"""
V12 — FINAL FUSION: TẤT CẢ V3-V11 kết hợp + Test TOÀN BỘ
============================================================
Kết hợp TINH HOA:
  • V8 XGBoost (best ML +11.3%)
  • V6 Stable transitions (13) + Pairs (5) + Digit transitions (21)
  • V11 Per-column Markov + Companion + Tăng giảm + Chẵn lẻ + Block
  • V7 Gap timing + Decade flow + KNN + Momentum + Streak

Walk-forward: train→predict→test TỪNG KỲ MỘT
"""
import sys, os, math, json, time
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper.data_manager import get_mega645_numbers, get_mega645_all
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# Load V6 findings
V6_PATH = os.path.join(os.path.dirname(__file__), 'models', 'full_spectrum_v6.json')
V6 = {}
if os.path.exists(V6_PATH):
    with open(V6_PATH, 'r', encoding='utf-8') as f:
        V6 = json.load(f)

STABLE_TRANS = {}
STABLE_PAIRS = {}
DIGIT_TRANS = {}
for finding in V6.get('findings', []):
    if finding['type'] == 'stable_transition':
        STABLE_TRANS[(finding.get('from',0), finding.get('to',0))] = finding
    elif finding['type'] == 'stable_pair':
        STABLE_PAIRS[tuple(finding.get('pair',[0,0]))] = finding
    elif finding['type'] == 'stable_digit_transitions':
        for dt in finding.get('details', []):
            pos = 0 if 'smallest' in dt.get('pos','') else 5
            DIGIT_TRANS[(pos, dt.get('from',0), dt.get('to',0))] = dt

MAX, PICK = 45, 6
BASE_P = PICK / MAX


def build_xgb_features(data, idx):
    if idx < 20: return None
    curr = sorted(data[idx][:PICK])
    f = []
    f.extend(curr)
    f.extend([sum(curr), max(curr)-min(curr), np.mean(curr), np.std(curr),
              sum(1 for x in curr if x%2==1), sum(1 for x in curr if x>22)])
    for x in curr:
        f.extend([x//10, x%10])
    for num in range(1, MAX+1):
        w5 = sum(1 for d in data[max(0,idx-4):idx+1] if num in d[:PICK]) / 5
        w15 = sum(1 for d in data[max(0,idx-14):idx+1] if num in d[:PICK]) / 15
        w50 = sum(1 for d in data[max(0,idx-49):idx+1] if num in d[:PICK]) / min(50,idx+1)
        f.extend([w5, w15, w50])
    last_seen = {}
    for i in range(idx+1):
        for num in data[i][:PICK]:
            last_seen[num] = i
    for num in range(1, MAX+1):
        f.append(idx - last_seen.get(num, 0))
    return f


def predict_fusion(data, train_end, xgb_probs):
    """Combine ALL signals into final scores."""
    n = train_end + 1
    last = sorted(data[train_end][:PICK])
    last_set = set(last)
    scores = {num: 0.0 for num in range(1, MAX+1)}

    # ── S1: XGBoost probabilities (weight=5) ──
    if xgb_probs is not None:
        for num in range(1, MAX+1):
            scores[num] += (xgb_probs.get(num, BASE_P) - BASE_P) * 30

    # ── S2: V6 Stable Transitions (weight=4) ──
    for prev_num in last_set:
        for nxt in range(1, MAX+1):
            st = STABLE_TRANS.get((prev_num, nxt))
            if st:
                rate = st.get('overall_rate', st.get('avg_rate', BASE_P))
                cons = st.get('consistency', 0.5)
                scores[nxt] += (rate / BASE_P - 1) * cons * 4

    # ── S3: V6 Digit Transitions (weight=5) ──
    for pos_idx in [0, 5]:
        last_dig = last[pos_idx] % 10
        for (pos, from_d, to_d), info in DIGIT_TRANS.items():
            if pos == pos_idx and from_d == last_dig:
                rate = info.get('avg_rate', info.get('rate', 0.1))
                cons = info.get('consistency', 0.5)
                for num in range(1, MAX+1):
                    if num % 10 == to_d:
                        scores[num] += rate * cons * 5

    # ── S4: V6 Stable Pairs (weight=3) ──
    for p in last_set:
        for nxt in range(1, MAX+1):
            sp = STABLE_PAIRS.get(tuple(sorted([p, nxt])))
            if sp:
                z = sp.get('z', 2)
                cons = sp.get('consistency', 0.5)
                scores[nxt] += z * cons * 0.5

    # ── S5: Per-column Markov (V11 style, weight=4) ──
    for col in range(PICK):
        trans = defaultdict(Counter)
        for i in range(n - 1):
            sd = sorted(data[i][:PICK])
            sd_next = sorted(data[i+1][:PICK])
            trans[sd[col]][sd_next[col]] += 1
        val = last[col]
        if val in trans:
            total = sum(trans[val].values())
            for nxt_val, cnt in trans[val].most_common(5):
                rate = cnt / total
                scores[nxt_val] += rate * 4

    # ── S6: Live follow transitions (weight=3) ──
    follow = defaultdict(Counter)
    appear = Counter()
    for i in range(n - 1):
        for p in data[i][:PICK]:
            appear[p] += 1
            for nx in data[i+1][:PICK]:
                follow[p][nx] += 1
    for p in last_set:
        if appear[p] < 30: continue
        for nxt in range(1, MAX+1):
            rate = follow[p].get(nxt, 0) / appear[p]
            if rate > BASE_P * 1.2:
                scores[nxt] += (rate / BASE_P - 1) * 3

    # ── S7: Gap timing (weight=2) ──
    last_seen = {}
    gap_data = defaultdict(list)
    for i, d in enumerate(data[:n]):
        for num in d[:PICK]:
            if num in last_seen:
                gap_data[num].append(i - last_seen[num])
            last_seen[num] = i
    for num in range(1, MAX+1):
        gaps = gap_data.get(num, [])
        if len(gaps) < 5: continue
        curr_gap = n - 1 - last_seen.get(num, 0)
        avg_gap = np.mean(gaps)
        if curr_gap > avg_gap:
            scores[num] += min((curr_gap / avg_gap - 1) * 2, 3)

    # ── S8: Decade flow (weight=3) ──
    def dec(x): return min(x // 10, 4)
    last_dec = tuple(sorted(dec(x) for x in last))
    dec_trans = defaultdict(Counter)
    for i in range(1, n):
        prev = tuple(sorted(dec(x) for x in sorted(data[i-1][:PICK])))
        for x in data[i][:PICK]:
            dec_trans[prev][dec(x)] += 1
    expected = dec_trans.get(last_dec, Counter())
    total_exp = sum(expected.values()) or 1
    for num in range(1, MAX+1):
        scores[num] += (expected.get(dec(num), 0) / total_exp - 0.2) * 3

    # ── S9: Column trend (V11 up/down, weight=2) ──
    if n >= 3:
        for col in range(PICK):
            d1 = sorted(data[train_end][:PICK])[col] - sorted(data[train_end-1][:PICK])[col]
            trend = 'up' if d1 > 0 else 'down'
            local_tt = defaultdict(Counter)
            for i in range(2, n):
                d = sorted(data[i-1][:PICK])[col] - sorted(data[i-2][:PICK])[col]
                t = 'up' if d > 0 else ('down' if d < 0 else 'same')
                d2 = sorted(data[i][:PICK])[col] - sorted(data[i-1][:PICK])[col]
                t2 = 'up' if d2 > 0 else ('down' if d2 < 0 else 'same')
                local_tt[t][t2] += 1
            nxt_dist = local_tt.get(trend, Counter())
            total_tt = sum(nxt_dist.values()) or 1
            best = max(nxt_dist, key=nxt_dist.get) if nxt_dist else 'up'
            curr_v = last[col]
            if best == 'up':
                for num in range(curr_v+1, min(curr_v+8, MAX+1)):
                    scores[num] += 1
            elif best == 'down':
                for num in range(max(1, curr_v-8), curr_v):
                    scores[num] += 1

    # ── S10: KNN similarity (weight=2) ──
    knn_sc = Counter()
    total_w = 0
    for i in range(n - 2):
        sim = len(set(data[i][:PICK]) & last_set)
        if sim >= 3:
            w = sim ** 2
            total_w += w
            for num in data[i+1][:PICK]:
                knn_sc[num] += w
    if total_w > 0:
        for num in range(1, MAX+1):
            scores[num] += (knn_sc.get(num, 0) / total_w / BASE_P - 1) * 2

    # ── S11: Momentum (weight=1.5) ──
    if n >= 20:
        for num in range(1, MAX+1):
            r5 = sum(1 for d in data[n-5:n] if num in d[:PICK]) / 5
            r20 = sum(1 for d in data[n-20:n] if num in d[:PICK]) / 20
            scores[num] += (r5 - r20) * 6

    # ── S12: Block companion (V11, weight=2) ──
    def blk(x): return min((x-1)//9, 4)
    curr_bp = tuple(blk(x) for x in last)
    bp_trans = defaultdict(Counter)
    for i in range(n - 1):
        bp = tuple(blk(x) for x in sorted(data[i][:PICK]))
        next_bp = tuple(blk(x) for x in sorted(data[i+1][:PICK]))
        bp_trans[bp][next_bp] += 1
    if curr_bp in bp_trans:
        total = sum(bp_trans[curr_bp].values())
        for next_bp, count in bp_trans[curr_bp].most_common(3):
            weight = count / total
            for b in next_bp:
                lo = b * 9 + 1
                hi = min(lo + 8, MAX)
                for num in range(lo, hi + 1):
                    scores[num] += weight * 2

    # ── Anti-repeat (adaptive) ──
    rr = []
    for i in range(1, min(n, 50)):
        rr.append(len(set(data[n-i-1][:PICK]) & set(data[n-i][:PICK])))
    avg_rr = np.mean(rr) if rr else 0.8
    for num in last_set:
        scores[num] += (avg_rr - 1.0) * 1.5

    return scores


def generate_portfolio(scores, data, train_end, n_portfolio=500):
    recent = [sorted(data[i][:PICK]) for i in range(max(0,train_end-30), train_end+1)]
    sums = [sum(d) for d in recent]
    ranges = [d[-1]-d[0] for d in recent]
    sum_lo, sum_hi = np.percentile(sums, 3), np.percentile(sums, 97)
    rng_lo, rng_hi = np.percentile(ranges, 3), np.percentile(ranges, 97)

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    pool = [num for num, _ in ranked[:22]]

    results = []
    seen = set()
    for combo in combinations(pool[:18], PICK):
        s = sum(combo)
        if s < sum_lo or s > sum_hi: continue
        r = combo[-1] - combo[0]
        if r < rng_lo or r > rng_hi: continue
        odds = sum(1 for x in combo if x%2==1)
        if odds < 1 or odds > 5: continue
        sc = sum(scores.get(n, 0) for n in combo)
        key = tuple(sorted(combo))
        if key not in seen:
            seen.add(key)
            results.append({'numbers': sorted(combo), 'score': sc})
        if len(results) >= n_portfolio * 3:
            break
    results.sort(key=lambda x: -x['score'])
    return [r['numbers'] for r in results[:n_portfolio]]


def run_v12():
    data = get_mega645_numbers()
    N = len(data)
    t0 = time.time()

    print("=" * 80)
    print("  🔥 V12 FINAL FUSION — ALL V3-V11 combined")
    print(f"  {N} draws | 12 signals | Walk-forward EVERY draw")
    print("=" * 80)

    # Pre-train XGBoost models with expanding window
    # Phase 1: Initial training (first 70%)
    train_split = int(N * 0.6)
    min_train = 200

    print(f"\n  Phase 1: Training XGBoost on first {train_split} draws...")
    X_train, Y_train = [], {num: [] for num in range(1, MAX+1)}
    for idx in range(20, train_split):
        f = build_xgb_features(data, idx)
        if f:
            X_train.append(f)
            next_draw = set(data[idx+1][:PICK])
            for num in range(1, MAX+1):
                Y_train[num].append(1 if num in next_draw else 0)

    X_train = np.array(X_train)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    xgb_models = {}
    for num in range(1, MAX+1):
        y = np.array(Y_train[num])
        xgb_models[num] = GradientBoostingClassifier(
            n_estimators=80, max_depth=4, learning_rate=0.1,
            subsample=0.8, min_samples_leaf=10, random_state=42
        )
        xgb_models[num].fit(X_train_s, y)

    print(f"  Trained {len(xgb_models)} XGBoost models ({X_train.shape[1]} features)")

    # Phase 2: Walk-forward backtest
    port_sizes = [1, 10, 50, 100, 500]
    results = {ps: [] for ps in port_sizes}
    random_results = []
    np.random.seed(42)
    test_start = max(min_train, train_split)
    n_test = N - test_start - 1

    print(f"  Phase 2: Walk-forward test [{test_start}..{N-2}] ({n_test} tests)")

    for ti in range(test_start, N - 1):
        actual = set(sorted(data[ti + 1][:PICK]))

        # Get XGBoost probabilities
        f = build_xgb_features(data, ti)
        xgb_probs = {}
        if f:
            f_s = scaler.transform([f])
            for num, model in xgb_models.items():
                p = model.predict_proba(f_s)[0]
                xgb_probs[num] = p[1] if len(p) > 1 else BASE_P

        # Fusion scores
        scores = predict_fusion(data, ti, xgb_probs)

        # Generate portfolio
        portfolio = generate_portfolio(scores, data, ti, n_portfolio=max(port_sizes))

        for ps in port_sizes:
            port = portfolio[:ps]
            if port:
                best = max(len(set(p) & actual) for p in port)
            else:
                best = 0
            results[ps].append(best)

        rand = set(np.random.choice(range(1, MAX+1), PICK, replace=False).tolist())
        random_results.append(len(rand & actual))

        idx_i = ti - test_start
        if (idx_i + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (idx_i + 1) * (n_test - idx_i - 1)
            p500 = np.mean(results[500])
            print(f"    [{idx_i+1}/{n_test}] port500={p500:.3f}, "
                  f"elapsed={elapsed:.0f}s, ETA={eta:.0f}s")

    elapsed = time.time() - t0
    rand_avg = np.mean(random_results)

    # ═══════════════ RESULTS ═══════════════
    print(f"\n{'='*80}")
    print(f"  📊 V12 FINAL RESULTS — {n_test} kỳ test")
    print(f"{'='*80}")

    header = f"  {'Port':>6} | {'Avg/6':>7} | {'≥2/6':>7} | {'≥3/6':>7} | {'≥4/6':>7} | {'≥5/6':>7} | {'6/6':>7} | {'vs Rand':>8}"
    print(f"\n{header}")
    print(f"  {'─'*6} | {'─'*7} | {'─'*7} | {'─'*7} | {'─'*7} | {'─'*7} | {'─'*7} | {'─'*8}")

    def row(label, bm, baseline=None):
        avg = np.mean(bm)
        n = len(bm)
        imp = f"{(avg/baseline-1)*100:+6.1f}%" if baseline else "   ---  "
        return (f"  {label:>6} | {avg:.4f}  | "
                f"{sum(1 for m in bm if m>=2)/n*100:5.1f}%  | "
                f"{sum(1 for m in bm if m>=3)/n*100:5.1f}%  | "
                f"{sum(1 for m in bm if m>=4)/n*100:5.1f}%  | "
                f"{sum(1 for m in bm if m>=5)/n*100:5.2f}%  | "
                f"{sum(1 for m in bm if m>=6)/n*100:5.2f}%  | "
                f"{imp}")

    print(row("Rand", random_results))
    for ps in port_sizes:
        print(row(f"{ps}", results[ps], rand_avg))

    # Distribution
    print(f"\n  Distribution (Portfolio 500):")
    bm500 = results[500]
    dist = Counter(bm500)
    for k in range(7):
        c = dist.get(k, 0)
        bar = '█' * int(c / n_test * 100)
        print(f"    {k}/6: {c:4d} ({c/n_test*100:6.2f}%) {bar}")

    h6 = dist.get(6, 0)
    h5 = dist.get(5, 0)
    h4 = dist.get(4, 0)
    h3 = dist.get(3, 0)

    print(f"\n  ╔══════════════════════════════════════════════════════════╗")
    print(f"  ║  V12 FINAL FUSION — KẾT QUẢ THỰC TẾ (Port 500)        ║")
    print(f"  ║  6/6: {h6:4d}/{n_test} kỳ = {h6/n_test*100:.4f}%                        ║")
    print(f"  ║  5/6: {h5:4d}/{n_test} kỳ = {h5/n_test*100:.4f}%                        ║")
    print(f"  ║  4/6: {h4:4d}/{n_test} kỳ = {h4/n_test*100:.4f}%                        ║")
    print(f"  ║  3/6: {h3:4d}/{n_test} kỳ = {h3/n_test*100:.4f}%                        ║")
    print(f"  ║  Time: {elapsed:.1f}s                                        ║")
    print(f"  ╚══════════════════════════════════════════════════════════╝")

    # Save
    output = {
        'version': '12.0 — Final Fusion (V3-V11 combined)',
        'n_draws': N, 'n_test': n_test,
        'signals': ['XGBoost', 'V6_StableTrans', 'V6_DigitTrans', 'V6_StablePairs',
                     'PerColumnMarkov', 'LiveFollow', 'GapTiming', 'DecadeFlow',
                     'ColumnTrend', 'KNN', 'Momentum', 'BlockCompanion'],
        'results': {},
        'random_avg': round(rand_avg, 4),
        'elapsed': round(elapsed, 1),
    }
    for ps in port_sizes:
        bm = results[ps]
        output['results'][f'port_{ps}'] = {
            'avg': round(np.mean(bm), 4),
            'pct_2plus': round(sum(1 for m in bm if m>=2)/n_test*100, 2),
            'pct_3plus': round(sum(1 for m in bm if m>=3)/n_test*100, 2),
            'pct_4plus': round(sum(1 for m in bm if m>=4)/n_test*100, 2),
            'pct_5plus': round(sum(1 for m in bm if m>=5)/n_test*100, 4),
            'pct_6': round(sum(1 for m in bm if m>=6)/n_test*100, 4),
            'distribution': {str(k): dist.get(k,0) for k in range(7)} if ps == 500 else {},
        }
    path = os.path.join(os.path.dirname(__file__), 'models', 'v12_final.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {path}")


if __name__ == '__main__':
    run_v12()
