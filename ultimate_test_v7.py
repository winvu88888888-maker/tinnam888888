"""
V7 — ULTIMATE ENGINE: Tích hợp ALL V6 findings + Full Backtest
Mục tiêu: Test xem bao nhiêu % trúng 6/6
"""
import sys, os, math, json, time
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper.data_manager import get_mega645_numbers, get_mega645_all


# ===================================================================
# LOAD V6 FINDINGS
# ===================================================================
V6_PATH = os.path.join(os.path.dirname(__file__), 'models', 'full_spectrum_v6.json')
with open(V6_PATH, 'r', encoding='utf-8') as f:
    V6 = json.load(f)

# Extract stable transitions
STABLE_TRANSITIONS = {}
for finding in V6['findings']:
    if finding['type'] == 'stable_transition':
        key = (finding['from'], finding['to'])
        STABLE_TRANSITIONS[key] = {
            'rate': finding['overall_rate'],
            'z': finding['overall_z'],
            'consistency': finding['consistency'],
        }

# Extract stable pairs
STABLE_PAIRS = {}
for finding in V6['findings']:
    if finding['type'] == 'stable_pair':
        pair = tuple(finding['pair'])
        STABLE_PAIRS[pair] = {
            'rate': finding['avg_rate'],
            'z': finding['z'],
            'consistency': finding['consistency'],
        }

# Extract digit transitions
DIGIT_TRANS = {}
for finding in V6['findings']:
    if finding['type'] == 'stable_digit_transitions':
        for dt in finding['details']:
            pos = 0 if 'smallest' in dt['pos'] else 5
            key = (pos, dt['from'], dt['to'])
            DIGIT_TRANS[key] = {
                'rate': dt['avg_rate'],
                'consistency': dt['consistency'],
            }

print(f"Loaded: {len(STABLE_TRANSITIONS)} transitions, "
      f"{len(STABLE_PAIRS)} pairs, {len(DIGIT_TRANS)} digit rules")


def predict_v7(train_data, n_portfolio=50):
    """Predict using ALL validated V6 findings."""
    MAX = 45
    PICK = 6
    base_p = PICK / MAX
    n = len(train_data)
    if n < 30:
        return [sorted(np.random.choice(range(1, MAX+1), PICK, replace=False).tolist())]

    scores = {num: 0.0 for num in range(1, MAX+1)}
    last = train_data[-1][:PICK]
    last_set = set(last)
    last_sorted = sorted(last)

    # ─────────────────────────────────────────────
    # SIGNAL 1: V6 Stable Transitions (weight=4)
    # ─────────────────────────────────────────────
    for prev_num in last_set:
        for nxt in range(1, MAX+1):
            key = (prev_num, nxt)
            if key in STABLE_TRANSITIONS:
                st = STABLE_TRANSITIONS[key]
                boost = (st['rate'] / base_p - 1) * st['consistency'] * 4
                scores[nxt] += boost

    # ─────────────────────────────────────────────
    # SIGNAL 2: Live Transition (trained on data, weight=3)
    # ─────────────────────────────────────────────
    follow = defaultdict(Counter)
    appear = Counter()
    for i in range(n - 1):
        for p in train_data[i][:PICK]:
            appear[p] += 1
            for nx in train_data[i+1][:PICK]:
                follow[p][nx] += 1
    for p in last_set:
        if appear[p] < 30:
            continue
        for nxt in range(1, MAX+1):
            rate = follow[p].get(nxt, 0) / appear[p]
            if rate > base_p * 1.2:
                scores[nxt] += (rate / base_p - 1) * 3

    # ─────────────────────────────────────────────
    # SIGNAL 3: V6 Digit Transitions (weight=5 — strongest signal)
    # ─────────────────────────────────────────────
    last_digit_pos0 = last_sorted[0] % 10
    last_digit_pos5 = last_sorted[5] % 10

    for pos_idx, last_dig in [(0, last_digit_pos0), (5, last_digit_pos5)]:
        best_targets = []
        for (pos, from_d, to_d), info in DIGIT_TRANS.items():
            if pos == pos_idx and from_d == last_dig:
                best_targets.append((to_d, info['rate'], info['consistency']))

        for to_d, rate, consistency in best_targets:
            # Boost numbers with this digit at this position
            for num in range(1, MAX+1):
                if num % 10 == to_d:
                    scores[num] += rate * consistency * 5

    # ─────────────────────────────────────────────
    # SIGNAL 4: V6 Stable Pairs (weight=3)
    # ─────────────────────────────────────────────
    for p in last_set:
        for nxt in range(1, MAX+1):
            key = tuple(sorted([p, nxt]))
            if key in STABLE_PAIRS:
                sp = STABLE_PAIRS[key]
                scores[nxt] += sp['z'] * sp['consistency'] * 0.5

    # ─────────────────────────────────────────────
    # SIGNAL 5: Gap Timing / Overdue (weight=2)
    # ─────────────────────────────────────────────
    last_seen = {}
    gap_data = defaultdict(list)
    for i, d in enumerate(train_data):
        for num in d[:PICK]:
            if num in last_seen:
                gap_data[num].append(i - last_seen[num])
            last_seen[num] = i
    for num in range(1, MAX+1):
        gaps = gap_data.get(num, [])
        if len(gaps) < 5:
            continue
        curr_gap = n - last_seen.get(num, 0)
        avg_gap = np.mean(gaps)
        if curr_gap > avg_gap:
            scores[num] += min((curr_gap / avg_gap - 1) * 2, 3)

    # ─────────────────────────────────────────────
    # SIGNAL 6: Decade Flow (weight=3)
    # ─────────────────────────────────────────────
    def dec(x):
        return min(x // 10, 4)
    last_dec = tuple(sorted(dec(x) for x in last))
    dec_trans = defaultdict(Counter)
    for i in range(1, n):
        prev = tuple(sorted(dec(x) for x in train_data[i-1][:PICK]))
        for x in train_data[i][:PICK]:
            dec_trans[prev][dec(x)] += 1
    expected = dec_trans.get(last_dec, Counter())
    total_exp = sum(expected.values()) or 1
    for num in range(1, MAX+1):
        d = dec(num)
        dp = expected.get(d, 0) / total_exp
        scores[num] += (dp - 0.2) * 3

    # ─────────────────────────────────────────────
    # SIGNAL 7: KNN Similarity (weight=2)
    # ─────────────────────────────────────────────
    knn_sc = Counter()
    total_w = 0
    for i in range(n - 2):
        sim = len(set(train_data[i][:PICK]) & last_set)
        if sim >= 3:
            w = sim ** 2
            total_w += w
            for num in train_data[i+1][:PICK]:
                knn_sc[num] += w
    if total_w > 0:
        for num in range(1, MAX+1):
            scores[num] += (knn_sc.get(num, 0) / total_w / base_p - 1) * 2

    # ─────────────────────────────────────────────
    # SIGNAL 8: Streak / Overdue (weight=1.5)
    # ─────────────────────────────────────────────
    for num in range(1, MAX+1):
        c = 0
        for x in reversed(train_data[-100:]):
            if num not in x[:PICK]:
                c += 1
            else:
                break
        expected_gap = MAX / PICK
        if c > expected_gap:
            scores[num] += min((c / expected_gap - 1) * 1.5, 2)

    # ─────────────────────────────────────────────
    # SIGNAL 9: Momentum (weight=1)
    # ─────────────────────────────────────────────
    if n >= 30:
        for num in range(1, MAX+1):
            r5 = sum(1 for d in train_data[-5:] if num in d[:PICK]) / 5
            r20 = sum(1 for d in train_data[-20:] if num in d[:PICK]) / 20
            scores[num] += (r5 - r20) * 4

    # ─────────────────────────────────────────────
    # ANTI-REPEAT (adaptive)
    # ─────────────────────────────────────────────
    rr = []
    for i in range(1, min(n, 60)):
        rr.append(len(set(train_data[i-1][:PICK]) & set(train_data[i][:PICK])))
    avg_rr = np.mean(rr) if rr else 0.8
    for num in last_set:
        scores[num] += (avg_rr - 1.0) * 1.5

    # ─────────────────────────────────────────────
    # GENERATE PORTFOLIO with constraints
    # ─────────────────────────────────────────────
    recent_sums = [sum(d[:PICK]) for d in train_data[-50:]]
    sum_lo = np.percentile(recent_sums, 3)
    sum_hi = np.percentile(recent_sums, 97)

    recent_ranges = [max(d[:PICK]) - min(d[:PICK]) for d in train_data[-50:]]
    range_lo = np.percentile(recent_ranges, 5)
    range_hi = np.percentile(recent_ranges, 95)

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    pool = [num for num, _ in ranked[:25]]

    results = []
    seen = set()
    for combo in combinations(pool[:18], PICK):
        s = sum(combo)
        if s < sum_lo or s > sum_hi:
            continue
        rng = combo[-1] - combo[0]
        if rng < range_lo or rng > range_hi:
            continue
        # Odd/even constraint
        odds = sum(1 for x in combo if x % 2 == 1)
        if odds < 1 or odds > 5:
            continue
        sc = sum(scores.get(n, 0) for n in combo)
        key = tuple(sorted(combo))
        if key not in seen:
            seen.add(key)
            results.append({'numbers': sorted(combo), 'score': sc})
        if len(results) >= n_portfolio * 5:
            break

    results.sort(key=lambda x: -x['score'])
    if not results:
        return [sorted(pool[:PICK])]
    return [r['numbers'] for r in results[:n_portfolio]]


def run_backtest():
    data = get_mega645_numbers()
    all_rows = get_mega645_all()
    dates = [r['draw_date'] for r in all_rows]
    N = len(data)
    MAX = 45
    PICK = 6
    t0 = time.time()

    print("=" * 80)
    print("  🏆 V7 ULTIMATE ENGINE — FULL BACKTEST: % TRÚNG 6/6")
    print(f"  {N} draws | Testing ALL portfolio sizes")
    print("=" * 80)

    # Test with different portfolio sizes
    portfolio_sizes = [1, 10, 50, 100, 200, 500]
    min_train = 200
    
    # Walk-forward: step every draw for max test coverage
    test_indices = list(range(min_train, N - 1))
    n_test = len(test_indices)
    
    print(f"  Test range: draw {min_train} → {N-1} ({n_test} tests)")
    print(f"  Portfolio sizes: {portfolio_sizes}")
    print(f"\n  Running...")

    # Results storage
    results = {ps: {'matches': [], 'best_matches': []} for ps in portfolio_sizes}
    random_matches = []
    np.random.seed(42)

    for idx_i, train_end in enumerate(test_indices):
        train = data[:train_end + 1]
        actual = set(data[train_end + 1][:PICK])

        # Generate max portfolio
        max_ps = max(portfolio_sizes)
        portfolio = predict_v7(train, n_portfolio=max_ps)

        # Score for each portfolio size
        for ps in portfolio_sizes:
            port = portfolio[:ps]
            best = max(len(set(p) & actual) for p in port) if port else 0
            results[ps]['best_matches'].append(best)
            if ps == 1:
                results[ps]['matches'].append(len(set(port[0]) & actual) if port else 0)

        # Random baseline
        rand = set(np.random.choice(range(1, MAX+1), PICK, replace=False).tolist())
        random_matches.append(len(rand & actual))

        if (idx_i + 1) % 200 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (idx_i + 1) * (n_test - idx_i - 1)
            print(f"    [{idx_i+1}/{n_test}] elapsed={elapsed:.0f}s, ETA={eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\n  Backtest complete in {elapsed:.1f}s")

    # ───────────────────────────────────────────────
    # RESULTS
    # ───────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  📊 KẾT QUẢ BACKTEST — {n_test} kỳ test")
    print(f"{'='*80}")

    # Random baseline
    rand_avg = np.mean(random_matches)
    rand_dist = Counter(random_matches)
    
    print(f"\n  ┌─────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐")
    print(f"  │ Port.   │  Avg/6   │  ≥3/6    │  ≥4/6    │  ≥5/6    │  6/6 🏆  │  vs Rand │")
    print(f"  ├─────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤")
    
    print(f"  │ Random  │  {rand_avg:.4f}  │"
          f"  {sum(1 for m in random_matches if m>=3)/n_test*100:5.2f}%  │"
          f"  {sum(1 for m in random_matches if m>=4)/n_test*100:5.2f}%  │"
          f"  {sum(1 for m in random_matches if m>=5)/n_test*100:5.2f}%  │"
          f"  {sum(1 for m in random_matches if m>=6)/n_test*100:5.2f}%  │"
          f"    ---   │")

    for ps in portfolio_sizes:
        bm = results[ps]['best_matches']
        avg = np.mean(bm)
        imp = (avg / rand_avg - 1) * 100 if rand_avg > 0 else 0
        pct3 = sum(1 for m in bm if m >= 3) / n_test * 100
        pct4 = sum(1 for m in bm if m >= 4) / n_test * 100
        pct5 = sum(1 for m in bm if m >= 5) / n_test * 100
        pct6 = sum(1 for m in bm if m >= 6) / n_test * 100
        
        label = f"{ps:3d} sets" if ps > 1 else "  1 set"
        print(f"  │ {label} │  {avg:.4f}  │  {pct3:5.2f}%  │  {pct4:5.2f}%  │"
              f"  {pct5:5.2f}%  │  {pct6:5.2f}%  │ {imp:+5.1f}%  │")

    print(f"  └─────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘")

    # Detailed distribution for 500 portfolio
    print(f"\n  Distribution chi tiết (Portfolio 500 sets):")
    bm500 = results[max(portfolio_sizes)]['best_matches']
    dist500 = Counter(bm500)
    for k in range(7):
        c = dist500.get(k, 0)
        pct = c / n_test * 100
        bar = '█' * int(pct)
        print(f"    {k}/6: {c:4d} ({pct:6.2f}%) {bar}")

    # Count actual 6/6 hits
    hits_6 = sum(1 for m in bm500 if m >= 6)
    hits_5 = sum(1 for m in bm500 if m >= 5)
    hits_4 = sum(1 for m in bm500 if m >= 4)

    print(f"\n  ╔═══════════════════════════════════════════════╗")
    print(f"  ║  KẾT QUẢ 6/6 JACKPOT (Portfolio {max(portfolio_sizes)} sets):     ║")
    print(f"  ║  Trúng 6/6: {hits_6:3d}/{n_test} kỳ = {hits_6/n_test*100:.4f}%          ║")
    print(f"  ║  Trúng 5/6: {hits_5:3d}/{n_test} kỳ = {hits_5/n_test*100:.4f}%          ║")
    print(f"  ║  Trúng 4/6: {hits_4:3d}/{n_test} kỳ = {hits_4/n_test*100:.4f}%          ║")
    print(f"  ╚═══════════════════════════════════════════════╝")

    # Save results
    output = {
        'version': '7.0 — Ultimate Engine',
        'total_draws': N,
        'n_test': n_test,
        'elapsed': round(elapsed, 1),
        'results': {},
    }
    for ps in portfolio_sizes:
        bm = results[ps]['best_matches']
        output['results'][f'portfolio_{ps}'] = {
            'avg': round(np.mean(bm), 4),
            'pct_3plus': round(sum(1 for m in bm if m>=3)/n_test*100, 4),
            'pct_4plus': round(sum(1 for m in bm if m>=4)/n_test*100, 4),
            'pct_5plus': round(sum(1 for m in bm if m>=5)/n_test*100, 4),
            'pct_6': round(sum(1 for m in bm if m>=6)/n_test*100, 4),
        }
    output['random'] = {
        'avg': round(rand_avg, 4),
        'pct_3plus': round(sum(1 for m in random_matches if m>=3)/n_test*100, 4),
        'pct_6': round(sum(1 for m in random_matches if m>=6)/n_test*100, 4),
    }

    path = os.path.join(os.path.dirname(__file__), 'models', 'ultimate_backtest_v7.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {path}")


if __name__ == '__main__':
    run_backtest()
