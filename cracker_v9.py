"""
V9 — CONSTRAINT CRACKER + THEORETICAL LIMIT
============================================
CHIẾN LƯỢC MỚI: Thay vì score từng số → LOẠI TRỪ.
Nếu loại bỏ đủ nhiều combo "không thể" → không gian tìm kiếm nhỏ lại.

ATTACKS:
1. XGBoost STACKING (multi-layer: model predicts models)
2. Hard constraint filter (loại hết combo vi phạm rules)
3. Coverage optimization (portfolio không overlap)
4. Massive portfolio (5000-50000 sets)
5. THEORETICAL LIMIT: ngay cả "God algorithm" tối đa được bao nhiêu?
"""
import sys, os, math, json, time
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper.data_manager import get_mega645_numbers, get_mega645_all

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


def build_features_compact(data, idx):
    """Compact but powerful features."""
    MAX, PICK = 45, 6
    if idx < 20:
        return None
    curr = sorted(data[idx][:PICK])
    f = []
    # Position values + stats (12)
    f.extend(curr)
    f.extend([sum(curr), max(curr)-min(curr), np.mean(curr), np.std(curr),
              sum(1 for x in curr if x%2==1), sum(1 for x in curr if x>22)])
    # Digits (12)
    for x in curr:
        f.extend([x//10, x%10])
    # Freq 3 windows per number (135)
    for num in range(1, MAX+1):
        w5 = sum(1 for d in data[max(0,idx-4):idx+1] if num in d[:PICK]) / 5
        w15 = sum(1 for d in data[max(0,idx-14):idx+1] if num in d[:PICK]) / 15
        w50 = sum(1 for d in data[max(0,idx-49):idx+1] if num in d[:PICK]) / min(50,idx+1)
        f.extend([w5, w15, w50])
    # Gap per number (45)
    last_seen = {}
    for i in range(idx+1):
        for num in data[i][:PICK]:
            last_seen[num] = i
    for num in range(1, MAX+1):
        f.append(idx - last_seen.get(num, 0))
    return f


def run_v9():
    data = get_mega645_numbers()
    all_rows = get_mega645_all()
    dates = [r['draw_date'] for r in all_rows]
    N = len(data)
    MAX = 45
    PICK = 6
    base_p = PICK / MAX
    C_45_6 = math.comb(MAX, PICK)
    t0 = time.time()

    print("=" * 80)
    print("  ⚡ V9 — CONSTRAINT CRACKER + THEORETICAL LIMIT")
    print(f"  {N} draws | C(45,6) = {C_45_6:,}")
    print("=" * 80)

    # ================================================================
    # PART 1: THEORETICAL MAXIMUM
    # "God Algorithm" — biết mọi pattern → tối đa được bao nhiêu?
    # ================================================================
    print(f"\n{'━'*80}")
    print("  PART 1: THEORETICAL MAXIMUM — God Algorithm")
    print("  Nếu biết HOÀN TOÀN mọi pattern → max bao nhiêu?")
    print(f"{'━'*80}")

    # Build per-number XGBoost with ALL data (no split — find theoretical max)
    print("\n  Training XGBoost on FULL data (theoretical upper bound)...")
    X_all = []
    for idx in range(20, N-1):
        f = build_features_compact(data, idx)
        if f:
            X_all.append(f)
    X_all = np.array(X_all)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # Train and get in-sample probabilities (OVERFIT ON PURPOSE = theoretical max)
    god_probs = np.zeros((len(X_all), MAX))
    for num_idx in range(MAX):
        num = num_idx + 1
        y = np.array([1 if num in data[idx+1][:PICK] else 0
                       for idx in range(20, N-1)][:len(X_all)])
        gb = GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                         learning_rate=0.1, random_state=42)
        gb.fit(X_scaled, y)
        probs = gb.predict_proba(X_scaled)
        god_probs[:, num_idx] = probs[:, 1] if probs.shape[1] > 1 else 0.5

    # For each test draw: pick top 6 by god algo
    god_matches = []
    for i in range(len(X_all)):
        ranked = np.argsort(-god_probs[i])[:PICK]
        predicted = set(r + 1 for r in ranked)
        actual = set(data[i + 21][:PICK])
        god_matches.append(len(predicted & actual))

    god_avg = np.mean(god_matches)
    god_dist = Counter(god_matches)
    n_god = len(god_matches)

    print(f"\n  GOD ALGORITHM (in-sample, overfit = theoretical max):")
    print(f"    Avg: {god_avg:.4f}/6")
    for k in range(7):
        c = god_dist.get(k, 0)
        print(f"    {k}/6: {c:4d} ({c/n_god*100:.2f}%)")
    pct6_god = god_dist.get(6, 0) / n_god * 100
    print(f"\n    6/6 (GOD): {god_dist.get(6,0)}/{n_god} = {pct6_god:.4f}%")

    # God portfolio: for each draw, generate top combos
    print(f"\n  God Portfolio test (top-K combos per draw)...")
    god_port = {k: [] for k in [1, 100, 1000, 5000]}
    for i in range(0, len(X_all), max(1, len(X_all)//200)):
        probs = god_probs[i]
        top20 = np.argsort(-probs)[:20]
        top20_nums = [int(r + 1) for r in top20]
        actual = set(data[i + 21][:PICK])

        combos = []
        for combo in combinations(top20_nums[:14], PICK):
            sc = sum(probs[n-1] for n in combo)
            combos.append((sorted(combo), sc))
        combos.sort(key=lambda x: -x[1])

        for k in god_port:
            port = [c[0] for c in combos[:k]]
            if port:
                best = max(len(set(p) & actual) for p in port)
            else:
                best = 0
            god_port[k].append(best)

    print(f"    {'Port':>6} | {'Avg':>6} | {'≥3':>6} | {'≥4':>6} | {'≥5':>6} | {'6/6':>6}")
    for k in sorted(god_port.keys()):
        bm = god_port[k]
        avg = np.mean(bm)
        p3 = sum(1 for m in bm if m>=3)/len(bm)*100
        p4 = sum(1 for m in bm if m>=4)/len(bm)*100
        p5 = sum(1 for m in bm if m>=5)/len(bm)*100
        p6 = sum(1 for m in bm if m>=6)/len(bm)*100
        print(f"    {k:6d} | {avg:.4f} | {p3:5.1f}% | {p4:5.1f}% | {p5:5.1f}% | {p6:5.2f}%")

    # ================================================================
    # PART 2: HARD CONSTRAINT FILTER
    # Không score → LOẠI TRỪ combo sai → narrow search space
    # ================================================================
    print(f"\n{'━'*80}")
    print("  PART 2: Hard Constraint Filter — Narrow Search Space")
    print(f"{'━'*80}")

    # Collect constraints from historical data
    all_sums = [sum(d[:PICK]) for d in data]
    all_ranges = [max(d[:PICK]) - min(d[:PICK]) for d in data]
    all_odds = [sum(1 for x in d[:PICK] if x % 2 == 1) for d in data]
    all_decades = []
    for d in data:
        dec = Counter(min(x//10, 4) for x in d[:PICK])
        all_decades.append(tuple(sorted(dec.items())))

    sum_lo, sum_hi = np.percentile(all_sums, 1), np.percentile(all_sums, 99)
    range_lo, range_hi = np.percentile(all_ranges, 1), np.percentile(all_ranges, 99)
    odd_counts = Counter(all_odds)

    print(f"  Constraints from {N} draws:")
    print(f"    Sum range: [{sum_lo:.0f}, {sum_hi:.0f}] (99% CI)")
    print(f"    Range: [{range_lo:.0f}, {range_hi:.0f}]")
    print(f"    Odd counts: {dict(sorted(odd_counts.items()))}")

    # Count how many of C(45,6) pass all constraints
    print(f"\n  Counting valid combos (sampling)...")
    n_sample = 500000
    np.random.seed(42)
    valid = 0
    for _ in range(n_sample):
        combo = sorted(np.random.choice(range(1, MAX+1), PICK, replace=False))
        s = sum(combo)
        if s < sum_lo or s > sum_hi:
            continue
        r = combo[-1] - combo[0]
        if r < range_lo or r > range_hi:
            continue
        odds = sum(1 for x in combo if x % 2 == 1)
        if odds < 1 or odds > 5:
            continue
        valid += 1

    valid_rate = valid / n_sample
    estimated_valid = int(C_45_6 * valid_rate)
    reduction = (1 - valid_rate) * 100

    print(f"    Valid combos: ~{estimated_valid:,} / {C_45_6:,}")
    print(f"    Reduction: {reduction:.1f}%")
    print(f"    Remaining search space: {valid_rate*100:.1f}%")

    # ================================================================
    # PART 3: STACKED XGBOOST + CONSTRAINT → Large Portfolio
    # ================================================================
    print(f"\n{'━'*80}")
    print("  PART 3: Stacked XGBoost + Constraints → Walk-Forward")
    print(f"{'━'*80}")

    # Walk-forward with proper split
    train_end = int(N * 0.75)
    print(f"  Train: {train_end}, Test: {N - train_end - 1}")

    # Train models on training set
    X_train = []
    Y_labels = {num: [] for num in range(1, MAX+1)}
    for idx in range(20, train_end):
        f = build_features_compact(data, idx)
        if f:
            X_train.append(f)
            next_draw = set(data[idx+1][:PICK])
            for num in range(1, MAX+1):
                Y_labels[num].append(1 if num in next_draw else 0)

    X_train = np.array(X_train)
    scaler2 = StandardScaler()
    X_train_s = scaler2.fit_transform(X_train)

    print(f"  Training {MAX} XGBoost models...")
    models = {}
    for num in range(1, MAX+1):
        y = np.array(Y_labels[num])
        gb = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, min_samples_leaf=10, random_state=42
        )
        gb.fit(X_train_s, y)
        models[num] = gb

    # Test with large portfolios
    print(f"  Testing with portfolios up to 5000...")
    port_sizes = [1, 50, 500, 1000, 2000, 5000]
    results = {ps: [] for ps in port_sizes}
    random_results = []
    np.random.seed(42)
    n_test = 0

    for idx in range(train_end, N-1):
        f = build_features_compact(data, idx)
        if f is None:
            continue
        n_test += 1
        f_s = scaler2.transform([f])
        actual = set(data[idx+1][:PICK])

        # Get probabilities
        probs = {}
        for num, model in models.items():
            p = model.predict_proba(f_s)[0]
            probs[num] = p[1] if len(p) > 1 else 0.5

        # Generate large portfolio with constraints
        last_draw = sorted(data[idx][:PICK])
        recent_sums = [sum(data[max(0,idx-i)][:PICK]) for i in range(min(50, idx+1))]
        s_lo = np.percentile(recent_sums, 2)
        s_hi = np.percentile(recent_sums, 98)

        ranked = sorted(probs.items(), key=lambda x: -x[1])
        pool = [num for num, _ in ranked[:22]]

        portfolio = []
        seen = set()
        # Generate from top pool with constraints
        for combo in combinations(pool[:18], PICK):
            s = sum(combo)
            if s < s_lo or s > s_hi:
                continue
            r = combo[-1] - combo[0]
            if r < 15 or r > 42:
                continue
            odds = sum(1 for x in combo if x % 2 == 1)
            if odds < 1 or odds > 5:
                continue
            key = tuple(combo)
            if key not in seen:
                seen.add(key)
                sc = sum(probs.get(n, 0) for n in combo)
                portfolio.append((sorted(combo), sc))
            if len(portfolio) >= max(port_sizes):
                break

        portfolio.sort(key=lambda x: -x[1])

        for ps in port_sizes:
            port = [p[0] for p in portfolio[:ps]]
            if port:
                best = max(len(set(p) & actual) for p in port)
            else:
                best = 0
            results[ps].append(best)

        rand = set(np.random.choice(range(1, MAX+1), PICK, replace=False).tolist())
        random_results.append(len(rand & actual))

        if n_test % 50 == 0:
            print(f"    [{n_test}] port5000 avg={np.mean(results[5000]):.3f}")

    # ================================================================
    # RESULTS
    # ================================================================
    elapsed = time.time() - t0
    rand_avg = np.mean(random_results)

    print(f"\n{'='*80}")
    print(f"  📊 V9 KẾT QUẢ — {n_test} kỳ test")
    print(f"{'='*80}")

    print(f"\n  {'Port':>6} | {'Avg/6':>7} | {'≥3/6':>7} | {'≥4/6':>7} | "
          f"{'≥5/6':>7} | {'6/6':>7} | {'vs Rand':>8}")
    print(f"  {'─'*6} | {'─'*7} | {'─'*7} | {'─'*7} | {'─'*7} | {'─'*7} | {'─'*8}")
    print(f"  {'Rand':>6} | {rand_avg:.4f}  | "
          f"{sum(1 for m in random_results if m>=3)/n_test*100:5.2f}%  | "
          f"{sum(1 for m in random_results if m>=4)/n_test*100:5.2f}%  | "
          f"{sum(1 for m in random_results if m>=5)/n_test*100:5.2f}%  | "
          f"{sum(1 for m in random_results if m>=6)/n_test*100:5.2f}%  | "
          f"   ---  ")

    for ps in port_sizes:
        bm = results[ps]
        avg = np.mean(bm)
        imp = (avg / rand_avg - 1) * 100
        print(f"  {ps:6d} | {avg:.4f}  | "
              f"{sum(1 for m in bm if m>=3)/n_test*100:5.2f}%  | "
              f"{sum(1 for m in bm if m>=4)/n_test*100:5.2f}%  | "
              f"{sum(1 for m in bm if m>=5)/n_test*100:5.2f}%  | "
              f"{sum(1 for m in bm if m>=6)/n_test*100:5.2f}%  | "
              f"{imp:+6.1f}%  ")

    # Distribution for largest portfolio
    max_ps = max(port_sizes)
    bm_max = results[max_ps]
    dist = Counter(bm_max)
    print(f"\n  Distribution (Portfolio {max_ps}):")
    for k in range(7):
        c = dist.get(k, 0)
        bar = '█' * int(c / n_test * 100)
        print(f"    {k}/6: {c:4d} ({c/n_test*100:6.2f}%) {bar}")

    hits6 = dist.get(6, 0)
    hits5 = dist.get(5, 0)
    hits4 = dist.get(4, 0)
    print(f"\n  ╔════════════════════════════════════════════════════╗")
    print(f"  ║  6/6: {hits6:3d}/{n_test} = {hits6/n_test*100:.4f}%                         ║")
    print(f"  ║  5/6: {hits5:3d}/{n_test} = {hits5/n_test*100:.4f}%                         ║")
    print(f"  ║  4/6: {hits4:3d}/{n_test} = {hits4/n_test*100:.4f}%                         ║")
    print(f"  ╚════════════════════════════════════════════════════╝")

    # ================================================================
    # MATHEMATICAL PROOF OF IMPOSSIBILITY
    # ================================================================
    print(f"\n{'━'*80}")
    print("  📐 MATHEMATICAL ANALYSIS")
    print(f"{'━'*80}")

    print(f"""
  C(45,6) = {C_45_6:,} possible combos

  With constraint filter: ~{estimated_valid:,} valid combos ({valid_rate*100:.1f}%)
  
  With {max_ps} tickets per draw:
    Random P(6/6) = {max_ps}/{C_45_6} = {max_ps/C_45_6*100:.6f}%
    Constraint P(6/6) = {max_ps}/{estimated_valid} = {max_ps/max(estimated_valid,1)*100:.6f}%
    
  Over {n_test} test draws:
    Expected random hits = {n_test * max_ps / C_45_6:.3f}
    Expected constrained hits = {n_test * max_ps / max(estimated_valid, 1):.3f}

  To have 50% chance of 6/6 in {n_test} draws:
    Need tickets/draw = {int(C_45_6 * 0.693 / n_test):,}
    Cost: {int(C_45_6 * 0.693 / n_test) * 10000 / 1e9:.1f} billion VND per draw
    
  Even GOD algorithm (in-sample overfit):
    6/6 rate: {pct6_god:.4f}%
    """)

    # Save
    output = {
        'version': '9.0',
        'god_algorithm': {
            'avg': round(god_avg, 4),
            'pct_6': round(pct6_god, 4),
            'note': 'in-sample overfit = theoretical maximum',
        },
        'constraint_filter': {
            'valid_combos': estimated_valid,
            'reduction_pct': round(reduction, 1),
        },
        'test_results': {},
        'elapsed': round(elapsed, 1),
    }
    for ps in port_sizes:
        bm = results[ps]
        output['test_results'][f'port_{ps}'] = {
            'avg': round(np.mean(bm), 4),
            'pct_3': round(sum(1 for m in bm if m>=3)/n_test*100, 2),
            'pct_4': round(sum(1 for m in bm if m>=4)/n_test*100, 2),
            'pct_5': round(sum(1 for m in bm if m>=5)/n_test*100, 2),
            'pct_6': round(sum(1 for m in bm if m>=6)/n_test*100, 2),
        }

    path = os.path.join(os.path.dirname(__file__), 'models', 'cracker_v9.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    print(f"  Saved to {path}")
    print(f"  Time: {elapsed:.1f}s")


if __name__ == '__main__':
    run_v9()
