"""
BACKTEST: Pick TOP-10 scored combos from each pool size.
Measure actual 6/6, 5/6, 4/6 rates.

This establishes the CEILING of what 10 sets can achieve.
"""
import sys, time, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from scraper.data_manager import get_mega645_numbers, get_mega645_all

data = get_mega645_numbers()
dates = [r['draw_date'] for r in get_mega645_all()]
total = len(data)
MAX_NUM = 45
PICK = 6

def compute_signals(data_slice):
    """Compute signal scores for each number 1-45."""
    n = len(data_slice)
    last = data_slice[-1][:6]
    last_set = set(last)
    scores = {num: 0.001 for num in range(1, MAX_NUM + 1)}

    # Transition
    follow = defaultdict(Counter)
    pc = Counter()
    for i in range(n - 1):
        for p in data_slice[i][:6]:
            pc[p] += 1
            for nx in data_slice[i + 1][:6]:
                follow[p][nx] += 1
    base = PICK / MAX_NUM
    for num in range(1, MAX_NUM + 1):
        prob = sum(follow[p].get(num, 0) for p in last_set) / max(sum(pc[p] for p in last_set), 1)
        scores[num] += max(0, (prob / base - 1) * 3)

    # Momentum
    if n >= 50:
        for num in range(1, MAX_NUM + 1):
            f5 = sum(1 for x in data_slice[-5:] if num in x[:6]) / 5
            f20 = sum(1 for x in data_slice[-20:] if num in x[:6]) / 20
            f50 = sum(1 for x in data_slice[-50:] if num in x[:6]) / 50
            scores[num] += (f5 - f20) * 10 + (f20 - f50) * 5

    # Gap timing
    for num in range(1, MAX_NUM + 1):
        apps = [i for i, x in enumerate(data_slice) if num in x[:6]]
        if len(apps) < 5: continue
        gaps = [apps[j+1] - apps[j] for j in range(len(apps)-1)]
        mg, sg = np.mean(gaps), np.std(gaps)
        z = (n - apps[-1] - mg) / sg if sg > 0 else 0
        if z > 0.5: scores[num] += z * 1.5

    # Co-occurrence
    pair_freq = Counter()
    for x in data_slice[-200:]:
        for p in combinations(sorted(x[:PICK]), 2):
            pair_freq[p] += 1
    for num in range(1, MAX_NUM + 1):
        scores[num] += sum(pair_freq.get(tuple(sorted([p, num])), 0) for p in last_set) * 0.1

    # KNN
    for i in range(n - 2):
        sim = len(set(data_slice[i][:6]) & last_set)
        if sim >= 3:
            for num in data_slice[i + 1][:6]:
                scores[num] += sim * sim * 0.1

    # Bigram
    bg = defaultdict(Counter)
    for i in range(1, n):
        for pn in data_slice[i-1][:6]:
            for cn in data_slice[i][:6]:
                bg[pn][cn] += 1
    for pn in last:
        t = sum(bg[pn].values())
        if t > 0:
            for nn, cnt in bg[pn].most_common(10):
                scores[nn] += cnt / t

    # Vulnerability serial
    w = min(100, n)
    in_after_in, in_count = Counter(), Counter()
    in_after_out, out_count = Counter(), Counter()
    for i in range(max(0, n-w), n-1):
        curr = set(data_slice[i][:6])
        nxt = set(data_slice[i+1][:6])
        for num in range(1, MAX_NUM + 1):
            if num in curr:
                in_count[num] += 1
                if num in nxt: in_after_in[num] += 1
            else:
                out_count[num] += 1
                if num in nxt: in_after_out[num] += 1
    for num in range(1, MAX_NUM + 1):
        p_in = in_after_in[num] / max(in_count[num], 1)
        p_out = in_after_out[num] / max(out_count[num], 1)
        if num in last_set:
            scores[num] += (p_in - p_out) * 5
        else:
            scores[num] += (p_out - p_in) * 3

    return scores


def build_super_pool(data_slice, scores, max_pool):
    """Build super-pool from 11 diverse pools."""
    n = len(data_slice)
    last_set = set(data_slice[-1][:6])
    ranked = sorted(scores.items(), key=lambda x: -x[1])

    pool_A = set(num for num, _ in ranked[:15])
    freq_30 = Counter(num for d in data_slice[-30:] for num in d[:6])
    pool_B = set(num for num, _ in freq_30.most_common(15))
    freq_50 = Counter(num for d in data_slice[-50:] for num in d[:6])
    pool_C = set(num for num, _ in freq_50.most_common(15))
    freq_100 = Counter(num for d in data_slice[-100:] for num in d[:6])
    pool_D = set(num for num, _ in freq_100.most_common(15))

    last_seen = {}
    for i, d in enumerate(data_slice):
        for num in d[:6]:
            last_seen[num] = i
    pool_E = set(sorted(range(1, MAX_NUM + 1),
                        key=lambda x: -(n - last_seen.get(x, 0)))[:15])

    follow = Counter()
    for i in range(n - 1):
        for p in data_slice[i][:6]:
            if p in last_set:
                for nx in data_slice[i + 1][:6]:
                    follow[nx] += 1
    pool_F = set(num for num, _ in follow.most_common(15))

    knn = Counter()
    for i in range(n - 2):
        sim = len(set(data_slice[i][:6]) & last_set)
        if sim >= 3:
            for num in data_slice[i + 1][:6]:
                knn[num] += sim * sim
    pool_G = set(num for num, _ in knn.most_common(15))

    cooc = Counter()
    for d in data_slice[-200:]:
        ds = set(d[:6])
        overlap = ds & last_set
        if len(overlap) >= 2:
            for num in ds - last_set:
                cooc[num] += len(overlap)
    pool_H = set(num for num, _ in cooc.most_common(15))

    mid_nums = [num for num, _ in ranked[10:30]]
    avg_f = np.mean(list(freq_100.values())) if freq_100 else 0
    pool_I = set(num for num in mid_nums if freq_100.get(num, 0) > avg_f * 0.9)

    pool_J = set()
    for num in last_set:
        for delta in [-2, -1, 1, 2]:
            nb = num + delta
            if 1 <= nb <= MAX_NUM:
                pool_J.add(nb)

    gap_sums = defaultdict(float)
    gap_counts = defaultdict(int)
    ls = {}
    for i, d in enumerate(data_slice):
        for num in d[:6]:
            if num in ls:
                gap_sums[num] += (i - ls[num])
                gap_counts[num] += 1
            ls[num] = i
    pool_K = set()
    for num in range(1, MAX_NUM + 1):
        gc = gap_counts.get(num, 0)
        if gc < 5: continue
        mg = gap_sums[num] / gc
        cg = n - ls.get(num, 0)
        if mg > 0 and cg > mg * 1.2:
            pool_K.add(num)

    raw_union = (pool_A | pool_B | pool_C | pool_D | pool_E |
                 pool_F | pool_G | pool_H | pool_I | pool_J | pool_K)
    super_pool = sorted(raw_union, key=lambda x: -scores.get(x, 0))
    if len(super_pool) > max_pool:
        super_pool = super_pool[:max_pool]
    return sorted(super_pool)


def score_combo(combo, scores):
    """Score a combo by sum of individual number scores."""
    return sum(scores.get(n, 0) for n in combo)


# ================================================================
# BACKTEST: Top-N sets from pool, measure exact match rates
# ================================================================
print(f"Data: {total} draws\n")
print(f"{'='*70}")
print(f" TOP-10 SETS BACKTEST — Can 10 best combos hit 6/6?")
print(f" Testing pool sizes: 10, 12, 14, 16, 18, 20")
print(f"{'='*70}\n")

# Pool sizes to test (smaller = fewer combos = FASTER scoring)
pool_sizes = [10, 12, 14, 16, 18, 20]
N_SETS = 10  # How many sets to pick

# Track results
results = {ps: {'h6': 0, 'h5': 0, 'h4': 0, 'h3': 0, 'total': 0}
           for ps in pool_sizes}

# Also track: how many of 6 actual numbers are in top-N ranked?
top_N_contain = {n: [] for n in [6, 8, 10, 12, 15, 20]}

t0 = time.time()
START = 80
for i in range(START, total - 1):
    d_slice = data[:i+1]
    scores = compute_signals(d_slice)
    actual = set(data[i+1][:6])

    # Check how many actual numbers appear in top-N ranked
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    for n_top in top_N_contain:
        top_set = set(num for num, _ in ranked[:n_top])
        top_N_contain[n_top].append(len(top_set & actual))

    for ps in pool_sizes:
        pool = build_super_pool(d_slice, scores, ps)

        # Generate ALL C(pool, 6) and pick top-10 by score
        all_combos = []
        for combo in combinations(pool, 6):
            sc = score_combo(combo, scores)
            all_combos.append((combo, sc))

        all_combos.sort(key=lambda x: -x[1])
        top_combos = [c for c, _ in all_combos[:N_SETS]]

        # Check matches
        best_match = 0
        for combo in top_combos:
            match = len(set(combo) & actual)
            best_match = max(best_match, match)

        results[ps]['total'] += 1
        if best_match == 6: results[ps]['h6'] += 1
        if best_match >= 5: results[ps]['h5'] += 1
        if best_match >= 4: results[ps]['h4'] += 1
        if best_match >= 3: results[ps]['h3'] += 1

    tested = results[pool_sizes[0]]['total']
    if tested % 50 == 0:
        el = time.time() - t0
        eta = el / tested * (total - START - 1 - tested) / 60
        print(f"  [{tested}/{total-START-1}] {el:.0f}s ETA={eta:.1f}m")
        for ps in pool_sizes:
            r = results[ps]
            print(f"    pool={ps}: 6/6={r['h6']} 5/6={r['h5']} "
                  f"4/6={r['h4']} 3/6={r['h3']}")
        sys.stdout.flush()

# ================================================================
# FINAL RESULTS
# ================================================================
el = time.time() - t0
tested = results[pool_sizes[0]]['total']

print(f"\n{'='*70}")
print(f" RESULTS — {tested} draws, {el:.0f}s")
print(f"{'='*70}\n")

print(f"=== PART 1: Top-10 Combos from Pool (exact backtest) ===\n")
print(f"  {'Pool':>6}  {'C(p,6)':>10}  {'6/6':>8}  {'>=5/6':>8}  {'>=4/6':>8}  {'>=3/6':>8}")
for ps in pool_sizes:
    r = results[ps]
    from math import comb
    cpk = comb(ps, 6)
    print(f"  {ps:>6}  {cpk:>10,}  "
          f"{r['h6']/tested*100:>7.2f}%  "
          f"{r['h5']/tested*100:>7.2f}%  "
          f"{r['h4']/tested*100:>7.2f}%  "
          f"{r['h3']/tested*100:>7.2f}%")

print(f"\n=== PART 2: How many actual numbers in Top-N ranked? ===\n")
print(f"  {'Top-N':>6}  {'Avg in top':>10}  {'All 6 in':>10}  {'>=5 in':>10}  {'>=4 in':>10}")
for n_top in sorted(top_N_contain.keys()):
    vals = top_N_contain[n_top]
    avg = np.mean(vals)
    all6 = sum(1 for v in vals if v == 6) / len(vals) * 100
    ge5 = sum(1 for v in vals if v >= 5) / len(vals) * 100
    ge4 = sum(1 for v in vals if v >= 4) / len(vals) * 100
    print(f"  {n_top:>6}  {avg:>10.3f}  {all6:>9.2f}%  {ge5:>9.2f}%  {ge4:>9.2f}%")

# MATHEMATICAL CEILING
print(f"\n=== PART 3: Mathematical Ceiling ===\n")
print(f"  Random baseline: 10 sets / C(45,6) = 10 / 8,145,060")
print(f"  = {10/8145060*100:.6f}% per draw")
print(f"  = 1 hit per {8145060/10:.0f} draws")
print(f"\n  Even with PERFECT pool of 10 numbers (if all 6 inside):")
print(f"  C(10,6) = {comb(10,6)} combos, picking 10 of 210 = {10/210*100:.1f}% conditional")
print(f"  But P(all 6 in top-10) = {sum(1 for v in top_N_contain[10] if v==6)/len(top_N_contain[10])*100:.2f}%")
print(f"  → Effective 6/6 = both events combined")

print(f"\n{'='*70}")
