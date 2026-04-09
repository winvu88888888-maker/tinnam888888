"""
V14 PURE 6/6 MAXIMIZER — Fast Pool Quality Sweep.

KEY INSIGHT: If we enumerate ALL C(pool, 6) with NO filters,
then 6/6 = TRUE if and only if ALL 6 actual winners are in the pool.
This means we DON'T need to enumerate at all for testing!
Just check: is actual ⊆ pool?

This script sweeps pool sizes 25-40 with 10+ diverse pools
to find the MAXIMUM 6/6 rate achievable.
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

def compute_signals(data_slice, dates_slice):
    """Compute 23 signal scores for each number 1-45."""
    n = len(data_slice)
    last = data_slice[-1][:6]
    last_set = set(last)
    scores = {num: 0.001 for num in range(1, MAX_NUM + 1)}

    # Signal 1: Transition (conditional P given last draw)
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

    # Signal 2: Momentum (short vs long term)
    if n >= 50:
        for num in range(1, MAX_NUM + 1):
            f5 = sum(1 for x in data_slice[-5:] if num in x[:6]) / 5
            f20 = sum(1 for x in data_slice[-20:] if num in x[:6]) / 20
            f50 = sum(1 for x in data_slice[-50:] if num in x[:6]) / 50
            scores[num] += (f5 - f20) * 10 + (f20 - f50) * 5

    # Signal 3: Gap timing (overdue with std deviation)
    for num in range(1, MAX_NUM + 1):
        apps = [i for i, x in enumerate(data_slice) if num in x[:6]]
        if len(apps) < 5: continue
        gaps = [apps[j+1] - apps[j] for j in range(len(apps)-1)]
        mg, sg = np.mean(gaps), np.std(gaps)
        z = (n - apps[-1] - mg) / sg if sg > 0 else 0
        if z > 0.5: scores[num] += z * 1.5

    # Signal 4: Co-occurrence with last draw
    pair_freq = Counter()
    for x in data_slice[-200:]:
        for p in combinations(sorted(x[:PICK]), 2):
            pair_freq[p] += 1
    for num in range(1, MAX_NUM + 1):
        scores[num] += sum(pair_freq.get(tuple(sorted([p, num])), 0) for p in last_set) * 0.1

    # Signal 5: KNN (similar past draws)
    for i in range(n - 2):
        sim = len(set(data_slice[i][:6]) & last_set)
        if sim >= 3:
            for num in data_slice[i + 1][:6]:
                scores[num] += sim * sim * 0.1

    # Signal 6: Bigram (lag-1 conditional)
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

    # Signal 7: Vulnerability - serial correlation
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

def build_super_pool(data_slice, dates_slice, scores, max_pool):
    """Build super-pool from 10 diverse independent pools."""
    n = len(data_slice)
    last_set = set(data_slice[-1][:6])

    ranked = sorted(scores.items(), key=lambda x: -x[1])

    # Pool A: Ensemble (top signal scores)
    pool_A = set(num for num, _ in ranked[:15])

    # Pool B: Frequency recent-30 (very short term)
    freq_30 = Counter(num for d in data_slice[-30:] for num in d[:6])
    pool_B = set(num for num, _ in freq_30.most_common(15))

    # Pool C: Frequency recent-50
    freq_50 = Counter(num for d in data_slice[-50:] for num in d[:6])
    pool_C = set(num for num, _ in freq_50.most_common(15))

    # Pool D: Frequency recent-100
    freq_100 = Counter(num for d in data_slice[-100:] for num in d[:6])
    pool_D = set(num for num, _ in freq_100.most_common(15))

    # Pool E: Overdue (longest absence)
    last_seen = {}
    for i, d in enumerate(data_slice):
        for num in d[:6]:
            last_seen[num] = i
    pool_E = set(sorted(range(1, MAX_NUM + 1),
                        key=lambda x: -(n - last_seen.get(x, 0)))[:15])

    # Pool F: Transition (conditional P given last draw)
    follow = Counter()
    for i in range(n - 1):
        for p in data_slice[i][:6]:
            if p in last_set:
                for nx in data_slice[i + 1][:6]:
                    follow[nx] += 1
    pool_F = set(num for num, _ in follow.most_common(15))

    # Pool G: KNN best (from most similar past draws)
    knn = Counter()
    for i in range(n - 2):
        sim = len(set(data_slice[i][:6]) & last_set)
        if sim >= 3:
            for num in data_slice[i + 1][:6]:
                knn[num] += sim * sim
    pool_G = set(num for num, _ in knn.most_common(15))

    # Pool H: Co-occurrence partners
    cooc = Counter()
    for d in data_slice[-200:]:
        ds = set(d[:6])
        overlap = ds & last_set
        if len(overlap) >= 2:
            for num in ds - last_set:
                cooc[num] += len(overlap)
    pool_H = set(num for num, _ in cooc.most_common(15))

    # Pool I: Rescue (anti-consensus: mid-ranked but frequent)
    mid_nums = [num for num, _ in ranked[10:30]]
    avg_f = np.mean(list(freq_100.values())) if freq_100 else 0
    pool_I = set(num for num in mid_nums if freq_100.get(num, 0) > avg_f * 0.9)

    # Pool J: Last draw neighbors (±1, ±2 of each last number)
    pool_J = set()
    for num in last_set:
        for delta in [-2, -1, 1, 2]:
            nb = num + delta
            if 1 <= nb <= MAX_NUM:
                pool_J.add(nb)

    # Pool K: Stat overdue (numbers past their mean gap)
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

    # UNION all pools
    raw_union = (pool_A | pool_B | pool_C | pool_D | pool_E |
                 pool_F | pool_G | pool_H | pool_I | pool_J | pool_K)

    # Rank and cap
    super_pool = sorted(raw_union, key=lambda x: -scores.get(x, 0))
    if len(super_pool) > max_pool:
        super_pool = set(super_pool[:max_pool])
    else:
        super_pool = set(super_pool)

    return super_pool

# ================================================================
# SWEEP: test multiple pool sizes
# ================================================================
print(f"Data: {total} draws\n")
print(f"{'='*70}")
print(f" V14 PURE 6/6 MAXIMIZER — Pool Size Sweep")
print(f" (11 diverse pools, no filters, 6/6 = pool containment)")
print(f"{'='*70}\n")

pool_sizes = [28, 30, 32, 34, 36, 38, 40]
results = {ps: {'hits6': 0, 'hits5': 0, 'total': 0} for ps in pool_sizes}

t0 = time.time()
for i in range(80, total - 1):
    d_slice = data[:i+1]
    dt_slice = dates[:i+1]
    scores = compute_signals(d_slice, dt_slice)
    actual = set(data[i+1][:6])

    for ps in pool_sizes:
        pool = build_super_pool(d_slice, dt_slice, scores, ps)
        overlap = len(pool & actual)
        results[ps]['total'] += 1
        if overlap == 6:
            results[ps]['hits6'] += 1
        elif overlap == 5:
            results[ps]['hits5'] += 1

    tested = results[pool_sizes[0]]['total']
    if tested % 100 == 0:
        el = time.time() - t0
        eta = el / tested * (total - 81 - tested) / 60
        line = f"  [{tested}/{total-81}] {el:.0f}s ETA={eta:.1f}m | "
        for ps in pool_sizes:
            r = results[ps]
            pct = r['hits6'] / r['total'] * 100
            line += f"P{ps}={r['hits6']}({pct:.1f}%) "
        print(line)
        sys.stdout.flush()

# Final results
el = time.time() - t0
tested = results[pool_sizes[0]]['total']
print(f"\n{'='*70}")
print(f" V14 RESULTS — {tested} draws, {el:.0f}s ({el/60:.1f}m)")
print(f"{'='*70}\n")
print(f"  {'Pool':>6}  {'6/6 Hits':>10}  {'6/6 Rate':>10}  {'5/6 Hits':>10}  {'>=5/6 Rate':>10}")
for ps in pool_sizes:
    r = results[ps]
    rate6 = r['hits6'] / r['total'] * 100
    rate5p = (r['hits6'] + r['hits5']) / r['total'] * 100
    print(f"  {ps:>6}  {r['hits6']:>6}/{tested}  {rate6:>8.2f}%  "
          f"{r['hits5']:>6}/{tested}  {rate5p:>8.2f}%")
print(f"\n{'='*70}")
