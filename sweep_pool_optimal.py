"""
FAST POOL CONTAINMENT SWEEP — Find optimal pool size for max 6/6.

KEY MATH:
  Expected 6/6 = P(all 6 in pool) × P(portfolio hits 6/6 | pool contains all 6)
  P(portfolio hits) ≈ N_tickets / C(pool_size, 6)
  
  Expected = containment_rate × tickets / C(pool, 6)
  
  We test pool sizes 18-40 to find which maximizes EXPECTED 6/6.
  Uses test_v14_sweep's proven 11-pool diversity system.
"""
import sys, os, time, math
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper.data_manager import get_mega645_numbers, get_mega645_all

MAX_NUM = 45
PICK = 6


def compute_signals(data_slice):
    """Compute per-number scores (compact 7-signal engine)."""
    n = len(data_slice)
    last_set = set(data_slice[-1][:6])
    scores = {num: 0.001 for num in range(1, MAX_NUM + 1)}
    base = PICK / MAX_NUM

    # S1: Transition
    follow = defaultdict(Counter)
    pc = Counter()
    for i in range(n - 1):
        for p in data_slice[i][:6]:
            pc[p] += 1
            for nx in data_slice[i + 1][:6]:
                follow[p][nx] += 1
    for num in range(1, MAX_NUM + 1):
        prob = sum(follow[p].get(num, 0) for p in last_set) / max(sum(pc[p] for p in last_set), 1)
        scores[num] += max(0, (prob / base - 1) * 3)

    # S2: Multi-scale frequency
    for w, wt in [(3, 4.0), (5, 3.0), (10, 2.0), (20, 1.5), (50, 1.0)]:
        if n < w: continue
        for num in range(1, MAX_NUM + 1):
            f = sum(1 for d in data_slice[-w:] if num in d[:6]) / w
            scores[num] += f * wt

    # S3: Gap timing
    for num in range(1, MAX_NUM + 1):
        apps = [i for i, d in enumerate(data_slice) if num in d[:6]]
        if len(apps) < 5: continue
        gaps = [apps[j+1]-apps[j] for j in range(len(apps)-1)]
        mg, sg = np.mean(gaps), np.std(gaps)
        z = (n - apps[-1] - mg) / sg if sg > 0 else 0
        if z > 0.5: scores[num] += z * 1.5

    # S4: KNN
    knn = Counter()
    for i in range(n - 2):
        sim = len(set(data_slice[i][:6]) & last_set)
        if sim >= 3:
            for num in data_slice[i + 1][:6]:
                knn[num] += sim * sim
    mx = max(knn.values()) if knn else 1
    for num in range(1, MAX_NUM + 1):
        scores[num] += knn.get(num, 0) / mx * 2.5

    # S5: Serial correlation
    w = min(100, n)
    in_after_in, in_count = Counter(), Counter()
    for i in range(max(0, n-w), n-1):
        curr = set(data_slice[i][:6])
        nxt = set(data_slice[i+1][:6])
        for num in range(1, MAX_NUM + 1):
            if num in curr:
                in_count[num] += 1
                if num in nxt: in_after_in[num] += 1
    for num in range(1, MAX_NUM + 1):
        if num in last_set and in_count[num] > 5:
            p_in = in_after_in[num] / in_count[num]
            scores[num] += (p_in - base) * 5

    return scores


def build_super_pool(data_slice, scores, max_pool):
    """Build super-pool from 11 diverse pools (proven from test_v14_sweep)."""
    n = len(data_slice)
    last_set = set(data_slice[-1][:6])
    ranked = sorted(scores.items(), key=lambda x: -x[1])

    # Pool A: Top signals
    pool_A = set(num for num, _ in ranked[:15])
    # Pool B: Freq-30
    freq_30 = Counter(num for d in data_slice[-30:] for num in d[:6])
    pool_B = set(num for num, _ in freq_30.most_common(15))
    # Pool C: Freq-50
    freq_50 = Counter(num for d in data_slice[-50:] for num in d[:6])
    pool_C = set(num for num, _ in freq_50.most_common(15))
    # Pool D: Freq-100
    freq_100 = Counter(num for d in data_slice[-min(100,n):] for num in d[:6])
    pool_D = set(num for num, _ in freq_100.most_common(15))
    # Pool E: Overdue
    last_seen = {}
    for i, d in enumerate(data_slice):
        for num in d[:6]: last_seen[num] = i
    pool_E = set(sorted(range(1, MAX_NUM+1), key=lambda x: -(n - last_seen.get(x, 0)))[:15])
    # Pool F: Transition followers
    follow_c = Counter()
    for i in range(n-1):
        for p in data_slice[i][:6]:
            if p in last_set:
                for nx in data_slice[i+1][:6]:
                    follow_c[nx] += 1
    pool_F = set(num for num, _ in follow_c.most_common(15))
    # Pool G: KNN
    knn_c = Counter()
    for i in range(n-2):
        sim = len(set(data_slice[i][:6]) & last_set)
        if sim >= 3:
            for num in data_slice[i+1][:6]:
                knn_c[num] += sim * sim
    pool_G = set(num for num, _ in knn_c.most_common(15))
    # Pool H: Co-occurrence
    cooc = Counter()
    for d in data_slice[-200:]:
        ds = set(d[:6])
        overlap = ds & last_set
        if len(overlap) >= 2:
            for num in ds - last_set:
                cooc[num] += len(overlap)
    pool_H = set(num for num, _ in cooc.most_common(15))
    # Pool I: Rescue (mid-ranked but frequent)
    mid = [num for num, _ in ranked[10:30]]
    avg_f = np.mean(list(freq_100.values())) if freq_100 else 0
    pool_I = set(num for num in mid if freq_100.get(num, 0) > avg_f * 0.9)
    # Pool J: Neighbors
    pool_J = set()
    for num in last_set:
        for delta in [-2, -1, 1, 2]:
            nb = num + delta
            if 1 <= nb <= MAX_NUM: pool_J.add(nb)
    # Pool K: Stat overdue
    gap_sums, gap_counts, ls = defaultdict(float), defaultdict(int), {}
    for i, d in enumerate(data_slice):
        for num in d[:6]:
            if num in ls:
                gap_sums[num] += (i - ls[num])
                gap_counts[num] += 1
            ls[num] = i
    pool_K = set()
    for num in range(1, MAX_NUM+1):
        gc = gap_counts.get(num, 0)
        if gc < 5: continue
        mg = gap_sums[num] / gc
        cg = n - ls.get(num, 0)
        if mg > 0 and cg > mg * 1.2:
            pool_K.add(num)

    raw = pool_A | pool_B | pool_C | pool_D | pool_E | pool_F | pool_G | pool_H | pool_I | pool_J | pool_K
    ranked_union = sorted(raw, key=lambda x: -scores.get(x, 0))
    return set(ranked_union[:max_pool]) if len(ranked_union) > max_pool else set(ranked_union)


# ============ SWEEP ============
data = get_mega645_numbers()
total = len(data)
t0 = time.time()

print(f"Data: {total} draws")
print(f"{'='*80}")
print(f" POOL CONTAINMENT + EXPECTED 6/6 SWEEP")
print(f" Testing pool sizes 18-40 with 11 diverse pools")
print(f"{'='*80}\n")

pool_sizes = [18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
results = {ps: {'contain': 0, 'hits5': 0, 'total': 0} for ps in pool_sizes}
WARMUP = 200

for i in range(WARMUP, total - 1):
    d_slice = data[:i+1]
    scores = compute_signals(d_slice)
    actual = set(data[i+1][:6])

    for ps in pool_sizes:
        pool = build_super_pool(d_slice, scores, ps)
        overlap = len(pool & actual)
        results[ps]['total'] += 1
        if overlap == 6: results[ps]['contain'] += 1
        elif overlap == 5: results[ps]['hits5'] += 1

    tested = results[pool_sizes[0]]['total']
    if tested % 200 == 0:
        el = time.time() - t0
        eta = el / tested * (total - WARMUP - 1 - tested) / 60
        line = f"  [{tested}/{total-WARMUP-1}] {el:.0f}s ETA={eta:.1f}m | "
        for ps in [22, 26, 30, 34, 38]:
            r = results[ps]
            pct = r['contain'] / max(r['total'],1) * 100
            line += f"P{ps}={r['contain']}({pct:.1f}%) "
        print(line)
        sys.stdout.flush()

# ============ ANALYSIS ============
el = time.time() - t0
tested = results[pool_sizes[0]]['total']

print(f"\n{'='*80}")
print(f" RESULTS — {tested} draws, {el:.0f}s")
print(f"{'='*80}\n")

ticket_counts = [500, 1000, 2000, 3000, 5000]

print(f"  {'Pool':>6}  {'C(P,6)':>12}  {'6/6 contain':>12}  {'5/6':>6}  ", end='')
for tc in ticket_counts:
    print(f"{'E[6/6] @'+str(tc):>14}", end='')
print()

best_expected = {}
for ps in pool_sizes:
    r = results[ps]
    cpk = math.comb(ps, 6)
    contain_rate = r['contain'] / max(r['total'], 1)
    contain_pct = contain_rate * 100
    hits5_pct = r['hits5'] / max(r['total'], 1) * 100
    
    print(f"  {ps:6d}  {cpk:12,d}  {r['contain']:5d} ({contain_pct:5.2f}%)  {r['hits5']:5d}  ", end='')
    for tc in ticket_counts:
        # Expected 6/6 = containment_rate × tickets / C(pool,6)
        expected = contain_rate * tc / cpk * tested
        expected_pct = contain_rate * tc / cpk * 100
        print(f"  {expected_pct:7.4f}% ({expected:4.1f})", end='')
        best_expected[(ps, tc)] = expected_pct
    print()

# Find optimal
print(f"\n  {'='*60}")
print(f"  OPTIMAL POOL SIZE per ticket count:")
print(f"  {'='*60}")
for tc in ticket_counts:
    best_ps = max(pool_sizes, key=lambda ps: best_expected.get((ps, tc), 0))
    best_val = best_expected.get((best_ps, tc), 0)
    print(f"    {tc:5d} tickets → Pool {best_ps:2d}: E[6/6] = {best_val:.4f}%")

print(f"\n  Time: {el:.1f}s")
