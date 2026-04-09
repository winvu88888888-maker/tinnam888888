"""
V11 DEEP REVERSE ANALYSIS — Pool quality measurement.
For every draw, measure: how many of 6 winners are in our top-K pool?
This tells us the CEILING for 6/6 at each pool size.
"""
import sys, time, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter, defaultdict
from models.ultimate_engine import UltimateEngine
from scraper.data_manager import get_mega645_numbers, get_mega645_all

data = get_mega645_numbers()
dates = [r['draw_date'] for r in get_mega645_all()]
total = len(data)
engine = UltimateEngine(45, 6)

print(f"Data: {total} draws\n")
print(f"{'='*70}")
print(f" POOL QUALITY ANALYSIS — How many of 6 winners in top-K?")
print(f"{'='*70}\n")

# For each draw (i=80..total-1), get engine's top-K pool
# Then check how many of 6 actual winners are in that pool
pool_hits = {k: [] for k in [10, 12, 15, 18, 20, 22, 25, 30, 35]}
all6_in_k = {k: 0 for k in pool_hits}
all5_in_k = {k: 0 for k in pool_hits}

t0 = time.time()
step = max(1, (total-81) // 200)  # Sample ~200 draws for speed
tested = 0
for i in range(80, total-1, step):
    r = engine.predict(data[:i+1], dates[:i+1])
    actual = set(data[i+1][:6])
    pool = r['top_30']
    
    # Extend pool with rescue nums and overdue
    rescue_nums = [rn[0] for rn in r.get('rescue_nums', [])]
    
    for k in pool_hits:
        pk = set(pool[:min(k, len(pool))])
        # Also add rescue nums to expand the pool
        if k >= 20:
            pk.update(rescue_nums[:min(5, k-15)])
        hits = len(pk & actual)
        pool_hits[k].append(hits)
        if hits == 6:
            all6_in_k[k] += 1
        if hits >= 5:
            all5_in_k[k] += 1
    
    tested += 1
    if tested % 50 == 0:
        print(f"  [{tested}] {time.time()-t0:.0f}s")
        sys.stdout.flush()

print(f"\n  Tested: {tested} draws ({time.time()-t0:.0f}s)\n")

print(f"{'='*70}")
print(f" RESULTS: P(all 6 in top-K)")
print(f"{'='*70}\n")
print(f"  {'K':>4}  {'All 6':>10}  {'>=5':>10}  {'Avg hits':>10}  {'C(K,6)':>12}  {'Filter~':>10}")
from math import comb
for k in sorted(pool_hits.keys()):
    a6 = all6_in_k[k]
    a5 = all5_in_k[k]
    avg = np.mean(pool_hits[k])
    ck = comb(k, 6)
    filt = int(ck * 0.15)  # ~15% survive constraints
    print(f"  {k:>4}  {a6:>5}/{tested} ({a6/tested*100:5.1f}%)  "
          f"{a5:>5}/{tested} ({a5/tested*100:5.1f}%)  "
          f"{avg:>6.2f}/6  {ck:>12,}  ~{filt:>8,}")

print(f"\n{'='*70}")
print(f" MATHEMATICAL LIMITS")
print(f"{'='*70}\n")

for k in [15, 18, 20, 22, 25, 30]:
    ck = comb(k, 6)
    filt = int(ck * 0.15)
    p_all6 = all6_in_k[k] / tested
    p_with_enum = p_all6  # If we enumerate all valid combos from top-K
    expected_per_1400 = 1405 * p_with_enum
    print(f"  Top-{k}: P(all6)={p_all6*100:.1f}% → If enumerate {filt:,} valid → "
          f"Expected 6/6 in 1405 draws: {expected_per_1400:.1f}")

print(f"\n  To reach 50% 6/6 rate, need P(all 6 in pool) >= 50%")
print(f"  Current best: top-35 → {all6_in_k[35]/tested*100:.1f}%")
print(f"  Need to improve top-30 pool quality by ~{50/(all6_in_k[30]/tested*100+0.01):.0f}x")

# Also check: what if we use MULTIPLE pools?
# Union of top-15 from different signal groups
print(f"\n{'='*70}")
print(f" MULTI-POOL UNION ANALYSIS")
print(f"{'='*70}\n")

# For each draw, get scores from multiple signal subsets
# Then check if UNION of their top-15 captures all 6
t0 = time.time()
union_hits = {20: [], 25: [], 30: [], 35: []}
union6 = {k: 0 for k in union_hits}

for i in range(80, total-1, step):
    r = engine.predict(data[:i+1], dates[:i+1])
    actual = set(data[i+1][:6])
    
    # Pool 1: ensemble (standard scoring)
    pool1 = set(r['top_30'][:15])
    
    # Pool 2: frequency-based (last 50 draws)
    freq_50 = Counter(num for d in data[:i+1][-50:] for num in d[:6])
    pool2 = set(num for num, _ in freq_50.most_common(15))
    
    # Pool 3: overdue-based
    last_seen = {}
    for ii, d in enumerate(data[:i+1]):
        for num in d[:6]:
            last_seen[num] = ii
    overdue = sorted(range(1, 46), key=lambda x: -(i+1 - last_seen.get(x, 0)))
    pool3 = set(overdue[:15])
    
    # Pool 4: transition-based (conditional probability)
    last_set = set(data[i][:6])
    follow = Counter()
    for ii in range(len(data[:i+1]) - 1):
        for p in data[ii][:6]:
            if p in last_set:
                for nx in data[ii+1][:6]:
                    follow[nx] += 1
    pool4 = set(num for num, _ in follow.most_common(15))
    
    # Pool 5: rescue numbers
    rescue = set(rn[0] for rn in r.get('rescue_nums', [])[:10])
    
    for target_k in union_hits:
        if target_k == 20:
            union = pool1 | pool2
        elif target_k == 25:
            union = pool1 | pool2 | pool3
        elif target_k == 30:
            union = pool1 | pool2 | pool3 | pool4
        else:  # 35
            union = pool1 | pool2 | pool3 | pool4 | rescue
        
        hits = len(union & actual)
        union_hits[target_k].append(hits)
        if hits == 6:
            union6[target_k] += 1

print(f"  MULTI-POOL UNION: P(all 6 in union)")
print(f"  {'Union size':>12}  {'All 6':>15}  {'Avg hits':>10}")
for k in sorted(union_hits.keys()):
    a6 = union6[k]
    avg = np.mean(union_hits[k])
    print(f"  ~{k:>3} numbers  {a6:>5}/{tested} ({a6/tested*100:5.1f}%)  {avg:>6.2f}/6")

print(f"\n{'='*70}")
