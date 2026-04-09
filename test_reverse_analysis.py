"""
REVERSE ENGINEERING 6/6 FAILURE

For EVERY draw in history, analyze:
1. The 5/6 near-misses: what was the 1 missing number? Where was it ranked?
2. How many of 6 winners were in top-K pool? (K=10,15,20,25,30)
3. The BEST portfolio match: which number was swapped to get 6/6?
4. Was missing number from the last draw (repeat that we excluded)?
5. Pattern of misses: decade, odd/even, position

Goal: Find the EXACT bottleneck blocking 6/6.
"""
import sys, os, time
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
print(f" REVERSE ENGINEERING: Why can't we hit 6/6?")
print(f"{'='*70}")

# Collect detailed stats for every draw
five_of_six = []          # Details of 5/6 hits
four_of_six = []          # Details of 4/6 hits
missed_numbers = Counter() # Which numbers we miss most
missed_decade = Counter()  # Which decade we miss most
missed_was_repeat = 0      # How often miss was from last draw
missed_was_in_pool = 0     # How often miss was in our top-30 pool
missed_rank = []           # Rank of missed number in our scoring
all_pool_containment = []  # How many of 6 winners in top-K

t0 = time.time()
tested = 0
for i in range(80, total - 1):
    r = engine.predict(data[:i+1], dates[:i+1], 500)
    actual = set(data[i+1])
    last_draw = set(data[i])
    pool_30 = r['top_30']
    
    # Best match in portfolio
    best_match = 0
    best_set = None
    for p in r['portfolio']:
        h = len(set(p['numbers']) & actual)
        if h > best_match:
            best_match = h
            best_set = p
    
    tested += 1
    
    # Pool containment
    for k in [15, 20, 25, 30]:
        pool_k = set(pool_30[:k]) if k <= len(pool_30) else set(pool_30)
        hits = len(pool_k & actual)
        if k == 30:
            all_pool_containment.append(hits)
    
    if best_match >= 5:
        predicted = set(best_set['numbers'])
        missing = actual - predicted  # What we should have had
        extra = predicted - actual    # What we predicted wrong
        
        for m in missing:
            missed_numbers[m] += 1
            # Decade
            if m <= 9: missed_decade['01-09'] += 1
            elif m <= 19: missed_decade['10-19'] += 1
            elif m <= 29: missed_decade['20-29'] += 1
            elif m <= 39: missed_decade['30-39'] += 1
            else: missed_decade['40-45'] += 1
            
            # Was it a repeat?
            if m in last_draw: missed_was_repeat += 1
            
            # Was it in our top-30 pool?
            if m in set(pool_30[:30]): missed_was_in_pool += 1
            
            # What rank was it?
            scores = r['scores']
            if m in scores:
                rank = sorted(scores.keys(), key=lambda x: -scores[x]).index(m) + 1
                missed_rank.append(rank)
            else:
                missed_rank.append(99)  # Not even in top 30
        
        info = {
            'draw': i+1,
            'actual': sorted(actual),
            'predicted': sorted(predicted),
            'missing': sorted(missing),
            'extra': sorted(extra),
            'strategy': best_set['strategy'],
            'missing_in_pool30': all(m in set(pool_30[:30]) for m in missing),
            'missing_in_last': any(m in last_draw for m in missing),
            'match': best_match,
        }
        
        if best_match == 5:
            five_of_six.append(info)
        # Also track 4/6 cases
        
    if best_match >= 4:
        predicted = set(best_set['numbers'])
        missing = actual - predicted
        for m in missing:
            if m in last_draw:
                pass
        four_of_six.append({
            'match': best_match,
            'missing_count': len(missing),
            'missing_in_last': len(missing & last_draw),
        })
    
    if tested % 300 == 0:
        print(f"  [{tested}/{total-81}] {time.time()-t0:.0f}s")

print(f"\n Total: {tested} draws, {time.time()-t0:.0f}s\n")

# ============================================
# ANALYSIS 1: 5/6 Near-Miss Autopsy
# ============================================
print(f"{'='*70}")
print(f" ANALYSIS 1: 5/6 NEAR-MISS AUTOPSY ({len(five_of_six)} cases)")
print(f"{'='*70}\n")
for case in five_of_six:
    print(f"  Draw #{case['draw']}: [{case['strategy']}]")
    print(f"    Actual:    {case['actual']}")
    print(f"    Predicted: {case['predicted']}")
    print(f"    MISSING:   {case['missing']} ← this blocked 6/6")
    print(f"    Extra:     {case['extra']} ← this was wrong")
    print(f"    Miss in pool-30: {case['missing_in_pool30']}")
    print(f"    Miss was repeat: {case['missing_in_last']}")
    print()

# ============================================
# ANALYSIS 2: Missing Number Patterns
# ============================================
print(f"{'='*70}")
print(f" ANALYSIS 2: WHAT WE MISS")
print(f"{'='*70}\n")

n_5of6 = len(five_of_six)
if n_5of6 > 0:
    print(f"  Most frequently missed numbers (in 5/6 cases):")
    for num, cnt in missed_numbers.most_common(10):
        print(f"    #{num:2d}: missed {cnt} times")
    
    print(f"\n  Missed by decade:")
    for dec in ['01-09', '10-19', '20-29', '30-39', '40-45']:
        c = missed_decade.get(dec, 0)
        print(f"    {dec}: {c} times ({c/max(n_5of6,1)*100:.0f}%)")
    
    print(f"\n  Was missing number a REPEAT from last draw?")
    print(f"    Yes: {missed_was_repeat}/{n_5of6} ({missed_was_repeat/max(n_5of6,1)*100:.0f}%)")
    
    print(f"\n  Was missing number in our TOP-30 pool?")
    print(f"    Yes: {missed_was_in_pool}/{n_5of6} ({missed_was_in_pool/max(n_5of6,1)*100:.0f}%)")
    
    if missed_rank:
        print(f"\n  Rank of missed number in our scoring:")
        print(f"    Mean rank: {np.mean(missed_rank):.1f}")
        print(f"    Median rank: {np.median(missed_rank):.0f}")
        for r in sorted(set(missed_rank)):
            c = missed_rank.count(r)
            label = f"rank {r}" if r < 99 else "NOT IN TOP 30"
            print(f"    {label}: {c} times")

# ============================================
# ANALYSIS 3: Pool Containment
# ============================================
print(f"\n{'='*70}")
print(f" ANALYSIS 3: POOL QUALITY (all 6 in top-K?)")
print(f"{'='*70}\n")
cont = Counter(all_pool_containment)
print(f"  How many of 6 winners in top-30:")
for h in range(7):
    c = cont.get(h, 0)
    print(f"    {h}/6: {c:5d} ({c/tested*100:5.1f}%)")
all6_in30 = sum(1 for x in all_pool_containment if x == 6)
print(f"\n  ALL 6 in top-30: {all6_in30}/{tested} = {all6_in30/tested*100:.1f}%")
print(f"  → This is the CEILING for 6/6 with top-30 pool")

# ============================================
# ANALYSIS 4: What would it take for 6/6?
# ============================================
print(f"\n{'='*70}")
print(f" ANALYSIS 4: MATHEMATICAL PATH TO 6/6")
print(f"{'='*70}\n")

# If all 6 are in top-30 (8.2% of time):
# C(30,6) = 593,775 combos
# With 500 sets: coverage = 500/593775 = 0.084%
# P(6/6 | all6 in pool) = 500/593775 = 0.00084
# Expected hits in 1405 draws: 1405 * 0.082 * 0.00084 = 0.097

from math import comb
c30 = comb(30, 6)
c25 = comb(25, 6)
c20 = comb(20, 6)
c15 = comb(15, 6)

print(f"  If all 6 winners are in pool of size K:")
print(f"  And we generate N sets from that pool:")
print(f"")
print(f"  {'Pool':>6} {'C(K,6)':>10} {'500 sets':>12} {'1000':>10} {'2000':>10} {'5000':>10}")
for k, ck in [(15, c15), (20, c20), (25, c25), (30, c30)]:
    for n_sets in [500, 1000, 2000, 5000]:
        p = n_sets / ck
        if k == 15:
            end = ''
        print(f"  top-{k:>2} {ck:>10,} {500/ck*100:>10.4f}% {1000/ck*100:>8.4f}% {2000/ck*100:>8.4f}% {5000/ck*100:>8.4f}%")
        break

print()
for k, ck in [(15, c15), (20, c20), (25, c25), (30, c30)]:
    # How often all 6 in top-K?
    # From our data: approximate
    all6_in_k = sum(1 for x in all_pool_containment if x >= 6)  # Only have top-30 data
    p_all6 = all6_in30 / tested if k == 30 else 0
    # Approximate for smaller pools
    if k == 25: p_all6 = all6_in30 / tested * 0.35  # rough estimate
    if k == 20: p_all6 = all6_in30 / tested * 0.07  
    if k == 15: p_all6 = all6_in30 / tested * 0.008
    
    for n_sets in [500, 1000, 2000, 5000]:
        p_hit = n_sets / ck
        p_total = p_all6 * p_hit
        expected_per_1400 = 1405 * p_total
        print(f"  top-{k}, {n_sets} sets: P(6/6)={p_total*100:.6f}% → "
              f"Expected in 1405 draws: {expected_per_1400:.3f}")

# ============================================
# ANALYSIS 5: 4/6 → What's needed for 6/6?
# ============================================
print(f"\n{'='*70}")
print(f" ANALYSIS 5: FROM 4/6 TO 6/6")
print(f"{'='*70}\n")
n_4of6 = sum(1 for x in four_of_six if x['match'] == 4)
n_4of6_miss_repeat = sum(1 for x in four_of_six if x['match'] == 4 and x['missing_in_last'] > 0)
print(f"  4/6 cases: {n_4of6}")
print(f"  4/6 where missing nums include repeat: {n_4of6_miss_repeat} ({n_4of6_miss_repeat/max(n_4of6,1)*100:.0f}%)")
print(f"  → In 4/6, we miss 2 numbers. If we could get just 1 more right → 5/6")
print(f"  → To go 5/6 → 6/6, need the EXACT right swap")

print(f"\n{'='*70}")
print(f" CONCLUSION & PROPOSED SOLUTIONS")
print(f"{'='*70}")
print(f"""
  ROOT CAUSES of 6/6 failure:
  1. POOL QUALITY: Only {all6_in30/tested*100:.1f}% of draws have all 6 in top-30
     → We can NEVER hit 6/6 for the other {100-all6_in30/tested*100:.1f}%
  
  2. COMBINATORIAL EXPLOSION: Even when all 6 are in top-30,
     C(30,6) = {c30:,} combos vs our 500 sets = {500/c30*100:.4f}% coverage
  
  3. The 6th number is typically ranked {np.mean(missed_rank):.0f}th 
     (if available) — it's often outside our primary focus
  
  PROPOSED V6 SOLUTIONS:
  A. CONCENTRATE POOL: If we can reliably get all 6 into top-20,
     C(20,6) = {c20:,} and 2000 sets = {2000/c20*100:.3f}% coverage
  
  B. SMART ENUMERATION: For top-15 (C=5,005), we can ENUMERATE 
     ALL {c15:,} combos! If all 6 are in top-15, we GUARANTEE 6/6!
  
  C. HYBRID: Generate {c15:,} sets from top-15 + 500 from wider pools
     Total ~5,500 sets = covers top-15 exhaustively + hedges wider
""")
print(f"{'='*70}")
