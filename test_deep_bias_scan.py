"""
V5 DEEP BIAS SCAN — Find REAL exploitable patterns in Vietlott data.

Tests:
1. ELIMINATION POWER: If we eliminate numbers via various rules, how small can the pool get
   while still containing all 6 winners?
2. STRUCTURAL PATTERNS: Sum mod, decade distribution, consecutive patterns
3. CONDITIONAL PROBABILITY: Under what conditions do specific groups of numbers appear?
4. ANTI-REPEAT POWER: How many draws before a number repeats? Best exclusion window?
5. ZONE PATTERNS: Hot/cold zones that actually persist
"""
import sys, os, time
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from scraper.data_manager import get_mega645_numbers, get_mega645_all

data = get_mega645_numbers()
total = len(data)
print(f"Data: {total} draws\n")

# ==========================================
# TEST 1: ELIMINATION POWER
# How effective are various exclusion rules?
# ==========================================
print(f"{'='*70}")
print(f" TEST 1: ELIMINATION POWER")
print(f" Goal: Find rules that reliably exclude numbers")
print(f"{'='*70}\n")

# Rule A: Exclude numbers from last draw
rule_a_safe = 0
rule_a_pool = []
for i in range(1, total):
    excluded = set(data[i-1])
    actual = set(data[i])
    safe = len(actual - excluded) == 6  # All 6 are NOT in excluded
    if safe: rule_a_safe += 1
    pool_size = 45 - len(excluded)
    rule_a_pool.append(pool_size)
rate_a = rule_a_safe / (total-1) * 100
print(f"  Rule A: Exclude last draw (6 nums)")
print(f"    Safe rate: {rule_a_safe}/{total-1} = {rate_a:.1f}%")
print(f"    Pool size: 39 (eliminated 6)")

# Rule B: Exclude last 2 draws
rule_b_safe = 0
for i in range(2, total):
    excluded = set(data[i-1]) | set(data[i-2])
    actual = set(data[i])
    if len(actual & excluded) == 0: rule_b_safe += 1
rate_b = rule_b_safe / (total-2) * 100
print(f"\n  Rule B: Exclude last 2 draws")
print(f"    Safe rate: {rule_b_safe}/{total-2} = {rate_b:.1f}%")
print(f"    Pool size: ~33 (eliminated ~12)")

# Rule C: Exclude numbers that appeared in last 3 draws
rule_c_safe = 0
for i in range(3, total):
    excluded = set(data[i-1]) | set(data[i-2]) | set(data[i-3])
    actual = set(data[i])
    if len(actual & excluded) == 0: rule_c_safe += 1
rate_c = rule_c_safe / (total-3) * 100
print(f"\n  Rule C: Exclude last 3 draws")
print(f"    Safe rate: {rule_c_safe}/{total-3} = {rate_c:.1f}%")
print(f"    Pool size: ~27 (eliminated ~18)")

# Rule D: What's the OPTIMAL exclusion window?
print(f"\n  Rule D: Optimal exclusion window")
for window in [1, 2, 3, 4, 5]:
    safe = 0
    for i in range(window, total):
        excluded = set()
        for w in range(1, window+1):
            excluded |= set(data[i-w])
        actual = set(data[i])
        if len(actual & excluded) == 0: safe += 1
    rate = safe / (total-window) * 100
    avg_pool = 45 - window*6*(1 - (window-1)*6/45/2)  # approximate
    print(f"    Window {window}: safe={rate:.1f}%, unique excluded≈{len(excluded)}")

# ==========================================
# TEST 2: REPEAT ANALYSIS
# ==========================================
print(f"\n{'='*70}")
print(f" TEST 2: REPEAT ANALYSIS")
print(f"{'='*70}\n")

# How many numbers repeat from draw N to N+1?
repeat_dist = Counter()
for i in range(1, total):
    repeats = len(set(data[i]) & set(data[i-1]))
    repeat_dist[repeats] += 1
print(f"  Repeats from draw N to N+1:")
for k in range(7):
    c = repeat_dist.get(k, 0)
    pct = c / (total-1) * 100
    print(f"    {k} nums repeat: {c:5d} ({pct:.1f}%)")

# Average gap per number
all_gaps = defaultdict(list)
last_seen = {}
for i, draw in enumerate(data):
    for num in draw:
        if num in last_seen:
            all_gaps[num].append(i - last_seen[num])
        last_seen[num] = i

print(f"\n  Gap statistics per number:")
avg_gaps = {num: np.mean(gaps) for num, gaps in all_gaps.items() if gaps}
std_gaps = {num: np.std(gaps) for num, gaps in all_gaps.items() if gaps}
gap_values = list(avg_gaps.values())
print(f"    Overall avg gap: {np.mean(gap_values):.2f} draws")
print(f"    Gap range: {min(gap_values):.1f} - {max(gap_values):.1f}")
print(f"    Most frequent (shortest gap): {min(avg_gaps, key=avg_gaps.get)} (avg {min(gap_values):.1f})")
print(f"    Least frequent (longest gap): {max(avg_gaps, key=avg_gaps.get)} (avg {max(gap_values):.1f})")

# ==========================================
# TEST 3: STRUCTURAL PATTERN - SUM
# ==========================================
print(f"\n{'='*70}")
print(f" TEST 3: STRUCTURAL PATTERNS")
print(f"{'='*70}\n")

sums = [sum(d) for d in data]
print(f"  SUM distribution:")
print(f"    Mean: {np.mean(sums):.1f}, Std: {np.std(sums):.1f}")
print(f"    Min: {min(sums)}, Max: {max(sums)}")
print(f"    P10: {int(np.percentile(sums, 10))}, P90: {int(np.percentile(sums, 90))}")

# Sum mod patterns
print(f"\n  SUM mod patterns:")
for mod in [3, 5, 7, 9]:
    prev_mod = [sum(data[i-1]) % mod for i in range(1, total)]
    next_mod = [sum(data[i]) % mod for i in range(1, total)]
    cross = defaultdict(Counter)
    for p, n in zip(prev_mod, next_mod):
        cross[p][n] += 1
    # Check if any conditional probability is significantly different from uniform
    max_lift = 0
    for p_val, counts in cross.items():
        total_p = sum(counts.values())
        for n_val, cnt in counts.items():
            observed = cnt / total_p
            expected = 1 / mod
            lift = observed / expected
            if lift > max_lift: max_lift = lift
    print(f"    mod {mod}: max conditional lift = {max_lift:.3f}x (1.0 = random)")

# Odd/Even pattern
print(f"\n  ODD/EVEN sequence:")
odd_counts = [sum(1 for x in d if x % 2 == 1) for d in data]
for i in range(1, total):
    pass  # Just checking pattern
odd_transitions = Counter()
for i in range(1, total):
    odd_transitions[(odd_counts[i-1], odd_counts[i])] += 1
print(f"    Transition (prev_odd -> next_odd) top patterns:")
for (p, n), c in odd_transitions.most_common(10):
    pct = c / (total-1) * 100
    expected = 1/7 * 100  # roughly
    print(f"      {p}odd->{n}odd: {c:4d} ({pct:.1f}%)")

# Decade distribution (1-9, 10-19, 20-29, 30-39, 40-45)
print(f"\n  DECADE distribution per draw:")
decades = defaultdict(list)
for d in data:
    dec_count = [0]*5
    for n in d:
        if n <= 9: dec_count[0] += 1
        elif n <= 19: dec_count[1] += 1
        elif n <= 29: dec_count[2] += 1
        elif n <= 39: dec_count[3] += 1
        else: dec_count[4] += 1
    for di in range(5):
        decades[di].append(dec_count[di])
for di in range(5):
    ranges = ['01-09', '10-19', '20-29', '30-39', '40-45']
    avg = np.mean(decades[di])
    print(f"    {ranges[di]}: avg={avg:.2f}/6")

# ==========================================
# TEST 4: CONDITIONAL ELIMINATION
# Find conditions that allow safe elimination
# ==========================================
print(f"\n{'='*70}")
print(f" TEST 4: CONDITIONAL ELIMINATION")
print(f" Find: conditions where specific numbers are safely excludable")
print(f"{'='*70}\n")

# Decade carry-over
print(f"  DECADE CARRY-OVER analysis:")
print(f"  (How often do numbers from same decade appear in consecutive draws?)")
for di in range(5):
    ranges = ['01-09', '10-19', '20-29', '30-39', '40-45']
    sizes = [9, 10, 10, 10, 6]
    carry = 0
    for i in range(1, total):
        prev_in_dec = [n for n in data[i-1] if (di*10+1 <= n <= di*10+9 if di < 4 else 40 <= n <= 45)]
        curr_in_dec = [n for n in data[i] if (di*10+1 <= n <= di*10+9 if di < 4 else 40 <= n <= 45)]
        if prev_in_dec and curr_in_dec:
            carry += 1
    pct = carry / (total-1) * 100
    print(f"    {ranges[di]}: carry-over rate = {pct:.1f}%")

# High number continuation
print(f"\n  HIGH/LOW transition:")
for i in range(1, total):
    pass
high_counts = [sum(1 for x in d if x > 22) for d in data]
high_trans = Counter()
for i in range(1, total):
    high_trans[(high_counts[i-1], high_counts[i])] += 1
for (p, n), c in sorted(high_trans.items(), key=lambda x:-x[1])[:8]:
    print(f"    {p}high->{n}high: {c:4d} ({c/(total-1)*100:.1f}%)")

# ==========================================
# TEST 5: BEST ELIMINATION STRATEGY
# Combine multiple rules and measure pool reduction + safety
# ==========================================
print(f"\n{'='*70}")
print(f" TEST 5: COMBINED ELIMINATION (how small can pool get safely?)")
print(f"{'='*70}\n")

# For each draw, eliminate based on multiple criteria and check if all 6 winners remain
test_results = []
for i in range(5, total):
    excluded = set()
    
    # E1: Numbers from last draw (unless repeat rate suggests otherwise)
    # Find which numbers from last draw are LEAST likely to repeat
    last = set(data[i-1])
    prev = set(data[i-2]) if i >= 2 else set()
    
    # E2: Numbers that appeared in both last 2 draws (double-hot = cooling)
    double_hot = last & prev
    excluded |= double_hot
    
    # E3: Numbers with very short current gap (appeared too recently in window)
    recent_window = set()
    for w in range(1, 3):
        if i-w >= 0:
            recent_window |= set(data[i-w])
    # Only exclude numbers that appeared in BOTH of last 2 draws
    excluded |= double_hot
    
    # E4: Based on sum constraint — eliminate extreme numbers
    recent_sums = [sum(data[j]) for j in range(max(0, i-20), i)]
    target_sum_lo = int(np.percentile(recent_sums, 15))
    target_sum_hi = int(np.percentile(recent_sums, 85))
    
    # E5: Decade balance — if last draw had 0 from a decade, don't over-exclude that decade
    
    actual = set(data[i])
    pool_size = 45 - len(excluded)
    winners_in_pool = len(actual - excluded)
    test_results.append({
        'pool_size': pool_size,
        'winners_in_pool': winners_in_pool,
        'all_safe': winners_in_pool == 6,
        'excluded_count': len(excluded),
    })

safe_rate = sum(1 for r in test_results if r['all_safe']) / len(test_results) * 100
avg_pool = np.mean([r['pool_size'] for r in test_results])
avg_excluded = np.mean([r['excluded_count'] for r in test_results])
avg_winners = np.mean([r['winners_in_pool'] for r in test_results])

print(f"  Combined elimination (double-hot only):")
print(f"    Avg excluded: {avg_excluded:.1f}")
print(f"    Avg pool: {avg_pool:.1f}")
print(f"    Safe rate: {safe_rate:.1f}%")
print(f"    Avg winners in pool: {avg_winners:.2f}/6")

# More aggressive: exclude ALL of last draw
test2 = []
for i in range(5, total):
    excluded = set(data[i-1])  # All 6 from last draw
    actual = set(data[i])
    safe = len(actual & excluded) == 0
    test2.append(safe)
safe2 = sum(test2) / len(test2) * 100
print(f"\n  Exclude ALL last draw (6 nums):")
print(f"    Pool: 39, Safe: {safe2:.1f}%")

# Even more: exclude numbers with gap < 3
test3_results = []
for i in range(5, total):
    excluded = set()
    for j in range(1, 3):
        if i-j >= 0:
            excluded |= set(data[i-j])
    actual = set(data[i])
    winners_lost = len(actual & excluded)
    test3_results.append({
        'excluded': len(excluded),
        'pool': 45 - len(excluded),
        'safe': winners_lost == 0,
        'lost': winners_lost,
    })
safe3 = sum(1 for r in test3_results if r['safe']) / len(test3_results) * 100
avg_lost3 = np.mean([r['lost'] for r in test3_results])
avg_pool3 = np.mean([r['pool'] for r in test3_results])
print(f"\n  Exclude last 2 draws (gap < 3):")
print(f"    Avg pool: {avg_pool3:.1f}, Safe: {safe3:.1f}%, Avg lost: {avg_lost3:.2f}")

# SMART elimination: exclude last draw + numbers NOT in top-30 frequency
print(f"\n  FREQUENCY-BASED elimination:")
for top_k in [25, 30, 35]:
    safe_count = 0
    for i in range(100, total):
        # Use rolling frequency from last 100 draws
        freq = Counter(n for d in data[max(0,i-100):i] for n in d)
        top_nums = set(n for n, _ in freq.most_common(top_k))
        # Also exclude last draw
        pool = top_nums - set(data[i-1])
        actual = set(data[i])
        if len(actual - pool) == 0:
            safe_count += 1
    rate = safe_count / (total-100) * 100
    print(f"    Top-{top_k} freq (excl last draw): pool≈{top_k-6}, safe={rate:.1f}%")

print(f"\n{'='*70}")
print(f" DONE — Use these findings to build V5 elimination engine")
print(f"{'='*70}")
