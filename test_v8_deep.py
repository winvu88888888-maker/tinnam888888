"""
V8 DEEP REVERSE: Multi-perspective pool analysis.

Current problem: ONE ensemble ranking → all 6 in top-15 only 0.14%
Hypothesis: If we create MULTIPLE DIVERSE rankings, each might capture
            all 6 in its top-10 independently.

C(10,6) = 210 combos per ranking.
With 10 diverse rankings: 210 × 10 = 2,100 sets.
If each has independent 0.07% chance of all-6-in-top10, combined = ~0.7%

Tests:
1. Per-signal top-10 containment: how often does each signal's top-10 hold all 6?
2. Signal GROUPS: if we group 4-5 signals, does any group consistently beat random?
3. OPPOSITE scoring: sometimes the WORST-scoring signal's top-10 captures all 6?!
4. RANDOM SHUFFLED rankings: how many random top-10 pools needed for 50% all-6?
5. Ideal: for each draw, find THE smallest pool containing all 6 → reverse engineer
"""
import sys, os, time
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from models.ultimate_engine import UltimateEngine
from scraper.data_manager import get_mega645_numbers, get_mega645_all

data = get_mega645_numbers()
dates = [r['draw_date'] for r in get_mega645_all()]
total = len(data)
engine = UltimateEngine(45, 6)

print(f"Data: {total} draws\n")

# Store per-draw, per-signal rankings
all_signal_rankings = []  # list of {sig_name: [ranked_numbers]}
all_actuals = []

# Also test: signal SUBGROUPS
group_defs = {
    'recent': ['momentum', 'streak', 'runlength', 'gap_timing'],
    'pattern': ['transition', 'ngram', 'context3', 'seq_pattern'],
    'pair': ['cooccurrence', 'pair_boost', 'triplet', 'knn'],
    'structure': ['oddeven', 'highlow', 'entropy', 'position'],
    'cycle': ['fft_cycle', 'lag_repeat', 'day_profile', 'consecutive'],
}
group_top10_all6 = {g: 0 for g in group_defs}

# Per signal top-10 all-6 count
sig_top10_all6 = Counter()
sig_top10_hits = defaultdict(list)

# Multi-perspective: if we combine N DIFFERENT signal rankings,
# does at least one have all 6 in its top-10?
multi_perspective_hits = {n: 0 for n in [1, 2, 3, 5, 10, 15, 20]}

t0 = time.time()
tested = 0

for i in range(80, total - 1):
    sub_data = data[:i+1]
    sub_dates = dates[:i+1]
    signals = engine._compute_all_signals(sub_data, sub_dates)
    actual = set(data[i+1])
    all_actuals.append(actual)
    
    # Per-signal rankings
    sig_rankings = {}
    sig_top10_sets = {}
    for sig_name, sig_scores in signals.items():
        if not sig_scores: continue
        ranked = [num for num, _ in sorted(sig_scores.items(), key=lambda x: -x[1])]
        sig_rankings[sig_name] = ranked
        top10 = set(ranked[:10])
        sig_top10_sets[sig_name] = top10
        hits = len(top10 & actual)
        sig_top10_hits[sig_name].append(hits)
        if hits == 6:
            sig_top10_all6[sig_name] += 1
    
    # Group rankings (vote within group)
    for group_name, members in group_defs.items():
        group_votes = Counter()
        for sig in members:
            if sig in sig_rankings:
                for rank, num in enumerate(sig_rankings[sig][:12]):
                    group_votes[num] += 12 - rank
        group_top10 = set(num for num, _ in group_votes.most_common(10))
        if len(group_top10 & actual) == 6:
            group_top10_all6[group_name] += 1
    
    # Multi-perspective: shuffle all signal rankings, check if ANY has all 6 in top-10
    sig_names = list(sig_top10_sets.keys())
    any_hit = {n: False for n in multi_perspective_hits}
    # Check each individual signal
    for idx, sig in enumerate(sig_names):
        if len(sig_top10_sets[sig] & actual) == 6:
            for n in multi_perspective_hits:
                if idx < n:
                    any_hit[n] = True
    for n in multi_perspective_hits:
        if any_hit[n]:
            multi_perspective_hits[n] += 1
    
    tested += 1
    if tested % 300 == 0:
        print(f"  [{tested}/{total-81}] {time.time()-t0:.0f}s")

print(f"\n Total: {tested} draws, {time.time()-t0:.0f}s")

# ============================================
# ANALYSIS 1: Per-Signal Top-10 All-6
# ============================================
print(f"\n{'='*70}")
print(f" ANALYSIS 1: INDIVIDUAL SIGNAL top-10 containment")
print(f"{'='*70}\n")
print(f"  Random baseline: P(all 6 in top-10) = C(10,6)/C(45,6) = 0.0026%")
for sig, cnt in sorted(sig_top10_all6.items(), key=lambda x: -x[1]):
    avg = np.mean(sig_top10_hits.get(sig, [0]))
    print(f"  {sig:16s}: all6_in_top10={cnt} ({cnt/tested*100:.3f}%)  avg={avg:.3f}/6")

# ============================================  
# ANALYSIS 2: Signal Groups
# ============================================
print(f"\n{'='*70}")
print(f" ANALYSIS 2: SIGNAL GROUP top-10 containment")
print(f"{'='*70}\n")
for group, cnt in sorted(group_top10_all6.items(), key=lambda x: -x[1]):
    print(f"  {group:16s}: all6={cnt} ({cnt/tested*100:.3f}%)")

# ============================================
# ANALYSIS 3: Multi-Perspective
# ============================================
print(f"\n{'='*70}")
print(f" ANALYSIS 3: MULTI-PERSPECTIVE (any of first N signals has all 6 in top-10)")
print(f"{'='*70}\n")
for n, cnt in sorted(multi_perspective_hits.items()):
    print(f"  First {n:2d} signals: {cnt} times all6 ({cnt/tested*100:.3f}%)")

# ============================================
# ANALYSIS 4: DIVERSE top-10 UNION coverage
# ============================================
print(f"\n{'='*70}")
print(f" ANALYSIS 4: UNION of diverse top-10 pools")
print(f"{'='*70}\n")

# For each draw, take top-10 from each signal group → union them → check coverage
union_hits = []
for i in range(tested):
    actual = all_actuals[i]
    # Reconstruct group top-10 (simplified: use first signal in group)
    # This is a proxy metric
    all_top10 = set()
    for sig in sig_top10_hits:
        if i < len(sig_top10_hits[sig]):
            pass  # We don't have the actual sets stored
    # Use a simpler metric: what's the max hits across all signals?
    max_h = max(sig_top10_hits[sig][i] for sig in sig_top10_hits if i < len(sig_top10_hits[sig]))
    union_hits.append(max_h)

avg_best = np.mean(union_hits)
all6_best = sum(1 for h in union_hits if h == 6)
print(f"  Best single signal per draw (oracle):")
print(f"    avg best hits in top-10: {avg_best:.2f}/6")
print(f"    all 6 in best signal's top-10: {all6_best}/{tested} = {all6_best/tested*100:.3f}%")
print(f"    → Even oracle can't reliably get all 6 in any top-10!")

# ============================================
# ANALYSIS 5: WHAT IF WE USE DIFFERENT POOL SIZES?
# ============================================
print(f"\n{'='*70}")
print(f" ANALYSIS 5: OPTIMAL POOL SIZE × NUMBER OF PERSPECTIVES")
print(f"{'='*70}\n")

from math import comb
print(f"  {'K':>3} {'C(K,6)':>8} {'1 rank':>10} {'5 rank':>10} {'10 rank':>10} {'20 rank':>10}")
for k in [8, 10, 12, 15]:
    ck = comb(k, 6)
    # P(all 6 in top-K) for random = C(K,6)/C(45,6) ≈ K^6/45^6 approximation
    p_one = comb(k, 6) / comb(45, 6)
    for n_ranks in [1, 5, 10, 20]:
        p_any = 1 - (1 - p_one) ** n_ranks
        total_sets = ck * n_ranks
        expected_jackpots = tested * p_any
        if n_ranks == 1:
            print(f"  {k:3d} {ck:8d} {p_any*100:8.4f}%", end='')
        else:
            print(f" {p_any*100:8.4f}%", end='')
    print()

print(f"\n  Cost comparison (sets × per-draw):")
for k in [8, 10, 12]:
    ck = comb(k, 6)
    for n in [5, 10, 20]:
        total = ck * n
        p_one = comb(k, 6) / comb(45, 6)
        p_any = 1 - (1 - p_one) ** n
        exp = tested * p_any
        print(f"    {n}×C({k},6) = {total:5d} sets, P(6/6)={p_any*100:.4f}%, expected={exp:.2f}")

# ============================================
# ANALYSIS 6: RANDOM DIVERSE RANKINGS (empirical)
# ============================================
print(f"\n{'='*70}")
print(f" ANALYSIS 6: EMPIRICAL - Random diverse rankings")
print(f"{'='*70}\n")

# Generate N random rankings of 45 numbers, exhaust top-10 from each
# Check: how many random rankings needed for P(6/6) > 1/1405?
np.random.seed(42)
for n_random in [10, 20, 50, 100]:
    jackpots = 0
    for i in range(tested):
        actual = all_actuals[i]
        hit = False
        for _ in range(n_random):
            random_top10 = set(np.random.choice(range(1, 46), 10, replace=False))
            if len(random_top10 & actual) == 6:
                hit = True
                break
        if hit: jackpots += 1
    print(f"  {n_random:3d} random top-10 rankings: {jackpots} jackpots in {tested} draws "
          f"({jackpots/tested*100:.3f}%)")

print(f"\n{'='*70}")
print(f" CONCLUSION")  
print(f"{'='*70}\n")
print(f"""  Since scoring = random, the MATH is clear:
  - P(all 6 in any top-10 of 45) = C(10,6)/C(45,6) = {comb(10,6)/comb(45,6)*100:.4f}%
  - N independent rankings: P = 1-(1-p)^N
  - For P ≈ 1%: need N ≈ {int(0.01 / (comb(10,6)/comb(45,6)))} rankings at 210 sets = {int(0.01 / (comb(10,6)/comb(45,6))) * 210} sets
  - For P ≈ 5%: need N ≈ {int(0.05 / (comb(10,6)/comb(45,6)))} rankings = {int(0.05 / (comb(10,6)/comb(45,6))) * 210} sets
  
  V8 STRATEGY: Use 10 DIVERSE rankings (signal groups, random perturbations,
  reverse rankings), exhaust C(10,6)=210 from each = 2,100 total sets.
  This gives ~{10 * comb(10,6)/comb(45,6)*100:.3f}% P(6/6) per draw.
""")
