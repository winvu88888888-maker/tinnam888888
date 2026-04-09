"""
V7 REVERSE ANALYSIS: Learn when scoring works best.

Questions:
1. When ALL 6 land in top-K (K=10,12,15,20): what conditions exist?
2. Which INDIVIDUAL signals had all 6 in their top-12?
3. Can we identify "high confidence" draws where pool is more concentrated?
4. If we could predict confidence level, could we use smaller portfolio on good draws?
5. What if we COMBINE only the signals that scored well on a per-draw basis?

Goal: Reduce from 5,600 sets to <1000 while maintaining or increasing 6/6.
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

# Per-signal accuracy tracking
signal_names = None
signal_top12_hits = defaultdict(list)  # signal -> list of hits when signal puts num in top-12
signal_top6_hits = defaultdict(list)

# Pool quality tracking
pool_quality = {k: [] for k in [10, 12, 15, 20]}
per_draw_best_signals = []

t0 = time.time()
tested = 0

for i in range(80, total - 1):
    # Get signals directly
    sub_data = data[:i+1]
    sub_dates = dates[:i+1]
    
    signals = engine._compute_all_signals(sub_data, sub_dates)
    actual = set(data[i+1])
    
    if signal_names is None:
        signal_names = list(signals.keys())
    
    # Per-signal analysis: how many of 6 winners in each signal's top-K?
    best_sig = None
    best_hits = 0
    for sig_name, sig_scores in signals.items():
        if not sig_scores: continue
        ranked = sorted(sig_scores.items(), key=lambda x: -x[1])
        top12 = set(num for num, _ in ranked[:12])
        top6 = set(num for num, _ in ranked[:6])
        h12 = len(top12 & actual)
        h6 = len(top6 & actual)
        signal_top12_hits[sig_name].append(h12)
        signal_top6_hits[sig_name].append(h6)
        if h12 > best_hits:
            best_hits = h12
            best_sig = sig_name
    
    per_draw_best_signals.append((best_sig, best_hits))
    
    # Ensemble pool quality
    weights = engine._walk_forward_weights(sub_data, signals)
    vote_counts = Counter()
    for sig_name, sig_scores in signals.items():
        if not sig_scores: continue
        w = weights.get(sig_name, 1.0)
        for rank, (num, _) in enumerate(sorted(sig_scores.items(), key=lambda x: -x[1])[:12]):
            vote_counts[num] += w * (12 - rank) / 12
    
    ranked_pool = [num for num, _ in vote_counts.most_common()]
    for k in pool_quality:
        pool_k = set(ranked_pool[:k]) if k <= len(ranked_pool) else set(ranked_pool)
        pool_quality[k].append(len(pool_k & actual))
    
    tested += 1
    if tested % 300 == 0:
        print(f"  [{tested}/{total-81}] {time.time()-t0:.0f}s")

print(f"\n Total: {tested} draws, {time.time()-t0:.0f}s")

# ============================================
# ANALYSIS 1: Per-Signal Quality
# ============================================
print(f"\n{'='*70}")
print(f" ANALYSIS 1: INDIVIDUAL SIGNAL QUALITY")
print(f" (avg hits when using each signal's top-12)")
print(f"{'='*70}\n")

sig_quality = {}
for sig in signal_names:
    if sig in signal_top12_hits:
        avg12 = np.mean(signal_top12_hits[sig])
        avg6 = np.mean(signal_top6_hits[sig])
        all6_in12 = sum(1 for h in signal_top12_hits[sig] if h == 6)
        sig_quality[sig] = avg12
        print(f"  {sig:16s}: top12 avg={avg12:.3f}/6  top6 avg={avg6:.3f}/6  "
              f"all6_in_top12={all6_in12} ({all6_in12/tested*100:.2f}%)")

# Rank signals
print(f"\n  BEST signals (by top-12 avg):")
for sig, avg in sorted(sig_quality.items(), key=lambda x: -x[1])[:5]:
    print(f"    {sig}: {avg:.3f}/6")

# ============================================
# ANALYSIS 2: Signal Combination
# ============================================
print(f"\n{'='*70}")
print(f" ANALYSIS 2: BEST SIGNAL PER DRAW")
print(f"{'='*70}\n")

best_sig_count = Counter(s for s, _ in per_draw_best_signals)
best_sig_avg = np.mean([h for _, h in per_draw_best_signals])
print(f"  Avg best single signal: {best_sig_avg:.2f}/6 in top-12")
print(f"  Best signal frequency:")
for sig, cnt in best_sig_count.most_common(10):
    print(f"    {sig:16s}: best {cnt} times ({cnt/tested*100:.1f}%)")

# How often does the BEST single signal have all 6 in top-12?
all6_best = sum(1 for _, h in per_draw_best_signals if h == 6)
print(f"\n  Best signal has ALL 6 in top-12: {all6_best}/{tested} = {all6_best/tested*100:.2f}%")

# ============================================
# ANALYSIS 3: Ensemble Pool Quality
# ============================================
print(f"\n{'='*70}")
print(f" ANALYSIS 3: ENSEMBLE POOL QUALITY")
print(f"{'='*70}\n")

for k in sorted(pool_quality.keys()):
    hits = pool_quality[k]
    avg = np.mean(hits)
    all6 = sum(1 for h in hits if h == 6)
    at5 = sum(1 for h in hits if h >= 5)
    rand = k * 6 / 45
    lift = avg / rand
    print(f"  top-{k:2d}: avg={avg:.3f}/6 (random={rand:.2f}, lift={lift:.2f}x) "
          f"all6={all6} ({all6/tested*100:.2f}%) 5+={at5} ({at5/tested*100:.1f}%)")

# ============================================
# ANALYSIS 4: Dynamic Signal Selection
# ============================================
print(f"\n{'='*70}")
print(f" ANALYSIS 4: IF WE PICK TOP-3 SIGNALS DYNAMICALLY")
print(f" (oracle: pick the 3 signals that perform best for each draw)")
print(f"{'='*70}\n")

union_hits = []
for i in range(tested):
    # For each draw, get top-3 signals' top-12, union them
    sig_scores = []
    for sig in signal_names:
        if sig in signal_top12_hits and i < len(signal_top12_hits[sig]):
            sig_scores.append((sig, signal_top12_hits[sig][i]))
    sig_scores.sort(key=lambda x: -x[1])
    # If we knew which 3 signals were best (oracle), how good is their union?
    # This is the CEILING for dynamic selection
    union_hits.append(sig_scores[0][1] if sig_scores else 0)

avg_oracle = np.mean(union_hits)
all6_oracle = sum(1 for h in union_hits if h == 6)
print(f"  Oracle best-1 signal: avg={avg_oracle:.3f}/6, all6={all6_oracle} ({all6_oracle/tested*100:.2f}%)")

# What about INTERSECTION of top-K signals?
# If multiple signals agree on a number, it's more likely
print(f"\n  CONSENSUS approach (numbers voted by >=N signals in their top-12):")
for min_votes in [3, 5, 7, 10]:
    consensus_hits = []
    for i in range(tested):
        actual_draw = set(data[81 + i])
        # Count votes for each number
        votes = Counter()
        for sig in signal_names:
            if sig in signal_top12_hits and i < len(signal_top12_hits[sig]):
                # Need actual top-12 for this draw... we don't have it stored
                pass
        # Skip - we need a different approach
    break

# ============================================
# ANALYSIS 5: Portfolio size analysis
# ============================================
print(f"\n{'='*70}")
print(f" ANALYSIS 5: MINIMUM PORTFOLIO SIZE FOR 6/6")
print(f"{'='*70}\n")

from math import comb
for k in [10, 12, 15, 20]:
    all6 = sum(1 for h in pool_quality[k] if h == 6)
    ck = comb(k, 6)
    rate = all6 / tested * 100
    print(f"  top-{k}: all6 in {rate:.2f}% of draws, C({k},6)={ck:,}")
    for n_sets in [210, 500, 924, 2000, 5005]:
        if n_sets <= ck:
            p = n_sets / ck
            expected = tested * (rate/100) * p
            print(f"    {n_sets:5d} sets → cover {p*100:.1f}% → expected 6/6 in {tested}: {expected:.2f}")

print(f"\n{'='*70}")
print(f" CONCLUSION")
print(f"{'='*70}")
print(f"""
  KEY FINDING: Individual signals have ~1.60/6 avg in top-12 (random=1.60)
  → Signals are STILL essentially random individually too.
  
  But ensemble is slightly better for larger pools:
  → top-15 might have marginally above random concentration.
  
  PRACTICAL PATH TO FEWER SETS:
  1. If we accept C(12,6)=924 sets exhaustive from top-12:
     Need all 6 in top-12 → very rare
  2. If we keep C(15,6)~3200 valid + 500 broad = ~3700 total
     (already much less than 5600)
  3. Best approach: FILTER exhaustive top-15 more aggressively
     (stricter decade + sum constraints → fewer valid combos)
""")
