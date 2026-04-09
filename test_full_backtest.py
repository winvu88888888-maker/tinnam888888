"""
Full Backtest — Test Deep Forensic across ALL draws.
Reports exact 6/6, 5/6, 4/6, 3/6 rates for both primary and portfolio.
"""
import sys
import os
import time
import numpy as np
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from scraper.data_manager import get_mega645_numbers, get_mega645_all
from models.deep_forensic import DeepForensic

print("Loading Mega 6/45 data...")
data = get_mega645_numbers()
all_rows = get_mega645_all()
dates = [r['draw_date'] for r in all_rows]
total = len(data)

print(f"Total draws: {total}")
print(f"Last draw: {dates[-1]} -> {data[-1]}")

engine = DeepForensic(45, 6)

# Test from draw 80 to end (need at least 70 history for signals)
start_from = 80
primary_matches = []
portfolio_matches = []

t0 = time.time()
tested = 0

for test_end in range(start_from, total - 1):
    train = data[:test_end + 1]
    train_dates = dates[:test_end + 1] if dates else None
    actual = set(data[test_end + 1])
    
    try:
        result = engine.analyze(train, train_dates)
        
        # Primary
        pred = set(result['primary'])
        m = len(pred & actual)
        primary_matches.append(m)
        
        # Portfolio best
        best_p = max((len(set(p) & actual) for p in result['portfolio']), default=m)
        portfolio_matches.append(best_p)
        
        tested += 1
        
        # Progress
        if tested % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / tested * (total - 1 - start_from - tested) / 60
            p_dist = Counter(primary_matches)
            port_dist = Counter(portfolio_matches)
            print(f"  [{tested}/{total-1-start_from}] {elapsed:.0f}s (ETA {eta:.1f}m) | "
                  f"Primary: avg={np.mean(primary_matches):.3f}/6 | "
                  f"Portfolio: avg={np.mean(portfolio_matches):.3f}/6, "
                  f"4+={port_dist.get(4,0)+port_dist.get(5,0)+port_dist.get(6,0)}, "
                  f"5+={port_dist.get(5,0)+port_dist.get(6,0)}, "
                  f"6={port_dist.get(6,0)}")
    except Exception as e:
        primary_matches.append(0)
        portfolio_matches.append(0)
        tested += 1

elapsed = time.time() - t0

# === FINAL REPORT ===
print(f"\n{'='*70}")
print(f"  FULL BACKTEST REPORT — Deep Forensic Engine")
print(f"  Tested: {tested} draws | Time: {elapsed:.1f}s")
print(f"{'='*70}")

p_dist = Counter(primary_matches)
port_dist = Counter(portfolio_matches)

print(f"\n  === PRIMARY (1 set per draw) ===")
print(f"  Average: {np.mean(primary_matches):.4f}/6")
print(f"  Random:  {6*6/45:.4f}/6")
print(f"  Improvement: {((np.mean(primary_matches)/(6*6/45))-1)*100:.1f}% vs random")
print()
for k in range(7):
    c = p_dist.get(k, 0)
    pct = c / tested * 100
    bar = '#' * int(pct)
    print(f"    {k}/6: {c:5d} ({pct:6.2f}%) {bar}")

print(f"\n  === PORTFOLIO (20 sets per draw) ===")
print(f"  Average: {np.mean(portfolio_matches):.4f}/6")
print(f"  Improvement: {((np.mean(portfolio_matches)/(6*6/45))-1)*100:.1f}% vs random")
print()
for k in range(7):
    c = port_dist.get(k, 0)
    pct = c / tested * 100
    bar = '#' * int(pct)
    print(f"    {k}/6: {c:5d} ({pct:6.2f}%) {bar}")

print(f"\n  === KEY RATES ===")
print(f"  6/6 Primary: {p_dist.get(6,0)}/{tested} = {p_dist.get(6,0)/tested*100:.4f}%")
print(f"  5/6 Primary: {p_dist.get(5,0)}/{tested} = {p_dist.get(5,0)/tested*100:.4f}%")
print(f"  4/6 Primary: {p_dist.get(4,0)}/{tested} = {p_dist.get(4,0)/tested*100:.4f}%")
print(f"  3/6 Primary: {p_dist.get(3,0)}/{tested} = {p_dist.get(3,0)/tested*100:.4f}%")
print()
print(f"  6/6 Portfolio: {port_dist.get(6,0)}/{tested} = {port_dist.get(6,0)/tested*100:.4f}%")
print(f"  5/6 Portfolio: {port_dist.get(5,0)}/{tested} = {port_dist.get(5,0)/tested*100:.4f}%")
print(f"  4/6 Portfolio: {port_dist.get(4,0)}/{tested} = {port_dist.get(4,0)/tested*100:.4f}%")
print(f"  3/6 Portfolio: {port_dist.get(3,0)}/{tested} = {port_dist.get(3,0)/tested*100:.4f}%")

# For context: random probabilities
from math import comb
total_combos = comb(45, 6)  # 8,145,060
print(f"\n  === RANDOM PROBABILITIES (for reference) ===")
for k in range(7):
    ways = comb(6, k) * comb(39, 6 - k)  # hit k from 6, miss rest from 39
    p = ways / total_combos * 100
    portfolio_p = 1 - (1 - ways/total_combos)**20
    print(f"    {k}/6: 1-set={p:.6f}%, 20-set={portfolio_p*100:.6f}%")

print(f"\n{'='*70}")
