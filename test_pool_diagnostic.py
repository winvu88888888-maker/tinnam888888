"""
DIAGNOSTIC: How good is our scoring pool?
Measures: In what % of draws, all 6 winning numbers are within top-K?
This is the CEILING — if all 6 aren't in pool, 6/6 is IMPOSSIBLE.
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
print(f"Data: {total} draws\n")

engine = UltimateEngine(45, 6)

# Track how many of the 6 winning numbers are in top-K pool
pool_hits = {k: [] for k in [10, 15, 20, 25, 30, 35, 40]}
all_6_in = {k: 0 for k in pool_hits}
at_least_5_in = {k: 0 for k in pool_hits}

t0 = time.time()
tested = 0

for i in range(80, total - 1):
    r = engine.predict(data[:i+1], dates[:i+1], n_portfolio=200)
    actual = set(data[i+1])
    
    # Get full ranked list
    ranked = sorted(r['scores'].items(), key=lambda x: -x[1])
    # But scores only has top 30 — we need full ranking
    # Use final_scores from top_30, extend with remaining
    top_30 = r['top_30']
    
    for k in pool_hits:
        pool = set(top_30[:k]) if k <= len(top_30) else set(top_30)
        hits = len(pool & actual)
        pool_hits[k].append(hits)
        if hits == 6:
            all_6_in[k] += 1
        if hits >= 5:
            at_least_5_in[k] += 1
    
    tested += 1
    if tested % 200 == 0:
        el = time.time() - t0
        print(f"  [{tested}] ", end="")
        for k in [15, 20, 25, 30]:
            avg = np.mean(pool_hits[k])
            a6 = all_6_in[k]
            print(f"top{k}: avg={avg:.2f} all6={a6} ", end="")
        print(f"  {el:.0f}s")

print(f"\n{'='*70}")
print(f" POOL QUALITY DIAGNOSTIC — {tested} draws")
print(f"{'='*70}")
print(f"\n {'Pool':>6} {'Avg hits':>10} {'All 6/6':>10} {'5+/6':>10} {'All6 %':>10} {'5+ %':>10}")
print(f" {'-'*60}")
for k in sorted(pool_hits.keys()):
    avg = np.mean(pool_hits[k])
    a6 = all_6_in[k]
    a5 = at_least_5_in[k]
    p6 = a6 / tested * 100
    p5 = a5 / tested * 100
    print(f" top-{k:>2}: {avg:>8.3f}/6  {a6:>8d}  {a5:>8d}  {p6:>8.2f}%  {p5:>8.2f}%")

# Distribution for top-20 and top-25
for k in [20, 25]:
    dist = Counter(pool_hits[k])
    print(f"\n top-{k} distribution:")
    for h in range(7):
        c = dist.get(h, 0)
        print(f"   {h}/6: {c:5d} ({c/tested*100:.1f}%)")

print(f"\n INSIGHT:")
print(f" Random top-20: C(20,6)/C(45,6) = {len(list(range(1)))}")
# Expected hits: 20*6/45 = 2.67
exp20 = 20 * 6 / 45
print(f" Random top-20 avg: {exp20:.2f}/6")
print(f" Actual top-20 avg: {np.mean(pool_hits[20]):.3f}/6 (lift = {np.mean(pool_hits[20])/exp20:.2f}x)")
exp25 = 25 * 6 / 45
print(f" Random top-25 avg: {exp25:.2f}/6") 
print(f" Actual top-25 avg: {np.mean(pool_hits[25]):.3f}/6 (lift = {np.mean(pool_hits[25])/exp25:.2f}x)")
print(f"{'='*70}")
