"""Full backtest V13 — Maximum 6/6 via Minimal Filters."""
import sys, time, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter
from models.ultimate_engine import UltimateEngine
from scraper.data_manager import get_mega645_numbers, get_mega645_all

data = get_mega645_numbers()
dates = [r['draw_date'] for r in get_mega645_all()]
total = len(data)
engine = UltimateEngine(45, 6)

t0 = time.time()
r = engine.predict(data, dates)
print(f"V13: {r['total_sets']} sets, pool={r['super_pool_size']}, {time.time()-t0:.1f}s")
sys.stdout.flush()

print(f"{'='*70}\n V13 FULL BACKTEST (Max 6/6)\n{'='*70}")
portm, set_counts, six_hits = [], [], []
t0 = time.time()
for i in range(80, total-1):
    r = engine.predict(data[:i+1], dates[:i+1])
    actual = set(data[i+1])
    # FAST matching: convert portfolio to set-based lookup
    best = 0
    for p in r['portfolio']:
        h = len(actual.intersection(p['numbers']))
        if h > best:
            best = h
            if h == 6: break  # Found jackpot, no need to check more
    portm.append(best); set_counts.append(r['total_sets'])
    if best == 6:
        six_hits.append(i+1)
        print(f"  #{i+1}: ***6/6*** pool={r['super_pool_size']} sets={r['total_sets']}")
        sys.stdout.flush()
    if len(portm) % 100 == 0:
        pd = Counter(portm); el = time.time()-t0
        eta = el/len(portm)*(total-81-len(portm))/60
        print(f"  [{len(portm)}/{total-81}] avg={np.mean(portm):.3f} sets={np.mean(set_counts):.0f} "
              f"6={pd.get(6,0)} 5={pd.get(5,0)} 4={pd.get(4,0)} {el:.0f}s ETA={eta:.1f}m")
        sys.stdout.flush()
el = time.time()-t0; tested = len(portm); dp = Counter(portm)
print(f"\n{'='*70}\n V13 RESULTS — {tested} draws, {el:.0f}s ({el/60:.1f}m)")
print(f" Avg sets: {np.mean(set_counts):.0f}\n{'='*70}")
print(f" PORTFOLIO: avg={np.mean(portm):.4f}/6")
for k in range(7): c=dp.get(k,0); print(f"   {k}/6: {c:5d} ({c/tested*100:6.2f}%)")
print(f"\n KEY:")
for t in [6,5,4,3]:
    h=sum(1 for m in portm if m>=t); print(f"   >={t}/6: {h:5d} ({h/tested*100:.2f}%)")
if six_hits: print(f"\n *** 6/6 HITS ({len(six_hits)}): {six_hits} ***")
print(f"{'='*70}")
