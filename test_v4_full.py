"""V4 sanity + full backtest."""
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
print(f"Data: {total} draws")

engine = UltimateEngine(45, 6)

# Quick test
t0 = time.time()
r = engine.predict(data, dates, n_portfolio=500)
print(f"V4 sanity: {r['n_signals']} sig, {r['total_sets']} sets, cov={r['coverage']}/45, "
      f"{time.time()-t0:.1f}s")
print(f"Primary: {r['primary']}")
strat = Counter(p['strategy'] for p in r['portfolio'])
for s, c in strat.most_common():
    print(f"  {s:16s} {c:3d}")

# Full backtest
print(f"\n{'='*70}")
print(f" V4 FULL BACKTEST")
print(f"{'='*70}")
pm, portm = [], []
t0 = time.time()
for i in range(80, total - 1):
    r = engine.predict(data[:i+1], dates[:i+1], n_portfolio=500)
    actual = set(data[i+1])
    pm.append(len(set(r['primary']) & actual))
    portm.append(max(len(set(p['numbers']) & actual) for p in r['portfolio']))
    
    if portm[-1] >= 5:
        best_set = None
        for p in r['portfolio']:
            h = len(set(p['numbers']) & actual)
            if h == portm[-1]:
                best_set = p; break
        print(f"  ***** #{i+1}: {portm[-1]}/6! [{best_set['strategy']}] "
              f"Pred={best_set['numbers']} Actual={sorted(actual)} *****")
    
    if len(pm) % 200 == 0:
        pd = Counter(portm)
        el = time.time() - t0
        eta = el / len(pm) * (total-81-len(pm)) / 60
        print(f"  [{len(pm)}/{total-81}] prim={np.mean(pm):.3f} port={np.mean(portm):.3f} "
              f"6={pd.get(6,0)} 5={pd.get(5,0)} 4={pd.get(4,0)} "
              f"{el:.0f}s ETA={eta:.1f}m")

el = time.time() - t0
tested = len(pm)
dp = Counter(pm)
dport = Counter(portm)

print(f"\n{'='*70}")
print(f" V4 RESULTS — {tested} draws, {el:.0f}s ({el/60:.1f}m)")
print(f"{'='*70}")
print(f"\n PRIMARY: avg={np.mean(pm):.4f}/6")
for k in range(7):
    c = dp.get(k,0); print(f"   {k}/6: {c:5d} ({c/tested*100:6.2f}%)")
print(f"\n PORTFOLIO ({r.get('total_sets', 500)} sets): avg={np.mean(portm):.4f}/6")
for k in range(7):
    c = dport.get(k,0); print(f"   {k}/6: {c:5d} ({c/tested*100:6.2f}%)")
print(f"\n KEY:")
for t in [6,5,4,3]:
    h = sum(1 for m in portm if m >= t)
    print(f"   >={t}/6: {h:5d} ({h/tested*100:.2f}%)")
rand = 6*6/45
print(f" vs Random: +{((np.mean(portm)/rand)-1)*100:.0f}%")
print(f"{'='*70}")
