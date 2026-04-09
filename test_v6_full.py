"""Full backtest V6 — exhaustive enumeration."""
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

# Sanity
t0 = time.time()
r = engine.predict(data, dates, 500)
print(f"V6: {r['total_sets']} sets ({r['tier1_count']}+{r['tier2_count']}+{r['tier3_count']}), "
      f"cov={r['coverage']}/45, {time.time()-t0:.1f}s\n")

# Full backtest
print(f"{'='*70}\n V6 FULL BACKTEST\n{'='*70}")
pm, portm = [], []
six_of_six_draws = []
t0 = time.time()
for i in range(80, total-1):
    r = engine.predict(data[:i+1], dates[:i+1], 500)
    actual = set(data[i+1])
    pm.append(len(set(r['primary']) & actual))
    best = max(len(set(p['numbers']) & actual) for p in r['portfolio'])
    portm.append(best)
    
    if best >= 5:
        bp = [p for p in r['portfolio'] if len(set(p['numbers']) & actual) == best][0]
        marker = "****** 6/6 JACKPOT!!! ******" if best == 6 else f"5/6 [{bp['strategy']}]"
        print(f"  #{i+1}: {marker} Pred={bp['numbers']} Act={sorted(actual)}")
        if best == 6: six_of_six_draws.append(i+1)
    
    if len(pm) % 200 == 0:
        pd = Counter(portm)
        el = time.time()-t0
        eta = el/len(pm)*(total-81-len(pm))/60
        print(f"  [{len(pm)}/{total-81}] avg={np.mean(portm):.3f} 6={pd.get(6,0)} "
              f"5={pd.get(5,0)} 4={pd.get(4,0)} {el:.0f}s ETA={eta:.1f}m")

el = time.time()-t0
tested = len(pm)
dp, dport = Counter(pm), Counter(portm)
print(f"\n{'='*70}")
print(f" V6 RESULTS — {tested} draws, {el:.0f}s ({el/60:.1f}m)")
print(f"{'='*70}")
print(f"\n PORTFOLIO (~{r['total_sets']} sets): avg={np.mean(portm):.4f}/6")
for k in range(7): c=dport.get(k,0); print(f"   {k}/6: {c:5d} ({c/tested*100:6.2f}%)")
print(f"\n KEY:")
for t in [6,5,4,3]:
    h = sum(1 for m in portm if m >= t)
    print(f"   >={t}/6: {h:5d} ({h/tested*100:.2f}%)")
print(f" vs Random: +{((np.mean(portm)/0.8)-1)*100:.0f}%")
if six_of_six_draws:
    print(f"\n ★★★ 6/6 JACKPOT DRAWS: {six_of_six_draws} ★★★")
print(f"{'='*70}")
