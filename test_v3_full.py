"""FULL Backtest Ultimate Engine V3 — ALL draws."""
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
pm, portm = [], []
t0 = time.time()

for i in range(80, total - 1):
    r = engine.predict(data[:i+1], dates[:i+1], n_portfolio=200)
    actual = set(data[i+1])
    pm.append(len(set(r['primary']) & actual))
    portm.append(max(len(set(p['numbers']) & actual) for p in r['portfolio']))
    
    if portm[-1] >= 5:
        print(f"  ***** #{i+1}: {portm[-1]}/6! Actual={sorted(actual)} *****")
    
    if len(pm) % 200 == 0:
        pd = Counter(portm)
        el = time.time() - t0
        eta = el / len(pm) * (total - 81 - len(pm)) / 60
        print(f"  [{len(pm)}/{total-81}] prim={np.mean(pm):.3f} port={np.mean(portm):.3f} "
              f"6={pd.get(6,0)} 5={pd.get(5,0)} 4={pd.get(4,0)} "
              f"{el:.0f}s ETA={eta:.1f}m")

el = time.time() - t0
tested = len(pm)
dp = Counter(pm)
dport = Counter(portm)
rand = 6*6/45

print(f"\n{'='*70}")
print(f" ULTIMATE V3 FULL BACKTEST — {tested} draws, {el:.0f}s ({el/60:.1f}m)")
print(f"{'='*70}")
print(f"\n PRIMARY: avg={np.mean(pm):.4f}/6 (vs Random {rand:.3f})")
for k in range(7):
    c = dp.get(k, 0)
    print(f"   {k}/6: {c:5d} ({c/tested*100:6.2f}%)")
print(f"\n PORTFOLIO (200 sets): avg={np.mean(portm):.4f}/6")
for k in range(7):
    c = dport.get(k, 0)
    print(f"   {k}/6: {c:5d} ({c/tested*100:6.2f}%)")
print(f"\n KEY RATES:")
for t in [6,5,4,3]:
    h = sum(1 for m in portm if m >= t)
    print(f"   >={t}/6: {h:5d} ({h/tested*100:.2f}%)")
print(f" vs Random: +{((np.mean(portm)/rand)-1)*100:.0f}%")
print(f"{'='*70}")
