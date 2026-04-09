"""Backtest Ultimate Engine V2 on last 200 draws."""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter

from models.ultimate_engine import UltimateEngine
from scraper.data_manager import get_mega645_numbers, get_mega645_all

data = get_mega645_numbers()
rows = get_mega645_all()
dates = [r['draw_date'] for r in rows]
print(f"Data: {len(data)} draws")

engine = UltimateEngine(45, 6)
pm, portm = [], []
t0 = time.time()
test_start = max(80, len(data) - 200)

for i in range(test_start, len(data) - 1):
    r = engine.predict(data[:i+1], dates[:i+1], n_portfolio=100)
    actual = set(data[i+1])
    
    p_hit = len(set(r['primary']) & actual)
    pm.append(p_hit)
    
    best_p = max(len(set(p['numbers']) & actual) for p in r['portfolio'])
    portm.append(best_p)
    
    if best_p >= 4:
        print(f"  *** Draw #{i+1}: port {best_p}/6! Actual={sorted(actual)}")
    
    if len(pm) % 50 == 0:
        pd = Counter(portm)
        elapsed = time.time() - t0
        print(f"  [{len(pm)}] prim={np.mean(pm):.3f} port={np.mean(portm):.3f} "
              f"4+={pd.get(4,0)+pd.get(5,0)+pd.get(6,0)} "
              f"5+={pd.get(5,0)+pd.get(6,0)} 6={pd.get(6,0)} "
              f"time={elapsed:.0f}s")

elapsed = time.time() - t0
pd_p = Counter(pm)
pd_port = Counter(portm)

print(f"\n{'='*60}")
print(f" ULTIMATE V2 BACKTEST — 200 draws, {elapsed:.0f}s")
print(f"{'='*60}")
print(f" PRIMARY: avg={np.mean(pm):.4f}/6")
for k in range(7):
    c = pd_p.get(k, 0)
    print(f"   {k}/6: {c:4d} ({c/len(pm)*100:.1f}%)")
print(f"\n PORTFOLIO (100 sets): avg={np.mean(portm):.4f}/6")
for k in range(7):
    c = pd_port.get(k, 0)
    print(f"   {k}/6: {c:4d} ({c/len(portm)*100:.1f}%)")
print(f"\n Random: {6*6/45:.3f}/6")
print(f" Port vs Random: +{((np.mean(portm)/(6*6/45))-1)*100:.0f}%")
print(f" 3+ rate: {sum(1 for m in portm if m>=3)/len(portm)*100:.1f}%")
print(f"{'='*60}")
