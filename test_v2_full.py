"""
FULL Backtest Ultimate Engine V2 — ALL draws from draw 81 to latest.
Measures exact 6/6, 5/6, 4/6 rates across entire history.
"""
import sys, time, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter

from models.ultimate_engine import UltimateEngine
from scraper.data_manager import get_mega645_numbers, get_mega645_all

data = get_mega645_numbers()
rows = get_mega645_all()
dates = [r['draw_date'] for r in rows]
total = len(data)
print(f"Data: {total} draws, Last: {dates[-1]}")

engine = UltimateEngine(45, 6)
pm, portm = [], []
t0 = time.time()
test_start = 80  # Start from draw 81

for i in range(test_start, total - 1):
    r = engine.predict(data[:i+1], dates[:i+1], n_portfolio=100)
    actual = set(data[i+1])
    
    p_hit = len(set(r['primary']) & actual)
    pm.append(p_hit)
    
    best_p = max(len(set(p['numbers']) & actual) for p in r['portfolio'])
    portm.append(best_p)
    
    if best_p >= 5:
        print(f"  ***** Draw #{i+1}: PORTFOLIO {best_p}/6! Actual={sorted(actual)} *****")
    
    if len(pm) % 200 == 0:
        pd = Counter(portm)
        elapsed = time.time() - t0
        eta = elapsed / len(pm) * (total - 1 - test_start - len(pm)) / 60
        print(f"  [{len(pm)}/{total-1-test_start}] "
              f"prim={np.mean(pm):.3f} port={np.mean(portm):.3f} "
              f"6/6={pd.get(6,0)} 5/6={pd.get(5,0)} 4/6={pd.get(4,0)} "
              f"time={elapsed:.0f}s ETA={eta:.1f}min")

elapsed = time.time() - t0
tested = len(pm)
pd_p = Counter(pm)
pd_port = Counter(portm)
random_exp = 6*6/45

print(f"\n{'='*70}")
print(f" ULTIMATE ENGINE V2 — FULL BACKTEST")
print(f" {tested} draws tested | {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"{'='*70}")

print(f"\n === PRIMARY (1 set) ===")
print(f" Avg: {np.mean(pm):.4f}/6 (Random: {random_exp:.3f}/6)")
print(f" vs Random: +{((np.mean(pm)/random_exp)-1)*100:.1f}%")
for k in range(7):
    c = pd_p.get(k, 0)
    pct = c / tested * 100
    bar = '#' * int(pct)
    print(f"   {k}/6: {c:5d} ({pct:6.2f}%) {bar}")

print(f"\n === PORTFOLIO (100 sets) ===")
print(f" Avg: {np.mean(portm):.4f}/6 (Random: {random_exp:.3f}/6)")
print(f" vs Random: +{((np.mean(portm)/random_exp)-1)*100:.1f}%")
for k in range(7):
    c = pd_port.get(k, 0)
    pct = c / tested * 100
    bar = '#' * int(pct)
    print(f"   {k}/6: {c:5d} ({pct:6.2f}%) {bar}")

print(f"\n === KEY METRICS ===")
for threshold in [6, 5, 4, 3]:
    hits_p = sum(1 for m in pm if m >= threshold)
    hits_port = sum(1 for m in portm if m >= threshold)
    pct_p = hits_p / tested * 100
    pct_port = hits_port / tested * 100
    print(f"   >= {threshold}/6:  Primary {hits_p:5d} ({pct_p:.2f}%)  |  Portfolio {hits_port:5d} ({pct_port:.2f}%)")

print(f"\n === CONCLUSION ===")
p5_rate = sum(1 for m in portm if m >= 5) / tested * 100
p4_rate = sum(1 for m in portm if m >= 4) / tested * 100
p3_rate = sum(1 for m in portm if m >= 3) / tested * 100
print(f" 6/6 jackpot rate: {pd_port.get(6,0)}/{tested} = {pd_port.get(6,0)/tested*100:.4f}%")
print(f" 5/6 rate: {pd_port.get(5,0)}/{tested} = {pd_port.get(5,0)/tested*100:.3f}%")
print(f" 4/6 rate: {sum(1 for m in portm if m>=4)}/{tested} = {p4_rate:.2f}%")
print(f" 3+ rate: {sum(1 for m in portm if m>=3)}/{tested} = {p3_rate:.1f}%")
print(f"{'='*70}")
