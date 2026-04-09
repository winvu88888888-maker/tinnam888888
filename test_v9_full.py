"""Full backtest V9 — block puzzle."""
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
print(f"V9: {r['total_sets']} sets, cov={r['coverage']}/45, {time.time()-t0:.1f}s\n")

print(f"{'='*70}\n V9 FULL BACKTEST (Block Puzzle)\n{'='*70}")
portm = []
set_counts = []
six_hits = []
strat_hits = Counter()  # which strategy hits 5/6 and 6/6?
t0 = time.time()
for i in range(80, total-1):
    r = engine.predict(data[:i+1], dates[:i+1])
    actual = set(data[i+1])
    best = max(len(set(p['numbers']) & actual) for p in r['portfolio'])
    portm.append(best)
    set_counts.append(r['total_sets'])
    if best >= 5:
        bp = [p for p in r['portfolio'] if len(set(p['numbers']) & actual) == best][0]
        strat_hits[bp['strategy']] += 1
        marker = "★★★ 6/6 JACKPOT!!! ★★★" if best == 6 else f"5/6 [{bp['strategy']}]"
        print(f"  #{i+1}: {marker} Pred={bp['numbers']} Act={sorted(actual)}")
        if best == 6: six_hits.append(i+1)
    if len(portm) % 200 == 0:
        pd = Counter(portm)
        el = time.time()-t0
        eta = el/len(portm)*(total-81-len(portm))/60
        print(f"  [{len(portm)}/{total-81}] avg={np.mean(portm):.3f} sets={np.mean(set_counts):.0f} "
              f"6={pd.get(6,0)} 5={pd.get(5,0)} 4={pd.get(4,0)} {el:.0f}s ETA={eta:.1f}m")

el = time.time()-t0
tested = len(portm)
dport = Counter(portm)
print(f"\n{'='*70}")
print(f" V9 RESULTS — {tested} draws, {el:.0f}s ({el/60:.1f}m)")
print(f" Avg sets: {np.mean(set_counts):.0f}")
print(f"{'='*70}")
print(f"\n PORTFOLIO: avg={np.mean(portm):.4f}/6")
for k in range(7): c=dport.get(k,0); print(f"   {k}/6: {c:5d} ({c/tested*100:6.2f}%)")
print(f"\n KEY:")
for t in [6,5,4,3]:
    h = sum(1 for m in portm if m >= t)
    print(f"   >={t}/6: {h:5d} ({h/tested*100:.2f}%)")
print(f"\n Strategy hits (5/6+):")
for s, cnt in strat_hits.most_common():
    print(f"   {s:22s}: {cnt}")
if six_hits: print(f"\n ★★★ 6/6: {six_hits} ★★★")
print(f"{'='*70}")
