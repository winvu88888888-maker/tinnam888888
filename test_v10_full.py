"""Full backtest V10 — Reverse Vulnerability Engine."""
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

# Quick sanity check
t0 = time.time()
r = engine.predict(data, dates)
print(f"V10: {r['total_sets']} sets, cov={r['coverage']}/45, "
      f"conf={r['confidence']}, {time.time()-t0:.1f}s")
print(f"Rescue nums: {r['rescue_nums'][:5]}\n")

print(f"{'='*70}\n V10 FULL BACKTEST (Reverse Vulnerability Engine)\n{'='*70}")
portm = []
set_counts = []
six_hits = []
conf_levels = []
rescue_hits = 0  # how many times rescue strategy got best match
strat_hits = Counter()
t0 = time.time()

for i in range(80, total-1):
    r = engine.predict(data[:i+1], dates[:i+1])
    actual = set(data[i+1])
    best = 0
    best_set = None
    for p in r['portfolio']:
        h = len(set(p['numbers']) & actual)
        if h > best:
            best = h
            best_set = p
    portm.append(best)
    set_counts.append(r['total_sets'])
    conf_levels.append(r['confidence'])

    if best >= 5:
        strat_hits[best_set['strategy']] += 1
        if best_set['strategy'] == 'rescue':
            rescue_hits += 1
        marker = "****** 6/6 JACKPOT!!! ******" if best == 6 else f"5/6 [{best_set['strategy']}]"
        print(f"  #{i+1}: {marker} Pred={best_set['numbers']} Act={sorted(actual)} "
              f"conf={r['confidence']:.2f}")
        if best == 6:
            six_hits.append(i+1)

    if len(portm) % 100 == 0:
        pd = Counter(portm)
        el = time.time()-t0
        eta = el/len(portm)*(total-81-len(portm))/60
        print(f"  [{len(portm)}/{total-81}] avg={np.mean(portm):.3f} sets={np.mean(set_counts):.0f} "
              f"conf={np.mean(conf_levels):.3f} "
              f"6={pd.get(6,0)} 5={pd.get(5,0)} 4={pd.get(4,0)} {el:.0f}s ETA={eta:.1f}m")
        sys.stdout.flush()

el = time.time()-t0
tested = len(portm)
dport = Counter(portm)
print(f"\n{'='*70}")
print(f" V10 RESULTS — {tested} draws, {el:.0f}s ({el/60:.1f}m)")
print(f" Avg sets: {np.mean(set_counts):.0f}, Avg confidence: {np.mean(conf_levels):.3f}")
print(f"{'='*70}")
print(f"\n PORTFOLIO: avg={np.mean(portm):.4f}/6")
for k in range(7):
    c = dport.get(k, 0)
    print(f"   {k}/6: {c:5d} ({c/tested*100:6.2f}%)")
print(f"\n KEY:")
for t in [6, 5, 4, 3]:
    h = sum(1 for m in portm if m >= t)
    print(f"   >={t}/6: {h:5d} ({h/tested*100:.2f}%)")
print(f"\n Strategy hits (5/6+):")
for s, cnt in strat_hits.most_common():
    print(f"   {s:22s}: {cnt}")
print(f"\n Rescue strategy hits: {rescue_hits}")
if six_hits:
    print(f"\n ****** 6/6 HITS: {six_hits} ******")
print(f"{'='*70}")
