"""Quick sanity test for Ultimate Engine V2."""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.ultimate_engine import UltimateEngine
from scraper.data_manager import get_mega645_numbers, get_mega645_all

data = get_mega645_numbers()
rows = get_mega645_all()
dates = [r['draw_date'] for r in rows]
print(f"Data: {len(data)} draws, Last: {dates[-1]}")

engine = UltimateEngine(45, 6)
t0 = time.time()
r = engine.predict(data, dates, n_portfolio=100)
elapsed = time.time() - t0

print(f"OK: {r['n_signals']} signals, {r['total_sets']} sets, coverage={r['coverage']}/45, time={elapsed:.1f}s")
print(f"Primary: {r['primary']}")
print(f"Top 5 weights: {dict(list(sorted(r['weights'].items(), key=lambda x: -x[1]))[:5])}")

from collections import Counter
strat = Counter(p['strategy'] for p in r['portfolio'])
print(f"Strategies: {dict(strat.most_common())}")
