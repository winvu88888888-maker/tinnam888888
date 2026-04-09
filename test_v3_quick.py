"""Quick sanity test for V3."""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.ultimate_engine import UltimateEngine
from scraper.data_manager import get_mega645_numbers, get_mega645_all
from collections import Counter

data = get_mega645_numbers()
dates = [r['draw_date'] for r in get_mega645_all()]
print(f"Data: {len(data)} draws")

engine = UltimateEngine(45, 6)
t0 = time.time()
r = engine.predict(data, dates, n_portfolio=200)
print(f"OK: {r['n_signals']} signals, {r['total_sets']} sets, coverage={r['coverage']}/45, {time.time()-t0:.1f}s")
print(f"Primary: {r['primary']}")
print(f"Strategies: {dict(Counter(p['strategy'] for p in r['portfolio']).most_common())}")
