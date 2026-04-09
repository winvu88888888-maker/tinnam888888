"""Quick sanity test V10 — single prediction."""
import sys, time, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.ultimate_engine import UltimateEngine
from scraper.data_manager import get_mega645_numbers, get_mega645_all

data = get_mega645_numbers()
dates = [r['draw_date'] for r in get_mega645_all()]
e = UltimateEngine(45, 6)

# Single prediction with 100 draws
t = time.time()
r = e.predict(data[:100], dates[:100])
print(f"Quick 100: {r['total_sets']} sets, conf={r['confidence']}, {time.time()-t:.1f}s")

# Single prediction with full data
t = time.time()
r = e.predict(data, dates)
print(f"Full: {r['total_sets']} sets, cov={r['coverage']}/45, conf={r['confidence']}, {time.time()-t:.1f}s")
print(f"Rescue: {r['rescue_nums'][:5]}")
