"""Quick test V12 speed."""
import sys, time, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.ultimate_engine import UltimateEngine
from scraper.data_manager import get_mega645_numbers, get_mega645_all
d = get_mega645_numbers(); dt = [r['draw_date'] for r in get_mega645_all()]
e = UltimateEngine(45, 6)
t = time.time(); r = e.predict(d, dt)
print(f"V12: {r['total_sets']} sets, pool={r['super_pool_size']}, cov={r['coverage']}/45, {time.time()-t:.1f}s")
