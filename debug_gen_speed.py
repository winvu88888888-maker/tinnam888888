"""Quick debug: test gen speed for 1 draw."""
import sys, os, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper.data_manager import get_mega645_numbers
from v20_apex import compute_signals_v20, build_fusion_pool, gen_hybrid_portfolio

data = get_mega645_numbers()
te = 200

print("Computing signals...")
t0 = time.time()
sc = compute_signals_v20(data, te)
print(f"  Signals: {time.time()-t0:.2f}s")

print("Building pool...")
t0 = time.time()
pool = build_fusion_pool(data, te, sc, 42)
print(f"  Pool: {time.time()-t0:.2f}s, size={len(pool)}")

for n_tickets in [5000, 10000, 20000, 50000]:
    t0 = time.time()
    rng = random.Random(42)
    port = gen_hybrid_portfolio(pool, sc, n_tickets, rng)
    elapsed = time.time()-t0
    print(f"  Gen {n_tickets//1000}K: {elapsed:.2f}s → {len(port)} combos")
