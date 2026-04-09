"""Quick V21 syntax + method existence test."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from models.master_predictor import MasterPredictor
import random, inspect

# Verify method exists
p = MasterPredictor(45, 6)
assert hasattr(p, '_generate_portfolio_v21'), "FAIL: _generate_portfolio_v21 missing!"
print("[OK] _generate_portfolio_v21 exists")

# Verify no Unicode arrows in print statements
source = inspect.getsource(MasterPredictor.predict)
assert '\u2192' not in source, "FAIL: Unicode arrow still in predict()"
print("[OK] No Unicode arrows in predict()")

# Test with minimal data (70 draws = minimum viable)
random.seed(42)
draws = [sorted(random.sample(range(1, 46), 6)) for _ in range(70)]
p2 = MasterPredictor(45, 6)
result = p2.predict(draws)

print(f"[OK] Numbers: {result['numbers']}")
print(f"[OK] Portfolio: {len(result['portfolio'])} sets")
print(f"[OK] Coverage: {result['coverage_pct']}%")
print(f"[OK] Method: {result['method']}")
print("ALL TESTS PASSED!")
