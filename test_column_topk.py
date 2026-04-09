"""
TOP-K ENSEMBLE PER POSITION — COMBO EXPANSION BACKTEST
=======================================================
For each position, use MULTIPLE methods to generate top-K candidates.
Then combine positions to form valid combos, and check 6/6 rate.

Strategy:
  For each position, take the union of predictions from BEST methods.
  Then cross-product with ordering constraint (pos[i] < pos[i+1]).
"""
import sys, time, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter, defaultdict
from itertools import product as iterproduct
from scraper.data_manager import get_mega645_numbers, get_mega645_all

data = get_mega645_numbers()
dates = [r['draw_date'] for r in get_mega645_all()]
total = len(data)
MAX_NUM = 45
PICK = 6

sorted_draws = [sorted(d[:6]) for d in data]

# ================================================================
# Per-position prediction methods (compact versions)
# ================================================================
def get_candidates(history, pos, K=5):
    """Get top-K candidate values for position `pos` using ensemble of methods."""
    candidates = Counter()

    # Method 1: Mode (multiple windows)
    for w in [20, 30, 50, 80, 100]:
        vals = [h[pos] for h in history[-w:]]
        for v, c in Counter(vals).most_common(3):
            candidates[v] += c / len(vals) * 2

    # Method 2: Transition matrix — top-3
    trans = defaultdict(Counter)
    for i in range(len(history) - 1):
        trans[history[i][pos]][history[i+1][pos]] += 1
    last_val = history[-1][pos]
    if last_val in trans:
        t = sum(trans[last_val].values())
        for v, c in trans[last_val].most_common(5):
            candidates[v] += c / t * 3

    # Method 3: Bigram — top-3
    if len(history) >= 3:
        bg = defaultdict(Counter)
        for i in range(len(history) - 2):
            key = (history[i][pos], history[i+1][pos])
            bg[key][history[i+2][pos]] += 1
        key = (history[-2][pos], history[-1][pos])
        if key in bg:
            t = sum(bg[key].values())
            for v, c in bg[key].most_common(3):
                candidates[v] += c / t * 4  # Higher weight for specific pattern

    # Method 4: Moving average ±1
    for w in [10, 20, 30]:
        vals = [h[pos] for h in history[-w:]]
        ma = np.mean(vals)
        for delta in range(-1, 2):
            v = int(round(ma)) + delta
            if 1 <= v <= MAX_NUM:
                candidates[v] += 1.5

    # Method 5: Weighted MA ±1
    vals = [h[pos] for h in history[-15:]]
    weights = np.arange(1, len(vals) + 1, dtype=float)
    wma = np.average(vals, weights=weights)
    for delta in range(-1, 2):
        v = int(round(wma)) + delta
        if 1 <= v <= MAX_NUM:
            candidates[v] += 1.5

    # Method 6: Conditional on previous position
    if pos > 0:
        cond = defaultdict(Counter)
        for h in history:
            cond[h[pos-1]][h[pos]] += 1
        prev_val = history[-1][pos-1]
        if prev_val in cond:
            t = sum(cond[prev_val].values())
            for v, c in cond[prev_val].most_common(3):
                candidates[v] += c / t * 2

    # Method 7: KNN (similar past draws → next draw's pos value)
    last = history[-1]
    knn_scores = Counter()
    for i in range(len(history) - 2):
        sim = sum(1 for j in range(6) if abs(history[i][j] - last[j]) <= 2)
        if sim >= 4 and i + 1 < len(history):
            knn_scores[history[i + 1][pos]] += sim ** 2
    for v, s in knn_scores.most_common(3):
        candidates[v] += s / max(knn_scores.values()) * 2 if knn_scores else 0

    # Method 8: Delta prediction (most common delta)
    vals = [h[pos] for h in history[-50:]]
    deltas = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
    if deltas:
        for d_val, d_cnt in Counter(deltas).most_common(3):
            v = vals[-1] + d_val
            if 1 <= v <= MAX_NUM:
                candidates[v] += d_cnt / len(deltas) * 2

    # Method 9: Overdue values at this position
    vals = [h[pos] for h in history]
    last_seen_p = {}
    for i, v in enumerate(vals):
        last_seen_p[v] = i
    n = len(vals)
    overdue = sorted(last_seen_p.items(), key=lambda x: x[1])[:3]
    for v, ls in overdue:
        if n - ls > 10:
            candidates[v] += 1.0

    # Method 10: Repeat last draw value
    candidates[history[-1][pos]] += 1.0

    # Method 11: Last ±1, ±2
    for delta in [-2, -1, 1, 2]:
        v = history[-1][pos] + delta
        if 1 <= v <= MAX_NUM:
            candidates[v] += 0.5

    # Return top-K
    return [v for v, _ in candidates.most_common(K)]


# ================================================================
# BACKTEST: K candidates per position → cross-product → check 6/6
# ================================================================
print(f"Data: {total} draws\n")
print(f"{'='*80}")
print(f" TOP-K ENSEMBLE PER POSITION → COMBO EXPANSION BACKTEST")
print(f"{'='*80}\n")

START = 100
TESTED = total - START - 1

K_VALUES = [3, 5, 7, 10, 15]
results = {k: {'h6': 0, 'h5': 0, 'h4': 0, 'h3': 0, 'combos': [],
                'pos_hit': [0]*6} for k in K_VALUES}

t0 = time.time()
for idx in range(START, total - 1):
    history = sorted_draws[:idx+1]
    actual = sorted_draws[idx+1]
    actual_tuple = tuple(actual)

    for K in K_VALUES:
        # Get candidates per position
        pos_candidates = []
        for pos in range(6):
            cands = get_candidates(history, pos, K)
            pos_candidates.append(cands)
            # Check if actual value is in candidates
            if actual[pos] in cands:
                results[K]['pos_hit'][pos] += 1

        # Cross-product with ordering constraint
        valid_combos = []
        for combo in iterproduct(*pos_candidates):
            # Check strictly increasing (valid sorted combo)
            valid = True
            for j in range(5):
                if combo[j] >= combo[j+1]:
                    valid = False
                    break
            if valid:
                valid_combos.append(combo)

        results[K]['combos'].append(len(valid_combos))

        # Check matches
        best_match = 0
        hit6 = False
        for combo in valid_combos:
            match = len(set(combo) & set(actual))
            best_match = max(best_match, match)
            if combo == actual_tuple:
                hit6 = True

        if hit6:
            results[K]['h6'] += 1
        if best_match >= 5:
            results[K]['h5'] += 1
        if best_match >= 4:
            results[K]['h4'] += 1
        if best_match >= 3:
            results[K]['h3'] += 1

    done = idx - START + 1
    if done % 100 == 0:
        el = time.time() - t0
        eta = el / done * (TESTED - done) / 60
        line = f"  [{done}/{TESTED}] {el:.0f}s ETA={eta:.1f}m |"
        for K in K_VALUES:
            r = results[K]
            avg_c = np.mean(r['combos']) if r['combos'] else 0
            line += f" K{K}: 6/6={r['h6']},≥4={r['h4']},avg_c={avg_c:.0f} |"
        print(line)
        sys.stdout.flush()

# ================================================================
# FINAL RESULTS
# ================================================================
el = time.time() - t0
print(f"\n{'='*80}")
print(f" RESULTS — {TESTED} draws, {el:.0f}s ({el/60:.1f}m)")
print(f"{'='*80}\n")

print(f"  {'K':>4}  {'Avg Combos':>12}  {'6/6':>8}  {'≥5/6':>8}  {'≥4/6':>8}  {'≥3/6':>8}")
print(f"  {'-'*60}")
for K in K_VALUES:
    r = results[K]
    avg_c = np.mean(r['combos'])
    print(f"  {K:>4}  {avg_c:>12,.0f}  "
          f"{r['h6']/TESTED*100:>7.2f}%  "
          f"{r['h5']/TESTED*100:>7.2f}%  "
          f"{r['h4']/TESTED*100:>7.2f}%  "
          f"{r['h3']/TESTED*100:>7.2f}%")

print(f"\n  --- Per-Position Coverage (actual value in Top-K candidates) ---\n")
print(f"  {'K':>4}  {'Pos1':>8}  {'Pos2':>8}  {'Pos3':>8}  {'Pos4':>8}  {'Pos5':>8}  {'Pos6':>8}  | Joint")
for K in K_VALUES:
    r = results[K]
    accs = [r['pos_hit'][p] / TESTED * 100 for p in range(6)]
    joint = 1.0
    for a in accs:
        joint *= a / 100
    line = f"  {K:>4}"
    for a in accs:
        line += f"  {a:>7.2f}%"
    line += f"  | {joint*100:.4f}%"
    print(line)

print(f"\n{'='*80}")
print(f" KEY INSIGHT: Per-position coverage tells the CEILING.")
print(f" Even if we generate ALL valid combos from K candidates,")
print(f" 6/6 can ONLY happen if actual[i] is in candidates[i] for ALL 6 positions.")
print(f"{'='*80}")
