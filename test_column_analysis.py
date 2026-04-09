"""
COLUMN-BY-COLUMN POSITIONAL ANALYSIS
=====================================
Goal: For each of the 6 sorted positions, test EVERY prediction method
to find the one that predicts the EXACT value most often.

If per-position accuracy = A%, then joint 6/6 = A1 * A2 * ... * A6

Position 1 = smallest number in the draw (sorted)
Position 6 = largest number in the draw (sorted)

METHODS TESTED PER POSITION:
  1. Mode (most common value at this position)
  2. Last value (repeat last draw's value at this pos)
  3. Moving average (rounded)
  4. Weighted moving average (recent = heavier)
  5. Transition matrix (P(next_pos=x | last_pos=y))
  6. Bigram (2-draw history)
  7. Trigram (3-draw history pattern)
  8. Linear regression (trend)
  9. Momentum-adjusted (short vs long term trend)
  10. Median (rolling median)
  11. Most-frequent-in-window (mode of recent N)
  12. Gap-adjusted (overdue value detection)
  13. KNN (find most similar past draws → predict next)
  14. Cycle FFT (periodic pattern detection)
  15. Markov chain (stationary distribution)
  16. Conditional on other positions (if pos1=X, P(pos2=Y))
  17. Exponential smoothing
  18. Range-center (mid of recent range)
  19. Combined vote (ensemble of top methods)
"""
import sys, time, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter, defaultdict
from scraper.data_manager import get_mega645_numbers, get_mega645_all

data = get_mega645_numbers()
dates = [r['draw_date'] for r in get_mega645_all()]
total = len(data)
MAX_NUM = 45
PICK = 6

# Pre-sort all draws
sorted_draws = [sorted(d[:6]) for d in data]

print(f"Data: {total} draws\n")
print(f"{'='*80}")
print(f" COLUMN-BY-COLUMN POSITIONAL ANALYSIS")
print(f" Testing {18} prediction methods × {6} positions")
print(f"{'='*80}\n")

# ================================================================
# PHASE 1: Distribution analysis per position
# ================================================================
print("--- PHASE 1: POSITIONAL DISTRIBUTION ---\n")
for pos in range(6):
    values = [sd[pos] for sd in sorted_draws]
    print(f"  Position {pos+1}: min={min(values)}, max={max(values)}, "
          f"mean={np.mean(values):.1f}, std={np.std(values):.1f}, "
          f"median={np.median(values):.0f}, unique={len(set(values))}")
print()

# ================================================================
# PHASE 2: Test ALL methods per position
# ================================================================
START = 100  # Need enough history
TESTED = total - START - 1

# Methods return a single predicted value for position `pos`
# given history sorted_draws[:idx+1]

def method_mode(history, pos, window=50):
    """Most common value at this position in recent window."""
    vals = [h[pos] for h in history[-window:]]
    return Counter(vals).most_common(1)[0][0]

def method_last(history, pos):
    """Repeat last draw's value at this position."""
    return history[-1][pos]

def method_ma(history, pos, window=10):
    """Moving average (rounded)."""
    vals = [h[pos] for h in history[-window:]]
    return int(round(np.mean(vals)))

def method_wma(history, pos, window=15):
    """Weighted moving average (recent draws weighted more)."""
    vals = [h[pos] for h in history[-window:]]
    weights = np.arange(1, len(vals) + 1, dtype=float)
    return int(round(np.average(vals, weights=weights)))

def method_transition(history, pos):
    """Transition matrix: P(next_pos=x | last_pos=y)."""
    trans = defaultdict(Counter)
    for i in range(len(history) - 1):
        trans[history[i][pos]][history[i+1][pos]] += 1
    last_val = history[-1][pos]
    if last_val in trans and trans[last_val]:
        return trans[last_val].most_common(1)[0][0]
    return last_val

def method_bigram(history, pos):
    """2-draw history: P(next | last_2)."""
    if len(history) < 3:
        return history[-1][pos]
    trans = defaultdict(Counter)
    for i in range(len(history) - 2):
        key = (history[i][pos], history[i+1][pos])
        trans[key][history[i+2][pos]] += 1
    key = (history[-2][pos], history[-1][pos])
    if key in trans and trans[key]:
        return trans[key].most_common(1)[0][0]
    return method_transition(history, pos)

def method_trigram(history, pos):
    """3-draw history pattern."""
    if len(history) < 4:
        return history[-1][pos]
    trans = defaultdict(Counter)
    for i in range(len(history) - 3):
        key = (history[i][pos], history[i+1][pos], history[i+2][pos])
        trans[key][history[i+3][pos]] += 1
    key = (history[-3][pos], history[-2][pos], history[-1][pos])
    if key in trans and trans[key]:
        return trans[key].most_common(1)[0][0]
    return method_bigram(history, pos)

def method_linear_reg(history, pos, window=30):
    """Linear regression trend extrapolation."""
    vals = [h[pos] for h in history[-window:]]
    x = np.arange(len(vals))
    coeffs = np.polyfit(x, vals, 1)
    pred = coeffs[0] * len(vals) + coeffs[1]
    return max(1, min(MAX_NUM, int(round(pred))))

def method_momentum(history, pos):
    """Momentum: short-term vs long-term trend adjustment."""
    v_short = np.mean([h[pos] for h in history[-5:]])
    v_long = np.mean([h[pos] for h in history[-30:]])
    momentum = v_short - v_long
    pred = v_short + momentum * 0.5
    return max(1, min(MAX_NUM, int(round(pred))))

def method_median(history, pos, window=20):
    """Rolling median."""
    vals = [h[pos] for h in history[-window:]]
    return int(round(np.median(vals)))

def method_mode_short(history, pos):
    """Mode of recent 20 draws."""
    return method_mode(history, pos, window=20)

def method_mode_long(history, pos):
    """Mode of recent 100 draws."""
    return method_mode(history, pos, window=100)

def method_gap(history, pos):
    """Gap-adjusted: find value that is 'overdue' at this position."""
    vals = [h[pos] for h in history]
    last_seen = {}
    for i, v in enumerate(vals):
        last_seen[v] = i
    n = len(vals)
    # Compute mean gap for each value
    gap_data = defaultdict(list)
    prev_idx = {}
    for i, v in enumerate(vals):
        if v in prev_idx:
            gap_data[v].append(i - prev_idx[v])
        prev_idx[v] = i
    # Find most overdue
    best_val = vals[-1]
    best_ratio = 0
    for v in set(vals):
        if len(gap_data[v]) < 3:
            continue
        mg = np.mean(gap_data[v])
        cg = n - last_seen.get(v, 0)
        ratio = cg / mg if mg > 0 else 0
        if ratio > best_ratio:
            best_ratio = ratio
            best_val = v
    return best_val

def method_knn(history, pos, k=5):
    """KNN: find most similar past draws, predict from their successors."""
    last = history[-1]
    sims = []
    for i in range(len(history) - 2):
        sim = sum(1 for j in range(6) if abs(history[i][j] - last[j]) <= 2)
        sims.append((sim, i))
    sims.sort(key=lambda x: -x[0])
    preds = Counter()
    for sim, idx in sims[:k]:
        if idx + 1 < len(history):
            preds[history[idx + 1][pos]] += sim
    if preds:
        return preds.most_common(1)[0][0]
    return last[pos]

def method_exp_smooth(history, pos, alpha=0.3):
    """Exponential smoothing."""
    vals = [h[pos] for h in history[-30:]]
    s = vals[0]
    for v in vals[1:]:
        s = alpha * v + (1 - alpha) * s
    return max(1, min(MAX_NUM, int(round(s))))

def method_range_center(history, pos, window=15):
    """Center of recent range at this position."""
    vals = [h[pos] for h in history[-window:]]
    return int(round((min(vals) + max(vals)) / 2))

def method_conditional(history, pos):
    """Conditional on previous position's value."""
    if pos == 0:
        return method_transition(history, pos)
    # P(pos_value | prev_pos_value in same draw)
    trans = defaultdict(Counter)
    for h in history:
        trans[h[pos-1]][h[pos]] += 1
    last_prev = history[-1][pos-1]
    if last_prev in trans and trans[last_prev]:
        return trans[last_prev].most_common(1)[0][0]
    return history[-1][pos]

def method_delta_predict(history, pos):
    """Predict based on most common delta (change from draw to draw)."""
    vals = [h[pos] for h in history[-50:]]
    deltas = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
    if not deltas:
        return vals[-1]
    most_common_delta = Counter(deltas).most_common(1)[0][0]
    pred = vals[-1] + most_common_delta
    return max(1, min(MAX_NUM, pred))

def method_top2_transition(history, pos):
    """Top-2 from transition, pick the one with highest frequency."""
    trans = defaultdict(Counter)
    for i in range(len(history) - 1):
        trans[history[i][pos]][history[i+1][pos]] += 1
    last_val = history[-1][pos]
    if last_val in trans and trans[last_val]:
        top2 = trans[last_val].most_common(2)
        # Pick the one that's also the most globally frequent
        freq = Counter(h[pos] for h in history[-50:])
        if len(top2) >= 2:
            if freq.get(top2[0][0], 0) >= freq.get(top2[1][0], 0):
                return top2[0][0]
            else:
                return top2[1][0]
        return top2[0][0]
    return last_val

# All methods list
METHODS = {
    'mode_50':          lambda h, p: method_mode(h, p, 50),
    'mode_20':          method_mode_short,
    'mode_100':         method_mode_long,
    'last':             method_last,
    'ma_10':            lambda h, p: method_ma(h, p, 10),
    'ma_20':            lambda h, p: method_ma(h, p, 20),
    'wma_15':           lambda h, p: method_wma(h, p, 15),
    'transition':       method_transition,
    'bigram':           method_bigram,
    'trigram':           method_trigram,
    'linear_reg':       method_linear_reg,
    'momentum':         method_momentum,
    'median_20':        method_median,
    'gap_overdue':      method_gap,
    'knn_5':            lambda h, p: method_knn(h, p, 5),
    'knn_10':           lambda h, p: method_knn(h, p, 10),
    'exp_smooth':       method_exp_smooth,
    'range_center':     method_range_center,
    'conditional':      method_conditional,
    'delta_predict':    method_delta_predict,
    'top2_transition':  method_top2_transition,
}

# ================================================================
# RUN BACKTEST
# ================================================================
print(f"--- PHASE 2: BACKTEST {len(METHODS)} METHODS × 6 POSITIONS ---\n")

# results[method][pos] = count of exact hits
results = {m: [0]*6 for m in METHODS}
# Also track top-K accuracy (within ±1, ±2)
results_pm1 = {m: [0]*6 for m in METHODS}
results_pm2 = {m: [0]*6 for m in METHODS}

t0 = time.time()
for idx in range(START, total - 1):
    history = sorted_draws[:idx+1]
    actual = sorted_draws[idx+1]

    for mname, mfunc in METHODS.items():
        for pos in range(6):
            try:
                pred = mfunc(history, pos)
                if pred == actual[pos]:
                    results[mname][pos] += 1
                if abs(pred - actual[pos]) <= 1:
                    results_pm1[mname][pos] += 1
                if abs(pred - actual[pos]) <= 2:
                    results_pm2[mname][pos] += 1
            except:
                pass

    done = idx - START + 1
    if done % 200 == 0:
        el = time.time() - t0
        eta = el / done * (TESTED - done) / 60
        print(f"  [{done}/{TESTED}] {el:.0f}s ETA={eta:.1f}m")

el = time.time() - t0
print(f"\n  Completed in {el:.0f}s ({el/60:.1f}m)\n")

# ================================================================
# RESULTS
# ================================================================
print(f"{'='*80}")
print(f" EXACT HIT ACCURACY PER METHOD × POSITION ({TESTED} draws)")
print(f"{'='*80}\n")

pos_labels = ['Pos1', 'Pos2', 'Pos3', 'Pos4', 'Pos5', 'Pos6']
header = f"  {'Method':<20}" + "".join(f"  {pl:>8}" for pl in pos_labels) + "  | Joint%"
print(header)
print("  " + "-"*len(header))

best_per_pos = [('', 0)] * 6  # (method, accuracy)

for mname in sorted(METHODS.keys()):
    accs = [results[mname][pos] / TESTED * 100 for pos in range(6)]
    # Joint probability = product of per-position accuracies
    joint = 1.0
    for a in accs:
        joint *= (a / 100)
    joint *= 100

    line = f"  {mname:<20}"
    for pos in range(6):
        line += f"  {accs[pos]:>7.2f}%"
    line += f"  |  {joint:.6f}%"
    print(line)

    # Track best per position
    for pos in range(6):
        if accs[pos] > best_per_pos[pos][1]:
            best_per_pos[pos] = (mname, accs[pos])

# Best per position summary
print(f"\n{'='*80}")
print(f" BEST METHOD PER POSITION")
print(f"{'='*80}\n")

joint_best = 1.0
for pos in range(6):
    mname, acc = best_per_pos[pos]
    joint_best *= (acc / 100)
    print(f"  Position {pos+1}: {mname:<20} = {acc:.2f}% exact")
print(f"\n  BEST JOINT 6/6 (product) = {joint_best*100:.6f}%")
print(f"  Expected 6/6 in {TESTED} draws = {joint_best * TESTED:.2f}")

# ================================================================
# ±1 and ±2 accuracy
# ================================================================
print(f"\n{'='*80}")
print(f" WITHIN ±1 ACCURACY (best method per position)")
print(f"{'='*80}\n")

best_pm1 = [('', 0)] * 6
for mname in METHODS:
    for pos in range(6):
        acc = results_pm1[mname][pos] / TESTED * 100
        if acc > best_pm1[pos][1]:
            best_pm1[pos] = (mname, acc)

joint_pm1 = 1.0
for pos in range(6):
    mname, acc = best_pm1[pos]
    joint_pm1 *= (acc / 100)
    print(f"  Position {pos+1}: {mname:<20} = {acc:.2f}% (±1)")
print(f"\n  JOINT ±1 ALL 6 = {joint_pm1*100:.4f}%")

print(f"\n{'='*80}")
print(f" WITHIN ±2 ACCURACY (best method per position)")
print(f"{'='*80}\n")

best_pm2 = [('', 0)] * 6
for mname in METHODS:
    for pos in range(6):
        acc = results_pm2[mname][pos] / TESTED * 100
        if acc > best_pm2[pos][1]:
            best_pm2[pos] = (mname, acc)

joint_pm2 = 1.0
for pos in range(6):
    mname, acc = best_pm2[pos]
    joint_pm2 *= (acc / 100)
    print(f"  Position {pos+1}: {mname:<20} = {acc:.2f}% (±2)")
print(f"\n  JOINT ±2 ALL 6 = {joint_pm2*100:.4f}%")

# ================================================================
# TOP-K expansion: how many combos if we take top-K per position?
# ================================================================
print(f"\n{'='*80}")
print(f" TOP-K CANDIDATES PER POSITION → COMBO COUNT")
print(f"{'='*80}\n")

for k in [1, 2, 3, 4, 5]:
    combos = k ** 6
    print(f"  Top-{k} per position → {k}^6 = {combos:>6,} possible combos")

print(f"\n  (Need to subtract invalid: pos[i] >= pos[i+1] cases)")

# ================================================================
# POSITION VALUE RANGE (unique values at each position)
# ================================================================
print(f"\n{'='*80}")
print(f" ENTROPY PER POSITION (How many distinct values?)")
print(f"{'='*80}\n")

for pos in range(6):
    vals = [sd[pos] for sd in sorted_draws]
    vc = Counter(vals)
    n_unique = len(vc)
    top5 = vc.most_common(5)
    top5_pct = sum(c for _, c in top5) / len(vals) * 100
    entropy = -sum((c/len(vals)) * np.log2(c/len(vals)) for c in vc.values())
    print(f"  Position {pos+1}: {n_unique} unique values, "
          f"top-5 cover {top5_pct:.1f}%, entropy={entropy:.2f} bits")
    print(f"    Top-5: {', '.join(f'{v}({c})' for v, c in top5)}")

print(f"\n{'='*80}")
