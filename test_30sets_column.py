"""
30-SET COLUMN-BASED PREDICTOR
==============================
Strategy:
  1. For each column, generate Top-K candidates (from hot zone)
  2. Build ALL valid sorted combos from cross-product
  3. Score each combo using ensemble scoring
  4. Pick TOP-30 best combos
  5. Backtest: how often do 30 combos contain 6/6, 5/6, 4/6?

Test different K per column to find sweet spot.
"""
import sys, time, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations, product as iterproduct
from scraper.data_manager import get_mega645_numbers, get_mega645_all

data = get_mega645_numbers()
total = len(data)
MAX_NUM = 45
sorted_draws = [sorted(d[:6]) for d in data]

print(f"Data: {total} draws")
print(f"{'='*80}")
print(f" 30-SET COLUMN-BASED PREDICTOR — Backtest")
print(f"{'='*80}\n")

# ================================================================
# COLUMN CANDIDATE GENERATOR (fast version)
# ================================================================
def get_col_candidates(history, pos, K):
    """Get top-K candidates for column `pos` using ensemble scoring."""
    scores = Counter()
    n = len(history)
    vals = [h[pos] for h in history]
    lv = vals[-1]

    # Mode (multi-window)
    for w in [10, 20, 30, 50, 100]:
        seg = vals[-min(w,n):]
        for v, c in Counter(seg).most_common(5):
            scores[v] += c / len(seg) * 2

    # Transition
    trans = defaultdict(Counter)
    for i in range(n-1):
        trans[vals[i]][vals[i+1]] += 1
    if lv in trans:
        t = sum(trans[lv].values())
        for v, c in trans[lv].most_common(6):
            scores[v] += c / t * 4

    # Bigram
    if n >= 3:
        bg = defaultdict(Counter)
        for i in range(n-2):
            bg[(vals[i],vals[i+1])][vals[i+2]] += 1
        key = (vals[-2], vals[-1])
        if key in bg:
            t = sum(bg[key].values())
            for v, c in bg[key].most_common(4):
                scores[v] += c / t * 5

    # WMA ±1
    for w in [10, 15, 20]:
        seg = vals[-min(w,n):]
        weights = np.arange(1, len(seg)+1, dtype=float)
        wma = np.average(seg, weights=weights)
        for delta in range(-1, 2):
            v = int(round(wma)) + delta
            if 1 <= v <= MAX_NUM:
                scores[v] += 1.5

    # Conditional on prev/next col
    if pos > 0:
        cond = defaultdict(Counter)
        for h in history: cond[h[pos-1]][h[pos]] += 1
        pv = history[-1][pos-1]
        if pv in cond:
            t = sum(cond[pv].values())
            for v, c in cond[pv].most_common(4):
                scores[v] += c / t * 3
    if pos < 5:
        cond = defaultdict(Counter)
        for h in history: cond[h[pos+1]][h[pos]] += 1
        nv = history[-1][pos+1]
        if nv in cond:
            t = sum(cond[nv].values())
            for v, c in cond[nv].most_common(4):
                scores[v] += c / t * 3

    # Delta
    seg = vals[-50:]
    deltas = [seg[i+1]-seg[i] for i in range(len(seg)-1)]
    if deltas:
        for d, c in Counter(deltas).most_common(3):
            v = lv + d
            if 1 <= v <= MAX_NUM:
                scores[v] += c / len(deltas) * 2

    # Gap/overdue
    last_seen = {}
    for i, v in enumerate(vals): last_seen[v] = i
    gap_data = defaultdict(list)
    prev_idx = {}
    for i, v in enumerate(vals):
        if v in prev_idx: gap_data[v].append(i - prev_idx[v])
        prev_idx[v] = i
    for v in set(vals):
        if len(gap_data[v]) < 3: continue
        mg = np.mean(gap_data[v])
        cg = n - last_seen.get(v, 0)
        if mg > 0 and cg > mg * 1.2:
            scores[v] += min((cg/mg-1)*1.5, 3.0)

    # Last ±1
    for delta in [-1, 0, 1]:
        v = lv + delta
        if 1 <= v <= MAX_NUM:
            scores[v] += 1.0

    # Spacing (if pos > 0)
    if pos > 0:
        spacings = [h[pos]-h[pos-1] for h in history[-50:]]
        avg_sp = np.mean(spacings)
        pred = history[-1][pos-1] + int(round(avg_sp))
        for delta in [-1, 0, 1]:
            v = pred + delta
            if 1 <= v <= MAX_NUM:
                scores[v] += 1.0

    return [v for v, _ in scores.most_common(K)]


def score_combo(combo, history):
    """Score a combo using multiple signals."""
    score = 0
    last_draw = set(history[-1])
    n = len(history)

    # Signal 1: Sum of frequency scores
    freq = Counter()
    for h in history[-50:]:
        for num in h:
            freq[num] += 1
    for num in combo:
        score += freq.get(num, 0) * 0.1

    # Signal 2: Transition bonus
    follow = defaultdict(Counter)
    for i in range(n-1):
        for p in history[i]:
            for nx in history[i+1]:
                follow[p][nx] += 1
    for num in combo:
        for p in last_draw:
            score += follow[p].get(num, 0) * 0.02

    # Signal 3: Co-occurrence within combo
    pair_freq = Counter()
    for h in history[-200:]:
        hs = sorted(h)
        for p in combinations(hs, 2):
            pair_freq[p] += 1
    for p in combinations(sorted(combo), 2):
        score += pair_freq.get(p, 0) * 0.05

    # Signal 4: KNN similarity
    last = history[-1]
    for i in range(n-2):
        sim = len(set(history[i]) & set(last))
        if sim >= 3:
            match = len(set(history[i+1]) & set(combo))
            score += match * sim * 0.1

    # Signal 5: Diversity — prefer combos spread across range
    combo_range = combo[-1] - combo[0]
    score += combo_range * 0.1

    return score


# ================================================================
# BACKTEST
# ================================================================
START = 100
TESTED = total - START - 1
N_SETS = 30

# Test different K configurations
K_CONFIGS = [
    ("K3_uniform", [3,3,3,3,3,3]),
    ("K4_uniform", [4,4,4,4,4,4]),
    ("K5_uniform", [5,5,5,5,5,5]),
    ("K3_adaptive", [2,4,4,4,4,2]),  # fewer for pos1/6 (concentrated)
    ("K4_adaptive", [3,5,5,5,5,3]),
    ("K5_adaptive", [3,6,7,7,6,3]),
    ("K6_adaptive", [4,7,8,8,7,4]),
]

results = {}
for name, ks in K_CONFIGS:
    max_combos = 1
    for k in ks: max_combos *= k
    results[name] = {
        'ks': ks,
        'max_combos': max_combos,
        'h6': 0, 'h5': 0, 'h4': 0, 'h3': 0,
        'pos_hit': [0]*6,
        'valid_combos': [],
        'best_match_hist': [],
    }

t0 = time.time()
for idx in range(START, total-1):
    history = sorted_draws[:idx+1]
    actual = sorted_draws[idx+1]
    actual_set = set(actual)
    actual_tuple = tuple(actual)

    for name, ks in K_CONFIGS:
        r = results[name]

        # Get candidates per column
        pos_cands = []
        for pos in range(6):
            cands = get_col_candidates(history, pos, ks[pos])
            pos_cands.append(cands)
            if actual[pos] in cands:
                r['pos_hit'][pos] += 1

        # Generate valid combos (strictly increasing)
        valid = []
        for combo in iterproduct(*pos_cands):
            ok = True
            for j in range(5):
                if combo[j] >= combo[j+1]:
                    ok = False
                    break
            if ok:
                valid.append(combo)

        r['valid_combos'].append(len(valid))

        if not valid:
            r['best_match_hist'].append(0)
            continue

        # Score and pick top-30
        if len(valid) <= N_SETS:
            top_combos = valid
        else:
            scored = [(c, score_combo(c, history)) for c in valid]
            scored.sort(key=lambda x: -x[1])
            top_combos = [c for c, _ in scored[:N_SETS]]

        # Check matches
        best_match = 0
        hit6 = False
        for combo in top_combos:
            match = len(set(combo) & actual_set)
            best_match = max(best_match, match)
            if combo == actual_tuple:
                hit6 = True

        r['best_match_hist'].append(best_match)
        if hit6: r['h6'] += 1
        if best_match >= 5: r['h5'] += 1
        if best_match >= 4: r['h4'] += 1
        if best_match >= 3: r['h3'] += 1

    done = idx - START + 1
    if done % 100 == 0:
        el = time.time() - t0
        eta = el / done * (TESTED - done) / 60
        print(f"  [{done}/{TESTED}] {el:.0f}s ETA={eta:.1f}m")
        for name, _ in K_CONFIGS[:3]:
            r = results[name]
            avg_vc = np.mean(r['valid_combos']) if r['valid_combos'] else 0
            print(f"    {name}: 6/6={r['h6']} 5/6={r['h5']} "
                  f"4/6={r['h4']} 3/6={r['h3']} avg_valid={avg_vc:.0f}")
        sys.stdout.flush()

el = time.time() - t0
print(f"\n  Completed in {el:.0f}s ({el/60:.1f}m)\n")

# ================================================================
# RESULTS
# ================================================================
print(f"{'='*80}")
print(f" RESULTS — 30 SETS PER DRAW × {TESTED} DRAWS")
print(f"{'='*80}\n")

print(f"  {'Config':<16} {'K per col':<22} {'MaxCb':<8} {'AvgValid':<10} "
      f"{'6/6':<8} {'>=5/6':<8} {'>=4/6':<8} {'>=3/6':<8}")
print(f"  {'-'*96}")

for name, ks in K_CONFIGS:
    r = results[name]
    avg_vc = np.mean(r['valid_combos'])
    print(f"  {name:<16} {str(ks):<22} {r['max_combos']:<8} {avg_vc:<10.0f} "
          f"{r['h6']/TESTED*100:<8.2f} {r['h5']/TESTED*100:<8.2f} "
          f"{r['h4']/TESTED*100:<8.2f} {r['h3']/TESTED*100:<8.2f}")

print(f"\n  --- Per-Position Coverage (actual in candidates) ---\n")
print(f"  {'Config':<16}", end='')
for p in range(6):
    print(f" {'Col'+str(p+1):<8}", end='')
print(f" {'Joint':<8}")

for name, ks in K_CONFIGS:
    r = results[name]
    print(f"  {name:<16}", end='')
    joint = 1.0
    for p in range(6):
        cov = r['pos_hit'][p] / TESTED * 100
        joint *= cov / 100
        print(f" {cov:<8.1f}", end='')
    print(f" {joint*100:<8.2f}")

print(f"\n  --- Best Match Distribution ---\n")
for name, ks in K_CONFIGS:
    r = results[name]
    bm = r['best_match_hist']
    if bm:
        avg = np.mean(bm)
        print(f"  {name:<16}: avg_best_match={avg:.2f}/6, "
              f"0-match={sum(1 for x in bm if x==0)/len(bm)*100:.1f}%, "
              f">=2={sum(1 for x in bm if x>=2)/len(bm)*100:.1f}%, "
              f">=3={sum(1 for x in bm if x>=3)/len(bm)*100:.1f}%")

# ================================================================
# COMPARISON WITH RANDOM
# ================================================================
print(f"\n{'='*80}")
print(f" COMPARISON WITH RANDOM")
print(f"{'='*80}\n")

from math import comb
random_6 = 30 / comb(45, 6) * 100
print(f"  Random 30 sets: 6/6 = {random_6:.6f}%")
print(f"  Random 30 sets per draw = 1 hit per {comb(45,6)/30:.0f} draws")

for name, ks in K_CONFIGS:
    r = results[name]
    rate = r['h6'] / TESTED * 100
    if rate > 0:
        improvement = rate / random_6
        print(f"  {name}: 6/6={rate:.4f}% = {improvement:.0f}x random")
    else:
        print(f"  {name}: 6/6=0 (same as random for this sample)")

print(f"\n{'='*80}")
