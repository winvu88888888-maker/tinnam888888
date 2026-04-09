"""
ADAPTIVE PER-COLUMN TOP-K PREDICTOR
=====================================
Goal: For EACH column, use specialized ensemble of best methods
to generate Top-K candidates such that coverage → 100%.

Strategy per column:
  - Pos 1: mod3, transition, mode_50, top2_trans, cond_prev (concentrated dist)
  - Pos 2: wma15, digit_sum, linreg30, wma20, top2_trans (wide dist)
  - Pos 3: ma20, linreg30, wma15, ma10, transition (widest dist)
  - Pos 4: knn_exact, mode_30, transition, cond_prev, mod3 (wide dist)
  - Pos 5: ma20, top2_trans, linreg30, wma30, ema02 (wide dist)
  - Pos 6: streak, mode_all, mod7, mode_50, digit_sum (concentrated dist)

For each position, generate candidates from ALL methods (union),
then rank by score. Test K=1..45 to find minimum K for 100% coverage.
"""
import sys, time, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter, defaultdict
from itertools import product as iterproduct
from scraper.data_manager import get_mega645_numbers, get_mega645_all

data = get_mega645_numbers()
total = len(data)
MAX_NUM = 45
sorted_draws = [sorted(d[:6]) for d in data]

print(f"Data: {total} draws")
print(f"{'='*90}")
print(f" ADAPTIVE PER-COLUMN TOP-K PREDICTOR — Target 100% Coverage")
print(f"{'='*90}\n")

# ================================================================
# PER-POSITION CANDIDATE GENERATORS
# Each returns a SCORED dict {value: score}
# ================================================================

def gen_candidates_pos(history, pos):
    """Generate scored candidates for position `pos` using ALL relevant methods.
    Returns dict {value: score}."""
    candidates = Counter()
    n = len(history)
    vals = [h[pos] for h in history]
    lv = vals[-1]

    # === METHOD 1: Multi-window MODE ===
    for w in [10, 20, 30, 50, 80, 100, 200, n]:
        ww = min(w, n)
        seg = vals[-ww:]
        for v, c in Counter(seg).most_common(5):
            candidates[v] += c / len(seg) * 2

    # === METHOD 2: Transition matrix ===
    trans = defaultdict(Counter)
    for i in range(n-1):
        trans[vals[i]][vals[i+1]] += 1
    if lv in trans:
        t = sum(trans[lv].values())
        for v, c in trans[lv].most_common(8):
            candidates[v] += c / t * 4

    # === METHOD 3: Bigram ===
    if n >= 3:
        bg = defaultdict(Counter)
        for i in range(n-2):
            key = (vals[i], vals[i+1])
            bg[key][vals[i+2]] += 1
        key = (vals[-2], vals[-1])
        if key in bg:
            t = sum(bg[key].values())
            for v, c in bg[key].most_common(5):
                candidates[v] += c / t * 5

    # === METHOD 4: Trigram ===
    if n >= 4:
        tg = defaultdict(Counter)
        for i in range(n-3):
            key = (vals[i], vals[i+1], vals[i+2])
            tg[key][vals[i+3]] += 1
        key = (vals[-3], vals[-2], vals[-1])
        if key in tg:
            t = sum(tg[key].values())
            for v, c in tg[key].most_common(3):
                candidates[v] += c / t * 6

    # === METHOD 5: 4-gram ===
    if n >= 5:
        fg = defaultdict(Counter)
        for i in range(n-4):
            key = tuple(vals[i:i+4])
            fg[key][vals[i+4]] += 1
        key = tuple(vals[-4:])
        if key in fg:
            t = sum(fg[key].values())
            for v, c in fg[key].most_common(3):
                candidates[v] += c / t * 8

    # === METHOD 6: Multi-window MA ±1, ±2 ===
    for w in [5, 10, 15, 20, 30, 50]:
        ww = min(w, n)
        seg = vals[-ww:]
        ma = np.mean(seg)
        for delta in range(-2, 3):
            v = int(round(ma)) + delta
            if 1 <= v <= MAX_NUM:
                weight = 2.0 if delta == 0 else (1.5 if abs(delta) == 1 else 0.8)
                candidates[v] += weight

    # === METHOD 7: WMA ±1, ±2 ===
    for w in [10, 15, 20, 30]:
        ww = min(w, n)
        seg = vals[-ww:]
        weights = np.arange(1, len(seg)+1, dtype=float)
        wma = np.average(seg, weights=weights)
        for delta in range(-2, 3):
            v = int(round(wma)) + delta
            if 1 <= v <= MAX_NUM:
                weight = 2.0 if delta == 0 else (1.0 if abs(delta) == 1 else 0.5)
                candidates[v] += weight

    # === METHOD 8: EMA ±1 ===
    for alpha in [0.1, 0.2, 0.3, 0.5, 0.7]:
        seg = vals[-30:]
        s = seg[0]
        for vv in seg[1:]:
            s = alpha * vv + (1-alpha) * s
        for delta in range(-1, 2):
            v = int(round(s)) + delta
            if 1 <= v <= MAX_NUM:
                candidates[v] += 1.5

    # === METHOD 9: Linear regression ±1 ===
    for w in [20, 30, 50]:
        ww = min(w, n)
        seg = vals[-ww:]
        x = np.arange(len(seg))
        c = np.polyfit(x, seg, 1)
        pred = c[0]*len(seg) + c[1]
        for delta in range(-1, 2):
            v = int(round(pred)) + delta
            if 1 <= v <= MAX_NUM:
                candidates[v] += 1.5

    # === METHOD 10: Quadratic regression ===
    seg = vals[-30:]
    x = np.arange(len(seg))
    try:
        c = np.polyfit(x, seg, 2)
        pred = c[0]*len(seg)**2 + c[1]*len(seg) + c[2]
        for delta in range(-1, 2):
            v = int(round(pred)) + delta
            if 1 <= v <= MAX_NUM:
                candidates[v] += 1.0
    except:
        pass

    # === METHOD 11: Conditional on previous position ===
    if pos > 0:
        cond = defaultdict(Counter)
        for h in history:
            cond[h[pos-1]][h[pos]] += 1
        pv = history[-1][pos-1]
        if pv in cond:
            t = sum(cond[pv].values())
            for v, c in cond[pv].most_common(5):
                candidates[v] += c / t * 3

    # === METHOD 12: Conditional on next position ===
    if pos < 5:
        cond = defaultdict(Counter)
        for h in history:
            cond[h[pos+1]][h[pos]] += 1
        nv = history[-1][pos+1]
        if nv in cond:
            t = sum(cond[nv].values())
            for v, c in cond[nv].most_common(5):
                candidates[v] += c / t * 3

    # === METHOD 13: Cross-2-position ===
    if pos > 0 and pos < 5:
        cross = defaultdict(Counter)
        for i in range(n-1):
            key = (history[i][pos-1], history[i][pos+1])
            cross[key][history[i+1][pos]] += 1
        key = (history[-1][pos-1], history[-1][pos+1])
        if key in cross:
            t = sum(cross[key].values())
            for v, c in cross[key].most_common(3):
                candidates[v] += c / t * 4

    # === METHOD 14: KNN (similar full draws) ===
    last = history[-1]
    knn = Counter()
    for i in range(n-2):
        sim = sum(1 for j in range(6) if abs(history[i][j]-last[j]) <= 2)
        if sim >= 4 and i+1 < n:
            knn[history[i+1][pos]] += sim**2
    if knn:
        mx = max(knn.values())
        for v, s in knn.most_common(5):
            candidates[v] += s / mx * 3

    # === METHOD 15: KNN exact position ===
    knn_p = Counter()
    for i in range(n-1):
        if abs(vals[i] - lv) <= 1:
            knn_p[vals[i+1]] += 1
    if knn_p:
        mx = max(knn_p.values())
        for v, c in knn_p.most_common(5):
            candidates[v] += c / mx * 3

    # === METHOD 16: Delta prediction ===
    for w in [20, 50, 100]:
        ww = min(w, n)
        seg = vals[-ww:]
        deltas = [seg[i+1]-seg[i] for i in range(len(seg)-1)]
        if deltas:
            for d, c in Counter(deltas).most_common(5):
                v = lv + d
                if 1 <= v <= MAX_NUM:
                    candidates[v] += c / len(deltas) * 2

    # === METHOD 17: Delta bigram ===
    seg = vals[-100:]
    deltas = [seg[i+1]-seg[i] for i in range(len(seg)-1)]
    if len(deltas) >= 3:
        dtrans = defaultdict(Counter)
        for i in range(len(deltas)-2):
            key = (deltas[i], deltas[i+1])
            dtrans[key][deltas[i+2]] += 1
        key = (deltas[-2], deltas[-1])
        if key in dtrans:
            t = sum(dtrans[key].values())
            for d, c in dtrans[key].most_common(3):
                v = lv + d
                if 1 <= v <= MAX_NUM:
                    candidates[v] += c / t * 4

    # === METHOD 18: Modular arithmetic ===
    for m in [2, 3, 5, 7]:
        mods = [v % m for v in vals]
        mtrans = defaultdict(Counter)
        for i in range(len(mods)-1):
            mtrans[mods[i]][mods[i+1]] += 1
        lm = mods[-1]
        if lm in mtrans:
            top_mod = mtrans[lm].most_common(1)[0][0]
            recent_with_mod = [v for v in vals[-50:] if v % m == top_mod]
            if recent_with_mod:
                for v, c in Counter(recent_with_mod).most_common(3):
                    candidates[v] += c / len(recent_with_mod) * 2

    # === METHOD 19: Gap / overdue ===
    last_seen = {}
    for i, v in enumerate(vals):
        last_seen[v] = i
    gap_data = defaultdict(list)
    prev_idx = {}
    for i, v in enumerate(vals):
        if v in prev_idx:
            gap_data[v].append(i - prev_idx[v])
        prev_idx[v] = i
    for v in set(vals):
        if len(gap_data[v]) < 3: continue
        mg = np.mean(gap_data[v])
        cg = n - last_seen.get(v, 0)
        if mg > 0 and cg > mg * 1.2:
            candidates[v] += min((cg / mg - 1) * 1.5, 3.0)

    # === METHOD 20: Gap expected (DUE NOW) ===
    for v in set(vals):
        if len(gap_data[v]) < 5: continue
        mg = np.mean(gap_data[v])
        cg = n - last_seen.get(v, 0)
        if mg > 0 and abs(cg - mg) < mg * 0.3:
            candidates[v] += 1.5

    # === METHOD 21: Lag repeat ===
    for lag in range(1, 15):
        if lag < n:
            candidates[vals[-lag]] += 0.3

    # === METHOD 22: Cycle detection ===
    v_arr = np.array(vals[-200:], dtype=float)
    if len(v_arr) >= 30:
        v_arr_c = v_arr - v_arr.mean()
        std = v_arr_c.std()
        if std > 0:
            v_arr_c = v_arr_c / std
            for lag in range(2, min(30, len(v_arr_c)//3)):
                corr = np.mean(v_arr_c[:len(v_arr_c)-lag] * v_arr_c[lag:])
                if corr > 0.15 and lag <= len(vals):
                    candidates[vals[-lag]] += corr * 2

    # === METHOD 23: Streak continuation ===
    streak_val = lv
    streak_len = 1
    for i in range(n-2, -1, -1):
        if vals[i] == streak_val:
            streak_len += 1
        else:
            break
    if streak_len >= 2:
        candidates[streak_val] += streak_len * 1.5
    # After streak breaks:
    streak_break = Counter()
    for i in range(1, n):
        sl = 1
        for j in range(i-1, -1, -1):
            if vals[j] == vals[i]: sl += 1
            else: break
        if sl == streak_len and i+1 < n:
            streak_break[vals[i+1]] += 1
    if streak_break:
        for v, c in streak_break.most_common(3):
            candidates[v] += c * 0.5

    # === METHOD 24: Direction-based ===
    if n >= 3:
        d1 = 1 if vals[-1] > vals[-2] else (-1 if vals[-1] < vals[-2] else 0)
        dir_next = Counter()
        for i in range(1, n-1):
            d = 1 if vals[i] > vals[i-1] else (-1 if vals[i] < vals[i-1] else 0)
            if d == d1:
                dir_next[vals[i+1]] += 1
        if dir_next:
            mx = max(dir_next.values())
            for v, c in dir_next.most_common(5):
                candidates[v] += c / mx * 2

    # === METHOD 25: Median ±1 ===
    for w in [10, 20, 30, 50]:
        ww = min(w, n)
        seg = vals[-ww:]
        med = np.median(seg)
        for delta in range(-1, 2):
            v = int(round(med)) + delta
            if 1 <= v <= MAX_NUM:
                candidates[v] += 1.0

    # === METHOD 26: Last value ±1, ±2 ===
    for delta in range(-2, 3):
        v = lv + delta
        if 1 <= v <= MAX_NUM:
            weight = 1.5 if delta == 0 else (1.0 if abs(delta) == 1 else 0.5)
            candidates[v] += weight

    # === METHOD 27: Digit sum transition ===
    dsums = [sum(int(c) for c in str(v)) for v in vals]
    ds_trans = defaultdict(Counter)
    for i in range(len(dsums)-1):
        ds_trans[dsums[i]][dsums[i+1]] += 1
    ld = dsums[-1]
    if ld in ds_trans:
        top_ds = ds_trans[ld].most_common(2)
        for next_ds, cnt in top_ds:
            recent = [v for v in vals[-50:] if sum(int(c) for c in str(v)) == next_ds]
            if recent:
                for v, c in Counter(recent).most_common(3):
                    candidates[v] += c / len(recent) * 1.5

    # === METHOD 28: Conditional on draw sum ===
    sum_trans = defaultdict(Counter)
    for h in history:
        s = sum(h)
        sum_trans[s][h[pos]] += 1
    ls = sum(history[-1])
    for ds in range(-5, 6):
        key = ls + ds
        if key in sum_trans:
            t = sum(sum_trans[key].values())
            for v, c in sum_trans[key].most_common(3):
                weight = 1.0 if ds == 0 else 0.3
                candidates[v] += c / t * weight

    # === METHOD 29: Conditional on draw range ===
    range_trans = defaultdict(Counter)
    for h in history:
        r = h[5] - h[0]
        range_trans[r][h[pos]] += 1
    lr = history[-1][5] - history[-1][0]
    for dr in range(-3, 4):
        key = lr + dr
        if key in range_trans:
            t = sum(range_trans[key].values())
            for v, c in range_trans[key].most_common(3):
                weight = 1.0 if dr == 0 else 0.3
                candidates[v] += c / t * weight

    # === METHOD 30: Volatility regime ===
    std_recent = np.std(vals[-20:])
    if std_recent < 3:
        # Low vol → favor mode
        for v, c in Counter(vals[-20:]).most_common(3):
            candidates[v] += 2.0
    else:
        # High vol → favor trend
        trend = np.mean(vals[-5:]) - np.mean(vals[-20:])
        pred = np.mean(vals[-5:]) + trend * 0.5
        for delta in range(-2, 3):
            v = int(round(pred)) + delta
            if 1 <= v <= MAX_NUM:
                candidates[v] += 1.0

    return candidates


# ================================================================
# BACKTEST: Per-column coverage at various K values
# ================================================================
START = 100
TESTED = total - START - 1

# Track per-position coverage at each K
MAX_K = 45
pos_coverage = [[0]*MAX_K for _ in range(6)]  # pos_coverage[pos][k-1] = count of draws where actual in top-k

# Track joint coverage (all 6 positions covered)
joint_coverage = [0]*MAX_K

# Track combo counts for selected K values
K_TEST = [3, 5, 7, 10, 12, 15, 20, 25, 30]
combo_data = {k: {'combos': [], 'h6': 0, 'h5': 0, 'h4': 0, 'h3': 0} for k in K_TEST}

t0 = time.time()
for idx in range(START, total-1):
    history = sorted_draws[:idx+1]
    actual = sorted_draws[idx+1]

    # Generate candidates per position
    pos_ranked = []  # pos_ranked[pos] = sorted list of candidates
    pos_hit_at_k = []  # pos_hit_at_k[pos] = rank where actual value appears (0-indexed), or -1

    for pos in range(6):
        scores = gen_candidates_pos(history, pos)
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        ranked_vals = [v for v, _ in ranked]
        pos_ranked.append(ranked_vals)

        # Find where actual value is in ranked list
        actual_v = actual[pos]
        if actual_v in ranked_vals:
            rank = ranked_vals.index(actual_v)
            pos_hit_at_k.append(rank)
            # Update coverage counts
            for k in range(rank, MAX_K):
                pos_coverage[pos][k] += 1
        else:
            pos_hit_at_k.append(-1)

    # Update joint coverage
    for k in range(MAX_K):
        if all(pos_hit_at_k[p] != -1 and pos_hit_at_k[p] <= k for p in range(6)):
            joint_coverage[k] += 1

    # Test specific K values for combo analysis
    for K in K_TEST:
        cands_per_pos = [pr[:K] for pr in pos_ranked]

        # Check position hits
        all_pos_hit = all(actual[p] in cands_per_pos[p] for p in range(6))

        # Count valid combos (strictly increasing)
        valid_count = 0
        best_match = 0
        hit6 = False

        # For small K, enumerate; for large K, just count
        if K <= 15:
            for combo in iterproduct(*cands_per_pos):
                valid = True
                for j in range(5):
                    if combo[j] >= combo[j+1]:
                        valid = False
                        break
                if valid:
                    valid_count += 1
                    match = len(set(combo) & set(actual))
                    best_match = max(best_match, match)
                    if combo == tuple(actual):
                        hit6 = True
        else:
            # For large K, just check if all 6 actuals are in candidates
            valid_count = -1  # too many to count
            if all_pos_hit:
                hit6 = True  # actual combo IS one of the valid combos

        combo_data[K]['combos'].append(valid_count)
        if hit6: combo_data[K]['h6'] += 1
        if best_match >= 5 or (K > 15 and all_pos_hit): combo_data[K]['h5'] += 1
        if best_match >= 4: combo_data[K]['h4'] += 1
        if best_match >= 3: combo_data[K]['h3'] += 1

    done = idx - START + 1
    if done % 100 == 0:
        el = time.time() - t0
        eta = el / done * (TESTED - done) / 60
        # Quick status
        cov_k10 = [pos_coverage[p][9]/done*100 for p in range(6)]
        j_k10 = joint_coverage[9]/done*100
        print(f"  [{done}/{TESTED}] {el:.0f}s ETA={eta:.1f}m | "
              f"K10 per-pos: {' '.join(f'{c:.0f}%' for c in cov_k10)} | joint={j_k10:.1f}%")
        sys.stdout.flush()

el = time.time() - t0
print(f"\n  Completed in {el:.0f}s ({el/60:.1f}m)\n")

# ================================================================
# RESULTS
# ================================================================
print(f"{'='*90}")
print(f" PER-POSITION COVERAGE AT VARIOUS K")
print(f"{'='*90}\n")

print(f"  {'K':<5}", end='')
for p in range(6):
    print(f"  {'Pos'+str(p+1):<10}", end='')
print(f"  {'JOINT':<10}")
print(f"  {'-'*75}")

for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 35, 40, 45]:
    if k > MAX_K: continue
    print(f"  {k:<5}", end='')
    for p in range(6):
        cov = pos_coverage[p][k-1] / TESTED * 100
        print(f"  {cov:<10.2f}", end='')
    j = joint_coverage[k-1] / TESTED * 100
    print(f"  {j:<10.2f}")

# ================================================================
# MINIMUM K FOR TARGET COVERAGE
# ================================================================
print(f"\n{'='*90}")
print(f" MINIMUM K PER POSITION FOR TARGET COVERAGE")
print(f"{'='*90}\n")

for target in [90, 95, 99, 100]:
    print(f"  Target: {target}% coverage")
    for p in range(6):
        for k in range(MAX_K):
            cov = pos_coverage[p][k] / TESTED * 100
            if cov >= target:
                print(f"    Pos {p+1}: K = {k+1} ({cov:.2f}%)")
                break
        else:
            print(f"    Pos {p+1}: K > {MAX_K} (max coverage = "
                  f"{pos_coverage[p][MAX_K-1]/TESTED*100:.2f}%)")
    # Joint
    for k in range(MAX_K):
        j = joint_coverage[k] / TESTED * 100
        if j >= target:
            print(f"    JOINT: K = {k+1} ({j:.2f}%)")
            break
    else:
        print(f"    JOINT: K > {MAX_K} (max = {joint_coverage[MAX_K-1]/TESTED*100:.2f}%)")
    print()

# ================================================================
# COMBO ANALYSIS
# ================================================================
print(f"{'='*90}")
print(f" COMBO ANALYSIS — Can actual combo be found in Top-K expansion?")
print(f"{'='*90}\n")

print(f"  {'K':<5}  {'6/6 Rate':<12}  {'Avg Combos':<15}  {'6/6 Count':<12}  {'Joint Cov%':<12}")
print(f"  {'-'*60}")
for K in K_TEST:
    r = combo_data[K]
    rate = r['h6'] / TESTED * 100
    valid = [c for c in r['combos'] if c > 0]
    avg_c = np.mean(valid) if valid else 0
    j_cov = joint_coverage[K-1] / TESTED * 100
    print(f"  {K:<5}  {rate:<12.2f}  {avg_c:<15,.0f}  {r['h6']:<12}  {j_cov:<12.2f}")

# ================================================================
# OPTIMAL ADAPTIVE K PER POSITION
# ================================================================
print(f"\n{'='*90}")
print(f" OPTIMAL ADAPTIVE K — Different K per position for 95%+ coverage")
print(f"{'='*90}\n")

# Find min K for 95% per position
adaptive_K = []
for p in range(6):
    for k in range(MAX_K):
        cov = pos_coverage[p][k] / TESTED * 100
        if cov >= 95:
            adaptive_K.append(k+1)
            break
    else:
        adaptive_K.append(MAX_K)

print(f"  Adaptive K for 95% per-position coverage:")
total_combos_est = 1
for p in range(6):
    k = adaptive_K[p]
    cov = pos_coverage[p][k-1] / TESTED * 100
    total_combos_est *= k
    print(f"    Pos {p+1}: K = {k} ({cov:.2f}%)")
print(f"\n  Max combos (before ordering filter): {total_combos_est:,}")
print(f"  Estimated valid combos (after strictly-increasing filter): ~{total_combos_est//10:,}")

# Same for 99%
print(f"\n  Adaptive K for 99% per-position coverage:")
adaptive_K99 = []
total_99 = 1
for p in range(6):
    for k in range(MAX_K):
        cov = pos_coverage[p][k] / TESTED * 100
        if cov >= 99:
            adaptive_K99.append(k+1)
            break
    else:
        adaptive_K99.append(MAX_K)
for p in range(6):
    k = adaptive_K99[p]
    cov = pos_coverage[p][k-1] / TESTED * 100
    total_99 *= k
    print(f"    Pos {p+1}: K = {k} ({cov:.2f}%)")
print(f"\n  Max combos: {total_99:,}")

# Same for 100%
print(f"\n  Adaptive K for 100% per-position coverage:")
adaptive_K100 = []
total_100 = 1
for p in range(6):
    for k in range(MAX_K):
        cov = pos_coverage[p][k] / TESTED * 100
        if cov >= 100:
            adaptive_K100.append(k+1)
            break
    else:
        adaptive_K100.append(MAX_K)
for p in range(6):
    k = adaptive_K100[p]
    cov = pos_coverage[p][k-1] / TESTED * 100
    total_100 *= k
    print(f"    Pos {p+1}: K = {k} ({cov:.2f}%)")
print(f"\n  Max combos: {total_100:,}")

print(f"\n{'='*90}")
print(f" SUMMARY: Per-column coverage tells us the CEILING.")
print(f" If actual value is NOT in Top-K candidates for ANY column,")
print(f" then 6/6 is IMPOSSIBLE regardless of combo selection.")
print(f"{'='*90}\n")

# The true 6/6 ceiling = joint coverage
print(f"  K=10  → Joint coverage = {joint_coverage[9]/TESTED*100:.2f}% "
      f"(6/6 CEILING)")
print(f"  K=15  → Joint coverage = {joint_coverage[14]/TESTED*100:.2f}% "
      f"(6/6 CEILING)")
print(f"  K=20  → Joint coverage = {joint_coverage[19]/TESTED*100:.2f}% "
      f"(6/6 CEILING)")
print(f"  K=25  → Joint coverage = {joint_coverage[24]/TESTED*100:.2f}% "
      f"(6/6 CEILING)")
print(f"  K=30  → Joint coverage = {joint_coverage[29]/TESTED*100:.2f}% "
      f"(6/6 CEILING)")

print(f"\n{'='*90}")
