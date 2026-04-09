"""
ADAPTIVE PER-COLUMN COVERAGE — FAST VERSION
=============================================
ONLY measures per-position coverage (no combo enumeration).
Goal: Find minimum K per column for 100% coverage.
"""
import sys, time, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter, defaultdict
from scraper.data_manager import get_mega645_numbers, get_mega645_all

data = get_mega645_numbers()
total = len(data)
MAX_NUM = 45
sorted_draws = [sorted(d[:6]) for d in data]

print(f"Data: {total} draws")
print(f"{'='*90}")
print(f" ADAPTIVE PER-COLUMN COVERAGE — FAST (no combo enum)")
print(f" 30 methods per position, K=1..45")
print(f"{'='*90}\n")

def gen_candidates_pos(history, pos):
    """Generate scored candidates for position `pos`. Returns Counter."""
    candidates = Counter()
    n = len(history)
    vals = [h[pos] for h in history]
    lv = vals[-1]

    # 1. Multi-window MODE
    for w in [10, 20, 30, 50, 80, 100, 200, n]:
        seg = vals[-min(w,n):]
        for v, c in Counter(seg).most_common(5):
            candidates[v] += c / len(seg) * 2

    # 2. Transition matrix
    trans = defaultdict(Counter)
    for i in range(n-1):
        trans[vals[i]][vals[i+1]] += 1
    if lv in trans:
        t = sum(trans[lv].values())
        for v, c in trans[lv].most_common(8):
            candidates[v] += c / t * 4

    # 3. Bigram
    if n >= 3:
        bg = defaultdict(Counter)
        for i in range(n-2):
            bg[(vals[i],vals[i+1])][vals[i+2]] += 1
        key = (vals[-2], vals[-1])
        if key in bg:
            t = sum(bg[key].values())
            for v, c in bg[key].most_common(5):
                candidates[v] += c / t * 5

    # 4. Trigram
    if n >= 4:
        tg = defaultdict(Counter)
        for i in range(n-3):
            tg[(vals[i],vals[i+1],vals[i+2])][vals[i+3]] += 1
        key = (vals[-3], vals[-2], vals[-1])
        if key in tg:
            t = sum(tg[key].values())
            for v, c in tg[key].most_common(3):
                candidates[v] += c / t * 6

    # 5. 4-gram + 5-gram
    for gram_len in [4, 5]:
        if n >= gram_len + 1:
            ng = defaultdict(Counter)
            for i in range(n - gram_len):
                key = tuple(vals[i:i+gram_len])
                ng[key][vals[i+gram_len]] += 1
            key = tuple(vals[-gram_len:])
            if key in ng:
                t = sum(ng[key].values())
                for v, c in ng[key].most_common(3):
                    candidates[v] += c / t * (gram_len + 2)

    # 6. Multi-window MA ±2
    for w in [5, 10, 15, 20, 30, 50]:
        seg = vals[-min(w,n):]
        ma = np.mean(seg)
        for delta in range(-2, 3):
            v = int(round(ma)) + delta
            if 1 <= v <= MAX_NUM:
                candidates[v] += (2.0 if delta==0 else 1.5 if abs(delta)==1 else 0.8)

    # 7. WMA ±2
    for w in [10, 15, 20, 30]:
        seg = vals[-min(w,n):]
        weights = np.arange(1, len(seg)+1, dtype=float)
        wma = np.average(seg, weights=weights)
        for delta in range(-2, 3):
            v = int(round(wma)) + delta
            if 1 <= v <= MAX_NUM:
                candidates[v] += (2.0 if delta==0 else 1.0 if abs(delta)==1 else 0.5)

    # 8. EMA ±1
    for alpha in [0.1, 0.2, 0.3, 0.5, 0.7]:
        seg = vals[-30:]
        s = seg[0]
        for vv in seg[1:]: s = alpha*vv + (1-alpha)*s
        for delta in [-1, 0, 1]:
            v = int(round(s)) + delta
            if 1 <= v <= MAX_NUM:
                candidates[v] += 1.5

    # 9. Linear regression ±1
    for w in [20, 30, 50]:
        seg = vals[-min(w,n):]
        x = np.arange(len(seg))
        c = np.polyfit(x, seg, 1)
        pred = c[0]*len(seg) + c[1]
        for delta in [-1, 0, 1]:
            v = int(round(pred)) + delta
            if 1 <= v <= MAX_NUM:
                candidates[v] += 1.5

    # 10. Quadratic regression ±1
    try:
        seg = vals[-30:]
        x = np.arange(len(seg))
        c = np.polyfit(x, seg, 2)
        pred = c[0]*len(seg)**2 + c[1]*len(seg) + c[2]
        for delta in [-1, 0, 1]:
            v = int(round(pred)) + delta
            if 1 <= v <= MAX_NUM: candidates[v] += 1.0
    except: pass

    # 11. Conditional on prev/next position
    for other_pos in [pos-1, pos+1]:
        if 0 <= other_pos <= 5:
            cond = defaultdict(Counter)
            for h in history:
                cond[h[other_pos]][h[pos]] += 1
            ov = history[-1][other_pos]
            if ov in cond:
                t = sum(cond[ov].values())
                for v, c in cond[ov].most_common(5):
                    candidates[v] += c / t * 3

    # 12. Cross-2pos
    if 0 < pos < 5:
        cross = defaultdict(Counter)
        for i in range(n-1):
            cross[(history[i][pos-1], history[i][pos+1])][history[i+1][pos]] += 1
        key = (history[-1][pos-1], history[-1][pos+1])
        if key in cross:
            t = sum(cross[key].values())
            for v, c in cross[key].most_common(3):
                candidates[v] += c / t * 4

    # 13. KNN full draw
    last = history[-1]
    knn = Counter()
    for i in range(n-2):
        sim = sum(1 for j in range(6) if abs(history[i][j]-last[j]) <= 2)
        if sim >= 4: knn[history[i+1][pos]] += sim**2
    if knn:
        mx = max(knn.values())
        for v, s in knn.most_common(5):
            candidates[v] += s / mx * 3

    # 14. KNN exact pos
    knn_p = Counter()
    for i in range(n-1):
        if abs(vals[i] - lv) <= 1:
            knn_p[vals[i+1]] += 1
    if knn_p:
        mx = max(knn_p.values())
        for v, c in knn_p.most_common(5):
            candidates[v] += c / mx * 3

    # 15. Delta prediction
    for w in [20, 50, 100]:
        seg = vals[-min(w,n):]
        deltas = [seg[i+1]-seg[i] for i in range(len(seg)-1)]
        if deltas:
            for d, c in Counter(deltas).most_common(5):
                v = lv + d
                if 1 <= v <= MAX_NUM:
                    candidates[v] += c / len(deltas) * 2

    # 16. Delta bigram
    seg = vals[-min(100,n):]
    deltas = [seg[i+1]-seg[i] for i in range(len(seg)-1)]
    if len(deltas) >= 3:
        dtrans = defaultdict(Counter)
        for i in range(len(deltas)-2):
            dtrans[(deltas[i],deltas[i+1])][deltas[i+2]] += 1
        key = (deltas[-2], deltas[-1])
        if key in dtrans:
            t = sum(dtrans[key].values())
            for d, c in dtrans[key].most_common(3):
                v = lv + d
                if 1 <= v <= MAX_NUM:
                    candidates[v] += c / t * 4

    # 17. Modular arithmetic
    for m in [2, 3, 5, 7]:
        mods = [v % m for v in vals]
        mt = defaultdict(Counter)
        for i in range(len(mods)-1):
            mt[mods[i]][mods[i+1]] += 1
        lm = mods[-1]
        if lm in mt:
            top_mod = mt[lm].most_common(1)[0][0]
            recent = [v for v in vals[-50:] if v % m == top_mod]
            if recent:
                for v, c in Counter(recent).most_common(3):
                    candidates[v] += c / len(recent) * 2

    # 18. Gap / overdue
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
            candidates[v] += min((cg/mg-1)*1.5, 3.0)
        if mg > 0 and abs(cg-mg) < mg*0.3:
            candidates[v] += 1.5

    # 19. Lag repeat
    for lag in range(1, 15):
        if lag < n: candidates[vals[-lag]] += 0.3

    # 20. Cycle detection
    v_arr = np.array(vals[-200:], dtype=float)
    if len(v_arr) >= 30:
        vc = v_arr - v_arr.mean()
        std = vc.std()
        if std > 0:
            vc = vc / std
            for lag in range(2, min(30, len(vc)//3)):
                corr = np.mean(vc[:len(vc)-lag] * vc[lag:])
                if corr > 0.15 and lag <= len(vals):
                    candidates[vals[-lag]] += corr * 2

    # 21. Streak
    streak_val = lv
    streak_len = 1
    for i in range(n-2, -1, -1):
        if vals[i] == streak_val: streak_len += 1
        else: break
    if streak_len >= 2:
        candidates[streak_val] += streak_len * 1.5

    # 22. Direction
    if n >= 3:
        d1 = 1 if vals[-1]>vals[-2] else (-1 if vals[-1]<vals[-2] else 0)
        dir_next = Counter()
        for i in range(1, n-1):
            d = 1 if vals[i]>vals[i-1] else (-1 if vals[i]<vals[i-1] else 0)
            if d == d1: dir_next[vals[i+1]] += 1
        if dir_next:
            mx = max(dir_next.values())
            for v, c in dir_next.most_common(5):
                candidates[v] += c / mx * 2

    # 23. Median ±1
    for w in [10, 20, 30, 50]:
        seg = vals[-min(w,n):]
        med = np.median(seg)
        for delta in [-1, 0, 1]:
            v = int(round(med)) + delta
            if 1 <= v <= MAX_NUM: candidates[v] += 1.0

    # 24. Last ±2
    for delta in range(-2, 3):
        v = lv + delta
        if 1 <= v <= MAX_NUM:
            candidates[v] += (1.5 if delta==0 else 1.0 if abs(delta)==1 else 0.5)

    # 25. Digit sum transition
    dsums = [sum(int(c) for c in str(v)) for v in vals]
    ds_t = defaultdict(Counter)
    for i in range(len(dsums)-1):
        ds_t[dsums[i]][dsums[i+1]] += 1
    ld = dsums[-1]
    if ld in ds_t:
        for nds, cnt in ds_t[ld].most_common(2):
            recent = [v for v in vals[-50:] if sum(int(c) for c in str(v))==nds]
            if recent:
                for v, c in Counter(recent).most_common(3):
                    candidates[v] += c / len(recent) * 1.5

    # 26. Conditional on draw sum ± range
    sum_t = defaultdict(Counter)
    range_t = defaultdict(Counter)
    for h in history:
        sum_t[sum(h)][h[pos]] += 1
        range_t[h[5]-h[0]][h[pos]] += 1
    ls, lr = sum(history[-1]), history[-1][5]-history[-1][0]
    for bucket, trans_map, lval in [(sum_t, sum_t, ls), (range_t, range_t, lr)]:
        for d in range(-3, 4):
            key = lval + d
            if key in trans_map:
                t = sum(trans_map[key].values())
                for v, c in trans_map[key].most_common(3):
                    candidates[v] += c / t * (1.0 if d==0 else 0.3)

    # 27. Volatility regime
    std_r = np.std(vals[-20:])
    if std_r < 3:
        for v, c in Counter(vals[-20:]).most_common(3):
            candidates[v] += 2.0
    else:
        trend = np.mean(vals[-5:]) - np.mean(vals[-20:])
        pred = np.mean(vals[-5:]) + trend*0.5
        for delta in range(-2, 3):
            v = int(round(pred)) + delta
            if 1 <= v <= MAX_NUM: candidates[v] += 1.0

    # 28. DEMA (double exponential moving average)
    seg = vals[-30:]
    s1 = seg[0]; s2 = seg[0]
    alpha = 0.3
    for vv in seg[1:]:
        s1 = alpha*vv + (1-alpha)*s1
        s2 = alpha*s1 + (1-alpha)*s2
    pred = 2*s1 - s2
    for delta in [-1, 0, 1]:
        v = int(round(pred)) + delta
        if 1 <= v <= MAX_NUM: candidates[v] += 1.5

    # 29. ALL unique values ever seen at this position (fallback)
    for v in set(vals):
        if v not in candidates:
            candidates[v] += 0.01  # tiny score as fallback

    return candidates


# ================================================================
# BACKTEST: Coverage only (no combo enumeration)
# ================================================================
START = 100
TESTED = total - START - 1
MAX_K = 45

pos_coverage = [[0]*MAX_K for _ in range(6)]
joint_coverage = [0]*MAX_K

# Also track: at what rank does actual value appear? (distribution)
pos_rank_hist = [[] for _ in range(6)]

t0 = time.time()
for idx in range(START, total-1):
    history = sorted_draws[:idx+1]
    actual = sorted_draws[idx+1]

    ranks = []
    for pos in range(6):
        scores = gen_candidates_pos(history, pos)
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        ranked_vals = [v for v, _ in ranked]

        actual_v = actual[pos]
        if actual_v in ranked_vals:
            rank = ranked_vals.index(actual_v)
            ranks.append(rank)
            pos_rank_hist[pos].append(rank)
            for k in range(rank, MAX_K):
                pos_coverage[pos][k] += 1
        else:
            ranks.append(999)
            pos_rank_hist[pos].append(999)

    for k in range(MAX_K):
        if all(r <= k for r in ranks):
            joint_coverage[k] += 1

    done = idx - START + 1
    if done % 50 == 0:
        el = time.time() - t0
        eta = el / done * (TESTED - done) / 60
        cov5 = [pos_coverage[p][4]/done*100 for p in range(6)]
        cov10 = [pos_coverage[p][9]/done*100 for p in range(6)]
        j5 = joint_coverage[4]/done*100
        j10 = joint_coverage[9]/done*100
        print(f"  [{done}/{TESTED}] {el:.0f}s ETA={eta:.1f}m")
        print(f"    K5  per-pos: {' '.join(f'{c:.0f}%' for c in cov5)} | joint={j5:.1f}%")
        print(f"    K10 per-pos: {' '.join(f'{c:.0f}%' for c in cov10)} | joint={j10:.1f}%")
        sys.stdout.flush()

el = time.time() - t0
print(f"\n  Completed in {el:.0f}s ({el/60:.1f}m)\n")

# ================================================================
# RESULTS: PER-POSITION COVERAGE TABLE
# ================================================================
print(f"{'='*90}")
print(f" PER-POSITION COVERAGE AT VARIOUS K — {TESTED} draws")
print(f"{'='*90}\n")

header = f"  {'K':<4}"
for p in range(6):
    header += f"  {'Pos'+str(p+1):<9}"
header += f"  {'JOINT':<9}  {'Combos_max':<12}"
print(header)
print(f"  {'-'*85}")

for k in [1,2,3,4,5,6,7,8,9,10,12,15,18,20,25,30,35,40,45]:
    if k > MAX_K: continue
    line = f"  {k:<4}"
    for p in range(6):
        cov = pos_coverage[p][k-1] / TESTED * 100
        line += f"  {cov:<9.2f}"
    j = joint_coverage[k-1] / TESTED * 100
    combos = k**6
    line += f"  {j:<9.2f}  {combos:>12,}"
    print(line)

# ================================================================
# MINIMUM K FOR TARGETS
# ================================================================
print(f"\n{'='*90}")
print(f" MINIMUM K PER POSITION FOR TARGET COVERAGE")
print(f"{'='*90}\n")

for target in [80, 90, 95, 99, 99.5, 100]:
    print(f"  ── Target: {target}% ──")
    ks = []
    for p in range(6):
        found = False
        for k in range(MAX_K):
            cov = pos_coverage[p][k] / TESTED * 100
            if cov >= target:
                ks.append(k+1)
                print(f"    Pos {p+1}: K = {k+1:>3} ({cov:.2f}%)")
                found = True
                break
        if not found:
            ks.append(MAX_K+1)
            mc = pos_coverage[p][MAX_K-1]/TESTED*100
            print(f"    Pos {p+1}: K > {MAX_K} (max = {mc:.2f}%)")
    # Joint
    for k in range(MAX_K):
        if joint_coverage[k]/TESTED*100 >= target:
            print(f"    JOINT: K = {k+1} ({joint_coverage[k]/TESTED*100:.2f}%)")
            break
    else:
        print(f"    JOINT: K > {MAX_K} ({joint_coverage[MAX_K-1]/TESTED*100:.2f}%)")
    # Combo count for adaptive K
    if all(kk <= MAX_NUM for kk in ks):
        total_c = 1
        for kk in ks: total_c *= kk
        print(f"    Adaptive combos (max): {total_c:,}")
    print()

# ================================================================
# RANK DISTRIBUTION PER POSITION
# ================================================================
print(f"{'='*90}")
print(f" RANK DISTRIBUTION — Where does the actual value typically appear?")
print(f"{'='*90}\n")

for p in range(6):
    ranks = [r for r in pos_rank_hist[p] if r < 999]
    missed = sum(1 for r in pos_rank_hist[p] if r >= 999)
    if ranks:
        print(f"  Pos {p+1}: mean_rank={np.mean(ranks):.1f}, "
              f"median_rank={np.median(ranks):.0f}, "
              f"p90_rank={np.percentile(ranks,90):.0f}, "
              f"p99_rank={np.percentile(ranks,99):.0f}, "
              f"max_rank={max(ranks)}, "
              f"missed={missed}/{TESTED}")
    else:
        print(f"  Pos {p+1}: all missed!")

# ================================================================
# SUMMARY
# ================================================================
print(f"\n{'='*90}")
print(f" SUMMARY — CEILING FOR 6/6")
print(f"{'='*90}\n")

for k in [5, 10, 15, 20, 25, 30]:
    if k > MAX_K: continue
    j = joint_coverage[k-1] / TESTED * 100
    combos = k**6
    print(f"  K={k:>2} → Joint coverage = {j:.2f}% "
          f"(max {combos:>12,} combos)")

print(f"\n  Random baseline: 6/6 = 1/C(45,6) = 1/8,145,060 = 0.0000123%")
best_j = max(joint_coverage[k]/TESTED*100 for k in range(MAX_K))
best_k = max(range(MAX_K), key=lambda k: joint_coverage[k])
print(f"  Best joint coverage: K={best_k+1}, {best_j:.2f}%")

print(f"\n{'='*90}")
