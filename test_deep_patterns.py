"""
DEEP PATTERN MINING PER COLUMN — Phase 2
==========================================
1. Transition chain mining (markov, bigram, trigram, 4-gram with entropy)
2. Cycle/periodicity detection (autocorrelation, FFT spectral peaks)
3. Run-length & streak patterns
4. Inter-column dependencies (spacing distribution, conditional MI)
5. ULTRA predictor combining ALL discovered patterns → BACKTEST

Goal: Find exploitable structural patterns WITHIN each column's narrow range,
then combine with inter-column constraints for a BETTER predictor.
"""
import sys, time, os, warnings
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
import numpy as np
from collections import Counter, defaultdict
from scraper.data_manager import get_mega645_numbers

data = get_mega645_numbers()
total = len(data)
MAX_NUM = 45
sorted_draws = [sorted(d[:6]) for d in data]

print(f"Data: {total} draws")
print(f"{'='*90}")
print(f" DEEP PATTERN MINING — Per-Column Analysis")
print(f"{'='*90}\n")

# ================================================================
# PART 1: TRANSITION CHAIN ANALYSIS
# ================================================================
print(f"{'='*90}")
print(f" PART 1: TRANSITION CHAIN ANALYSIS")
print(f"{'='*90}\n")

for pos in range(6):
    vals = [sd[pos] for sd in sorted_draws]
    n = len(vals)
    vc = Counter(vals)
    n_unique = len(vc)

    print(f"  ===== Column {pos+1} ({n_unique} unique values) =====\n")

    # --- 1a. Markov transition matrix entropy ---
    trans = defaultdict(Counter)
    for i in range(n-1):
        trans[vals[i]][vals[i+1]] += 1

    # For top-10 most frequent values, show their transition distribution
    top10 = vc.most_common(10)
    print(f"    --- Transition Matrix (top sources → top destinations) ---")
    for src, src_count in top10[:5]:
        if src not in trans: continue
        t = sum(trans[src].values())
        dests = trans[src].most_common(5)
        dest_str = " | ".join(f"{v}:{c/t*100:.0f}%" for v, c in dests)
        # Entropy of transition distribution
        probs = [c/t for c in trans[src].values()]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        print(f"      {src:>2} → [{dest_str}] (H={entropy:.2f} bits, n={t})")
    print()

    # --- 1b. Transition predictability ---
    # What % of transitions can be predicted by just picking the mode?
    correct_1 = 0
    correct_2 = 0
    correct_3 = 0
    for i in range(n-1):
        if vals[i] in trans:
            mc = trans[vals[i]].most_common(3)
            if mc[0][0] == vals[i+1]: correct_1 += 1
            if any(v == vals[i+1] for v, _ in mc[:2]): correct_2 += 1
            if any(v == vals[i+1] for v, _ in mc[:3]): correct_3 += 1
    print(f"    Transition prediction accuracy:")
    print(f"      Top-1: {correct_1/(n-1)*100:.1f}%  "
          f"Top-2: {correct_2/(n-1)*100:.1f}%  "
          f"Top-3: {correct_3/(n-1)*100:.1f}%")

    # --- 1c. Bigram prediction accuracy ---
    bg = defaultdict(Counter)
    for i in range(n-2):
        bg[(vals[i], vals[i+1])][vals[i+2]] += 1
    bg_correct = 0
    bg_total = 0
    for i in range(2, n):
        key = (vals[i-2], vals[i-1])
        if key in bg and bg[key]:
            pred = bg[key].most_common(1)[0][0]
            if pred == vals[i]: bg_correct += 1
            bg_total += 1
    if bg_total > 0:
        print(f"      Bigram top-1: {bg_correct/bg_total*100:.1f}% (of {bg_total} testable)")

    # --- 1d. Repeat probability ---
    repeat = sum(1 for i in range(1, n) if vals[i] == vals[i-1])
    print(f"      Repeat probability: {repeat/(n-1)*100:.1f}%")
    print()

# ================================================================
# PART 2: CYCLE / PERIODICITY DETECTION
# ================================================================
print(f"\n{'='*90}")
print(f" PART 2: CYCLE / PERIODICITY DETECTION (Autocorrelation)")
print(f"{'='*90}\n")

for pos in range(6):
    vals = np.array([sd[pos] for sd in sorted_draws], dtype=float)
    n = len(vals)
    # Normalize
    vm = vals - vals.mean()
    std = vm.std()
    if std < 0.01: continue
    vm = vm / std

    print(f"  ===== Column {pos+1} =====")

    # Autocorrelation for lags 1..50
    max_lag = min(50, n//3)
    acf = []
    for lag in range(1, max_lag+1):
        corr = np.mean(vm[:n-lag] * vm[lag:])
        acf.append((lag, corr))

    # Find significant peaks (above 2/sqrt(n) threshold)
    threshold = 2 / np.sqrt(n)
    significant = [(lag, c) for lag, c in acf if abs(c) > threshold]

    if significant:
        top5 = sorted(significant, key=lambda x: -abs(x[1]))[:5]
        for lag, c in top5:
            direction = "+" if c > 0 else "-"
            print(f"    Lag {lag:>3}: r={c:>+.4f} {direction} "
                  f"({'***' if abs(c) > 0.1 else '**' if abs(c) > 0.05 else '*'})")
    else:
        print(f"    No significant autocorrelation found (threshold={threshold:.4f})")

    # FFT spectral analysis
    if n >= 64:
        fft_vals = np.abs(np.fft.rfft(vm))
        freqs = np.fft.rfftfreq(n)
        # Skip DC component (index 0)
        fft_vals[0] = 0
        top_indices = np.argsort(fft_vals)[-3:][::-1]
        print(f"    FFT dominant periods: ", end="")
        for idx in top_indices:
            if freqs[idx] > 0:
                period = 1/freqs[idx]
                power = fft_vals[idx]
                print(f"{period:.1f} draws (power={power:.1f}), ", end="")
        print()

    print()

# ================================================================
# PART 3: RUN-LENGTH & STREAK PATTERNS
# ================================================================
print(f"{'='*90}")
print(f" PART 3: RUN-LENGTH & STREAK PATTERNS")
print(f"{'='*90}\n")

for pos in range(6):
    vals = [sd[pos] for sd in sorted_draws]
    n = len(vals)

    # Direction runs: up, down, same
    directions = []
    for i in range(1, n):
        if vals[i] > vals[i-1]: directions.append('U')
        elif vals[i] < vals[i-1]: directions.append('D')
        else: directions.append('S')

    dir_counts = Counter(directions)
    print(f"  Column {pos+1}: Up={dir_counts.get('U',0)/(n-1)*100:.1f}% "
          f"Down={dir_counts.get('D',0)/(n-1)*100:.1f}% "
          f"Same={dir_counts.get('S',0)/(n-1)*100:.1f}%")

    # Run length analysis (consecutive same-direction moves)
    run_lengths = {'U': [], 'D': [], 'S': []}
    if directions:
        cur_dir = directions[0]
        cur_len = 1
        for i in range(1, len(directions)):
            if directions[i] == cur_dir:
                cur_len += 1
            else:
                run_lengths[cur_dir].append(cur_len)
                cur_dir = directions[i]
                cur_len = 1
        run_lengths[cur_dir].append(cur_len)

    for d in ['U', 'D', 'S']:
        runs = run_lengths[d]
        if runs:
            print(f"    {d}-runs: avg={np.mean(runs):.2f}, max={max(runs)}, "
                  f"dist: {Counter(runs).most_common(5)}")

    # Direction transition → what follows a streak of U/D/S?
    # After 2+ consecutive Up moves, what happens next?
    for d in ['U', 'D']:
        after_streak = Counter()
        for i in range(2, len(directions)):
            if directions[i-2] == d and directions[i-1] == d:
                after_streak[directions[i]] += 1
        if after_streak:
            t = sum(after_streak.values())
            print(f"    After {d}{d}: {', '.join(f'{k}={v/t*100:.0f}%' for k,v in after_streak.most_common())}")

    print()

# ================================================================
# PART 4: INTER-COLUMN DEPENDENCIES
# ================================================================
print(f"{'='*90}")
print(f" PART 4: INTER-COLUMN DEPENDENCIES")
print(f"{'='*90}\n")

# --- 4a. Spacing distribution between adjacent columns ---
print(f"  --- Adjacent Column Spacing ---\n")
for gap_pair in range(5):
    spacings = [sd[gap_pair+1] - sd[gap_pair] for sd in sorted_draws]
    sc = Counter(spacings)
    top5 = sc.most_common(5)
    print(f"  Col{gap_pair+1}→Col{gap_pair+2}: "
          f"mean={np.mean(spacings):.1f}, std={np.std(spacings):.1f}, "
          f"median={np.median(spacings):.0f}")
    print(f"    Top spacings: {', '.join(f'{s}({c/len(spacings)*100:.0f}%)' for s,c in top5)}")
    # Min spacing (must be >= 1)
    print(f"    Range: [{min(spacings)}, {max(spacings)}]")
    print()

# --- 4b. Cross-column correlation ---
print(f"  --- Cross-Column Correlation Matrix ---\n")
cols = np.array([[sd[p] for sd in sorted_draws] for p in range(6)], dtype=float)
corr = np.corrcoef(cols)
print(f"    {'':>6}", end='')
for p in range(6): print(f" Col{p+1:>2}", end='')
print()
for p1 in range(6):
    print(f"    Col{p1+1}", end='')
    for p2 in range(6):
        print(f"  {corr[p1,p2]:>5.2f}", end='')
    print()
print()

# --- 4c. Delta correlation (change in col_i vs change in col_j) ---
print(f"  --- Delta Correlation (simultaneous changes) ---\n")
deltas = np.diff(cols, axis=1)
dcorr = np.corrcoef(deltas)
print(f"    {'':>6}", end='')
for p in range(6): print(f" dC{p+1:>2} ", end='')
print()
for p1 in range(6):
    print(f"    dC{p1+1}", end='')
    for p2 in range(6):
        print(f"  {dcorr[p1,p2]:>5.2f}", end='')
    print()
print()

# --- 4d. Conditional spacing: given col_i value, predict col_j spacing ---
print(f"  --- Spacing Prediction Accuracy ---\n")
for gap_pair in range(5):
    spacings = [sd[gap_pair+1] - sd[gap_pair] for sd in sorted_draws]
    vals_prev = [sd[gap_pair] for sd in sorted_draws]
    # Build spacing predictor: given col_i value, predict spacing
    sp_trans = defaultdict(Counter)
    for i in range(len(vals_prev)):
        sp_trans[vals_prev[i]][spacings[i]] += 1

    correct = 0
    for i in range(1, len(vals_prev)):
        vp = vals_prev[i]
        if vp in sp_trans:
            pred_sp = sp_trans[vp].most_common(1)[0][0]
            if pred_sp == spacings[i]:
                correct += 1
    print(f"  Col{gap_pair+1}→Col{gap_pair+2} spacing predictable: "
          f"{correct/(len(vals_prev)-1)*100:.1f}%")

print()

# --- 4e. Draw-level sum & range patterns ---
print(f"  --- Draw Sum & Range Patterns ---\n")
sums = [sum(sd) for sd in sorted_draws]
ranges = [sd[5]-sd[0] for sd in sorted_draws]
print(f"  Sum:   mean={np.mean(sums):.1f}, std={np.std(sums):.1f}, "
      f"range=[{min(sums)}, {max(sums)}]")
print(f"  Range: mean={np.mean(ranges):.1f}, std={np.std(ranges):.1f}, "
      f"range=[{min(ranges)}, {max(ranges)}]")

# Sum given previous sum
sum_trans = defaultdict(Counter)
for i in range(len(sums)-1):
    # Bucket sums into ranges of 10
    bucket = sums[i] // 10 * 10
    sum_trans[bucket][sums[i+1] // 10 * 10] += 1
print(f"\n  Sum transition (bucketed by 10):")
for bucket in sorted(sum_trans.keys()):
    t = sum(sum_trans[bucket].values())
    top = sum_trans[bucket].most_common(3)
    print(f"    [{bucket},{bucket+9}] → "
          f"{', '.join(f'[{b},{b+9}]:{c/t*100:.0f}%' for b,c in top)}")
print()

# ================================================================
# PART 5: EXPLOITABILITY SCORE — How much can we beat random?
# ================================================================
print(f"{'='*90}")
print(f" PART 5: EXPLOITABILITY SCORE PER COLUMN")
print(f"{'='*90}\n")

for pos in range(6):
    vals = [sd[pos] for sd in sorted_draws]
    n = len(vals)
    vc = Counter(vals)
    n_unique = len(vc)

    # Random baseline for this column
    random_acc = 1.0 / n_unique * 100

    # Method 1: Mode (overall)
    mode_val = vc.most_common(1)[0][0]
    mode_acc = vc[mode_val] / n * 100

    # Method 2: Transition top-1
    trans = defaultdict(Counter)
    for i in range(n-1):
        trans[vals[i]][vals[i+1]] += 1
    trans_correct = 0
    for i in range(1, n):
        if vals[i-1] in trans:
            pred = trans[vals[i-1]].most_common(1)[0][0]
            if pred == vals[i]: trans_correct += 1
    trans_acc = trans_correct / (n-1) * 100

    # Method 3: Conditional on previous column
    cond_acc = 0
    if pos > 0:
        cond = defaultdict(Counter)
        for sd in sorted_draws:
            cond[sd[pos-1]][sd[pos]] += 1
        cond_correct = 0
        for i in range(1, len(sorted_draws)):
            pv = sorted_draws[i][pos-1]
            if pv in cond:
                pred = cond[pv].most_common(1)[0][0]
                if pred == sorted_draws[i][pos]: cond_correct += 1
        cond_acc = cond_correct / (len(sorted_draws)-1) * 100

    # Best achievable
    best = max(mode_acc, trans_acc, cond_acc if pos > 0 else 0)
    exploit = best / random_acc

    print(f"  Col {pos+1}: random={random_acc:.1f}% | mode={mode_acc:.1f}% | "
          f"trans={trans_acc:.1f}% | cond={cond_acc:.1f}% | "
          f"BEST={best:.1f}% ({exploit:.1f}x random)")

# ================================================================
# PART 6: BACKTEST — ULTRA PREDICTOR
# Combines: transition + bigram + spacing + cycle-aware + hot-zone
# ================================================================
print(f"\n{'='*90}")
print(f" PART 6: ULTRA PREDICTOR BACKTEST")
print(f" Combining ALL discovered patterns into scored candidates")
print(f"{'='*90}\n")

def ultra_candidates(history, pos, K=10):
    """Generate Top-K candidates using ALL pattern sources."""
    scores = Counter()
    n = len(history)
    vals = [h[pos] for h in history]
    lv = vals[-1]

    # === SOURCE 1: Multi-window mode ===
    for w in [10, 20, 30, 50, 80, 100]:
        seg = vals[-min(w,n):]
        for v, c in Counter(seg).most_common(5):
            scores[v] += c / len(seg) * 2

    # === SOURCE 2: Markov transition ===
    trans = defaultdict(Counter)
    for i in range(n-1):
        trans[vals[i]][vals[i+1]] += 1
    if lv in trans:
        t = sum(trans[lv].values())
        for v, c in trans[lv].most_common(8):
            scores[v] += c / t * 5  # Higher weight - proven useful

    # === SOURCE 3: Bigram ===
    if n >= 3:
        bg = defaultdict(Counter)
        for i in range(n-2):
            bg[(vals[i],vals[i+1])][vals[i+2]] += 1
        key = (vals[-2], vals[-1])
        if key in bg:
            t = sum(bg[key].values())
            for v, c in bg[key].most_common(5):
                scores[v] += c / t * 6

    # === SOURCE 4: Trigram ===
    if n >= 4:
        tg = defaultdict(Counter)
        for i in range(n-3):
            tg[(vals[i],vals[i+1],vals[i+2])][vals[i+3]] += 1
        key = (vals[-3], vals[-2], vals[-1])
        if key in tg:
            t = sum(tg[key].values())
            for v, c in tg[key].most_common(3):
                scores[v] += c / t * 8

    # === SOURCE 5: 4-gram ===
    if n >= 5:
        fg = defaultdict(Counter)
        for i in range(n-4):
            fg[tuple(vals[i:i+4])][vals[i+4]] += 1
        key = tuple(vals[-4:])
        if key in fg:
            t = sum(fg[key].values())
            for v, c in fg[key].most_common(3):
                scores[v] += c / t * 10

    # === SOURCE 6: WMA ±1,±2 ===
    for w in [10, 15, 20, 30]:
        seg = vals[-min(w,n):]
        weights = np.arange(1, len(seg)+1, dtype=float)
        wma = np.average(seg, weights=weights)
        for delta in range(-2, 3):
            v = int(round(wma)) + delta
            if 1 <= v <= MAX_NUM:
                weight = 2.5 if delta == 0 else (1.5 if abs(delta) == 1 else 0.5)
                scores[v] += weight

    # === SOURCE 7: EMA ===
    for alpha in [0.1, 0.2, 0.3, 0.5]:
        seg = vals[-30:]
        s = seg[0]
        for vv in seg[1:]:
            s = alpha * vv + (1-alpha) * s
        for delta in range(-1, 2):
            v = int(round(s)) + delta
            if 1 <= v <= MAX_NUM:
                scores[v] += 1.5

    # === SOURCE 8: Linear regression ===
    for w in [20, 30, 50]:
        seg = vals[-min(w,n):]
        x = np.arange(len(seg))
        c = np.polyfit(x, seg, 1)
        pred = c[0]*len(seg) + c[1]
        for delta in range(-1, 2):
            v = int(round(pred)) + delta
            if 1 <= v <= MAX_NUM:
                scores[v] += 1.5

    # === SOURCE 9: Spacing-based (inter-column) ===
    if pos > 0:
        spacings = [h[pos]-h[pos-1] for h in history[-50:]]
        avg_sp = np.mean(spacings)
        std_sp = np.std(spacings)
        prev_val = history[-1][pos-1]
        for delta in range(-2, 3):
            v = prev_val + int(round(avg_sp)) + delta
            if 1 <= v <= MAX_NUM:
                weight = 3.0 if delta == 0 else (2.0 if abs(delta)==1 else 0.8)
                scores[v] += weight

        # Conditional spacing (given prev col value)
        sp_cond = defaultdict(Counter)
        for h in history:
            sp_cond[h[pos-1]][h[pos]-h[pos-1]] += 1
        pv = history[-1][pos-1]
        if pv in sp_cond:
            t = sum(sp_cond[pv].values())
            for sp, c in sp_cond[pv].most_common(5):
                v = pv + sp
                if 1 <= v <= MAX_NUM:
                    scores[v] += c / t * 4

    # === SOURCE 10: Conditional on previous column value ===
    if pos > 0:
        cond = defaultdict(Counter)
        for h in history: cond[h[pos-1]][h[pos]] += 1
        pv = history[-1][pos-1]
        if pv in cond:
            t = sum(cond[pv].values())
            for v, c in cond[pv].most_common(6):
                scores[v] += c / t * 4

    # === SOURCE 11: Conditional on next column value ===
    if pos < 5:
        cond = defaultdict(Counter)
        for h in history: cond[h[pos+1]][h[pos]] += 1
        nv = history[-1][pos+1]
        if nv in cond:
            t = sum(cond[nv].values())
            for v, c in cond[nv].most_common(6):
                scores[v] += c / t * 4

    # === SOURCE 12: Cross-2 context ===
    if pos > 0 and pos < 5:
        cross = defaultdict(Counter)
        for i in range(n-1):
            key = (history[i][pos-1], history[i][pos+1])
            cross[key][history[i+1][pos]] += 1
        key = (history[-1][pos-1], history[-1][pos+1])
        if key in cross:
            t = sum(cross[key].values())
            for v, c in cross[key].most_common(3):
                scores[v] += c / t * 5

    # === SOURCE 13: Delta prediction ===
    for w in [20, 50, 100]:
        seg = vals[-min(w,n):]
        deltas = [seg[i+1]-seg[i] for i in range(len(seg)-1)]
        if deltas:
            for d, c in Counter(deltas).most_common(5):
                v = lv + d
                if 1 <= v <= MAX_NUM:
                    scores[v] += c / len(deltas) * 2.5

    # === SOURCE 14: Delta bigram ===
    seg = vals[-100:]
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
                    scores[v] += c / t * 5

    # === SOURCE 15: Gap/overdue ===
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
            scores[v] += min((cg/mg-1)*2, 4.0)
        # Due now
        if mg > 0 and abs(cg - mg) < mg * 0.3:
            scores[v] += 2.0

    # === SOURCE 16: KNN full draw ===
    last = history[-1]
    knn = Counter()
    for i in range(n-2):
        sim = sum(1 for j in range(6) if abs(history[i][j]-last[j]) <= 2)
        if sim >= 4 and i+1 < n:
            knn[history[i+1][pos]] += sim**2
    if knn:
        mx = max(knn.values())
        for v, s in knn.most_common(5):
            scores[v] += s / mx * 3

    # === SOURCE 17: KNN position-only ===
    knn_p = Counter()
    for i in range(n-1):
        if abs(vals[i] - lv) <= 1:
            knn_p[vals[i+1]] += 1
    if knn_p:
        mx = max(knn_p.values())
        for v, c in knn_p.most_common(5):
            scores[v] += c / mx * 3

    # === SOURCE 18: Modular arithmetic ===
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
                    scores[v] += c / len(recent_with_mod) * 2

    # === SOURCE 19: Direction-based ===
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
                scores[v] += c / mx * 2

    # === SOURCE 20: Streak analysis ===
    streak_val = lv
    streak_len = 1
    for i in range(n-2, -1, -1):
        if vals[i] == streak_val: streak_len += 1
        else: break
    if streak_len >= 2:
        scores[streak_val] += streak_len * 2

    # === SOURCE 21: Autocorrelation-based ===
    if n >= 60:
        v_arr = np.array(vals[-200:], dtype=float)
        v_arr_c = v_arr - v_arr.mean()
        std = v_arr_c.std()
        if std > 0:
            v_arr_c = v_arr_c / std
            for lag in range(2, min(20, len(v_arr_c)//3)):
                corr = np.mean(v_arr_c[:len(v_arr_c)-lag] * v_arr_c[lag:])
                if corr > 0.15 and lag <= n:
                    scores[vals[-lag]] += corr * 3

    # === SOURCE 22: Draw sum context ===
    sum_trans = defaultdict(Counter)
    for h in history:
        s = sum(h) // 10 * 10  # bucket
        sum_trans[s][h[pos]] += 1
    ls = sum(history[-1]) // 10 * 10
    for ds in [-10, 0, 10]:
        key = ls + ds
        if key in sum_trans:
            t = sum(sum_trans[key].values())
            for v, c in sum_trans[key].most_common(5):
                weight = 1.5 if ds == 0 else 0.5
                scores[v] += c / t * weight

    # === SOURCE 23: Draw range context ===
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
                scores[v] += c / t * weight

    # === SOURCE 24: Volatility regime ===
    std_recent = np.std(vals[-20:])
    if std_recent < 3:
        for v, c in Counter(vals[-20:]).most_common(3):
            scores[v] += 2.5
    else:
        trend = np.mean(vals[-5:]) - np.mean(vals[-20:])
        pred = np.mean(vals[-5:]) + trend * 0.5
        for delta in range(-2, 3):
            v = int(round(pred)) + delta
            if 1 <= v <= MAX_NUM:
                scores[v] += 1.0

    # === SOURCE 25: Digit-sum transition ===
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
                    scores[v] += c / len(recent) * 1.5

    # === SOURCE 26: Last ±1,±2 ===
    for delta in range(-2, 3):
        v = lv + delta
        if 1 <= v <= MAX_NUM:
            weight = 2.0 if delta == 0 else (1.5 if abs(delta)==1 else 0.5)
            scores[v] += weight

    # === SOURCE 27: Median ±1 ===
    for w in [10, 20, 30]:
        seg = vals[-min(w,n):]
        med = np.median(seg)
        for delta in range(-1, 2):
            v = int(round(med)) + delta
            if 1 <= v <= MAX_NUM:
                scores[v] += 1.0

    return [v for v, _ in scores.most_common(K)]


# ================================================================
# BACKTEST ULTRA PREDICTOR
# ================================================================
START = 100
TESTED = total - START - 1
N_SETS = 30

# Test K configurations
K_CONFIGS = [
    ("K5_adaptive",  [3,6,7,7,6,3]),
    ("K6_adaptive",  [4,7,8,8,7,4]),
    ("K7_adaptive",  [4,8,9,9,8,4]),
    ("K8_extended",  [5,9,10,10,9,5]),
]

results = {}
for name, ks in K_CONFIGS:
    mc = 1
    for k in ks: mc *= k
    results[name] = {
        'ks': ks, 'max_combos': mc,
        'h6': 0, 'h5': 0, 'h4': 0, 'h3': 0,
        'pos_hit': [0]*6,
        'valid_combos': [],
        'best_match_hist': [],
    }

def score_combo_ultra(combo, history):
    """Score a combo by sum of individual scores + pair bonus."""
    score = 0
    last_draw = set(history[-1])
    n = len(history)

    # Frequency
    freq = Counter()
    for h in history[-50:]:
        for num in h: freq[num] += 1
    for num in combo: score += freq.get(num, 0) * 0.1

    # Transition
    follow = defaultdict(Counter)
    for i in range(n-1):
        for p in history[i]:
            for nx in history[i+1]:
                follow[p][nx] += 1
    for num in combo:
        for p in last_draw:
            score += follow[p].get(num, 0) * 0.02

    # Co-occurrence
    from itertools import combinations
    pair_freq = Counter()
    for h in history[-200:]:
        for p in combinations(sorted(h), 2):
            pair_freq[p] += 1
    for p in combinations(sorted(combo), 2):
        score += pair_freq.get(p, 0) * 0.05

    # KNN
    last = history[-1]
    for i in range(n-2):
        sim = len(set(history[i]) & set(last))
        if sim >= 3:
            match = len(set(history[i+1]) & set(combo))
            score += match * sim * 0.1

    # Spacing regularity
    spacings = [combo[i+1]-combo[i] for i in range(5)]
    score += (combo[-1]-combo[0]) * 0.1  # Range
    # Penalize extreme spacings
    for sp in spacings:
        if sp > 15: score -= 0.5
        if sp < 2: score -= 0.3

    return score


from itertools import product as iterproduct, combinations

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
            cands = ultra_candidates(history, pos, ks[pos])
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
            scored = [(c, score_combo_ultra(c, history)) for c in valid]
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
    if done % 50 == 0:
        el = time.time() - t0
        eta = el / done * (TESTED - done) / 60
        print(f"  [{done}/{TESTED}] {el:.0f}s ETA={eta:.1f}m")
        for name, _ in K_CONFIGS[:2]:
            r = results[name]
            avg_vc = np.mean(r['valid_combos']) if r['valid_combos'] else 0
            pos_cov = [r['pos_hit'][p]/(done)*100 for p in range(6)]
            print(f"    {name}: 6/6={r['h6']} 5/6={r['h5']} "
                  f"4/6={r['h4']} 3/6={r['h3']} "
                  f"pos=[{' '.join(f'{c:.0f}' for c in pos_cov)}]%")
        sys.stdout.flush()

el = time.time() - t0
print(f"\n  Completed in {el:.0f}s ({el/60:.1f}m)\n")

# ================================================================
# RESULTS
# ================================================================
print(f"{'='*90}")
print(f" ULTRA PREDICTOR RESULTS — {N_SETS} SETS × {TESTED} DRAWS")
print(f"{'='*90}\n")

print(f"  {'Config':<16} {'K per col':<22} {'MaxCb':<8} {'AvgValid':<10} "
      f"{'6/6':<8} {'>=5/6':<8} {'>=4/6':<8} {'>=3/6':<8}")
print(f"  {'-'*96}")

for name, ks in K_CONFIGS:
    r = results[name]
    avg_vc = np.mean(r['valid_combos'])
    print(f"  {name:<16} {str(ks):<22} {r['max_combos']:<8} {avg_vc:<10.0f} "
          f"{r['h6']/TESTED*100:<8.3f} {r['h5']/TESTED*100:<8.2f} "
          f"{r['h4']/TESTED*100:<8.2f} {r['h3']/TESTED*100:<8.2f}")

print(f"\n  --- Per-Position Coverage ---\n")
print(f"  {'Config':<16}", end='')
for p in range(6):
    print(f" {'Col'+str(p+1):<8}", end='')
print(f" {'Joint':<8}")
print(f"  {'-'*75}")

for name, ks in K_CONFIGS:
    r = results[name]
    print(f"  {name:<16}", end='')
    joint = 1.0
    for p in range(6):
        cov = r['pos_hit'][p] / TESTED * 100
        joint *= cov / 100
        print(f" {cov:<8.1f}", end='')
    print(f" {joint*100:<8.2f}")

# Random comparison
from math import comb
random_rate = N_SETS / comb(45, 6) * 100
print(f"\n  Random baseline ({N_SETS} sets): {random_rate:.7f}%")
for name, ks in K_CONFIGS:
    r = results[name]
    rate = r['h6']/TESTED*100
    if rate > 0:
        print(f"  {name}: {rate:.4f}% = {rate/random_rate:.0f}x random")

# Best match distribution
print(f"\n  --- Best Match Distribution ---\n")
for name, ks in K_CONFIGS:
    r = results[name]
    bm = r['best_match_hist']
    if bm:
        avg = np.mean(bm)
        print(f"  {name:<16}: avg={avg:.2f}/6, "
              f"0={sum(1 for x in bm if x==0)/len(bm)*100:.1f}% "
              f"≥2={sum(1 for x in bm if x>=2)/len(bm)*100:.1f}% "
              f"≥3={sum(1 for x in bm if x>=3)/len(bm)*100:.1f}% "
              f"≥4={sum(1 for x in bm if x>=4)/len(bm)*100:.1f}%")

print(f"\n{'='*90}")
print(f" KEY INSIGHT: Per-position coverage is the HARD CEILING.")
print(f" The ULTRA predictor adds spacing, cross-column, cycle-aware")
print(f" and delta-bigram sources to squeeze more signal per column.")
print(f"{'='*90}")
