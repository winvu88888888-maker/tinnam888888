"""
DEEP COLUMN-SPECIFIC RANGE ANALYSIS
=====================================
Each sorted position has a NARROW practical range:
  Pos1 (smallest) → ~1-10
  Pos2 → ~5-20
  Pos3 → ~10-30
  Pos4 → ~20-35
  Pos5 → ~25-42
  Pos6 (largest) → ~35-45

Strategy: For EACH column, analyze ONLY its practical range.
Find deep patterns: value sequences, cycles, spacing rules,
hot/cold streaks, transition chains within the narrow range.

Then build COLUMN-SPECIFIC predictors that exploit each column's
unique statistical fingerprint.
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
print(f" DEEP COLUMN-SPECIFIC RANGE ANALYSIS")
print(f" Focus: narrow range of each position, patterns within that range")
print(f"{'='*90}\n")

# ================================================================
# PHASE 1: MAP EACH COLUMN'S ACTUAL RANGE
# ================================================================
print(f"{'='*90}")
print(f" PHASE 1: COLUMN RANGES & VALUE MAPS")
print(f"{'='*90}\n")

for pos in range(6):
    vals = [sd[pos] for sd in sorted_draws]
    vc = Counter(vals)
    n = len(vals)

    # Full distribution
    print(f"  ── COLUMN {pos+1} ({'SMALLEST' if pos==0 else 'LARGEST' if pos==5 else f'#{pos+1}'}) ──")
    print(f"  Range: {min(vals)}-{max(vals)}")
    print(f"  Mean: {np.mean(vals):.1f} | Std: {np.std(vals):.1f} | Median: {np.median(vals):.0f}")

    # P5/P95 practical range
    p5, p95 = np.percentile(vals, 5), np.percentile(vals, 95)
    p10, p90 = np.percentile(vals, 10), np.percentile(vals, 90)
    print(f"  P5-P95 range: {p5:.0f}-{p95:.0f} ({(p95-p5):.0f} span)")
    print(f"  P10-P90 range: {p10:.0f}-{p90:.0f} ({(p90-p10):.0f} span)")

    # Count values in practical range
    in_range = sum(1 for v in vals if p5 <= v <= p95)
    print(f"  Values in P5-P95: {in_range}/{n} ({in_range/n*100:.1f}%)")

    # Full value distribution (sorted by value)
    all_vals = sorted(vc.keys())
    print(f"\n  Value Distribution:")
    row = "    "
    for v in all_vals:
        pct = vc[v]/n*100
        bar = '#' * max(1, int(pct/2))
        row_item = f"{v:>2}:{vc[v]:>4}({pct:>5.1f}%) {bar}"
        print(f"    {row_item}")

    # Top-10 most common
    print(f"\n  Top-10 most common:")
    cum = 0
    for v, c in vc.most_common(10):
        cum += c
        print(f"    {v:>2}: {c:>4} ({c/n*100:.1f}%) cumulative={cum/n*100:.1f}%")

    print()

# ================================================================
# PHASE 2: DEEP PATTERN ANALYSIS PER COLUMN
# ================================================================
print(f"\n{'='*90}")
print(f" PHASE 2: DEEP PATTERNS WITHIN EACH COLUMN")
print(f"{'='*90}\n")

for pos in range(6):
    vals = [sd[pos] for sd in sorted_draws]
    n = len(vals)
    vc = Counter(vals)

    print(f"\n{'─'*90}")
    print(f" COLUMN {pos+1} — Deep Pattern Analysis")
    print(f"{'─'*90}\n")

    # --- A. Transition patterns: what comes after what ---
    print(f"  A. TRANSITION PATTERNS (value→next_value at this position)")
    trans = defaultdict(Counter)
    for i in range(n-1):
        trans[vals[i]][vals[i+1]] += 1

    # For top-5 values, show their top transitions
    for v, c in vc.most_common(8):
        if v in trans:
            t = sum(trans[v].values())
            top3 = trans[v].most_common(3)
            top3_str = ", ".join(f"{nv}({nc}/{t}={nc/t*100:.0f}%)" for nv, nc in top3)
            print(f"    After {v:>2} ({c} times): → {top3_str}")

    # --- B. Repeat rate: how often does same value repeat? ---
    print(f"\n  B. REPEAT RATE")
    repeats = sum(1 for i in range(1,n) if vals[i]==vals[i-1])
    print(f"    Same value repeats: {repeats}/{n-1} ({repeats/(n-1)*100:.1f}%)")

    # Repeat streaks
    max_streak = 1; cur_streak = 1; streak_vals = defaultdict(int)
    for i in range(1, n):
        if vals[i] == vals[i-1]:
            cur_streak += 1
            max_streak = max(max_streak, cur_streak)
        else:
            if cur_streak >= 2:
                streak_vals[vals[i-1]] += 1
            cur_streak = 1
    if cur_streak >= 2:
        streak_vals[vals[-1]] += 1
    print(f"    Max repeat streak: {max_streak}")
    if streak_vals:
        top_streakers = sorted(streak_vals.items(), key=lambda x: -x[1])[:5]
        print(f"    Values that streak most: {', '.join(f'{v}({c})' for v,c in top_streakers)}")

    # --- C. Delta patterns: change between draws ---
    print(f"\n  C. DELTA PATTERNS (change from draw to draw)")
    deltas = [vals[i+1]-vals[i] for i in range(n-1)]
    dc = Counter(deltas)
    print(f"    Mean delta: {np.mean(deltas):.2f} | Std: {np.std(deltas):.2f}")
    print(f"    Top-10 deltas:")
    for d, c in dc.most_common(10):
        print(f"      Δ={d:>+3}: {c:>4} ({c/len(deltas)*100:.1f}%)")

    # --- D. Delta-delta (acceleration) ---
    print(f"\n  D. DELTA-DELTA (acceleration patterns)")
    if len(deltas) >= 2:
        dd = [deltas[i+1]-deltas[i] for i in range(len(deltas)-1)]
        ddc = Counter(dd)
        for d, c in ddc.most_common(5):
            print(f"      ΔΔ={d:>+3}: {c:>4} ({c/len(dd)*100:.1f}%)")

    # --- E. Cycle detection: does value X appear every N draws? ---
    print(f"\n  E. CYCLE ANALYSIS (how often does each value reappear?)")
    gap_data = defaultdict(list)
    last_seen = {}
    for i, v in enumerate(vals):
        if v in last_seen:
            gap_data[v].append(i - last_seen[v])
        last_seen[v] = i

    for v, c in vc.most_common(8):
        gaps = gap_data.get(v, [])
        if len(gaps) >= 3:
            mg = np.mean(gaps)
            sg = np.std(gaps)
            min_g, max_g = min(gaps), max(gaps)
            # Check if periodic (low std relative to mean)
            periodicity = 1 - (sg / mg) if mg > 0 else 0
            periodic_str = " ★PERIODIC" if periodicity > 0.5 else ""
            print(f"    Value {v:>2}: mean_gap={mg:.1f}, std={sg:.1f}, "
                  f"range=[{min_g}-{max_g}], period_score={periodicity:.2f}{periodic_str}")

    # --- F. Direction patterns: up/down/same sequences ---
    print(f"\n  F. DIRECTION SEQUENCES (up↑/down↓/same=)")
    dirs = []
    for i in range(1, n):
        if vals[i] > vals[i-1]: dirs.append('U')
        elif vals[i] < vals[i-1]: dirs.append('D')
        else: dirs.append('S')
    dir_c = Counter(dirs)
    print(f"    Up: {dir_c.get('U',0)} ({dir_c.get('U',0)/len(dirs)*100:.1f}%)")
    print(f"    Down: {dir_c.get('D',0)} ({dir_c.get('D',0)/len(dirs)*100:.1f}%)")
    print(f"    Same: {dir_c.get('S',0)} ({dir_c.get('S',0)/len(dirs)*100:.1f}%)")

    # Direction bigram
    dir_bg = defaultdict(Counter)
    for i in range(len(dirs)-1):
        dir_bg[dirs[i]][dirs[i+1]] += 1
    print(f"    Direction transitions:")
    for d in ['U', 'D', 'S']:
        if d in dir_bg:
            t = sum(dir_bg[d].values())
            trans_str = ", ".join(f"{nd}({nc/t*100:.0f}%)" for nd, nc in dir_bg[d].most_common(3))
            print(f"      After {d}: → {trans_str}")

    # --- G. Spacing to next position ---
    if pos < 5:
        print(f"\n  G. SPACING TO COLUMN {pos+2}")
        spacings = [sd[pos+1] - sd[pos] for sd in sorted_draws]
        sc = Counter(spacings)
        print(f"    Mean spacing: {np.mean(spacings):.1f} | Std: {np.std(spacings):.1f}")
        print(f"    Top-5 spacings:")
        for s, c in sc.most_common(5):
            print(f"      Gap={s:>2}: {c:>4} ({c/n*100:.1f}%)")

    # --- H. Hot/Cold streaks ---
    print(f"\n  H. HOT VALUES (recent 50 draws) vs COLD")
    recent = vals[-50:]
    older = vals[-200:-50]
    rc = Counter(recent)
    oc = Counter(older)
    hot = []
    cold = []
    for v in set(list(rc.keys()) + list(oc.keys())):
        r_pct = rc.get(v, 0) / 50 * 100
        o_pct = oc.get(v, 0) / 150 * 100
        if r_pct > o_pct * 1.5 and rc.get(v, 0) >= 3:
            hot.append((v, r_pct, o_pct))
        elif o_pct > r_pct * 1.5 and oc.get(v, 0) >= 5:
            cold.append((v, r_pct, o_pct))
    hot.sort(key=lambda x: -x[1])
    cold.sort(key=lambda x: -x[2])
    if hot:
        print(f"    HOT: {', '.join(f'{v}({r:.0f}% vs {o:.0f}%)' for v,r,o in hot[:5])}")
    if cold:
        print(f"    COLD: {', '.join(f'{v}({r:.0f}% vs {o:.0f}%)' for v,r,o in cold[:5])}")

    # --- I. N-gram patterns (sequence detection) ---
    print(f"\n  I. SEQUENCE PATTERNS (most common 2-value and 3-value sequences)")
    bi_seqs = Counter()
    tri_seqs = Counter()
    for i in range(n-1):
        bi_seqs[(vals[i], vals[i+1])] += 1
    for i in range(n-2):
        tri_seqs[(vals[i], vals[i+1], vals[i+2])] += 1

    print(f"    Top-5 bigrams:")
    for seq, c in bi_seqs.most_common(5):
        print(f"      {seq[0]:>2}→{seq[1]:>2}: {c:>3} ({c/(n-1)*100:.1f}%)")
    print(f"    Top-5 trigrams:")
    for seq, c in tri_seqs.most_common(5):
        print(f"      {seq[0]:>2}→{seq[1]:>2}→{seq[2]:>2}: {c:>2} ({c/(n-2)*100:.1f}%)")

    # --- J. Conditional on neighboring columns ---
    print(f"\n  J. CONDITIONAL ON NEIGHBORING COLUMNS")
    if pos > 0:
        cond_prev = defaultdict(Counter)
        for sd in sorted_draws:
            cond_prev[sd[pos-1]][sd[pos]] += 1
        # Show top conditionals
        prev_vals = Counter(sd[pos-1] for sd in sorted_draws).most_common(5)
        print(f"    Given Col{pos} (prev column):")
        for pv, pc in prev_vals:
            t = sum(cond_prev[pv].values())
            top2 = cond_prev[pv].most_common(3)
            top_str = ", ".join(f"{v}({c/t*100:.0f}%)" for v,c in top2)
            print(f"      Col{pos}={pv:>2} → Col{pos+1}={top_str}")

    if pos < 5:
        cond_next = defaultdict(Counter)
        for sd in sorted_draws:
            cond_next[sd[pos+1]][sd[pos]] += 1
        next_vals = Counter(sd[pos+1] for sd in sorted_draws).most_common(5)
        print(f"    Given Col{pos+2} (next column):")
        for nv, nc in next_vals:
            t = sum(cond_next[nv].values())
            top2 = cond_next[nv].most_common(3)
            top_str = ", ".join(f"{v}({c/t*100:.0f}%)" for v,c in top2)
            print(f"      Col{pos+2}={nv:>2} → Col{pos+1}={top_str}")

# ================================================================
# PHASE 3: BACKTEST COLUMN-SPECIFIC METHODS
# Within each column's range, test specialized predictions
# ================================================================
print(f"\n{'='*90}")
print(f" PHASE 3: COLUMN-SPECIFIC PREDICTOR BACKTEST")
print(f" Using range-aware methods within each column's practical range")
print(f"{'='*90}\n")

START = 100
TESTED = total - START - 1

# For each column, define its practical range
def get_column_range(history, pos):
    """Get P5-P95 range for this column from history."""
    vals = [h[pos] for h in history]
    return int(np.percentile(vals, 3)), int(np.percentile(vals, 97))

# Column-specific methods that exploit narrow range
def predict_column(history, pos):
    """Return dict of {method_name: predicted_value} for this column."""
    vals = [h[pos] for h in history]
    n = len(vals)
    lo, hi = get_column_range(history, pos)
    range_vals = [v for v in vals if lo <= v <= hi]  # values in range

    predictions = {}

    # M1: Mode in range (recent windows)
    for w in [10, 20, 30, 50]:
        recent = [v for v in vals[-w:] if lo <= v <= hi]
        if recent:
            predictions[f'range_mode_{w}'] = Counter(recent).most_common(1)[0][0]

    # M2: Transition within range
    trans = defaultdict(Counter)
    for i in range(n-1):
        trans[vals[i]][vals[i+1]] += 1
    lv = vals[-1]
    if lv in trans:
        # Only consider in-range transitions
        in_range_trans = {v: c for v, c in trans[lv].items() if lo <= v <= hi}
        if in_range_trans:
            predictions['range_transition'] = max(in_range_trans, key=in_range_trans.get)

    # M3: Bigram within range
    if n >= 3:
        bg = defaultdict(Counter)
        for i in range(n-2):
            bg[(vals[i], vals[i+1])][vals[i+2]] += 1
        key = (vals[-2], vals[-1])
        if key in bg:
            in_range = {v: c for v, c in bg[key].items() if lo <= v <= hi}
            if in_range:
                predictions['range_bigram'] = max(in_range, key=in_range.get)

    # M4: WMA within range (clamp to range)
    for w in [10, 15, 20]:
        seg = vals[-w:]
        weights = np.arange(1, len(seg)+1, dtype=float)
        wma = np.average(seg, weights=weights)
        predictions[f'range_wma_{w}'] = max(lo, min(hi, int(round(wma))))

    # M5: Most common delta → predict → clamp to range
    seg = vals[-50:]
    deltas = [seg[i+1]-seg[i] for i in range(len(seg)-1)]
    if deltas:
        d = Counter(deltas).most_common(1)[0][0]
        pred = vals[-1] + d
        predictions['range_delta_mode'] = max(lo, min(hi, pred))

    # M6: Delta transition (what delta usually follows current delta?)
    if len(vals) >= 3:
        all_deltas = [vals[i+1]-vals[i] for i in range(n-1)]
        if len(all_deltas) >= 2:
            dt = defaultdict(Counter)
            for i in range(len(all_deltas)-1):
                dt[all_deltas[i]][all_deltas[i+1]] += 1
            ld = all_deltas[-1]
            if ld in dt:
                next_d = dt[ld].most_common(1)[0][0]
                pred = vals[-1] + next_d
                predictions['range_delta_trans'] = max(lo, min(hi, pred))

    # M7: Conditional on previous column
    if pos > 0:
        cond = defaultdict(Counter)
        for h in history:
            cond[h[pos-1]][h[pos]] += 1
        pv = history[-1][pos-1]
        if pv in cond:
            in_range = {v: c for v, c in cond[pv].items() if lo <= v <= hi}
            if in_range:
                predictions['cond_prev_col'] = max(in_range, key=in_range.get)

    # M8: Conditional on next column
    if pos < 5:
        cond = defaultdict(Counter)
        for h in history:
            cond[h[pos+1]][h[pos]] += 1
        nv = history[-1][pos+1]
        if nv in cond:
            in_range = {v: c for v, c in cond[nv].items() if lo <= v <= hi}
            if in_range:
                predictions['cond_next_col'] = max(in_range, key=in_range.get)

    # M9: Repeat last (if still in range)
    if lo <= vals[-1] <= hi:
        predictions['repeat_last'] = vals[-1]

    # M10: Repeat last-1
    if n >= 2 and lo <= vals[-2] <= hi:
        predictions['repeat_last2'] = vals[-2]

    # M11: Spacing-based (predict based on prev column + typical spacing)
    if pos > 0:
        spacings = [h[pos]-h[pos-1] for h in history]
        for w in [20, 50]:
            seg = spacings[-w:]
            avg_sp = np.mean(seg)
            pred = history[-1][pos-1] + int(round(avg_sp))
            predictions[f'spacing_{w}'] = max(lo, min(hi, pred))

        # Spacing transition
        sp_trans = defaultdict(Counter)
        for i in range(len(spacings)-1):
            sp_trans[spacings[i]][spacings[i+1]] += 1
        ls = spacings[-1]
        if ls in sp_trans:
            next_sp = sp_trans[ls].most_common(1)[0][0]
            pred = history[-1][pos-1] + next_sp
            predictions['spacing_trans'] = max(lo, min(hi, pred))

    # M12: Direction-based
    if n >= 3:
        d = 1 if vals[-1]>vals[-2] else (-1 if vals[-1]<vals[-2] else 0)
        dir_next = Counter()
        for i in range(1, n-1):
            di = 1 if vals[i]>vals[i-1] else (-1 if vals[i]<vals[i-1] else 0)
            if di == d and lo <= vals[i+1] <= hi:
                dir_next[vals[i+1]] += 1
        if dir_next:
            predictions['direction'] = dir_next.most_common(1)[0][0]

    # M13: Cycle-based (most overdue value in range)
    last_seen_p = {}
    for i, v in enumerate(vals):
        last_seen_p[v] = i
    best_v, best_ratio = vals[-1], 0
    gap_data = defaultdict(list)
    prev_idx = {}
    for i, v in enumerate(vals):
        if v in prev_idx:
            gap_data[v].append(i - prev_idx[v])
        prev_idx[v] = i
    for v in set(vals):
        if not (lo <= v <= hi): continue
        if len(gap_data[v]) < 3: continue
        mg = np.mean(gap_data[v])
        cg = n - last_seen_p.get(v, 0)
        if mg > 0:
            ratio = cg / mg
            if ratio > best_ratio:
                best_ratio = ratio
                best_v = v
    predictions['overdue_in_range'] = best_v

    # M14: MA of range-only values
    rv = [v for v in vals[-30:] if lo <= v <= hi]
    if rv:
        predictions['range_ma'] = int(round(np.mean(rv)))

    # M15: Median of recent range values
    if rv:
        predictions['range_median'] = int(round(np.median(rv)))

    # M16: Streak-aware
    streak_val = vals[-1]
    streak_len = 1
    for i in range(n-2, -1, -1):
        if vals[i] == streak_val: streak_len += 1
        else: break
    if streak_len >= 2 and lo <= streak_val <= hi:
        predictions['streak_continue'] = streak_val

    return predictions


# Run backtest
hits = defaultdict(lambda: [0]*6)  # hits[method][pos]
total_preds = defaultdict(lambda: [0]*6)

# Also track: best method and combined accuracy
top3_hits = [0]*6  # if actual is in top-3 predictions (by frequency)

t0 = time.time()
for idx in range(START, total-1):
    history = sorted_draws[:idx+1]
    actual = sorted_draws[idx+1]

    for pos in range(6):
        preds = predict_column(history, pos)
        actual_v = actual[pos]

        # Check each method
        for mname, pred_v in preds.items():
            total_preds[mname][pos] += 1
            if pred_v == actual_v:
                hits[mname][pos] += 1

        # Check if actual is in top-3 unique predictions
        pred_counter = Counter(preds.values())
        top3_vals = [v for v, _ in pred_counter.most_common(3)]
        if actual_v in top3_vals:
            top3_hits[pos] += 1

    done = idx - START + 1
    if done % 200 == 0:
        el = time.time() - t0
        eta = el / done * (TESTED - done) / 60
        print(f"  [{done}/{TESTED}] {el:.0f}s ETA={eta:.1f}m")
        sys.stdout.flush()

el = time.time() - t0
print(f"\n  Completed in {el:.0f}s ({el/60:.1f}m)\n")

# ================================================================
# RESULTS PER COLUMN
# ================================================================
for pos in range(6):
    vals = [sd[pos] for sd in sorted_draws]
    lo, hi = get_column_range(sorted_draws, pos)

    print(f"\n{'='*90}")
    print(f" COLUMN {pos+1} — Range [{lo}-{hi}] — {len(set(vals))} unique values")
    print(f"{'='*90}\n")

    # Rank methods
    ranked = []
    for mname in sorted(hits.keys()):
        t = total_preds[mname][pos]
        h = hits[mname][pos]
        if t > 0:
            acc = h / t * 100
            ranked.append((mname, acc, h, t))

    ranked.sort(key=lambda x: -x[1])

    print(f"  {'Rank':<5} {'Method':<22} {'Exact%':<10} {'Hits':<8} {'Total':<8}")
    print(f"  {'-'*55}")
    for i, (mname, acc, h, t) in enumerate(ranked[:15]):
        marker = " ◄" if i == 0 else ""
        print(f"  {i+1:<5} {mname:<22} {acc:<10.2f} {h:<8} {t:<8}{marker}")

    # Top-3 vote accuracy
    t3_acc = top3_hits[pos] / TESTED * 100
    print(f"\n  Top-3 vote coverage: {top3_hits[pos]}/{TESTED} = {t3_acc:.2f}%")

# ================================================================
# GRAND SUMMARY
# ================================================================
print(f"\n{'='*90}")
print(f" GRAND SUMMARY — Best Range-Aware Method Per Column")
print(f"{'='*90}\n")

joint = 1.0
for pos in range(6):
    best_name = None
    best_acc = 0
    for mname in hits:
        t = total_preds[mname][pos]
        if t > 0:
            acc = hits[mname][pos] / t * 100
            if acc > best_acc:
                best_acc = acc
                best_name = mname
    joint *= best_acc / 100
    lo, hi = get_column_range(sorted_draws, pos)
    print(f"  Col {pos+1} [{lo:>2}-{hi:>2}]: {best_name:<22} = {best_acc:.2f}%")

print(f"\n  JOINT 6/6 = {joint*100:.6f}%")
print(f"  Top-3 vote coverage = " +
      " × ".join(f"{top3_hits[p]/TESTED*100:.1f}%" for p in range(6)))

joint_top3 = 1.0
for p in range(6):
    joint_top3 *= top3_hits[p] / TESTED
print(f"  Joint top-3 coverage = {joint_top3*100:.4f}%")

print(f"\n{'='*90}")
