"""
ULTRA-FAST PER-COLUMN COVERAGE BACKTEST
========================================
Focus ONLY on: Can each column's Top-K candidates contain the actual value?
Joint coverage = product of per-column coverage = the HARD CEILING for 6/6.
No combo enumeration. No combo scoring. Just pure per-position accuracy.

Also includes Parts 4-5: Inter-column analysis + Exploitability.
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

# ================================================================
# PART 4: INTER-COLUMN DEPENDENCIES
# ================================================================
print(f"\n{'='*90}")
print(f" PART 4: INTER-COLUMN DEPENDENCIES")
print(f"{'='*90}\n")

for gap_pair in range(5):
    spacings = [sd[gap_pair+1] - sd[gap_pair] for sd in sorted_draws]
    sc = Counter(spacings)
    top5 = sc.most_common(5)
    print(f"  Col{gap_pair+1}->Col{gap_pair+2}: "
          f"mean={np.mean(spacings):.1f} std={np.std(spacings):.1f} "
          f"med={np.median(spacings):.0f} [{min(spacings)},{max(spacings)}]")
    print(f"    Top: {', '.join(f'{s}({c/len(spacings)*100:.0f}%)' for s,c in top5)}")

print()

# Correlation matrix
cols = np.array([[sd[p] for sd in sorted_draws] for p in range(6)], dtype=float)
corr = np.corrcoef(cols)
print(f"  --- Correlation Matrix ---")
print(f"    {'':>5}", end='')
for p in range(6): print(f" C{p+1:>2} ", end='')
print()
for p1 in range(6):
    print(f"    C{p1+1}", end='')
    for p2 in range(6):
        print(f" {corr[p1,p2]:>4.2f}", end='')
    print()

# Delta correlation
print(f"\n  --- Delta Correlation ---")
deltas = np.diff(cols, axis=1)
dcorr = np.corrcoef(deltas)
print(f"    {'':>5}", end='')
for p in range(6): print(f" dC{p+1:>1} ", end='')
print()
for p1 in range(6):
    print(f"    dC{p1+1}", end='')
    for p2 in range(6):
        print(f" {dcorr[p1,p2]:>4.2f}", end='')
    print()

# Sum & Range
print()
sums = [sum(sd) for sd in sorted_draws]
ranges = [sd[5]-sd[0] for sd in sorted_draws]
print(f"  Sum:   mean={np.mean(sums):.1f} std={np.std(sums):.1f}")
print(f"  Range: mean={np.mean(ranges):.1f} std={np.std(ranges):.1f}")

# ================================================================
# PART 5: EXPLOITABILITY
# ================================================================
print(f"\n{'='*90}")
print(f" PART 5: EXPLOITABILITY PER COLUMN")
print(f"{'='*90}\n")

for pos in range(6):
    vals = [sd[pos] for sd in sorted_draws]
    n = len(vals)
    vc = Counter(vals)
    n_unique = len(vc)
    random_acc = 1.0 / n_unique * 100
    mode_acc = vc.most_common(1)[0][1] / n * 100

    trans = defaultdict(Counter)
    for i in range(n-1):
        trans[vals[i]][vals[i+1]] += 1
    t_correct = sum(1 for i in range(1, n) if vals[i-1] in trans and
                    trans[vals[i-1]].most_common(1)[0][0] == vals[i])
    trans_acc = t_correct / (n-1) * 100

    cond_acc = 0
    if pos > 0:
        cond = defaultdict(Counter)
        for sd in sorted_draws:
            cond[sd[pos-1]][sd[pos]] += 1
        c_correct = sum(1 for i in range(1, len(sorted_draws))
                       if sorted_draws[i][pos-1] in cond and
                       cond[sorted_draws[i][pos-1]].most_common(1)[0][0] == sorted_draws[i][pos])
        cond_acc = c_correct / (len(sorted_draws)-1) * 100

    best = max(mode_acc, trans_acc, cond_acc)
    print(f"  Col{pos+1}: random={random_acc:.1f}% mode={mode_acc:.1f}% "
          f"trans={trans_acc:.1f}% cond={cond_acc:.1f}% "
          f"BEST={best:.1f}% ({best/random_acc:.1f}x)")

# ================================================================
# PART 6: FAST BACKTEST — PER-COLUMN COVERAGE ONLY
# ================================================================
print(f"\n{'='*90}")
print(f" PART 6: FAST ULTRA BACKTEST — Per-Column Coverage")
print(f" (Joint coverage = HARD CEILING for 6/6)")
print(f"{'='*90}\n")

def fast_candidates_v2(history, pos, K=10):
    """Optimized 15-source candidate generator."""
    scores = Counter()
    n = len(history)
    vals = [h[pos] for h in history]
    lv = vals[-1]

    # S1: Multi-window mode
    for w in [10, 20, 50, 100]:
        ww = min(w,n)
        seg = vals[-ww:]
        for v, c in Counter(seg).most_common(4):
            scores[v] += c / len(seg) * 2

    # S2: Transition
    trans = defaultdict(Counter)
    for i in range(n-1):
        trans[vals[i]][vals[i+1]] += 1
    if lv in trans:
        t = sum(trans[lv].values())
        for v, c in trans[lv].most_common(6):
            scores[v] += c / t * 5

    # S3: Bigram
    if n >= 3:
        bg = defaultdict(Counter)
        for i in range(n-2):
            bg[(vals[i],vals[i+1])][vals[i+2]] += 1
        key = (vals[-2], vals[-1])
        if key in bg:
            t = sum(bg[key].values())
            for v, c in bg[key].most_common(4):
                scores[v] += c / t * 6

    # S4: WMA
    for w in [15, 20]:
        ww = min(w,n)
        seg = vals[-ww:]
        weights = np.arange(1, len(seg)+1, dtype=float)
        wma = np.average(seg, weights=weights)
        for delta in range(-2, 3):
            v = int(round(wma)) + delta
            if 1 <= v <= MAX_NUM:
                scores[v] += 2.5 if delta == 0 else (1.5 if abs(delta)==1 else 0.5)

    # S5: Spacing-based
    if pos > 0:
        sps = [h[pos]-h[pos-1] for h in history[-50:]]
        avg_sp = np.mean(sps)
        pv = history[-1][pos-1]
        for delta in range(-2, 3):
            v = pv + int(round(avg_sp)) + delta
            if 1 <= v <= MAX_NUM:
                scores[v] += 3.0 if delta == 0 else (2.0 if abs(delta)==1 else 0.8)

    # S6: Conditional on prev col
    if pos > 0:
        cond = defaultdict(Counter)
        for h in history: cond[h[pos-1]][h[pos]] += 1
        pv = history[-1][pos-1]
        if pv in cond:
            t = sum(cond[pv].values())
            for v, c in cond[pv].most_common(5):
                scores[v] += c / t * 4

    # S7: Conditional on next col
    if pos < 5:
        cond = defaultdict(Counter)
        for h in history: cond[h[pos+1]][h[pos]] += 1
        nv = history[-1][pos+1]
        if nv in cond:
            t = sum(cond[nv].values())
            for v, c in cond[nv].most_common(5):
                scores[v] += c / t * 4

    # S8: Delta
    seg = vals[-min(50,n):]
    dts = [seg[i+1]-seg[i] for i in range(len(seg)-1)]
    if dts:
        for d, c in Counter(dts).most_common(4):
            v = lv + d
            if 1 <= v <= MAX_NUM:
                scores[v] += c / len(dts) * 2.5

    # S9: Gap/overdue
    last_seen = {}
    for i, v in enumerate(vals): last_seen[v] = i
    gap_data = defaultdict(list)
    pi = {}
    for i, v in enumerate(vals):
        if v in pi: gap_data[v].append(i - pi[v])
        pi[v] = i
    for v in set(vals):
        if len(gap_data[v]) < 3: continue
        mg = np.mean(gap_data[v])
        cg = n - last_seen.get(v, 0)
        if mg > 0 and cg > mg * 1.2:
            scores[v] += min((cg/mg-1)*2, 4.0)
        if mg > 0 and abs(cg - mg) < mg * 0.3:
            scores[v] += 2.0

    # S10: Last ±2
    for delta in range(-2, 3):
        v = lv + delta
        if 1 <= v <= MAX_NUM:
            scores[v] += 2.0 if delta == 0 else (1.5 if abs(delta)==1 else 0.5)

    # S11: Direction
    if n >= 3:
        d1 = 1 if vals[-1] > vals[-2] else (-1 if vals[-1] < vals[-2] else 0)
        dn = Counter()
        for i in range(1, n-1):
            d = 1 if vals[i] > vals[i-1] else (-1 if vals[i] < vals[i-1] else 0)
            if d == d1:
                dn[vals[i+1]] += 1
        if dn:
            mx = max(dn.values())
            for v, c in dn.most_common(4):
                scores[v] += c / mx * 2

    # S12: Modular
    for m in [3, 5]:
        mods = [v % m for v in vals]
        mt = defaultdict(Counter)
        for i in range(len(mods)-1):
            mt[mods[i]][mods[i+1]] += 1
        lm = mods[-1]
        if lm in mt:
            top_mod = mt[lm].most_common(1)[0][0]
            rm = [v for v in vals[-50:] if v % m == top_mod]
            if rm:
                for v, c in Counter(rm).most_common(3):
                    scores[v] += c / len(rm) * 2

    # S13: Linear regression
    for w in [20, 30]:
        ww = min(w,n)
        seg = vals[-ww:]
        x = np.arange(len(seg))
        c = np.polyfit(x, seg, 1)
        pred = c[0]*len(seg) + c[1]
        for delta in range(-1, 2):
            v = int(round(pred)) + delta
            if 1 <= v <= MAX_NUM:
                scores[v] += 1.5

    # S14: Volatility regime
    std_r = np.std(vals[-20:])
    if std_r < 3:
        for v, c in Counter(vals[-20:]).most_common(3):
            scores[v] += 2.5
    else:
        trend = np.mean(vals[-5:]) - np.mean(vals[-20:])
        pred = np.mean(vals[-5:]) + trend * 0.5
        for delta in range(-2, 3):
            v = int(round(pred)) + delta
            if 1 <= v <= MAX_NUM:
                scores[v] += 1.0

    # S15: Delta bigram
    seg = vals[-min(80,n):]
    dts = [seg[i+1]-seg[i] for i in range(len(seg)-1)]
    if len(dts) >= 3:
        dtrans = defaultdict(Counter)
        for i in range(len(dts)-2):
            dtrans[(dts[i],dts[i+1])][dts[i+2]] += 1
        key = (dts[-2], dts[-1])
        if key in dtrans:
            t = sum(dtrans[key].values())
            for d, c in dtrans[key].most_common(3):
                v = lv + d
                if 1 <= v <= MAX_NUM:
                    scores[v] += c / t * 5

    return [v for v, _ in scores.most_common(K)]


# ================================================================
# BACKTEST — Per-position coverage at K=3..20
# ================================================================
START = 100
TESTED = total - START - 1
MAX_K = 20

# pos_coverage[pos][k] = count of draws where actual in top-(k+1)
pos_coverage = [[0]*MAX_K for _ in range(6)]
joint_coverage = [0]*MAX_K

t0 = time.time()
for idx in range(START, total-1):
    history = sorted_draws[:idx+1]
    actual = sorted_draws[idx+1]

    pos_rank = []  # rank (0-indexed) of actual value in candidates
    for pos in range(6):
        cands = fast_candidates_v2(history, pos, MAX_K)
        if actual[pos] in cands:
            rank = cands.index(actual[pos])
            pos_rank.append(rank)
            for k in range(rank, MAX_K):
                pos_coverage[pos][k] += 1
        else:
            pos_rank.append(-1)

    # Joint coverage
    for k in range(MAX_K):
        if all(r != -1 and r <= k for r in pos_rank):
            joint_coverage[k] += 1

    done = idx - START + 1
    if done % 200 == 0:
        el = time.time() - t0
        eta = el / done * (TESTED - done) / 60
        cov_k5 = [pos_coverage[p][4]/done*100 for p in range(6)]
        cov_k10 = [pos_coverage[p][9]/done*100 for p in range(6)]
        j5 = joint_coverage[4]/done*100
        j10 = joint_coverage[9]/done*100
        print(f"  [{done}/{TESTED}] {el:.0f}s ETA={eta:.1f}m")
        print(f"    K5  per-col: {' '.join(f'{c:.0f}%' for c in cov_k5)} | joint={j5:.1f}%")
        print(f"    K10 per-col: {' '.join(f'{c:.0f}%' for c in cov_k10)} | joint={j10:.1f}%")
        sys.stdout.flush()

el = time.time() - t0
print(f"\n  Completed in {el:.0f}s ({el/60:.1f}m)\n")

# ================================================================
# RESULTS
# ================================================================
print(f"{'='*90}")
print(f" PER-POSITION COVERAGE AT VARIOUS K ({TESTED} draws)")
print(f"{'='*90}\n")

print(f"  {'K':<5}", end='')
for p in range(6): print(f"  {'Col'+str(p+1):<9}", end='')
print(f"  {'JOINT':<9}  {'Est_combos':<12}")
print(f"  {'-'*85}")

for k in [1,2,3,4,5,6,7,8,9,10,12,15,20]:
    if k > MAX_K: continue
    print(f"  {k:<5}", end='')
    for p in range(6):
        cov = pos_coverage[p][k-1] / TESTED * 100
        print(f"  {cov:<9.2f}", end='')
    j = joint_coverage[k-1] / TESTED * 100
    # Estimate combos (C(k,1)^6 / 6! approximately)
    combos_est = k**6
    print(f"  {j:<9.2f}  ~{combos_est:>10,}")

# ================================================================
# ADAPTIVE K FOR TARGETS
# ================================================================
print(f"\n{'='*90}")
print(f" ADAPTIVE K — Min K per position for target coverage")
print(f"{'='*90}\n")

for target in [70, 80, 90, 95, 99]:
    ks = []
    for p in range(6):
        for k in range(MAX_K):
            if pos_coverage[p][k] / TESTED * 100 >= target:
                ks.append(k+1)
                break
        else:
            ks.append(MAX_K+1)

    total_combos = 1
    for k in ks: total_combos *= k

    # Joint with these adaptive Ks
    joint = 0
    for i in range(TESTED):
        # We need to re-check... but we don't have per-draw data
        pass  # Use the precomputed uniform K joint instead

    print(f"  Target {target}%: K = {ks} → {total_combos:,} max combos")

# ================================================================
# MINIMUM K FOR JOINT COVERAGE
# ================================================================
print(f"\n{'='*90}")
print(f" JOINT COVERAGE AT EACH K (the 6/6 hard ceiling)")
print(f"{'='*90}\n")

from math import comb
for k in [3,5,7,10,12,15,20]:
    if k > MAX_K: continue
    j = joint_coverage[k-1] / TESTED * 100
    valid_combos_est = comb(k, 1)**6  # Upper bound
    print(f"  K={k:>2}: Joint = {j:.2f}%  "
          f"(if we pick ANY 30 of ~{valid_combos_est:,} combos → "
          f"ceiling = {j:.2f}%)")

print(f"\n  Random baseline (30 sets): "
      f"{30/comb(45,6)*100:.7f}%")

print(f"\n{'='*90}")
print(f" BOTTOM LINE")
print(f"{'='*90}")
print(f"  Joint coverage tells us: even with PERFECT combo selection,")
print(f"  we can NEVER exceed this rate unless we increase K.")
print(f"  The only path to higher 6/6 is HIGHER per-column coverage.")
print(f"{'='*90}")
