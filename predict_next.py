"""
ULTRA PREDICTION PIPELINE v2 — OPTIMIZED
==========================================
Pre-computes ALL signal tables once, then scores combos with fast lookups.
1. Per-column candidate generation (19 sources, Col3/Col4 boosted)
2. Quick 200-draw validation backtest (NO combo scoring — per-col only)
3. Generate 30 OPTIMAL sets for next draw (with pre-computed scoring)
"""
import sys, time, os, warnings
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations, product as iterproduct
from math import comb
from scraper.data_manager import get_mega645_numbers

data = get_mega645_numbers()
total = len(data)
MAX_NUM = 45
sorted_draws = [sorted(d[:6]) for d in data]

print(f"Data: {total} draws")
print(f"{'='*90}")
print(f" ULTRA PREDICTION PIPELINE v2.0")
print(f"{'='*90}\n")

# ================================================================
# CANDIDATE GENERATOR
# ================================================================
def ultra_gen(history, pos, K=10):
    """Generate Top-K candidates with Col3/Col4 boost."""
    scores = Counter()
    n = len(history)
    vals = [h[pos] for h in history]
    lv = vals[-1]
    is_mid = pos in [2, 3]

    # S1: Mode
    for w in [10, 20, 50, 100] + ([30, 80] if is_mid else []):
        seg = vals[-min(w,n):]
        for v, c in Counter(seg).most_common(5 if is_mid else 4):
            scores[v] += c / len(seg) * 2

    # S2: Transition
    trans = defaultdict(Counter)
    for i in range(n-1): trans[vals[i]][vals[i+1]] += 1
    if lv in trans:
        t = sum(trans[lv].values())
        for v, c in trans[lv].most_common(8 if is_mid else 6):
            scores[v] += c / t * 5

    # S3: Bigram
    if n >= 3:
        bg = defaultdict(Counter)
        for i in range(n-2): bg[(vals[i],vals[i+1])][vals[i+2]] += 1
        key = (vals[-2], vals[-1])
        if key in bg:
            t = sum(bg[key].values())
            for v, c in bg[key].most_common(5): scores[v] += c / t * 6

    # S4: Trigram
    if n >= 4:
        tg = defaultdict(Counter)
        for i in range(n-3): tg[(vals[i],vals[i+1],vals[i+2])][vals[i+3]] += 1
        key = (vals[-3], vals[-2], vals[-1])
        if key in tg:
            t = sum(tg[key].values())
            for v, c in tg[key].most_common(3): scores[v] += c / t * 8

    # S5: WMA
    for w in [15, 20, 30]:
        seg = vals[-min(w,n):]
        weights = np.arange(1, len(seg)+1, dtype=float)
        wma = np.average(seg, weights=weights)
        sp = 3 if is_mid else 2
        for d in range(-sp, sp+1):
            v = int(round(wma)) + d
            if 1 <= v <= MAX_NUM:
                scores[v] += 2.5 if d==0 else (1.5 if abs(d)==1 else 0.5)

    # S6: Spacing
    if pos > 0:
        sps = [h[pos]-h[pos-1] for h in history[-50:]]
        pv = history[-1][pos-1]
        avg_sp = np.mean(sps)
        for d in range(-3 if is_mid else -2, (3 if is_mid else 2)+1):
            v = pv + int(round(avg_sp)) + d
            if 1 <= v <= MAX_NUM:
                scores[v] += 3.0 if d==0 else (2.0 if abs(d)==1 else 0.8)

    # S7: Conditional prev col
    if pos > 0:
        cond = defaultdict(Counter)
        for h in history: cond[h[pos-1]][h[pos]] += 1
        pv = history[-1][pos-1]
        if pv in cond:
            t = sum(cond[pv].values())
            for v, c in cond[pv].most_common(6): scores[v] += c / t * 4

    # S8: Conditional next col
    if pos < 5:
        cond = defaultdict(Counter)
        for h in history: cond[h[pos+1]][h[pos]] += 1
        nv = history[-1][pos+1]
        if nv in cond:
            t = sum(cond[nv].values())
            for v, c in cond[nv].most_common(6): scores[v] += c / t * 4

    # S9: Cross-2
    if 0 < pos < 5:
        cross = defaultdict(Counter)
        for i in range(n-1):
            cross[(history[i][pos-1],history[i][pos+1])][history[i+1][pos]] += 1
        key = (history[-1][pos-1], history[-1][pos+1])
        if key in cross:
            t = sum(cross[key].values())
            for v, c in cross[key].most_common(4): scores[v] += c / t * 5

    # S10: Delta
    for w in [20, 50]:
        seg = vals[-min(w,n):]
        dts = [seg[i+1]-seg[i] for i in range(len(seg)-1)]
        if dts:
            for d, c in Counter(dts).most_common(5):
                v = lv + d
                if 1 <= v <= MAX_NUM: scores[v] += c / len(dts) * 2.5

    # S11: Delta bigram
    seg = vals[-min(80,n):]
    dts = [seg[i+1]-seg[i] for i in range(len(seg)-1)]
    if len(dts) >= 3:
        dtrans = defaultdict(Counter)
        for i in range(len(dts)-2): dtrans[(dts[i],dts[i+1])][dts[i+2]] += 1
        key = (dts[-2], dts[-1])
        if key in dtrans:
            t = sum(dtrans[key].values())
            for d, c in dtrans[key].most_common(3):
                v = lv + d
                if 1 <= v <= MAX_NUM: scores[v] += c / t * 5

    # S12: Gap/overdue
    ls = {}
    gd = defaultdict(list)
    pi = {}
    for i, v in enumerate(vals):
        ls[v] = i
        if v in pi: gd[v].append(i - pi[v])
        pi[v] = i
    for v in set(vals):
        if len(gd[v]) < 3: continue
        mg = np.mean(gd[v]); cg = n - ls.get(v, 0)
        if mg > 0 and cg > mg * 1.2: scores[v] += min((cg/mg-1)*2, 4.0)
        if mg > 0 and abs(cg - mg) < mg * 0.3: scores[v] += 2.0

    # S13: Direction
    if n >= 3:
        d1 = 1 if vals[-1]>vals[-2] else (-1 if vals[-1]<vals[-2] else 0)
        dn = Counter()
        for i in range(1, n-1):
            d = 1 if vals[i]>vals[i-1] else (-1 if vals[i]<vals[i-1] else 0)
            if d == d1: dn[vals[i+1]] += 1
        if dn:
            mx = max(dn.values())
            for v, c in dn.most_common(5): scores[v] += c / mx * 2

    # S14: Modular
    for m in [3, 5]:
        mods = [v % m for v in vals]
        mt = defaultdict(Counter)
        for i in range(len(mods)-1): mt[mods[i]][mods[i+1]] += 1
        lm = mods[-1]
        if lm in mt:
            top_mod = mt[lm].most_common(1)[0][0]
            rm = [v for v in vals[-50:] if v % m == top_mod]
            if rm:
                for v, c in Counter(rm).most_common(3): scores[v] += c/len(rm)*2

    # S15: Last ±2
    for d in range(-2, 3):
        v = lv + d
        if 1 <= v <= MAX_NUM:
            scores[v] += 2.0 if d==0 else (1.5 if abs(d)==1 else 0.5)

    # S16: LinReg
    for w in [20, 30]:
        seg = vals[-min(w,n):]
        x = np.arange(len(seg))
        c = np.polyfit(x, seg, 1)
        pred = c[0]*len(seg) + c[1]
        for d in range(-1, 2):
            v = int(round(pred)) + d
            if 1 <= v <= MAX_NUM: scores[v] += 1.5

    # S17: Volatility regime
    std_r = np.std(vals[-20:])
    if std_r < 3:
        for v, c in Counter(vals[-20:]).most_common(3): scores[v] += 2.5
    else:
        trend = np.mean(vals[-5:]) - np.mean(vals[-20:])
        pred = np.mean(vals[-5:]) + trend * 0.5
        for d in range(-2, 3):
            v = int(round(pred)) + d
            if 1 <= v <= MAX_NUM: scores[v] += 1.0

    # S18: EMA (boost for middle)
    if is_mid:
        for alpha in [0.1, 0.2, 0.3, 0.5]:
            seg = vals[-30:]
            s = seg[0]
            for vv in seg[1:]: s = alpha*vv + (1-alpha)*s
            for d in range(-1, 2):
                v = int(round(s)) + d
                if 1 <= v <= MAX_NUM: scores[v] += 1.5

    # S19: KNN position (boost for middle)
    if is_mid:
        kp = Counter()
        for i in range(n-1):
            if abs(vals[i] - lv) <= 2: kp[vals[i+1]] += 1
        if kp:
            mx = max(kp.values())
            for v, c in kp.most_common(5): scores[v] += c / mx * 3

    return [v for v, _ in scores.most_common(K)]


# ================================================================
# SECTION 1: FAST BACKTEST (per-col coverage only, last 200)
# ================================================================
print(f" SECTION 1: VALIDATION (per-col coverage, last 200 draws)\n")

START_BT = max(100, total - 201)
TESTED_BT = total - START_BT - 1

K_CONFIGS = [
    ("K5_adapt",  [3, 6, 7, 7, 6, 3]),
    ("K6_adapt",  [4, 7, 8, 8, 7, 4]),
    ("K7_adapt",  [4, 8, 10, 10, 8, 4]),
]

bt = {name: {'pos_hit': [0]*6, 'joint': 0} for name, _ in K_CONFIGS}

t0 = time.time()
for idx in range(START_BT, total-1):
    history = sorted_draws[:idx+1]
    actual = sorted_draws[idx+1]

    for name, ks in K_CONFIGS:
        all_hit = True
        for pos in range(6):
            cands = ultra_gen(history, pos, ks[pos])
            if actual[pos] in cands:
                bt[name]['pos_hit'][pos] += 1
            else:
                all_hit = False
        if all_hit:
            bt[name]['joint'] += 1

    done = idx - START_BT + 1
    if done % 50 == 0:
        el = time.time() - t0
        eta = el / done * (TESTED_BT - done) / 60
        print(f"  [{done}/{TESTED_BT}] {el:.0f}s ETA={eta:.1f}m")
        sys.stdout.flush()

el = time.time() - t0
print(f"\n  Done in {el:.0f}s\n")

print(f"  {'Config':<12}", end='')
for p in range(6): print(f" C{p+1:<5}", end='')
print(f" {'Joint':<8}")
print(f"  {'-'*65}")

best_name = None
best_joint = -1
for name, ks in K_CONFIGS:
    print(f"  {name:<12}", end='')
    for p in range(6):
        cov = bt[name]['pos_hit'][p] / TESTED_BT * 100
        print(f" {cov:<5.1f}%", end='')
    j = bt[name]['joint'] / TESTED_BT * 100
    print(f" {j:<8.2f}%")
    if j > best_joint:
        best_joint = j
        best_name = name

BEST_KS = dict(K_CONFIGS)[best_name]
print(f"\n  >> BEST: {best_name} = {BEST_KS} (joint={best_joint:.2f}%)\n")

# ================================================================
# SECTION 2: GENERATE 30 SETS FOR NEXT DRAW
# ================================================================
print(f"{'='*90}")
print(f" SECTION 2: PREDICTION FOR NEXT DRAW")
print(f"{'='*90}\n")

history = sorted_draws
last_draw = history[-1]
print(f"  Last draw: {last_draw}\n")

# Candidates
pos_cands = []
for pos in range(6):
    K = BEST_KS[pos]
    cands = ultra_gen(history, pos, K)
    pos_cands.append(cands)
    print(f"  Col{pos+1} (K={K}): {cands}")

# Generate valid combos
print(f"\n  Generating combos...")
valid = []
for combo in iterproduct(*pos_cands):
    ok = True
    for j in range(5):
        if combo[j] >= combo[j+1]:
            ok = False
            break
    if ok:
        valid.append(combo)
print(f"  Valid combos: {len(valid)}")

# Pre-compute ALL scoring signals ONCE
print(f"  Pre-computing signals...")
t0 = time.time()

# Signal A: per-number frequency score
freq_score = Counter()
for h in history[-50:]:
    for num in h: freq_score[num] += 1

# Signal B: transition probability from last draw
follow_score = defaultdict(float)
last_set = set(last_draw)
follow = defaultdict(Counter)
for i in range(len(history)-1):
    for p in history[i]:
        for nx in history[i+1]:
            follow[p][nx] += 1
for num in range(1, MAX_NUM+1):
    for p in last_set:
        follow_score[num] += follow[p].get(num, 0) * 0.02

# Signal C: pair co-occurrence
pair_freq = Counter()
for h in history[-100:]:
    for p in combinations(sorted(h), 2):
        pair_freq[p] += 1

# Signal D: average sum & range (for combo-level scoring)
avg_sum = np.mean([sum(h) for h in history[-50:]])
avg_range = np.mean([h[5]-h[0] for h in history[-50:]])

el = time.time() - t0
print(f"  Signals pre-computed in {el:.1f}s\n")

# Score combos with FAST lookups
def score_fast(combo):
    s = 0
    # Per-number scores
    for num in combo:
        s += freq_score.get(num, 0) * 0.1
        s += follow_score.get(num, 0)
    # Pair co-occurrence
    for p in combinations(sorted(combo), 2):
        s += pair_freq.get(p, 0) * 0.05
    # Spacing regularity
    for i in range(5):
        sp = combo[i+1]-combo[i]
        if sp > 15: s -= 0.5
        if sp < 2: s -= 0.3
    # Range
    s += (combo[-1]-combo[0]) * 0.08
    # Sum proximity
    s -= abs(sum(combo) - avg_sum) * 0.01
    return s

print(f"  Scoring {len(valid)} combos...")
t0 = time.time()
scored = [(c, score_fast(c)) for c in valid]
scored.sort(key=lambda x: -x[1])
el = time.time() - t0
print(f"  Done in {el:.1f}s\n")

N_SETS = 30
top30 = scored[:N_SETS]

print(f"  {'='*70}")
print(f"  TOP {min(N_SETS, len(top30))} PREDICTED SETS")
print(f"  {'='*70}\n")
print(f"  {'#':<4} {'Numbers':<30} {'Score':<10} {'Sum':<6} {'Range':<6}")
print(f"  {'-'*60}")
for i, (combo, score) in enumerate(top30):
    nums = " ".join(f"{n:>2}" for n in combo)
    print(f"  {i+1:<4} [{nums}]  {score:<10.2f} {sum(combo):<6} {combo[-1]-combo[0]:<6}")

# ================================================================
# SECTION 3: PORTFOLIO ANALYSIS
# ================================================================
print(f"\n{'='*90}")
print(f" SECTION 3: PORTFOLIO ANALYSIS")
print(f"{'='*90}\n")

all_nums = Counter()
for combo, _ in top30:
    for n in combo: all_nums[n] += 1

print(f"  Unique numbers: {len(all_nums)}")
print(f"  Numbers in portfolio:")
for num, cnt in sorted(all_nums.most_common(), key=lambda x: x[0]):
    bar = '█' * cnt
    print(f"    {num:>2}: {bar} ({cnt})")

# Per-position
print(f"\n  Per-Column values:")
for pos in range(6):
    vals = sorted(set(c[pos] for c, _ in top30))
    print(f"  Col{pos+1}: {vals}")

# Sum/Range
sums_t = [sum(c) for c, _ in top30]
ranges_t = [c[-1]-c[0] for c, _ in top30]
print(f"\n  Sum: [{min(sums_t)}, {max(sums_t)}] avg={np.mean(sums_t):.0f} (hist avg={avg_sum:.0f})")
print(f"  Range: [{min(ranges_t)}, {max(ranges_t)}] avg={np.mean(ranges_t):.0f} (hist avg={avg_range:.0f})")

# Confidence
random_rate = N_SETS / comb(45, 6) * 100
print(f"\n  Random 6/6: {random_rate:.7f}%")
print(f"  Joint ceiling: {best_joint:.2f}%")
print(f"  Improvement vs random: ~{best_joint/random_rate:.0f}x")

print(f"\n{'='*90}")
print(f" DONE — {len(top30)} sets generated")
print(f"{'='*90}")
