"""
MINIMUM PORTFOLIO OPTIMIZER — 6/6 với số sets ÍT NHẤT
=======================================================
Câu hỏi chính: Cần bao nhiêu sets tối thiểu để hit 6/6?

Strategy:
1. Cho mỗi draw, generate TOP-N sets (N=1,2,3,5,10,20,30)
2. Đo P(6/6) cho mỗi N
3. Tìm N tối ưu: max(P(6/6)/N) = hiệu suất cao nhất
4. So sánh multiple strategies:
   - A: Narrow K (ít candidate → ít combo → xác suất cao hơn/set)
   - B: Wide K (nhiều candidate → cover nhiều hơn)
   - C: Diverse portfolio (spread risk)
   - D: Concentrated portfolio (all-in vào best combo)
"""
import sys, time, os, warnings
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations, product as iterproduct
from math import comb as mcomb
from scraper.data_manager import get_mega645_numbers

data = get_mega645_numbers()
total = len(data)
MAX_NUM = 45
sorted_draws = [sorted(d[:6]) for d in data]

print(f"Data: {total} draws")
print(f"{'='*90}")
print(f" MINIMUM PORTFOLIO OPTIMIZER — 6/6 với SỐ SETS ÍT NHẤT")
print(f"{'='*90}\n")

# ================================================================
# CANDIDATE GENERATOR (same as predict_next.py)
# ================================================================
def ultra_gen(history, pos, K=10):
    scores = Counter()
    n = len(history)
    vals = [h[pos] for h in history]
    lv = vals[-1]
    is_mid = pos in [2, 3]

    for w in [10, 20, 50, 100] + ([30, 80] if is_mid else []):
        seg = vals[-min(w,n):]
        for v, c in Counter(seg).most_common(5 if is_mid else 4):
            scores[v] += c / len(seg) * 2

    trans = defaultdict(Counter)
    for i in range(n-1): trans[vals[i]][vals[i+1]] += 1
    if lv in trans:
        t = sum(trans[lv].values())
        for v, c in trans[lv].most_common(8 if is_mid else 6):
            scores[v] += c / t * 5

    if n >= 3:
        bg = defaultdict(Counter)
        for i in range(n-2): bg[(vals[i],vals[i+1])][vals[i+2]] += 1
        key = (vals[-2], vals[-1])
        if key in bg:
            t = sum(bg[key].values())
            for v, c in bg[key].most_common(5): scores[v] += c / t * 6

    if n >= 4:
        tg = defaultdict(Counter)
        for i in range(n-3): tg[(vals[i],vals[i+1],vals[i+2])][vals[i+3]] += 1
        key = (vals[-3], vals[-2], vals[-1])
        if key in tg:
            t = sum(tg[key].values())
            for v, c in tg[key].most_common(3): scores[v] += c / t * 8

    for w in [15, 20, 30]:
        seg = vals[-min(w,n):]
        weights = np.arange(1, len(seg)+1, dtype=float)
        wma = np.average(seg, weights=weights)
        sp = 3 if is_mid else 2
        for d in range(-sp, sp+1):
            v = int(round(wma)) + d
            if 1 <= v <= MAX_NUM:
                scores[v] += 2.5 if d==0 else (1.5 if abs(d)==1 else 0.5)

    if pos > 0:
        sps = [h[pos]-h[pos-1] for h in history[-50:]]
        pv = history[-1][pos-1]
        avg_sp = np.mean(sps)
        for d in range(-3 if is_mid else -2, (3 if is_mid else 2)+1):
            v = pv + int(round(avg_sp)) + d
            if 1 <= v <= MAX_NUM:
                scores[v] += 3.0 if d==0 else (2.0 if abs(d)==1 else 0.8)

    if pos > 0:
        cond = defaultdict(Counter)
        for h in history: cond[h[pos-1]][h[pos]] += 1
        pv = history[-1][pos-1]
        if pv in cond:
            t = sum(cond[pv].values())
            for v, c in cond[pv].most_common(6): scores[v] += c / t * 4

    if pos < 5:
        cond = defaultdict(Counter)
        for h in history: cond[h[pos+1]][h[pos]] += 1
        nv = history[-1][pos+1]
        if nv in cond:
            t = sum(cond[nv].values())
            for v, c in cond[nv].most_common(6): scores[v] += c / t * 4

    if 0 < pos < 5:
        cross = defaultdict(Counter)
        for i in range(n-1):
            cross[(history[i][pos-1],history[i][pos+1])][history[i+1][pos]] += 1
        key = (history[-1][pos-1], history[-1][pos+1])
        if key in cross:
            t = sum(cross[key].values())
            for v, c in cross[key].most_common(4): scores[v] += c / t * 5

    for w in [20, 50]:
        seg = vals[-min(w,n):]
        dts = [seg[i+1]-seg[i] for i in range(len(seg)-1)]
        if dts:
            for d, c in Counter(dts).most_common(5):
                v = lv + d
                if 1 <= v <= MAX_NUM: scores[v] += c / len(dts) * 2.5

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

    if n >= 3:
        d1 = 1 if vals[-1]>vals[-2] else (-1 if vals[-1]<vals[-2] else 0)
        dn = Counter()
        for i in range(1, n-1):
            d = 1 if vals[i]>vals[i-1] else (-1 if vals[i]<vals[i-1] else 0)
            if d == d1: dn[vals[i+1]] += 1
        if dn:
            mx = max(dn.values())
            for v, c in dn.most_common(5): scores[v] += c / mx * 2

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

    for d in range(-2, 3):
        v = lv + d
        if 1 <= v <= MAX_NUM:
            scores[v] += 2.0 if d==0 else (1.5 if abs(d)==1 else 0.5)

    for w in [20, 30]:
        seg = vals[-min(w,n):]
        x = np.arange(len(seg))
        c = np.polyfit(x, seg, 1)
        pred = c[0]*len(seg) + c[1]
        for d in range(-1, 2):
            v = int(round(pred)) + d
            if 1 <= v <= MAX_NUM: scores[v] += 1.5

    std_r = np.std(vals[-20:])
    if std_r < 3:
        for v, c in Counter(vals[-20:]).most_common(3): scores[v] += 2.5
    else:
        trend = np.mean(vals[-5:]) - np.mean(vals[-20:])
        pred = np.mean(vals[-5:]) + trend * 0.5
        for d in range(-2, 3):
            v = int(round(pred)) + d
            if 1 <= v <= MAX_NUM: scores[v] += 1.0

    if is_mid:
        for alpha in [0.1, 0.2, 0.3, 0.5]:
            seg = vals[-30:]
            s = seg[0]
            for vv in seg[1:]: s = alpha*vv + (1-alpha)*s
            for d in range(-1, 2):
                v = int(round(s)) + d
                if 1 <= v <= MAX_NUM: scores[v] += 1.5

    if is_mid:
        kp = Counter()
        for i in range(n-1):
            if abs(vals[i] - lv) <= 2: kp[vals[i+1]] += 1
        if kp:
            mx = max(kp.values())
            for v, c in kp.most_common(5): scores[v] += c / mx * 3

    return scores  # Return RAW scores, not just top-K


# ================================================================
# PRE-COMPUTE SCORING SIGNALS
# ================================================================
def precompute_signals(history):
    """Pre-compute all combo-level scoring signals."""
    freq = Counter()
    for h in history[-50:]:
        for num in h: freq[num] += 1

    last_set = set(history[-1])
    follow = defaultdict(Counter)
    for i in range(len(history)-1):
        for p in history[i]:
            for nx in history[i+1]:
                follow[p][nx] += 1
    follow_score = {}
    for num in range(1, MAX_NUM+1):
        follow_score[num] = sum(follow[p].get(num, 0) for p in last_set) * 0.02

    pair_freq = Counter()
    for h in history[-100:]:
        for p in combinations(sorted(h), 2):
            pair_freq[p] += 1

    avg_sum = np.mean([sum(h) for h in history[-50:]])
    avg_range = np.mean([h[5]-h[0] for h in history[-50:]])

    return freq, follow_score, pair_freq, avg_sum, avg_range


def score_combo(combo, freq, follow_score, pair_freq, avg_sum):
    s = 0
    for num in combo:
        s += freq.get(num, 0) * 0.1
        s += follow_score.get(num, 0)
    for p in combinations(sorted(combo), 2):
        s += pair_freq.get(p, 0) * 0.05
    for i in range(5):
        sp = combo[i+1]-combo[i]
        if sp > 15: s -= 0.5
        if sp < 2: s -= 0.3
    s += (combo[-1]-combo[0]) * 0.08
    s -= abs(sum(combo) - avg_sum) * 0.01
    return s


# ================================================================
# STRATEGY DEFINITIONS
# ================================================================
# Test multiple K configs — from very narrow to wider
STRATEGIES = [
    # name, K_per_col, description
    ("ULTRA_NARROW",  [2, 3, 3, 3, 3, 2],  "Minimum K → fewest combos"),
    ("NARROW",        [2, 4, 5, 5, 4, 2],  "Narrow K"),
    ("BALANCED",      [3, 5, 6, 6, 5, 3],  "Balanced K"),
    ("STANDARD",      [3, 6, 7, 7, 6, 3],  "Standard K"),
    ("WIDE",          [4, 7, 8, 8, 7, 4],  "Wide K for more coverage"),
]

# ================================================================
# FULL BACKTEST — Test N=1,2,3,5,10,20 for each strategy
# ================================================================
N_VALUES = [1, 2, 3, 5, 10, 20]

print(f" FULL BACKTEST — {len(STRATEGIES)} strategies x {len(N_VALUES)} portfolio sizes\n")

START = 200
TESTED = total - START - 1

# results[strategy][N] = {h6, h5, h4, h3, total_valid, joint_hit}
results = {}
for sname, ks, _ in STRATEGIES:
    results[sname] = {}
    for N in N_VALUES:
        results[sname][N] = {'h6': 0, 'h5': 0, 'h4': 0, 'h3': 0}
    results[sname]['pos_hit'] = [0]*6
    results[sname]['valid_total'] = []

t0 = time.time()
for idx in range(START, total-1):
    history = sorted_draws[:idx+1]
    actual = sorted_draws[idx+1]
    actual_set = set(actual)
    actual_tuple = tuple(actual)

    # Pre-compute signals for this draw
    freq, follow_sc, pair_freq, avg_sum, avg_range = precompute_signals(history)

    for sname, ks, _ in STRATEGIES:
        # Generate candidates per column
        pos_cands = []
        for pos in range(6):
            raw_scores = ultra_gen(history, pos, ks[pos])
            cands = [v for v, _ in raw_scores.most_common(ks[pos])]
            pos_cands.append(cands)
            if actual[pos] in cands:
                results[sname]['pos_hit'][pos] += 1

        # Generate valid combos
        valid = []
        for combo in iterproduct(*pos_cands):
            ok = True
            for j in range(5):
                if combo[j] >= combo[j+1]:
                    ok = False
                    break
            if ok:
                valid.append(combo)

        results[sname]['valid_total'].append(len(valid))

        if not valid:
            continue

        # Score ALL valid combos
        scored = sorted(
            [(c, score_combo(c, freq, follow_sc, pair_freq, avg_sum)) for c in valid],
            key=lambda x: -x[1]
        )

        # Test different portfolio sizes
        for N in N_VALUES:
            top_n = [c for c, _ in scored[:N]]
            best_match = 0
            hit6 = False
            for combo in top_n:
                m = len(set(combo) & actual_set)
                best_match = max(best_match, m)
                if combo == actual_tuple: hit6 = True

            if hit6: results[sname][N]['h6'] += 1
            if best_match >= 5: results[sname][N]['h5'] += 1
            if best_match >= 4: results[sname][N]['h4'] += 1
            if best_match >= 3: results[sname][N]['h3'] += 1

    done = idx - START + 1
    if done % 100 == 0:
        el = time.time() - t0
        eta = el / done * (TESTED - done) / 60
        print(f"  [{done}/{TESTED}] {el:.0f}s ETA={eta:.1f}m")
        # Show progress for first strategy
        sn = STRATEGIES[0][0]
        for N in [1, 5, 10]:
            r = results[sn][N]
            print(f"    {sn} N={N}: 6/6={r['h6']} >=4={r['h4']} >=3={r['h3']}")
        sys.stdout.flush()

el = time.time() - t0
print(f"\n  Completed in {el:.0f}s ({el/60:.1f}m)\n")

# ================================================================
# RESULTS TABLE
# ================================================================
random_1 = 1 / mcomb(45, 6) * 100  # P(6/6) with 1 random set

print(f"{'='*90}")
print(f" RESULTS — {TESTED} draws backtested")
print(f"{'='*90}\n")

# Table 1: 6/6 rate by strategy × N
print(f"  === 6/6 HIT RATE ===\n")
print(f"  {'Strategy':<16} {'AvgCombos':<10}", end='')
for N in N_VALUES: print(f" {'N='+str(N):<9}", end='')
print(f"  {'Best N':<10}")
print(f"  {'-'*95}")

for sname, ks, desc in STRATEGIES:
    avg_vc = np.mean(results[sname]['valid_total'])
    print(f"  {sname:<16} {avg_vc:<10.0f}", end='')
    best_eff = 0
    best_n = 1
    for N in N_VALUES:
        rate = results[sname][N]['h6'] / TESTED * 100
        print(f" {rate:<9.3f}", end='')
        eff = rate / N
        if eff > best_eff:
            best_eff = eff
            best_n = N
    print(f"  N={best_n}")

# Table 2: Efficiency = P(6/6) / N
print(f"\n  === EFFICIENCY (P(6/6) per set) ===\n")
print(f"  {'Strategy':<16}", end='')
for N in N_VALUES: print(f" {'N='+str(N):<9}", end='')
print()
print(f"  {'-'*75}")

best_overall_eff = 0
best_overall_config = ""
for sname, ks, desc in STRATEGIES:
    print(f"  {sname:<16}", end='')
    for N in N_VALUES:
        rate = results[sname][N]['h6'] / TESTED * 100
        eff = rate / N
        print(f" {eff:<9.5f}", end='')
        if eff > best_overall_eff:
            best_overall_eff = eff
            best_overall_config = f"{sname} N={N}"
    print()

print(f"\n  Random efficiency: {random_1:.7f}% per set")
print(f"  BEST efficiency:  {best_overall_config} = {best_overall_eff:.5f}% per set")
print(f"  Improvement:      {best_overall_eff/random_1:.0f}x random")

# Table 3: >= 4/6 and >= 3/6 rates (for secondary prizes)
print(f"\n  === >=4/6 RATE (secondary prizes) ===\n")
print(f"  {'Strategy':<16}", end='')
for N in N_VALUES: print(f" {'N='+str(N):<9}", end='')
print()
print(f"  {'-'*75}")

for sname, ks, desc in STRATEGIES:
    print(f"  {sname:<16}", end='')
    for N in N_VALUES:
        rate = results[sname][N]['h4'] / TESTED * 100
        print(f" {rate:<9.2f}", end='')
    print()

print(f"\n  === >=3/6 RATE ===\n")
print(f"  {'Strategy':<16}", end='')
for N in N_VALUES: print(f" {'N='+str(N):<9}", end='')
print()
print(f"  {'-'*75}")

for sname, ks, desc in STRATEGIES:
    print(f"  {sname:<16}", end='')
    for N in N_VALUES:
        rate = results[sname][N]['h3'] / TESTED * 100
        print(f" {rate:<9.2f}", end='')
    print()

# Table 4: Per-position coverage
print(f"\n  === PER-POSITION COVERAGE (joint = 6/6 ceiling) ===\n")
print(f"  {'Strategy':<16} {'K':<20}", end='')
for p in range(6): print(f" C{p+1:<5}", end='')
print(f" {'Joint':<8}")
print(f"  {'-'*80}")

for sname, ks, desc in STRATEGIES:
    print(f"  {sname:<16} {str(ks):<20}", end='')
    joint = 1.0
    for p in range(6):
        cov = results[sname]['pos_hit'][p] / TESTED * 100
        joint *= cov / 100
        print(f" {cov:<5.1f}%", end='')
    print(f" {joint*100:<8.3f}%")

# ================================================================
# VERDICT: OPTIMAL CONFIGURATION
# ================================================================
print(f"\n{'='*90}")
print(f" VERDICT: OPTIMAL CONFIGURATION")
print(f"{'='*90}\n")

# Find best config for different budgets
for budget in [1, 3, 5, 10]:
    best_s = ""
    best_rate = 0
    for sname, ks, desc in STRATEGIES:
        rate = results[sname][min(budget, max(N_VALUES))]['h6'] / TESTED * 100
        if rate > best_rate:
            best_rate = rate
            best_s = sname
    if best_rate > 0:
        print(f"  Budget {budget:>2} sets: BEST = {best_s} → 6/6={best_rate:.3f}% "
              f"({best_rate/random_1/budget:.0f}x random/set)")
    else:
        print(f"  Budget {budget:>2} sets: BEST = {best_s} → 6/6=0% "
              f"(need more data or wider K)")

# Best 4/6 configs
print()
for budget in [1, 3, 5, 10]:
    best_s = ""
    best_rate = 0
    for sname, ks, desc in STRATEGIES:
        n_key = min(budget, max(N_VALUES))
        if n_key in results[sname]:
            rate = results[sname][n_key]['h4'] / TESTED * 100
            if rate > best_rate:
                best_rate = rate
                best_s = sname
    print(f"  Budget {budget:>2} sets: BEST >=4/6 = {best_s} → {best_rate:.2f}%")

print(f"\n{'='*90}")
print(f" MATHEMATICAL REALITY CHECK")
print(f"{'='*90}")
print(f"  C(45,6) = {mcomb(45,6):,} total combos")
print(f"  1 set random  = {1/mcomb(45,6)*100:.7f}%")
print(f"  10 sets random = {10/mcomb(45,6)*100:.6f}%")
print(f"  Even 1000x improvement = {1000/mcomb(45,6)*100:.4f}%")
print(f"  Need ~{mcomb(45,6)//1000:,} sets for 1/1000 chance with 1000x improvement")
print(f"{'='*90}")
