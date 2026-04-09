"""
V22 NEXUS — FORENSIC-DRIVEN EVOLUTION
=======================================
Builds on V21 TITAN champion (MS-F42-70: 8@50K, 4@10K).

New from Forensic Analysis:
1. MS-F42-SR80: Multi-seed + SR80 (V21's best single + best multi)
2. 5-seed instead of 3: More diversity coverage
3. Adaptive weight: Each seed uses different weight method
4. Mean-reversion signal: Forensic proved all columns revert
5. Column-aware pool: C1/C6 stable, weight edge numbers more

V21 baselines:
  MS-F42-70:  1@5K, 4@10K, 5@18K, 5@20K, 6@30K, 8@50K (CHAMPION)
  F42-SR80:   0@5K, 3@10K, 4@18K, 4@20K, 5@30K, 7@50K (Best single)
"""
import sys, os, math, json, time, random
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper.data_manager import get_mega645_numbers

MAX = 45; PICK = 6


# ═══════════════════════════════════════
# SIGNAL ENGINE (V21 9-signal + forensic mean-reversion)
# ═══════════════════════════════════════

def compute_signals(data, at_index):
    """10-signal engine: V21's 9 + forensic mean-reversion boost."""
    relevant = data[:at_index]; n = len(relevant)
    if n < 50: return {num: PICK/MAX for num in range(1, MAX+1)}
    last = set(relevant[-1][:PICK]); base_p = PICK/MAX
    scores = {num: 0.0 for num in range(1, MAX+1)}

    # Signal 1-5: Multi-window frequency
    for w, wt in [(3,4),(5,3),(10,2),(20,1.5),(50,1)]:
        if n < w: continue
        fc = Counter()
        for d in relevant[-w:]:
            for num in d[:PICK]: fc[num] += 1
        for num in range(1, MAX+1): scores[num] += fc.get(num,0)/w * wt

    # Signal 6: Markov-1 follow
    follow = defaultdict(Counter); pc = Counter()
    for i in range(n-1):
        for p in relevant[i][:PICK]:
            pc[p] += 1
            for nx in relevant[i+1][:PICK]: follow[p][nx] += 1
    for num in range(1, MAX+1):
        tf = sum(follow[p].get(num,0) for p in last)
        tp = sum(pc[p] for p in last)
        if tp > 0: scores[num] += (tf/tp/base_p-1)*3

    # Signal 7: KNN pattern
    knn = Counter()
    for i in range(n-2):
        sim = len(set(relevant[i][:PICK]) & last)
        if sim >= 2:
            for num in relevant[i+1][:PICK]: knn[num] += sim*sim
    mx = max(knn.values()) if knn else 1
    for num in range(1, MAX+1): scores[num] += knn.get(num,0)/mx*2.5

    # Signal 8: Gap-due
    ls_idx = {}; gs, gn = defaultdict(float), defaultdict(int)
    for i, d in enumerate(relevant):
        for num in d[:PICK]:
            if num in ls_idx: gs[num] += i-ls_idx[num]; gn[num] += 1
            ls_idx[num] = i
    for num in range(1, MAX+1):
        if gn[num] < 5: continue
        mg = gs[num]/gn[num]; cg = n-ls_idx.get(num,0)
        if cg > mg*1.3: scores[num] += min((cg/mg-1)*1.5, 3)

    # Signal 9: Pair frequency
    pf = Counter()
    for d in relevant[-200:]:
        for p in combinations(sorted(d[:PICK]), 2): pf[p] += 1
    for num in range(1, MAX+1):
        scores[num] += sum(pf.get(tuple(sorted([p,num])),0) for p in last)*0.05

    # Signal 10: Pair-boost for non-top12
    top12 = sorted(scores.items(), key=lambda x: -x[1])[:12]
    top12_nums = [num for num, _ in top12]
    for num in range(1, MAX+1):
        if num in top12_nums: continue
        pair_bonus = sum(pf.get(tuple(sorted([num, t])), 0) for t in top12_nums)
        scores[num] += pair_bonus * 0.03

    # Signal 11: Repeat-adjacency (V21)
    ww = min(100, n)
    iai, ic = Counter(), Counter()
    for i in range(max(0,n-ww), n-1):
        curr, nxt = set(relevant[i][:PICK]), set(relevant[i+1][:PICK])
        for num in curr:
            ic[num] += 1
            if num in nxt: iai[num] += 1
    for num in range(1, MAX+1):
        if num in last and ic[num] > 5:
            scores[num] += (iai[num]/ic[num]-base_p)*3

    # Signal 12: Markov-2 (V21)
    if n >= 3:
        last2 = set(relevant[-2][:PICK])
        follow2 = defaultdict(Counter); pc2 = Counter()
        for i in range(n-2):
            for p in relevant[i][:PICK]:
                pc2[p] += 1
                for nx in relevant[i+2][:PICK]: follow2[p][nx] += 1
        for num in range(1, MAX+1):
            tf2 = sum(follow2[p].get(num,0) for p in last2)
            tp2 = sum(pc2[p] for p in last2)
            if tp2 > 0: scores[num] += (tf2/tp2/base_p-1)*2

    # Signal 13: Digit frequency (V21)
    if n >= 10:
        digit_freq = Counter()
        for d in relevant[-10:]:
            for num in d[:PICK]: digit_freq[num % 10] += 1
        total_digits = sum(digit_freq.values())
        for num in range(1, MAX+1):
            d = num % 10
            ed = total_digits / 10
            ad = digit_freq.get(d, 0)
            if ad > ed * 1.3:
                scores[num] += min((ad/ed - 1) * 1.0, 2)

    # ═══════════════════════════════════════
    # NEW Signal 14: FORENSIC MEAN-REVERSION
    # ═══════════════════════════════════════
    # Forensic proved: all columns show mean reversion (Sau↑ < 0, Sau↓ > 0)
    # If a number appeared in many recent draws → slightly penalize
    # If a number is "overdue" relative to its median gap → slightly boost
    if n >= 30:
        recent_30 = Counter()
        for d in relevant[-30:]:
            for num in d[:PICK]: recent_30[num] += 1
        all_time = Counter()
        for d in relevant:
            for num in d[:PICK]: all_time[num] += 1
        for num in range(1, MAX+1):
            expected_30 = (all_time.get(num, 0) / n) * 30
            actual_30 = recent_30.get(num, 0)
            if expected_30 > 0:
                deviation = (actual_30 - expected_30) / max(1, expected_30)
                # Mean reversion: if over-represented recently → penalize, if under → boost
                scores[num] -= deviation * 0.8

    return scores


def build_fusion_pool(data, at_index, scores, target_size):
    """V21 fusion pool + forensic edge-number boost."""
    relevant = data[:at_index]; n = at_index
    last_set = set(relevant[-1][:PICK])

    # Diverse pool (V21)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    pools = set(num for num, _ in ranked[:15])
    for w in [30, 50, 100]:
        fc = Counter(num for d in relevant[-min(w,n):] for num in d[:PICK])
        pools.update(num for num, _ in fc.most_common(15))
    ls = {}
    for i, d in enumerate(relevant):
        for num in d[:PICK]: ls[num] = i
    pools.update(sorted(range(1,MAX+1), key=lambda x: -(n-ls.get(x,0)))[:15])
    fc = Counter()
    for i in range(n-1):
        for p in relevant[i][:PICK]:
            if p in last_set:
                for nx in relevant[i+1][:PICK]: fc[nx] += 1
    pools.update(num for num, _ in fc.most_common(15))
    kc = Counter()
    for i in range(n-2):
        sim = len(set(relevant[i][:PICK]) & last_set)
        if sim >= 3:
            for num in relevant[i+1][:PICK]: kc[num] += sim*sim
    pools.update(num for num, _ in kc.most_common(15))
    cc = Counter()
    for d in relevant[-200:]:
        ds = set(d[:PICK]); ov = ds & last_set
        if len(ov) >= 2:
            for num in ds-last_set: cc[num] += len(ov)
    pools.update(num for num, _ in cc.most_common(15))
    for num in last_set:
        for delta in [-2,-1,1,2]:
            nb = num+delta
            if 1<=nb<=MAX: pools.add(nb)

    # Signal pool
    signal = set(num for num, _ in ranked[:35])

    # Gap-due pool
    gap_due = set(sorted(range(1, MAX+1), key=lambda x: -(n - ls.get(x, 0)))[:15])

    # Markov-2 pool
    if n >= 3:
        last2 = set(relevant[-2][:PICK])
        m2 = Counter()
        for i in range(n-2):
            for p in relevant[i][:PICK]:
                if p in last2:
                    for nx in relevant[i+2][:PICK]: m2[nx] += 1
        markov2 = set(num for num, _ in m2.most_common(15))
    else:
        markov2 = set()

    # Triplet transition pool
    if n >= 3:
        trip = Counter()
        for i in range(n-2):
            s1 = set(relevant[i][:PICK])
            overlap = s1 & last_set
            if len(overlap) >= 2 and i+2 < n:
                for num in relevant[i+2][:PICK]:
                    trip[num] += len(overlap)
        triplet_pool = set(num for num, _ in trip.most_common(10))
    else:
        triplet_pool = set()

    # ═══════════════════════════════
    # NEW: Forensic edge-number pool
    # ═══════════════════════════════
    # C1 stable: [1,2,3,4] and C6 stable: [43,44,45]
    # Always include these in pool since forensics showed they're fixed
    forensic_stable = {1, 2, 3, 4, 5, 43, 44, 45}

    fusion = pools | signal | gap_due | markov2 | triplet_pool | forensic_stable
    return sorted(fusion, key=lambda x: -scores.get(x, 0))[:target_size]


# ═══════════════════════════════════════
# GENERATOR — V21 Proven + Forensic Enhancements
# ═══════════════════════════════════════

def _make_weights(pool, scores, method):
    n = len(pool)
    ranked = sorted(pool, key=lambda x: -scores.get(x,0))
    wp = []
    if method == 'linear':
        for i, num in enumerate(ranked):
            wp.extend([num] * max(1, n-i))
    elif method == 'exp':
        for i, num in enumerate(ranked):
            wp.extend([num] * max(1, min(int(2**((n-i)/4)), 300)))
    elif method == 'step':
        for i, num in enumerate(ranked):
            wp.extend([num] * (10 if i < 12 else 1))
    elif method == 'zone':
        for i, num in enumerate(ranked):
            if i < 10: weight = 8
            elif i < 25: weight = 3
            else: weight = 1
            wp.extend([num] * weight)
    elif method == 'plateau':
        # NEW: Plateau weighting — top 15 equal, then linear decay
        for i, num in enumerate(ranked):
            if i < 15: weight = 6
            elif i < 30: weight = 3
            else: weight = 1
            wp.extend([num] * weight)
    return wp


def gen_hybrid(pool, scores, n_tickets, rng, sort_ratio=0.8):
    """V21's proven hybrid gen with configurable sort ratio."""
    n = len(pool)
    if n < PICK: return []

    raw_target = int(n_tickets * 1.2)
    methods = ['linear', 'exp', 'step', 'zone', 'plateau']
    wps = [_make_weights(pool, scores, m) for m in methods]
    per = raw_target // len(methods)

    selected = set()
    for j, wp in enumerate(wps):
        target = per if j < len(methods)-1 else raw_target - (len(methods)-1)*per
        att = 0
        while len(selected) < (j+1)*per and att < target*15:
            att += 1
            picked = set()
            while len(picked) < PICK:
                picked.add(rng.choice(wp))
            selected.add(tuple(sorted(picked)))

    raw_list = list(selected)
    scored = [(c, sum(scores.get(num,0) for num in c)) for c in raw_list]
    scored.sort(key=lambda x: -x[1])

    sort_count = int(n_tickets * sort_ratio)
    diverse_count = n_tickets - sort_count
    sorted_portion = [c for c, _ in scored[:sort_count]]
    remaining = [c for c, _ in scored[sort_count:]]

    if remaining and diverse_count > 0:
        sorted_set = set(sorted_portion)
        diverse_pool = [c for c in remaining if c not in sorted_set]
        if diverse_pool:
            diverse_portion = [diverse_pool[rng.randint(0, len(diverse_pool)-1)]
                             for _ in range(min(diverse_count, len(diverse_pool)))]
        else:
            diverse_portion = []
    else:
        diverse_portion = []

    portfolio = sorted_portion + diverse_portion
    return portfolio[:n_tickets]


def gen_multi_seed(pool, scores, n_tickets, base_seed, ti, sort_ratio, n_seeds=3):
    """Multi-seed: n_seeds → union → score-sort → top n_tickets."""
    all_combos = set()
    per_seed = int(n_tickets * (0.5 if n_seeds <= 3 else 0.35))

    for s in range(n_seeds):
        seed_val = base_seed + ti * 10000 + s * 777
        rng = random.Random(seed_val)
        port = gen_hybrid(pool, scores, per_seed, rng, sort_ratio)
        all_combos.update(port)

    scored = [(c, sum(scores.get(num,0) for num in c)) for c in all_combos]
    scored.sort(key=lambda x: -x[1])
    return [c for c, _ in scored[:n_tickets]]


def gen_adaptive_seed(pool, scores, n_tickets, base_seed, ti, sort_ratio, n_seeds=5):
    """NEW: Adaptive multi-seed — each seed uses different weight emphasis.
    Seed 0: Standard (balanced)
    Seed 1: Top-heavy (more firepower on best combos)
    Seed 2: Diverse (broader coverage)
    Seed 3: Mean-reversion adjusted
    Seed 4: Edge-number boosted
    """
    all_combos = set()
    per_seed = int(n_tickets * 0.3)

    for s in range(n_seeds):
        seed_val = base_seed + ti * 10000 + s * 777
        rng = random.Random(seed_val)

        # Each seed modifies scores slightly
        mod_scores = dict(scores)
        if s == 1:
            # Top-heavy: boost top 12 numbers
            top12 = sorted(scores.items(), key=lambda x: -x[1])[:12]
            for num, sc in top12:
                mod_scores[num] = sc * 1.2
        elif s == 2:
            # Diverse: flatten scores
            mean_sc = np.mean(list(scores.values()))
            for num in mod_scores:
                mod_scores[num] = mod_scores[num] * 0.6 + mean_sc * 0.4
        elif s == 3:
            # Edge boost: forensic stable numbers get bonus
            for num in [1,2,3,4,5,43,44,45]:
                mod_scores[num] = mod_scores.get(num, 0) + 1.5
        elif s == 4:
            # Anti-correlation: slightly prefer numbers NOT in last draw
            last = set(pool[:6]) if len(pool) >= 6 else set()
            for num in mod_scores:
                if num not in last:
                    mod_scores[num] += 0.5

        port = gen_hybrid(pool, mod_scores, per_seed, rng, sort_ratio)
        all_combos.update(port)

    scored = [(c, sum(scores.get(num,0) for num in c)) for c in all_combos]
    scored.sort(key=lambda x: -x[1])
    return [c for c, _ in scored[:n_tickets]]


# ═══════════════════════════════════════
# MAIN BACKTEST
# ═══════════════════════════════════════

def run():
    data = get_mega645_numbers()
    N = len(data); t0 = time.time()

    print("=" * 90)
    print("  ⚡ V22 NEXUS — FORENSIC-DRIVEN EVOLUTION")
    print(f"  {N} draws | 14-signal | Fusion pool + Forensic stable | Adaptive multi-seed")
    print("=" * 90)

    WARMUP = 200
    n_test = N - WARMUP

    # ── CONFIGS: 6 strategic variations ──
    #
    # V21 Champions:  MS-F42-70 (8@50K, 4@10K)
    #                 F42-SR80  (7@50K, 3@10K)
    #
    # V22 Hypothesis: Combining multi-seed + SR80 + forensic signals
    #                 should beat both V21 configs.

    CONFIGS = {
        # V21 CHAMPION baseline (verify same results)
        'V21-BASE':    {'pool_size': 42, 'sort_ratio': 0.7, 'gen': 'ms3'},
        # V22-A: MS + SR80 (combining V21's two best dimensions)
        'MS3-SR80':    {'pool_size': 42, 'sort_ratio': 0.8, 'gen': 'ms3'},
        # V22-B: 5-seed + SR70
        'MS5-SR70':    {'pool_size': 42, 'sort_ratio': 0.7, 'gen': 'ms5'},
        # V22-C: 5-seed + SR80 (full combo)
        'MS5-SR80':    {'pool_size': 42, 'sort_ratio': 0.8, 'gen': 'ms5'},
        # V22-D: Adaptive 5-seed (each seed different strategy)
        'ADAPT-70':    {'pool_size': 42, 'sort_ratio': 0.7, 'gen': 'adapt'},
        # V22-E: Adaptive 5-seed + SR80
        'ADAPT-80':    {'pool_size': 42, 'sort_ratio': 0.8, 'gen': 'adapt'},
    }

    PORT_SIZES = [5000, 10000, 18000, 20000, 30000, 50000]

    results = {}
    for cfg_name in CONFIGS:
        results[cfg_name] = {pt: {'six': 0, 'five': 0, 'four': 0, 'det': []} for pt in PORT_SIZES}

    contain = {'f42': 0}

    print(f"\n  {len(CONFIGS)} configs × {len(PORT_SIZES)} portfolio sizes")
    print(f"  Walk-forward: {n_test} draws")
    print(f"{'━'*90}")

    for ti in range(n_test):
        te = WARMUP + ti
        actual = set(data[te][:PICK])

        # Compute 14-signal scores
        sc = compute_signals(data, te)

        # Build fusion pool (42) with forensic stable numbers
        pool = build_fusion_pool(data, te, sc, 42)

        if len(actual & set(pool)) >= 6:
            contain['f42'] += 1

        # Generate portfolio for each config
        for cfg_name, cfg in CONFIGS.items():
            sr = cfg['sort_ratio']
            gen_type = cfg['gen']

            if gen_type == 'ms3':
                portfolio = gen_multi_seed(pool, sc, max(PORT_SIZES), 42, ti, sr, n_seeds=3)
            elif gen_type == 'ms5':
                portfolio = gen_multi_seed(pool, sc, max(PORT_SIZES), 42, ti, sr, n_seeds=5)
            elif gen_type == 'adapt':
                portfolio = gen_adaptive_seed(pool, sc, max(PORT_SIZES), 42, ti, sr, n_seeds=5)

            # Evaluate at each portfolio size
            for pt in PORT_SIZES:
                port = portfolio[:min(pt, len(portfolio))]
                if not port: continue
                best = 0
                for c in port:
                    m = len(actual & set(c))
                    if m > best:
                        best = m
                        if best >= 6: break
                if best >= 6:
                    results[cfg_name][pt]['six'] += 1
                    results[cfg_name][pt]['det'].append((te, sorted(actual)))
                if best >= 5: results[cfg_name][pt]['five'] += 1
                if best >= 4: results[cfg_name][pt]['four'] += 1

        if (ti+1) % 50 == 0:
            el = time.time() - t0
            eta = el/(ti+1)*(n_test-ti-1)
            ranking = [(k, results[k][20000]['six']) for k in results]
            ranking.sort(key=lambda x: -x[1])
            top3 = " | ".join(f"{k}={s}" for k, s in ranking[:3])
            draw_sec = el/(ti+1)
            print(f"  [{ti+1:4d}/{n_test}] @20K: {top3} | "
                  f"PC42={contain['f42']} | "
                  f"{draw_sec:.1f}s/draw | ETA:{eta/60:.1f}m")
            sys.stdout.flush()

    elapsed = time.time() - t0

    # ═══════════════════════════════════════
    # FINAL RESULTS
    # ═══════════════════════════════════════
    print(f"\n{'═'*90}")
    print(f"  ⚡ V22 NEXUS — FINAL RESULTS ({n_test} draws, {elapsed/60:.1f}min)")
    print(f"{'═'*90}")

    # Containment
    pct = contain['f42']/n_test*100
    print(f"\n  Pool Containment: Fusion 42 = {pct:.2f}% ({contain['f42']}/{n_test})")

    # 6/6 Hits table
    print(f"\n{'─'*90}")
    print(f"  🏆 6/6 HITS TABLE:")
    header = f"  {'Config':<14} │"
    for pt in PORT_SIZES:
        header += f" @{pt//1000}K"
    header += " │  Eff@10K"
    print(header)
    print(f"  {'─'*14} │{'─'*36} │ {'─'*8}")

    for cfg_name in CONFIGS:
        row = f"  {cfg_name:<14} │"
        for pt in PORT_SIZES:
            s6 = results[cfg_name][pt]['six']
            row += f" {s6:>4}"
        eff = results[cfg_name][10000]['six'] / 10
        row += f" │ {eff:.3f}/K"
        all_50k = [results[c][50000]['six'] for c in CONFIGS]
        if results[cfg_name][50000]['six'] == max(all_50k) and max(all_50k) > 0:
            row += " ⭐"
        print(row)

    # V21 comparison
    v21 = {5000: 1, 10000: 4, 18000: 5, 20000: 5, 30000: 6, 50000: 8}
    print(f"\n{'─'*90}")
    print(f"  🏆 V22 vs V21 CHAMPION (MS-F42-70):")
    for pt in PORT_SIZES:
        best_cfg = max(CONFIGS.keys(), key=lambda c: (results[c][pt]['six'], results[c][pt]['five']))
        s6 = results[best_cfg][pt]['six']
        s5 = results[best_cfg][pt]['five']
        baseline = v21.get(pt, 0)
        delta = s6 - baseline
        arrow = '🔺' if delta > 0 else ('🔻' if delta < 0 else '➖')
        eff = s6 / (pt/1000)
        print(f"    @{pt//1000:>2}K: {best_cfg:<14} = {s6:>3} hits 6/6, {s5:>3} hits 5/6 "
              f"(eff={eff:.3f}/K) {arrow} {delta:+d} vs V21={baseline}")

    # 5/6 table
    print(f"\n{'─'*90}")
    print(f"  📊 5/6 HITS:")
    header = f"  {'Config':<14} │"
    for pt in PORT_SIZES:
        header += f" @{pt//1000}K"
    print(header)
    print(f"  {'─'*14} │{'─'*36}")
    for cfg_name in CONFIGS:
        row = f"  {cfg_name:<14} │"
        for pt in PORT_SIZES:
            s5 = results[cfg_name][pt]['five']
            row += f" {s5:>4}"
        print(row)

    # Dimension analysis
    print(f"\n{'─'*90}")
    print(f"  📊 DIMENSION ANALYSIS @50K:")

    print(f"\n  1. Sort Ratio Effect (3-seed):")
    for cfg in ['V21-BASE', 'MS3-SR80']:
        s6 = results[cfg][50000]['six']
        s5 = results[cfg][50000]['five']
        print(f"     {cfg}: {s6} hits 6/6, {s5} hits 5/6")

    print(f"\n  2. Seed Count Effect (SR=70%):")
    for cfg in ['V21-BASE', 'MS5-SR70']:
        s6 = results[cfg][50000]['six']
        s5 = results[cfg][50000]['five']
        print(f"     {cfg}: {s6} hits 6/6, {s5} hits 5/6")

    print(f"\n  3. Full Combo (5-seed + SR80):")
    for cfg in ['MS5-SR80']:
        s6 = results[cfg][50000]['six']
        s5 = results[cfg][50000]['five']
        print(f"     {cfg}: {s6} hits 6/6, {s5} hits 5/6")

    print(f"\n  4. Adaptive Seeds:")
    for cfg in ['ADAPT-70', 'ADAPT-80']:
        s6 = results[cfg][50000]['six']
        s5 = results[cfg][50000]['five']
        print(f"     {cfg}: {s6} hits 6/6, {s5} hits 5/6")

    # 6/6 details
    best_50k_cfg = max(CONFIGS.keys(), key=lambda c: results[c][50000]['six'])
    det = results[best_50k_cfg][50000]['det']
    if det:
        print(f"\n  🎉 6/6 DRAWS ({best_50k_cfg} @50K):")
        for idx, nums in det[:20]:
            print(f"    Draw #{idx}: {nums}")

    # Save
    output = {
        'version': 'V22 NEXUS',
        'n_test': n_test,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'containment': {'f42': {'count': contain['f42'], 'pct': round(pct, 2)}},
        'configs': {},
        'elapsed_min': round(elapsed/60, 1),
    }
    for cfg_name in CONFIGS:
        output['configs'][cfg_name] = {}
        for pt in PORT_SIZES:
            r = results[cfg_name][pt]
            output['configs'][cfg_name][str(pt)] = {
                'six': r['six'], 'five': r['five'], 'four': r['four'],
            }
    for pt in PORT_SIZES:
        ranking = [(k, results[k][pt]['six'], results[k][pt]['five']) for k in CONFIGS]
        ranking.sort(key=lambda x: (-x[1], -x[2]))
        output[f'best_at_{pt//1000}k'] = {
            'config': ranking[0][0], 'six': ranking[0][1], 'five': ranking[0][2]
        }

    path = os.path.join(os.path.dirname(__file__), 'models', 'v22_nexus.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {path}")
    print(f"  Total: {elapsed/60:.1f}min ({elapsed/n_test:.2f}s/draw)")
    print(f"{'═'*90}")


if __name__ == '__main__':
    run()
