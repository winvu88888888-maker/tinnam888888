"""
V21 TITAN — CLEAN SWEEP
========================
Uses PROVEN V20 generator (0.6s/50K) with systematic parameter sweep.
V20 generator verified: 50K combos = 0.6s, signals = 0.01s

Sweep:
1. Pool sizes: 42, 44 (43/45 interpolate)
2. SORT ratios: 50%, 70%, 80%, 100%
3. Multi-seed: 3 seeds union vs single seed

V20 baseline: APEX=8@50K(constrained), APEX-RAW=9@50K(unconstrained)
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
# REUSE V20 PROVEN ENGINE (0.01s signals, 0.6s/50K gen)
# ═══════════════════════════════════════

def compute_signals(data, at_index):
    """9-signal engine: V20's 8 + pair-boost."""
    relevant = data[:at_index]; n = len(relevant)
    if n < 50: return {num: PICK/MAX for num in range(1, MAX+1)}
    last = set(relevant[-1][:PICK]); base_p = PICK/MAX
    scores = {num: 0.0 for num in range(1, MAX+1)}

    for w, wt in [(3,4),(5,3),(10,2),(20,1.5),(50,1)]:
        if n < w: continue
        fc = Counter()
        for d in relevant[-w:]:
            for num in d[:PICK]: fc[num] += 1
        for num in range(1, MAX+1): scores[num] += fc.get(num,0)/w * wt

    follow = defaultdict(Counter); pc = Counter()
    for i in range(n-1):
        for p in relevant[i][:PICK]:
            pc[p] += 1
            for nx in relevant[i+1][:PICK]: follow[p][nx] += 1
    for num in range(1, MAX+1):
        tf = sum(follow[p].get(num,0) for p in last)
        tp = sum(pc[p] for p in last)
        if tp > 0: scores[num] += (tf/tp/base_p-1)*3

    knn = Counter()
    for i in range(n-2):
        sim = len(set(relevant[i][:PICK]) & last)
        if sim >= 2:
            for num in relevant[i+1][:PICK]: knn[num] += sim*sim
    mx = max(knn.values()) if knn else 1
    for num in range(1, MAX+1): scores[num] += knn.get(num,0)/mx*2.5

    ls_idx = {}; gs, gn = defaultdict(float), defaultdict(int)
    for i, d in enumerate(relevant):
        for num in d[:PICK]:
            if num in ls_idx: gs[num] += i-ls_idx[num]; gn[num] += 1
            ls_idx[num] = i
    for num in range(1, MAX+1):
        if gn[num] < 5: continue
        mg = gs[num]/gn[num]; cg = n-ls_idx.get(num,0)
        if cg > mg*1.3: scores[num] += min((cg/mg-1)*1.5, 3)

    pf = Counter()
    for d in relevant[-200:]:
        for p in combinations(sorted(d[:PICK]), 2): pf[p] += 1
    for num in range(1, MAX+1):
        scores[num] += sum(pf.get(tuple(sorted([p,num])),0) for p in last)*0.05

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

    # Signal 9: Pair-boost
    top12 = sorted(scores.items(), key=lambda x: -x[1])[:12]
    top12_nums = [num for num, _ in top12]
    for num in range(1, MAX+1):
        if num in top12_nums: continue
        pair_bonus = sum(pf.get(tuple(sorted([num, t])), 0) for t in top12_nums)
        scores[num] += pair_bonus * 0.03

    return scores


def build_diverse_pool(data, at_index, scores, max_pool):
    relevant = data[:at_index]; n = at_index
    last_set = set(relevant[-1][:PICK])
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    pools = set()
    pools.update(num for num, _ in ranked[:15])
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
    return sorted(pools, key=lambda x: -scores.get(x,0))[:max_pool]


def build_fusion_pool(data, at_index, scores, target_size):
    relevant = data[:at_index]; n = at_index
    diverse = set(build_diverse_pool(data, at_index, scores, 40))
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    signal = set(num for num, _ in ranked[:35])
    ls = {}
    for i, d in enumerate(relevant):
        for num in d[:PICK]: ls[num] = i
    gap_due = set(sorted(range(1, MAX+1), key=lambda x: -(n - ls.get(x, 0)))[:15])
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
    if n >= 3:
        last_set = set(relevant[-1][:PICK])
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
    fusion = diverse | signal | gap_due | markov2 | triplet_pool
    return sorted(fusion, key=lambda x: -scores.get(x, 0))[:target_size]


# ═══════════════════════════════════════
# PROVEN V20 GENERATOR (0.6s per 50K)
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
    return wp


def gen_hybrid(pool, scores, n_tickets, rng, sort_ratio=0.7):
    """V20's proven hybrid SORT+SEQ generator with configurable sort_ratio."""
    n = len(pool)
    if n < PICK: return []

    raw_target = int(n_tickets * 1.4)
    methods = ['linear', 'exp', 'step', 'zone']
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


def gen_multi_seed(pool, scores, n_tickets, base_seed, ti, sort_ratio=0.7):
    """Multi-seed: 3 seeds → union → score-sort → top n_tickets."""
    all_combos = set()
    per_seed = int(n_tickets * 0.5)

    for s in range(3):
        seed_val = base_seed + ti * 10000 + s * 777
        rng = random.Random(seed_val)
        port = gen_hybrid(pool, scores, per_seed, rng, sort_ratio)
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
    print("  🔱 V21 TITAN — PARAMETER SWEEP (V20-proven generator)")
    print(f"  {N} draws | 9-signal | Fusion pool | Hybrid gen")
    print("=" * 90)

    WARMUP = 200
    n_test = N - WARMUP

    # ── CONFIGS: 6 strategic variations ──
    #
    # Dimension 1: Pool size (42 vs 44) — bigger pool = more containment but diluted scoring
    # Dimension 2: SORT ratio (50%, 70%, 80%) — more sorting = more top-heavy
    # Dimension 3: Multi-seed (single vs 3-seed union) — more diversity in combo space
    #
    # V20 APEX-RAW was: pool=42, sort=70%, single-seed = 9 hits @50K
    
    CONFIGS = {
        'F42-SR70':   {'pool_size': 42, 'sort_ratio': 0.7, 'multi_seed': False},  # = V20 APEX-RAW baseline
        'F44-SR70':   {'pool_size': 44, 'sort_ratio': 0.7, 'multi_seed': False},  # Bigger pool
        'F42-SR50':   {'pool_size': 42, 'sort_ratio': 0.5, 'multi_seed': False},  # More diverse
        'F42-SR80':   {'pool_size': 42, 'sort_ratio': 0.8, 'multi_seed': False},  # More top-heavy
        'MS-F42-70':  {'pool_size': 42, 'sort_ratio': 0.7, 'multi_seed': True},   # Multi-seed
        'MS-F44-70':  {'pool_size': 44, 'sort_ratio': 0.7, 'multi_seed': True},   # Multi-seed bigger
    }

    PORT_SIZES = [5000, 10000, 18000, 20000, 30000, 50000]

    results = {}
    for cfg_name in CONFIGS:
        results[cfg_name] = {pt: {'six': 0, 'five': 0, 'four': 0, 'det': []} for pt in PORT_SIZES}

    contain = {42: 0, 44: 0}

    print(f"\n  {len(CONFIGS)} configs × {len(PORT_SIZES)} portfolio sizes")
    print(f"  Walk-forward: {n_test} draws")
    print(f"  Expected: ~{0.7*len(CONFIGS)*n_test/60:.0f} min (0.7s/draw × {len(CONFIGS)} configs)")
    print(f"{'━'*90}")

    for ti in range(n_test):
        te = WARMUP + ti
        actual = set(data[te][:PICK])

        # Compute signals (0.01s)
        sc = compute_signals(data, te)

        # Build pools — only 2 unique sizes needed (0.00s each)
        pool42 = build_fusion_pool(data, te, sc, 42)
        pool44 = build_fusion_pool(data, te, sc, 44)
        
        if len(actual & set(pool42)) >= 6: contain[42] += 1
        if len(actual & set(pool44)) >= 6: contain[44] += 1

        pool_cache = {42: pool42, 44: pool44}

        # Generate portfolio for each config and evaluate
        for cfg_name, cfg in CONFIGS.items():
            pool = pool_cache[cfg['pool_size']]
            sr = cfg['sort_ratio']

            if cfg['multi_seed']:
                portfolio = gen_multi_seed(pool, sc, max(PORT_SIZES), 42, ti, sr)
            else:
                seed_val = 42 + ti * 10000 + hash(cfg_name) % 9973
                rng = random.Random(seed_val)
                portfolio = gen_hybrid(pool, sc, max(PORT_SIZES), rng, sr)

            # Evaluate at each portfolio size
            for pt in PORT_SIZES:
                port = portfolio[:min(pt, len(portfolio))]
                if not port:
                    continue
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
                  f"PC42={contain[42]} PC44={contain[44]} | "
                  f"{draw_sec:.1f}s/draw | ETA:{eta/60:.1f}m")
            sys.stdout.flush()

    elapsed = time.time() - t0

    # ═══════════════════════════════════════
    # FINAL RESULTS
    # ═══════════════════════════════════════
    print(f"\n{'═'*90}")
    print(f"  🔱 V21 TITAN — FINAL RESULTS ({n_test} draws, {elapsed/60:.1f}min)")
    print(f"{'═'*90}")

    # Containment
    print(f"\n  Pool Containment:")
    for ps in [42, 44]:
        pct = contain[ps]/n_test*100
        cpk = math.comb(ps, PICK)
        print(f"    Fusion {ps}: {pct:.2f}% ({contain[ps]}/{n_test}), C({ps},6)={cpk:,}")

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
        # Mark champion at 50K
        all_50k = [results[c][50000]['six'] for c in CONFIGS]
        if results[cfg_name][50000]['six'] == max(all_50k) and max(all_50k) > 0:
            row += " ⭐"
        print(row)

    # Best per size
    print(f"\n{'─'*90}")
    print(f"  🏆 CHAMPION PER SIZE:")
    v20 = {5000: 1, 10000: 3, 18000: 4, 20000: 4, 30000: 7, 50000: 9}
    for pt in PORT_SIZES:
        best_cfg = max(CONFIGS.keys(), key=lambda c: (results[c][pt]['six'], results[c][pt]['five']))
        s6 = results[best_cfg][pt]['six']
        s5 = results[best_cfg][pt]['five']
        baseline = v20.get(pt, 0)
        delta = s6 - baseline
        arrow = '🔺' if delta > 0 else ('🔻' if delta < 0 else '➖')
        eff = s6 / (pt/1000)
        print(f"    @{pt//1000:>2}K: {best_cfg:<14} = {s6:>3} hits (eff={eff:.3f}/K) "
              f"{arrow} {delta:+d} vs V20={baseline}")

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

    # Group analysis
    print(f"\n{'─'*90}")
    print(f"  📊 DIMENSION ANALYSIS @50K:")

    print(f"\n  1. Pool Size (SR=70%, single-seed):")
    for ps in [42, 44]:
        cfg = f'F{ps}-SR70'
        s6 = results[cfg][50000]['six']
        s5 = results[cfg][50000]['five']
        print(f"     F{ps}: {s6} hits 6/6, {s5} hits 5/6 (contain={contain[ps]/n_test*100:.1f}%)")

    print(f"\n  2. SORT Ratio (F42, single-seed):")
    for cfg, lbl in [('F42-SR50','50%'), ('F42-SR70','70%'), ('F42-SR80','80%')]:
        s6 = results[cfg][50000]['six']
        s5 = results[cfg][50000]['five']
        print(f"     SR={lbl}: {s6} hits 6/6, {s5} hits 5/6")

    print(f"\n  3. Multi-Seed Effect:")
    for ms, single in [('MS-F42-70','F42-SR70'), ('MS-F44-70','F44-SR70')]:
        ms_s = results[ms][50000]['six']
        sg_s = results[single][50000]['six']
        delta = ms_s - sg_s
        arrow = '🔺' if delta > 0 else ('🔻' if delta < 0 else '➖')
        print(f"     {ms} vs {single}: {ms_s} vs {sg_s} ({arrow} {delta:+d})")

    # 6/6 details
    best_50k_cfg = max(CONFIGS.keys(), key=lambda c: results[c][50000]['six'])
    det = results[best_50k_cfg][50000]['det']
    if det:
        print(f"\n  🎉 6/6 DRAWS ({best_50k_cfg} @50K):")
        for idx, nums in det[:20]:
            print(f"    Draw #{idx}: {nums}")

    # Save
    output = {
        'version': 'V21 TITAN',
        'n_test': n_test,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'containment': {str(ps): {'count': contain[ps], 'pct': round(contain[ps]/n_test*100,2)} for ps in [42, 44]},
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

    path = os.path.join(os.path.dirname(__file__), 'models', 'v21_titan.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {path}")
    print(f"  Total: {elapsed/60:.1f}min ({elapsed/n_test:.2f}s/draw)")
    print(f"{'═'*90}")


if __name__ == '__main__':
    run()
