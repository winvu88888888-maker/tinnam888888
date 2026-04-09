"""
V20 APEX — MAXIMUM 6/6 ENGINE
================================
Combines ALL proven insights from V15-V19:

1. 8-Signal Engine (V18's 6 + Markov-2 + Digit-transition)
2. Multi-pool Fusion (Pool40 diverse + Pool35 signal → union → top 42)
3. SORT-first Hybrid: 70% score-sorted + 30% diversity-random
4. Adaptive zone weighting (hot/cold/due zones)
5. Constraint envelope from training data distribution
6. Per-draw isolated RNG (no state leakage)

Baseline to beat: V18 Ensemble = 13 hits 6/6 @50K
"""
import sys, os, math, json, time, random
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper.data_manager import get_mega645_numbers

MAX = 45; PICK = 6


# ============================================================
# 8-SIGNAL ENGINE
# ============================================================

def compute_signals_v20(data, at_index):
    """8-signal engine: V18's 6 + Markov-2 + digit-transition."""
    relevant = data[:at_index]; n = len(relevant)
    if n < 50: return {num: PICK/MAX for num in range(1, MAX+1)}
    last = set(relevant[-1][:PICK]); base_p = PICK/MAX
    scores = {num: 0.0 for num in range(1, MAX+1)}

    # Signal 1: Multi-window frequency (proven)
    for w, wt in [(3,4),(5,3),(10,2),(20,1.5),(50,1)]:
        if n < w: continue
        fc = Counter()
        for d in relevant[-w:]:
            for num in d[:PICK]: fc[num] += 1
        for num in range(1, MAX+1): scores[num] += fc.get(num,0)/w * wt

    # Signal 2: Follow-on transition (proven)
    follow = defaultdict(Counter); pc = Counter()
    for i in range(n-1):
        for p in relevant[i][:PICK]:
            pc[p] += 1
            for nx in relevant[i+1][:PICK]: follow[p][nx] += 1
    for num in range(1, MAX+1):
        tf = sum(follow[p].get(num,0) for p in last)
        tp = sum(pc[p] for p in last)
        if tp > 0: scores[num] += (tf/tp/base_p-1)*3

    # Signal 3: KNN similarity (proven)
    knn = Counter()
    for i in range(n-2):
        sim = len(set(relevant[i][:PICK]) & last)
        if sim >= 2:
            for num in relevant[i+1][:PICK]: knn[num] += sim*sim
    mx = max(knn.values()) if knn else 1
    for num in range(1, MAX+1): scores[num] += knn.get(num,0)/mx*2.5

    # Signal 4: Gap/due analysis (proven)
    ls_idx = {}; gs, gn = defaultdict(float), defaultdict(int)
    for i, d in enumerate(relevant):
        for num in d[:PICK]:
            if num in ls_idx: gs[num] += i-ls_idx[num]; gn[num] += 1
            ls_idx[num] = i
    for num in range(1, MAX+1):
        if gn[num] < 5: continue
        mg = gs[num]/gn[num]; cg = n-ls_idx.get(num,0)
        if cg > mg*1.3: scores[num] += min((cg/mg-1)*1.5, 3)

    # Signal 5: Pair frequency (proven)
    pf = Counter()
    for d in relevant[-200:]:
        for p in combinations(sorted(d[:PICK]), 2): pf[p] += 1
    for num in range(1, MAX+1):
        scores[num] += sum(pf.get(tuple(sorted([p,num])),0) for p in last)*0.05

    # Signal 6: Inertia/autocorrelation (proven)
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

    # Signal 7: Markov-2 (2-step transition)
    # Look at which numbers appear 2 draws after a given set
    if n >= 3:
        last2 = set(relevant[-2][:PICK])  # draw before last
        follow2 = defaultdict(Counter); pc2 = Counter()
        for i in range(n-2):
            for p in relevant[i][:PICK]:
                pc2[p] += 1
                for nx in relevant[i+2][:PICK]: follow2[p][nx] += 1
        for num in range(1, MAX+1):
            tf2 = sum(follow2[p].get(num,0) for p in last2)
            tp2 = sum(pc2[p] for p in last2)
            if tp2 > 0: scores[num] += (tf2/tp2/base_p-1)*2

    # Signal 8: Digit-pattern transition
    # Numbers sharing same last digit as recent winners get a boost  
    if n >= 10:
        digit_freq = Counter()
        for d in relevant[-10:]:
            for num in d[:PICK]:
                digit_freq[num % 10] += 1
        # Boost numbers whose last digit is trending
        total_digits = sum(digit_freq.values())
        for num in range(1, MAX+1):
            d = num % 10
            expected_digit = total_digits / 10
            actual_digit = digit_freq.get(d, 0)
            if actual_digit > expected_digit * 1.3:
                scores[num] += min((actual_digit/expected_digit - 1) * 1.0, 2)

    return scores


# ============================================================
# MULTI-POOL FUSION
# ============================================================

def build_diverse_pool(data, at_index, scores, max_pool):
    """11 diverse sub-pools → union → top by score (V18 proven)."""
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


def build_signal_pool(scores, pool_size):
    """Pure signal-ranked pool."""
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [num for num, _ in ranked[:pool_size]]


def build_fusion_pool(data, at_index, scores, target_size=42):
    """
    MULTI-POOL FUSION: union of diverse(40) + signal(35) + gap-due(15)
    → deduplicate → rank by score → top target_size.
    Goal: increase containment beyond single pool.
    """
    diverse = set(build_diverse_pool(data, at_index, scores, 40))
    signal = set(build_signal_pool(scores, 35))
    
    # Gap-due pool: numbers most overdue
    relevant = data[:at_index]; n = at_index
    ls = {}
    for i, d in enumerate(relevant):
        for num in d[:PICK]: ls[num] = i
    gap_sorted = sorted(range(1, MAX+1), key=lambda x: -(n - ls.get(x, 0)))
    gap_due = set(gap_sorted[:15])
    
    # Markov-2 pool: top predicted from 2-step transition
    if n >= 3:
        last2 = set(relevant[-2][:PICK])
        m2_scores = Counter()
        for i in range(n-2):
            for p in relevant[i][:PICK]:
                if p in last2:
                    for nx in relevant[i+2][:PICK]:
                        m2_scores[nx] += 1
        markov2 = set(num for num, _ in m2_scores.most_common(15))
    else:
        markov2 = set()
    
    # Union all pools
    fusion = diverse | signal | gap_due | markov2
    
    # Rank by score, take top target_size
    return sorted(fusion, key=lambda x: -scores.get(x, 0))[:target_size]


# ============================================================
# PORTFOLIO GENERATORS
# ============================================================

def _make_weights(pool, scores, method):
    """Build weighted sampling list."""
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
        # Adaptive: top 10 get 8x, mid 10-25 get 3x, rest 1x
        for i, num in enumerate(ranked):
            if i < 10: weight = 8
            elif i < 25: weight = 3
            else: weight = 1
            wp.extend([num] * weight)
    return wp


def gen_hybrid_portfolio(pool, scores, n_tickets, rng, constraints=None):
    """
    HYBRID SORT+SEQ PORTFOLIO:
    1. Generate large raw portfolio (~1.3x target) with multi-weight ensemble
    2. Score-sort all combos
    3. Take top 70% by score (SORT portion - captured proven winning combos)
    4. Fill remaining 30% with diversity-random combos from leftover
    5. Apply constraint filter if provided
    """
    n = len(pool)
    if n < PICK: return []
    
    # Over-generate to have enough after filtering
    raw_target = int(n_tickets * 1.4)
    
    # Multi-weight ensemble: 4 methods × 25% each
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
            combo = tuple(sorted(picked))
            
            # Light constraint filter
            if constraints:
                s = sum(combo)
                if s < constraints['sum_lo'] or s > constraints['sum_hi']:
                    continue
                odd = sum(1 for x in combo if x % 2 == 1)
                if odd < constraints['odd_lo'] or odd > constraints['odd_hi']:
                    continue
            
            selected.add(combo)
    
    raw_list = list(selected)
    
    # Score-sort all combos
    scored = [(combo, sum(scores.get(num, 0) for num in combo)) for combo in raw_list]
    scored.sort(key=lambda x: -x[1])
    
    # HYBRID: 70% SORT (top-scored) + 30% DIVERSITY (random from remainder)
    sort_count = int(n_tickets * 0.7)
    diverse_count = n_tickets - sort_count
    
    sorted_portion = [c for c, _ in scored[:sort_count]]
    
    # Diversity portion: random sample from remaining combos
    remaining = [c for c, _ in scored[sort_count:]]
    if remaining and diverse_count > 0:
        diverse_portion = [remaining[rng.randint(0, len(remaining)-1)] 
                          for _ in range(min(diverse_count, len(remaining)))]
        # Deduplicate diversity portion against sorted portion
        sorted_set = set(map(tuple, sorted_portion))
        diverse_portion = [c for c in diverse_portion if tuple(c) not in sorted_set]
    else:
        diverse_portion = []
    
    # Combine: sorted first, then diversity
    portfolio = sorted_portion + diverse_portion
    return portfolio[:n_tickets]


def gen_pure_ensemble(pool, scores, n_tickets, rng):
    """V18-style pure ensemble (3-weight), for comparison."""
    n = len(pool)
    if n < PICK: return []
    wps = [_make_weights(pool, scores, m) for m in ['linear', 'exp', 'step']]
    per = n_tickets // 3
    selected = set()
    for j, wp in enumerate(wps):
        target = per if j < 2 else n_tickets - 2*per
        att = 0
        while len(selected) < (j+1)*per and att < target*15:
            att += 1
            picked = set()
            while len(picked) < PICK:
                picked.add(rng.choice(wp))
            selected.add(tuple(sorted(picked)))
    return list(selected)


# ============================================================
# CONSTRAINT LEARNING
# ============================================================

def learn_constraints(data, at_index):
    """Learn combo property distribution from training data."""
    train = data[:at_index]
    sums = [sum(d[:PICK]) for d in train]
    odds = [sum(1 for x in d[:PICK] if x % 2 == 1) for d in train]
    rngs = [max(d[:PICK]) - min(d[:PICK]) for d in train]
    return {
        'sum_lo': int(np.percentile(sums, 3)),
        'sum_hi': int(np.percentile(sums, 97)),
        'odd_lo': max(0, int(np.percentile(odds, 3))),
        'odd_hi': min(PICK, int(np.percentile(odds, 97))),
        'range_lo': int(np.percentile(rngs, 3)),
        'range_hi': int(np.percentile(rngs, 97)),
    }


# ============================================================
# MAIN BACKTEST
# ============================================================

def run():
    data = get_mega645_numbers()
    N = len(data); t0 = time.time()
    
    print("=" * 90)
    print("  ⚡ V20 APEX — MAXIMUM 6/6 ENGINE")
    print(f"  {N} draws | 8-signal | Fusion pool 42 | Hybrid SORT+SEQ | Constraints")
    print("=" * 90)
    
    WARMUP = 200
    n_test = N - WARMUP
    
    # Test configs:
    # A) V20 APEX: Fusion pool 42 + Hybrid gen + Constraints
    # B) V20 APEX-RAW: Fusion pool 42 + Hybrid gen, no constraints
    # C) V20 DIVERSE: Diverse pool 40 + Hybrid gen (like V18 but better gen)
    # D) V18 REPLAY: Diverse pool 40 + Pure ensemble (baseline replay)
    # E) V20 FUSION+ENS: Fusion pool 42 + Pure ensemble
    
    CONFIGS = {
        'APEX': {'pool': 'fusion', 'pool_size': 42, 'gen': 'hybrid', 'constraints': True},
        'APEX-RAW': {'pool': 'fusion', 'pool_size': 42, 'gen': 'hybrid', 'constraints': False},
        'DIVERSE-HYB': {'pool': 'diverse', 'pool_size': 40, 'gen': 'hybrid', 'constraints': False},
        'FUSION-ENS': {'pool': 'fusion', 'pool_size': 42, 'gen': 'ensemble', 'constraints': False},
        'V18-REPLAY': {'pool': 'diverse', 'pool_size': 40, 'gen': 'ensemble', 'constraints': False},
    }
    
    PORT_SIZES = [5000, 10000, 18000, 20000, 30000, 50000]
    
    results = {}
    for cfg_name in CONFIGS:
        results[cfg_name] = {pt: {'six': 0, 'five': 0, 'four': 0, 'det': []} for pt in PORT_SIZES}
    
    contain = {'fusion42': 0, 'diverse40': 0}
    
    print(f"\n  Walk-forward: {n_test} draws")
    print(f"  Configs: {list(CONFIGS.keys())}")
    print(f"  Ports: {[f'{p//1000}K' for p in PORT_SIZES]}")
    print(f"{'━'*90}")
    
    for ti in range(n_test):
        te = WARMUP + ti
        actual = set(data[te][:PICK])
        
        # Compute 8-signal scores
        sc = compute_signals_v20(data, te)
        
        # Build pools
        pool_fusion = build_fusion_pool(data, te, sc, 42)
        pool_diverse = build_diverse_pool(data, te, sc, 40)
        
        # Track containment
        if len(actual & set(pool_fusion)) >= 6: contain['fusion42'] += 1
        if len(actual & set(pool_diverse)) >= 6: contain['diverse40'] += 1
        
        # Learn constraints from training data
        constraints = learn_constraints(data, te)
        
        # Generate portfolios for each config with isolated RNG
        for cfg_name, cfg in CONFIGS.items():
            seed_val = 42 + ti * 10000 + hash(cfg_name) % 9973
            rng = random.Random(seed_val)
            
            pool = pool_fusion if cfg['pool'] == 'fusion' else pool_diverse
            max_port = max(PORT_SIZES)
            
            if cfg['gen'] == 'hybrid':
                cons = constraints if cfg['constraints'] else None
                portfolio = gen_hybrid_portfolio(pool, sc, max_port, rng, cons)
            else:  # ensemble
                portfolio = gen_pure_ensemble(pool, sc, max_port, rng)
            
            # Evaluate at each portfolio size
            for pt in PORT_SIZES:
                port = portfolio[:min(pt, len(portfolio))]
                if port:
                    best = max(len(actual & set(c)) for c in port)
                else:
                    best = 0
                if best >= 6:
                    results[cfg_name][pt]['six'] += 1
                    results[cfg_name][pt]['det'].append((te, sorted(actual)))
                if best >= 5: results[cfg_name][pt]['five'] += 1
                if best >= 4: results[cfg_name][pt]['four'] += 1
        
        if (ti+1) % 50 == 0:
            el = time.time() - t0
            eta = el/(ti+1)*(n_test-ti-1)
            parts = []
            for cfg_name in ['APEX', 'V18-REPLAY']:
                s6 = results[cfg_name][20000]['six']
                parts.append(f"{cfg_name}={s6}")
            pc_f = contain['fusion42']
            pc_d = contain['diverse40']
            print(f"  [{ti+1:4d}/{n_test}] @20K: {' | '.join(parts)} | "
                  f"PC(F42={pc_f} D40={pc_d}) | {el:.0f}s ETA:{eta/60:.1f}m")
            sys.stdout.flush()
    
    elapsed = time.time() - t0
    
    # ════════════════════════════════════════════════════════
    # FINAL RESULTS
    # ════════════════════════════════════════════════════════
    
    print(f"\n{'═'*90}")
    print(f"  ⚡ V20 APEX — FINAL RESULTS ({n_test} tests, {elapsed/60:.1f}min)")
    print(f"{'═'*90}")
    
    # Pool containment
    print(f"\n  Pool Containment:")
    for pk, pv in contain.items():
        pct = pv/n_test*100
        print(f"    {pk}: {pv}/{n_test} ({pct:.2f}%)")
    
    # Results table
    for pt in PORT_SIZES:
        print(f"\n{'─'*90}")
        print(f"  @{pt//1000}K PORTFOLIO:")
        print(f"  {'Config':<20} │ {'4/6':>6} {'5/6':>6} {'6/6':>6} │ {'Hits 6':>7} {'Hits 5':>7} │ {'Eff(/K)':>8}")
        print(f"  {'─'*20} │ {'─'*6} {'─'*6} {'─'*6} │ {'─'*7} {'─'*7} │ {'─'*8}")
        
        rows = []
        for cfg_name in CONFIGS:
            r = results[cfg_name][pt]
            p4 = r['four']/n_test*100
            p5 = r['five']/n_test*100
            p6 = r['six']/n_test*100
            eff = r['six']/(pt/1000)
            rows.append((cfg_name, p4, p5, p6, r['six'], r['five'], eff))
        
        rows.sort(key=lambda x: (-x[4], -x[5]))
        
        for cfg_name, p4, p5, p6, s6, s5, eff in rows:
            bar = '█' * s6
            marker = ' ⭐' if s6 == rows[0][4] and s6 > 0 else ''
            print(f"  {cfg_name:<20} │ {p4:5.1f}% {p5:5.1f}% {p6:5.2f}% │ {s6:7d} {s5:7d} │ {eff:7.3f}/K  {bar}{marker}")
    
    # Champion determination
    print(f"\n{'═'*90}")
    print(f"  🏆 TOURNAMENT: BEST CONFIG PER SIZE")
    print(f"{'═'*90}")
    
    overall_best = None
    overall_best_score = -1
    
    for pt in PORT_SIZES:
        best_cfg = None
        best_six = -1
        best_five = -1
        for cfg_name in CONFIGS:
            r = results[cfg_name][pt]
            if r['six'] > best_six or (r['six'] == best_six and r['five'] > best_five):
                best_six = r['six']
                best_five = r['five']
                best_cfg = cfg_name
        
        eff = best_six / (pt/1000)
        print(f"  @{pt//1000:>2}K: {best_cfg:<20} → {best_six:>3} hits 6/6, "
              f"{best_five:>3} hits 5/6 (eff={eff:.3f}/K)")
        
        if pt == 50000:
            overall_best = best_cfg
            overall_best_score = best_six
    
    # V18 baseline comparison
    print(f"\n{'─'*90}")
    print(f"  📊 VS V18 BASELINE (13 hits @50K):")
    for cfg_name in CONFIGS:
        s6_50k = results[cfg_name][50000]['six']
        delta = s6_50k - 13
        arrow = '🔺' if delta > 0 else ('🔻' if delta < 0 else '➖')
        print(f"    {cfg_name:<20}: {s6_50k:>3} hits @50K ({arrow} {delta:+d} vs V18)")
    
    # 6/6 draw details for best config
    best_at_50k = max(CONFIGS.keys(), key=lambda c: results[c][50000]['six'])
    det = results[best_at_50k][50000]['det']
    if det:
        print(f"\n  🎉 6/6 DRAWS ({best_at_50k} @50K):")
        for idx, nums in det[:15]:
            print(f"    Draw #{idx}: {nums}")
    
    # Sweet spot analysis
    print(f"\n{'─'*90}")
    print(f"  🎯 SWEET SPOT (Best config: {overall_best}):")
    for pt in PORT_SIZES:
        r = results[overall_best][pt]
        eff = r['six'] / (pt/1000)
        pct_of_max = r['six'] / max(1, overall_best_score) * 100
        bar = '█' * r['six']
        print(f"    @{pt//1000:>2}K: {r['six']:>3} hits ({pct_of_max:5.1f}% of max) "
              f"eff={eff:.3f}/K  {bar}")
    
    # Save results
    output = {
        'version': 'V20 APEX',
        'n_test': n_test,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'containment': {k: {'count': v, 'pct': round(v/n_test*100,2)} for k,v in contain.items()},
        'configs': {},
        'elapsed_min': round(elapsed/60, 1),
    }
    for cfg_name in CONFIGS:
        output['configs'][cfg_name] = {}
        for pt in PORT_SIZES:
            r = results[cfg_name][pt]
            output['configs'][cfg_name][str(pt)] = {
                'six': r['six'], 'five': r['five'], 'four': r['four'],
                'p6': round(r['six']/n_test*100, 3),
                'p5': round(r['five']/n_test*100, 3),
            }
    
    path = os.path.join(os.path.dirname(__file__), 'models', 'v20_apex.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {path}")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"{'═'*90}")


if __name__ == '__main__':
    run()
