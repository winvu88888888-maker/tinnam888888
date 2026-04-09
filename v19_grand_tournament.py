"""
V19 FOCUSED TOURNAMENT (FAST) — Based on Phase 1 results
==========================================================
Phase 1 proved: Pool 40 = BEST (7 hits, 2.19x lift @20K)
                Pool 28 = surprise runner-up (6 hits, 1.82x lift)
                Random = only 1 hit (SIGNAL WORKS!)

Now test the method combinations FAST:
- Only top 2 pools: 40, 28 (+ pool 34 as tiebreaker)
- Only test at 20K (proven sweet spot from sweep)
- 3 pool methods × 4 gen methods = 12 configs per pool = 36 total
- Reuse signals, compute once per draw
- FAST portfolio gen (cap at 20K, no 2x oversample for sorted)
"""
import sys, os, math, time, random, json
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper.data_manager import get_mega645_numbers

MAX = 45; PICK = 6

# ============================================================
# SIGNAL ENGINE (V18 6-signal — proven)
# ============================================================

def compute_signals(data, at_index):
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
    return scores


# ============================================================
# POOL BUILDERS
# ============================================================

def build_signal_pool(scores, pool_size):
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [num for num, _ in ranked[:pool_size]]

def build_diverse_pool(data, at_index, scores, pool_size):
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
    return sorted(pools, key=lambda x: -scores.get(x,0))[:pool_size]

def build_hybrid_pool(data, at_index, scores, pool_size):
    core = int(pool_size * 0.7)
    diverse = build_diverse_pool(data, at_index, scores, core)
    ds = set(diverse)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    ext = [num for num, _ in ranked if num not in ds][:pool_size - core]
    return sorted(diverse + ext, key=lambda x: -scores.get(x,0))[:pool_size]


# ============================================================
# COMBO GENERATORS (optimized for speed)
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
    return wp

def gen_ensemble(pool, scores, n_tickets, rng):
    """V18 ensemble: lin+exp+step."""
    n = len(pool)
    if n < PICK: return []
    wps = [_make_weights(pool, scores, m) for m in ['linear','exp','step']]
    per = n_tickets // 3
    selected = set()
    for j, wp in enumerate(wps):
        target = per if j < 2 else n_tickets - 2*per
        att = 0
        while len(selected) < (j+1)*per and att < target*12:
            att += 1
            picked = set()
            while len(picked) < PICK:
                picked.add(rng.choice(wp))
            selected.add(tuple(sorted(picked)))
    return list(selected)

def gen_linear(pool, scores, n_tickets, rng):
    """Linear-only weighted."""
    n = len(pool)
    if n < PICK: return []
    wp = _make_weights(pool, scores, 'linear')
    selected = set()
    att = 0
    while len(selected) < n_tickets and att < n_tickets*12:
        att += 1
        picked = set()
        while len(picked) < PICK:
            picked.add(rng.choice(wp))
        selected.add(tuple(sorted(picked)))
    return list(selected)

def gen_constrained(pool, scores, n_tickets, rng, constraints):
    """Constraint-filtered linear."""
    n = len(pool)
    if n < PICK: return []
    wp = _make_weights(pool, scores, 'linear')
    selected = set()
    att = 0
    while len(selected) < n_tickets and att < n_tickets*25:
        att += 1
        picked = set()
        while len(picked) < PICK:
            picked.add(rng.choice(wp))
        combo = tuple(sorted(picked))
        s = sum(combo)
        if s < constraints['sum_lo'] or s > constraints['sum_hi']: continue
        odd = sum(1 for x in combo if x % 2 == 1)
        if odd < constraints['odd_lo'] or odd > constraints['odd_hi']: continue
        rng_val = max(combo) - min(combo)
        if rng_val < constraints['range_lo'] or rng_val > constraints['range_hi']: continue
        selected.add(combo)
    return list(selected)

def gen_pair_boost(pool, scores, n_tickets, rng, pair_scores):
    """Pair-boosted linear."""
    n = len(pool)
    if n < PICK: return []
    ranked = sorted(pool, key=lambda x: -scores.get(x,0))
    wp = []
    for i, num in enumerate(ranked):
        base = max(1, n-i)
        pair_bonus = sum(pair_scores.get(tuple(sorted([num, o])), 0) for o in ranked[:12] if o != num)
        wp.extend([num] * (base + int(pair_bonus * 2)))
    selected = set()
    att = 0
    while len(selected) < n_tickets and att < n_tickets*12:
        att += 1
        picked = set()
        while len(picked) < PICK:
            picked.add(rng.choice(wp))
        selected.add(tuple(sorted(picked)))
    return list(selected)

def gen_pure_random(n_tickets, rng):
    """Pure random from C(45,6)."""
    universe = list(range(1, MAX+1))
    selected = set()
    att = 0
    while len(selected) < n_tickets and att < n_tickets*5:
        att += 1
        selected.add(tuple(sorted(rng.sample(universe, PICK))))
    return list(selected)


# ============================================================
# EVALUATION HELPERS
# ============================================================

def evaluate_portfolio(portfolio, actual, n_tickets):
    """Evaluate a portfolio at given size, return best match."""
    sub = portfolio[:min(n_tickets, len(portfolio))]
    if not sub: return 0
    return max(len(actual & set(c)) for c in sub)

def score_sort_portfolio(portfolio, scores):
    """Sort portfolio by combo score (best first)."""
    return sorted(portfolio, key=lambda c: sum(scores.get(n,0) for n in c), reverse=True)


# ============================================================
# MAIN TOURNAMENT
# ============================================================

def run():
    data = get_mega645_numbers()
    N = len(data); t0 = time.time()
    WARMUP = 200; n_test = N - WARMUP
    
    # From Phase 1: Pool 40 = best (7 hits), Pool 28/34 = runners-up
    TEST_POOLS = [28, 34, 40]
    PORT = 20000  # Proven sweet spot
    EXTRA_PORTS = [5000, 10000, 30000]  # Also track for comparison
    ALL_PORTS = [5000, 10000, 20000, 30000]
    
    # Config: (pool_size, pool_method, gen_method)
    CONFIGS = []
    for ps in TEST_POOLS:
        for pm in ['signal', 'diverse', 'hybrid']:
            for gm in ['ensemble', 'linear', 'constrained', 'pair_boost']:
                CONFIGS.append((ps, pm, gm))
    
    # + score-sorted variants of the best generators
    SORT_CONFIGS = []
    for ps in TEST_POOLS:
        for pm in ['diverse', 'hybrid']:
            for gm in ['ensemble', 'linear']:
                SORT_CONFIGS.append((ps, pm, gm))
    
    # + random baseline
    
    print("=" * 90)
    print("  🏆 V19 FOCUSED TOURNAMENT — FAST METHOD COMPARISON")
    print(f"  {N} draws | {n_test} tests | Pools: {TEST_POOLS}")
    print(f"  {len(CONFIGS)} configs + {len(SORT_CONFIGS)} sorted variants + random")
    print(f"  Portfolio: {ALL_PORTS}")
    print("=" * 90)
    
    # Results tracking
    results = {}
    for cfg in CONFIGS:
        results[cfg] = {pt: {'six': 0, 'five': 0, 'four': 0} for pt in ALL_PORTS}
    for cfg in SORT_CONFIGS:
        key = (*cfg, 'sorted')
        results[key] = {pt: {'six': 0, 'five': 0, 'four': 0} for pt in ALL_PORTS}
    results['random'] = {pt: {'six': 0, 'five': 0, 'four': 0} for pt in ALL_PORTS}
    
    # Pool containment
    contain = {ps: 0 for ps in TEST_POOLS}
    
    for ti in range(n_test):
        te = WARMUP + ti
        actual = set(data[te][:PICK])
        
        sc = compute_signals(data, te)
        
        # Constraints (learn from training data only)
        train = data[:te]
        sums = [sum(d[:PICK]) for d in train]
        odds = [sum(1 for x in d[:PICK] if x % 2 == 1) for d in train]
        rngs = [max(d[:PICK]) - min(d[:PICK]) for d in train]
        constraints = {
            'sum_lo': int(np.percentile(sums, 5)),
            'sum_hi': int(np.percentile(sums, 95)),
            'odd_lo': max(0, int(np.percentile(odds, 5))),
            'odd_hi': min(PICK, int(np.percentile(odds, 95))),
            'range_lo': int(np.percentile(rngs, 5)),
            'range_hi': int(np.percentile(rngs, 95)),
        }
        
        # Pair scores
        pair_sc = Counter()
        for d in train[-200:]:
            for p in combinations(sorted(d[:PICK]), 2): pair_sc[p] += 1
        
        # Build pools once per pool size × method
        pool_cache = {}
        for ps in TEST_POOLS:
            pool_cache[(ps, 'signal')] = build_signal_pool(sc, ps)
            pool_cache[(ps, 'diverse')] = build_diverse_pool(data, te, sc, ps)
            pool_cache[(ps, 'hybrid')] = build_hybrid_pool(data, te, sc, ps)
            
            # Track containment (diverse pool)
            if len(actual & set(pool_cache[(ps, 'diverse')])) >= 6:
                contain[ps] += 1
        
        # Generate & evaluate each config
        max_port = max(ALL_PORTS)
        
        for cfg in CONFIGS:
            ps, pm, gm = cfg
            pool = pool_cache[(ps, pm)]
            seed_val = 42 + ti * 10000 + ps * 100 + hash((pm, gm)) % 997
            rng = random.Random(seed_val)
            
            if gm == 'ensemble':
                portfolio = gen_ensemble(pool, sc, max_port, rng)
            elif gm == 'linear':
                portfolio = gen_linear(pool, sc, max_port, rng)
            elif gm == 'constrained':
                portfolio = gen_constrained(pool, sc, max_port, rng, constraints)
            else:  # pair_boost
                portfolio = gen_pair_boost(pool, sc, max_port, rng, pair_sc)
            
            for pt in ALL_PORTS:
                best = evaluate_portfolio(portfolio, actual, pt)
                if best >= 6: results[cfg][pt]['six'] += 1
                if best >= 5: results[cfg][pt]['five'] += 1
                if best >= 4: results[cfg][pt]['four'] += 1
            
            # Score-sorted variant (only for select configs)
            if cfg[:3] in [(ps2, pm2, gm2) for ps2, pm2, gm2 in SORT_CONFIGS if (ps2,pm2,gm2) == (ps,pm,gm)]:
                sorted_port = score_sort_portfolio(portfolio, sc)
                skey = (*cfg, 'sorted')
                for pt in ALL_PORTS:
                    best = evaluate_portfolio(sorted_port, actual, pt)
                    if best >= 6: results[skey][pt]['six'] += 1
                    if best >= 5: results[skey][pt]['five'] += 1
                    if best >= 4: results[skey][pt]['four'] += 1
        
        # Random baseline
        rng_r = random.Random(42 + ti * 10000 + 9999)
        rand_port = gen_pure_random(max_port, rng_r)
        for pt in ALL_PORTS:
            best = evaluate_portfolio(rand_port, actual, pt)
            if best >= 6: results['random'][pt]['six'] += 1
            if best >= 5: results['random'][pt]['five'] += 1
            if best >= 4: results['random'][pt]['four'] += 1
        
        if (ti+1) % 50 == 0:
            el = time.time() - t0
            eta = el/(ti+1)*(n_test-ti-1)
            # Show top 5 at 20K
            ranking = [(k, results[k][20000]['six']) for k in results if k != 'random']
            ranking.sort(key=lambda x: -x[1])
            top5_str = " | ".join(
                f"{'P'+str(k[0])+'/'+k[1][:3]+'/'+k[2][:4] if isinstance(k, tuple) else k}={s}"
                for k, s in ranking[:4]
            )
            rand_6 = results['random'][20000]['six']
            print(f"  [{ti+1:4d}/{n_test}] {top5_str} | RAND={rand_6} | {el:.0f}s ETA:{eta/60:.1f}m")
            sys.stdout.flush()
    
    elapsed = time.time() - t0
    
    # ════════════════════════════════════════════════════════
    # FINAL RESULTS
    # ════════════════════════════════════════════════════════
    
    print(f"\n{'═'*90}")
    print(f"  🏆 V19 GRAND TOURNAMENT — FINAL RESULTS ({n_test} tests, {elapsed/60:.1f}min)")
    print(f"{'═'*90}")
    
    # Pool containment
    print(f"\n  Pool Containment:")
    for ps in TEST_POOLS:
        pc = contain[ps] / n_test * 100
        cpk = math.comb(ps, PICK)
        print(f"    Pool {ps}: {pc:.1f}% ({contain[ps]}/{n_test}), C({ps},6)={cpk:,}")
    
    # Results table @ 20K
    print(f"\n{'─'*90}")
    print(f"  RESULTS AT 20K PORTFOLIO:")
    print(f"  {'Config':<40} │ {'4/6':>6} {'5/6':>6} {'6/6':>6} │ {'Eff':>8} │ {'Lift':>6}")
    print(f"  {'─'*40} │ {'─'*6} {'─'*6} {'─'*6} │ {'─'*8} │ {'─'*6}")
    
    all_rows = []
    for key in results:
        r = results[key][20000]
        if isinstance(key, tuple):
            if len(key) == 4:
                label = f"P{key[0]}/{key[1]}/{key[2]}+SORT"
            else:
                label = f"P{key[0]}/{key[1]}/{key[2]}"
        else:
            label = key.upper()
        
        p4 = r['four'] / n_test * 100
        p5 = r['five'] / n_test * 100
        p6 = r['six'] / n_test * 100
        eff = r['six'] / 20  # per 1K
        
        # Compute expected
        if isinstance(key, tuple):
            ps = key[0]
            pc = contain.get(ps, 0) / n_test
            cpk = math.comb(ps, PICK)
            expected = pc * 20000 / cpk * n_test
        else:
            expected = 20000 / math.comb(45, 6) * n_test
        
        lift = r['six'] / expected if expected > 0 else 0
        all_rows.append((label, r, eff, lift, key))
    
    all_rows.sort(key=lambda x: (-x[1]['six'], -x[2]))
    
    for label, r, eff, lift, key in all_rows:
        p4 = r['four'] / n_test * 100
        p5 = r['five'] / n_test * 100
        p6 = r['six'] / n_test * 100
        bar = '█' * r['six']
        print(f"  {label:<40} │ {p4:5.1f}% {p5:5.1f}% {p6:5.2f}% │ {eff:7.3f}/K │ {lift:5.2f}x  {bar}")
    
    # Best config at each portfolio size
    print(f"\n{'─'*90}")
    print(f"  BEST CONFIG PER PORTFOLIO SIZE:")
    for pt in ALL_PORTS:
        ranking = [(k, results[k][pt]['six'], results[k][pt]['five']) for k in results]
        ranking.sort(key=lambda x: (-x[1], -x[2]))
        top = ranking[0]
        k, six, five = top
        if isinstance(k, tuple):
            if len(k) == 4:
                label = f"P{k[0]}/{k[1]}/{k[2]}+SORT"
            else:
                label = f"P{k[0]}/{k[1]}/{k[2]}"
        else:
            label = "RANDOM"
        eff = six / (pt/1000)
        print(f"    @{pt//1000:>2}K: {label:<35} → {six:>3} hits 6/6, {five:>3} hits 5/6 (eff={eff:.3f}/K)")
    
    # Top 5 overall tournament rankings
    print(f"\n{'═'*90}")
    print(f"  🏆 TOURNAMENT FINAL STANDINGS (by 6/6 @20K, tiebreak by 5/6):")
    print(f"{'═'*90}")
    
    champion = all_rows[0]
    for i, (label, r, eff, lift, key) in enumerate(all_rows[:10]):
        medal = ['🥇','🥈','🥉','4️⃣','5️⃣','6️⃣','7️⃣','8️⃣','9️⃣','🔟'][i]
        print(f"  {medal} {label:<38} 6/6={r['six']:>3}  5/6={r['five']:>3}  eff={eff:.3f}/K  lift={lift:.2f}x")
    
    # Random comparison
    rand_r = results['random'][20000]
    print(f"\n  📊 RANDOM BASELINE @20K: 6/6={rand_r['six']}, 5/6={rand_r['five']}")
    print(f"  📈 CHAMPION vs RANDOM: {champion[1]['six']}x / {max(1,rand_r['six'])} = "
          f"{champion[1]['six']/max(1,rand_r['six']):.1f}x better")
    
    # Multi-portfolio analysis for champion
    print(f"\n{'─'*90}")
    print(f"  CHAMPION ({champion[0]}) ACROSS PORTFOLIO SIZES:")
    champ_key = champion[4]
    for pt in ALL_PORTS:
        r = results[champ_key][pt]
        eff = r['six'] / (pt/1000)
        print(f"    @{pt//1000:>2}K: 6/6={r['six']:>3}  5/6={r['five']:>3}  4/6={r['four']:>3}  eff={eff:.3f}/K")
    
    # Save results
    output = {
        'version': 'V19 Focused Tournament',
        'n_test': n_test,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'phase1_results': {str(ps): {
            'contain': contain[ps],
            'contain_pct': round(contain[ps]/n_test*100, 2),
            'cpk': math.comb(ps, PICK),
        } for ps in TEST_POOLS},
        'random_baseline': {str(pt): {
            'six': results['random'][pt]['six'],
            'five': results['random'][pt]['five'],
        } for pt in ALL_PORTS},
        'top10': [{
            'rank': i+1,
            'config': str(all_rows[i][0]),
            'six': all_rows[i][1]['six'],
            'five': all_rows[i][1]['five'],
            'eff': round(all_rows[i][2], 3),
            'lift': round(all_rows[i][3], 3),
        } for i in range(min(10, len(all_rows)))],
        'champion': {
            'config': str(champion[0]),
            'results': {str(pt): {
                'six': results[champ_key][pt]['six'],
                'five': results[champ_key][pt]['five'],
                'four': results[champ_key][pt]['four'],
            } for pt in ALL_PORTS},
        },
        'elapsed_min': round(elapsed/60, 1),
    }
    path = os.path.join(os.path.dirname(__file__), 'models', 'v19_tournament.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {path}")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"{'═'*90}")


if __name__ == '__main__':
    run()
