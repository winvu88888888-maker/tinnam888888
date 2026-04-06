"""
V18 HYPERNOVA FAST — OPTIMIZED FOR SPEED
==========================================
Key insight: V15 Supernova IS the real baseline (8 hits 6/6 at 50K, no leakage).
V16 tournament had DATA LEAKAGE (56 hits was FAKE).

Goal: Beat V15's 8 hits at 50K with better signals + diverse weights.

FAST DESIGN:
- Compute signals + pool ONCE per draw
- Generate ONE large portfolio using ensemble weights
- Test at 5K, 10K, 20K, 50K cuts of same portfolio
- Pool 40 (proven best containment) + Pool 35 (comparison)
"""
import sys, os, math, json, time, random
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper.data_manager import get_mega645_numbers

MAX = 45; PICK = 6
random.seed(42); np.random.seed(42)


def compute_signals(data, at_index):
    """6-signal engine."""
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


def build_diverse_pool(data, at_index, scores, max_pool):
    """11 diverse sub-pools → union → top by score."""
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


def fast_gen_ensemble(pool, scores, n_tickets):
    """FAST ensemble: 1/3 linear + 1/3 exp + 1/3 stepper."""
    n = len(pool)
    if n < PICK: return []
    ranked = sorted(pool, key=lambda x: -scores.get(x,0))
    
    # Build 3 w_pools
    wp_lin = []
    wp_exp = []
    wp_stp = []
    for i, num in enumerate(ranked):
        wp_lin.extend([num] * max(1, n-i))
        wp_exp.extend([num] * max(1, min(int(2**((n-i)/4)), 300)))
        wp_stp.extend([num] * (10 if i < 12 else 1))
    
    w_pools = [wp_lin, wp_exp, wp_stp]
    per = n_tickets // 3
    
    selected = set()
    for j, wp in enumerate(w_pools):
        target = per if j < 2 else n_tickets - 2*per
        att = 0
        while len(selected) < (j+1)*per and att < target*15:
            att += 1
            picked = set()
            while len(picked) < PICK:
                picked.add(random.choice(wp))
            selected.add(tuple(sorted(picked)))
    
    return list(selected)


def run():
    data = get_mega645_numbers()
    N = len(data); t0 = time.time()
    
    print("="*80)
    print("  🚀 V18 HYPERNOVA FAST — OPTIMIZED")
    print(f"  {N} draws | 6sig | Diverse pool 40 | Ensemble gen")
    print("="*80)
    
    WARMUP = 200; POOL_MAX = 40
    PORT_SIZES = [5000, 10000, 20000, 50000]
    n_test = N - WARMUP
    
    results = {pt: {'hits': [], 'six': 0, 'five': 0, 'det': []} for pt in PORT_SIZES}
    contain = 0
    
    # Also track V15-style (same diverse pool, linear only) for comparison
    results_v15 = {pt: {'hits': [], 'six': 0, 'five': 0} for pt in PORT_SIZES}
    
    print(f"\n  Walk-forward: {n_test} draws | Pool: {POOL_MAX}")
    print(f"  Ports: {PORT_SIZES}")
    print(f"{'━'*80}")
    
    for ti in range(n_test):
        te = WARMUP + ti
        actual = set(data[te][:PICK])
        
        sc = compute_signals(data, te)
        pool = build_diverse_pool(data, te, sc, POOL_MAX)
        
        if len(actual & set(pool)) >= 6:
            contain += 1
        
        # Generate ONE large ensemble portfolio
        portfolio = fast_gen_ensemble(pool, sc, max(PORT_SIZES))
        
        # Also generate V15-style (linear only) for comparison
        ranked_pool = sorted(pool, key=lambda x: -sc.get(x,0))
        n_p = len(ranked_pool)
        wp_v15 = []
        for i, num in enumerate(ranked_pool):
            wp_v15.extend([num] * max(1, n_p-i))
        
        v15_port = set()
        att = 0
        while len(v15_port) < max(PORT_SIZES) and att < max(PORT_SIZES)*15:
            att += 1
            picked = set()
            while len(picked) < PICK:
                picked.add(random.choice(wp_v15))
            v15_port.add(tuple(sorted(picked)))
        v15_list = list(v15_port)
        
        # Evaluate V18 ensemble
        for pt in PORT_SIZES:
            port = portfolio[:pt] if pt < len(portfolio) else portfolio
            if port:
                best = max(len(actual & set(c)) for c in port)
            else:
                best = 0
            results[pt]['hits'].append(best)
            if best >= 6:
                results[pt]['six'] += 1
                results[pt]['det'].append((te, sorted(actual)))
            if best >= 5:
                results[pt]['five'] += 1
        
        # Evaluate V15 linear
        for pt in PORT_SIZES:
            port = v15_list[:pt] if pt < len(v15_list) else v15_list
            if port:
                best = max(len(actual & set(c)) for c in port)
            else:
                best = 0
            results_v15[pt]['hits'].append(best)
            if best >= 6: results_v15[pt]['six'] += 1
            if best >= 5: results_v15[pt]['five'] += 1
        
        if (ti+1) % 100 == 0:
            el = time.time()-t0
            eta = el/(ti+1)*(n_test-ti-1)
            parts_v18 = [f"{pt//1000}K:6={results[pt]['six']}" for pt in PORT_SIZES]
            parts_v15 = [f"{pt//1000}K:6={results_v15[pt]['six']}" for pt in PORT_SIZES]
            print(f"  [{ti+1:4d}/{n_test}] PC={contain} V18({' '.join(parts_v18)}) "
                  f"V15({' '.join(parts_v15)}) | {el:.0f}s ETA:{eta/60:.1f}m")
            sys.stdout.flush()
    
    elapsed = time.time() - t0
    
    print(f"\n{'═'*80}")
    print(f"  🚀 V18 RESULTS ({n_test} tests, {elapsed:.0f}s)")
    print(f"{'═'*80}")
    
    print(f"\n  Pool containment: {contain}/{n_test} ({contain/n_test*100:.2f}%)")
    
    # V18 Ensemble results
    print(f"\n  ┌── V18 ENSEMBLE (6sig + diverse pool + 3-weight ensemble) ──")
    print(f"  │ {'Port':>7} │ {'≥3/6':>7} {'≥4/6':>7} {'≥5/6':>7} {'6/6':>7} │ {'5+':>5} {'6':>5}")
    print(f"  │ {'─'*7} │ {'─'*7} {'─'*7} {'─'*7} {'─'*7} │ {'─'*5} {'─'*5}")
    for pt in PORT_SIZES:
        bm = results[pt]['hits']
        p3 = sum(1 for m in bm if m>=3)/n_test*100
        p4 = sum(1 for m in bm if m>=4)/n_test*100
        p5 = sum(1 for m in bm if m>=5)/n_test*100
        p6 = sum(1 for m in bm if m>=6)/n_test*100
        print(f"  │ {pt:>7,} │ {p3:6.2f}% {p4:6.2f}% {p5:6.2f}% {p6:6.2f}% │ {results[pt]['five']:5d} {results[pt]['six']:5d}")
    print(f"  └──")
    
    # V15-style linear results (same pool, same signals, just linear weights)
    print(f"\n  ┌── V15-STYLE LINEAR (same pool, linear weights only) ──")
    print(f"  │ {'Port':>7} │ {'≥3/6':>7} {'≥4/6':>7} {'≥5/6':>7} {'6/6':>7} │ {'5+':>5} {'6':>5}")
    print(f"  │ {'─'*7} │ {'─'*7} {'─'*7} {'─'*7} {'─'*7} │ {'─'*5} {'─'*5}")
    for pt in PORT_SIZES:
        bm = results_v15[pt]['hits']
        p3 = sum(1 for m in bm if m>=3)/n_test*100
        p4 = sum(1 for m in bm if m>=4)/n_test*100
        p5 = sum(1 for m in bm if m>=5)/n_test*100
        p6 = sum(1 for m in bm if m>=6)/n_test*100
        print(f"  │ {pt:>7,} │ {p3:6.2f}% {p4:6.2f}% {p5:6.2f}% {p6:6.2f}% │ {results_v15[pt]['five']:5d} {results_v15[pt]['six']:5d}")
    print(f"  └──")
    
    # Compare vs V15 Supernova baseline
    print(f"\n  📊 COMPARISON:")
    print(f"    V15 Supernova (Pool 40, linear, 3sig): 8 hits 6/6 @ 50K (0.62%)")
    print(f"    V18 Ensemble  (Pool 40, ensemble, 6sig): {results[50000]['six']} hits 6/6 @ 50K ({results[50000]['six']/n_test*100:.2f}%)")
    print(f"    V15-style     (Pool 40, linear, 6sig): {results_v15[50000]['six']} hits 6/6 @ 50K ({results_v15[50000]['six']/n_test*100:.2f}%)")
    
    # Math check
    cpk = math.comb(POOL_MAX, PICK)
    pc = contain / n_test
    for pt in PORT_SIZES:
        expected = pc * pt / cpk * 100
        actual_v18 = results[pt]['six'] / n_test * 100
        actual_v15 = results_v15[pt]['six'] / n_test * 100
        print(f"    @{pt//1000}K: Expected={expected:.3f}%, V18={actual_v18:.3f}%, V15={actual_v15:.3f}%")
    
    # 6/6 details
    for pt in [50000, 20000]:
        det = results[pt]['det']
        if det:
            print(f"\n  🎉 V18 6/6 DRAWS @ {pt//1000}K:")
            for idx, nums in det[:10]:
                print(f"    Draw #{idx}: {nums}")
    
    # Save
    output = {
        'version': '18.0 FAST FINAL',
        'n_test': n_test, 'pool_max': POOL_MAX,
        'contain': contain, 'contain_pct': round(pc*100, 2),
        'v18_ensemble': {str(pt): {
            'p5': round(sum(1 for m in results[pt]['hits'] if m>=5)/n_test*100, 2),
            'p6': round(sum(1 for m in results[pt]['hits'] if m>=6)/n_test*100, 2),
            'six': results[pt]['six'], 'five': results[pt]['five'],
        } for pt in PORT_SIZES},
        'v15_linear': {str(pt): {
            'p5': round(sum(1 for m in results_v15[pt]['hits'] if m>=5)/n_test*100, 2),
            'p6': round(sum(1 for m in results_v15[pt]['hits'] if m>=6)/n_test*100, 2),
            'six': results_v15[pt]['six'], 'five': results_v15[pt]['five'],
        } for pt in PORT_SIZES},
        'elapsed': round(elapsed, 1),
    }
    path = os.path.join(os.path.dirname(__file__), 'models', 'v18_hypernova.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {path}")


if __name__ == '__main__':
    run()
