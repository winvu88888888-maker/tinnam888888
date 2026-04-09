"""
V15 SUPERNOVA v7 — MEGA PORT + POOL 40
========================================
V6 showed: Pool 40 × Port 5000 = 2 hits 6/6 (0.16%)
Theory: Pool 40 containment = 47%, C(40,6) = 3.8M
  Port 5000 = 0.13% coverage → only 0.06% of contained draws hit 6/6

To increase: Need MORE tickets! Test 10K, 20K, 30K, 50K.
Port 50K = 50000/3838380 = 1.3% coverage × 47% containment = 0.6% expected 6/6

FAST approach: Generate all tickets ONCE, then check all draws.
"""
import sys, os, math, json, time, random
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper.data_manager import get_mega645_numbers

MAX = 45; PICK = 6
random.seed(42)


def compute_signals(data, at_index):
    relevant = data[:at_index]; n = len(relevant)
    if n < 50: return {num: PICK/MAX for num in range(1,MAX+1)}
    last = set(relevant[-1][:PICK]); base_p = PICK/MAX
    scores = {num: 0.0 for num in range(1,MAX+1)}
    
    for w, wt in [(3,4),(5,3),(10,2),(20,1.5),(50,1)]:
        if n < w: continue
        fc = Counter()
        for d in relevant[-w:]:
            for num in d[:PICK]: fc[num] += 1
        for num in range(1,MAX+1): scores[num] += fc.get(num,0)/w * wt
    
    follow = defaultdict(Counter); pc = Counter()
    for i in range(n-1):
        for p in relevant[i][:PICK]:
            pc[p] += 1
            for nx in relevant[i+1][:PICK]: follow[p][nx] += 1
    for num in range(1,MAX+1):
        tf = sum(follow[p].get(num,0) for p in last)
        tp = sum(pc[p] for p in last)
        if tp > 0: scores[num] += (tf/tp/base_p-1)*3
    
    ls_idx = {}; gs, gn = defaultdict(float), defaultdict(int)
    for i, d in enumerate(relevant):
        for num in d[:PICK]:
            if num in ls_idx: gs[num] += i-ls_idx[num]; gn[num] += 1
            ls_idx[num] = i
    for num in range(1,MAX+1):
        if gn[num] < 5: continue
        mg = gs[num]/gn[num]; cg = n-ls_idx.get(num,0)
        if cg > mg*1.3: scores[num] += min((cg/mg-1)*1.5, 3)
    
    knn = Counter()
    for i in range(n-2):
        sim = len(set(relevant[i][:PICK]) & last)
        if sim >= 2:
            for num in relevant[i+1][:PICK]: knn[num] += sim*sim
    mx = max(knn.values()) if knn else 1
    for num in range(1,MAX+1): scores[num] += knn.get(num,0)/mx*2.5
    
    if n >= 10:
        l1, l2 = set(relevant[-1][:PICK]), set(relevant[-2][:PICK])
        bc, ec = Counter(), Counter(); tb, te = 0, 0
        for i in range(2,n):
            p2, p1, cu = set(relevant[i-2][:PICK]), set(relevant[i-1][:PICK]), set(relevant[i][:PICK])
            for num in cu:
                if num in p2 and num in p1: bc[num] += 1
                elif num in p2 or num in p1: ec[num] += 1
            for num in range(1,MAX+1):
                if num in p2 and num in p1: tb += 1
                elif num in p2 or num in p1: te += 1
        for num in range(1,MAX+1):
            if num in l1 and num in l2: p = bc.get(num,0)/max(tb/MAX,1)
            elif num in l1 or num in l2: p = ec.get(num,0)/max(te/MAX,1)
            else: p = 0
            scores[num] += (p-base_p)*5
    
    pf = Counter()
    for d in relevant[-200:]:
        for p in combinations(sorted(d[:PICK]), 2): pf[p] += 1
    for num in range(1,MAX+1):
        scores[num] += sum(pf.get(tuple(sorted([p,num])),0) for p in last)*0.05
    tf = Counter()
    for d in relevant[-150:]:
        for t in combinations(sorted(d[:PICK]),3): tf[t] += 1
    for t, c in tf.most_common(500):
        if c < 2: break
        ts = set(t)
        if len(ts & last) == 2: scores[(ts-last).pop()] += c*0.3
    
    if n >= 100:
        rw = min(100, n//3)
        fr, fo = Counter(), Counter()
        for d in relevant[-rw:]:
            for num in d[:PICK]: fr[num] += 1
        for d in relevant[:-rw]:
            for num in d[:PICK]: fo[num] += 1
        for num in range(1,MAX+1):
            scores[num] += (fr.get(num,0)/rw - fo.get(num,0)/max(n-rw,1))*10
    
    ww = min(100, n)
    iai, ic = Counter(), Counter()
    for i in range(max(0,n-ww), n-1):
        curr, nxt = set(relevant[i][:PICK]), set(relevant[i+1][:PICK])
        for num in curr:
            ic[num] += 1
            if num in nxt: iai[num] += 1
    for num in range(1,MAX+1):
        if num in last and ic[num] > 5:
            scores[num] += (iai[num]/ic[num]-base_p)*3
    
    return scores


def build_pool(data, at_index, scores, max_pool):
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


def fast_gen(pool, scores, n_tickets):
    """ULTRA-FAST weighted portfolio."""
    pool_list = list(pool)
    if len(pool_list) < PICK: return []
    
    ranked = sorted(pool_list, key=lambda x: -scores.get(x,0))
    w_pool = []
    for i, num in enumerate(ranked):
        w_pool.extend([num] * max(1, len(ranked)-i))
    
    selected = set()
    attempts = 0
    max_attempts = n_tickets * 15
    while len(selected) < n_tickets and attempts < max_attempts:
        attempts += 1
        picked = set()
        while len(picked) < PICK:
            picked.add(random.choice(w_pool))
        if len(picked) < PICK: continue
        combo = tuple(sorted(picked))
        if combo not in selected:
            selected.add(combo)
    
    return list(selected)


def run():
    data = get_mega645_numbers()
    N = len(data); t0 = time.time()
    
    print("="*80)
    print("  🌟 V15 SUPERNOVA v7 — MEGA PORTFOLIO (10K-50K tickets)")
    print(f"  {N} draws | Pool 40 + Port [5K, 10K, 20K, 30K, 50K]")
    print("="*80)
    
    WARMUP = 200; POOL_MAX = 40
    PORT_SIZES = [5000, 10000, 20000, 30000, 50000]
    n_test = N - WARMUP
    
    contain = 0
    results = {pt: {'hits': [], 'six': 0, 'five': 0, 'six_detail': []} for pt in PORT_SIZES}
    
    print(f"  Walk-forward: {n_test} | Pool: {POOL_MAX}")
    print(f"{'━'*80}")
    
    for ti in range(n_test):
        te = WARMUP + ti
        actual = set(data[te][:PICK])
        actual_tuple = tuple(sorted(actual))
        
        sc = compute_signals(data, te)
        pool = build_pool(data, te, sc, POOL_MAX)
        pool_set = set(pool)
        
        if len(actual & pool_set) >= 6:
            contain += 1
        
        # Generate max portfolio ONCE
        portfolio = fast_gen(pool, sc, max(PORT_SIZES))
        
        for pt in PORT_SIZES:
            port = portfolio[:pt]
            if port:
                # Check if actual combo is IN portfolio (exact 6/6)
                has_6 = actual_tuple in port
                if has_6:
                    best = 6
                else:
                    # Check best match (sample if too large)
                    if len(port) <= 10000:
                        best = max(len(actual & set(c)) for c in port)
                    else:
                        # Sample 10K for speed
                        sample = random.sample(port, 10000)
                        best = max(len(actual & set(c)) for c in sample)
                        # Also check top-scored combos
                        top = port[:1000]
                        best = max(best, max(len(actual & set(c)) for c in top))
            else:
                best = 0
            
            results[pt]['hits'].append(best)
            if best >= 6:
                results[pt]['six'] += 1
                results[pt]['six_detail'].append((te, sorted(actual)))
            if best >= 5:
                results[pt]['five'] += 1
        
        if (ti+1) % 200 == 0:
            el = time.time()-t0
            eta = el/(ti+1)*(n_test-ti-1)
            parts = []
            for pt in PORT_SIZES:
                h6 = results[pt]['six']; h5 = results[pt]['five']
                parts.append(f"{pt//1000}K:6={h6}")
            print(f"  [{ti+1}/{n_test}] PC={contain} {' '.join(parts)} | {el:.0f}s ETA:{eta:.0f}s")
            sys.stdout.flush()
    
    elapsed = time.time() - t0
    
    print(f"\n{'═'*80}")
    print(f"  🌟 V15 v7 RESULTS ({n_test} tests, {elapsed:.0f}s)")
    print(f"{'═'*80}")
    
    print(f"\n  Pool contain: {contain}/{n_test} ({contain/n_test*100:.2f}%)")
    
    print(f"\n  {'Port':>8} | {'≥3/6':>8} {'≥4/6':>8} {'≥5/6':>8} {'6/6':>8} | {'5+ hits':>8} {'6 hits':>8}")
    print(f"  {'─'*8} | {'─'*8} {'─'*8} {'─'*8} {'─'*8} | {'─'*8} {'─'*8}")
    
    for pt in PORT_SIZES:
        bm = results[pt]['hits']
        p3 = sum(1 for m in bm if m>=3)/n_test*100
        p4 = sum(1 for m in bm if m>=4)/n_test*100
        p5 = sum(1 for m in bm if m>=5)/n_test*100
        p6 = sum(1 for m in bm if m>=6)/n_test*100
        print(f"  {pt:8d} | {p3:7.2f}% {p4:7.2f}% {p5:7.2f}% {p6:7.2f}% | "
              f"{results[pt]['five']:8d} {results[pt]['six']:8d}")
    
    best_pt = max(PORT_SIZES, key=lambda pt: results[pt]['six'])
    best_six = results[best_pt]['six']
    print(f"\n  🏆 BEST: Port {best_pt} → {best_six} 6/6 hits ({best_six/n_test*100:.3f}%)")
    
    if results[best_pt]['six_detail']:
        for idx, nums in results[best_pt]['six_detail']:
            print(f"    🎉 Draw #{idx}: {nums}")
    
    output = {
        'version': '15.0v7 MEGA', 'n_test': n_test,
        'pool_max': POOL_MAX, 'contain': contain,
        'results': {},
        'elapsed': round(elapsed,1),
    }
    for pt in PORT_SIZES:
        bm = results[pt]['hits']
        output['results'][str(pt)] = {
            'p3': round(sum(1 for m in bm if m>=3)/n_test*100,2),
            'p5': round(sum(1 for m in bm if m>=5)/n_test*100,2),
            'p6': round(sum(1 for m in bm if m>=6)/n_test*100,2),
            'six': results[pt]['six'], 'five': results[pt]['five'],
        }
    path = os.path.join(os.path.dirname(__file__), 'models', 'v15_supernova.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {path}")


if __name__ == '__main__':
    run()
