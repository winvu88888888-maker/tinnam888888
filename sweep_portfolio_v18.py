"""
V18 PORTFOLIO SIZE SWEEP — FIND THE SWEET SPOT
================================================
Tests V18 ensemble at granular portfolio sizes (1K to 50K).

Two strategies:
  A) Sequential slice (current V18 behavior)
  B) Score-sorted (rank combos by sum of signal scores, best first)

Goal: Find the minimum portfolio size that captures most 6/6 hits
with the best cost-efficiency (hits per 1000 tickets).
"""
import sys, os, math, time, random
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper.data_manager import get_mega645_numbers

MAX = 45; PICK = 6
random.seed(42); np.random.seed(42)

# ============================================================
# V18 ENGINE (exact copy from v18_hypernova.py)
# ============================================================

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


# ============================================================
# SWEEP ENGINE
# ============================================================

def run():
    data = get_mega645_numbers()
    N = len(data); t0 = time.time()
    
    # Granular portfolio sizes to test
    PORT_SIZES = [1000, 2000, 3000, 5000, 7000, 10000, 12000, 15000,
                  18000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]
    
    WARMUP = 200; POOL_MAX = 40
    n_test = N - WARMUP
    MAX_PORT = max(PORT_SIZES)  # 50K
    
    print("=" * 80)
    print("  🔍 V18 PORTFOLIO SIZE SWEEP — FINDING THE SWEET SPOT")
    print(f"  {N} draws | Pool: {POOL_MAX} | Max: {MAX_PORT:,}")
    print(f"  Testing {len(PORT_SIZES)} sizes: {[f'{p//1000}K' for p in PORT_SIZES]}")
    print("=" * 80)
    
    # Strategy A: Sequential (current V18 — combos in generation order)
    # Strategy B: Score-sorted (rank by sum of signal scores, best combos first)
    
    res_seq = {pt: {'six': 0, 'five': 0, 'hits': []} for pt in PORT_SIZES}
    res_sort = {pt: {'six': 0, 'five': 0, 'hits': []} for pt in PORT_SIZES}
    contain = 0
    
    print(f"\n  Walk-forward: {n_test} draws")
    print(f"{'━' * 80}")
    
    for ti in range(n_test):
        te = WARMUP + ti
        actual = set(data[te][:PICK])
        
        sc = compute_signals(data, te)
        pool = build_diverse_pool(data, te, sc, POOL_MAX)
        
        if len(actual & set(pool)) >= 6:
            contain += 1
        
        # Generate full 50K portfolio
        portfolio = fast_gen_ensemble(pool, sc, MAX_PORT)
        
        if not portfolio:
            for pt in PORT_SIZES:
                res_seq[pt]['hits'].append(0)
                res_sort[pt]['hits'].append(0)
            continue
        
        # Strategy B: sort by sum of scores (descending)
        portfolio_sorted = sorted(portfolio, key=lambda c: sum(sc.get(n,0) for n in c), reverse=True)
        
        # Evaluate at each portfolio size
        for pt in PORT_SIZES:
            # Strategy A: sequential slice
            port_a = portfolio[:min(pt, len(portfolio))]
            if port_a:
                best_a = max(len(actual & set(c)) for c in port_a)
            else:
                best_a = 0
            res_seq[pt]['hits'].append(best_a)
            if best_a >= 6: res_seq[pt]['six'] += 1
            if best_a >= 5: res_seq[pt]['five'] += 1
            
            # Strategy B: score-sorted slice
            port_b = portfolio_sorted[:min(pt, len(portfolio_sorted))]
            if port_b:
                best_b = max(len(actual & set(c)) for c in port_b)
            else:
                best_b = 0
            res_sort[pt]['hits'].append(best_b)
            if best_b >= 6: res_sort[pt]['six'] += 1
            if best_b >= 5: res_sort[pt]['five'] += 1
        
        if (ti + 1) % 100 == 0:
            el = time.time() - t0
            eta = el / (ti + 1) * (n_test - ti - 1)
            # Show key sizes
            keys = [5000, 15000, 30000, 50000]
            seq_str = " ".join(f"{pt//1000}K:{res_seq[pt]['six']}" for pt in keys)
            srt_str = " ".join(f"{pt//1000}K:{res_sort[pt]['six']}" for pt in keys)
            print(f"  [{ti+1:4d}/{n_test}] SEQ({seq_str}) SORT({srt_str}) "
                  f"PC={contain} | {el:.0f}s ETA:{eta/60:.1f}m")
            sys.stdout.flush()
    
    elapsed = time.time() - t0
    pc = contain / n_test
    cpk = math.comb(POOL_MAX, PICK)  # C(40,6)
    
    print(f"\n{'═' * 80}")
    print(f"  🔍 SWEEP RESULTS ({n_test} tests, {elapsed:.0f}s)")
    print(f"{'═' * 80}")
    print(f"\n  Pool containment: {contain}/{n_test} ({pc*100:.2f}%)")
    print(f"  C({POOL_MAX},{PICK}) = {cpk:,}")
    
    # ====== STRATEGY A: Sequential ======
    print(f"\n{'─' * 80}")
    print(f"  ┌── STRATEGY A: SEQUENTIAL SLICE (current V18) ──")
    print(f"  │ {'Port':>7} │ {'5/6':>6} {'6/6':>6} │ {'5+':>5} {'6':>5} │ {'Eff':>8} │ {'Expected':>8} │ {'Lift':>6}")
    print(f"  │ {'─'*7} │ {'─'*6} {'─'*6} │ {'─'*5} {'─'*5} │ {'─'*8} │ {'─'*8} │ {'─'*6}")
    for pt in PORT_SIZES:
        p5 = res_seq[pt]['five'] / n_test * 100
        p6 = res_seq[pt]['six'] / n_test * 100
        eff = res_seq[pt]['six'] / (pt / 1000)  # hits per 1K tickets
        expected = pc * pt / cpk * 100  # random expected %
        lift = p6 / expected if expected > 0 else 0
        bar = '█' * res_seq[pt]['six']
        print(f"  │ {pt:>7,} │ {p5:5.2f}% {p6:5.2f}% │ {res_seq[pt]['five']:5d} {res_seq[pt]['six']:5d} │ "
              f"{eff:7.3f}/K │ {expected:7.3f}% │ {lift:5.1f}x  {bar}")
    print(f"  └──")
    
    # ====== STRATEGY B: Score-sorted ======
    print(f"\n  ┌── STRATEGY B: SCORE-SORTED (best combos first) ──")
    print(f"  │ {'Port':>7} │ {'5/6':>6} {'6/6':>6} │ {'5+':>5} {'6':>5} │ {'Eff':>8} │ {'Expected':>8} │ {'Lift':>6}")
    print(f"  │ {'─'*7} │ {'─'*6} {'─'*6} │ {'─'*5} {'─'*5} │ {'─'*8} │ {'─'*8} │ {'─'*6}")
    for pt in PORT_SIZES:
        p5 = res_sort[pt]['five'] / n_test * 100
        p6 = res_sort[pt]['six'] / n_test * 100
        eff = res_sort[pt]['six'] / (pt / 1000)
        expected = pc * pt / cpk * 100
        lift = p6 / expected if expected > 0 else 0
        bar = '█' * res_sort[pt]['six']
        print(f"  │ {pt:>7,} │ {p5:5.2f}% {p6:5.2f}% │ {res_sort[pt]['five']:5d} {res_sort[pt]['six']:5d} │ "
              f"{eff:7.3f}/K │ {expected:7.3f}% │ {lift:5.1f}x  {bar}")
    print(f"  └──")
    
    # ====== COMPARISON TABLE ======
    print(f"\n{'─' * 80}")
    print(f"  📊 DIRECT COMPARISON (6/6 hits)")
    print(f"  {'Port':>7} │ {'Sequential':>10} │ {'Score-Sort':>10} │ {'Winner':>10} │ {'Δ':>5}")
    print(f"  {'─'*7} │ {'─'*10} │ {'─'*10} │ {'─'*10} │ {'─'*5}")
    for pt in PORT_SIZES:
        s = res_seq[pt]['six']
        r = res_sort[pt]['six']
        w = "SEQ" if s > r else ("SORT" if r > s else "TIE")
        d = r - s
        print(f"  {pt:>7,} │ {s:>10} │ {r:>10} │ {w:>10} │ {d:>+5}")
    
    # ====== SWEET SPOT ANALYSIS ======
    print(f"\n{'─' * 80}")
    print(f"  🎯 SWEET SPOT ANALYSIS")
    best_strat = res_sort  # or decide based on data
    best_label = "SORT"
    
    # Compare both and pick best
    for pt in PORT_SIZES:
        if res_seq[pt]['six'] > res_sort[pt]['six']:
            best_strat = res_seq
            best_label = "SEQ"
            break
    
    # Actually pick best per size
    print(f"\n  Looking for: minimum portfolio with maximum efficiency")
    print(f"  (Efficiency = 6/6 hits per 1000 tickets)")
    print()
    
    # Find the knee point: where adding more tickets gives diminishing returns
    best_eff = 0
    best_size = 0
    for pt in PORT_SIZES:
        s_eff = res_seq[pt]['six'] / (pt / 1000)
        r_eff = res_sort[pt]['six'] / (pt / 1000)
        best = max(s_eff, r_eff)
        strat = "SEQ" if s_eff >= r_eff else "SORT"
        hits = res_seq[pt]['six'] if strat == "SEQ" else res_sort[pt]['six']
        marker = " ◀◀◀ BEST" if best > best_eff else ""
        if best > best_eff:
            best_eff = best
            best_size = pt
        print(f"    {pt//1000:>3}K: {best:.4f} hits/K ({strat}, {hits} hits){marker}")
    
    # Marginal analysis: hits gained per additional 1K tickets
    print(f"\n  Marginal analysis (additional hits per +size increase):")
    for i in range(1, len(PORT_SIZES)):
        prev_pt = PORT_SIZES[i-1]
        pt = PORT_SIZES[i]
        delta_tickets = pt - prev_pt
        
        prev_best = max(res_seq[prev_pt]['six'], res_sort[prev_pt]['six'])
        curr_best = max(res_seq[pt]['six'], res_sort[pt]['six'])
        delta_hits = curr_best - prev_best
        marginal = delta_hits / (delta_tickets / 1000) if delta_tickets > 0 else 0
        
        bar = '▓' * delta_hits if delta_hits > 0 else '░'
        print(f"    {prev_pt//1000:>3}K → {pt//1000:>3}K: +{delta_hits:>2} hits "
              f"(+{delta_tickets//1000}K tickets, marginal={marginal:.3f}/K) {bar}")
    
    # Final recommendation
    print(f"\n{'═' * 80}")
    max_hits_seq = res_seq[50000]['six']
    max_hits_sort = res_sort[50000]['six']
    max_hits = max(max_hits_seq, max_hits_sort)
    
    # Find smallest portfolio that captures >= 80% of max hits
    for threshold_pct in [90, 80, 70, 60, 50]:
        threshold = int(max_hits * threshold_pct / 100)
        for pt in PORT_SIZES:
            best_at_pt = max(res_seq[pt]['six'], res_sort[pt]['six'])
            if best_at_pt >= threshold:
                cost_ratio = pt / 50000 * 100
                print(f"  ≥{threshold_pct}% of max ({threshold}+ of {max_hits} hits): "
                      f"need {pt//1000}K tickets ({cost_ratio:.0f}% cost)")
                break
    
    print(f"\n  💡 RECOMMENDATION: Best efficiency at {best_size//1000}K "
          f"({best_eff:.4f} hits/K tickets)")
    print(f"{'═' * 80}")


if __name__ == '__main__':
    run()
