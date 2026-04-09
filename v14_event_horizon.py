"""
V14 EVENT HORIZON — MAXIMUM COVERAGE ASSAULT
=============================================
V13 breakthrough: 1 draw 6/6, pool containment 20.45% (263 draws).
Pool 30 + safety ring = best strategy.
V13 problem: confidence detector gave MINIMAL 99.9% → not useful.

V14 STRATEGY:
  1. KEEP pool 30 + safety 5 = 35 (proven optimal from V13)
  2. PORTFOLIO SIZE: 3000-5000 (V13 had 2000, need more for 6/6)
  3. SMARTER COVERAGE: Target C(35,6)=1,623,160. 
     With 5000 tickets = 0.31% coverage
     But with WEIGHTED scoring: concentrate on high-signal combos
  4. DROP failed confidence detector
  5. ADD: Combo-level scoring (score combos, not just numbers)
  6. ADD: Portfolio diversification via pair/triple balance
"""
import sys, os, math, json, time
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper.data_manager import get_mega645_numbers, get_mega645_all

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

MAX = 45
PICK = 6


# ================================================================
# UNIFIED SIGNAL ENGINE (battle-tested, compact)
# ================================================================
def compute_signals(data, at_index):
    """Returns combined per-number scores from all weapons."""
    relevant = data[:at_index]
    n = len(relevant)
    if n < 50:
        return np.ones(MAX) * PICK/MAX
    
    last = set(relevant[-1][:PICK])
    base_p = PICK / MAX
    scores = np.zeros(MAX)
    
    # 1. Multi-scale frequency (proven winner)
    for w, wt in [(3, 4.0), (5, 3.0), (10, 2.0), (20, 1.5), (50, 1.0)]:
        if n < w: continue
        for num in range(1, MAX+1):
            f = sum(1 for d in relevant[-w:] if num in d[:PICK]) / w
            scores[num-1] += f * wt
    
    # 2. Transition matrix
    follow = defaultdict(Counter)
    pc = Counter()
    for i in range(n-1):
        for p in relevant[i][:PICK]:
            pc[p] += 1
            for nx in relevant[i+1][:PICK]:
                follow[p][nx] += 1
    for num in range(1, MAX+1):
        tf = sum(follow[p].get(num, 0) for p in last)
        tp = sum(pc[p] for p in last)
        if tp > 0:
            scores[num-1] += (tf/tp / base_p - 1) * 3
    
    # 3. Gap timing  
    for num in range(1, MAX+1):
        apps = [i for i, d in enumerate(relevant) if num in d[:PICK]]
        if len(apps) < 5: continue
        gaps = [apps[j+1]-apps[j] for j in range(len(apps)-1)]
        mg, sg = np.mean(gaps), np.std(gaps)
        cg = n - apps[-1]
        z = (cg - mg) / sg if sg > 0 else 0
        if z > 0.5: scores[num-1] += z * 1.5
        elif z < -1: scores[num-1] -= 1
    
    # 4. KNN similarity attention
    knn = Counter()
    for i in range(n-2):
        sim = len(set(relevant[i][:PICK]) & last)
        if sim >= 2:
            for num in relevant[i+1][:PICK]:
                knn[num] += sim ** 2
    mx = max(knn.values()) if knn else 1
    for num in range(1, MAX+1):
        scores[num-1] += knn.get(num, 0) / mx * 2.5
    
    # 5. Markov Order-2
    if n >= 10:
        last1 = set(relevant[-1][:PICK])
        last2 = set(relevant[-2][:PICK])
        both_c, either_c = Counter(), Counter()
        tb, te = 0, 0
        for i in range(2, n):
            p2, p1, cu = set(relevant[i-2][:PICK]), set(relevant[i-1][:PICK]), set(relevant[i][:PICK])
            for num in range(1, MAX+1):
                ip2, ip1 = num in p2, num in p1
                if ip2 and ip1: tb += 1; both_c[num] += int(num in cu)
                elif ip2 or ip1: te += 1; either_c[num] += int(num in cu)
        for num in range(1, MAX+1):
            if num in last1 and num in last2: p = both_c[num] / max(tb/MAX, 1)
            elif num in last1 or num in last2: p = either_c[num] / max(te/MAX, 1)
            else: p = 0
            scores[num-1] += (p - base_p) * 5
    
    # 6. Co-occurrence + triplet completion
    pf = Counter()
    for d in relevant[-200:]:
        for p in combinations(sorted(d[:PICK]), 2):
            pf[p] += 1
    for num in range(1, MAX+1):
        scores[num-1] += sum(pf.get(tuple(sorted([p, num])), 0) for p in last) * 0.05
    
    tf = Counter()
    for d in relevant[-150:]:
        for t in combinations(sorted(d[:PICK]), 3):
            tf[t] += 1
    for t, c in tf.most_common(500):
        if c < 2: break
        ts = set(t)
        ov = ts & last
        if len(ov) == 2:
            scores[(ts-last).pop() - 1] += c * 0.3
    
    # 7. Ball wear drift (machine bias)
    if n >= 100:
        rw = min(100, n//3)
        fr = np.zeros(MAX)
        for d in relevant[-rw:]:
            for num in d[:PICK]: fr[num-1] += 1
        fr /= rw
        fo = np.zeros(MAX)
        for d in relevant[:-rw]:
            for num in d[:PICK]: fo[num-1] += 1
        fo /= max(n-rw, 1)
        drift = fr - fo
        mx = np.max(np.abs(drift))
        if mx > 0: scores += (drift/mx) * 2
    
    # 8. Decade balance boost
    freq50 = np.zeros(MAX)
    for d in relevant[-50:]:
        for num in d[:PICK]: freq50[num-1] += 1
    freq50 /= 50
    scores += freq50 * 3
    
    return scores


# ================================================================
# META-STACKER (proven from V13)
# ================================================================
class MetaStacker:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
    
    def train(self, data, start, end):
        X, Y = [], []
        for idx in range(max(start, 100), end-1, 5):
            s = compute_signals(data, idx)
            actual = set(data[idx][:PICK])
            sn = self._normalize(s)
            for num in range(1, MAX+1):
                X.append([sn[num-1], sn[num-1]**2, 
                         np.mean(sn[max(0,num-3):num+2])])
                Y.append(1 if num in actual else 0)
        if len(X) < 100: return False
        X, Y = np.array(X), np.array(Y)
        Xs = self.scaler.fit_transform(X)
        self.model = GradientBoostingClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42)
        self.model.fit(Xs, Y)
        return True
    
    def predict(self, raw_scores):
        sn = self._normalize(raw_scores)
        if self.model is None: return sn
        X = []
        for num in range(MAX):
            X.append([sn[num], sn[num]**2,
                     np.mean(sn[max(0,num-2):num+3])])
        Xs = self.scaler.transform(np.array(X))
        probs = self.model.predict_proba(Xs)
        return np.array([p[1] if len(self.model.classes_)==2 else p[0] 
                        for p in probs])
    
    def _normalize(self, s):
        mn, mx = np.min(s), np.max(s)
        return (s-mn)/(mx-mn) if mx-mn > 0.001 else np.ones_like(s)*0.5


# ================================================================
# COMBO-LEVEL SCORER (NEW — score whole combos, not just numbers)
# ================================================================
def score_combo(combo, num_scores, pair_history, data_recent):
    """Score a combo based on number scores + pair affinity + properties."""
    score = sum(num_scores[n] for n in combo)
    
    # Pair affinity bonus (pairs that historically appear together)
    pair_bonus = 0
    for a, b in combinations(combo, 2):
        key = tuple(sorted([a, b]))
        pair_bonus += pair_history.get(key, 0)
    score += pair_bonus * 0.01
    
    # Property bonus (sum/range in typical zone)
    s = sum(combo)
    rng = combo[-1] - combo[0]
    odd = sum(1 for x in combo if x % 2 == 1)
    
    # Typical lottery draw: sum ~138, range ~30, odd 2-4
    sum_dist = abs(s - 138) / 50
    range_dist = abs(rng - 30) / 15
    odd_dist = abs(odd - 3) / 3
    score -= (sum_dist + range_dist + odd_dist) * 0.5
    
    return score


# ================================================================
# MASSIVE PORTFOLIO GENERATOR (optimized for 3000-5000 tickets)
# ================================================================
def generate_massive_portfolio(pool, num_scores, pair_history, data_recent,
                                n_tickets, constraints):
    """Generate large portfolio with maximum diversity and scoring."""
    pool = sorted(pool)
    pool_size = len(pool)
    if pool_size < PICK: return []
    
    total = math.comb(pool_size, PICK)
    
    # Strategy depends on total combos vs desired tickets
    if total <= n_tickets * 2:
        # Small pool — enumerate all, sort by score
        all_combos = []
        for combo in combinations(pool, PICK):
            if not _check(combo, constraints): continue
            sc = score_combo(combo, num_scores, pair_history, data_recent)
            all_combos.append((list(combo), sc))
        all_combos.sort(key=lambda x: -x[1])
        return [{'numbers': c, 'score': round(s,3)} for c, s in all_combos[:n_tickets]]
    
    # Large pool — hybrid: top scored + random diverse sampling
    n_top = n_tickets // 2
    n_diverse = n_tickets - n_top
    
    # Phase 1: Top scored combos (greedy with diversity)
    selected = []
    used_pairs = Counter()
    
    # Sample candidates efficiently
    candidates = _smart_sample(pool, num_scores, pair_history, data_recent,
                                min(n_tickets * 8, total // 2), constraints)
    candidates.sort(key=lambda x: -x[1])
    
    for combo, score in candidates:
        if len(selected) >= n_top: break
        if selected:
            check = selected[-min(40, len(selected)):]
            if max(len(set(combo) & set(s['numbers'])) for s in check) >= 5:
                continue
        selected.append({'numbers': combo, 'score': round(score, 3)})
        for p in combinations(combo, 2):
            used_pairs[p] += 1
    
    # Phase 2: Diverse random combos (maximize pair coverage)
    attempts = 0
    while len(selected) < n_tickets and attempts < n_tickets * 5:
        attempts += 1
        # Weighted random selection
        weights = np.array([num_scores.get(n, 1) for n in pool])
        weights = weights - weights.min() + 0.1
        weights /= weights.sum()
        idx = np.random.choice(len(pool), PICK, replace=False, p=weights)
        combo = sorted([pool[i] for i in idx])
        if not _check(tuple(combo), constraints): continue
        
        if selected:
            check = selected[-min(30, len(selected)):]
            if max(len(set(combo) & set(s['numbers'])) for s in check) >= 5:
                continue
        
        # Pair novelty bonus
        new_pairs = sum(1 for p in combinations(combo, 2) if used_pairs[p] == 0)
        sc = score_combo(combo, num_scores, pair_history, data_recent)
        
        selected.append({'numbers': combo, 'score': round(sc + new_pairs * 0.02, 3)})
        for p in combinations(combo, 2):
            used_pairs[p] += 1
    
    return selected


def _smart_sample(pool, num_scores, pair_history, data_recent, n, constraints):
    """Weighted random sampling of combos from pool."""
    pool_arr = np.array(pool)
    weights = np.array([num_scores.get(p, 1) for p in pool])
    weights = weights - weights.min() + 0.1
    weights /= weights.sum()
    
    candidates = []
    seen = set()
    attempts = 0
    while len(candidates) < n and attempts < n * 5:
        attempts += 1
        idx = np.random.choice(len(pool), PICK, replace=False, p=weights)
        combo = tuple(sorted(pool_arr[idx]))
        if combo in seen: continue
        seen.add(combo)
        if not _check(combo, constraints): continue
        sc = score_combo(list(combo), num_scores, pair_history, data_recent)
        candidates.append((list(combo), sc))
    return candidates


def _check(combo, c):
    s = sum(combo)
    if s < c.get('sum_lo', 0) or s > c.get('sum_hi', 999): return False
    r = combo[-1] - combo[0]
    if r < c.get('range_lo', 0) or r > c.get('range_hi', 999): return False
    o = sum(1 for x in combo if x % 2 == 1)
    if o < c.get('odd_lo', 0) or o > c.get('odd_hi', PICK): return False
    return True


def learn_constraints(data, at_index):
    recent = data[max(0, at_index-50):at_index]
    sums = [sum(sorted(d[:PICK])) for d in recent]
    odds = [sum(1 for x in d[:PICK] if x % 2 == 1) for d in recent]
    ranges = [max(d[:PICK]) - min(d[:PICK]) for d in recent]
    return {
        'sum_lo': int(np.percentile(sums, 5)),
        'sum_hi': int(np.percentile(sums, 95)),
        'odd_lo': max(0, int(np.percentile(odds, 5))),
        'odd_hi': min(PICK, int(np.percentile(odds, 95))),
        'range_lo': int(np.percentile(ranges, 5)),
        'range_hi': int(np.percentile(ranges, 95)),
    }


# ================================================================
# WALK-FORWARD MASTER
# ================================================================
def run_v14():
    data = get_mega645_numbers()
    all_rows = get_mega645_all()
    N = len(data)
    t0 = time.time()
    
    print("=" * 80)
    print("  🕳️  V14 EVENT HORIZON — MAXIMUM COVERAGE ASSAULT")
    print(f"  {N} draws | Pool 30+5 | Port up to 5000")
    print("=" * 80)
    
    WARMUP = 200
    POOL_CORE = 30
    POOL_SAFETY = 5
    port_sizes = [1, 50, 100, 500, 1000, 2000, 3000, 5000]
    
    results = {ps: [] for ps in port_sizes}
    random_results = []
    pool_contain = 0
    five_hits, six_hits = [], []
    miss_info = []
    
    meta = MetaStacker()
    np.random.seed(42)
    n_test = N - WARMUP
    
    print(f"\n  Walk-forward: {n_test} | Port sizes: {port_sizes}")
    print(f"{'━' * 80}")
    
    for test_idx in range(n_test):
        train_end = WARMUP + test_idx
        actual = set(data[train_end][:PICK])
        
        # Retrain meta every 100
        if test_idx % 100 == 0:
            meta.train(data, WARMUP, train_end)
        
        # Compute signals + meta-stack
        raw = compute_signals(data, train_end)
        ensemble = meta.predict(raw)
        
        # Build pool: top 30 + 5 safety = 35
        ranked = np.argsort(-ensemble)
        core = [int(ranked[i])+1 for i in range(POOL_CORE)]
        safety = [int(ranked[i])+1 for i in range(POOL_CORE, POOL_CORE+POOL_SAFETY)]
        full_pool = sorted(set(core + safety))
        
        # Pair history for combo scoring
        pair_hist = Counter()
        for d in data[max(0,train_end-200):train_end]:
            for p in combinations(sorted(d[:PICK]), 2):
                pair_hist[p] += 1
        
        constraints = learn_constraints(data, train_end)
        num_sc = {n+1: float(ensemble[n]) for n in range(MAX)}
        
        # Generate massive portfolio
        max_port = max(port_sizes)
        portfolio = generate_massive_portfolio(
            full_pool, num_sc, pair_hist, data[:train_end],
            max_port + 200, constraints
        )
        
        # Score
        for ps in port_sizes:
            port = portfolio[:ps]
            if port:
                best = max(len(actual & set(p['numbers'])) for p in port)
            else:
                best = 0
            results[ps].append(best)
        
        # Track hits
        max_best = results[max(port_sizes)][-1]
        if max_best >= 5:
            five_hits.append((train_end, max_best, sorted(actual)))
            # Miss analysis for port-2000
            port2k = portfolio[:2000]
            if port2k:
                b2k = max(len(actual & set(p['numbers'])) for p in port2k)
                if b2k == 5:
                    for p in port2k:
                        if len(actual & set(p['numbers'])) == 5:
                            missed = actual - set(p['numbers'])
                            miss_info.append({
                                'draw': train_end,
                                'missed': sorted(missed),
                                'in_pool': sorted(missed & set(full_pool)),
                            })
                            break
        if max_best >= 6:
            six_hits.append((train_end, sorted(actual)))
        
        # Pool containment
        if len(actual & set(full_pool)) >= 6:
            pool_contain += 1
        
        # Random
        rand = set(np.random.choice(range(1, MAX+1), PICK, replace=False).tolist())
        random_results.append(len(rand & actual))
        
        # Progress
        if (test_idx+1) % 100 == 0:
            elapsed = time.time() - t0
            b500 = results[500][-100:]
            p3_500 = sum(1 for m in b500 if m>=3)/len(b500)*100
            p4_500 = sum(1 for m in b500 if m>=4)/len(b500)*100
            b5k = results[5000][-100:]
            p3_5k = sum(1 for m in b5k if m>=3)/len(b5k)*100
            p5_5k = sum(1 for m in b5k if m>=5)/len(b5k)*100
            p6_5k = sum(1 for m in b5k if m>=6)/len(b5k)*100
            speed = (test_idx+1)/elapsed
            eta = (n_test-test_idx-1)/speed if speed > 0 else 0
            print(f"  [{test_idx+1:5d}/{n_test}] "
                  f"P500: ≥3={p3_500:.0f}% ≥4={p4_500:.0f}% | "
                  f"P5K: ≥3={p3_5k:.0f}% ≥5={p5_5k:.0f}% 6/6={p6_5k:.0f}% | "
                  f"Pool:{pool_contain}x | 5/6:{len(five_hits)} 6/6:{len(six_hits)} | "
                  f"ETA:{eta:.0f}s")
    
    # ================================================================
    # FINAL RESULTS
    # ================================================================
    elapsed = time.time() - t0
    
    print(f"\n{'═' * 80}")
    print(f"  🕳️  V14 EVENT HORIZON — FINAL RESULTS")
    print(f"  {n_test} tests | Time: {elapsed:.1f}s")
    print(f"{'═' * 80}")
    
    print(f"\n  🎯 PORTFOLIO RESULTS:")
    print(f"  {'Port':>6} | {'Avg':>8} | {'≥3/6':>8} | {'≥4/6':>8} | {'≥5/6':>8} | {'6/6':>8}")
    print(f"  {'─'*6} | {'─'*8} | {'─'*8} | {'─'*8} | {'─'*8} | {'─'*8}")
    
    best_port = {}
    for ps in port_sizes:
        bm = results[ps]
        avg = np.mean(bm)
        p3 = sum(1 for m in bm if m>=3)/n_test*100
        p4 = sum(1 for m in bm if m>=4)/n_test*100
        p5 = sum(1 for m in bm if m>=5)/n_test*100
        p6 = sum(1 for m in bm if m>=6)/n_test*100
        print(f"  {ps:6d} | {avg:8.4f} | {p3:7.2f}% | {p4:7.2f}% | {p5:7.2f}% | {p6:7.2f}%")
        best_port[f'port_{ps}'] = {
            'avg': round(avg,4), 'pct_3': round(p3,2),
            'pct_4': round(p4,2), 'pct_5': round(p5,2), 'pct_6': round(p6,2),
        }
    
    print(f"\n  🔑 POOL CONTAINMENT: {pool_contain}/{n_test} ({pool_contain/n_test*100:.2f}%)")
    
    if six_hits:
        print(f"\n  🎉🎉🎉 6/6 JACKPOT: {len(six_hits)} draws!")
        for idx, actual_d in six_hits:
            print(f"    🏆 Draw #{idx}: {actual_d}")
    else:
        print(f"\n  ❌ 6/6: 0")
    
    print(f"\n  🏆 5/6+ HITS: {len(five_hits)}")
    for idx, m, actual_d in five_hits[:20]:
        print(f"    Draw #{idx}: {actual_d} → {m}/6")
    
    # Save
    output = {
        'version': '14.0 — EVENT HORIZON',
        'n_draws': N, 'n_test': n_test,
        'pool_config': f'{POOL_CORE}+{POOL_SAFETY}={POOL_CORE+POOL_SAFETY}',
        'portfolio_results': best_port,
        'pool_containment': {'count': pool_contain, 'pct': round(pool_contain/n_test*100, 2)},
        'five_hits': len(five_hits), 'six_hits': len(six_hits),
        'six_hit_draws': [{'draw': idx, 'numbers': nums} for idx, nums in six_hits],
        'elapsed_seconds': round(elapsed, 1),
    }
    path = os.path.join(os.path.dirname(__file__), 'models', 'v14_event_horizon.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n  Saved to {path}")
    print(f"{'═' * 80}")


if __name__ == '__main__':
    run_v14()
