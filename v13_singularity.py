"""
V13 SINGULARITY — ADAPTIVE POOL + MASSIVE COVERAGE
====================================================
V12 breakthrough: 52 draws (4%) had ALL 6 in pool, but 0 in portfolio.
V13 strategy: WHEN signals are strong → SHRINK pool → MORE coverage.

KEY INSIGHT:
  Pool 27: C(27,6) = 296,010 → 1000 vé = 0.34% coverage
  Pool 20: C(20,6) =  38,760 → 2000 vé = 5.16% coverage (15x better!)
  Pool 18: C(18,6) =  18,564 → 2000 vé = 10.8% coverage (30x better!)

IMPROVEMENTS OVER V12:
1. CONFIDENCE-BASED POOL SIZE — Strong signal → tiny pool, high coverage
2. PORTFOLIO UP TO 2000 — 2x V12 maximum
3. META-STACKER — GBM learns non-linear weapon combinations
4. ANTI-OVERLAP DIVERSITY — Minimum-overlap ticket generation
5. DOUBLE-LAYER POOL — Tight core + extended safety ring
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
# ALL WEAPONS (compact, battle-tested from V11+V12)
# ================================================================
def compute_all_signals(data, at_index):
    """Returns 3 separate score arrays: sequence, machine, safety-aware."""
    relevant = data[:at_index]
    n = len(relevant)
    if n < 50:
        return np.ones(MAX) * PICK/MAX, np.zeros(MAX), np.zeros(MAX)
    
    last = set(relevant[-1][:PICK])
    base_p = PICK / MAX
    
    # ---- SIGNAL 1: SEQUENCE (transition + momentum + gap + KNN + Markov) ----
    s1 = np.zeros(MAX)
    
    # Multi-scale frequency
    for w, wt in [(3, 4.0), (5, 3.0), (10, 2.0), (20, 1.5), (50, 1.0)]:
        if n < w: continue
        for num in range(1, MAX+1):
            f = sum(1 for d in relevant[-w:] if num in d[:PICK]) / w
            s1[num-1] += f * wt
    
    # Transition matrix
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
            cp = tf / tp
            s1[num-1] += (cp / base_p - 1) * 3
    
    # Gap timing
    for num in range(1, MAX+1):
        apps = [i for i, d in enumerate(relevant) if num in d[:PICK]]
        if len(apps) < 5: continue
        gaps = [apps[j+1] - apps[j] for j in range(len(apps)-1)]
        mg, sg = np.mean(gaps), np.std(gaps)
        cg = n - apps[-1]
        z = (cg - mg) / sg if sg > 0 else 0
        if z > 0.5: s1[num-1] += z * 1.5
        elif z < -1: s1[num-1] -= 1
    
    # KNN attention (lower threshold for more signal)
    knn = Counter()
    for i in range(n-2):
        sim = len(set(relevant[i][:PICK]) & last)
        if sim >= 2:
            w = sim ** 2
            for num in relevant[i+1][:PICK]:
                knn[num] += w
    mx = max(knn.values()) if knn else 1
    for num in range(1, MAX+1):
        s1[num-1] += knn.get(num, 0) / mx * 2.5
    
    # Markov Order-2
    if n >= 10:
        last1 = set(relevant[-1][:PICK])
        last2 = set(relevant[-2][:PICK])
        both_c, either_c = Counter(), Counter()
        tb, te = 0, 0
        for i in range(2, n):
            p2 = set(relevant[i-2][:PICK])
            p1 = set(relevant[i-1][:PICK])
            cu = set(relevant[i][:PICK])
            for num in range(1, MAX+1):
                ip2, ip1, ic = num in p2, num in p1, num in cu
                if ip2 and ip1: tb += 1; both_c[num] += int(ic)
                elif ip2 or ip1: te += 1; either_c[num] += int(ic)
        for num in range(1, MAX+1):
            if num in last1 and num in last2:
                p = both_c[num] / max(tb/MAX, 1)
            elif num in last1 or num in last2:
                p = either_c[num] / max(te/MAX, 1)
            else: p = 0
            s1[num-1] += (p - base_p) * 5
    
    # Co-occurrence pairs
    pf = Counter()
    for d in relevant[-200:]:
        for p in combinations(sorted(d[:PICK]), 2):
            pf[p] += 1
    for num in range(1, MAX+1):
        cooc = sum(pf.get(tuple(sorted([p, num])), 0) for p in last)
        s1[num-1] += cooc * 0.05
    
    # Triplet completion
    tf = Counter()
    for d in relevant[-150:]:
        for t in combinations(sorted(d[:PICK]), 3):
            tf[t] += 1
    for t, c in tf.most_common(500):
        if c < 2: break
        ts = set(t)
        ov = ts & last
        if len(ov) == 2:
            missing = (ts - last).pop()
            s1[missing-1] += c * 0.3
    
    # ---- SIGNAL 2: MACHINE BIAS (drift + inertia + temporal) ----
    s2 = np.zeros(MAX)
    
    if n >= 100:
        # Ball wear drift
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
        if mx > 0: s2 += (drift/mx) * 3
        
        # Draw inertia
        rep = Counter()
        pcount = Counter()
        for i in range(1, n):
            prev = set(relevant[i-1][:PICK])
            curr = set(relevant[i][:PICK])
            for num in prev:
                pcount[num] += 1
                if num in curr: rep[num] += 1
        for num in range(1, MAX+1):
            if num in last and pcount[num] > 10:
                pr = rep[num] / pcount[num]
                s2[num-1] += (pr/base_p - 1) * 2
    
    if n >= 300:
        # Temporal drift
        win = 150
        nw = max(n//win, 2)
        aw = n // nw
        wfs = []
        for w in range(nw):
            s, e = w*aw, min((w+1)*aw, n)
            freq = np.zeros(MAX)
            for d in relevant[s:e]:
                for num in d[:PICK]: freq[num-1] += 1
            freq /= (e-s)
            wfs.append(freq)
        for num in range(MAX):
            vals = [wf[num] for wf in wfs]
            if len(vals) >= 3:
                slope = stats.linregress(np.arange(len(vals)), vals).slope
                s2[num] += slope * 50
    
    # ---- SIGNAL 3: CONTEXTUAL (decade balance + neighbor + surprise) ----
    s3 = np.zeros(MAX)
    
    # Overall recent frequency for non-pool numbers
    freq50 = np.zeros(MAX)
    for d in relevant[-50:]:
        for num in d[:PICK]: freq50[num-1] += 1
    freq50 /= 50
    s3 += freq50 * 5
    
    # Decade balance
    recent_dec = Counter()
    for d in relevant[-30:]:
        for num in d[:PICK]:
            recent_dec[min((num-1)//10, 4)] += 1
    expected_dec = 30 * PICK / 5
    for dec in range(5):
        surplus = recent_dec.get(dec, 0) / expected_dec
        for num in range(dec*10+1, min((dec+1)*10+1, MAX+1)):
            s3[num-1] += (surplus - 1) * 2
    
    # Strong transitions from last draw
    for num in range(1, MAX+1):
        tf_score = sum(follow[p].get(num, 0) for p in last if p in follow)
        s3[num-1] += tf_score * 0.1
    
    return s1, s2, s3


# ================================================================
# META-STACKER: Learn optimal signal combination
# ================================================================
class MetaStacker:
    """Train a GBM that learns how to combine weapon signals."""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
    
    def train(self, data, start, end):
        """Build training data: for each past draw, features = signals, target = match."""
        X, Y = [], []
        for idx in range(max(start, 100), end-1, 3):  # Every 3rd draw for speed
            s1, s2, s3 = compute_all_signals(data, idx)
            actual = set(data[idx][:PICK])
            
            # Build per-number features
            for num in range(1, MAX+1):
                feats = [s1[num-1], s2[num-1], s3[num-1],
                         s1[num-1]*s2[num-1], s1[num-1]*s3[num-1],
                         s2[num-1]*s3[num-1]]
                X.append(feats)
                Y.append(1 if num in actual else 0)
        
        if len(X) < 100:
            return False
        
        X = np.array(X)
        Y = np.array(Y)
        X_s = self.scaler.fit_transform(X)
        
        self.model = GradientBoostingClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
        self.model.fit(X_s, Y)
        return True
    
    def predict(self, s1, s2, s3):
        """Get per-number probability from meta model."""
        if self.model is None:
            return (s1 + s2 + s3) / 3
        
        X = []
        for num in range(MAX):
            feats = [s1[num], s2[num], s3[num],
                     s1[num]*s2[num], s1[num]*s3[num],
                     s2[num]*s3[num]]
            X.append(feats)
        
        X = np.array(X)
        X_s = self.scaler.transform(X)
        probs = self.model.predict_proba(X_s)
        
        scores = np.zeros(MAX)
        for i in range(MAX):
            if len(self.model.classes_) == 2:
                scores[i] = probs[i][1]
            else:
                scores[i] = probs[i][0]
        return scores


# ================================================================
# CONFIDENCE-BASED ADAPTIVE POOL
# ================================================================
def compute_pool_and_confidence(ensemble_scores):
    """
    Analyze signal concentration to determine pool size.
    Strong signal = tight pool (18-20) = HIGH coverage per ticket.
    Weak signal = wide pool (28-30) = more containment but less coverage.
    """
    ranked = np.argsort(-ensemble_scores)
    top_scores = ensemble_scores[ranked[:30]]
    
    # Signal concentration metrics
    top6_avg = np.mean(top_scores[:6])
    top20_avg = np.mean(top_scores[:20])
    bottom_avg = np.mean(top_scores[20:])
    
    # Gap between top cluster and rest
    gap_ratio = (top6_avg - bottom_avg) / (top20_avg - bottom_avg + 1e-8)
    
    # Entropy of top scores (low entropy = concentrated)
    top_norm = top_scores[:20] / (np.sum(top_scores[:20]) + 1e-8)
    entropy = -np.sum(top_norm * np.log(top_norm + 1e-10))
    max_entropy = np.log(20)
    norm_entropy = entropy / max_entropy  # 0=concentrated, 1=spread
    
    # Determine pool size
    if norm_entropy < 0.85 and gap_ratio > 1.3:
        pool_size = 18  # VERY concentrated → tiny pool, massive coverage
        confidence = 'HIGH'
    elif norm_entropy < 0.90 and gap_ratio > 1.1:
        pool_size = 22  # Moderately concentrated
        confidence = 'MEDIUM'
    elif norm_entropy < 0.95:
        pool_size = 26  # Spread
        confidence = 'LOW'
    else:
        pool_size = 30  # Very spread → maximize containment
        confidence = 'MINIMAL'
    
    return pool_size, confidence, ranked


# ================================================================
# FAST DIVERSIFIED PORTFOLIO GENERATOR
# ================================================================
def generate_portfolio(pool, scores, n_tickets, constraints):
    """Generate max-diversity portfolio with constraint filtering."""
    pool = sorted(pool)
    pool_size = len(pool)
    if pool_size < PICK:
        return []
    
    total_combos = math.comb(pool_size, PICK)
    
    # If total combos <= n_tickets, return ALL (guaranteed cover!)
    if total_combos <= n_tickets * 1.5:
        all_c = []
        for combo in combinations(pool, PICK):
            if _passes_constraints(combo, constraints):
                all_c.append({'numbers': list(combo), 
                              'score': sum(scores.get(n, 0) for n in combo)})
        all_c.sort(key=lambda x: -x['score'])
        return all_c[:n_tickets]
    
    # Score and filter combos (sample if too many)
    if total_combos > 500000:
        # Random sampling for large pools
        candidates = _sample_combos(pool, scores, n_tickets * 5, constraints)
    else:
        # Enumerate all
        candidates = []
        for combo in combinations(pool, PICK):
            if not _passes_constraints(combo, constraints):
                continue
            sc = sum(scores.get(n, 0) for n in combo)
            candidates.append((list(combo), sc))
    
    if not candidates:
        return []
    
    candidates.sort(key=lambda x: -x[1])
    
    # Greedy diversified selection
    selected = []
    used_triples = set()
    
    for combo, score in candidates:
        if len(selected) >= n_tickets:
            break
        
        # Diversity: check overlap with recent selections
        if len(selected) > 0:
            check_range = selected[-min(50, len(selected)):]
            max_ov = max(len(set(combo) & set(s['numbers'])) for s in check_range)
            if max_ov >= 5:
                continue
        
        # Triple coverage bonus
        new_triples = 0
        combo_triples = list(combinations(combo, 3))
        for t in combo_triples[:10]:  # Check subset for speed
            if t not in used_triples:
                new_triples += 1
        
        selected.append({
            'numbers': combo,
            'score': round(score + new_triples * 0.02, 3),
        })
        
        for t in combo_triples:
            used_triples.add(t)
    
    return selected


def _passes_constraints(combo, constraints):
    s = sum(combo)
    if s < constraints.get('sum_lo', 0) or s > constraints.get('sum_hi', 999):
        return False
    rng = combo[-1] - combo[0]
    if rng < constraints.get('range_lo', 0) or rng > constraints.get('range_hi', 999):
        return False
    odd = sum(1 for x in combo if x % 2 == 1)
    if odd < constraints.get('odd_lo', 0) or odd > constraints.get('odd_hi', PICK):
        return False
    return True


def _sample_combos(pool, scores, n_samples, constraints):
    """Random-sample combos from large pools, weighted by scores."""
    pool = sorted(pool)
    pool_arr = np.array(pool)
    
    # Weight by score
    weights = np.array([scores.get(n, 0) for n in pool])
    weights = weights - weights.min() + 0.1
    weights /= weights.sum()
    
    candidates = []
    seen = set()
    attempts = 0
    
    while len(candidates) < n_samples and attempts < n_samples * 10:
        attempts += 1
        idx = np.random.choice(len(pool), PICK, replace=False, p=weights)
        combo = tuple(sorted(pool_arr[idx]))
        if combo in seen:
            continue
        seen.add(combo)
        if _passes_constraints(combo, constraints):
            sc = sum(scores.get(n, 0) for n in combo)
            candidates.append((list(combo), sc))
    
    return candidates


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
# MASTER WALK-FORWARD
# ================================================================
def run_v13():
    data = get_mega645_numbers()
    all_rows = get_mega645_all()
    dates = [r['draw_date'] for r in all_rows]
    N = len(data)
    t0 = time.time()
    
    print("=" * 80)
    print("  🌌 V13 SINGULARITY — ADAPTIVE POOL + META-STACKER")
    print(f"  {N} draws | Target: break through to 6/6")
    print("=" * 80)
    
    WARMUP = 200
    port_sizes = [1, 10, 50, 100, 200, 500, 1000, 2000]
    
    results = {ps: [] for ps in port_sizes}
    random_results = []
    pool_contain_count = 0
    five_hits = []
    six_hits = []
    miss_analysis = []
    confidence_dist = Counter()
    pool_size_dist = []
    
    meta = MetaStacker()
    META_RETRAIN = 100
    
    np.random.seed(42)
    n_test = N - WARMUP
    
    print(f"\n  Walk-forward: {n_test} iterations")
    print(f"  V13: Adaptive Pool (18-30) + Meta-Stacker + Port-2000")
    print(f"{'━' * 80}")
    
    for test_idx in range(n_test):
        train_end = WARMUP + test_idx
        actual = set(data[train_end][:PICK])
        
        # Retrain meta-stacker periodically
        if test_idx % META_RETRAIN == 0:
            meta.train(data, WARMUP, train_end)
        
        # ---- COMPUTE ALL SIGNALS ----
        s1, s2, s3 = compute_all_signals(data, train_end)
        
        # ---- META-STACKER COMBINATION ----
        def normalize(s):
            mn, mx = np.min(s), np.max(s)
            if mx - mn < 0.001: return np.ones_like(s) * 0.5
            return (s - mn) / (mx - mn)
        
        s1n, s2n, s3n = normalize(s1), normalize(s2), normalize(s3)
        ensemble = meta.predict(s1n, s2n, s3n)
        
        # ---- ADAPTIVE POOL SIZE ----
        pool_size, confidence, ranked = compute_pool_and_confidence(ensemble)
        confidence_dist[confidence] += 1
        pool_size_dist.append(pool_size)
        
        # Build double-layer pool: CORE + SAFETY RING
        core_pool = [int(ranked[i])+1 for i in range(pool_size)]
        
        # Safety ring: next 5 numbers outside core
        safety_ring = []
        for i in range(pool_size, min(pool_size+5, MAX)):
            safety_ring.append(int(ranked[i])+1)
        
        extended_pool = sorted(set(core_pool + safety_ring))
        
        # ---- GENERATE PORTFOLIO ----
        constraints = learn_constraints(data, train_end)
        num_scores = {n+1: float(ensemble[n]) for n in range(MAX)}
        
        max_port = max(port_sizes)
        portfolio_all = generate_portfolio(
            extended_pool, num_scores, max_port + 100, constraints
        )
        
        # ---- SCORE AGAINST ACTUAL ----
        for ps in port_sizes:
            port = portfolio_all[:ps]
            if port:
                best = max(len(actual & set(p['numbers'])) for p in port)
            else:
                best = 0
            results[ps].append(best)
            
            if ps == max(500, min(port_sizes)):
                if best >= 5:
                    five_hits.append((train_end, best, sorted(actual), confidence, pool_size))
                    # Miss analysis
                    for p in port:
                        matched = actual & set(p['numbers'])
                        if len(matched) == best:
                            missed = actual - set(p['numbers'])
                            miss_analysis.append({
                                'draw': train_end, 'actual': sorted(actual),
                                'missed': sorted(missed),
                                'in_pool': sorted(missed & set(extended_pool)),
                                'out_pool': sorted(missed - set(extended_pool)),
                                'pool_size': len(extended_pool),
                                'confidence': confidence,
                            })
                            break
        
        # Check maximum portfolio for 6/6
        max_port_result = results[max(port_sizes)][-1] if results[max(port_sizes)] else 0
        if max_port_result >= 6:
            six_hits.append((train_end, sorted(actual), confidence, pool_size))
        
        # Pool containment
        pool_hits = len(actual & set(extended_pool))
        if pool_hits >= 6:
            pool_contain_count += 1
        
        # Random baseline
        rand = set(np.random.choice(range(1, MAX+1), PICK, replace=False).tolist())
        random_results.append(len(rand & actual))
        
        # Progress
        if (test_idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            b500 = results[500][-100:]
            p3 = sum(1 for m in b500 if m>=3)/len(b500)*100
            p4 = sum(1 for m in b500 if m>=4)/len(b500)*100
            p5 = sum(1 for m in b500 if m>=5)/len(b500)*100
            b2000 = results[2000][-100:]
            p3k = sum(1 for m in b2000 if m>=3)/len(b2000)*100 if b2000 else 0
            p5k = sum(1 for m in b2000 if m>=5)/len(b2000)*100 if b2000 else 0
            p6k = sum(1 for m in b2000 if m>=6)/len(b2000)*100 if b2000 else 0
            speed = (test_idx+1)/elapsed
            eta = (n_test-test_idx-1)/speed if speed > 0 else 0
            avg_pool = np.mean(pool_size_dist[-100:])
            print(f"  [{test_idx+1:5d}/{n_test}] "
                  f"P500: ≥3={p3:.0f}% ≥4={p4:.0f}% ≥5={p5:.0f}% | "
                  f"P2K: ≥3={p3k:.0f}% ≥5={p5k:.0f}% 6/6={p6k:.0f}% | "
                  f"Pool:{pool_contain_count}x 6in | "
                  f"Avg pool={avg_pool:.0f} | ETA:{eta:.0f}s")
    
    # ================================================================
    # FINAL RESULTS
    # ================================================================
    elapsed = time.time() - t0
    
    print(f"\n{'═' * 80}")
    print(f"  🌌 V13 SINGULARITY — FINAL RESULTS")
    print(f"  {n_test} tests | Time: {elapsed:.1f}s")
    print(f"{'═' * 80}")
    
    print(f"\n  🎯 PORTFOLIO RESULTS:")
    print(f"  {'Port':>6} | {'Avg':>8} | {'≥3/6':>8} | {'≥4/6':>8} | {'≥5/6':>8} | {'6/6':>8}")
    print(f"  {'─'*6} | {'─'*8} | {'─'*8} | {'─'*8} | {'─'*8} | {'─'*8}")
    ra = np.mean(random_results)
    print(f"  {'Rand':>6} | {ra:8.4f} | "
          f"{sum(1 for m in random_results if m>=3)/n_test*100:7.2f}% | "
          f"{sum(1 for m in random_results if m>=4)/n_test*100:7.2f}% | "
          f"{sum(1 for m in random_results if m>=5)/n_test*100:7.2f}% | "
          f"{sum(1 for m in random_results if m>=6)/n_test*100:7.2f}%")
    
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
    
    print(f"\n  🔑 POOL CONTAINMENT: {pool_contain_count}/{n_test} "
          f"({pool_contain_count/n_test*100:.2f}%)")
    
    print(f"\n  📊 CONFIDENCE DISTRIBUTION:")
    for conf in ['HIGH', 'MEDIUM', 'LOW', 'MINIMAL']:
        cnt = confidence_dist.get(conf, 0)
        print(f"    {conf:8s}: {cnt:5d} ({cnt/n_test*100:.1f}%)")
    
    print(f"\n  📏 POOL SIZE STATS: "
          f"avg={np.mean(pool_size_dist):.1f} | "
          f"min={min(pool_size_dist)} | max={max(pool_size_dist)}")
    
    # 5/6 and 6/6
    print(f"\n  🏆 5/6+ HITS: {len(five_hits)} draws")
    for draw_idx, matches, actual_d, conf, ps in five_hits[:20]:
        print(f"    Draw #{draw_idx}: {actual_d} → {matches}/6 "
              f"(conf={conf}, pool={ps})")
    
    if six_hits:
        print(f"\n  🎉🎉🎉 6/6 JACKPOT: {len(six_hits)} draws!")
        for draw_idx, actual_d, conf, ps in six_hits:
            print(f"    🏆 Draw #{draw_idx}: {actual_d} (conf={conf}, pool={ps})")
    else:
        print(f"\n  ❌ 6/6 JACKPOT: 0 draws")
    
    # Miss analysis
    if miss_analysis:
        print(f"\n  🔍 MISS ANALYSIS:")
        out_pool = sum(1 for m in miss_analysis if m['out_pool'])
        in_pool = sum(1 for m in miss_analysis if m['in_pool'] and not m['out_pool'])
        print(f"    IN POOL (coverage gap): {in_pool}")
        print(f"    OUT OF POOL: {out_pool}")
        for m in miss_analysis[:10]:
            tag = "✅ IN" if m['in_pool'] and not m['out_pool'] else "❌ OUT"
            print(f"    Draw #{m['draw']}: missed={m['missed']} → {tag} "
                  f"(pool={m['pool_size']}, conf={m['confidence']})")
    
    # Save
    output = {
        'version': '13.0 — SINGULARITY',
        'improvements': ['Meta-Stacker', 'Confidence-Adaptive Pool 18-30',
                          'Double-Layer Pool', 'Port-2000'],
        'n_draws': N, 'n_test': n_test,
        'portfolio_results': best_port,
        'pool_containment': {'count': pool_contain_count,
                              'pct': round(pool_contain_count/n_test*100, 2)},
        'five_hits': len(five_hits),
        'six_hits': len(six_hits),
        'confidence_distribution': dict(confidence_dist),
        'avg_pool_size': round(np.mean(pool_size_dist), 1),
        'miss_analysis_summary': {
            'in_pool': sum(1 for m in miss_analysis if m['in_pool'] and not m['out_pool']),
            'out_pool': sum(1 for m in miss_analysis if m['out_pool']),
        },
        'elapsed_seconds': round(elapsed, 1),
    }
    path = os.path.join(os.path.dirname(__file__), 'models', 'v13_singularity.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n  Saved to {path}")
    print(f"{'═' * 80}")


if __name__ == '__main__':
    run_v13()
