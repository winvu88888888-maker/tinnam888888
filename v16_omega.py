"""
V16 OMEGA — RADICALLY DIFFERENT APPROACHES
============================================
V15 Deep Analysis revealed: combo scoring = RANDOM (50th percentile).
Pool containment = RANDOM (matches C(n,6)/C(45,6) exactly).

THIS ENGINE tests FUNDAMENTALLY NEW strategies:

APPROACH 1: XGBoost COMBO Classifier
  - Instead of scoring individual numbers → score COMBOS directly
  - Features: last draw properties → predict next draw properties
  - Train: does this STRUCTURAL TYPE of combo follow this draw?

APPROACH 2: State Space Prediction  
  - Represent each draw as a STATE vector (sum, range, odd%, decade_dist)
  - Learn state transitions → predict next state → generate matching combos
  - Think: Markov chain on DRAW PROPERTIES, not individual numbers

APPROACH 3: Conditional Pair Tables
  - For each pair (a,b) in last draw, build conditional probability table
  - P(next contains pair (x,y) | last contained pair (a,b))
  - Score combo by product of pair chances

APPROACH 4: Anti-Pattern (Contrarian)
  - What if AVOIDING recently hot patterns works better?
  - Generate combos that are maximally DIFFERENT from recent draws
  
BASELINE: Pure Random (for honest comparison)

All approaches compared with STATISTICAL SIGNIFICANCE TESTING.
"""
import sys, os, math, json, time, random
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper.data_manager import get_mega645_numbers

MAX = 45; PICK = 6
random.seed(42); np.random.seed(42)


# ================================================================
# DRAW FEATURE EXTRACTION (for combo-level ML)
# ================================================================
def draw_features(draw):
    """Extract 15 structural features from a draw."""
    d = sorted(draw[:PICK])
    s = sum(d)
    r = d[-1] - d[0]
    odd = sum(1 for x in d if x%2==1)
    high = sum(1 for x in d if x > 22)
    gaps = [d[i+1]-d[i] for i in range(PICK-1)]
    decades = [0]*5
    for x in d:
        decades[min(x//10, 4)] += 1
    return [
        s, r, odd, high, np.mean(d), np.std(d),
        min(gaps), max(gaps), np.mean(gaps), np.std(gaps),
        decades[0], decades[1], decades[2], decades[3], decades[4],
    ]


def combo_features(combo, last_draw, prev2_draw=None):
    """Features for a CANDIDATE combo vs last draw."""
    cf = draw_features(combo)
    lf = draw_features(last_draw)
    
    # Delta features (how different from last draw)
    delta = [cf[i]-lf[i] for i in range(len(cf))]
    
    # Overlap
    overlap = len(set(combo[:PICK]) & set(last_draw[:PICK]))
    
    # Pair overlap with last draw
    last_pairs = set(combinations(sorted(last_draw[:PICK]), 2))
    combo_pairs = set(combinations(sorted(combo[:PICK]), 2))
    pair_overlap = len(last_pairs & combo_pairs)
    
    features = cf + delta + [overlap, pair_overlap]
    
    if prev2_draw:
        p2f = draw_features(prev2_draw)
        delta2 = [cf[i]-p2f[i] for i in range(len(cf))]
        overlap2 = len(set(combo[:PICK]) & set(prev2_draw[:PICK]))
        features += delta2 + [overlap2]
    
    return features


# ================================================================
# APPROACH 1: XGBoost Combo Classifier
# ================================================================
def approach_xgb_combo(data, train_end, n_tickets=5000):
    """Train XGBoost on combo-level features, generate scored combos."""
    # Training: for each draw, score combos from a pool
    # Positive: the actual winning draw
    # Negative: random combos
    
    # Build training data from recent 200 draws
    X_train, Y_train = [], []
    start = max(2, train_end - 300)
    
    for idx in range(start, train_end - 1):
        last = data[idx]
        prev2 = data[idx-1] if idx > 0 else data[idx]
        actual = sorted(data[idx+1][:PICK])
        
        # Positive
        cf = combo_features(actual, last, prev2)
        X_train.append(cf)
        Y_train.append(1)
        
        # 5 negatives (random combos)
        for _ in range(5):
            neg = sorted(random.sample(range(1, MAX+1), PICK))
            cf = combo_features(neg, last, prev2)
            X_train.append(cf)
            Y_train.append(0)
    
    if len(X_train) < 100:
        return None
    
    X = np.array(X_train)
    Y = np.array(Y_train)
    
    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    
    model = GradientBoostingClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.1,
        subsample=0.8, random_state=42)
    model.fit(Xs, Y)
    
    # Generate candidates and score
    last = data[train_end]
    prev2 = data[train_end-1]
    
    candidates = []
    seen = set()
    for _ in range(n_tickets * 3):
        combo = sorted(random.sample(range(1, MAX+1), PICK))
        key = tuple(combo)
        if key in seen: continue
        seen.add(key)
        cf = combo_features(combo, last, prev2)
        cf_arr = np.nan_to_num(np.array([cf]), nan=0, posinf=0, neginf=0)
        prob = model.predict_proba(scaler.transform(cf_arr))[0]
        score = prob[1] if len(model.classes_) == 2 else 0.5
        candidates.append((combo, score))
    
    candidates.sort(key=lambda x: -x[1])
    return [c for c, _ in candidates[:n_tickets]]


# ================================================================
# APPROACH 2: State Space Prediction
# ================================================================
def approach_state_prediction(data, train_end, n_tickets=5000):
    """Predict next draw's structural state, generate matching combos."""
    # Build state transition model
    states = []
    for i in range(train_end + 1):
        d = sorted(data[i][:PICK])
        s = sum(d)
        r = d[-1] - d[0]
        odd = sum(1 for x in d if x%2==1)
        states.append((s, r, odd))
    
    # Learn: what sum/range/odd follows what?
    sum_trans = defaultdict(list)
    range_trans = defaultdict(list)
    odd_trans = defaultdict(list)
    
    for i in range(len(states)-1):
        s1, r1, o1 = states[i]
        s2, r2, o2 = states[i+1]
        # Bin sum into 5 groups
        sb1 = min(s1//30, 7)
        sb2 = min(s2//30, 7)
        sum_trans[sb1].append(s2)
        range_trans[r1//5].append(r2)
        odd_trans[o1].append(o2)
    
    # Predict next state
    last = states[-1]
    sb = min(last[0]//30, 7)
    pred_sum_range = (np.mean(sum_trans[sb]) if sum_trans[sb] else 138,
                      14 if not sum_trans[sb] else np.std(sum_trans[sb]))
    rb = last[1]//5
    pred_range_range = (np.mean(range_trans[rb]) if range_trans[rb] else 30,
                        5 if not range_trans[rb] else np.std(range_trans[rb]))
    pred_odd = Counter(odd_trans[last[2]]).most_common(3) if odd_trans[last[2]] else [(3,1)]
    
    # Generate combos matching predicted state
    candidates = []
    seen = set()
    target_sum = pred_sum_range[0]
    target_sum_std = max(pred_sum_range[1], 10)
    target_range = pred_range_range[0]
    target_range_std = max(pred_range_range[1], 3)
    
    for _ in range(n_tickets * 5):
        combo = sorted(random.sample(range(1, MAX+1), PICK))
        key = tuple(combo)
        if key in seen: continue
        seen.add(key)
        
        s = sum(combo)
        r = combo[-1] - combo[0]
        
        # Score by how well it matches predicted state
        sum_score = -abs(s - target_sum) / target_sum_std
        range_score = -abs(r - target_range) / target_range_std
        score = sum_score + range_score
        
        candidates.append((combo, score))
        if len(candidates) >= n_tickets * 3:
            break
    
    candidates.sort(key=lambda x: -x[1])
    return [c for c, _ in candidates[:n_tickets]]


# ================================================================
# APPROACH 3: Conditional Pair Scoring
# ================================================================
def approach_pair_conditional(data, train_end, n_tickets=5000):
    """Score combos by conditional pair probabilities."""
    # Build: P(pair (x,y) in next | pair (a,b) in current)
    pair_follow = defaultdict(Counter)
    
    for i in range(max(0, train_end-300), train_end):
        curr_pairs = set(combinations(sorted(data[i][:PICK]), 2))
        next_pairs = set(combinations(sorted(data[i+1][:PICK]), 2))
        for cp in curr_pairs:
            for np2 in next_pairs:
                pair_follow[cp][np2] += 1
    
    # Current draw pairs
    last_pairs = set(combinations(sorted(data[train_end][:PICK]), 2))
    
    # Score candidates
    candidates = []
    seen = set()
    
    for _ in range(n_tickets * 4):
        combo = sorted(random.sample(range(1, MAX+1), PICK))
        key = tuple(combo)
        if key in seen: continue
        seen.add(key)
        
        combo_pairs = set(combinations(combo, 2))
        
        # Score: sum of conditional pair frequencies
        score = 0
        for lp in last_pairs:
            for cp in combo_pairs:
                score += pair_follow[lp].get(cp, 0)
        
        candidates.append((combo, score))
        if len(candidates) >= n_tickets * 2:
            break
    
    candidates.sort(key=lambda x: -x[1])
    return [c for c, _ in candidates[:n_tickets]]


# ================================================================
# APPROACH 4: Anti-Pattern (Contrarian)
# ================================================================
def approach_contrarian(data, train_end, n_tickets=5000):
    """Generate combos MAXIMALLY DIFFERENT from recent draws."""
    recent = [set(data[i][:PICK]) for i in range(max(0,train_end-10), train_end+1)]
    recent_freq = Counter()
    for d in recent:
        for num in d:
            recent_freq[num] += 1
    
    candidates = []
    seen = set()
    
    for _ in range(n_tickets * 4):
        combo = sorted(random.sample(range(1, MAX+1), PICK))
        key = tuple(combo)
        if key in seen: continue
        seen.add(key)
        
        # Score: LOWER frequency = BETTER (contrarian)
        freq_score = -sum(recent_freq.get(n, 0) for n in combo)
        
        # Also: no overlap with last 3 draws
        overlap_penalty = sum(len(set(combo) & d) for d in recent[:3])
        
        score = freq_score - overlap_penalty * 2
        candidates.append((combo, score))
        if len(candidates) >= n_tickets * 2:
            break
    
    candidates.sort(key=lambda x: -x[1])
    return [c for c, _ in candidates[:n_tickets]]


# ================================================================
# APPROACH 5: V15 Signal Engine (baseline comparison)
# ================================================================
def approach_v15_signals(data, train_end, n_tickets=5000):
    """V15's signal engine for comparison."""
    relevant = data[:train_end+1]; n = train_end+1
    last = set(relevant[-1][:PICK]); base_p = PICK/MAX
    scores = {num: 0.0 for num in range(1,MAX+1)}
    for w, wt in [(3,4),(5,3),(10,2),(20,1.5),(50,1)]:
        if n < w: continue
        fc = Counter()
        for d in relevant[-w:]:
            for num in d[:PICK]: fc[num]+=1
        for num in range(1,MAX+1): scores[num]+=fc.get(num,0)/w*wt
    follow = defaultdict(Counter); pc = Counter()
    for i in range(n-1):
        for p in relevant[i][:PICK]:
            pc[p]+=1
            for nx in relevant[i+1][:PICK]: follow[p][nx]+=1
    for num in range(1,MAX+1):
        tf = sum(follow[p].get(num,0) for p in last)
        tp = sum(pc[p] for p in last)
        if tp>0: scores[num]+=(tf/tp/base_p-1)*3
    knn = Counter()
    for i in range(n-2):
        sim = len(set(relevant[i][:PICK])&last)
        if sim>=2:
            for num in relevant[i+1][:PICK]: knn[num]+=sim*sim
    mx = max(knn.values()) if knn else 1
    for num in range(1,MAX+1): scores[num]+=knn.get(num,0)/mx*2.5
    
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    pool = [num for num,_ in ranked[:35]]
    
    w_pool = []
    for i, num in enumerate(pool):
        w_pool.extend([num]*max(1,len(pool)-i))
    
    selected = set(); results = []
    while len(results) < n_tickets and len(results) < 50000:
        picked = set()
        while len(picked)<PICK: picked.add(random.choice(w_pool))
        if len(picked)<PICK: continue
        combo = tuple(sorted(picked))
        if combo not in selected:
            selected.add(combo)
            results.append(list(combo))
    return results


# ================================================================
# APPROACH 6: Pure Random (TRUE baseline)
# ================================================================
def approach_random(data, train_end, n_tickets=5000):
    """Pure random — the TRUE baseline."""
    results = set()
    while len(results) < n_tickets:
        combo = tuple(sorted(random.sample(range(1, MAX+1), PICK)))
        results.add(combo)
    return [list(c) for c in results]


# ================================================================
# TOURNAMENT
# ================================================================
def run_tournament():
    data = get_mega645_numbers()
    N = len(data); t0 = time.time()
    
    print("="*80)
    print("  🏆 V16 OMEGA — APPROACH TOURNAMENT")
    print(f"  {N} draws | 6 approaches head-to-head")
    print("="*80)
    
    WARMUP = 300
    PORT = 5000
    n_test = N - WARMUP
    
    approaches = {
        'XGB_Combo': approach_xgb_combo,
        'StatePred': approach_state_prediction,
        'PairCond':  approach_pair_conditional,
        'Contrarian': approach_contrarian,
        'V15Signal': approach_v15_signals,
        'Random':    approach_random,
    }
    
    results = {name: {'hits': [], 'six': 0, 'five': 0, 'six_det': []} 
               for name in approaches}
    
    print(f"\n  Tournament: {n_test} rounds, {PORT} tickets each")
    print(f"  Approaches: {list(approaches.keys())}")
    print(f"{'━'*80}")
    
    for ti in range(n_test):
        te = WARMUP + ti
        actual = set(data[te][:PICK])
        
        for name, fn in approaches.items():
            try:
                portfolio = fn(data, te, PORT)
            except Exception as e:
                portfolio = []
            
            if portfolio:
                best = max(len(actual & set(c)) for c in portfolio)
            else:
                best = 0
            
            results[name]['hits'].append(best)
            if best >= 6:
                results[name]['six'] += 1
                results[name]['six_det'].append((te, sorted(actual)))
            if best >= 5:
                results[name]['five'] += 1
        
        if (ti+1) % 50 == 0:
            el = time.time()-t0
            eta = el/(ti+1)*(n_test-ti-1)
            print(f"  [{ti+1:4d}/{n_test}] ", end='')
            for name in approaches:
                h5 = results[name]['five']; h6 = results[name]['six']
                print(f"{name[:5]}:5+={h5} 6={h6} | ", end='')
            print(f"{el:.0f}s ETA:{eta:.0f}s")
            sys.stdout.flush()
    
    elapsed = time.time() - t0
    
    print(f"\n{'═'*80}")
    print(f"  🏆 V16 OMEGA TOURNAMENT RESULTS ({n_test} rounds, {elapsed:.0f}s)")
    print(f"{'═'*80}")
    
    print(f"\n  {'Approach':<15} | {'≥3/6':>8} {'≥4/6':>8} {'≥5/6':>8} {'6/6':>8} | {'5+':>5} {'6':>5}")
    print(f"  {'─'*15} | {'─'*8} {'─'*8} {'─'*8} {'─'*8} | {'─'*5} {'─'*5}")
    
    for name in approaches:
        bm = results[name]['hits']
        p3 = sum(1 for m in bm if m>=3)/n_test*100
        p4 = sum(1 for m in bm if m>=4)/n_test*100
        p5 = sum(1 for m in bm if m>=5)/n_test*100
        p6 = sum(1 for m in bm if m>=6)/n_test*100
        h5 = results[name]['five']; h6 = results[name]['six']
        marker = " ★" if name == max(approaches, key=lambda n: results[n]['six']) else ""
        print(f"  {name:<15} | {p3:7.2f}% {p4:7.2f}% {p5:7.2f}% {p6:7.2f}% | {h5:5d} {h6:5d}{marker}")
    
    # Statistical significance
    print(f"\n  📊 STATISTICAL SIGNIFICANCE (vs Random):")
    rand_hits = results['Random']['hits']
    for name in approaches:
        if name == 'Random': continue
        bm = results[name]['hits']
        t_stat, p_val = stats.ttest_rel(bm, rand_hits)
        sig = "✅ SIGNIFICANT" if p_val < 0.05 else "❌ NOT significant"
        print(f"    {name:<15}: t={t_stat:+.3f}, p={p_val:.4f} → {sig}")
    
    # Best
    best_name = max(approaches, key=lambda n: results[n]['six'])
    print(f"\n  🏆 WINNER: {best_name} with {results[best_name]['six']} 6/6 hits")
    
    if results[best_name]['six_det']:
        for idx, nums in results[best_name]['six_det']:
            print(f"    🎉 Draw #{idx}: {nums}")
    
    # Save
    output = {
        'version': '16.0 OMEGA', 'n_test': n_test, 'port': PORT,
        'results': {}, 'elapsed': round(elapsed,1),
    }
    for name in approaches:
        bm = results[name]['hits']
        output['results'][name] = {
            'p3': round(sum(1 for m in bm if m>=3)/n_test*100,2),
            'p4': round(sum(1 for m in bm if m>=4)/n_test*100,2),
            'p5': round(sum(1 for m in bm if m>=5)/n_test*100,2),
            'p6': round(sum(1 for m in bm if m>=6)/n_test*100,2),
            'six': results[name]['six'], 'five': results[name]['five'],
        }
    path = os.path.join(os.path.dirname(__file__), 'models', 'v16_omega.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {path}")


if __name__ == '__main__':
    run_tournament()
