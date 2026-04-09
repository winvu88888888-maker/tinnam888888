"""
MEGA HUNT: TÌM MỌI CÁCH TRÚNG CỘT 2-5
=========================================
Thử 15+ phương pháp dự đoán C2, C3, C4, C5.
Mỗi phương pháp đoán 1 giá trị/cột, đo % đúng TỪNG CỘT và % đúng CẢ 4.
Walk-forward backtest: 1,286 kỳ.
"""
import sys, os, math, time
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper.data_manager import get_mega645_numbers

data = get_mega645_numbers()
N = len(data)
WARMUP = 200
n_test = N - WARMUP
COLS = [1, 2, 3, 4]  # indices for C2-C5 (0-indexed)

# Prepare sorted columns
all_sorted = [sorted(d[:6]) for d in data]

print("=" * 100)
print("  🎯 MEGA HUNT — TÌM MỌI CÁCH TRÚNG CỘT 2-5")
print(f"  {N} kỳ | Backtest: {n_test} kỳ | 15+ phương pháp")
print("=" * 100)

# ═══════════════════════════════════════
# PREDICTION METHODS
# ═══════════════════════════════════════

def get_train_cols(te, window=200):
    """Get training column data."""
    cols = [[] for _ in range(6)]
    start = max(0, te - window)
    for d in all_sorted[start:te]:
        for i in range(6):
            cols[i].append(d[i])
    return cols

# ── METHOD 1: Mode (giá trị hay xuất hiện nhất) ──
def predict_mode(te):
    cols = get_train_cols(te)
    return {i: Counter(cols[i]).most_common(1)[0][0] for i in COLS}

# ── METHOD 2: Recent Mode (30 kỳ gần nhất) ──
def predict_recent_mode(te):
    cols = get_train_cols(te, 30)
    return {i: Counter(cols[i]).most_common(1)[0][0] for i in COLS}

# ── METHOD 3: Median ──
def predict_median(te):
    cols = get_train_cols(te)
    return {i: int(np.median(cols[i])) for i in COLS}

# ── METHOD 4: Weighted Recent (exponential decay) ──
def predict_weighted(te):
    cols = get_train_cols(te)
    result = {}
    for i in COLS:
        col = cols[i]
        n = len(col)
        weights = np.exp(np.linspace(-2, 0, n))  # recent = higher weight
        weighted_freq = defaultdict(float)
        for j, val in enumerate(col):
            weighted_freq[val] += weights[j]
        result[i] = max(weighted_freq.items(), key=lambda x: x[1])[0]
    return result

# ── METHOD 5: Markov-1 (transition from last value) ──
def predict_markov1(te):
    result = {}
    prev = all_sorted[te-1]
    for i in COLS:
        # Build transition table
        transitions = defaultdict(Counter)
        for j in range(max(0, te-500), te-1):
            transitions[all_sorted[j][i]][all_sorted[j+1][i]] += 1
        prev_val = prev[i]
        if prev_val in transitions and transitions[prev_val]:
            result[i] = transitions[prev_val].most_common(1)[0][0]
        else:
            cols = get_train_cols(te, 50)
            result[i] = Counter(cols[i]).most_common(1)[0][0]
    return result

# ── METHOD 6: Markov-2 (2-step transition) ──
def predict_markov2(te):
    result = {}
    if te < 3:
        return predict_mode(te)
    prev2 = all_sorted[te-2]
    for i in COLS:
        transitions = defaultdict(Counter)
        for j in range(max(0, te-500), te-2):
            transitions[all_sorted[j][i]][all_sorted[j+2][i]] += 1
        prev_val = prev2[i]
        if prev_val in transitions and transitions[prev_val]:
            result[i] = transitions[prev_val].most_common(1)[0][0]
        else:
            cols = get_train_cols(te, 50)
            result[i] = Counter(cols[i]).most_common(1)[0][0]
    return result

# ── METHOD 7: Delta-based (predict change) ──
def predict_delta(te):
    result = {}
    prev = all_sorted[te-1]
    for i in COLS:
        deltas = []
        for j in range(max(0, te-100), te-1):
            deltas.append(all_sorted[j+1][i] - all_sorted[j][i])
        avg_delta = int(round(np.mean(deltas))) if deltas else 0
        predicted = prev[i] + avg_delta
        predicted = max(1, min(45, predicted))
        result[i] = predicted
    return result

# ── METHOD 8: Momentum (recent trend) ──
def predict_momentum(te):
    result = {}
    for i in COLS:
        recent10 = [all_sorted[j][i] for j in range(max(0, te-10), te)]
        recent5 = recent10[-5:] if len(recent10) >= 5 else recent10
        if len(recent5) >= 3:
            # Linear trend
            x = np.arange(len(recent5))
            slope = np.polyfit(x, recent5, 1)[0]
            predicted = int(round(recent5[-1] + slope))
            predicted = max(1, min(45, predicted))
            result[i] = predicted
        else:
            result[i] = recent10[-1] if recent10 else 20
    return result

# ── METHOD 9: KNN Pattern Match (find similar past, predict next) ──
def predict_knn(te):
    last = all_sorted[te-1]
    best_sim = -1
    best_next = None
    for j in range(max(0, te-500), te-2):
        past = all_sorted[j]
        sim = sum(1 for k in range(6) if abs(past[k] - last[k]) <= 2)
        if sim > best_sim:
            best_sim = sim
            best_next = all_sorted[j+1]
    if best_next:
        return {i: best_next[i] for i in COLS}
    return predict_mode(te)

# ── METHOD 10: Conditional Chain (C1→C2→C3→C4→C5) ──
def predict_conditional(te):
    result = {}
    actual_c1 = None  # We don't know C1, use predicted
    prev = all_sorted[te-1]
    
    for i in COLS:
        # Predict C[i] given C[i-1] from previous draw
        prev_col_val = prev[i-1]  # value of column before this one, last draw
        transitions = defaultdict(Counter)
        for j in range(max(0, te-500), te):
            transitions[all_sorted[j][i-1]][all_sorted[j][i]] += 1
        
        if prev_col_val in transitions and transitions[prev_col_val]:
            result[i] = transitions[prev_col_val].most_common(1)[0][0]
        else:
            cols = get_train_cols(te, 50)
            result[i] = Counter(cols[i]).most_common(1)[0][0]
    return result

# ── METHOD 11: Gap/Due (most overdue value per column) ──
def predict_gap_due(te):
    result = {}
    for i in COLS:
        last_seen = {}
        for j in range(te):
            last_seen[all_sorted[j][i]] = j
        # Most overdue = longest gap since last seen
        cols = get_train_cols(te, 200)
        freq_vals = [v for v, _ in Counter(cols[i]).most_common(10)]
        overdue = sorted(freq_vals, key=lambda v: te - last_seen.get(v, 0), reverse=True)
        result[i] = overdue[0] if overdue else Counter(cols[i]).most_common(1)[0][0]
    return result

# ── METHOD 12: Repeat Last ──
def predict_repeat(te):
    prev = all_sorted[te-1]
    return {i: prev[i] for i in COLS}

# ── METHOD 13: Mean ──
def predict_mean(te):
    cols = get_train_cols(te, 100)
    return {i: int(round(np.mean(cols[i]))) for i in COLS}

# ── METHOD 14: Pair-transition (C2,C3 pair → next C2,C3) ──
def predict_pair_transition(te):
    result = {}
    prev = all_sorted[te-1]
    
    # For C2,C3: use (C2,C3) pair transition
    pair_23 = defaultdict(Counter)
    pair_45 = defaultdict(Counter)
    for j in range(max(0, te-500), te-1):
        key23 = (all_sorted[j][1], all_sorted[j][2])
        key45 = (all_sorted[j][3], all_sorted[j][4])
        pair_23[key23][(all_sorted[j+1][1], all_sorted[j+1][2])] += 1
        pair_45[key45][(all_sorted[j+1][3], all_sorted[j+1][4])] += 1
    
    # C2, C3
    key = (prev[1], prev[2])
    if key in pair_23 and pair_23[key]:
        best_pair = pair_23[key].most_common(1)[0][0]
        result[1] = best_pair[0]
        result[2] = best_pair[1]
    else:
        cols = get_train_cols(te, 50)
        result[1] = Counter(cols[1]).most_common(1)[0][0]
        result[2] = Counter(cols[2]).most_common(1)[0][0]
    
    # C4, C5
    key = (prev[3], prev[4])
    if key in pair_45 and pair_45[key]:
        best_pair = pair_45[key].most_common(1)[0][0]
        result[3] = best_pair[0]
        result[4] = best_pair[1]
    else:
        cols = get_train_cols(te, 50)
        result[3] = Counter(cols[3]).most_common(1)[0][0]
        result[4] = Counter(cols[4]).most_common(1)[0][0]
    
    return result

# ── METHOD 15: Ensemble Vote (top 5 methods vote) ──
def predict_ensemble(te):
    preds = [
        predict_mode(te),
        predict_recent_mode(te),
        predict_markov1(te),
        predict_weighted(te),
        predict_median(te),
    ]
    result = {}
    for i in COLS:
        votes = Counter(p[i] for p in preds)
        result[i] = votes.most_common(1)[0][0]
    return result

# ── METHOD 16: Ensemble±1 (vote considers ±1 neighbors) ──
def predict_ensemble_fuzzy(te):
    preds = [
        predict_mode(te),
        predict_recent_mode(te),
        predict_markov1(te),
        predict_weighted(te),
        predict_median(te),
        predict_momentum(te),
        predict_delta(te),
    ]
    result = {}
    for i in COLS:
        # Each prediction votes for itself AND neighbors ±1
        votes = Counter()
        for p in preds:
            v = p[i]
            votes[v] += 3
            votes[v-1] += 1
            votes[v+1] += 1
        result[i] = votes.most_common(1)[0][0]
    return result

# ── METHOD 17: Window-Adaptive (try multiple windows, pick best recent) ──
def predict_adaptive(te):
    result = {}
    for i in COLS:
        best_window = 50
        best_score = 0
        for w in [20, 30, 50, 80, 100, 150]:
            if te < w + 10:
                continue
            # Score: how well did mode of this window predict RECENT draws?
            correct = 0
            for t in range(max(WARMUP, te-20), te):
                train = [all_sorted[j][i] for j in range(max(0, t-w), t)]
                mode_v = Counter(train).most_common(1)[0][0]
                if all_sorted[t][i] == mode_v:
                    correct += 1
            if correct > best_score:
                best_score = correct
                best_window = w
        
        train = [all_sorted[j][i] for j in range(max(0, te-best_window), te)]
        result[i] = Counter(train).most_common(1)[0][0]
    return result

# ═══════════════════════════════════════
# BACKTEST ALL METHODS
# ═══════════════════════════════════════

METHODS = {
    'Mode-200':        predict_mode,
    'Recent-30':       predict_recent_mode,
    'Median':          predict_median,
    'Weighted-Exp':    predict_weighted,
    'Markov-1':        predict_markov1,
    'Markov-2':        predict_markov2,
    'Delta-Avg':       predict_delta,
    'Momentum':        predict_momentum,
    'KNN-Pattern':     predict_knn,
    'Conditional':     predict_conditional,
    'Gap-Due':         predict_gap_due,
    'Repeat-Last':     predict_repeat,
    'Mean':            predict_mean,
    'Pair-Trans':      predict_pair_transition,
    'Ensemble-5':      predict_ensemble,
    'Ensemble-Fuzzy':  predict_ensemble_fuzzy,
    'Adaptive':        predict_adaptive,
}

# Results: per-column accuracy and multi-column accuracy
results = {}
for name in METHODS:
    results[name] = {
        'per_col': {i: 0 for i in COLS},      # exact match per column
        'per_col_1': {i: 0 for i in COLS},     # within ±1
        'per_col_2': {i: 0 for i in COLS},     # within ±2
        'match_4': 0,   # all 4 exact
        'match_3': 0,   # 3 of 4 exact
        'match_2': 0,   # 2 of 4 exact
        'match_4_f1': 0, # all 4 within ±1
        'match_3_f1': 0, # 3 of 4 within ±1
    }

t0 = time.time()

for ti in range(n_test):
    te = WARMUP + ti
    actual = all_sorted[te]
    
    for name, func in METHODS.items():
        try:
            pred = func(te)
        except Exception:
            pred = {i: 20 for i in COLS}
        
        exact_count = 0
        fuzzy1_count = 0
        
        for i in COLS:
            p = pred.get(i, 0)
            a = actual[i]
            
            if p == a:
                results[name]['per_col'][i] += 1
                exact_count += 1
            if abs(p - a) <= 1:
                results[name]['per_col_1'][i] += 1
                fuzzy1_count += 1
            if abs(p - a) <= 2:
                results[name]['per_col_2'][i] += 1
        
        if exact_count >= 4: results[name]['match_4'] += 1
        if exact_count >= 3: results[name]['match_3'] += 1
        if exact_count >= 2: results[name]['match_2'] += 1
        if fuzzy1_count >= 4: results[name]['match_4_f1'] += 1
        if fuzzy1_count >= 3: results[name]['match_3_f1'] += 1
    
    if (ti+1) % 100 == 0:
        el = time.time() - t0
        eta = el/(ti+1)*(n_test-ti-1)
        print(f"  [{ti+1:4d}/{n_test}] {el:.0f}s ETA:{eta:.0f}s")
        sys.stdout.flush()

elapsed = time.time() - t0

# ═══════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════

print(f"\n{'═'*100}")
print(f"  🎯 KẾT QUẢ — ĐOÁN CHÍNH XÁC C2-C5 ({n_test} kỳ, {elapsed:.0f}s)")
print(f"{'═'*100}")

# ── A. Per-column EXACT accuracy ──
print(f"\n{'━'*100}")
print(f"  📊 A. ĐỘ CHÍNH XÁC TỪNG CỘT (Exact Match)")
print(f"{'━'*100}\n")

print(f"  {'Method':<18} │ {'C2':>7} {'C3':>7} {'C4':>7} {'C5':>7} │ {'TB':>7} │ {'Rank':>5}")
print(f"  {'─'*18} │ {'─'*7} {'─'*7} {'─'*7} {'─'*7} │ {'─'*7} │ {'─'*5}")

method_avg = {}
for name in METHODS:
    r = results[name]
    avg = sum(r['per_col'][i]/n_test*100 for i in COLS) / len(COLS)
    method_avg[name] = avg

ranked = sorted(method_avg.items(), key=lambda x: -x[1])
rank_map = {name: rank+1 for rank, (name, _) in enumerate(ranked)}

for name, avg in ranked:
    r = results[name]
    print(f"  {name:<18} │", end='')
    for i in COLS:
        pct = r['per_col'][i]/n_test*100
        print(f" {pct:>6.1f}%", end='')
    print(f" │ {avg:>6.2f}% │ #{rank_map[name]:>3}")

# ── B. Per-column ±1 accuracy ──
print(f"\n{'━'*100}")
print(f"  📊 B. ĐỘ CHÍNH XÁC TỪNG CỘT (±1, trong khoảng lân cận)")
print(f"{'━'*100}\n")

print(f"  {'Method':<18} │ {'C2±1':>7} {'C3±1':>7} {'C4±1':>7} {'C5±1':>7} │ {'TB':>7}")
print(f"  {'─'*18} │ {'─'*7} {'─'*7} {'─'*7} {'─'*7} │ {'─'*7}")

method_avg_f1 = {}
for name, _ in ranked:
    r = results[name]
    avg = sum(r['per_col_1'][i]/n_test*100 for i in COLS) / len(COLS)
    method_avg_f1[name] = avg
    print(f"  {name:<18} │", end='')
    for i in COLS:
        pct = r['per_col_1'][i]/n_test*100
        print(f" {pct:>6.1f}%", end='')
    print(f" │ {avg:>6.2f}%")

# ── C. Per-column ±2 accuracy ──
print(f"\n{'━'*100}")
print(f"  📊 C. ĐỘ CHÍNH XÁC TỪNG CỘT (±2)")
print(f"{'━'*100}\n")

print(f"  {'Method':<18} │ {'C2±2':>7} {'C3±2':>7} {'C4±2':>7} {'C5±2':>7} │ {'TB':>7}")
print(f"  {'─'*18} │ {'─'*7} {'─'*7} {'─'*7} {'─'*7} │ {'─'*7}")

for name, _ in ranked:
    r = results[name]
    avg = sum(r['per_col_2'][i]/n_test*100 for i in COLS) / len(COLS)
    print(f"  {name:<18} │", end='')
    for i in COLS:
        pct = r['per_col_2'][i]/n_test*100
        print(f" {pct:>6.1f}%", end='')
    print(f" │ {avg:>6.2f}%")

# ── D. Multi-column match ──
print(f"\n{'━'*100}")
print(f"  📊 D. ĐOÁN ĐÚNG NHIỀU CỘT CÙNG LÚC")
print(f"{'━'*100}\n")

print(f"  {'Method':<18} │ {'4/4 exact':>10} {'3/4 exact':>10} {'2/4 exact':>10} │ {'4/4 ±1':>8} {'3/4 ±1':>8}")
print(f"  {'─'*18} │ {'─'*10} {'─'*10} {'─'*10} │ {'─'*8} {'─'*8}")

for name, _ in ranked:
    r = results[name]
    m4 = r['match_4']
    m3 = r['match_3']
    m2 = r['match_2']
    m4f = r['match_4_f1']
    m3f = r['match_3_f1']
    
    print(f"  {name:<18} │ {m4:>10} {m3:>10} {m2:>10} │ {m4f:>8} {m3f:>8}")

# ── E. Best methods summary ──
print(f"\n{'━'*100}")
print(f"  ⭐ E. TÓM TẮT — PHƯƠNG PHÁP TỐT NHẤT")
print(f"{'━'*100}\n")

# Best per category
best_exact = ranked[0]
best_f1 = max(method_avg_f1.items(), key=lambda x: x[1])
best_multi = max(METHODS.keys(), key=lambda n: results[n]['match_3'])
best_multi4 = max(METHODS.keys(), key=lambda n: results[n]['match_4'])
best_multi_f1 = max(METHODS.keys(), key=lambda n: results[n]['match_4_f1'])

print(f"  🏆 Exact đúng TB cao nhất:  {best_exact[0]} = {best_exact[1]:.2f}%")
print(f"  🏆 ±1 đúng TB cao nhất:     {best_f1[0]} = {best_f1[1]:.2f}%")
print(f"  🏆 3/4 cột exact nhiều nhất: {best_multi} = {results[best_multi]['match_3']} lần")
print(f"  🏆 4/4 cột exact nhiều nhất: {best_multi4} = {results[best_multi4]['match_4']} lần")
print(f"  🏆 4/4 cột ±1 nhiều nhất:    {best_multi_f1} = {results[best_multi_f1]['match_4_f1']} lần")

# Per-column best method
print(f"\n  🏆 Phương pháp tốt nhất PER COLUMN (exact):")
for i in COLS:
    best_name = max(METHODS.keys(), key=lambda n: results[n]['per_col'][i])
    best_pct = results[best_name]['per_col'][i] / n_test * 100
    print(f"    C{i+1}: {best_name} = {best_pct:.1f}%")

print(f"\n  🏆 Phương pháp tốt nhất PER COLUMN (±1):")
for i in COLS:
    best_name = max(METHODS.keys(), key=lambda n: results[n]['per_col_1'][i])
    best_pct = results[best_name]['per_col_1'][i] / n_test * 100
    print(f"    C{i+1}: {best_name} = {best_pct:.1f}%")

# ── F. Mathematical ceiling ──
print(f"\n{'━'*100}")
print(f"  📊 F. GIỚI HẠN TOÁN HỌC")
print(f"{'━'*100}\n")

# If each column independently predicted at best rate, what's probability of all 4?
best_per_col_exact = []
best_per_col_f1 = []
for i in COLS:
    be = max(results[n]['per_col'][i] for n in METHODS) / n_test
    bf = max(results[n]['per_col_1'][i] for n in METHODS) / n_test
    best_per_col_exact.append(be)
    best_per_col_f1.append(bf)

prod_exact = 1
prod_f1 = 1
for p in best_per_col_exact: prod_exact *= p
for p in best_per_col_f1: prod_f1 *= p

print(f"  Best exact per column: C2={best_per_col_exact[0]*100:.1f}%, "
      f"C3={best_per_col_exact[1]*100:.1f}%, C4={best_per_col_exact[2]*100:.1f}%, "
      f"C5={best_per_col_exact[3]*100:.1f}%")
print(f"  → Nếu độc lập: P(4/4 exact) = {prod_exact*100:.4f}% "
      f"= ~1 lần / {int(1/prod_exact)} kỳ")

print(f"\n  Best ±1 per column:    C2={best_per_col_f1[0]*100:.1f}%, "
      f"C3={best_per_col_f1[1]*100:.1f}%, C4={best_per_col_f1[2]*100:.1f}%, "
      f"C5={best_per_col_f1[3]*100:.1f}%")
print(f"  → Nếu độc lập: P(4/4 ±1)   = {prod_f1*100:.4f}% "
      f"= ~1 lần / {int(1/prod_f1)} kỳ")

# Random baseline
random_exact = 1
for i in COLS:
    cols_all = [all_sorted[j][i] for j in range(N)]
    unique = len(set(cols_all))
    random_exact *= 1/unique
print(f"\n  Random baseline: P(4/4 exact) = {random_exact*100:.6f}% "
      f"= ~1 lần / {int(1/random_exact):,} kỳ")

# Improvement over random
actual_best_4 = max(results[n]['match_4'] for n in METHODS)
random_expected = n_test * random_exact
improvement = actual_best_4 / random_expected if random_expected > 0 else 0
print(f"  Thực tế: {actual_best_4} lần 4/4 exact vs random kỳ vọng {random_expected:.2f}")
print(f"  → Hơn random: {improvement:.0f}x")

print(f"\n{'═'*100}")
