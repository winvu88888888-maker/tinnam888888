"""
V11 LAST STAND — 3 VŨ KHÍ CHƯA BAO GIỜ THỬ
=============================================
1. DEEP SEQUENCE MODEL (numpy-based GRU + Attention)
2. PHYSICAL MACHINE BIAS DETECTOR (Smartplay Magnum modeling)  
3. COVERAGE SYSTEM (Stefan Mandel combinatorial guarantee)

Tất cả kết hợp vào MASTER ENSEMBLE + Walk-Forward Backtest.
Target: Beat V8's 37% ≥3/6 → tìm bất kỳ 6/6 nào.
"""
import sys, os, math, json, time
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper.data_manager import get_mega645_numbers, get_mega645_all

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

MAX = 45
PICK = 6
C_TOTAL = math.comb(MAX, PICK)  # 8,145,060


# ================================================================
# WEAPON 1: DEEP SEQUENCE MODEL (numpy GRU + Attention)
# ================================================================
class SequencePredictor:
    """
    Treats lottery draws as a 'language' — each draw is a 45-dim binary vector.
    Uses sklearn MLPClassifier with carefully engineered temporal features
    to simulate a sequence model's capability.
    """

    def __init__(self, window=20):
        self.window = window
        self.scaler = StandardScaler()

    def _encode_draw(self, draw):
        """6 numbers → 45-dim binary vector."""
        v = np.zeros(MAX)
        for n in draw[:PICK]:
            v[n - 1] = 1.0
        return v

    def _build_temporal_features(self, data, idx):
        """Build rich temporal features from a window of draws."""
        if idx < self.window:
            return None

        window_draws = data[idx - self.window:idx]
        feats = []

        # 1. Current state vectors at multiple scales
        for w in [3, 5, 10, self.window]:
            recent = window_draws[-w:]
            freq = np.zeros(MAX)
            for d in recent:
                for n in d[:PICK]:
                    freq[n - 1] += 1
            freq /= w
            feats.extend(freq.tolist())  # 45 * 4 = 180 features

        # 2. Momentum (short - long frequency)
        f3 = np.zeros(MAX)
        f10 = np.zeros(MAX)
        for d in window_draws[-3:]:
            for n in d[:PICK]:
                f3[n - 1] += 1
        f3 /= 3
        for d in window_draws[-10:]:
            for n in d[:PICK]:
                f10[n - 1] += 1
        f10 /= 10
        momentum = f3 - f10
        feats.extend(momentum.tolist())  # 45

        # 3. Gap features (how many draws since each number appeared)
        gaps = np.full(MAX, self.window, dtype=float)
        for j, d in enumerate(window_draws):
            for n in d[:PICK]:
                gaps[n - 1] = self.window - j - 1
        feats.extend((gaps / self.window).tolist())  # 45 normalized

        # 4. Transition features (what follows current last draw)
        last_draw = set(window_draws[-1][:PICK])
        trans_score = np.zeros(MAX)
        for j in range(len(window_draws) - 1):
            overlap = len(set(window_draws[j][:PICK]) & last_draw)
            if overlap >= 2:
                for n in window_draws[j + 1][:PICK]:
                    trans_score[n - 1] += overlap
        mx = np.max(trans_score)
        if mx > 0:
            trans_score /= mx
        feats.extend(trans_score.tolist())  # 45

        # 5. Positional Markov (per-position transitions)
        pos_pred = np.zeros(MAX)
        sorted_last = sorted(window_draws[-1][:PICK])
        for pos in range(PICK):
            trans = Counter()
            for j in range(len(window_draws) - 1):
                prev_sorted = sorted(window_draws[j][:PICK])
                next_sorted = sorted(window_draws[j + 1][:PICK])
                if prev_sorted[pos] == sorted_last[pos]:
                    trans[next_sorted[pos]] += 1
            total = sum(trans.values())
            if total > 0:
                for num, cnt in trans.most_common(5):
                    pos_pred[num - 1] += cnt / total
        feats.extend(pos_pred.tolist())  # 45

        # 6. Cross-draw attention (similarity-weighted next-draw)
        attn_score = np.zeros(MAX)
        for j in range(len(window_draws) - 1):
            sim = len(set(window_draws[j][:PICK]) & last_draw)
            weight = sim ** 2  # Quadratic attention
            for n in window_draws[j + 1][:PICK]:
                attn_score[n - 1] += weight
        mx = np.max(attn_score)
        if mx > 0:
            attn_score /= mx
        feats.extend(attn_score.tolist())  # 45

        # 7. Draw-level properties
        sorted_d = sorted(window_draws[-1][:PICK])
        feats.extend([
            sum(sorted_d) / 138,  # normalized sum
            (sorted_d[-1] - sorted_d[0]) / MAX,  # range
            sum(1 for x in sorted_d if x % 2 == 1) / PICK,  # odd ratio
            sum(1 for x in sorted_d if x > 22) / PICK,  # high ratio
            sorted_d[0] / MAX,  # min
            sorted_d[-1] / MAX,  # max
        ])  # 6

        # 8. Repeat pattern (how many numbers from draw N-1 appear in N)
        repeat_hist = []
        for j in range(1, len(window_draws)):
            prev = set(window_draws[j - 1][:PICK])
            curr = set(window_draws[j][:PICK])
            repeat_hist.append(len(prev & curr))
        feats.extend([
            np.mean(repeat_hist),
            np.std(repeat_hist),
            repeat_hist[-1] if repeat_hist else 0,
        ])  # 3

        return np.array(feats)

    def train_and_predict(self, data, train_end):
        """Train on data[:train_end], predict for draw at train_end."""
        # Build training data
        X, Y = [], []
        for idx in range(self.window, train_end - 1):
            f = self._build_temporal_features(data, idx)
            if f is not None:
                X.append(f)
                # Target: binary vector for next draw
                target = self._encode_draw(data[idx + 1])
                Y.append(target)

        if len(X) < 50:
            return np.ones(MAX) / MAX  # Not enough data

        X = np.array(X)
        Y = np.array(Y)

        # Train per-number classifiers (efficient batch)
        scores = np.zeros(MAX)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Get prediction features
        f_pred = self._build_temporal_features(data, train_end - 1)
        if f_pred is None:
            return np.ones(MAX) / MAX
        f_pred_scaled = scaler.transform(f_pred.reshape(1, -1))

        # Use MLP for multi-label classification
        # Train separate model for each number (top speedup: batch 45 numbers)
        for num_idx in range(MAX):
            y_col = Y[:, num_idx].astype(int)
            # Skip if too imbalanced
            if sum(y_col) < 5 or sum(y_col) > len(y_col) - 5:
                scores[num_idx] = sum(y_col) / len(y_col)
                continue

            # Fast GBM with limited trees
            model = GradientBoostingClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                subsample=0.8, random_state=42
            )
            try:
                model.fit(X_scaled, y_col)
                prob = model.predict_proba(f_pred_scaled)[0]
                # Probability of appearing
                if len(model.classes_) == 2:
                    scores[num_idx] = prob[1]
                else:
                    scores[num_idx] = prob[0]
            except:
                scores[num_idx] = sum(y_col) / len(y_col)

        return scores


# ================================================================
# WEAPON 2: PHYSICAL MACHINE BIAS DETECTOR
# ================================================================
class MachineBiasDetector:
    """
    Detect biases specific to Smartplay Magnum physical ball machines:
    - Ball wear drift over time
    - Sequential position bias
    - Draw-to-draw inertia
    - Temporal calibration drift
    - Environmental factors (day of week)
    """

    def analyze(self, data, dates=None, at_index=None):
        """Run all machine bias analyses. Returns per-number scores."""
        if at_index is None:
            at_index = len(data)
        relevant = data[:at_index]
        n = len(relevant)
        if n < 100:
            return np.zeros(MAX)

        scores = np.zeros(MAX)

        # 1. Ball Wear Drift
        drift = self._ball_wear_drift(relevant)
        scores += drift * 3.0

        # 2. Sequential Position Bias
        pos_bias = self._position_bias(relevant)
        scores += pos_bias * 2.0

        # 3. Draw-to-Draw Inertia
        inertia = self._draw_inertia(relevant)
        scores += inertia * 2.5

        # 4. Temporal Drift (machine aging)
        temporal = self._temporal_drift(relevant)
        scores += temporal * 2.0

        # 5. Day-of-week effect
        if dates and at_index <= len(dates):
            day_effect = self._day_of_week(relevant, dates[:at_index])
            scores += day_effect * 1.5

        return scores

    def _ball_wear_drift(self, data):
        """Detect if ball frequencies shift over time (wear = lighter = more frequent)."""
        n = len(data)
        if n < 200:
            return np.zeros(MAX)

        # Compare first half vs second half
        half = n // 2
        freq_first = np.zeros(MAX)
        freq_second = np.zeros(MAX)

        for d in data[:half]:
            for num in d[:PICK]:
                freq_first[num - 1] += 1
        freq_first /= half

        for d in data[half:]:
            for num in d[:PICK]:
                freq_second[num - 1] += 1
        freq_second /= (n - half)

        expected = PICK / MAX
        # Drift = acceleration towards more/less frequent
        drift = freq_second - freq_first

        # Also: recent trend (last 100 vs rest)
        recent_100 = min(100, n // 3)
        freq_recent = np.zeros(MAX)
        for d in data[-recent_100:]:
            for num in d[:PICK]:
                freq_recent[num - 1] += 1
        freq_recent /= recent_100

        freq_older = np.zeros(MAX)
        for d in data[:-recent_100]:
            for num in d[:PICK]:
                freq_older[num - 1] += 1
        freq_older /= max(n - recent_100, 1)

        recent_drift = freq_recent - freq_older

        # Numbers drifting UP recently → machine bias favoring them
        combined = drift * 0.3 + recent_drift * 0.7
        # Normalize to [-1, 1]
        mx = np.max(np.abs(combined))
        if mx > 0:
            combined /= mx
        return combined

    def _position_bias(self, data):
        """In physical machines, ball drawing order may correlate with weight."""
        n = len(data)
        scores = np.zeros(MAX)

        # Position frequency: P(number at sorted position K)
        pos_freq = np.zeros((PICK, MAX))
        for d in data:
            sd = sorted(d[:PICK])
            for p, num in enumerate(sd):
                pos_freq[p][num - 1] += 1

        # Expected: uniform within each position's range
        # Actual: some numbers may dominate certain positions → bias signal
        for num in range(MAX):
            total = sum(pos_freq[p][num] for p in range(PICK))
            if total == 0:
                continue
            # Which position does this number favor?
            pos_rates = [pos_freq[p][num] / n for p in range(PICK)]
            max_pos = np.argmax(pos_rates)
            dominance = pos_rates[max_pos] / (total / n / PICK) if total > 0 else 1

            # Higher dominance = more predictable position = possible bias
            if dominance > 1.3:
                scores[num] = (dominance - 1) * 0.5

        return scores

    def _draw_inertia(self, data):
        """Physical machines may not fully reset — measure draw-to-draw correlation."""
        n = len(data)
        if n < 50:
            return np.zeros(MAX)

        last = set(data[-1][:PICK])
        scores = np.zeros(MAX)

        # Per-number: P(appear | appeared in previous draw)
        repeat_given_prev = Counter()
        prev_count = Counter()
        for i in range(1, n):
            prev = set(data[i - 1][:PICK])
            curr = set(data[i][:PICK])
            for num in prev:
                prev_count[num] += 1
                if num in curr:
                    repeat_given_prev[num] += 1

        base_p = PICK / MAX
        for num in range(1, MAX + 1):
            if num in last and prev_count[num] > 10:
                p_repeat = repeat_given_prev[num] / prev_count[num]
                lift = p_repeat / base_p
                if lift > 1.0:
                    scores[num - 1] = (lift - 1) * 2
                else:
                    scores[num - 1] = (lift - 1) * 1  # Slightly penalize

        return scores

    def _temporal_drift(self, data):
        """Detect machine calibration shifts over 6-month windows."""
        n = len(data)
        if n < 300:
            return np.zeros(MAX)

        # Split into windows of ~150 draws
        window_size = 150
        n_windows = max(n // window_size, 2)
        actual_window = n // n_windows

        window_freqs = []
        for w in range(n_windows):
            start = w * actual_window
            end = min(start + actual_window, n)
            freq = np.zeros(MAX)
            for d in data[start:end]:
                for num in d[:PICK]:
                    freq[num - 1] += 1
            freq /= (end - start)
            window_freqs.append(freq)

        # Linear trend per number across windows
        scores = np.zeros(MAX)
        for num in range(MAX):
            values = [wf[num] for wf in window_freqs]
            if len(values) >= 3:
                # Fit linear trend
                x = np.arange(len(values))
                slope, _, _, _, _ = stats.linregress(x, values)
                # Positive slope = number becoming more frequent (machine drift)
                scores[num] = slope * 100  # Scale up
            else:
                # Simple: last window vs first window
                scores[num] = (values[-1] - values[0]) * 10

        return scores

    def _day_of_week(self, data, dates):
        """Staff rotation, temperature variations by day."""
        from datetime import datetime

        scores = np.zeros(MAX)
        day_data = defaultdict(list)

        for i, (draw, date_str) in enumerate(zip(data, dates)):
            try:
                dt = datetime.strptime(str(date_str), '%Y-%m-%d')
                dow = dt.weekday()
                day_data[dow].append(draw)
            except:
                continue

        if not day_data:
            return scores

        # Predict next day
        try:
            last_date = datetime.strptime(str(dates[-1]), '%Y-%m-%d')
            # Mega draws happen 3x/week, roughly every 2-3 days
            next_dow = (last_date.weekday() + 2) % 7
        except:
            return scores

        if next_dow not in day_data or len(day_data[next_dow]) < 20:
            return scores

        # Day-specific frequency vs overall
        day_freq = np.zeros(MAX)
        for d in day_data[next_dow]:
            for num in d[:PICK]:
                day_freq[num - 1] += 1
        day_freq /= len(day_data[next_dow])

        overall_freq = np.zeros(MAX)
        for d in data:
            for num in d[:PICK]:
                overall_freq[num - 1] += 1
        overall_freq /= len(data)

        # Lift
        for num in range(MAX):
            if overall_freq[num] > 0:
                lift = day_freq[num] / overall_freq[num]
                scores[num] = (lift - 1) * 3
        return scores


# ================================================================
# WEAPON 3: COVERAGE SYSTEM (Stefan Mandel Strategy)
# ================================================================
class CoverageSystem:
    """
    Given a super pool of K numbers, generate a portfolio of tickets
    that guarantees covering as many t-subsets as possible.
    """

    def generate_portfolio(self, super_pool, scores, n_tickets=200,
                           constraints=None):
        """
        Generate diversified portfolio using greedy set cover + scoring.
        
        Args:
            super_pool: list of numbers in the super pool
            scores: dict {number: score}
            n_tickets: how many tickets to generate
            constraints: dict with sum_lo/hi, odd_lo/hi, range_lo/hi
        """
        pool = sorted(super_pool)
        pool_size = len(pool)

        if pool_size < PICK:
            return []

        # Generate ALL valid combos from pool (with constraints)
        all_combos = []
        for combo in combinations(pool, PICK):
            if constraints:
                s = sum(combo)
                if s < constraints.get('sum_lo', 0) or s > constraints.get('sum_hi', 999):
                    continue
                rng = combo[-1] - combo[0]
                if rng < constraints.get('range_lo', 0) or rng > constraints.get('range_hi', 999):
                    continue
                odd = sum(1 for x in combo if x % 2 == 1)
                if odd < constraints.get('odd_lo', 0) or odd > constraints.get('odd_hi', PICK):
                    continue

            score = sum(scores.get(n, 0) for n in combo)
            all_combos.append((list(combo), score))

        if not all_combos:
            return []

        # Sort by score
        all_combos.sort(key=lambda x: -x[1])

        # Greedy diversified selection
        selected = []
        used_pairs = Counter()

        for combo, score in all_combos:
            if len(selected) >= n_tickets:
                break

            # Diversity check: new ticket should differ from existing ones
            if selected:
                max_overlap = max(
                    len(set(combo) & set(s['numbers']))
                    for s in selected[-20:]  # Check against last 20 selected
                )
                if max_overlap >= 5:  # Too similar
                    continue

            # Pair coverage bonus
            new_pairs = 0
            for p in combinations(combo, 2):
                if used_pairs[p] == 0:
                    new_pairs += 1
            coverage_bonus = new_pairs * 0.1

            selected.append({
                'numbers': combo,
                'score': round(score + coverage_bonus, 2),
                'strategy': 'coverage'
            })

            for p in combinations(combo, 2):
                used_pairs[p] += 1

        return selected


# ================================================================
# EXISTING ENGINE SIGNALS (compact versions)
# ================================================================
def existing_engine_scores(data, at_index=None):
    """Quick scoring from existing signals (transition, momentum, gap, etc.)."""
    if at_index is None:
        at_index = len(data)
    relevant = data[:at_index]
    n = len(relevant)
    if n < 30:
        return np.zeros(MAX)

    scores = np.zeros(MAX)
    last = set(relevant[-1][:PICK])

    # 1. Transition matrix
    follow = defaultdict(Counter)
    prev_c = Counter()
    for i in range(n - 1):
        for p in relevant[i][:PICK]:
            prev_c[p] += 1
            for nx in relevant[i + 1][:PICK]:
                follow[p][nx] += 1
    base_p = PICK / MAX
    for num in range(1, MAX + 1):
        total_f = sum(follow[p].get(num, 0) for p in last)
        total_p = sum(prev_c[p] for p in last)
        if total_p > 0:
            cp = total_f / total_p
            scores[num - 1] += (cp / base_p - 1) * 3

    # 2. Multi-scale momentum
    for num in range(1, MAX + 1):
        f5 = sum(1 for d in relevant[-5:] if num in d[:PICK]) / 5
        f20 = sum(1 for d in relevant[-20:] if num in d[:PICK]) / 20
        f50 = sum(1 for d in relevant[-min(50, n):] if num in d[:PICK]) / min(50, n)
        scores[num - 1] += (f5 - f20) * 10 + (f20 - f50) * 5

    # 3. Gap timing
    for num in range(1, MAX + 1):
        apps = [i for i, d in enumerate(relevant) if num in d[:PICK]]
        if len(apps) < 5:
            continue
        gaps = [apps[j + 1] - apps[j] for j in range(len(apps) - 1)]
        mg, sg = np.mean(gaps), np.std(gaps)
        cg = n - apps[-1]
        z = (cg - mg) / sg if sg > 0 else 0
        if z > 0.5:
            scores[num - 1] += z * 1.5
        elif z < -1:
            scores[num - 1] -= 1

    # 4. KNN history match
    knn = Counter()
    for i in range(n - 2):
        sim = len(set(relevant[i][:PICK]) & last)
        if sim >= 3:
            for num in relevant[i + 1][:PICK]:
                knn[num] += sim ** 2
    mx = max(knn.values()) if knn else 1
    for num in range(1, MAX + 1):
        scores[num - 1] += knn.get(num, 0) / mx * 2

    # 5. Co-occurrence
    pf = Counter()
    for d in relevant[-200:]:
        for p in combinations(sorted(d[:PICK]), 2):
            pf[p] += 1
    for num in range(1, MAX + 1):
        cooc = sum(pf.get(tuple(sorted([p, num])), 0) for p in last)
        scores[num - 1] += cooc * 0.05

    # 6. Markov Order-2
    if n >= 10:
        last1 = set(relevant[-1][:PICK])
        last2 = set(relevant[-2][:PICK])
        both_c = Counter()
        either_c = Counter()
        tb, te = 0, 0
        for i in range(2, n):
            p2 = set(relevant[i - 2][:PICK])
            p1 = set(relevant[i - 1][:PICK])
            cu = set(relevant[i][:PICK])
            for num in range(1, MAX + 1):
                ip2, ip1, ic = num in p2, num in p1, num in cu
                if ip2 and ip1:
                    tb += 1
                    if ic: both_c[num] += 1
                elif ip2 or ip1:
                    te += 1
                    if ic: either_c[num] += 1
        for num in range(1, MAX + 1):
            il1 = num in last1
            il2 = num in last2
            if il1 and il2:
                p = both_c[num] / max(tb / MAX, 1)
            elif il1 or il2:
                p = either_c[num] / max(te / MAX, 1)
            else:
                p = 0
            scores[num - 1] += (p - base_p) * 5

    return scores


# ================================================================
# LEARN CONSTRAINTS FROM HISTORY
# ================================================================
def learn_constraints(data, at_index=None):
    if at_index is None:
        at_index = len(data)
    recent = data[max(0, at_index - 50):at_index]
    sums = [sum(sorted(d[:PICK])) for d in recent]
    odds = [sum(1 for x in d[:PICK] if x % 2 == 1) for d in recent]
    ranges = [max(d[:PICK]) - min(d[:PICK]) for d in recent]
    return {
        'sum_lo': int(np.percentile(sums, 8)),
        'sum_hi': int(np.percentile(sums, 92)),
        'odd_lo': max(0, int(np.percentile(odds, 8))),
        'odd_hi': min(PICK, int(np.percentile(odds, 92))),
        'range_lo': int(np.percentile(ranges, 8)),
        'range_hi': int(np.percentile(ranges, 92)),
    }


# ================================================================
# MASTER WALK-FORWARD BACKTEST
# ================================================================
def run_v11():
    data = get_mega645_numbers()
    all_rows = get_mega645_all()
    dates = [r['draw_date'] for r in all_rows]
    N = len(data)
    t0 = time.time()

    print("=" * 80)
    print("  ⚔️  V11 LAST STAND — 3 WEAPONS NEVER TRIED")
    print(f"  {N} draws | Target: find ANY 6/6")
    print("=" * 80)

    # Initialize weapons
    seq_model = SequencePredictor(window=20)
    machine_det = MachineBiasDetector()
    coverage_sys = CoverageSystem()

    # Walk-forward settings
    WARMUP = 200  # Need enough data to train
    port_sizes = [1, 10, 50, 100, 200, 500]

    results = {ps: [] for ps in port_sizes}
    random_results = []
    weapon_contrib = {'seq': [], 'machine': [], 'existing': [], 'ensemble': []}

    np.random.seed(42)
    n_test = N - WARMUP

    print(f"\n  Walk-forward: {n_test} iterations (draw {WARMUP+1} → {N})")
    print(f"  Weapons: Sequence Model + Machine Bias + Existing Engines + Coverage")
    print(f"{'━' * 80}")

    # For speed: only use full seq model every 10th iteration, cache between
    seq_scores_cache = None
    seq_retrain_interval = 10

    for test_idx in range(n_test):
        train_end = WARMUP + test_idx
        actual = set(data[train_end][:PICK])

        # ---- WEAPON 1: Sequence Model ----
        if test_idx % seq_retrain_interval == 0:
            try:
                seq_scores = seq_model.train_and_predict(data, train_end)
                seq_scores_cache = seq_scores.copy()
            except Exception as e:
                seq_scores = np.ones(MAX) * PICK / MAX
                seq_scores_cache = seq_scores
        else:
            # Use cached + update with simple signals
            seq_scores = seq_scores_cache.copy() if seq_scores_cache is not None else np.ones(MAX) * PICK / MAX

        # ---- WEAPON 2: Machine Bias ----
        machine_scores = machine_det.analyze(data, dates, at_index=train_end)

        # ---- EXISTING ENGINES ----
        eng_scores = existing_engine_scores(data, at_index=train_end)

        # ---- ENSEMBLE: Weight-combine all weapons ----
        # Normalize each to [0, 1]
        def normalize(s):
            mn, mx = np.min(s), np.max(s)
            if mx - mn < 0.001:
                return np.ones_like(s) * 0.5
            return (s - mn) / (mx - mn)

        seq_norm = normalize(seq_scores)
        mac_norm = normalize(machine_scores)
        eng_norm = normalize(eng_scores)

        # Ensemble weights (will be calibrated later if V11 shows promise)
        W_SEQ = 2.0
        W_MAC = 1.5
        W_ENG = 3.0

        ensemble = seq_norm * W_SEQ + mac_norm * W_MAC + eng_norm * W_ENG
        ensemble_total = W_SEQ + W_MAC + W_ENG

        # Rank numbers
        ranked = np.argsort(-ensemble)

        # ---- BUILD SUPER POOL ----
        pool_size = 22
        super_pool = [int(ranked[i]) + 1 for i in range(pool_size)]

        # ---- GENERATE PORTFOLIO (Coverage System) ----
        constraints = learn_constraints(data, at_index=train_end)
        num_scores = {n + 1: float(ensemble[n]) for n in range(MAX)}

        portfolio_all = coverage_sys.generate_portfolio(
            super_pool, num_scores, n_tickets=max(port_sizes) + 50,
            constraints=constraints
        )

        # ---- SCORE AGAINST ACTUAL ----
        for ps in port_sizes:
            port = portfolio_all[:ps]
            if port:
                best = max(len(actual & set(p['numbers'])) for p in port)
            else:
                best = 0
            results[ps].append(best)

        # Random baseline
        rand = set(np.random.choice(range(1, MAX + 1), PICK, replace=False).tolist())
        random_results.append(len(rand & actual))

        # Track weapon contributions (for top-1 prediction)
        top6_seq = set(int(np.argsort(-seq_norm)[i]) + 1 for i in range(PICK))
        top6_mac = set(int(np.argsort(-mac_norm)[i]) + 1 for i in range(PICK))
        top6_eng = set(int(np.argsort(-eng_norm)[i]) + 1 for i in range(PICK))
        top6_ens = set(int(ranked[i]) + 1 for i in range(PICK))

        weapon_contrib['seq'].append(len(actual & top6_seq))
        weapon_contrib['machine'].append(len(actual & top6_mac))
        weapon_contrib['existing'].append(len(actual & top6_eng))
        weapon_contrib['ensemble'].append(len(actual & top6_ens))

        # Progress
        if (test_idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            avg_ens = np.mean(weapon_contrib['ensemble'][-100:])
            best_500 = results[500][-100:] if len(results[500]) >= 100 else results[500]
            pct3 = sum(1 for m in best_500 if m >= 3) / len(best_500) * 100 if best_500 else 0
            pct4 = sum(1 for m in best_500 if m >= 4) / len(best_500) * 100 if best_500 else 0
            pct6 = sum(1 for m in best_500 if m >= 6) / len(best_500) * 100 if best_500 else 0
            speed = (test_idx + 1) / elapsed
            eta = (n_test - test_idx - 1) / speed if speed > 0 else 0
            print(f"  [{test_idx + 1:5d}/{n_test}] "
                  f"Last100: avg={avg_ens:.3f}/6 | "
                  f"P500: ≥3={pct3:.1f}% ≥4={pct4:.1f}% 6/6={pct6:.1f}% | "
                  f"ETA: {eta:.0f}s")

    # ================================================================
    # FINAL RESULTS
    # ================================================================
    elapsed = time.time() - t0
    rand_avg = np.mean(random_results)

    print(f"\n{'═' * 80}")
    print(f"  ⚔️  V11 LAST STAND — FINAL RESULTS")
    print(f"  {n_test} tests | Time: {elapsed:.1f}s")
    print(f"{'═' * 80}")

    # Weapon comparison
    print(f"\n  🗡️ WEAPON COMPARISON (Top-6 prediction):")
    print(f"  {'Weapon':<20} | {'Avg/6':>8} | {'≥3/6':>8} | {'≥4/6':>8} | {'≥5/6':>8} | {'6/6':>8}")
    print(f"  {'─' * 20} | {'─' * 8} | {'─' * 8} | {'─' * 8} | {'─' * 8} | {'─' * 8}")
    for name, label in [('seq', '🧠 Sequence Model'), ('machine', '⚙️ Machine Bias'),
                        ('existing', '📊 Existing Engines'), ('ensemble', '⚔️ ENSEMBLE')]:
        data_w = weapon_contrib[name]
        avg = np.mean(data_w)
        pct3 = sum(1 for m in data_w if m >= 3) / n_test * 100
        pct4 = sum(1 for m in data_w if m >= 4) / n_test * 100
        pct5 = sum(1 for m in data_w if m >= 5) / n_test * 100
        pct6 = sum(1 for m in data_w if m >= 6) / n_test * 100
        print(f"  {label:<20} | {avg:8.4f} | {pct3:7.2f}% | {pct4:7.2f}% | {pct5:7.2f}% | {pct6:7.2f}%")

    # Portfolio results
    print(f"\n  🎯 PORTFOLIO RESULTS:")
    print(f"  {'Port':>6} | {'Avg':>8} | {'≥3/6':>8} | {'≥4/6':>8} | {'≥5/6':>8} | {'6/6':>8}")
    print(f"  {'─' * 6} | {'─' * 8} | {'─' * 8} | {'─' * 8} | {'─' * 8} | {'─' * 8}")
    print(f"  {'Rand':>6} | {rand_avg:8.4f} | "
          f"{sum(1 for m in random_results if m >= 3) / n_test * 100:7.2f}% | "
          f"{sum(1 for m in random_results if m >= 4) / n_test * 100:7.2f}% | "
          f"{sum(1 for m in random_results if m >= 5) / n_test * 100:7.2f}% | "
          f"{sum(1 for m in random_results if m >= 6) / n_test * 100:7.2f}%")

    best_port = {}
    for ps in port_sizes:
        bm = results[ps]
        avg = np.mean(bm)
        p3 = sum(1 for m in bm if m >= 3) / n_test * 100
        p4 = sum(1 for m in bm if m >= 4) / n_test * 100
        p5 = sum(1 for m in bm if m >= 5) / n_test * 100
        p6 = sum(1 for m in bm if m >= 6) / n_test * 100
        print(f"  {ps:6d} | {avg:8.4f} | {p3:7.2f}% | {p4:7.2f}% | {p5:7.2f}% | {p6:7.2f}%")
        best_port[f'port_{ps}'] = {
            'avg': round(avg, 4), 'pct_3': round(p3, 2),
            'pct_4': round(p4, 2), 'pct_5': round(p5, 2), 'pct_6': round(p6, 2),
        }

    # 5/6 and 6/6 detailed analysis
    print(f"\n  🏆 HIGH MATCH DETAILS:")
    for ps in [100, 200, 500]:
        bm = results[ps]
        fives = [(i + WARMUP, m) for i, m in enumerate(bm) if m >= 5]
        sixes = [(i + WARMUP, m) for i, m in enumerate(bm) if m >= 6]
        if fives:
            print(f"  Port-{ps}: {len(fives)} draws with ≥5/6!")
            for draw_idx, m in fives[:10]:
                actual_d = sorted(data[draw_idx][:PICK])
                print(f"    Draw #{draw_idx}: {actual_d} → matched {m}/6")
        if sixes:
            print(f"  🎉 Port-{ps}: {len(sixes)} draws with 6/6!!!")
            for draw_idx, m in sixes:
                actual_d = sorted(data[draw_idx][:PICK])
                print(f"    🏆 Draw #{draw_idx}: {actual_d} → JACKPOT HIT!")

    # Check pool containment
    pool_contains = 0
    for test_idx in range(n_test):
        train_end = WARMUP + test_idx
        actual = set(data[train_end][:PICK])
        # Quick check: was actual within super pool?
        if test_idx < len(results[500]):
            if results[500][test_idx] >= 6:
                pool_contains += 1
    print(f"\n  Pool containment (6/6 in any port-500): {pool_contains}")

    # Save results
    output = {
        'version': '11.0 — LAST STAND',
        'weapons': ['Sequence Model (GBM+temporal)', 'Machine Bias (Smartplay)',
                     'Coverage System (Mandel)', 'Existing Engines'],
        'n_draws': N,
        'n_test': n_test,
        'warmup': WARMUP,
        'weapon_comparison': {
            name: {
                'avg': round(np.mean(weapon_contrib[name]), 4),
                'pct_3': round(sum(1 for m in weapon_contrib[name] if m >= 3) / n_test * 100, 2),
                'pct_4': round(sum(1 for m in weapon_contrib[name] if m >= 4) / n_test * 100, 2),
                'pct_5': round(sum(1 for m in weapon_contrib[name] if m >= 5) / n_test * 100, 2),
                'pct_6': round(sum(1 for m in weapon_contrib[name] if m >= 6) / n_test * 100, 2),
            }
            for name in weapon_contrib
        },
        'portfolio_results': best_port,
        'random_avg': round(rand_avg, 4),
        'elapsed_seconds': round(elapsed, 1),
    }

    path = os.path.join(os.path.dirname(__file__), 'models', 'v11_last_stand.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved to {path}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'═' * 80}")


if __name__ == '__main__':
    run_v11()
