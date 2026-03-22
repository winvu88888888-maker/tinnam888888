"""
Master Predictor V14.0 — Nâng cấp với 20 tín hiệu + Column-Pool + Spectral
=============================================================================
V13: 15 signals + coordinate descent
V14: 20 signals + Column-Pool + Spectral Lag-10 + Regime-Adaptive
     + Enhanced Run-Length + Cross-Scale Agreement
     + Upgraded optimizer (finer deltas, 5 iterations)
     + Enhanced backtest metrics

Walk-forward backtest tự động → chọn trọng số tối ưu.
"""
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
import math
import warnings
warnings.filterwarnings('ignore')


class MasterPredictor:
    """V14.0: 20 tín hiệu + Column-Pool + Spectral → 1 dãy số."""
    
    VERSION = "V14.0"
    NUM_SIGNALS = 20
    
    def __init__(self, max_number, pick_count):
        self.max_number = max_number
        self.pick_count = pick_count
    
    def predict(self, data):
        """Return prediction result with backtest stats."""
        self.data = [d[:self.pick_count] for d in data]
        self.flat = [n for d in self.data for n in d]
        n = len(self.data)
        
        print(f"[Master V14] Analyzing {n} draws with {self.NUM_SIGNALS} signals...")
        
        # Step 1: Pre-compute column pool candidates
        self._column_pool = self._column_pool_candidates(self.data)
        print(f"  Column-Pool: {sum(len(v) for v in self._column_pool)} candidates across {self.pick_count} positions")
        
        # Step 2: Auto-tune weights via backtest
        best_weights = self._optimize_weights()
        print(f"  Weights optimized ({len(best_weights)} signals)")
        
        # Step 3: Generate prediction with optimized weights
        numbers, score_details = self._predict_with_weights(self.data, best_weights)
        
        # Step 4: Backtest to show accuracy
        bt = self._backtest(best_weights)
        print(f"  Backtest: {bt['avg']:.4f}/6 ({bt['improvement']:+.1f}%)")
        if bt.get('match_3plus', 0) > 0:
            print(f"  >=3 match: {bt['match_3plus']} times ({bt['hit_rate_3plus_pct']:.1f}%)")
        
        # Step 5: Confidence analysis
        confidence = self._confidence_analysis(score_details)
        
        print(f"[Master V14] Prediction: {numbers}")
        
        return {
            'numbers': numbers,
            'score_details': score_details[:15],
            'backtest': bt,
            'confidence': confidence,
            'version': self.VERSION,
            'method': f'Master AI V14 ({n} draws, {self.NUM_SIGNALS} signals, {bt["tests"]} backtested)',
        }
    
    def _column_pool_candidates(self, history):
        """
        Column-Pool strategy: per-position block prediction + hot number filtering.
        Returns: list of sets, one per position, containing candidate numbers.
        [Nguồn: dan_predictor.py, Phase 4 Position-Specific]
        """
        n = len(history)
        if n < 30:
            return [set(range(1, self.max_number + 1))] * self.pick_count
        
        # Extract per-position data
        pos_data = [[] for _ in range(self.pick_count)]
        for d in history:
            sd = sorted(d[:self.pick_count])
            for p in range(self.pick_count):
                if p < len(sd):
                    pos_data[p].append(sd[p])
        
        # Block definitions
        if self.max_number <= 45:
            blocks = {'A': (1, 9), 'B': (10, 19), 'C': (20, 29), 'D': (30, 39), 'E': (40, 45)}
        else:
            blocks = {'A': (1, 9), 'B': (10, 19), 'C': (20, 29), 'D': (30, 39), 'E': (40, 49), 'F': (50, 55)}
        
        def to_block(n_val):
            for bname, (lo, hi) in blocks.items():
                if lo <= n_val <= hi:
                    return bname
            return list(blocks.keys())[-1]
        
        candidates = []
        for pos in range(self.pick_count):
            h = pos_data[pos]
            if len(h) < 3:
                candidates.append(set(range(1, self.max_number + 1)))
                continue
            
            # Predict block using 3-gram pattern
            bseq = [to_block(v) for v in h]
            pred_blocks = None
            if len(bseq) >= 3:
                p3 = (bseq[-3], bseq[-2], bseq[-1])
                p3n = Counter()
                for i in range(len(bseq) - 3):
                    if (bseq[i], bseq[i+1], bseq[i+2]) == p3:
                        p3n[bseq[i+3]] += 1
                if sum(p3n.values()) >= 3:
                    pred_blocks = [b for b, _ in p3n.most_common(2)]
            if not pred_blocks:
                bc = Counter(bseq[-30:])
                pred_blocks = [b for b, _ in bc.most_common(3)]
            
            # Hot numbers within predicted blocks
            freq = Counter(h[-50:])
            valid = set()
            for b in pred_blocks:
                blo, bhi = blocks[b]
                for num in range(blo, bhi + 1):
                    if freq.get(num, 0) > 0:
                        valid.add(num)
            
            # Keep top by frequency, covering >= 70% cumulative
            ranked = sorted(valid, key=lambda x: -freq.get(x, 0))
            total_pct = 0
            hot = set()
            nh = len(h[-50:])
            for num in ranked:
                hot.add(num)
                total_pct += freq[num] / nh * 100
                if total_pct >= 70 or len(hot) >= 10:
                    break
            
            candidates.append(hot if hot else set(range(1, self.max_number + 1)))
        
        return candidates
    
    def _spectral_lag_signal(self, history, num):
        """
        Spectral Lag-10: detect 10-draw periodicity via autocorrelation.
        Returns score [0, 1] — higher = stronger lag-10 signal predicting appearance.
        [Nguồn: Phase 7, Spectral Periodicity exploit]
        """
        n = len(history)
        if n < 25:
            return 0.0
        
        # Binary appearance sequence (last 100 draws)
        window = min(100, n)
        seq = [1.0 if num in history[n - window + i] else 0.0 for i in range(window)]
        
        if len(seq) < 20:
            return 0.0
        
        # Compute autocorrelation at lag 10
        mean = np.mean(seq)
        var = np.var(seq)
        if var < 1e-10:
            return 0.0
        
        lag = 10
        if len(seq) <= lag:
            return 0.0
        
        autocorr = np.mean([(seq[i] - mean) * (seq[i + lag] - mean) 
                           for i in range(len(seq) - lag)]) / var
        
        # If autocorrelation is positive and the number appeared at t-10,
        # it's more likely to appear now
        appeared_at_lag = 1.0 if num in history[-lag] else 0.0
        
        # Score: positive autocorrelation * appeared_at_lag
        score = max(0, autocorr) * appeared_at_lag
        
        return float(min(score, 1.0))
    
    def _detect_regime(self, history, num, window=20):
        """
        Regime detection: hot/cold/neutral for a single number.
        Returns regime score modifier.
        [Nguồn: phase6_deep.py, _hmm_regime()]
        """
        n_draws = len(history)
        if n_draws < window:
            return 0.0
        
        freq_recent = sum(1 for d in history[-window:] if num in d) / window
        freq_old = sum(1 for d in history[-window*2:-window] if num in d) / window if n_draws > window*2 else freq_recent
        
        expected = self.pick_count / self.max_number
        trend = freq_recent - freq_old
        level = freq_recent - expected
        
        if trend > 0.03 and level > 0:
            # Hot regime: boost
            return trend * 5 + level * 3
        elif trend < -0.03 and level < 0:
            # Cold regime: mild penalty
            return -0.3
        else:
            return 0.0
    
    def _score_numbers(self, history, weights):
        """Score all numbers using 20 weighted signals. Returns scores dict."""
        n_draws = len(history)
        flat = [n for d in history for n in d]
        last = set(history[-1])
        scores = {}
        
        # Pre-compute shared data
        last_seen = {}
        for i, d in enumerate(history):
            for n in d:
                last_seen[n] = i
        
        exp_gap = self.max_number / self.pick_count
        freq_10 = Counter(n for d in history[-10:] for n in d)
        freq_30 = Counter(n for d in history[-30:] for n in d)
        freq_50 = Counter(n for d in history[-50:] for n in d)
        total_freq = Counter(flat)
        expected_total = len(flat) / self.max_number
        
        # Momentum
        r10 = Counter(n for d in history[-10:] for n in d)
        p10 = Counter(n for d in history[-20:-10] for n in d) if n_draws > 20 else r10
        
        # Pair network
        pair_scores = Counter()
        for d in history[-50:]:
            for pair in combinations(sorted(d), 2):
                pair_scores[pair] += 1
        
        # KNN: similar draws
        knn_scores = Counter()
        for i in range(len(history) - 2):
            overlap = len(set(history[i]) & last)
            if overlap >= 2:
                for n in history[i+1]:
                    knn_scores[n] += overlap ** 1.5
        
        # Pre-compute column pool membership (for S15)
        col_pool_flat = set()
        if hasattr(self, '_column_pool'):
            for cset in self._column_pool:
                col_pool_flat.update(cset)
        
        # Pre-compute cross-scale data (for S19)
        scale_appearances = {}
        scales = [5, 10, 20, 50, 100]
        for num in range(1, self.max_number + 1):
            count = 0
            for scale in scales:
                if n_draws >= scale:
                    if any(num in d for d in history[-scale:]):
                        count += 1
                elif any(num in d for d in history):
                    count += 1
            scale_appearances[num] = count
        
        for num in range(1, self.max_number + 1):
            s = [0.0] * self.NUM_SIGNALS
            
            # === Original 15 signals (S0-S14) ===
            
            # S0: Freq last 10 (hot)
            s[0] = freq_10.get(num, 0) / 10
            
            # S1: Freq last 30
            s[1] = freq_30.get(num, 0) / 30
            
            # S2: Freq last 50
            s[2] = freq_50.get(num, 0) / 50
            
            # S3: Gap overdue
            gap = n_draws - last_seen.get(num, 0)
            s[3] = max(0, gap / exp_gap - 0.8)
            
            # S4: Anti-repeat (negative if in last draw)
            s[4] = -1.0 if num in last else 0.0
            
            # S5: Momentum
            s[5] = (r10.get(num, 0) - p10.get(num, 0)) / 5
            
            # S6: Position frequency
            positions = []
            for d in history[-50:]:
                sd = sorted(d)
                if num in sd:
                    positions.append(sd.index(num))
            s[6] = len(positions) / 50
            
            # S7: Pair network bonus
            pair_bonus = 0
            for n in last:
                for pair, c in pair_scores.most_common(100):
                    if n in pair:
                        partner = pair[0] if pair[1] == n else pair[1]
                        if partner == num:
                            pair_bonus += c
            s[7] = pair_bonus / max(1, len(last) * 50)
            
            # S8: KNN conditional
            s[8] = knn_scores.get(num, 0) / max(1, max(knn_scores.values()) if knn_scores else 1)
            
            # S9: Frequency correction (under-represented boost)
            dev = (expected_total - total_freq.get(num, 0)) / max(1, expected_total)
            s[9] = max(0, dev)
            
            # S10: Run-length turning point
            curr_absence = 0
            for d in reversed(history):
                if num not in d:
                    curr_absence += 1
                else:
                    break
            s[10] = min(curr_absence / exp_gap, 2.0) if curr_absence > exp_gap * 0.7 else 0
            
            # S11: Temporal gradient (acceleration)
            f5 = sum(1 for d in history[-5:] if num in d) / 5
            f15 = sum(1 for d in history[-15:] if num in d) / 15
            f30 = sum(1 for d in history[-30:] if num in d) / 30
            v1 = f5 - f15
            v2 = f15 - f30
            s[11] = v1 + (v1 - v2) * 0.5
            
            # S12: Regime trend (hot streak boost)
            f_r = sum(1 for d in history[-15:] if num in d) / 15
            f_o = sum(1 for d in history[-45:-15] if num in d) / 30 if n_draws > 45 else f_r
            trend = f_r - f_o
            s[12] = max(0, trend) * 10
            
            # S13: Sum balance target
            avg_sum = np.mean([sum(d) for d in history[-20:]])
            target = avg_sum / self.pick_count
            s[13] = max(0, 1 - abs(num - target) / self.max_number)
            
            # S14: Anti-repeat double (second-to-last)
            s[14] = -0.5 if n_draws > 1 and num in history[-2] and num in last else 0
            
            # === NEW V14 signals (S15-S19) ===
            
            # S15: Column-Pool membership bonus
            # Numbers in the predicted column pool get a boost
            if col_pool_flat:
                s[15] = 1.0 if num in col_pool_flat else -0.3
            else:
                s[15] = 0.0
            
            # S16: Spectral Lag-10 periodicity
            s[16] = self._spectral_lag_signal(history, num)
            
            # S17: Regime-Adaptive modifier
            s[17] = self._detect_regime(history, num)
            
            # S18: Enhanced Run-Length (sigmoid turning point)
            # [Nguồn: phase6_deep.py, _run_length_predictor()]
            if curr_absence > 0:
                # Compute average absence run length
                seq = [1 if num in d else 0 for d in history]
                absence_runs = []
                run = 0
                for sv in seq:
                    if sv == 0:
                        run += 1
                    else:
                        if run > 0:
                            absence_runs.append(run)
                        run = 0
                avg_absence = np.mean(absence_runs) if absence_runs else exp_gap
                ratio = curr_absence / avg_absence if avg_absence > 0 else 0
                # Sigmoid: probability of turning point
                s[18] = 1 / (1 + math.exp(-3 * (ratio - 0.8)))
            else:
                s[18] = 0.0
            
            # S19: Cross-Scale Agreement bonus
            # Bonus when number appears across ≥4 of 5 time scales
            appearances_count = scale_appearances.get(num, 0)
            s[19] = max(0, (appearances_count - 3)) * 0.5  # 0 for ≤3, 0.5 for 4, 1.0 for 5
            
            # Weighted sum
            total_score = sum(w * si for w, si in zip(weights, s))
            scores[num] = total_score
        
        return scores
    
    def _predict_with_weights(self, history, weights):
        """Generate prediction using weights."""
        scores = self._score_numbers(history, weights)
        
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        numbers = sorted([n for n, _ in ranked[:self.pick_count]])
        
        max_s = max(s for _, s in ranked[:20]) if ranked else 1
        details = [{'number': int(n), 'score': round(float(s), 2),
                     'confidence': round(s / max(max_s, 0.01) * 100, 1),
                     'selected': n in numbers}
                    for n, s in ranked[:18]]
        
        return numbers, details
    
    def _optimize_weights(self):
        """Optimize 20 signal weights using coordinate descent on backtest."""
        n = len(self.data)
        train_end = min(n - 1, n - 50)
        test_range = range(max(60, train_end - 80), train_end)
        
        # Initial weights: original 15 + 5 new signals
        weights = [
            3.0, 2.0, 1.5, 2.5, 5.0,   # S0-S4: freq10, freq30, freq50, gap, anti-repeat
            2.0, 1.0, 3.0, 2.5, 1.5,    # S5-S9: momentum, position, pair, knn, freq_corr
            1.5, 2.0, 1.0, 0.5, 2.0,    # S10-S14: run-length, gradient, regime, sum, anti-dbl
            2.0, 3.0, 1.5, 2.0, 1.0,    # S15-S19: col-pool, spectral, regime-adapt, run-enh, cross-scale
        ]
        
        best_score = 0
        best_weights = weights[:]
        
        # Finer delta steps + more iterations for V14
        deltas = [-3, -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 3]
        
        for iteration in range(5):
            for w_idx in range(len(weights)):
                best_w = weights[w_idx]
                best_s = 0
                for delta in deltas:
                    weights[w_idx] = best_w + delta
                    # Quick eval
                    matches = []
                    for i in test_range:
                        scores = self._score_numbers(self.data[:i+1], weights)
                        pred = sorted(scores, key=lambda x: -scores[x])[:self.pick_count]
                        actual = set(self.data[i+1])
                        matches.append(len(set(pred) & actual))
                    avg = np.mean(matches) if matches else 0
                    if avg > best_s:
                        best_s = avg
                        best_w = weights[w_idx]
                weights[w_idx] = best_w
            
            if best_s > best_score:
                best_score = best_s
                best_weights = weights[:]
        
        return best_weights
    
    def _backtest(self, weights, test_count=200):
        """Walk-forward backtest with final weights. Enhanced V14 metrics."""
        n = len(self.data)
        start = max(60, n - test_count)
        matches = []
        for i in range(start, n - 1):
            scores = self._score_numbers(self.data[:i+1], weights)
            pred = sorted(scores, key=lambda x: -scores[x])[:self.pick_count]
            actual = set(self.data[i+1])
            matches.append(len(set(pred) & actual))
        
        if not matches:
            return {'avg': 0, 'max': 0, 'improvement': 0, 'tests': 0, 'distribution': {}}
        
        avg = float(np.mean(matches))
        rexp = self.pick_count ** 2 / self.max_number
        imp = (avg / rexp - 1) * 100 if rexp > 0 else 0
        
        m3plus = sum(1 for m in matches if m >= 3)
        total_tests = len(matches)
        
        result = {
            'avg': round(avg, 4),
            'max': int(max(matches)),
            'random_expected': round(rexp, 3),
            'improvement': round(float(imp), 2),
            'tests': total_tests,
            'match_3plus': m3plus,
            'match_4plus': sum(1 for m in matches if m >= 4),
            'match_5plus': sum(1 for m in matches if m >= 5),
            'match_6': sum(1 for m in matches if m >= 6),
            'hit_rate_3plus_pct': round(m3plus / total_tests * 100, 2) if total_tests > 0 else 0,
            'avg_last_50': round(float(np.mean(matches[-50:])), 4) if len(matches) >= 50 else round(avg, 4),
            'distribution': {str(k): int(v) for k, v in sorted(Counter(matches).items())},
        }
        
        return result
    
    def _confidence_analysis(self, score_details):
        """Analyze confidence of the prediction."""
        if not score_details:
            return {'level': 'low', 'score': 0}
        
        selected = [s for s in score_details if s.get('selected')]
        if not selected:
            return {'level': 'low', 'score': 0}
        
        avg_conf = np.mean([s['confidence'] for s in selected])
        min_conf = min(s['confidence'] for s in selected)
        
        # Gap between selected and non-selected
        non_selected = [s for s in score_details if not s.get('selected')]
        if non_selected:
            gap = selected[-1]['score'] - non_selected[0]['score']  
        else:
            gap = 0
        
        conf_score = avg_conf * 0.6 + min_conf * 0.3 + min(gap * 10, 10) * 0.1
        
        if conf_score >= 70:
            level = 'high'
        elif conf_score >= 40:
            level = 'medium'
        else:
            level = 'low'
        
        return {
            'level': level,
            'score': round(conf_score, 1),
            'avg_confidence': round(avg_conf, 1),
            'min_confidence': round(min_conf, 1),
        }
