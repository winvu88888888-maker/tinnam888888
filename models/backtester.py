"""
Backtesting Engine - Walk-Forward Validation System.
Tests each model against REAL historical data to find which approach works best.

Logic:
1. Take draws 1..N as training data
2. Use each model to predict draw N+1
3. Compare prediction vs actual draw N+1
4. Score: count matching numbers (0-6)
5. Slide window: N+1 → N+2, repeat
6. Track accuracy per model across all iterations
7. Find the best model/strategy

Scoring Levels:
- 6/6 = JACKPOT (exact match)
- 5/6 = Excellent
- 4/6 = Very Good
- 3/6 = Good
- 2/6 = Fair
- 1/6 = Poor
- 0/6 = Miss
"""
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
import time
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)


class BacktestEngine:
    """Walk-forward backtesting against real lottery draws."""
    
    def __init__(self, max_number, pick_count):
        self.max_number = max_number
        self.pick_count = pick_count
        self.results = {}  # model_name -> list of results
    
    def run_backtest(self, all_data, start_from=50, step=1, max_tests=None, 
                     progress_callback=None):
        """
        Run walk-forward backtest on all models.
        
        Args:
            all_data: Complete historical data [[n1..n6], ...]  (oldest first)
            start_from: Minimum training data size before testing
            step: Step size between tests (1=test every draw)
            max_tests: Maximum number of test iterations
            progress_callback: function(current, total, model_name) for progress
        """
        total_draws = len(all_data)
        if total_draws < start_from + 5:
            return {'error': 'Not enough data for backtesting'}
        
        # Determine test range
        test_indices = list(range(start_from, total_draws - 1, step))
        if max_tests and len(test_indices) > max_tests:
            # Sample evenly across the range
            indices = np.linspace(0, len(test_indices) - 1, max_tests, dtype=int)
            test_indices = [test_indices[i] for i in indices]
        
        total_tests = len(test_indices)
        
        # Define all models to test
        model_configs = [
            {'name': 'Frequency Hot', 'func': self._predict_frequency_hot},
            {'name': 'Frequency Cold (Due)', 'func': self._predict_frequency_cold},
            {'name': 'Frequency Weighted', 'func': self._predict_frequency_weighted},
            {'name': 'Markov Chain', 'func': self._predict_markov},
            {'name': 'Bayesian Posterior', 'func': self._predict_bayesian},
            {'name': 'Gap/Overdue', 'func': self._predict_gap_overdue},
            {'name': 'Recency Weighted', 'func': self._predict_recency},
            {'name': 'Monte Carlo', 'func': self._predict_monte_carlo},
            {'name': 'Pattern Match', 'func': self._predict_pattern},
            {'name': 'Sliding Window', 'func': self._predict_sliding_window},
            {'name': 'Anti-Repeat', 'func': self._predict_anti_repeat},
            {'name': 'Ensemble Vote (V15)', 'func': self._predict_ensemble_vote},
            {'name': 'Momentum+Regime', 'func': self._predict_momentum_regime},
            {'name': 'Consensus Engine', 'func': self._predict_consensus},
            {'name': 'Deep Forensic V2', 'func': self._predict_deep_forensic},
            {'name': 'Rule Engine (69 rules)', 'func': self._predict_rule_engine},
            {'name': 'Random Baseline', 'func': self._predict_random},
        ]
        
        # Initialize results
        self.results = {}
        for cfg in model_configs:
            self.results[cfg['name']] = {
                'scores': [],         # Number of matches per test
                'details': [],        # Detailed per-test results
                'total_tests': 0,
                'best_score': 0,
                'match_distribution': Counter(),  # {0: N, 1: N, 2: N, ...}
            }
        
        print(f"\n{'='*60}")
        print(f"  BACKTESTING ENGINE - {total_tests} iterations")
        print(f"  Data: {total_draws} draws | Models: {len(model_configs)}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for test_idx, train_end in enumerate(test_indices):
            # Training data: all draws up to train_end
            train_data = all_data[:train_end + 1]
            # Actual next draw
            actual = set(all_data[train_end + 1])
            
            for cfg in model_configs:
                try:
                    predicted = cfg['func'](train_data)
                    predicted_set = set(predicted[:self.pick_count])
                    
                    # Count matching numbers
                    matches = len(predicted_set & actual)
                    
                    self.results[cfg['name']]['scores'].append(matches)
                    self.results[cfg['name']]['match_distribution'][matches] += 1
                    self.results[cfg['name']]['total_tests'] += 1
                    
                    if matches > self.results[cfg['name']]['best_score']:
                        self.results[cfg['name']]['best_score'] = matches
                        self.results[cfg['name']]['best_detail'] = {
                            'draw_index': train_end + 1,
                            'predicted': sorted(list(predicted_set)),
                            'actual': sorted(list(actual)),
                            'matches': matches
                        }
                    
                    # Store details for recent tests
                    if test_idx >= total_tests - 20:
                        self.results[cfg['name']]['details'].append({
                            'draw_index': train_end + 1,
                            'predicted': sorted(list(predicted_set)),
                            'actual': sorted(list(actual)),
                            'matches': matches
                        })
                    
                except Exception as e:
                    self.results[cfg['name']]['scores'].append(0)
                    self.results[cfg['name']]['match_distribution'][0] += 1
                    self.results[cfg['name']]['total_tests'] += 1
            
            # Progress
            if progress_callback:
                progress_callback(test_idx + 1, total_tests, '')
            
            if (test_idx + 1) % 100 == 0 or test_idx == 0:
                elapsed = time.time() - start_time
                print(f"  [{test_idx+1}/{total_tests}] {elapsed:.1f}s elapsed")
        
        elapsed = time.time() - start_time
        print(f"\n  Backtest complete in {elapsed:.1f}s")
        
        # Compute final statistics
        return self._compute_statistics()
    
    def _compute_statistics(self):
        """Compute final statistics for all models."""
        summary = []
        
        for name, data in self.results.items():
            scores = data['scores']
            if not scores:
                continue
            
            total = len(scores)
            avg_matches = np.mean(scores)
            
            # Calculate percentages for each match level
            dist = data['match_distribution']
            
            entry = {
                'model': name,
                'total_tests': total,
                'avg_matches': round(float(avg_matches), 3),
                'best_score': data['best_score'],
                'best_detail': data.get('best_detail'),
                'match_distribution': {
                    str(k): v for k, v in sorted(dist.items())
                },
                'match_pct': {
                    f'{k}_match': round(v / total * 100, 2) 
                    for k, v in sorted(dist.items())
                },
                'at_least_3': round(sum(v for k, v in dist.items() if k >= 3) / total * 100, 2),
                'at_least_2': round(sum(v for k, v in dist.items() if k >= 2) / total * 100, 2),
                'recent_results': data.get('details', [])[-10:]
            }
            summary.append(entry)
        
        # Sort by average matches descending
        summary.sort(key=lambda x: -x['avg_matches'])
        
        # Assign rank
        for rank, entry in enumerate(summary):
            entry['rank'] = rank + 1
        
        return {
            'models': summary,
            'best_model': summary[0] if summary else None,
            'total_iterations': summary[0]['total_tests'] if summary else 0,
        }
    
    # ======= PREDICTION MODELS =======
    
    def _predict_frequency_hot(self, data):
        """Pick the most frequent numbers overall."""
        freq = Counter()
        for draw in data:
            for n in draw:
                freq[n] += 1
        top = [n for n, _ in freq.most_common(self.pick_count)]
        return top
    
    def _predict_frequency_cold(self, data):
        """Pick the LEAST frequent numbers (they're 'due')."""
        freq = Counter()
        for draw in data:
            for n in draw:
                freq[n] += 1
        # Ensure all numbers present
        for n in range(1, self.max_number + 1):
            if n not in freq:
                freq[n] = 0
        bottom = [n for n, _ in freq.most_common()[-self.pick_count:]]
        return bottom
    
    def _predict_frequency_weighted(self, data):
        """Weighted combination of hot + overdue numbers."""
        total = len(data)
        freq = Counter()
        last_seen = {}
        
        for idx, draw in enumerate(data):
            for n in draw:
                freq[n] += 1
                last_seen[n] = idx
        
        scores = {}
        expected = total * self.pick_count / self.max_number
        
        for n in range(1, self.max_number + 1):
            f = freq.get(n, 0)
            gap = total - last_seen.get(n, 0)
            avg_gap = total / (f + 1)
            
            # Score: combine frequency deviation + overdue factor
            freq_score = f / expected if expected > 0 else 1
            overdue_score = gap / avg_gap if avg_gap > 0 else 1
            scores[n] = freq_score * 0.4 + overdue_score * 0.6
        
        top = sorted(scores, key=lambda x: -scores[x])[:self.pick_count]
        return top
    
    def _predict_markov(self, data):
        """Markov chain: what numbers follow the last draw?"""
        if len(data) < 2:
            return self._predict_random(data)
        
        transition = defaultdict(Counter)
        for i in range(1, len(data)):
            prev = set(data[i-1])
            curr = set(data[i])
            for p in prev:
                for c in curr:
                    transition[p][c] += 1
        
        last_draw = set(data[-1])
        scores = Counter()
        
        for prev_num in last_draw:
            for num, count in transition[prev_num].items():
                scores[num] += count
        
        if not scores:
            return self._predict_random(data)
        
        top = [n for n, _ in scores.most_common(self.pick_count)]
        return top
    
    def _predict_bayesian(self, data):
        """Bayesian posterior with exponential recency weight."""
        alpha = np.ones(self.max_number)
        total = len(data)
        
        for idx, draw in enumerate(data):
            weight = np.exp((idx - total) / max(total * 0.25, 1))
            for n in draw:
                alpha[n - 1] += weight
        
        posterior = alpha / alpha.sum()
        top_indices = np.argsort(posterior)[-self.pick_count:]
        return [i + 1 for i in top_indices]
    
    def _predict_gap_overdue(self, data):
        """Pick numbers that are most overdue based on their cycle."""
        total = len(data)
        last_seen = {}
        freq = Counter()
        
        for idx, draw in enumerate(data):
            for n in draw:
                last_seen[n] = idx
                freq[n] += 1
        
        overdue = {}
        for n in range(1, self.max_number + 1):
            f = freq.get(n, 0)
            if f == 0:
                overdue[n] = total  # Never appeared
            else:
                avg_gap = total / f
                current_gap = total - last_seen.get(n, 0)
                overdue[n] = current_gap / avg_gap  # overdue ratio
        
        top = sorted(overdue, key=lambda x: -overdue[x])[:self.pick_count]
        return top
    
    def _predict_recency(self, data):
        """Weighted by recency: recent draws count more."""
        decay = 0.95
        total = len(data)
        scores = defaultdict(float)
        
        for idx, draw in enumerate(data):
            weight = decay ** (total - 1 - idx)
            for n in draw:
                scores[n] += weight
        
        top = sorted(scores, key=lambda x: -scores[x])[:self.pick_count]
        return top
    
    def _predict_monte_carlo(self, data, n_sims=10000):
        """Monte Carlo simulation based on weighted frequencies."""
        recent = data[-min(50, len(data)):]
        
        weighted = np.zeros(self.max_number)
        for idx, draw in enumerate(recent):
            w = 1.0 + idx / len(recent)
            for n in draw:
                weighted[n - 1] += w
        
        weighted += 0.3
        probs = weighted / weighted.sum()
        
        counts = np.zeros(self.max_number)
        for _ in range(n_sims):
            draw = np.random.choice(self.max_number, self.pick_count, replace=False, p=probs)
            for d in draw:
                counts[d] += 1
        
        top_indices = np.argsort(counts)[-self.pick_count:]
        return [i + 1 for i in top_indices]
    
    def _predict_pattern(self, data):
        """Match structural pattern of recent draws."""
        if len(data) < 10:
            return self._predict_random(data)
        
        recent = data[-10:]
        mid = self.max_number // 2
        
        # Determine most common odd count
        odd_counts = [sum(1 for n in d if n % 2 == 1) for d in recent]
        target_odd = int(round(np.mean(odd_counts)))
        
        # Determine most common sum range
        sums = [sum(d) for d in recent]
        target_sum = int(np.mean(sums))
        sum_std = max(int(np.std(sums)), 10)
        
        # Score numbers
        freq = Counter()
        for d in data:
            for n in d:
                freq[n] += 1
        
        # Generate candidates and pick best pattern match
        best = None
        best_score = -1
        
        for _ in range(500):
            # Weighted random selection
            probs = np.array([freq.get(n, 1) for n in range(1, self.max_number + 1)], dtype=float)
            probs /= probs.sum()
            
            candidate = sorted(np.random.choice(
                range(1, self.max_number + 1), self.pick_count, replace=False, p=probs
            ).tolist())
            
            # Score pattern match
            odd = sum(1 for n in candidate if n % 2 == 1)
            s = sum(candidate)
            
            score = 0
            if odd == target_odd: score += 3
            elif abs(odd - target_odd) == 1: score += 1
            
            if abs(s - target_sum) < sum_std: score += 3
            elif abs(s - target_sum) < sum_std * 2: score += 1
            
            if score > best_score:
                best_score = score
                best = candidate
        
        return best if best else self._predict_random(data)
    
    def _predict_sliding_window(self, data):
        """Use a small sliding window of recent draws to pick numbers."""
        window = min(5, len(data))
        recent = data[-window:]
        
        freq = Counter()
        for draw in recent:
            for n in draw:
                freq[n] += 1
        
        top = [n for n, _ in freq.most_common(self.pick_count)]
        if len(top) < self.pick_count:
            remaining = [n for n in range(1, self.max_number + 1) if n not in top]
            top.extend(np.random.choice(remaining, self.pick_count - len(top), replace=False).tolist())
        return top
    
    def _predict_anti_repeat(self, data):
        """Pick numbers that did NOT appear in the last draw."""
        if len(data) == 0:
            return self._predict_random(data)
        
        last = set(data[-1])
        non_last = [n for n in range(1, self.max_number + 1) if n not in last]
        
        # Among non-last, pick by overall frequency
        freq = Counter()
        for draw in data:
            for n in draw:
                freq[n] += 1
        
        scores = {n: freq.get(n, 0) for n in non_last}
        top = sorted(scores, key=lambda x: -scores[x])[:self.pick_count]
        return top
    
    def _predict_ensemble_vote(self, data):
        """Ensemble of 5 sub-strategies with weighted voting (V15 core)."""
        n_draws = len(data)
        last = set(data[-1])
        votes = Counter()
        
        # Sub-strategy A: Weighted frequency (last 50)
        for j, d in enumerate(data[-50:]):
            w = 1 + j / 50
            for n in d:
                votes[n] += w * 0.3
        
        # Sub-strategy B: KNN similarity
        for i in range(len(data) - 2):
            overlap = len(set(data[i]) & last)
            if overlap >= 2:
                for n in data[i+1]:
                    votes[n] += overlap * 0.6
        
        # Sub-strategy C: Gap overdue
        last_seen = {}
        for i, d in enumerate(data):
            for n in d:
                last_seen[n] = i
        exp_gap = self.max_number / self.pick_count
        for n in range(1, self.max_number + 1):
            gap = n_draws - last_seen.get(n, 0)
            if gap > exp_gap * 1.2:
                votes[n] += (gap / exp_gap) * 1.8
        
        # Sub-strategy D: Pair network
        pair_scores = Counter()
        for d in data[-60:]:
            for pair in combinations(sorted(d[:self.pick_count]), 2):
                pair_scores[pair] += 1
        for n in last:
            for pair, c in pair_scores.most_common(100):
                if n in pair:
                    partner = pair[0] if pair[1] == n else pair[1]
                    if partner not in last:
                        votes[partner] += c * 0.1
        
        # Sub-strategy E: Momentum
        if n_draws > 20:
            r10 = Counter(n for d in data[-10:] for n in d)
            p10 = Counter(n for d in data[-20:-10] for n in d)
            for n in range(1, self.max_number + 1):
                mom = r10.get(n, 0) - p10.get(n, 0)
                if mom > 0:
                    votes[n] += mom * 1.5
        
        # Anti-repeat
        for n in last:
            votes[n] -= 8
        
        return [n for n, _ in votes.most_common(self.pick_count)]
    
    def _predict_momentum_regime(self, data):
        """Momentum + Regime detection strategy."""
        n_draws = len(data)
        scores = {}
        for num in range(1, self.max_number + 1):
            f5 = sum(1 for d in data[-5:] if num in d) / 5
            f15 = sum(1 for d in data[-15:] if num in d) / 15
            f30 = sum(1 for d in data[-30:] if num in d) / 30
            v1 = f5 - f15
            v2 = f15 - f30
            accel = v1 - v2
            f_r = f15
            f_o = sum(1 for d in data[-45:-15] if num in d) / 30 if n_draws > 45 else f_r
            trend = f_r - f_o
            scores[num] = f5 * 4 + v1 * 8 + accel * 4 + max(0, trend) * 15
            if num in data[-1]:
                scores[num] -= 3
        return sorted(scores, key=lambda x: -scores[x])[:self.pick_count]
    
    def _predict_consensus(self, data):
        """Use full Consensus Engine to produce golden set prediction."""
        try:
            from models.consensus_engine import ConsensusEngine
            engine = ConsensusEngine(self.max_number, self.pick_count)
            result = engine.predict(data, n_portfolio=5)
            return result.get('golden_set', self._predict_random(data))
        except Exception:
            return self._predict_ensemble_vote(data)
    
    def _predict_deep_forensic(self, data):
        """Use Deep Forensic V2 engine for prediction."""
        try:
            from models.deep_forensic import DeepForensic
            engine = DeepForensic(self.max_number, self.pick_count)
            result = engine.analyze(data)
            return result.get('primary', self._predict_random(data))
        except Exception:
            return self._predict_ensemble_vote(data)
    
    def _predict_rule_engine(self, data):
        """Use Rule Engine with 69 validated statistical rules."""
        try:
            from models.rule_engine import RuleEngine
            engine = RuleEngine(self.max_number, self.pick_count)
            result = engine.predict(data, n_portfolio=5)
            return result.get('primary', self._predict_random(data))
        except Exception:
            return self._predict_ensemble_vote(data)
    
    def _predict_random(self, data):
        """Random baseline for comparison."""
        return sorted(np.random.choice(
            range(1, self.max_number + 1), self.pick_count, replace=False
        ).tolist())


def run_backtest_for_type(lottery_type, max_tests=200):
    """Run full backtest for a lottery type."""
    from scraper.data_manager import get_mega645_numbers, get_power655_numbers
    
    if lottery_type == 'mega':
        data = get_mega645_numbers()
        max_num, pick = 45, 6
    else:
        data = get_power655_numbers()
        max_num, pick = 55, 6
    
    if len(data) < 60:
        return {'error': 'Not enough data'}
    
    engine = BacktestEngine(max_num, pick)
    results = engine.run_backtest(
        data, 
        start_from=50, 
        step=max(1, (len(data) - 50) // max_tests) if max_tests else 1,
        max_tests=max_tests
    )
    
    return results


if __name__ == '__main__':
    print("Running backtest on Mega 6/45...")
    results = run_backtest_for_type('mega', max_tests=300)
    
    if 'error' in results:
        print(f"Error: {results['error']}")
    else:
        print(f"\n{'='*60}")
        print(f"  RESULTS - {results['total_iterations']} iterations")
        print(f"{'='*60}\n")
        
        for m in results['models']:
            dist_str = ', '.join(f"{k}match={v}" for k, v in sorted(m['match_distribution'].items()))
            print(f"  #{m['rank']} {m['model']}")
            print(f"     Avg matches: {m['avg_matches']:.3f}/6")
            print(f"     Best: {m['best_score']}/6")
            print(f"     >=3 match: {m['at_least_3']}%")
            print(f"     Distribution: {dist_str}")
            print()
