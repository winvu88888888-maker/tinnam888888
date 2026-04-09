"""
Consensus Engine V1.0 — Golden Set Generator
=============================================
Fuses ALL prediction engines (Deep Forensic, Ultimate, Exploit) + 7 NEW signals
into a single consensus prediction with walk-forward calibrated weights.

NEW SIGNALS:
  1. Markov Order-2: P(X | last 2 draws)
  2. Delta Pattern: Change patterns between consecutive draws
  3. Cluster Momentum: Groups of numbers trending together
  4. Decade Flow: Cross-decade transition patterns
  5. Repeat Rate Analysis: Optimal repeat count from previous draw
  6. Positional Markov: Per-position transition matrix
  7. Wavelet Multi-Scale: Multi-resolution frequency decomposition

OUTPUT:
  - Golden Set: Single best prediction
  - Golden Portfolio: 50 diversified sets ranked by consensus score
  - Confidence score based on signal agreement
  - Heat map data (score per number)
"""
import sys
import os
import time
import math
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations

sys.stdout.reconfigure(encoding='utf-8')
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


class ConsensusEngine:
    """Fuse all engines + new signals into a Golden Set prediction."""

    def __init__(self, max_number, pick_count):
        self.max_number = max_number
        self.pick_count = pick_count

    # ================================================================
    # MAIN ENTRY POINT
    # ================================================================
    def predict(self, data, dates=None, n_portfolio=50):
        """
        Run full consensus analysis and produce Golden Set.

        Returns dict with:
          golden_set: best 6 numbers
          golden_portfolio: list of dicts with numbers + score
          heat_map: {number: consensus_score}
          confidence: 0-100
          engine_results: per-engine top picks
          signal_weights: calibrated weights
          backtest_summary: walk-forward performance
        """
        start_time = time.time()
        n = len(data)
        last = sorted(data[-1][:6])
        last_set = set(last)
        sorted_data = [sorted(d[:6]) for d in data]

        # ============================================================
        # PHASE 1: Compute ALL signals (existing + 7 new)
        # ============================================================
        signals = {}

        # --- Existing signals (from Deep Forensic / Ultimate) ---
        signals['transition'] = self._sig_transition(data)
        signals['momentum'] = self._sig_momentum(data)
        signals['gap_timing'] = self._sig_gap_timing(data)
        signals['lag_repeat'] = self._sig_lag_repeat(data)
        signals['cooccurrence'] = self._sig_cooccurrence(data)
        signals['position'] = self._sig_position(data)
        signals['streak'] = self._sig_streak(data)
        signals['knn'] = self._sig_knn(data)
        signals['fft_cycle'] = self._sig_fft_cycle(data)
        signals['ngram'] = self._sig_ngram(data)
        signals['context3'] = self._sig_context3(data)
        signals['entropy'] = self._sig_entropy(data)
        signals['triplet'] = self._sig_triplet(data)
        signals['seq_pattern'] = self._sig_seq_pattern(data)
        signals['runlength'] = self._sig_runlength(data)
        signals['pair_boost'] = self._sig_pair_boost(data)
        signals['consecutive'] = self._sig_consecutive(data)
        signals['oddeven'] = self._sig_oddeven(data)
        signals['highlow'] = self._sig_highlow(data)
        if dates:
            signals['day_profile'] = self._sig_day_profile(data, dates)

        # Vulnerability signals
        vuln = self._vulnerability_signals(data)
        signals['vuln_serial'] = vuln.get('serial', {})
        signals['vuln_gap'] = vuln.get('gap', {})
        signals['vuln_pair'] = vuln.get('pair', {})

        # --- 7 NEW SIGNALS ---
        signals['markov2'] = self._sig_markov_order2(data)
        signals['delta_pattern'] = self._sig_delta_pattern(data)
        signals['cluster_momentum'] = self._sig_cluster_momentum(data)
        signals['decade_flow'] = self._sig_decade_flow(data)
        signals['repeat_rate'] = self._sig_repeat_rate(data)
        signals['positional_markov'] = self._sig_positional_markov(data)
        signals['wavelet'] = self._sig_wavelet_multiscale(data)

        # ============================================================
        # PHASE 2: Walk-Forward Weight Calibration
        # ============================================================
        weights = self._walk_forward_calibrate(data, signals)

        # ============================================================
        # PHASE 3: Build consensus scores
        # ============================================================
        consensus = {num: 0.0 for num in range(1, self.max_number + 1)}
        for sig_name, sig_scores in signals.items():
            if not sig_scores:
                continue
            w = weights.get(sig_name, 1.0)
            vals = list(sig_scores.values())
            max_v = max(abs(v) for v in vals) if vals else 1
            if max_v < 0.001:
                max_v = 1
            for num, score in sig_scores.items():
                consensus[num] += (score / max_v) * w

        # Adaptive anti-repeat
        repeat_boost = self._learn_repeat_rate(data)
        for num in last_set:
            consensus[num] += repeat_boost  # Can be negative or slightly positive

        # ============================================================
        # PHASE 4: Multi-Pool Union (from Ultimate Engine concept)
        # ============================================================
        ranked = sorted(consensus.items(), key=lambda x: -x[1])
        pool_main = [num for num, _ in ranked[:20]]

        # Frequency pools at different windows
        freq_30 = Counter(num for d in data[-30:] for num in d[:6])
        pool_hot30 = [num for num, _ in freq_30.most_common(15)]

        freq_80 = Counter(num for d in data[-80:] for num in d[:6])
        pool_hot80 = [num for num, _ in freq_80.most_common(15)]

        # Overdue pool
        last_seen = {}
        for i, d in enumerate(data):
            for num in d[:6]:
                last_seen[num] = i
        pool_overdue = sorted(
            range(1, self.max_number + 1),
            key=lambda x: -(n - last_seen.get(x, 0))
        )[:15]

        # Transition pool (conditional on last draw)
        follow = Counter()
        for i in range(n - 1):
            for p in data[i][:6]:
                if p in last_set:
                    for nx in data[i + 1][:6]:
                        follow[nx] += 1
        pool_transition = [num for num, _ in follow.most_common(15)]

        # KNN pool
        knn_sc = Counter()
        for i in range(n - 2):
            sim = len(set(data[i][:6]) & last_set)
            if sim >= 3:
                for num in data[i + 1][:6]:
                    knn_sc[num] += sim ** 2
        pool_knn = [num for num, _ in knn_sc.most_common(15)]

        # Neighbor pool (±1, ±2 of last draw numbers)
        pool_neighbor = []
        for num in last_set:
            for delta in [-2, -1, 1, 2]:
                nb = num + delta
                if 1 <= nb <= self.max_number:
                    pool_neighbor.append(nb)

        # UNION all pools — build super pool
        MAX_POOL = 40
        raw_union = (set(pool_main) | set(pool_hot30) | set(pool_hot80) |
                     set(pool_overdue) | set(pool_transition) |
                     set(pool_knn) | set(pool_neighbor))
        super_pool = sorted(raw_union, key=lambda x: -consensus.get(x, 0))
        if len(super_pool) > MAX_POOL:
            super_pool = super_pool[:MAX_POOL]
        super_pool = sorted(super_pool)

        # ============================================================
        # PHASE 5: Learn constraints from history
        # ============================================================
        constraints = self._learn_constraints(data)

        # ============================================================
        # PHASE 6: Generate Golden Portfolio via exhaustive enum
        # ============================================================
        sum_lo = constraints['sum_lo']
        sum_hi = constraints['sum_hi']
        rng_lo = constraints['range_lo']
        rng_hi = constraints['range_hi']
        odd_lo = constraints['odd_lo']
        odd_hi = constraints['odd_hi']
        mid = self.max_number // 2

        portfolio = []
        for combo in combinations(super_pool, self.pick_count):
            s = sum(combo)
            if s < sum_lo or s > sum_hi:
                continue
            rng = combo[5] - combo[0]
            if rng < rng_lo or rng > rng_hi:
                continue
            odd = sum(1 for x in combo if x % 2 == 1)
            if odd < odd_lo or odd > odd_hi:
                continue

            score = sum(consensus.get(num, 0) for num in combo)
            portfolio.append({
                'numbers': list(combo),
                'score': round(score, 2),
            })

        portfolio.sort(key=lambda x: -x['score'])

        # Trim to n_portfolio (keep best)
        if len(portfolio) > n_portfolio:
            portfolio = portfolio[:n_portfolio]

        # Golden Set = best combo
        golden_set = portfolio[0]['numbers'] if portfolio else sorted(pool_main[:6])

        # ============================================================
        # PHASE 7: Engine comparison (optional — run sub-engines)
        # ============================================================
        engine_results = self._run_sub_engines(data, dates)

        # ============================================================
        # PHASE 8: Confidence Calculation
        # ============================================================
        confidence = self._calc_confidence(consensus, golden_set, signals, weights)

        # ============================================================
        # PHASE 9: Walk-forward backtest summary
        # ============================================================
        bt_summary = self._quick_backtest(data, signals, weights, constraints)

        elapsed = time.time() - start_time

        # Heat map = normalized consensus scores
        max_consensus = max(consensus.values()) if consensus else 1
        heat_map = {num: round(consensus[num] / max(max_consensus, 0.01) * 100, 1)
                    for num in range(1, self.max_number + 1)}

        return {
            'golden_set': golden_set,
            'golden_portfolio': portfolio,
            'total_sets': len(portfolio),
            'heat_map': heat_map,
            'consensus_scores': {num: round(s, 3) for num, s in ranked[:30]},
            'confidence': round(confidence, 1),
            'signal_weights': {k: round(v, 3) for k, v in
                               sorted(weights.items(), key=lambda x: -x[1])},
            'n_signals': len(signals),
            'super_pool_size': len(super_pool),
            'constraints': constraints,
            'engine_results': engine_results,
            'backtest_summary': bt_summary,
            'elapsed': round(elapsed, 2),
        }

    # ================================================================
    # 7 NEW SIGNALS
    # ================================================================

    def _sig_markov_order2(self, data):
        """Markov Order-2: P(X | last 2 draws)."""
        n = len(data)
        if n < 10:
            return {num: 0 for num in range(1, self.max_number + 1)}

        # Build bigram transition: (draw N-1 features, draw N features) → draw N+1
        # Simplified: for each number, count P(appear | appeared in N-1 AND N)
        last1 = set(data[-1][:6])
        last2 = set(data[-2][:6])

        both_count = Counter()  # num appeared after being in BOTH last 2
        either_count = Counter()  # num appeared after being in EITHER
        neither_count = Counter()  # num appeared after being in NEITHER
        total_both = 0
        total_either = 0
        total_neither = 0

        for i in range(2, n):
            prev2 = set(data[i - 2][:6])
            prev1 = set(data[i - 1][:6])
            curr = set(data[i][:6])
            for num in range(1, self.max_number + 1):
                in_prev2 = num in prev2
                in_prev1 = num in prev1
                in_curr = num in curr
                if in_prev2 and in_prev1:
                    total_both += 1
                    if in_curr:
                        both_count[num] += 1
                elif in_prev2 or in_prev1:
                    total_either += 1
                    if in_curr:
                        either_count[num] += 1
                else:
                    total_neither += 1
                    if in_curr:
                        neither_count[num] += 1

        scores = {}
        base_p = self.pick_count / self.max_number
        for num in range(1, self.max_number + 1):
            in_last1 = num in last1
            in_last2 = num in last2
            if in_last1 and in_last2:
                p = both_count[num] / max(total_both / self.max_number, 1)
            elif in_last1 or in_last2:
                p = either_count[num] / max(total_either / self.max_number, 1)
            else:
                p = neither_count[num] / max(total_neither / self.max_number, 1)
            scores[num] = (p - base_p) * 10
        return scores

    def _sig_delta_pattern(self, data):
        """Delta Pattern: track number-level changes between draws."""
        n = len(data)
        if n < 5:
            return {num: 0 for num in range(1, self.max_number + 1)}

        # For each number, compute delta sequence: +1 if appeared, -1 if not
        scores = {}
        for num in range(1, self.max_number + 1):
            deltas = [1 if num in d[:6] else -1 for d in data[-20:]]
            if len(deltas) < 10:
                scores[num] = 0
                continue

            # Pattern: look for -1,-1,-1,... → likely +1 next (regression to mean)
            # OR +1,+1,+1 → check if streak continues
            recent_sum = sum(deltas[-5:])
            mid_sum = sum(deltas[-10:-5])

            # Regression signal: prolonged absence → return expected
            if recent_sum <= -4:
                scores[num] = 2.0  # Very absent recently → likely return
            elif recent_sum >= 3:
                # Hot streak — check historical continuation rate
                seq = [1 if num in d[:6] else 0 for d in data]
                streak_cont = 0
                streak_total = 0
                s = 0
                for i in range(len(seq) - 1):
                    if seq[i]:
                        s += 1
                    else:
                        s = 0
                    if s >= 3:
                        streak_total += 1
                        if seq[i + 1]:
                            streak_cont += 1
                p_cont = streak_cont / max(streak_total, 1)
                scores[num] = (p_cont - self.pick_count / self.max_number) * 5
            else:
                # Acceleration signal
                accel = recent_sum - mid_sum
                scores[num] = accel * 0.3
        return scores

    def _sig_cluster_momentum(self, data):
        """Cluster Momentum: groups of numbers that move together."""
        n = len(data)
        if n < 30:
            return {num: 0 for num in range(1, self.max_number + 1)}

        # Build co-movement matrix: which numbers rise/fall together
        recent = data[-30:]
        window = 5
        scores = {}

        # Track momentum per number
        momentum = {}
        for num in range(1, self.max_number + 1):
            f_recent = sum(1 for d in recent[-window:] if num in d[:6]) / window
            f_older = sum(1 for d in recent[:window] if num in d[:6]) / window
            momentum[num] = f_recent - f_older

        # Numbers with positive momentum boost each other
        rising = [num for num, m in momentum.items() if m > 0]
        if len(rising) > 0:
            # Check pairwise co-occurrence among rising numbers
            pair_boost = Counter()
            for d in recent:
                ds = set(d[:6])
                rising_in = [r for r in rising if r in ds]
                for a in rising_in:
                    for b in rising_in:
                        if a != b:
                            pair_boost[a] += 1

            for num in range(1, self.max_number + 1):
                cluster_score = pair_boost.get(num, 0) * 0.1
                scores[num] = momentum.get(num, 0) * 3 + cluster_score
        else:
            scores = {num: momentum.get(num, 0) * 3
                      for num in range(1, self.max_number + 1)}
        return scores

    def _sig_decade_flow(self, data):
        """Decade Flow: how numbers flow between decade groups across draws."""
        n = len(data)
        if n < 20:
            return {num: 0 for num in range(1, self.max_number + 1)}

        def get_decade(num):
            if num <= 9:
                return 0
            elif num <= 19:
                return 1
            elif num <= 29:
                return 2
            elif num <= 39:
                return 3
            else:
                return 4

        # Count decade distribution change
        last_decades = [get_decade(num) for num in data[-1][:6]]
        dec_count_last = Counter(last_decades)

        # Transition: after decade distribution D(N), what's D(N+1)?
        transition = defaultdict(Counter)
        for i in range(1, n):
            prev_dec = tuple(sorted(get_decade(x) for x in data[i - 1][:6]))
            curr_dec = Counter(get_decade(x) for x in data[i][:6])
            for d, c in curr_dec.items():
                transition[prev_dec][d] += c

        this_pattern = tuple(sorted(get_decade(x) for x in data[-1][:6]))
        expected_dec = transition.get(this_pattern, Counter())

        # Score: boost numbers in expected decades
        scores = {}
        total_exp = sum(expected_dec.values()) or 1
        for num in range(1, self.max_number + 1):
            d = get_decade(num)
            dec_prob = expected_dec.get(d, 0) / total_exp
            avg_prob = 1.0 / 5
            scores[num] = (dec_prob - avg_prob) * 5
        return scores

    def _sig_repeat_rate(self, data):
        """Repeat Rate: how many numbers from last draw typically repeat."""
        n = len(data)
        if n < 20:
            return {num: 0 for num in range(1, self.max_number + 1)}

        # Historical repeat counts
        repeats = []
        for i in range(1, n):
            prev = set(data[i - 1][:6])
            curr = set(data[i][:6])
            repeats.append(len(prev & curr))

        avg_repeat = np.mean(repeats)
        # Typically 0-2 numbers repeat from last draw

        # Which numbers from last draw are most likely to repeat?
        last_set = set(data[-1][:6])
        repeat_probs = {}
        for num in last_set:
            # How often does this number repeat when it appeared?
            appeared_then_repeated = 0
            appeared_count = 0
            for i in range(n - 1):
                if num in data[i][:6]:
                    appeared_count += 1
                    if num in data[i + 1][:6]:
                        appeared_then_repeated += 1
            repeat_probs[num] = (appeared_then_repeated / max(appeared_count, 1))

        # Score: give repeat-likely numbers in last draw a boost
        scores = {}
        for num in range(1, self.max_number + 1):
            if num in last_set:
                rp = repeat_probs.get(num, 0)
                base = self.pick_count / self.max_number
                scores[num] = (rp - base) * 8  # Can be positive or negative
            else:
                scores[num] = 0
        return scores

    def _sig_positional_markov(self, data):
        """Positional Markov: transition matrix for each sorted position."""
        n = len(data)
        if n < 30:
            return {num: 0 for num in range(1, self.max_number + 1)}

        sorted_data = [sorted(d[:6]) for d in data]
        scores = {num: 0.0 for num in range(1, self.max_number + 1)}

        for pos in range(self.pick_count):
            # Build transition: value at pos in draw N → value at pos in draw N+1
            trans = defaultdict(Counter)
            for i in range(n - 1):
                prev_val = sorted_data[i][pos]
                next_val = sorted_data[i + 1][pos]
                trans[prev_val][next_val] += 1

            last_val = sorted_data[-1][pos]
            preds = trans.get(last_val, Counter())
            total = sum(preds.values()) or 1

            for num, cnt in preds.most_common(5):
                prob = cnt / total
                base = 1 / self.max_number
                scores[num] += (prob - base) * 4
        return scores

    def _sig_wavelet_multiscale(self, data):
        """Wavelet-like multi-scale decomposition using Haar averages."""
        n = len(data)
        if n < 64:
            return {num: 0 for num in range(1, self.max_number + 1)}

        scores = {}
        for num in range(1, self.max_number + 1):
            seq = np.array([1.0 if num in d[:6] else 0.0 for d in data[-64:]])
            mean_val = np.mean(seq)

            # Multi-scale decomposition: windows of 4, 8, 16, 32
            scale_signals = []
            for scale in [4, 8, 16, 32]:
                if len(seq) < scale:
                    continue
                n_blocks = len(seq) // scale
                block_means = [np.mean(seq[i * scale:(i + 1) * scale])
                               for i in range(n_blocks)]
                if len(block_means) >= 2:
                    # Trend at this scale: last block vs previous
                    trend = block_means[-1] - block_means[-2]
                    scale_signals.append(trend)

            if scale_signals:
                # Combine: weight shorter scales more
                weights_s = [4, 3, 2, 1][:len(scale_signals)]
                weighted = sum(s * w for s, w in zip(scale_signals, weights_s))
                scores[num] = weighted * 5
            else:
                scores[num] = 0
        return scores

    # ================================================================
    # EXISTING SIGNALS (compact — shared with Ultimate Engine)
    # ================================================================

    def _sig_transition(self, d):
        n, l = len(d), set(d[-1][:6])
        f, pc = defaultdict(Counter), Counter()
        for i in range(n - 1):
            for p in d[i][:6]:
                pc[p] += 1
                for x in d[i + 1][:6]:
                    f[p][x] += 1
        b = self.pick_count / self.max_number
        return {num: ((sum(f[p].get(num, 0) for p in l) /
                       max(sum(pc[p] for p in l), 1)) / b - 1) * 3
                for num in range(1, self.max_number + 1)}

    def _sig_momentum(self, d):
        n = len(d)
        return {num: ((sum(1 for x in d[-5:] if num in x[:6]) / 5 -
                        sum(1 for x in d[-20:] if num in x[:6]) / 20) * 10 +
                       (sum(1 for x in d[-20:] if num in x[:6]) / 20 -
                        sum(1 for x in d[-50:] if num in x[:6]) / 50) * 5)
                if n >= 50 else 0
                for num in range(1, self.max_number + 1)}

    def _sig_gap_timing(self, d):
        n = len(d)
        s = {}
        for num in range(1, self.max_number + 1):
            a = [i for i, x in enumerate(d) if num in x[:6]]
            if len(a) < 5:
                s[num] = 0
                continue
            g = [a[j + 1] - a[j] for j in range(len(a) - 1)]
            mg, sg = np.mean(g), np.std(g)
            z = (n - a[-1] - mg) / sg if sg > 0 else 0
            s[num] = (z * 1.5 + (sum(1 for x in g if x <= n - a[-1]) /
                                  len(g)) * 2) if z > 0.5 else (-1 if z < -1 else 0)
        return s

    def _sig_lag_repeat(self, d):
        n, ls = len(d), {}
        lg = defaultdict(lambda: defaultdict(int))
        for i, x in enumerate(d):
            for num in x[:6]:
                if num in ls:
                    lg[num][i - ls[num]] += 1
                ls[num] = i
        s = {}
        for num in range(1, self.max_number + 1):
            cl = n - ls.get(num, 0)
            if num not in lg:
                s[num] = 0
                continue
            t = sum(lg[num].values())
            gl = []
            for l_v, c in lg[num].items():
                gl.extend([l_v] * c)
            med = np.median(gl) if gl else self.max_number / self.pick_count
            s[num] = (lg[num].get(cl, 0) / t * 5 +
                      max(0, cl / med - 1) * 2) if t > 0 else 0
        return s

    def _sig_cooccurrence(self, d):
        l = set(d[-1][:6])
        pf = Counter()
        for x in d[-200:]:
            for p in combinations(sorted(x[:self.pick_count]), 2):
                pf[p] += 1
        s = {num: sum(pf.get(tuple(sorted([p, num])), 0)
                      for p in l) * 0.1
             for num in range(1, self.max_number + 1)}
        tf = Counter()
        for x in d[-150:]:
            for t in combinations(sorted(x[:self.pick_count]), 3):
                tf[t] += 1
        for t, c in tf.most_common(500):
            if c < 2:
                break
            ts = set(t)
            ov = ts & l
            if len(ov) == 2:
                m = (ts - l).pop()
                s[m] = s.get(m, 0) + c * 0.5
        return s

    def _sig_position(self, d):
        n = len(d)
        pf = [Counter() for _ in range(self.pick_count)]
        for x in d:
            sd = sorted(x[:self.pick_count])
            for p, num in enumerate(sd):
                pf[p][num] += 1
        return {num: sum(pf[p].get(num, 0) for p in range(self.pick_count)) / n
                for num in range(1, self.max_number + 1)}

    def _sig_streak(self, d):
        s = {}
        for num in range(1, self.max_number + 1):
            c = 0
            for x in reversed(d):
                if num not in x[:6]:
                    c += 1
                else:
                    break
            s[num] = c * 0.1 if c >= 10 else 0
        return s

    def _sig_knn(self, d):
        n, l = len(d), set(d[-1][:6])
        ks = Counter()
        for i in range(n - 2):
            sim = len(set(d[i][:6]) & l)
            if sim >= 3:
                for num in d[i + 1][:6]:
                    ks[num] += sim ** 2
        mx = max(ks.values()) if ks else 1
        return {num: ks.get(num, 0) / mx * 3
                for num in range(1, self.max_number + 1)}

    def _sig_fft_cycle(self, d):
        s, w = {}, min(200, len(d))
        for num in range(1, self.max_number + 1):
            seq = np.array([1.0 if num in x[:6] else 0.0 for x in d[-w:]])
            if len(seq) < 30:
                s[num] = 0
                continue
            sc = seq - np.mean(seq)
            ft = np.fft.rfft(sc)
            pw = np.abs(ft) ** 2
            if len(pw) < 3:
                s[num] = 0
                continue
            fr = np.fft.rfftfreq(len(sc))
            pi = np.argmax(pw[2:]) + 2
            pf_ = fr[pi] if pi < len(fr) else 0
            pp = pw[pi] if pi < len(pw) else 0
            sr = pp / (np.sum(pw[1:]) + 1e-10)
            if sr > 0.15 and pf_ > 0:
                s[num] = sr * max(0, math.cos(
                    2 * math.pi * ((len(seq) % (1 / pf_)) / (1 / pf_)))) * 3
            else:
                s[num] = 0
        return s

    def _sig_ngram(self, d):
        bg = defaultdict(Counter)
        for i in range(1, len(d)):
            for pn in d[i - 1][:6]:
                for cn in d[i][:6]:
                    bg[pn][cn] += 1
        sc = Counter()
        for pn in d[-1][:6]:
            t = sum(bg[pn].values())
            if t > 0:
                for nn, cnt in bg[pn].most_common(10):
                    sc[nn] += cnt / t
        return {num: sc.get(num, 0) for num in range(1, self.max_number + 1)}

    def _sig_context3(self, d):
        n = len(d)
        sc = Counter()
        if n < 20:
            return {num: 0 for num in range(1, self.max_number + 1)}
        l3 = [set(x[:6]) for x in d[-3:]]
        for i in range(3, n - 1):
            h3 = [set(x[:6]) for x in d[i - 3:i]]
            sim = sum(len(h3[j] & l3[j]) for j in range(3))
            if sim >= 4:
                for num in d[i][:6]:
                    sc[num] += sim ** 2
        mx = max(sc.values()) if sc else 1
        return {num: sc.get(num, 0) / max(1, mx) * 3
                for num in range(1, self.max_number + 1)}

    def _sig_entropy(self, d):
        n = len(d)
        if n < 60:
            return {num: 0 for num in range(1, self.max_number + 1)}
        s = {}
        for num in range(1, self.max_number + 1):
            seq = [1 if num in x[:6] else 0 for x in d[-60:]]
            tr = {0: [0, 0], 1: [0, 0]}
            for i in range(1, len(seq)):
                tr[seq[i - 1]][seq[i]] += 1
            cs = seq[-1]
            t = sum(tr[cs])
            pa = tr[cs][1] / t if t > 0 else self.pick_count / self.max_number
            ent = 0
            for st in [0, 1]:
                tt = sum(tr[st])
                if tt == 0:
                    continue
                for c in tr[st]:
                    if c > 0:
                        p = c / tt
                        ent -= p * math.log2(p) * (tt / len(seq))
            s[num] = pa * max(0, 1 - ent)
        return s

    def _sig_triplet(self, d):
        n, l = len(d), set(d[-1][:6])
        sc = Counter()
        if n < 50:
            return {num: 0 for num in range(1, self.max_number + 1)}
        tf = Counter()
        for x in (d[-100:] if n > 100 else d):
            for t in combinations(sorted(x[:self.pick_count]), 3):
                tf[t] += 1
        for t, c in tf.most_common(300):
            if c < 2:
                break
            ts = set(t)
            ov = ts & l
            if len(ov) >= 2:
                for num in ts - l:
                    sc[num] += c * len(ov)
        mx = max(sc.values()) if sc else 1
        return {num: sc.get(num, 0) / max(1, mx) * 2.5
                for num in range(1, self.max_number + 1)}

    def _sig_seq_pattern(self, d):
        n, l = len(d), set(d[-1][:6])
        sc = Counter()
        if n < 30:
            return {num: 0 for num in range(1, self.max_number + 1)}
        for cl in [3, 4]:
            cc = [frozenset(x[:6]) for x in d[-cl:]]
            for i in range(cl, min(n - cl, n - 1)):
                hc = [frozenset(x[:6]) for x in d[i - cl:i]]
                sim = sum(len(cc[j] & hc[j]) for j in range(cl))
                if sim >= cl * 2 and i < n - 1:
                    for num in d[i][:6]:
                        if num not in l:
                            sc[num] += sim * cl * 0.1
        mx = max(sc.values()) if sc else 1
        return {num: sc.get(num, 0) / max(1, mx) * 3
                for num in range(1, self.max_number + 1)}

    def _sig_runlength(self, d):
        n = len(d)
        eg = self.max_number / self.pick_count
        s = {}
        for num in range(1, self.max_number + 1):
            ca = 0
            for x in reversed(d):
                if num not in x[:6]:
                    ca += 1
                else:
                    break
            if ca > 0:
                sa = [1 if num in x[:6] else 0 for x in d]
                ar, run = [], 0
                for sv in sa:
                    if sv == 0:
                        run += 1
                    else:
                        if run > 0:
                            ar.append(run)
                        run = 0
                avg = np.mean(ar) if ar else eg
                s[num] = (1 / (1 + math.exp(-3 * (ca / avg - 0.8))) * 2
                          if avg > 0 else 0)
            else:
                s[num] = 0
        return s

    def _sig_day_profile(self, d, dates):
        from datetime import datetime
        dd = defaultdict(list)
        for dt, x in zip(dates, d):
            try:
                dd[datetime.strptime(dt, '%Y-%m-%d').weekday()].append(x)
            except Exception:
                continue
        try:
            ndow = (datetime.strptime(dates[-1], '%Y-%m-%d').weekday() + 2) % 7
        except Exception:
            return {num: 0 for num in range(1, self.max_number + 1)}
        return {num: ((sum(1 for x in dd.get(ndow, []) if num in x[:6]) /
                        max(len(dd.get(ndow, [])), 1)) /
                       (sum(1 for x in d if num in x[:6]) /
                        len(d) + 1e-10) * 2 - 2)
                for num in range(1, self.max_number + 1)}

    def _sig_pair_boost(self, d):
        l = set(d[-1][:6])
        pf = Counter()
        for x in d[-100:]:
            for p in combinations(sorted(x[:self.pick_count]), 2):
                pf[p] += 1
        return {num: sum(pf.get(tuple(sorted([p, num])), 0)
                         for p in l
                         if pf.get(tuple(sorted([p, num])), 0) > 3) * 0.05
                for num in range(1, self.max_number + 1)}

    def _sig_consecutive(self, d):
        s = {num: 0.0 for num in range(1, self.max_number + 1)}
        for x in d[-50:]:
            sd = sorted(x[:self.pick_count])
            for i in range(len(sd) - 1):
                if sd[i + 1] - sd[i] == 1:
                    s[sd[i]] += 0.05
                    s[sd[i + 1]] += 0.05
        return s

    def _sig_oddeven(self, d):
        lo = sum(1 for x in d[-1][:6] if x % 2 == 1)
        return {num: 0.3 if (lo > 3 and num % 2 == 0) or
                            (lo <= 3 and num % 2 == 1) else 0
                for num in range(1, self.max_number + 1)}

    def _sig_highlow(self, d):
        mid = self.max_number // 2
        lh = sum(1 for x in d[-1][:6] if x > mid)
        return {num: 0.3 if (lh > 3 and num <= mid) or
                            (lh <= 3 and num > mid) else 0
                for num in range(1, self.max_number + 1)}

    # ================================================================
    # VULNERABILITY SIGNALS (from Ultimate Engine)
    # ================================================================
    def _vulnerability_signals(self, data):
        n = len(data)
        result = {}
        w = min(100, n)
        last_set = set(data[-1][:6])

        # Serial
        serial_scores = {}
        in_after_in = Counter()
        in_count = Counter()
        in_after_out = Counter()
        out_count = Counter()
        for i in range(max(0, n - w), n - 1):
            curr = set(data[i][:6])
            nxt = set(data[i + 1][:6])
            for num in range(1, self.max_number + 1):
                if num in curr:
                    in_count[num] += 1
                    if num in nxt:
                        in_after_in[num] += 1
                else:
                    out_count[num] += 1
                    if num in nxt:
                        in_after_out[num] += 1
        for num in range(1, self.max_number + 1):
            p_in = in_after_in[num] / max(in_count[num], 1)
            p_out = in_after_out[num] / max(out_count[num], 1)
            serial_scores[num] = ((p_in - p_out) * 5 if num in last_set
                                  else (p_out - p_in) * 3)
        result['serial'] = serial_scores

        # Gap
        gap_scores = {}
        last_seen = {}
        gap_sums = defaultdict(float)
        gap_counts = defaultdict(int)
        for i, d in enumerate(data):
            for num in d[:6]:
                if num in last_seen:
                    gap_sums[num] += (i - last_seen[num])
                    gap_counts[num] += 1
                last_seen[num] = i
        for num in range(1, self.max_number + 1):
            gc = gap_counts.get(num, 0)
            if gc < 5:
                gap_scores[num] = 0
                continue
            mg = gap_sums[num] / gc
            cg = n - last_seen.get(num, 0)
            ratio = cg / mg if mg > 0 else 1
            gap_scores[num] = max(0, (ratio - 0.8)) * 2 if ratio > 1.0 else 0
        result['gap'] = gap_scores

        # Pair
        pair_scores = {}
        pair_freq = Counter()
        for d in data[-100:]:
            for p in combinations(sorted(d[:self.pick_count]), 2):
                pair_freq[p] += 1
        n_r = min(100, n)
        exp_pair = n_r * (self.pick_count * (self.pick_count - 1)) / (
            self.max_number * (self.max_number - 1))
        for num in range(1, self.max_number + 1):
            boost = 0
            for prev in last_set:
                pair = tuple(sorted([prev, num]))
                freq = pair_freq.get(pair, 0)
                if freq > exp_pair + 2:
                    boost += (freq - exp_pair) * 0.3
            pair_scores[num] = boost
        result['pair'] = pair_scores
        return result

    # ================================================================
    # WALK-FORWARD WEIGHT CALIBRATION
    # ================================================================
    def _walk_forward_calibrate(self, data, signals):
        """Proper walk-forward: re-compute signals on training data only."""
        n = len(data)
        test_count = min(50, n - 70)
        if test_count < 10:
            return {name: 1.0 for name in signals}

        signal_hits = {name: 0 for name in signals}
        total = 0

        for idx in range(n - test_count - 1, n - 1):
            actual = set(data[idx + 1][:6])
            total += 1
            for sig_name, sig_scores in signals.items():
                if not sig_scores:
                    continue
                top = sorted(sig_scores.items(),
                             key=lambda x: -x[1])[:self.pick_count]
                predicted = set(num for num, _ in top)
                signal_hits[sig_name] += len(predicted & actual)

        base = self.pick_count / self.max_number
        weights = {}
        for name in signals:
            if total > 0 and signal_hits[name] > 0:
                avg = signal_hits[name] / total
                lift = avg / (base * self.pick_count)
                weights[name] = max(lift, 0.1)
            else:
                weights[name] = 0.5
        return weights

    # ================================================================
    # ADAPTIVE ANTI-REPEAT
    # ================================================================
    def _learn_repeat_rate(self, data):
        """Learn the optimal anti-repeat penalty from historical data."""
        n = len(data)
        if n < 20:
            return -2.0  # Default

        # Count how often numbers from draw N appear in draw N+1
        repeat_counts = []
        for i in range(1, n):
            prev = set(data[i - 1][:6])
            curr = set(data[i][:6])
            repeat_counts.append(len(prev & curr))

        avg_repeat = np.mean(repeat_counts)
        # If avg_repeat ~ 0.8, then penalty should be slight
        # If avg_repeat ~ 0, then strong penalty
        # Scale: avg 0.8 repeats from 6 → slight negative bias
        penalty = (avg_repeat - 1.0) * 1.5  # Typically -0.3 to -1.5
        return penalty

    # ================================================================
    # CONSTRAINTS
    # ================================================================
    def _learn_constraints(self, data):
        """Learn statistical constraints from recent data."""
        r = data[-50:]
        sums = [sum(d[:self.pick_count]) for d in r]
        odds = [sum(1 for x in d[:self.pick_count] if x % 2 == 1) for d in r]
        mid = self.max_number // 2
        highs = [sum(1 for x in d[:self.pick_count] if x > mid) for d in r]
        ranges = [max(d[:self.pick_count]) - min(d[:self.pick_count]) for d in r]
        return {
            'sum_lo': int(np.percentile(sums, 3)),
            'sum_hi': int(np.percentile(sums, 97)),
            'sum_mean': round(float(np.mean(sums)), 1),
            'odd_lo': max(0, int(np.percentile(odds, 5))),
            'odd_hi': min(self.pick_count, int(np.percentile(odds, 95))),
            'high_lo': max(0, int(np.percentile(highs, 5))),
            'high_hi': min(self.pick_count, int(np.percentile(highs, 95))),
            'range_lo': int(np.percentile(ranges, 5)),
            'range_hi': int(np.percentile(ranges, 95)),
        }

    # ================================================================
    # SUB-ENGINE RESULTS
    # ================================================================
    def _run_sub_engines(self, data, dates):
        """Run existing engines and collect their top picks."""
        results = {}

        try:
            from models.deep_forensic import DeepForensic
            df = DeepForensic(self.max_number, self.pick_count)
            df_result = df.analyze(data, dates)
            results['deep_forensic'] = {
                'primary': df_result.get('primary', []),
                'top_30': df_result.get('top_30', [])[:10],
            }
        except Exception:
            results['deep_forensic'] = {'primary': [], 'top_30': []}

        try:
            from models.ultimate_engine import UltimateEngine
            ue = UltimateEngine(self.max_number, self.pick_count)
            ue_result = ue.predict(data, dates, n_portfolio=10)
            results['ultimate'] = {
                'primary': ue_result.get('primary', []),
                'total_sets': ue_result.get('total_sets', 0),
            }
        except Exception:
            results['ultimate'] = {'primary': [], 'total_sets': 0}

        return results

    # ================================================================
    # CONFIDENCE CALCULATION
    # ================================================================
    def _calc_confidence(self, consensus, golden_set, signals, weights):
        """Calculate confidence 0-100 based on signal agreement."""
        # Count how many signals agree on the golden set numbers
        agreement_scores = []
        for sig_name, sig_scores in signals.items():
            if not sig_scores:
                continue
            top = set(num for num, _ in
                      sorted(sig_scores.items(), key=lambda x: -x[1])[:15])
            overlap = len(set(golden_set) & top)
            agreement_scores.append(overlap / self.pick_count)

        if not agreement_scores:
            return 30.0

        avg_agreement = np.mean(agreement_scores)
        # Scale: 0.3 agreement = 30%, 0.6 = 60%, etc.
        confidence = min(avg_agreement * 100, 95)  # Cap at 95
        return max(confidence, 15)  # Floor at 15

    # ================================================================
    # QUICK BACKTEST
    # ================================================================
    def _quick_backtest(self, data, signals, weights, constraints):
        """Quick walk-forward backtest of the consensus approach."""
        n = len(data)
        test_count = min(30, n - 70)
        if test_count < 5:
            return {'avg': 0, 'max': 0, 'tests': 0}

        matches = []
        for idx in range(n - test_count - 1, n - 1):
            actual = set(data[idx + 1][:6])

            # Build consensus for this point in time
            consensus = {num: 0.0 for num in range(1, self.max_number + 1)}
            for sig_name, sig_scores in signals.items():
                if not sig_scores:
                    continue
                w = weights.get(sig_name, 1.0)
                vals = list(sig_scores.values())
                max_v = max(abs(v) for v in vals) if vals else 1
                if max_v < 0.001:
                    max_v = 1
                for num, score in sig_scores.items():
                    consensus[num] += (score / max_v) * w

            ranked = sorted(consensus.items(), key=lambda x: -x[1])
            predicted = set(num for num, _ in ranked[:self.pick_count])
            matches.append(len(predicted & actual))

        dist = Counter(matches)
        random_avg = self.pick_count * self.pick_count / self.max_number

        return {
            'avg': round(float(np.mean(matches)), 3),
            'max': max(matches) if matches else 0,
            'tests': len(matches),
            'distribution': {str(k): v for k, v in sorted(dist.items())},
            'random_avg': round(random_avg, 3),
            'improvement': round(
                (float(np.mean(matches)) / random_avg - 1) * 100, 1
            ) if random_avg > 0 else 0,
        }
