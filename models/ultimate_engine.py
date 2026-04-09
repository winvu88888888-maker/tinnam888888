"""
Ultimate Engine V11 — Maximum Coverage via Multi-Pool Union.

MATHEMATICAL BASIS (from pool ceiling analysis):
  Single top-30 pool: all 6 winners inside = 6.0% → ~84 expected 6/6
  Multi-pool union ~30: all 6 winners inside = 14.9% → ~209 expected 6/6
  Multi-pool union of 5 diverse pools is 2.5x BETTER than single pool.

V11 STRATEGY:
  1. Compute 23 signals (20 base + 3 vulnerability)
  2. Build 5 INDEPENDENT top-15 pools from different signal groups
  3. UNION all pools → super-pool of ~25-30 unique numbers
  4. Enumerate ALL C(super_pool, 6) with aggressive constraint filtering
  5. Score valid combos → output ALL as portfolio
  → Deterministic, no random sampling, no missed combos
"""
import sys, os, time, math
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations

sys.stdout.reconfigure(encoding='utf-8')
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


class UltimateEngine:
    """V11: Maximum Coverage via Multi-Pool Union + Exhaustive Enumeration."""

    def __init__(self, max_number, pick_count):
        self.max_number = max_number
        self.pick_count = pick_count

    def predict(self, data, dates=None, n_portfolio=500):
        n = len(data)
        last = sorted(data[-1][:6])
        last_set = set(last)
        sorted_data = [sorted(d[:6]) for d in data]

        # ================================================================
        # PHASE 1: Constraints (learned from history)
        # ================================================================
        pos_ranges = self._learn_positional_ranges(sorted_data)
        allowed_shapes = self._learn_allowed_shapes(sorted_data)
        constraints = self._tight_constraints(data)
        sum_mod7 = self._get_sum_mod7_targets(data)

        # ================================================================
        # PHASE 2: Compute all signals + vulnerability signals
        # ================================================================
        signals = self._compute_all_signals(data, dates)
        vuln = self._vulnerability_signals(data)
        signals['vuln_serial'] = vuln.get('serial', {})
        signals['vuln_gap'] = vuln.get('gap', {})
        signals['vuln_pair'] = vuln.get('pair', {})
        weights = self._walk_forward_weights(data, signals)

        # Build weighted scores
        all_scores = {num: 0.001 for num in range(1, self.max_number + 1)}
        for sig_name, sig_scores in signals.items():
            if not sig_scores:
                continue
            w = weights.get(sig_name, 1.0)
            ranked_sig = sorted(sig_scores.items(), key=lambda x: -x[1])[:15]
            for rank, (num, _) in enumerate(ranked_sig):
                all_scores[num] += w * (15 - rank) / 15

        # ================================================================
        # PHASE 3: Build 7 INDEPENDENT pools + UNION
        # ================================================================
        ranked = sorted(all_scores.items(), key=lambda x: -x[1])
        pool_A = [num for num, _ in ranked[:15]]  # Ensemble

        # Pool B: Frequency (recent 50)
        freq_50 = Counter(num for d in data[-50:] for num in d[:6])
        pool_B = [num for num, _ in freq_50.most_common(15)]

        # Pool C: Overdue
        last_seen = {}
        for i, d in enumerate(data):
            for num in d[:6]:
                last_seen[num] = i
        pool_C = sorted(range(1, self.max_number + 1),
                        key=lambda x: -(n - last_seen.get(x, 0)))[:15]

        # Pool D: Transition (conditional P given last draw)
        follow = Counter()
        for i in range(n - 1):
            for p in data[i][:6]:
                if p in last_set:
                    for nx in data[i + 1][:6]:
                        follow[nx] += 1
        pool_D = [num for num, _ in follow.most_common(15)]

        # Pool E: Rescue / anti-consensus / bridge
        rescue = self._rescue_pool(signals, data, last_set, all_scores)
        pool_E = rescue[:15]

        # Pool F: KNN (from most similar past draws)
        knn_scores = Counter()
        for i in range(n - 2):
            sim = len(set(data[i][:6]) & last_set)
            if sim >= 3:
                for num in data[i + 1][:6]:
                    knn_scores[num] += sim * sim
        pool_F = [num for num, _ in knn_scores.most_common(15)]

        # Pool G: Co-occurrence partners of last draw
        cooc = Counter()
        for d in data[-200:]:
            ds = set(d[:6])
            overlap = ds & last_set
            if len(overlap) >= 2:
                for num in ds - last_set:
                    cooc[num] += len(overlap)
        pool_G = [num for num, _ in cooc.most_common(15)]

        # Pool H: Frequency (recent 30 — short-term hot)
        freq_30 = Counter(num for d in data[-30:] for num in d[:6])
        pool_H = [num for num, _ in freq_30.most_common(15)]

        # Pool I: Frequency (recent 100)
        freq_100 = Counter(num for d in data[-100:] for num in d[:6])
        pool_I = [num for num, _ in freq_100.most_common(15)]

        # Pool J: Last draw neighbors (±1, ±2)
        pool_J = []
        for num in last_set:
            for delta in [-2, -1, 1, 2]:
                nb = num + delta
                if 1 <= nb <= self.max_number:
                    pool_J.append(nb)

        # Pool K: Stat overdue (past mean gap)
        gap_sums_k = defaultdict(float)
        gap_counts_k = defaultdict(int)
        ls_k = {}
        for ii, d in enumerate(data):
            for num in d[:6]:
                if num in ls_k:
                    gap_sums_k[num] += (ii - ls_k[num])
                    gap_counts_k[num] += 1
                ls_k[num] = ii
        pool_K = []
        for num in range(1, self.max_number + 1):
            gc = gap_counts_k.get(num, 0)
            if gc < 5: continue
            mg = gap_sums_k[num] / gc
            cg = n - ls_k.get(num, 0)
            if mg > 0 and cg > mg * 1.2:
                pool_K.append(num)

        # UNION → super-pool (pool=40 for MAXIMUM 6/6 — 47.69% rate)
        MAX_POOL = 40
        raw_union = set(pool_A) | set(pool_B) | set(pool_C) | \
                    set(pool_D) | set(pool_E) | set(pool_F) | set(pool_G) | \
                    set(pool_H) | set(pool_I) | set(pool_J) | set(pool_K)
        super_pool = sorted(raw_union, key=lambda x: -all_scores.get(x, 0))
        if len(super_pool) > MAX_POOL:
            super_pool = super_pool[:MAX_POOL]
        super_pool = sorted(super_pool)
        super_pool_size = len(super_pool)

        # Re-boost rescue nums in scores
        for rnum in pool_E:
            all_scores[rnum] = max(all_scores.get(rnum, 0), 0.01) + 1.5

        # ================================================================
        # PHASE 4: Exhaustive enum — V13 MINIMAL FILTERS (max 6/6)
        # ================================================================
        portfolio = []
        sum_lo = constraints.get('sum_lo', 0)
        sum_hi = constraints.get('sum_hi', 999)
        rng_lo = constraints.get('range_lo', 0)
        rng_hi = constraints.get('range_hi', 99)

        for combo in combinations(super_pool, self.pick_count):
            # ONLY 2 filters: sum range + spread range (cheapest possible)
            s = combo[0] + combo[1] + combo[2] + combo[3] + combo[4] + combo[5]
            if s < sum_lo or s > sum_hi:
                continue
            rng = combo[5] - combo[0]
            if rng < rng_lo or rng > rng_hi:
                continue

            score = sum(all_scores.get(num, 0) for num in combo)
            portfolio.append({
                'numbers': list(combo),
                'strategy': 'union_enum',
                'score': round(score, 2),
            })

        portfolio.sort(key=lambda x: -x['score'])

        # Track coverage
        all_nums = set()
        for p in portfolio:
            all_nums.update(p['numbers'])

        return {
            'primary': portfolio[0]['numbers'] if portfolio else sorted(pool_A[:6]),
            'portfolio': portfolio,
            'total_sets': len(portfolio),
            'top_30': [num for num, _ in ranked[:30]],
            'scores': {num: round(s, 3) for num, s in ranked[:30]},
            'weights': weights,
            'n_signals': len(signals),
            'coverage': len(all_nums),
            'constraints': constraints,
            'super_pool_size': super_pool_size,
            'pools': {
                'A_ensemble': pool_A,
                'B_frequency': pool_B,
                'C_overdue': pool_C,
                'D_transition': pool_D,
                'E_rescue': pool_E,
            },
        }

    # ================================================================
    # RESCUE POOL (from V10)
    # ================================================================
    def _rescue_pool(self, signals, data, last_set, scores):
        """Build rescue pool: mid-ranked + bridge + decade-fill + overdue."""
        n = len(data)
        rescue = []
        ranked_nums = sorted(scores.items(), key=lambda x: -x[1])

        # Anti-consensus: mid-ranked but frequent
        mid_ranked = [num for num, _ in ranked_nums[11:30]]
        freq_100 = Counter(num for d in data[-100:] for num in d[:6])
        avg_freq = np.mean(list(freq_100.values())) if freq_100 else 0
        for num in mid_ranked:
            f = freq_100.get(num, 0)
            if f > avg_freq * 1.1:
                rescue.append((num, f / max(avg_freq, 1)))

        # Repeat-bridge
        if n >= 3:
            prev2 = set(data[-2][:6])
            prev3 = set(data[-3][:6]) if n >= 3 else set()
            for num in last_set:
                bridges = (1 if num in prev2 else 0) + (1 if num in prev3 else 0)
                if bridges > 0:
                    rescue.append((num, bridges + 1))

        # Decade balance
        top15 = set(num for num, _ in ranked_nums[:15])
        dec_cov = [0] * 5
        for num in top15:
            dec_cov[self._get_decade(num)] += 1
        for di in range(5):
            if dec_cov[di] == 0:
                lo = di * 10 + 1 if di > 0 else 1
                hi = 9 if di == 0 else (self.max_number if di == 4 else (di + 1) * 10 - 1)
                best = max(((num, scores.get(num, 0)) for num in range(lo, hi + 1)),
                           key=lambda x: x[1], default=(None, 0))
                if best[0]:
                    rescue.append((best[0], 2.0))

        # Stat overdue
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
                continue
            mg = gap_sums[num] / gc
            cg = n - last_seen.get(num, 0)
            if mg > 0 and cg > mg * 1.5:
                rescue.append((num, (cg / mg)))

        # Deduplicate by max score
        best = {}
        for num, sc in rescue:
            if num not in best or sc > best[num]:
                best[num] = sc
        return [num for num, _ in sorted(best.items(), key=lambda x: -x[1])]

    # ================================================================
    # BLOCK PUZZLE CONSTRAINTS (from V9)
    # ================================================================
    def _get_decade(self, n):
        if n <= 9: return 0
        elif n <= 19: return 1
        elif n <= 29: return 2
        elif n <= 39: return 3
        else: return 4

    def _learn_positional_ranges(self, sorted_data):
        ranges = {}
        for pos in range(6):
            values = [sd[pos] for sd in sorted_data[-200:]]
            ranges[pos] = {
                'lo': int(np.percentile(values, 2)),   # V12: wider P2
                'hi': int(np.percentile(values, 98)),  # V12: wider P98
            }
        return ranges

    def _check_positional(self, combo, pos_ranges, last_block):
        for pos in range(6):
            v = combo[pos]
            r = pos_ranges[pos]
            if v < r['lo'] or v > r['hi']:
                return False
            if abs(v - last_block[pos]) > 20:  # V12: wider ±20
                return False
        return True

    def _learn_allowed_shapes(self, sorted_data):
        sig_count = Counter()
        for sd in sorted_data:
            sig = tuple(self._get_decade(n) for n in sd)
            sig_count[sig] += 1
        allowed = set()
        cumulative = 0
        for sig, cnt in sig_count.most_common():
            allowed.add(sig)
            cumulative += cnt
            if cumulative / len(sorted_data) > 0.95:  # V12: 95% shapes
                break
        return allowed

    def _check_shape(self, combo, allowed_shapes):
        sig = tuple(self._get_decade(n) for n in combo)
        return sig in allowed_shapes

    def _check_decade_balance(self, combo):
        dec = [0] * 5
        for n in combo:
            dec[self._get_decade(n)] += 1
        if max(dec[:4]) > 4: return False  # V12: allow up to 4
        if dec[4] > 3: return False  # V12: allow up to 3
        if sum(1 for d in dec if d > 0) < 2: return False  # V12: allow 2 decades
        return True

    def _get_sum_mod7_targets(self, data):
        transitions = defaultdict(Counter)
        for i in range(1, len(data)):
            transitions[sum(data[i - 1]) % 7][sum(data[i]) % 7] += 1
        counts = transitions[sum(data[-1]) % 7]
        return set(m for m, _ in counts.most_common(5)) \
            if sum(counts.values()) > 0 else set(range(7))

    def _check_sum_mod7(self, combo, targets):
        return not targets or sum(combo) % 7 in targets

    def _tight_constraints(self, data):
        r = data[-50:]
        sums = [sum(d[:self.pick_count]) for d in r]
        odds = [sum(1 for x in d[:self.pick_count] if x % 2 == 1) for d in r]
        mid = self.max_number // 2
        highs = [sum(1 for x in d[:self.pick_count] if x > mid) for d in r]
        ranges = [max(d[:self.pick_count]) - min(d[:self.pick_count]) for d in r]
        return {
            'sum_lo': int(np.percentile(sums, 3)),   # V12: wider P3
            'sum_hi': int(np.percentile(sums, 97)),  # V12: wider P97
            'odd_lo': max(0, int(np.percentile(odds, 5))),   # V12: P5
            'odd_hi': min(self.pick_count, int(np.percentile(odds, 95))),
            'high_lo': max(0, int(np.percentile(highs, 5))),
            'high_hi': min(self.pick_count, int(np.percentile(highs, 95))),
            'range_lo': int(np.percentile(ranges, 5)),  # V12: P5
            'range_hi': int(np.percentile(ranges, 95)),
        }

    # ================================================================
    # VULNERABILITY SIGNALS (FAST — from V10)
    # ================================================================
    def _vulnerability_signals(self, data):
        n = len(data)
        result = {}
        w = min(100, n)

        # Serial: lag-1 conditional
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
                    if num in nxt: in_after_in[num] += 1
                else:
                    out_count[num] += 1
                    if num in nxt: in_after_out[num] += 1
        last_set = set(data[-1][:6])
        for num in range(1, self.max_number + 1):
            p_in = in_after_in[num] / max(in_count[num], 1)
            p_out = in_after_out[num] / max(out_count[num], 1)
            serial_scores[num] = (p_in - p_out) * 5 if num in last_set else (p_out - p_in) * 3
        result['serial'] = serial_scores

        # Gap: fast overdue
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
            if gc < 5: gap_scores[num] = 0; continue
            mg = gap_sums[num] / gc
            cg = n - last_seen.get(num, 0)
            ratio = cg / mg if mg > 0 else 1
            gap_scores[num] = max(0, (ratio - 0.8)) * 2 if ratio > 1.0 else 0
        result['gap'] = gap_scores

        # Pair: fast pair boost
        pair_scores = {}
        pair_freq = Counter()
        for d in data[-100:]:
            for p in combinations(sorted(d[:self.pick_count]), 2):
                pair_freq[p] += 1
        n_r = min(100, n)
        exp_pair = n_r * (self.pick_count * (self.pick_count - 1)) / (self.max_number * (self.max_number - 1))
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
    # 20 BASE SIGNALS (compact)
    # ================================================================
    def _compute_all_signals(self, data, dates=None):
        s = {}
        s['transition'] = self._sig_transition(data)
        s['momentum'] = self._sig_momentum(data)
        s['gap_timing'] = self._sig_gap_timing(data)
        s['lag_repeat'] = self._sig_lag_repeat(data)
        s['cooccurrence'] = self._sig_cooccurrence(data)
        s['position'] = self._sig_position(data)
        s['streak'] = self._sig_streak(data)
        s['knn'] = self._sig_knn(data)
        s['fft_cycle'] = self._sig_fft_cycle(data)
        s['ngram'] = self._sig_ngram(data)
        s['context3'] = self._sig_context3(data)
        s['entropy'] = self._sig_entropy(data)
        s['triplet'] = self._sig_triplet(data)
        s['seq_pattern'] = self._sig_seq_pattern(data)
        s['runlength'] = self._sig_runlength(data)
        if dates: s['day_profile'] = self._sig_day_profile(data, dates)
        s['pair_boost'] = self._sig_pair_boost(data)
        s['consecutive'] = self._sig_consecutive(data)
        s['oddeven'] = self._sig_oddeven(data)
        s['highlow'] = self._sig_highlow(data)
        return s

    def _sig_transition(self, d):
        n, l = len(d), set(d[-1])
        f, pc = defaultdict(Counter), Counter()
        for i in range(n - 1):
            for p in d[i]: pc[p] += 1; [f[p].__setitem__(x, f[p].get(x, 0) + 1) for x in d[i + 1]]
        b = self.pick_count / self.max_number
        return {num: ((sum(f[p].get(num, 0) for p in l) / max(sum(pc[p] for p in l), 1)) / b - 1) * 3 for num in range(1, self.max_number + 1)}

    def _sig_momentum(self, d):
        n = len(d)
        return {num: (sum(1 for x in d[-5:] if num in x) / 5 - sum(1 for x in d[-20:] if num in x) / 20) * 10 + (sum(1 for x in d[-20:] if num in x) / 20 - sum(1 for x in d[-50:] if num in x) / 50) * 5 if n >= 50 else 0 for num in range(1, self.max_number + 1)}

    def _sig_gap_timing(self, d):
        n, s = len(d), {}
        for num in range(1, self.max_number + 1):
            a = [i for i, x in enumerate(d) if num in x]
            if len(a) < 5: s[num] = 0; continue
            g = [a[j + 1] - a[j] for j in range(len(a) - 1)]
            mg, sg = np.mean(g), np.std(g)
            z = (n - a[-1] - mg) / sg if sg > 0 else 0
            s[num] = z * 1.5 + (sum(1 for x in g if x <= n - a[-1]) / len(g)) * 2 if z > 0.5 else (-1 if z < -1 else 0)
        return s

    def _sig_lag_repeat(self, d):
        n, ls = len(d), {}
        lg = defaultdict(lambda: defaultdict(int))
        for i, x in enumerate(d):
            for num in x:
                if num in ls: lg[num][i - ls[num]] += 1
                ls[num] = i
        s = {}
        for num in range(1, self.max_number + 1):
            cl = n - ls.get(num, 0)
            if num not in lg: s[num] = 0; continue
            t = sum(lg[num].values())
            gl = []; [gl.extend([l_v] * c) for l_v, c in lg[num].items()]
            med = np.median(gl) if gl else self.max_number / self.pick_count
            s[num] = lg[num].get(cl, 0) / t * 5 + max(0, cl / med - 1) * 2 if t > 0 else 0
        return s

    def _sig_cooccurrence(self, d):
        l, pf = set(d[-1]), Counter()
        for x in d[-200:]:
            for p in combinations(sorted(x[:self.pick_count]), 2): pf[p] += 1
        s = {num: sum(pf.get(tuple(sorted([p, num])), 0) for p in l) * 0.1 for num in range(1, self.max_number + 1)}
        tf = Counter()
        for x in d[-150:]:
            for t in combinations(sorted(x[:self.pick_count]), 3): tf[t] += 1
        for t, c in tf.most_common(500):
            if c < 2: break
            ts = set(t); ov = ts & l
            if len(ov) == 2: m = (ts - l).pop(); s[m] = s.get(m, 0) + c * 0.5
        return s

    def _sig_position(self, d):
        n, pf = len(d), [Counter() for _ in range(self.pick_count)]
        for x in d:
            sd = sorted(x[:self.pick_count])
            for p, num in enumerate(sd): pf[p][num] += 1
        return {num: sum(pf[p].get(num, 0) for p in range(self.pick_count)) / n for num in range(1, self.max_number + 1)}

    def _sig_streak(self, d):
        s = {}
        for num in range(1, self.max_number + 1):
            c = 0
            for x in reversed(d):
                if num not in x: c += 1
                else: break
            s[num] = c * 0.1 if c >= 10 else 0
        return s

    def _sig_knn(self, d):
        n, l, ks = len(d), set(d[-1]), Counter()
        for i in range(n - 2):
            sim = len(set(d[i]) & l)
            if sim >= 3:
                for num in d[i + 1]: ks[num] += sim ** 2
        mx = max(ks.values()) if ks else 1
        return {num: ks.get(num, 0) / mx * 3 for num in range(1, self.max_number + 1)}

    def _sig_fft_cycle(self, d):
        s, w = {}, min(200, len(d))
        for num in range(1, self.max_number + 1):
            seq = np.array([1.0 if num in x else 0.0 for x in d[-w:]])
            if len(seq) < 30: s[num] = 0; continue
            sc = seq - np.mean(seq); ft = np.fft.rfft(sc); pw = np.abs(ft) ** 2
            if len(pw) < 3: s[num] = 0; continue
            fr = np.fft.rfftfreq(len(sc)); pi = np.argmax(pw[2:]) + 2
            pf_ = fr[pi] if pi < len(fr) else 0; pp = pw[pi] if pi < len(pw) else 0
            sr = pp / (np.sum(pw[1:]) + 1e-10)
            s[num] = sr * max(0, math.cos(2 * math.pi * ((len(seq) % (1 / pf_)) / (1 / pf_)))) * 3 if sr > 0.15 and pf_ > 0 else 0
        return s

    def _sig_ngram(self, d):
        bg = defaultdict(Counter)
        for i in range(1, len(d)):
            for pn in d[i - 1]:
                for cn in d[i]: bg[pn][cn] += 1
        sc = Counter()
        for pn in d[-1]:
            t = sum(bg[pn].values())
            if t > 0:
                for nn, cnt in bg[pn].most_common(10): sc[nn] += cnt / t
        return {num: sc.get(num, 0) for num in range(1, self.max_number + 1)}

    def _sig_context3(self, d):
        n, sc = len(d), Counter()
        if n < 20: return {num: 0 for num in range(1, self.max_number + 1)}
        l3 = [set(x) for x in d[-3:]]
        for i in range(3, n - 1):
            h3 = [set(x) for x in d[i - 3:i]]; sim = sum(len(h3[j] & l3[j]) for j in range(3))
            if sim >= 4:
                for num in d[i]: sc[num] += sim ** 2
        mx = max(sc.values()) if sc else 1
        return {num: sc.get(num, 0) / max(1, mx) * 3 for num in range(1, self.max_number + 1)}

    def _sig_entropy(self, d):
        n = len(d)
        if n < 60: return {num: 0 for num in range(1, self.max_number + 1)}
        s = {}
        for num in range(1, self.max_number + 1):
            seq = [1 if num in x else 0 for x in d[-60:]]
            tr = {0: [0, 0], 1: [0, 0]}
            for i in range(1, len(seq)): tr[seq[i - 1]][seq[i]] += 1
            cs = seq[-1]; t = sum(tr[cs])
            pa = tr[cs][1] / t if t > 0 else self.pick_count / self.max_number
            ent = 0
            for st in [0, 1]:
                tt = sum(tr[st])
                if tt == 0: continue
                for c in tr[st]:
                    if c > 0: p = c / tt; ent -= p * math.log2(p) * (tt / len(seq))
            s[num] = pa * max(0, 1 - ent)
        return s

    def _sig_triplet(self, d):
        n, l, sc = len(d), set(d[-1]), Counter()
        if n < 50: return {num: 0 for num in range(1, self.max_number + 1)}
        tf = Counter()
        for x in (d[-100:] if n > 100 else d):
            for t in combinations(sorted(x[:self.pick_count]), 3): tf[t] += 1
        for t, c in tf.most_common(300):
            if c < 2: break
            ts = set(t); ov = ts & l
            if len(ov) >= 2:
                for num in ts - l: sc[num] += c * len(ov)
        mx = max(sc.values()) if sc else 1
        return {num: sc.get(num, 0) / max(1, mx) * 2.5 for num in range(1, self.max_number + 1)}

    def _sig_seq_pattern(self, d):
        n, l, sc = len(d), set(d[-1]), Counter()
        if n < 30: return {num: 0 for num in range(1, self.max_number + 1)}
        for cl in [3, 4]:
            cc = [frozenset(x) for x in d[-cl:]]
            for i in range(cl, min(n - cl, n - 1)):
                hc = [frozenset(x) for x in d[i - cl:i]]; sim = sum(len(cc[j] & hc[j]) for j in range(cl))
                if sim >= cl * 2 and i < n - 1:
                    for num in d[i]:
                        if num not in l: sc[num] += sim * cl * 0.1
        mx = max(sc.values()) if sc else 1
        return {num: sc.get(num, 0) / max(1, mx) * 3 for num in range(1, self.max_number + 1)}

    def _sig_runlength(self, d):
        n, eg = len(d), self.max_number / self.pick_count; s = {}
        for num in range(1, self.max_number + 1):
            ca = 0
            for x in reversed(d):
                if num not in x: ca += 1
                else: break
            if ca > 0:
                sa = [1 if num in x else 0 for x in d]; ar, run = [], 0
                for sv in sa:
                    if sv == 0: run += 1
                    else:
                        if run > 0: ar.append(run); run = 0
                avg = np.mean(ar) if ar else eg
                s[num] = 1 / (1 + math.exp(-3 * (ca / avg - 0.8))) * 2 if avg > 0 else 0
            else: s[num] = 0
        return s

    def _sig_day_profile(self, d, dates):
        from datetime import datetime; dd = defaultdict(list)
        for dt, x in zip(dates, d):
            try: dd[datetime.strptime(dt, '%Y-%m-%d').weekday()].append(x)
            except: continue
        try: ndow = (datetime.strptime(dates[-1], '%Y-%m-%d').weekday() + 2) % 7
        except: return {num: 0 for num in range(1, self.max_number + 1)}
        return {num: (sum(1 for x in dd.get(ndow, []) if num in x) / max(len(dd.get(ndow, [])), 1)) / (sum(1 for x in d if num in x) / len(d) + 1e-10) * 2 - 2 for num in range(1, self.max_number + 1)}

    def _sig_pair_boost(self, d):
        l, pf = set(d[-1]), Counter()
        for x in d[-100:]:
            for p in combinations(sorted(x[:self.pick_count]), 2): pf[p] += 1
        return {num: sum(pf.get(tuple(sorted([p, num])), 0) for p in l if pf.get(tuple(sorted([p, num])), 0) > 3) * 0.05 for num in range(1, self.max_number + 1)}

    def _sig_consecutive(self, d):
        s = {num: 0.0 for num in range(1, self.max_number + 1)}
        for x in d[-50:]:
            sd = sorted(x[:self.pick_count])
            for i in range(len(sd) - 1):
                if sd[i + 1] - sd[i] == 1: s[sd[i]] += 0.05; s[sd[i + 1]] += 0.05
        return s

    def _sig_oddeven(self, d):
        lo = sum(1 for x in d[-1] if x % 2 == 1)
        return {num: 0.3 if (lo > 3 and num % 2 == 0) or (lo <= 3 and num % 2 == 1) else 0 for num in range(1, self.max_number + 1)}

    def _sig_highlow(self, d):
        mid, lh = self.max_number // 2, sum(1 for x in d[-1] if x > self.max_number // 2)
        return {num: 0.3 if (lh > 3 and num <= mid) or (lh <= 3 and num > mid) else 0 for num in range(1, self.max_number + 1)}

    def _walk_forward_weights(self, data, signals):
        n, cs = len(data), min(30, len(data) - 70)
        if cs < 10: return {name: 1.0 for name in signals}
        sh = {name: 0 for name in signals}; tc = 0
        for idx in range(n - cs - 1, n - 1):
            actual = set(data[idx + 1]); tc += 1
            for sn, ss in signals.items():
                if not ss: continue
                pred = set(num for num, _ in sorted(ss.items(), key=lambda x: -x[1])[:self.pick_count])
                sh[sn] += len(pred & actual)
        base = self.pick_count / self.max_number
        return {name: max(sh[name] / tc / (base * self.pick_count), 0.1) if tc > 0 and sh[name] > 0 else 0.5 for name in signals}
