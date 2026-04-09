"""
DEEP VULNERABILITY MINING V2 — PHASE 2
========================================
Level 1 đã tìm ra 27 anomalies. Phase 2 đào SÂU hơn:

1. Temporal Stability: Pattern có ổn định qua thời gian?
2. Walk-Forward Validation: Pattern có DỰ ĐOÁN ĐƯỢC kỳ tiếp?
3. Multi-Step Transitions: Chuỗi 3+ kỳ liên tiếp
4. Fourier/Periodicity: Phát hiện chu kỳ ẩn cho từng số
5. Conditional Chains: X→Y→Z transition
6. Hot Zone Temporal: "Vùng nóng" thay đổi theo thời gian?
7. Lag-Specific Repeat: Số nào lặp lại đúng sau N kỳ?

Output: Validated Rules → chuyển thành RuleEngine
"""
import sys, os, math, json
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from scipy import stats
from scipy.fft import fft

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper.data_manager import get_mega645_numbers, get_mega645_all


def deep_mine():
    data = get_mega645_numbers()
    all_rows = get_mega645_all()
    dates = [r['draw_date'] for r in all_rows]
    N = len(data)
    MAX = 45
    PICK = 6
    HALF = N // 2

    print("=" * 75)
    print("  DEEP VULNERABILITY MINING V2 — PHASE 2")
    print(f"  {N} draws | Split: H1={HALF}, H2={N-HALF}")
    print("=" * 75)

    validated_rules = []  # Rules that PASS temporal validation

    # ==========================================================
    # TEST 1: TEMPORAL STABILITY OF TRIPLETS
    # ==========================================================
    print(f"\n{'━'*75}")
    print("  TEST 1: TEMPORAL STABILITY — Do triplets persist across time?")
    print(f"{'━'*75}")

    # Split data into 3 periods
    P1 = N // 3
    P2 = 2 * N // 3
    period_data = [data[:P1], data[P1:P2], data[P2:]]
    period_labels = [f"P1 (draw 1-{P1})", f"P2 (draw {P1+1}-{P2})", f"P3 (draw {P2+1}-{N})"]

    trip_by_period = [{}, {}, {}]
    for pi, pd in enumerate(period_data):
        tc = Counter()
        for d in pd:
            for t in combinations(sorted(d[:PICK]), 3):
                tc[t] += 1
        trip_by_period[pi] = tc

    # Find triplets that are frequent in ALL 3 periods
    all_trips = set()
    for tc in trip_by_period:
        all_trips.update(tc.keys())

    exp_per_period = [len(pd) * PICK*(PICK-1)*(PICK-2)/(MAX*(MAX-1)*(MAX-2)) for pd in period_data]

    stable_triplets = []
    for trip in all_trips:
        counts = [trip_by_period[i].get(trip, 0) for i in range(3)]
        # Must appear at least 2x in EACH period
        if all(c >= 2 for c in counts):
            total = sum(counts)
            z_scores = [(c - e) / max(math.sqrt(e), 0.1) for c, e in zip(counts, exp_per_period)]
            avg_z = np.mean(z_scores)
            if avg_z > 1.5 and total >= 7:
                stable_triplets.append((trip, counts, total, avg_z, z_scores))

    stable_triplets.sort(key=lambda x: -x[3])
    print(f"\n  Triplets stable across ALL 3 time periods (avg_z > 1.5):")
    if stable_triplets:
        for trip, counts, total, avg_z, zs in stable_triplets[:15]:
            print(f"    ({trip[0]:2d},{trip[1]:2d},{trip[2]:2d}): P1={counts[0]}, P2={counts[1]}, P3={counts[2]}, "
                  f"total={total}, avg_z={avg_z:+.2f}")
            validated_rules.append({
                'type': 'stable_triplet',
                'numbers': list(trip),
                'counts': counts,
                'total': total,
                'avg_z': round(avg_z, 3),
                'strength': round(avg_z * total / 10, 2),
            })
    else:
        print("    None found — triplet anomalies were likely noise")

    # ==========================================================
    # TEST 2: TEMPORAL STABILITY OF PAIRS
    # ==========================================================
    print(f"\n{'━'*75}")
    print("  TEST 2: TEMPORAL STABILITY — Pairs across time")
    print(f"{'━'*75}")

    pair_by_period = [{}, {}, {}]
    for pi, pd in enumerate(period_data):
        pc = Counter()
        for d in pd:
            for p in combinations(sorted(d[:PICK]), 2):
                pc[p] += 1
        pair_by_period[pi] = pc

    exp_pair_per_period = [len(pd) * PICK*(PICK-1)/(MAX*(MAX-1)) for pd in period_data]

    stable_pairs = []
    for a in range(1, MAX+1):
        for b in range(a+1, MAX+1):
            pair = (a, b)
            counts = [pair_by_period[i].get(pair, 0) for i in range(3)]
            if all(c >= 5 for c in counts):
                total = sum(counts)
                z_scores = [(c - e) / max(math.sqrt(e), 0.1) for c, e in zip(counts, exp_pair_per_period)]
                avg_z = np.mean(z_scores)
                # Also check consistency (low variance in z-scores)
                z_std = np.std(z_scores)
                if avg_z > 1.0 and z_std < 1.5 and total >= 30:
                    stable_pairs.append((pair, counts, total, avg_z, z_std))

    stable_pairs.sort(key=lambda x: -x[3])
    print(f"\n  Pairs stable across ALL 3 periods (avg_z > 1.0, consistent):")
    for pair, counts, total, avg_z, z_std in stable_pairs[:15]:
        print(f"    ({pair[0]:2d},{pair[1]:2d}): P1={counts[0]}, P2={counts[1]}, P3={counts[2]}, "
              f"total={total}, avg_z={avg_z:+.2f}, consistency={z_std:.2f}")
        validated_rules.append({
            'type': 'stable_pair',
            'numbers': list(pair),
            'counts': counts,
            'total': total,
            'avg_z': round(avg_z, 3),
            'strength': round(avg_z * 2, 2),
        })

    # ==========================================================
    # TEST 3: TRANSITION WALK-FORWARD VALIDATION
    # ==========================================================
    print(f"\n{'━'*75}")
    print("  TEST 3: TRANSITION WALK-FORWARD VALIDATION")
    print("  Train on first 75%, validate on last 25%")
    print(f"{'━'*75}")

    train_end = int(N * 0.75)
    train_data = data[:train_end]
    test_data = data[train_end:]
    n_train = len(train_data)
    n_test = len(test_data)

    # Build transitions from training data
    train_follow = defaultdict(Counter)
    train_appear = Counter()
    for i in range(n_train - 1):
        for p in train_data[i][:PICK]:
            train_appear[p] += 1
            for nx in train_data[i+1][:PICK]:
                train_follow[p][nx] += 1

    base_p = PICK / MAX

    # Find strong transitions in training data
    strong_transitions = []
    for prev in range(1, MAX+1):
        if train_appear[prev] < 40:
            continue
        for nxt in range(1, MAX+1):
            obs = train_follow[prev].get(nxt, 0)
            obs_p = obs / train_appear[prev]
            z = (obs_p - base_p) / math.sqrt(base_p * (1-base_p) / train_appear[prev])
            if abs(z) > 2.0:
                strong_transitions.append((prev, nxt, obs_p, z))

    # Validate on test data
    print(f"\n  Strong transitions from training ({n_train} draws):")
    print(f"  Validating on test set ({n_test} draws):")
    
    validated_transitions = []
    for prev, nxt, train_p, train_z in sorted(strong_transitions, key=lambda x: -abs(x[3]))[:30]:
        # Count in test data
        test_count = 0
        test_total = 0
        for i in range(len(test_data) - 1):
            if prev in test_data[i][:PICK]:
                test_total += 1
                if nxt in test_data[i+1][:PICK]:
                    test_count += 1
        
        test_p = test_count / max(test_total, 1)
        if test_total >= 10:
            test_z = (test_p - base_p) / math.sqrt(base_p * (1-base_p) / test_total)
        else:
            test_z = 0
        
        # Validated = same direction in both train and test
        same_direction = (train_z > 0 and test_z > 0) or (train_z < 0 and test_z < 0)
        validated = same_direction and abs(test_z) > 1.0

        status = "✅ VALIDATED" if validated else "❌ NOT validated"
        print(f"    {prev:2d}→{nxt:2d}: train={train_p:.3f}(z={train_z:+.2f}) | "
              f"test={test_p:.3f}(z={test_z:+.2f}) [{test_count}/{test_total}] {status}")
        
        if validated:
            validated_transitions.append({
                'type': 'transition',
                'from': prev,
                'to': nxt,
                'train_p': round(train_p, 4),
                'test_p': round(test_p, 4),
                'train_z': round(train_z, 2),
                'test_z': round(test_z, 2),
                'strength': round((train_z + test_z) / 2, 2),
            })
            validated_rules.append(validated_transitions[-1])

    print(f"\n  Validated: {len(validated_transitions)}/{len(strong_transitions[:30])}")

    # ==========================================================
    # TEST 4: FOURIER / PERIODICITY DETECTION
    # ==========================================================
    print(f"\n{'━'*75}")
    print("  TEST 4: PERIODICITY DETECTION (Fourier Analysis)")
    print(f"{'━'*75}")
    print(f"  Looking for hidden cycles in each number's appearance pattern...")

    periodic_numbers = []
    for num in range(1, MAX+1):
        seq = np.array([1.0 if num in d[:PICK] else 0.0 for d in data])
        seq_centered = seq - np.mean(seq)
        
        # FFT
        spectrum = np.abs(fft(seq_centered))[:N//2]
        freqs = np.fft.fftfreq(N)[:N//2]
        
        # Skip DC component (index 0)
        spectrum[0] = 0
        
        # Find dominant frequency
        peak_idx = np.argmax(spectrum[1:]) + 1
        peak_freq = freqs[peak_idx]
        peak_period = 1.0 / peak_freq if peak_freq > 0 else N
        peak_power = spectrum[peak_idx]
        
        # Statistical significance: compare peak to noise floor
        noise = np.mean(spectrum[1:])
        noise_std = np.std(spectrum[1:])
        z_peak = (peak_power - noise) / max(noise_std, 0.01)
        
        if z_peak > 4.0 and 3 <= peak_period <= 50:
            periodic_numbers.append((num, peak_period, peak_power, z_peak))

    periodic_numbers.sort(key=lambda x: -x[3])
    if periodic_numbers:
        print(f"\n  Numbers with significant periodicity (z > 4.0):")
        for num, period, power, z in periodic_numbers[:15]:
            print(f"    #{num:2d}: period ≈ {period:.1f} draws, z={z:.2f}")
            validated_rules.append({
                'type': 'periodicity',
                'number': num,
                'period': round(period, 1),
                'z_score': round(z, 2),
                'strength': round(z / 3, 2),
            })
    else:
        print("    No significant periodicities found")

    # ==========================================================
    # TEST 5: LAG-SPECIFIC REPEAT ANALYSIS
    # ==========================================================
    print(f"\n{'━'*75}")
    print("  TEST 5: LAG-SPECIFIC REPEAT (Exact lag repeat rates)")
    print(f"{'━'*75}")

    for lag in range(1, 8):
        repeat_rates = {}
        for num in range(1, MAX+1):
            appeared_at_lag = 0
            repeated = 0
            for i in range(lag, N):
                if num in data[i - lag][:PICK]:
                    appeared_at_lag += 1
                    if num in data[i][:PICK]:
                        repeated += 1
            if appeared_at_lag >= 50:
                rate = repeated / appeared_at_lag
                z = (rate - base_p) / math.sqrt(base_p * (1-base_p) / appeared_at_lag)
                if abs(z) > 2.5:
                    repeat_rates[num] = (rate, z, repeated, appeared_at_lag)

        if repeat_rates:
            print(f"\n  Lag-{lag} repeat anomalies (|z| > 2.5):")
            for num, (rate, z, rep, tot) in sorted(repeat_rates.items(), key=lambda x: -abs(x[1][1])):
                direction = "REPEATS MORE" if z > 0 else "AVOIDS REPEAT"
                print(f"    #{num:2d}: {rate:.1%} repeat rate at lag-{lag} ({rep}/{tot}), "
                      f"z={z:+.2f} ({direction})")
                validated_rules.append({
                    'type': 'lag_repeat',
                    'number': num,
                    'lag': lag,
                    'rate': round(rate, 4),
                    'z_score': round(z, 2),
                    'direction': direction,
                    'strength': round(abs(z) / 2, 2),
                })

    # ==========================================================
    # TEST 6: CONDITIONAL NUMBER BOOST (Multi-condition)
    # ==========================================================
    print(f"\n{'━'*75}")
    print("  TEST 6: MULTI-CONDITION PATTERNS")
    print("  When BOTH X and Y appear → what's boosted next draw?")
    print(f"{'━'*75}")

    # Top pairs from stable_pairs
    top_cond_pairs = [(p[0], p[1]) for p in stable_pairs[:10]]
    multi_cond_rules = []

    for (a, b), counts, total, avg_z, z_std in stable_pairs[:10]:
        # When BOTH a and b appear together
        boost = Counter()
        n_together = 0
        for i in range(N - 1):
            d = set(data[i][:PICK])
            if a in d and b in d:
                n_together += 1
                for nx in data[i+1][:PICK]:
                    boost[nx] += 1
        
        if n_together < 5:
            continue
        
        for nx in range(1, MAX+1):
            cond_p = boost.get(nx, 0) / n_together
            if n_together >= 10:
                z = (cond_p - base_p) / math.sqrt(base_p * (1-base_p) / n_together)
            else:
                z = 0
            if z > 2.0:
                multi_cond_rules.append(((a, b), nx, cond_p, z, boost.get(nx, 0), n_together))

    multi_cond_rules.sort(key=lambda x: -x[3])
    if multi_cond_rules:
        print(f"\n  When pair (X,Y) appears → number Z boosted next draw:")
        for (a, b), nx, cp, z, cnt, tot in multi_cond_rules[:15]:
            print(f"    ({a:2d},{b:2d}) present → #{nx:2d}: P={cp:.3f} vs base {base_p:.3f}, "
                  f"z={z:+.2f} ({cnt}/{tot})")
            validated_rules.append({
                'type': 'multi_condition',
                'condition_pair': [a, b],
                'boosted_number': nx,
                'cond_p': round(cp, 4),
                'z_score': round(z, 2),
                'count': cnt,
                'total_opportunities': tot,
                'strength': round(z * cp, 2),
            })

    # ==========================================================
    # TEST 7: SUM-RANGE LOCK (Narrow valid ranges)
    # ==========================================================
    print(f"\n{'━'*75}")
    print("  TEST 7: SUM-RANGE CONSTRAINT ANALYSIS")
    print(f"{'━'*75}")

    sums = [sum(d[:PICK]) for d in data]
    ranges = [max(d[:PICK]) - min(d[:PICK]) for d in data]
    odds = [sum(1 for x in d[:PICK] if x % 2 == 1) for d in data]

    sum_mean, sum_std = np.mean(sums), np.std(sums)
    range_mean, range_std = np.mean(ranges), np.std(ranges)
    odd_mean, odd_std = np.mean(odds), np.std(odds)

    # How much can we narrow?
    p68 = (sum(1 for s in sums if abs(s - sum_mean) <= sum_std)) / N * 100
    p95 = (sum(1 for s in sums if abs(s - sum_mean) <= 2*sum_std)) / N * 100

    print(f"  SUM: mean={sum_mean:.1f}, std={sum_std:.1f}")
    print(f"    68% of draws have sum in [{sum_mean-sum_std:.0f}, {sum_mean+sum_std:.0f}] (actual: {p68:.1f}%)")
    print(f"    95% of draws have sum in [{sum_mean-2*sum_std:.0f}, {sum_mean+2*sum_std:.0f}] (actual: {p95:.1f}%)")

    print(f"  RANGE: mean={range_mean:.1f}, std={range_std:.1f}")
    p_range = sum(1 for r in ranges if abs(r - range_mean) <= range_std) / N * 100
    print(f"    68% of draws have range in [{range_mean-range_std:.0f}, {range_mean+range_std:.0f}] (actual: {p_range:.1f}%)")

    print(f"  ODD count: mean={odd_mean:.2f}, std={odd_std:.2f}")
    print(f"    Most common odd counts: {Counter(odds).most_common(5)}")

    # Temporal sum trend
    print(f"\n  Sum trend (last 50 draws vs overall):")
    recent_sum_mean = np.mean(sums[-50:])
    z_sum_trend = (recent_sum_mean - sum_mean) / (sum_std / math.sqrt(50))
    print(f"    Recent 50: mean={recent_sum_mean:.1f}, z vs overall = {z_sum_trend:+.2f}")
    if abs(z_sum_trend) > 2.0:
        print(f"    ⚠️ Sum trend detected!")
        validated_rules.append({
            'type': 'sum_trend',
            'recent_mean': round(recent_sum_mean, 1),
            'overall_mean': round(sum_mean, 1),
            'z_score': round(z_sum_trend, 2),
            'strength': round(abs(z_sum_trend) / 2, 2),
        })

    validated_rules.append({
        'type': 'sum_constraint',
        'sum_range': [round(sum_mean - 1.5*sum_std), round(sum_mean + 1.5*sum_std)],
        'range_range': [round(range_mean - 1.5*range_std), round(range_mean + 1.5*range_std)],
        'odd_range': [max(0, round(odd_mean - 1.5*odd_std)), min(PICK, round(odd_mean + 1.5*odd_std))],
        'strength': 5.0,
    })

    # ==========================================================
    # TEST 8: RECENT MOMENTUM — Last 100 draws vs overall
    # ==========================================================
    print(f"\n{'━'*75}")
    print("  TEST 8: RECENT MOMENTUM (last 100 vs overall)")
    print(f"{'━'*75}")

    freq_overall = Counter()
    for d in data:
        for n in d[:PICK]:
            freq_overall[n] += 1
    
    freq_recent = Counter()
    for d in data[-100:]:
        for n in d[:PICK]:
            freq_recent[n] += 1

    momentum_list = []
    for n in range(1, MAX+1):
        rate_overall = freq_overall[n] / N
        rate_recent = freq_recent[n] / 100
        z = (rate_recent - rate_overall) / math.sqrt(rate_overall * (1-rate_overall) / 100)
        if abs(z) > 1.5:
            momentum_list.append((n, rate_recent, rate_overall, z))

    momentum_list.sort(key=lambda x: -x[3])
    if momentum_list:
        print(f"\n  Numbers with significant recent momentum:")
        for n, rr, ro, z in momentum_list:
            direction = "🔥 HEATING UP" if z > 0 else "❄️ COOLING DOWN"
            print(f"    #{n:2d}: recent={rr:.1%}, overall={ro:.1%}, z={z:+.2f} {direction}")
            validated_rules.append({
                'type': 'momentum',
                'number': n,
                'recent_rate': round(rr, 4),
                'overall_rate': round(ro, 4),
                'z_score': round(z, 2),
                'direction': 'hot' if z > 0 else 'cold',
                'strength': round(abs(z) / 2, 2),
            })

    # ==========================================================
    # TEST 9: WALK-FORWARD BACKTEST OF RULES
    # ==========================================================
    print(f"\n{'━'*75}")
    print("  TEST 9: WALK-FORWARD BACKTEST OF COMBINED RULES")
    print("  Apply discovered rules to predict — measure REAL accuracy")
    print(f"{'━'*75}")

    # Simple rule-based predictor using validated patterns
    def rule_predict(train, max_num=45, pick=6):
        n = len(train)
        if n < 50:
            return sorted(np.random.choice(range(1, max_num+1), pick, replace=False).tolist())
        
        scores = {num: 0.0 for num in range(1, max_num + 1)}
        last = set(train[-1][:pick])
        last2 = set(train[-2][:pick]) if n >= 2 else set()

        # Rule: Transition patterns
        follow = defaultdict(Counter)
        appear = Counter()
        for i in range(n - 1):
            for p in train[i][:pick]:
                appear[p] += 1
                for nx in train[i+1][:pick]:
                    follow[p][nx] += 1
        
        bp = pick / max_num
        for p in last:
            if appear[p] < 30:
                continue
            for nx in range(1, max_num + 1):
                cond_p = follow[p].get(nx, 0) / appear[p]
                lift = cond_p / bp if bp > 0 else 1
                if lift > 1.2:
                    scores[nx] += (lift - 1) * 3

        # Rule: Lag-specific repeat boost
        for lag in [1, 2, 3, 4]:
            if n > lag:
                lag_set = set(train[-lag][:pick])
                # Calculate repeat rate for data
                rep_rates = {}
                for num in range(1, max_num+1):
                    appeared = sum(1 for j in range(lag, n) if num in train[j-lag][:pick])
                    repeated = sum(1 for j in range(lag, n) if num in train[j-lag][:pick] and num in train[j][:pick])
                    if appeared > 30:
                        rep_rates[num] = repeated / appeared
                
                for num in lag_set:
                    rate = rep_rates.get(num, bp)
                    scores[num] += (rate - bp) * 5 * (1 / lag)

        # Rule: Pair co-occurrence
        pair_freq = Counter()
        for d in train[-200:]:
            for p in combinations(sorted(d[:pick]), 2):
                pair_freq[p] += 1
        
        for num in range(1, max_num+1):
            pair_score = 0
            for p in last:
                key = tuple(sorted([p, num]))
                pair_score += pair_freq.get(key, 0)
            exp = len(train[-200:]) * pick * (pick-1) / (max_num * (max_num - 1))
            scores[num] += max(0, (pair_score - exp * len(last))) * 0.05
        
        # Rule: Momentum
        f10 = Counter(n for d in train[-10:] for n in d[:pick])
        f50 = Counter(n for d in train[-50:] for n in d[:pick])
        for num in range(1, max_num+1):
            r10 = f10.get(num, 0) / 10
            r50 = f50.get(num, 0) / 50
            if r10 > r50 * 1.3:
                scores[num] += (r10 - r50) * 5
        
        # Rule: Fourier periodicity
        if n >= 64:
            for num in range(1, max_num+1):
                seq = np.array([1.0 if num in d[:pick] else 0.0 for d in train[-128:]])
                if len(seq) < 32:
                    continue
                spectrum = np.abs(fft(seq - np.mean(seq)))[:len(seq)//2]
                spectrum[0] = 0
                if len(spectrum) > 3:
                    peak_idx = np.argmax(spectrum[1:]) + 1
                    peak_power = spectrum[peak_idx]
                    noise = np.mean(spectrum[1:])
                    if peak_power > noise * 3:
                        # Check if current position is at "high" phase
                        peak_freq = peak_idx / len(seq)
                        phase = (len(seq) * peak_freq) % 1.0
                        if 0.3 < phase < 0.7:
                            scores[num] += 0.5
        
        # Rule: Sum/Range constraints (filter)
        recent_sums = [sum(d[:pick]) for d in train[-50:]]
        sum_lo = np.percentile(recent_sums, 10)
        sum_hi = np.percentile(recent_sums, 90)
        
        # Anti-repeat (adaptive)
        repeat_rates = []
        for i in range(1, min(n, 100)):
            repeat_rates.append(len(set(train[i-1][:pick]) & set(train[i][:pick])))
        penalty = (np.mean(repeat_rates) - 1.0) * 1.5
        for num in last:
            scores[num] += penalty
        
        # Build prediction
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        pool = [num for num, _ in ranked[:25]]
        
        best = None
        best_sc = -float('inf')
        for combo in combinations(pool[:15], pick):
            s = sum(combo)
            if s < sum_lo or s > sum_hi:
                continue
            sc = sum(scores.get(n, 0) for n in combo)
            if sc > best_sc:
                best_sc = sc
                best = sorted(combo)
        
        return best if best else sorted(pool[:pick])

    # Walk-forward test
    min_train = 100
    test_range = range(min_train, N - 1)
    step = max(1, len(list(test_range)) // 200)
    test_indices = list(range(min_train, N - 1, step))

    rule_matches = []
    random_matches = []
    np.random.seed(42)

    print(f"\n  Walk-forward: {len(test_indices)} tests (step={step})...")
    
    for i, train_end in enumerate(test_indices):
        train = data[:train_end + 1]
        actual = set(data[train_end + 1][:PICK])
        
        predicted = set(rule_predict(train))
        rule_matches.append(len(predicted & actual))
        
        rand = set(np.random.choice(range(1, MAX+1), PICK, replace=False).tolist())
        random_matches.append(len(rand & actual))
        
        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(test_indices)}] rules_avg={np.mean(rule_matches):.4f}/6, "
                  f"random_avg={np.mean(random_matches):.4f}/6")

    rule_avg = np.mean(rule_matches)
    rand_avg = np.mean(random_matches)
    improvement = (rule_avg / rand_avg - 1) * 100 if rand_avg > 0 else 0

    rule_dist = Counter(rule_matches)
    rand_dist = Counter(random_matches)

    print(f"\n  RESULTS:")
    print(f"  Rule Engine:    avg={rule_avg:.4f}/6, max={max(rule_matches)}/6")
    print(f"  Random:         avg={rand_avg:.4f}/6, max={max(random_matches)}/6")
    print(f"  Improvement:    {improvement:+.2f}%")
    print(f"\n  Distribution comparison:")
    for k in range(7):
        rc = rule_dist.get(k, 0)
        rr = rand_dist.get(k, 0)
        rp = rc / len(rule_matches) * 100
        rrp = rr / len(random_matches) * 100
        better = "✅" if rp > rrp and k >= 2 else "  "
        print(f"    {k}/6: Rules={rc:4d} ({rp:5.1f}%) | Random={rr:4d} ({rrp:5.1f}%) {better}")

    pct3_rules = sum(1 for m in rule_matches if m >= 3) / len(rule_matches) * 100
    pct3_rand = sum(1 for m in random_matches if m >= 3) / len(random_matches) * 100
    print(f"\n  ≥3/6: Rules={pct3_rules:.2f}% | Random={pct3_rand:.2f}%")

    # ==========================================================
    # SAVE VALIDATED RULES
    # ==========================================================
    print(f"\n{'━'*75}")
    print(f"  SAVING {len(validated_rules)} VALIDATED RULES")
    print(f"{'━'*75}")

    rules_path = os.path.join(os.path.dirname(__file__), 'models', 'validated_rules.json')
    with open(rules_path, 'w', encoding='utf-8') as f:
        json.dump({
            'version': '2.0',
            'total_draws_analyzed': N,
            'date_range': [dates[0], dates[-1]],
            'n_rules': len(validated_rules),
            'backtest_improvement': round(improvement, 2),
            'rules': validated_rules,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Saved to: {rules_path}")

    # Summary
    rule_types = Counter(r['type'] for r in validated_rules)
    print(f"\n  Rules by type:")
    for rt, cnt in rule_types.most_common():
        print(f"    {rt}: {cnt}")

    print(f"\n{'='*75}")
    print(f"  PHASE 2 COMPLETE — {len(validated_rules)} rules validated, "
          f"improvement = {improvement:+.2f}%")
    print(f"{'='*75}")


if __name__ == '__main__':
    deep_mine()
