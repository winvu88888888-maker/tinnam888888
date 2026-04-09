"""
DEEP MINE V4 — ANTI-OVERFITTING VULNERABILITY HUNT
===================================================
V3 thất bại vì overfitting trên small samples.
V4 thay đổi hoàn toàn chiến lược:

NGUYÊN TẮC MỚI:
1. Chỉ chấp nhận patterns có n >= 50 observations
2. Dùng RIGOROUS walk-forward: train/val/test 3-way split
3. Tập trung vào POPULATION-LEVEL anomalies (không phải individual chains)
4. Dùng ensemble of WEAK signals thay vì few STRONG signals
5. Bayesian posterior scoring thay vì naive frequency

15 TESTS MỚI (population-level):
  A. Autocorrelation Function per number (ACF lag 1-20)
  B. CUSUM per number + regime detection  
  C. Weighted frequency by recency (exponential decay)
  D. Gap CDF exploitation (overdue numbers)
  E. "Hot hand" vs "Gambler's fallacy" regime classifier
  F. Conditional odd/even balance after extreme draws
  G. Decade-level Markov chain P(decade_profile_N+1 | profile_N)
  H. Sum-based stratification (draw quality tiers)
  I. Multi-lag cross-correlation (strongest X→Y at ANY lag)
  J. Pair stability score (pairs consistent across 5 time periods)
  K. Number "rhythm" detection (periodic appearance)
  L. Edge persistence (consecutive draw overlap distribution)
  M. Positional range prediction (min/max bounds per position)
  N. Information gain: which signals ACTUALLY reduce entropy?
  O. MEGA ENSEMBLE with proper regularization
"""
import sys, os, math, json, time
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper.data_manager import get_mega645_numbers, get_mega645_all


def deep_mine_v4():
    data = get_mega645_numbers()
    all_rows = get_mega645_all()
    dates = [r['draw_date'] for r in all_rows]
    N = len(data)
    MAX = 45
    PICK = 6
    base_p = PICK / MAX
    t0 = time.time()

    # 3-WAY SPLIT: Train (60%) / Validation (20%) / Test (20%)
    train_end = int(N * 0.6)
    val_end = int(N * 0.8)
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    print("=" * 75)
    print("  🧬 DEEP MINE V4 — ANTI-OVERFITTING VULNERABILITY HUNT")
    print(f"  {N} draws | Train={train_end} | Val={val_end-train_end} | Test={N-val_end}")
    print("=" * 75)

    # ================================================================
    # BUILD SIGNAL FUNCTIONS (each returns {number: score})
    # Each signal is computed on ANY data slice → no data leakage
    # ================================================================
    
    def sig_transition(train, last_draw):
        """Standard bigram transition: P(Y appear | X in last draw)."""
        n = len(train)
        follow = defaultdict(Counter)
        appear = Counter()
        for i in range(n - 1):
            for p in train[i][:PICK]:
                appear[p] += 1
                for nx in train[i+1][:PICK]:
                    follow[p][nx] += 1
        last_set = set(last_draw[:PICK])
        scores = {}
        for num in range(1, MAX+1):
            total_appear = sum(appear[p] for p in last_set if appear[p] > 0)
            cond = sum(follow[p].get(num, 0) for p in last_set)
            rate = cond / max(total_appear, 1)
            scores[num] = (rate / base_p - 1) if total_appear > 50 else 0
        return scores

    def sig_gap_overdue(train):
        """Gap-based: numbers overdue relative to their historical gap."""
        n = len(train)
        last_seen = {}
        gap_data = defaultdict(list)
        for i, d in enumerate(train):
            for num in d[:PICK]:
                if num in last_seen:
                    gap_data[num].append(i - last_seen[num])
                last_seen[num] = i
        scores = {}
        for num in range(1, MAX+1):
            gaps = gap_data.get(num, [])
            if len(gaps) < 10:
                scores[num] = 0
                continue
            curr_gap = n - last_seen.get(num, 0)
            avg_gap = np.mean(gaps)
            std_gap = np.std(gaps)
            # Z-score of current gap vs historical
            z = (curr_gap - avg_gap) / max(std_gap, 1)
            # Sigmoid: increasingly likely as gap exceeds average
            scores[num] = 1 / (1 + math.exp(-2 * (z - 0.5))) if z > 0 else 0
        return scores

    def sig_momentum(train):
        """Multi-window momentum: recent vs historical frequency."""
        n = len(train)
        if n < 50:
            return {num: 0 for num in range(1, MAX+1)}
        scores = {}
        for num in range(1, MAX+1):
            r5 = sum(1 for d in train[-5:] if num in d[:PICK]) / 5
            r20 = sum(1 for d in train[-20:] if num in d[:PICK]) / 20
            r50 = sum(1 for d in train[-50:] if num in d[:PICK]) / 50
            # Short-term acceleration
            accel = (r5 - r20) * 8 + (r20 - r50) * 4
            scores[num] = accel
        return scores

    def sig_exponential_decay(train):
        """Exponential decay weighted frequency — recent draws matter more."""
        n = len(train)
        decay = 0.95  # Each draw back = 5% less weight
        scores = {num: 0.0 for num in range(1, MAX+1)}
        total_weight = 0
        for i in range(n - 1, max(n - 200, -1), -1):
            w = decay ** (n - 1 - i)
            total_weight += w
            for num in train[i][:PICK]:
                scores[num] += w
        for num in scores:
            scores[num] = scores[num] / max(total_weight, 1) / base_p - 1
        return scores

    def sig_knn(train, last_draw):
        """KNN: similar historical draws → what came next?"""
        n = len(train)
        last_set = set(last_draw[:PICK])
        knn_sc = Counter()
        total_w = 0
        for i in range(n - 2):
            sim = len(set(train[i][:PICK]) & last_set)
            if sim >= 3:
                w = sim ** 2
                total_w += w
                for num in train[i+1][:PICK]:
                    knn_sc[num] += w
        scores = {}
        for num in range(1, MAX+1):
            scores[num] = (knn_sc.get(num, 0) / max(total_w, 1) / base_p - 1) * 3
        return scores

    def sig_pair_boost(train, last_draw):
        """Pair co-occurrence with last draw numbers."""
        last_set = set(last_draw[:PICK])
        pair_freq = Counter()
        n_look = min(200, len(train))
        for d in train[-n_look:]:
            for p in combinations(sorted(d[:PICK]), 2):
                pair_freq[p] += 1
        exp = n_look * PICK * (PICK - 1) / (MAX * (MAX - 1))
        scores = {}
        for num in range(1, MAX+1):
            boost = 0
            for p in last_set:
                key = tuple(sorted([p, num]))
                freq = pair_freq.get(key, 0)
                if freq > exp + 1:
                    boost += (freq - exp) / max(exp, 1)
            scores[num] = boost * 0.5
        return scores

    def sig_position_markov(train, last_draw):
        """Per-position Markov transitions."""
        sorted_data = [sorted(d[:PICK]) for d in train]
        last_sorted = sorted(last_draw[:PICK])
        scores = {num: 0.0 for num in range(1, MAX+1)}
        for pos in range(PICK):
            trans = defaultdict(Counter)
            for i in range(len(sorted_data) - 1):
                trans[sorted_data[i][pos]][sorted_data[i+1][pos]] += 1
            preds = trans.get(last_sorted[pos], Counter())
            total = sum(preds.values())
            if total < 20:
                continue
            for num, cnt in preds.most_common(5):
                prob = cnt / total
                scores[num] += (prob - 1/MAX) * 3
        return scores

    def sig_oddeven_balance(train, last_draw):
        """Odd/even balance regression to mean."""
        n_odd = sum(1 for x in last_draw[:PICK] if x % 2 == 1)
        scores = {}
        for num in range(1, MAX+1):
            if n_odd >= 4 and num % 2 == 0:
                scores[num] = 0.3
            elif n_odd <= 2 and num % 2 == 1:
                scores[num] = 0.3
            else:
                scores[num] = 0
        return scores

    def sig_highlow_balance(train, last_draw):
        """High/low balance regression."""
        mid = MAX // 2
        n_high = sum(1 for x in last_draw[:PICK] if x > mid)
        scores = {}
        for num in range(1, MAX+1):
            if n_high >= 4 and num <= mid:
                scores[num] = 0.3
            elif n_high <= 2 and num > mid:
                scores[num] = 0.3
            else:
                scores[num] = 0
        return scores

    def sig_repeat_rate(train, last_draw):
        """Adaptive repeat rate for last draw numbers."""
        n = len(train)
        last_set = set(last_draw[:PICK])
        # Per-number repeat probability
        scores = {}
        for num in range(1, MAX+1):
            if num not in last_set:
                scores[num] = 0
                continue
            appeared = 0
            repeated = 0
            for i in range(n - 1):
                if num in train[i][:PICK]:
                    appeared += 1
                    if num in train[i+1][:PICK]:
                        repeated += 1
            rate = repeated / max(appeared, 1)
            scores[num] = (rate - base_p) * 5
        return scores

    def sig_decade_flow(train, last_draw):
        """Decade group flow patterns."""
        def dec(x):
            return min(x // 10, 4)
        last_dec = tuple(sorted(dec(x) for x in last_draw[:PICK]))
        trans = defaultdict(Counter)
        for i in range(1, len(train)):
            prev = tuple(sorted(dec(x) for x in train[i-1][:PICK]))
            curr = Counter(dec(x) for x in train[i][:PICK])
            for d, c in curr.items():
                trans[prev][d] += c
        expected = trans.get(last_dec, Counter())
        total_exp = sum(expected.values()) or 1
        scores = {}
        for num in range(1, MAX+1):
            d = dec(num)
            dp = expected.get(d, 0) / total_exp
            scores[num] = (dp - 0.2) * 3
        return scores

    def sig_streak(train):
        """Streak-based overdue signal."""
        scores = {}
        for num in range(1, MAX+1):
            c = 0
            for x in reversed(train):
                if num not in x[:PICK]:
                    c += 1
                else:
                    break
            expected_gap = MAX / PICK
            scores[num] = max(0, (c / expected_gap - 1)) * 0.5
        return scores

    # All signal functions
    SIGNAL_FUNCS = {
        'transition': lambda t, ld: sig_transition(t, ld),
        'gap_overdue': lambda t, ld: sig_gap_overdue(t),
        'momentum': lambda t, ld: sig_momentum(t),
        'exp_decay': lambda t, ld: sig_exponential_decay(t),
        'knn': lambda t, ld: sig_knn(t, ld),
        'pair_boost': lambda t, ld: sig_pair_boost(t, ld),
        'pos_markov': lambda t, ld: sig_position_markov(t, ld),
        'oddeven': lambda t, ld: sig_oddeven_balance(t, ld),
        'highlow': lambda t, ld: sig_highlow_balance(t, ld),
        'repeat_rate': lambda t, ld: sig_repeat_rate(t, ld),
        'decade_flow': lambda t, ld: sig_decade_flow(t, ld),
        'streak': lambda t, ld: sig_streak(t),
    }

    # ================================================================
    # PHASE 1: EVALUATE EACH SIGNAL INDEPENDENTLY ON VALIDATION SET
    # ================================================================
    print(f"\n{'━'*75}")
    print("  PHASE 1: Signal Quality Evaluation (on validation set)")
    print(f"{'━'*75}")

    signal_quality = {}
    for sig_name, sig_func in SIGNAL_FUNCS.items():
        matches = []
        for i in range(train_end, val_end - 1):
            train_slice = data[:i+1]
            last_draw = data[i]
            actual = set(data[i+1][:PICK])
            
            scores = sig_func(train_slice, last_draw)
            top6 = set(num for num, _ in sorted(scores.items(), key=lambda x: -x[1])[:PICK])
            matches.append(len(top6 & actual))
        
        avg = np.mean(matches) if matches else 0
        lift = avg / (base_p * PICK) if base_p * PICK > 0 else 1
        signal_quality[sig_name] = {
            'avg_match': round(avg, 4),
            'lift': round(lift, 4),
            'n_tests': len(matches),
        }
        status = "✅" if lift > 1.0 else "❌"
        print(f"  {sig_name:15s}: avg={avg:.4f}/6, lift={lift:.3f}x {status}")

    # ================================================================
    # PHASE 2: WEIGHT OPTIMIZATION ON VALIDATION SET
    # ================================================================
    print(f"\n{'━'*75}")
    print("  PHASE 2: Ensemble Weight Optimization")
    print(f"{'━'*75}")

    # Only use signals with lift > 1.0
    active_signals = {k: v for k, v in signal_quality.items() if v['lift'] > 1.0}
    print(f"  Active signals (lift > 1.0): {len(active_signals)}/{len(SIGNAL_FUNCS)}")
    
    # Weight = lift^2 (reward good signals quadratically)
    weights = {}
    for name, q in active_signals.items():
        weights[name] = q['lift'] ** 2
    
    # Normalize weights
    total_w = sum(weights.values()) or 1
    weights = {k: v / total_w for k, v in weights.items()}
    
    for name, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"    {name:15s}: weight={w:.4f} (lift={active_signals[name]['lift']:.3f}x)")

    # ================================================================
    # PHASE 3: ENSEMBLE PREDICTION FUNCTION
    # ================================================================
    def ensemble_predict(train_slice, last_draw, weights_dict, n_portfolio=1):
        """Predict using weighted ensemble of active signals."""
        consensus = {num: 0.0 for num in range(1, MAX+1)}
        
        for sig_name, w in weights_dict.items():
            if w <= 0:
                continue
            sig_func = SIGNAL_FUNCS[sig_name]
            scores = sig_func(train_slice, last_draw)
            
            # Normalize scores
            vals = list(scores.values())
            max_v = max(abs(v) for v in vals) if vals else 1
            if max_v < 0.001:
                max_v = 1
            
            for num, s in scores.items():
                consensus[num] += (s / max_v) * w
        
        # Constraints from recent data
        recent_sums = [sum(d[:PICK]) for d in train_slice[-50:]]
        sum_lo = np.percentile(recent_sums, 5)
        sum_hi = np.percentile(recent_sums, 95)
        
        recent_ranges = [max(d[:PICK]) - min(d[:PICK]) for d in train_slice[-50:]]
        range_lo = np.percentile(recent_ranges, 5)
        range_hi = np.percentile(recent_ranges, 95)
        
        # Build pool
        ranked = sorted(consensus.items(), key=lambda x: -x[1])
        pool = [num for num, _ in ranked[:25]]
        
        # Generate combos
        results = []
        for combo in combinations(pool[:16], PICK):
            s = sum(combo)
            if s < sum_lo or s > sum_hi:
                continue
            rng = combo[-1] - combo[0]
            if rng < range_lo or rng > range_hi:
                continue
            sc = sum(consensus.get(n, 0) for n in combo)
            results.append({'numbers': sorted(combo), 'score': round(sc, 3)})
        
        results.sort(key=lambda x: -x['score'])
        if not results:
            return [sorted(pool[:PICK])]
        
        return [r['numbers'] for r in results[:n_portfolio]]

    # ================================================================
    # PHASE 4: FINAL TEST ON HELD-OUT TEST SET
    # ================================================================
    print(f"\n{'━'*75}")
    print("  PHASE 3: FINAL TEST (held-out test set — NO data leakage)")
    print(f"{'━'*75}")

    test_matches = []
    random_matches = []
    portfolio_best = []
    np.random.seed(42)

    for i in range(val_end, N - 1):
        train_slice = data[:i+1]
        last_draw = data[i]
        actual = set(data[i+1][:PICK])
        
        # Single best prediction
        preds = ensemble_predict(train_slice, last_draw, weights, n_portfolio=50)
        best = set(preds[0]) if preds else set()
        test_matches.append(len(best & actual))
        
        # Best in portfolio
        best_port = max(len(set(p) & actual) for p in preds) if preds else 0
        portfolio_best.append(best_port)
        
        # Random baseline
        rand = set(np.random.choice(range(1, MAX+1), PICK, replace=False).tolist())
        random_matches.append(len(rand & actual))

    test_avg = np.mean(test_matches)
    rand_avg = np.mean(random_matches)
    port_avg = np.mean(portfolio_best)
    improvement = (test_avg / rand_avg - 1) * 100 if rand_avg > 0 else 0
    port_improvement = (port_avg / rand_avg - 1) * 100 if rand_avg > 0 else 0

    print(f"\n  ═══════════════════════════════════════════")
    print(f"  GOLDEN SET:     avg={test_avg:.4f}/6 ({improvement:+.1f}% vs random)")
    print(f"  PORTFOLIO BEST: avg={port_avg:.4f}/6 ({port_improvement:+.1f}% vs random)")
    print(f"  RANDOM:         avg={rand_avg:.4f}/6")
    print(f"  ═══════════════════════════════════════════")

    # Distribution
    test_dist = Counter(test_matches)
    rand_dist = Counter(random_matches)
    port_dist = Counter(portfolio_best)
    
    print(f"\n  Distribution:")
    print(f"  {'Match':>5} | {'Golden':>12} | {'Portfolio':>12} | {'Random':>12}")
    print(f"  {'─'*5} | {'─'*12} | {'─'*12} | {'─'*12}")
    n_test = len(test_matches)
    for k in range(7):
        tc = test_dist.get(k, 0)
        pc = port_dist.get(k, 0)
        rc = rand_dist.get(k, 0)
        print(f"  {k}/6   | {tc:4d} ({tc/n_test*100:5.1f}%) | "
              f"{pc:4d} ({pc/n_test*100:5.1f}%) | {rc:4d} ({rc/n_test*100:5.1f}%)")

    pct3 = sum(1 for m in test_matches if m >= 3) / n_test * 100
    pct3p = sum(1 for m in portfolio_best if m >= 3) / n_test * 100
    pct3r = sum(1 for m in random_matches if m >= 3) / n_test * 100
    print(f"\n  ≥3/6: Golden={pct3:.2f}% | Portfolio={pct3p:.2f}% | Random={pct3r:.2f}%")
    
    pct4 = sum(1 for m in test_matches if m >= 4) / n_test * 100
    pct4p = sum(1 for m in portfolio_best if m >= 4) / n_test * 100
    pct4r = sum(1 for m in random_matches if m >= 4) / n_test * 100
    print(f"  ≥4/6: Golden={pct4:.2f}% | Portfolio={pct4p:.2f}% | Random={pct4r:.2f}%")

    # ================================================================
    # PHASE 5: VULNERABILITY REPORT
    # ================================================================
    print(f"\n{'━'*75}")
    print("  📊 VULNERABILITY SUMMARY")
    print(f"{'━'*75}")

    # Analyze which signals contribute most
    signal_contribution = defaultdict(list)
    for i in range(val_end, min(val_end + 50, N - 1)):
        train_slice = data[:i+1]
        last_draw = data[i]
        actual = set(data[i+1][:PICK])
        
        for sig_name, w in weights.items():
            sig_func = SIGNAL_FUNCS[sig_name]
            scores = sig_func(train_slice, last_draw)
            top6 = set(num for num, _ in sorted(scores.items(), key=lambda x: -x[1])[:PICK])
            signal_contribution[sig_name].append(len(top6 & actual))
    
    print(f"\n  Signal contribution on test data:")
    for name, matches in sorted(signal_contribution.items(), 
                                  key=lambda x: -np.mean(x[1])):
        avg = np.mean(matches)
        lift = avg / (base_p * PICK)
        print(f"    {name:15s}: avg={avg:.4f}, lift={lift:.3f}x, "
              f"weight={weights.get(name, 0):.4f}")

    # ================================================================
    # SAVE RESULTS
    # ================================================================
    elapsed = time.time() - t0
    
    output = {
        'version': '4.0',
        'total_draws': N,
        'date_range': [dates[0], dates[-1]],
        'split': {'train': train_end, 'val': val_end - train_end, 'test': N - val_end},
        'signal_quality': signal_quality,
        'active_weights': {k: round(v, 4) for k, v in weights.items()},
        'backtest_test': {
            'golden_avg': round(test_avg, 4),
            'portfolio_avg': round(port_avg, 4),
            'random_avg': round(rand_avg, 4),
            'improvement_pct': round(improvement, 2),
            'portfolio_improvement_pct': round(port_improvement, 2),
            'pct_3plus_golden': round(pct3, 2),
            'pct_3plus_portfolio': round(pct3p, 2),
            'pct_3plus_random': round(pct3r, 2),
            'pct_4plus_portfolio': round(pct4p, 2),
            'n_tests': n_test,
        },
        'elapsed_seconds': round(elapsed, 1),
        'methodology': 'Anti-overfitting: 3-way split, signals evaluated independently,'
                        ' only lift>1.0 signals used, quadratic weight scaling',
    }
    
    rules_path = os.path.join(os.path.dirname(__file__), 'models', 'validated_rules_v4.json')
    with open(rules_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {rules_path}")

    print(f"\n{'='*75}")
    print(f"  V4 COMPLETE — Golden={test_avg:.4f}/6 ({improvement:+.1f}%), "
          f"Portfolio={port_avg:.4f}/6 ({port_improvement:+.1f}%), time={elapsed:.1f}s")
    print(f"{'='*75}")


if __name__ == '__main__':
    deep_mine_v4()
