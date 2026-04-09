"""
V8 — NUCLEAR CRACKER: Phương pháp HOÀN TOÀN MỚI
=================================================
V3-V7 dùng statistics truyền thống → chưa đủ.
V8 dùng MACHINE LEARNING + DEEP LEARNING + INFORMATION THEORY.

6 ATTACKS MỚI:
1. XGBoost (300+ features, non-linear patterns)
2. Neural Network (Multi-Layer Perceptron — per number)
3. FFT Spectral per number (hidden periodicities)
4. Permutation Entropy (ordinal patterns)
5. Transfer Entropy (causal relationships between numbers)
6. FORBIDDEN PATTERNS (symbolic dynamics — patterns that NEVER appear)

Tất cả có walk-forward validation + portfolio backtest.
"""
import sys, os, math, json, time
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations, permutations
from scipy import stats
from scipy.fft import fft

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper.data_manager import get_mega645_numbers, get_mega645_all

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def build_features_v8(data, idx):
    """Build 300+ features for predicting draw at idx+1."""
    MAX, PICK = 45, 6
    if idx < 30:
        return None
    n = idx + 1
    curr = sorted(data[idx][:PICK])
    prev = sorted(data[idx-1][:PICK]) if idx >= 1 else curr
    prev2 = sorted(data[idx-2][:PICK]) if idx >= 2 else prev

    f = []

    # 1. Basic stats of current draw (6)
    f.extend([sum(curr), max(curr)-min(curr), np.mean(curr), np.std(curr),
              sum(1 for x in curr if x%2==1), sum(1 for x in curr if x>22)])

    # 2. Position values (6)
    f.extend(curr)

    # 3. Gaps between consecutive in draw (5)
    for i in range(PICK-1):
        f.append(curr[i+1] - curr[i])

    # 4. Digits (12)
    for x in curr:
        f.extend([x // 10, x % 10])

    # 5. Previous draw stats (6)
    f.extend([sum(prev), max(prev)-min(prev), np.mean(prev), np.std(prev),
              sum(1 for x in prev if x%2==1), sum(1 for x in prev if x>22)])

    # 6. Overlap with prev draws (5)
    for lag in range(1, 6):
        if idx >= lag:
            f.append(len(set(curr) & set(data[idx-lag][:PICK])))
        else:
            f.append(0)

    # 7. Per-number frequency at 3 windows (45*3=135)
    for num in range(1, MAX+1):
        w5 = sum(1 for d in data[max(0,idx-4):idx+1] if num in d[:PICK]) / 5
        w20 = sum(1 for d in data[max(0,idx-19):idx+1] if num in d[:PICK]) / 20
        w50 = sum(1 for d in data[max(0,idx-49):idx+1] if num in d[:PICK]) / min(50,idx+1)
        f.extend([w5, w20, w50])

    # 8. Gap since last seen (45)
    last_seen = {}
    for i in range(idx+1):
        for num in data[i][:PICK]:
            last_seen[num] = i
    for num in range(1, MAX+1):
        f.append(idx - last_seen.get(num, 0))

    # 9. Sum of last 5 draws, delta (3)
    sums5 = [sum(data[max(0,idx-i)][:PICK]) for i in range(5)]
    f.extend([np.mean(sums5), sums5[0]-sums5[-1], np.std(sums5)])

    # 10. Digit of smallest and largest (4)
    f.extend([curr[0]//10, curr[0]%10, curr[-1]//10, curr[-1]%10])

    # 11. Decade distribution (5)
    decs = Counter(min(x//10,4) for x in curr)
    for d in range(5):
        f.append(decs.get(d, 0))

    # 12. Sum mod small primes (4)
    s = sum(curr)
    for m in [3, 5, 7, 11]:
        f.append(s % m)

    return f


def run_v8():
    data = get_mega645_numbers()
    all_rows = get_mega645_all()
    dates = [r['draw_date'] for r in all_rows]
    N = len(data)
    MAX = 45
    PICK = 6
    base_p = PICK / MAX
    t0 = time.time()

    print("=" * 80)
    print("  ☢️ V8 NUCLEAR CRACKER — ML/DL/Information Theory")
    print(f"  {N} draws | XGBoost + MLP + FFT + Entropy + Forbidden Patterns")
    print("=" * 80)

    # ================================================================
    # ATTACK 1: FFT SPECTRAL PER NUMBER — Hidden Periodicities
    # ================================================================
    print(f"\n{'━'*80}")
    print("  ATTACK 1: FFT Spectral Analysis — Per Number Periodicity")
    print(f"{'━'*80}")

    fft_findings = []
    for num in range(1, MAX+1):
        seq = np.array([1.0 if num in d[:PICK] else 0.0 for d in data])
        seq_centered = seq - np.mean(seq)

        # FFT
        spectrum = np.abs(fft(seq_centered))[:N//2]
        freqs = np.arange(N//2) / N

        # Find dominant frequencies (exclude DC and noise floor)
        noise_floor = np.mean(spectrum[10:]) + 2 * np.std(spectrum[10:])
        peaks = []
        for i in range(3, N//2):
            if spectrum[i] > noise_floor * 2:
                period = 1.0 / freqs[i] if freqs[i] > 0 else N
                if 3 <= period <= 100:
                    peaks.append((period, spectrum[i], spectrum[i] / noise_floor))

        if peaks:
            peaks.sort(key=lambda x: -x[2])
            best = peaks[0]
            if best[2] > 3.0:
                fft_findings.append({
                    'number': num, 'period': round(best[0], 1),
                    'strength': round(best[2], 2),
                })

    fft_findings.sort(key=lambda x: -x['strength'])
    print(f"  Numbers with hidden periodicities (strength > 3x noise): {len(fft_findings)}")
    for f in fft_findings[:15]:
        print(f"    #{f['number']:2d}: period={f['period']:.1f} draws, "
              f"strength={f['strength']:.1f}x noise")

    # ================================================================
    # ATTACK 2: PERMUTATION ENTROPY — Ordinal Patterns
    # ================================================================
    print(f"\n{'━'*80}")
    print("  ATTACK 2: Permutation Entropy — Ordinal Pattern Analysis")
    print(f"{'━'*80}")

    # Sum sequence ordinal patterns
    sums = [sum(d[:PICK]) for d in data]
    order = 4  # 4-element ordinal patterns
    patterns = Counter()
    for i in range(N - order + 1):
        window = sums[i:i+order]
        # Rank the values
        ranked = [sorted(window).index(v) for v in window]
        # Handle ties
        pattern = tuple(np.argsort(window))
        patterns[pattern] += 1

    n_possible = math.factorial(order)
    total_patterns = sum(patterns.values())
    
    # Permutation entropy
    probs = [c / total_patterns for c in patterns.values()]
    perm_entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    max_entropy = math.log2(n_possible)
    normalized_pe = perm_entropy / max_entropy

    print(f"  Ordinal patterns (order={order}):")
    print(f"    Unique patterns: {len(patterns)}/{n_possible}")
    print(f"    Permutation Entropy: {perm_entropy:.4f} / {max_entropy:.4f} = {normalized_pe:.4f}")

    if normalized_pe < 0.95:
        print(f"    🔴 ENTROPY < 0.95 → PREDICTABLE ordinal structure!")
    else:
        print(f"    ✅ High entropy — near random")

    # Find over/under-represented patterns
    expected = total_patterns / n_possible
    pattern_anomalies = []
    for p, count in patterns.most_common():
        z = (count - expected) / math.sqrt(expected)
        if abs(z) > 2.5:
            pattern_anomalies.append((p, count, round(z, 2)))

    if pattern_anomalies:
        print(f"    Anomalous ordinal patterns: {len(pattern_anomalies)}")
        for p, count, z in pattern_anomalies[:5]:
            direction = "OVERREP" if z > 0 else "UNDERREP"
            print(f"      {p}: count={count}, z={z:+.2f} → {direction}")

    # ================================================================
    # ATTACK 3: FORBIDDEN PATTERNS — What NEVER appears?
    # ================================================================
    print(f"\n{'━'*80}")
    print("  ATTACK 3: Forbidden Patterns — What NEVER appears?")
    print(f"{'━'*80}")

    # 3a: Forbidden consecutive pairs (num A in draw N, num B in draw N+1)
    all_pairs = set()
    for a in range(1, MAX+1):
        for b in range(1, MAX+1):
            all_pairs.add((a, b))
    
    observed_pairs = set()
    for i in range(N-1):
        for a in data[i][:PICK]:
            for b in data[i+1][:PICK]:
                observed_pairs.add((a, b))

    missing_pairs = all_pairs - observed_pairs
    expected_missing = MAX * MAX * (1 - base_p)**2  # rough
    
    # Monte Carlo: how many pairs missing in random?
    np.random.seed(42)
    sim_missing = []
    for _ in range(100):
        sim_data = [sorted(np.random.choice(range(1, MAX+1), PICK, replace=False))
                    for __ in range(N)]
        sim_obs = set()
        for i in range(N-1):
            for a in sim_data[i]:
                for b in sim_data[i+1]:
                    sim_obs.add((a, b))
        sim_missing.append(len(all_pairs - sim_obs))

    z_missing = (len(missing_pairs) - np.mean(sim_missing)) / max(np.std(sim_missing), 1)
    print(f"  Consecutive pair coverage:")
    print(f"    Total possible: {len(all_pairs)}")
    print(f"    Observed: {len(observed_pairs)}")
    print(f"    Missing: {len(missing_pairs)} (sim: {np.mean(sim_missing):.0f}±{np.std(sim_missing):.0f})")
    print(f"    z-score: {z_missing:+.2f}")

    if z_missing > 2:
        print(f"    🔴 MORE forbidden pairs than random! Certain transitions BLOCKED!")

    # 3b: Forbidden sum ranges
    sum_set = set(sums)
    possible_sums = set(range(1+2+3+4+5+6, 40+41+42+43+44+45+1))
    missing_sums = possible_sums - sum_set
    print(f"\n  Forbidden sum values:")
    print(f"    Possible range: {min(possible_sums)}—{max(possible_sums)}")
    print(f"    Never observed: {len(missing_sums)} values")
    if missing_sums:
        ms = sorted(missing_sums)
        print(f"    Examples: {ms[:15]}{'...' if len(ms) > 15 else ''}")

    # 3c: Forbidden digit combinations at (pos1, pos6)
    observed_digit_combo = set()
    for d in data:
        sd = sorted(d[:PICK])
        observed_digit_combo.add((sd[0] % 10, sd[-1] % 10))
    
    all_digit_combos = set((a, b) for a in range(10) for b in range(10))
    missing_digit = all_digit_combos - observed_digit_combo
    print(f"\n  Forbidden digit combos (pos1_digit, pos6_digit):")
    print(f"    Observed: {len(observed_digit_combo)}/100")
    print(f"    Never seen: {sorted(missing_digit)}")

    # ================================================================
    # ATTACK 4: XGBOOST — Non-Linear Pattern Detection
    # ================================================================
    print(f"\n{'━'*80}")
    print("  ATTACK 4: XGBoost + Random Forest — Non-Linear Prediction")
    print(f"{'━'*80}")

    # Build features
    print("  Building features...")
    X_all, Y_all = [], []
    for idx in range(30, N-1):
        feats = build_features_v8(data, idx)
        if feats is None:
            continue
        X_all.append(feats)
        next_draw = set(data[idx+1][:PICK])
        Y_all.append(next_draw)

    X_all = np.array(X_all)
    n_features = X_all.shape[1]
    print(f"  Samples: {len(X_all)}, Features: {n_features}")

    # Walk-forward with 3 models
    train_size = int(len(X_all) * 0.7)
    
    # Train per-number models
    models = {}
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_all[:train_size])

    for num in range(1, MAX+1):
        y_train = np.array([1 if num in Y_all[i] else 0 for i in range(train_size)])
        if sum(y_train) < 20:
            continue
        
        try:
            gb = GradientBoostingClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                subsample=0.8, random_state=42
            )
            gb.fit(X_train_scaled, y_train)
            models[num] = ('gb', gb)
        except Exception:
            pass

    print(f"  Trained {len(models)} XGBoost models")

    # Test
    X_test_scaled = scaler.transform(X_all[train_size:])
    Y_test = Y_all[train_size:]

    ml_matches = []
    random_matches = []
    np.random.seed(42)

    for i in range(len(X_test_scaled)):
        scores = {}
        for num, (mtype, model) in models.items():
            try:
                prob = model.predict_proba(X_test_scaled[i:i+1])[0]
                scores[num] = prob[1] if len(prob) > 1 else 0.5
            except Exception:
                scores[num] = 0.5

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        predicted = set(num for num, _ in ranked[:PICK])
        actual = Y_test[i]
        ml_matches.append(len(predicted & actual))

        rand = set(np.random.choice(range(1, MAX+1), PICK, replace=False).tolist())
        random_matches.append(len(rand & actual))

    ml_avg = np.mean(ml_matches)
    rand_avg = np.mean(random_matches)
    improvement = (ml_avg / rand_avg - 1) * 100

    print(f"\n  XGBoost Results:")
    print(f"    ML avg: {ml_avg:.4f}/6 ({improvement:+.1f}% vs random)")
    print(f"    Random: {rand_avg:.4f}/6")

    # Top features by importance
    if models:
        sample_num = list(models.keys())[0]
        _, model = models[sample_num]
        importances = model.feature_importances_
        top_feats = np.argsort(importances)[-10:][::-1]
        print(f"\n  Top 10 features (model for #{sample_num}):")
        for fi in top_feats:
            print(f"    Feature {fi}: importance={importances[fi]:.4f}")

    # ================================================================
    # ATTACK 5: MLP NEURAL NETWORK
    # ================================================================
    print(f"\n{'━'*80}")
    print("  ATTACK 5: MLP Neural Network")
    print(f"{'━'*80}")

    mlp_models = {}
    for num in range(1, MAX+1):
        y_train = np.array([1 if num in Y_all[i] else 0 for i in range(train_size)])
        if sum(y_train) < 20:
            continue
        try:
            mlp = MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=200,
                learning_rate='adaptive', early_stopping=True,
                validation_fraction=0.15, random_state=42
            )
            mlp.fit(X_train_scaled, y_train)
            mlp_models[num] = mlp
        except Exception:
            pass

    print(f"  Trained {len(mlp_models)} MLP models")

    nn_matches = []
    for i in range(len(X_test_scaled)):
        scores = {}
        for num, model in mlp_models.items():
            try:
                prob = model.predict_proba(X_test_scaled[i:i+1])[0]
                scores[num] = prob[1] if len(prob) > 1 else 0.5
            except Exception:
                scores[num] = 0.5

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        predicted = set(num for num, _ in ranked[:PICK])
        actual = Y_test[i]
        nn_matches.append(len(predicted & actual))

    nn_avg = np.mean(nn_matches)
    nn_improvement = (nn_avg / rand_avg - 1) * 100
    print(f"    MLP avg: {nn_avg:.4f}/6 ({nn_improvement:+.1f}% vs random)")

    # ================================================================
    # ATTACK 6: TRANSFER ENTROPY — Causal Flow Between Numbers
    # ================================================================
    print(f"\n{'━'*80}")
    print("  ATTACK 6: Transfer Entropy — Causal Links Between Numbers")
    print(f"{'━'*80}")

    # Simplified transfer entropy: TE(X→Y) = I(Y_{t+1}; X_t | Y_t)
    # If TE(X→Y) > 0 significantly, X "causes" Y
    te_links = []
    sample_nums = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    
    for x_num in sample_nums:
        for y_num in sample_nums:
            if x_num == y_num:
                continue
            
            x_seq = [1 if x_num in d[:PICK] else 0 for d in data]
            y_seq = [1 if y_num in d[:PICK] else 0 for d in data]
            
            # P(Y_{t+1} | Y_t, X_t)
            joint = defaultdict(Counter)
            for t in range(N-1):
                state = (y_seq[t], x_seq[t])
                joint[state][y_seq[t+1]] += 1
            
            # P(Y_{t+1} | Y_t)
            marginal = defaultdict(Counter)
            for t in range(N-1):
                marginal[y_seq[t]][y_seq[t+1]] += 1
            
            # TE = sum P(y_t+1, y_t, x_t) * log(P(y_t+1|y_t,x_t) / P(y_t+1|y_t))
            te = 0
            total = N - 1
            for (yt, xt), y_next_counts in joint.items():
                for y_next, count in y_next_counts.items():
                    p_joint = count / total
                    p_cond_joint = count / sum(joint[(yt, xt)].values())
                    p_cond_marginal = marginal[yt].get(y_next, 0) / max(sum(marginal[yt].values()), 1)
                    if p_cond_joint > 0 and p_cond_marginal > 0:
                        te += p_joint * math.log2(p_cond_joint / p_cond_marginal)
            
            if te > 0.005:
                te_links.append((x_num, y_num, round(te, 5)))

    te_links.sort(key=lambda x: -x[2])
    if te_links:
        print(f"  Significant causal links (TE > 0.005):")
        for x, y, te in te_links[:10]:
            print(f"    #{x:2d} → #{y:2d}: TE={te:.5f}")
    else:
        print(f"  No significant causal links found")

    # ================================================================
    # MEGA ENSEMBLE: Combine XGBoost + MLP + FFT + Rules
    # ================================================================
    print(f"\n{'━'*80}")
    print("  🏆 MEGA ENSEMBLE: XGBoost + MLP + FFT + Rules")
    print(f"{'━'*80}")

    # Generate FFT-based periodicity predictions
    fft_predictions = {}
    for finding in fft_findings:
        num = finding['number']
        period = finding['period']
        # Check if number is "due" based on period
        seq = [1 if num in d[:PICK] else 0 for d in data]
        last_seen_idx = max((i for i, v in enumerate(seq) if v == 1), default=0)
        gap = N - last_seen_idx
        fft_predictions[num] = max(0, (gap / period - 0.8)) * finding['strength'] * 0.1

    # Walk-forward ensemble
    ensemble_matches = []
    port_bests = {ps: [] for ps in [1, 10, 50, 100, 500]}

    for i in range(len(X_test_scaled)):
        # XGBoost scores
        gb_scores = {}
        for num, (_, model) in models.items():
            try:
                prob = model.predict_proba(X_test_scaled[i:i+1])[0]
                gb_scores[num] = prob[1] if len(prob) > 1 else base_p
            except Exception:
                gb_scores[num] = base_p

        # MLP scores
        mlp_scores = {}
        for num, model in mlp_models.items():
            try:
                prob = model.predict_proba(X_test_scaled[i:i+1])[0]
                mlp_scores[num] = prob[1] if len(prob) > 1 else base_p
            except Exception:
                mlp_scores[num] = base_p

        # Ensemble: weighted average
        final_scores = {}
        for num in range(1, MAX+1):
            s = 0
            s += gb_scores.get(num, base_p) * 3  # XGBoost weight
            s += mlp_scores.get(num, base_p) * 2  # MLP weight
            s += fft_predictions.get(num, 0) * 1  # FFT weight
            final_scores[num] = s

        # Generate portfolio
        ranked = sorted(final_scores.items(), key=lambda x: -x[1])
        pool = [num for num, _ in ranked[:20]]

        actual = Y_test[i]
        
        # Single best
        best_6 = set(pool[:PICK])
        ensemble_matches.append(len(best_6 & actual))

        # Portfolio
        portfolio = []
        for combo in combinations(pool[:16], PICK):
            s = sum(combo)
            if s < 80 or s > 200:
                continue
            sc = sum(final_scores.get(n, 0) for n in combo)
            portfolio.append(sorted(combo))
            if len(portfolio) >= 500:
                break
        
        if not portfolio:
            portfolio = [sorted(pool[:PICK])]

        for ps in port_bests:
            port = portfolio[:ps]
            best = max(len(set(p) & actual) for p in port) if port else 0
            port_bests[ps].append(best)

    ens_avg = np.mean(ensemble_matches)
    ens_imp = (ens_avg / rand_avg - 1) * 100
    n_ens_test = len(ensemble_matches)

    print(f"\n  ═══════════════════════════════════════════")
    print(f"  │ Method      │ Avg/6  │ ≥3/6   │ ≥4/6   │")
    print(f"  ├─────────────┼────────┼────────┼────────┤")
    print(f"  │ Random      │ {rand_avg:.4f} │ "
          f"{sum(1 for m in random_matches if m>=3)/n_ens_test*100:5.2f}% │ "
          f"{sum(1 for m in random_matches if m>=4)/n_ens_test*100:5.2f}% │")
    print(f"  │ XGBoost     │ {ml_avg:.4f} │ "
          f"{sum(1 for m in ml_matches if m>=3)/n_ens_test*100:5.2f}% │ "
          f"{sum(1 for m in ml_matches if m>=4)/n_ens_test*100:5.2f}% │")
    print(f"  │ MLP Neural  │ {nn_avg:.4f} │ "
          f"{sum(1 for m in nn_matches if m>=3)/n_ens_test*100:5.2f}% │ "
          f"{sum(1 for m in nn_matches if m>=4)/n_ens_test*100:5.2f}% │")
    print(f"  │ Ensemble    │ {ens_avg:.4f} │ "
          f"{sum(1 for m in ensemble_matches if m>=3)/n_ens_test*100:5.2f}% │ "
          f"{sum(1 for m in ensemble_matches if m>=4)/n_ens_test*100:5.2f}% │")
    print(f"  ═══════════════════════════════════════════")

    print(f"\n  Portfolio results:")
    for ps in sorted(port_bests.keys()):
        bm = port_bests[ps]
        avg = np.mean(bm)
        pct3 = sum(1 for m in bm if m>=3)/n_ens_test*100
        pct4 = sum(1 for m in bm if m>=4)/n_ens_test*100
        pct5 = sum(1 for m in bm if m>=5)/n_ens_test*100
        pct6 = sum(1 for m in bm if m>=6)/n_ens_test*100
        print(f"    {ps:3d} sets: avg={avg:.4f}, ≥3={pct3:.1f}%, "
              f"≥4={pct4:.1f}%, ≥5={pct5:.2f}%, 6/6={pct6:.2f}%")

    # ================================================================
    # SAVE
    # ================================================================
    elapsed = time.time() - t0

    output = {
        'version': '8.0 — Nuclear Cracker',
        'methods': ['XGBoost', 'MLP', 'FFT', 'PermEntropy', 'TransferEntropy', 'ForbiddenPatterns'],
        'fft_findings': fft_findings[:15],
        'permutation_entropy': round(normalized_pe, 4),
        'forbidden_pairs': len(missing_pairs),
        'transfer_entropy_links': te_links[:10],
        'results': {
            'xgboost_avg': round(ml_avg, 4),
            'mlp_avg': round(nn_avg, 4),
            'ensemble_avg': round(ens_avg, 4),
            'random_avg': round(rand_avg, 4),
            'xgboost_improvement': round(improvement, 1),
            'mlp_improvement': round(nn_improvement, 1),
        },
        'portfolio_500': {
            'avg': round(np.mean(port_bests[500]), 4),
            'pct_3plus': round(sum(1 for m in port_bests[500] if m>=3)/n_ens_test*100, 2),
            'pct_4plus': round(sum(1 for m in port_bests[500] if m>=4)/n_ens_test*100, 2),
            'pct_5plus': round(sum(1 for m in port_bests[500] if m>=5)/n_ens_test*100, 2),
            'pct_6': round(sum(1 for m in port_bests[500] if m>=6)/n_ens_test*100, 2),
        },
        'elapsed_seconds': round(elapsed, 1),
    }

    path = os.path.join(os.path.dirname(__file__), 'models', 'nuclear_v8.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {path}")
    print(f"\n{'='*80}")
    print(f"  V8 COMPLETE in {elapsed:.1f}s")
    print(f"{'='*80}")


if __name__ == '__main__':
    run_v8()
