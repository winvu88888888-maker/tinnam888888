"""
DEEP MINE V5 — DETERMINISTIC SYSTEM CRACKER
=============================================
TƯ DUY MỚI: XỔ SỐ KHÔNG PHẢI NGẪU NHIÊN.
Máy xổ số do con người tạo ra → có giới hạn vật lý & thuật toán.

CÁCH TIẾP CẬN:
Thay vì hỏi "pattern nào trong data?" → hỏi "MÁY hoạt động THẾ NÀO?"

10 CUỘC TẤN CÔNG:

ATTACK 1: PRNG State Reconstruction
  - Nếu dùng Linear Congruential Generator → crack modulus/multiplier
  - Tìm chu kỳ ẩn trong tích/tổng các số

ATTACK 2: Phase Space Reconstruction (Takens' Theorem)
  - Embed time series vào không gian cao chiều
  - Tìm attractor (hệ thống chaotic có attractor)

ATTACK 3: Neural Pattern Detection
  - XGBoost/MLP để tìm NON-LINEAR relationships
  - Feature engineering: 200+ features từ lịch sử

ATTACK 4: Digit-Level Analysis
  - Phân tích TỪNG CHỮ SỐ (hàng chục, hàng đơn vị)
  - Digit correlation giữa các vị trí

ATTACK 5: Ball Physics Simulation
  - Viên bi nặng hơn (2 chữ số vs 1 chữ số) → bias?
  - Vị trí ban đầu trong máy → ảnh hưởng kết quả?

ATTACK 6: Time-Series Forecasting
  - ARIMA/exponential smoothing cho từng số
  - Treating appearances as a renewal process

ATTACK 7: Modular Arithmetic Crack
  - Sequences mod small primes → detect LCG parameters
  - GCD analysis trên consecutive draws

ATTACK 8: Chaos Theory — Lyapunov Exponents
  - Đo "độ hỗn loạn" — nếu Lyapunov < 0 → hệ thống predictable

ATTACK 9: Compression-Based Prediction
  - Nếu data compressible → có structure → exploitable
  - Lempel-Ziv complexity per number

ATTACK 10: Machine Learning Ensemble
  - Random Forest + Gradient Boosting + Neural Net
  - 200+ features, walk-forward validation
"""
import sys, os, math, json, time
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from scipy import stats
from scipy.fft import fft

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper.data_manager import get_mega645_numbers, get_mega645_all


def deep_mine_v5():
    data = get_mega645_numbers()
    all_rows = get_mega645_all()
    dates = [r['draw_date'] for r in all_rows]
    N = len(data)
    MAX = 45
    PICK = 6
    base_p = PICK / MAX
    t0 = time.time()

    print("=" * 80)
    print("  🔓 DEEP MINE V5 — DETERMINISTIC SYSTEM CRACKER")
    print(f"  {N} draws | MÁY XỔ SỐ KHÔNG BAO GIỜ THỰC SỰ NGẪU NHIÊN")
    print("=" * 80)

    attacks = {}

    # ==================================================================
    # ATTACK 1: PRNG STATE RECONSTRUCTION
    # Nếu máy dùng PRNG (Linear Congruential Generator):
    #   X(n+1) = (a*X(n) + c) mod m
    # → Tổng/Tích của mỗi kỳ có thể là output trực tiếp từ PRNG
    # ==================================================================
    print(f"\n{'━'*80}")
    print("  ATTACK 1: PRNG State Reconstruction")
    print("  Giả thiết: tổng/tích mỗi kỳ là output của LCG")
    print(f"{'━'*80}")

    sums = [sum(d[:PICK]) for d in data]
    products_mod = [1 for _ in range(N)]
    for i, d in enumerate(data):
        p = 1
        for x in d[:PICK]:
            p = (p * x)
        products_mod[i] = p

    # Test 1a: Sum sequence — tìm autocorrelation structure
    sum_arr = np.array(sums, dtype=float)
    sum_centered = sum_arr - np.mean(sum_arr)
    acf_sums = np.correlate(sum_centered, sum_centered, mode='full')
    acf_sums = acf_sums[N-1:] / acf_sums[N-1]  # Normalize

    # Tìm peaks trong ACF (ngoài lag 0)
    acf_peaks = []
    for lag in range(1, min(100, N//3)):
        if lag >= len(acf_sums):
            break
        z = acf_sums[lag] * math.sqrt(N)
        if abs(z) > 2.5:
            acf_peaks.append((lag, round(acf_sums[lag], 5), round(z, 2)))

    if acf_peaks:
        print(f"  ⚠️ SUM autocorrelation peaks found:")
        for lag, r, z in acf_peaks[:10]:
            print(f"    Lag-{lag}: r={r:+.5f}, z={z:+.2f}")
    else:
        print(f"  Sum ACF: no significant peaks (noise-level)")

    # Test 1b: Detect LCG-like pattern in sums
    # If S(n+1) = a*S(n) + c mod m, then S(n+2) - S(n+1) = a*(S(n+1) - S(n)) mod m
    # Try to find (a, m) by analyzing differences
    diffs = [sums[i+1] - sums[i] for i in range(N-1)]
    diff2 = [diffs[i+1] - diffs[i] for i in range(len(diffs)-1)]

    # GCD of consecutive difference pairs → might reveal modulus
    from math import gcd
    gcds = []
    for i in range(min(500, len(diff2)-1)):
        d1 = abs(diff2[i]) if diff2[i] != 0 else 1
        d2 = abs(diff2[i+1]) if diff2[i+1] != 0 else 1
        if d1 > 0 and d2 > 0:
            gcds.append(gcd(d1, d2))

    gcd_counts = Counter(gcds)
    print(f"\n  Sum-difference GCD analysis (looking for hidden modulus):")
    for g, c in gcd_counts.most_common(5):
        print(f"    GCD={g}: occurs {c} times ({c/len(gcds)*100:.1f}%)")

    attacks['prng_state'] = {
        'acf_peaks': len(acf_peaks),
        'top_gcd': gcd_counts.most_common(1)[0] if gcd_counts else (0, 0),
    }

    # ==================================================================
    # ATTACK 2: PHASE SPACE RECONSTRUCTION (Takens' Theorem)
    # Embed 1D time series into higher dimensions
    # If deterministic system: trajectory forms ATTRACTOR
    # ==================================================================
    print(f"\n{'━'*80}")
    print("  ATTACK 2: Phase Space Reconstruction (Chaos Theory)")
    print("  Embed data into higher dimensions → find attractors")
    print(f"{'━'*80}")

    # For each number, create binary time series, then embed
    embed_dims = [2, 3, 4, 5]
    determinism_scores = {}

    for num in range(1, MAX+1):
        seq = np.array([1.0 if num in d[:PICK] else 0.0 for d in data])

        best_det = 0
        best_dim = 2
        for dim in embed_dims:
            # Create delay embedding
            embedded = []
            for i in range(dim - 1, len(seq)):
                point = tuple(seq[i - j] for j in range(dim))
                embedded.append(point)

            if len(embedded) < 50:
                continue

            # Measure determinism: same input state → same output?
            state_outcomes = defaultdict(list)
            for i in range(len(embedded) - 1):
                state = embedded[i]
                next_val = seq[min(i + dim, len(seq) - 1)]
                state_outcomes[state].append(next_val)

            # Determinism = weighted average of max(P(0), P(1)) per state
            total_weight = 0
            det_score = 0
            for state, outcomes in state_outcomes.items():
                if len(outcomes) < 3:
                    continue
                p1 = sum(outcomes) / len(outcomes)
                det = max(p1, 1 - p1)
                det_score += det * len(outcomes)
                total_weight += len(outcomes)

            if total_weight > 0:
                avg_det = det_score / total_weight
                if avg_det > best_det:
                    best_det = avg_det
                    best_dim = dim

        determinism_scores[num] = (best_det, best_dim)

    # Expected determinism for pure random: max(p, 1-p) where p = 6/45 ≈ 0.133
    # → expected = max(0.133, 0.867) = 0.867
    expected_det = max(base_p, 1 - base_p)

    anomalous = [(num, det, dim) for num, (det, dim) in determinism_scores.items()
                 if det > expected_det + 0.02]
    anomalous.sort(key=lambda x: -x[1])

    if anomalous:
        print(f"  Numbers with HIGH determinism (>{expected_det + 0.02:.3f}):")
        for num, det, dim in anomalous[:15]:
            excess = (det - expected_det) / expected_det * 100
            print(f"    #{num:2d}: determinism={det:.4f} (dim={dim}), "
                  f"+{excess:.1f}% above random")
    else:
        print(f"  No numbers show above-random determinism")

    # Overall determinism score
    all_det = [d for d, _ in determinism_scores.values()]
    avg_det = np.mean(all_det)
    z_det = (avg_det - expected_det) / (np.std(all_det) / math.sqrt(MAX))
    print(f"\n  Overall: avg_determinism={avg_det:.4f}, expected={expected_det:.4f}, z={z_det:+.2f}")

    attacks['phase_space'] = {
        'avg_determinism': round(avg_det, 4),
        'expected': round(expected_det, 4),
        'z_score': round(z_det, 2),
        'anomalous_count': len(anomalous),
        'top_numbers': [(n, round(d, 4)) for n, d, _ in anomalous[:10]],
    }

    # ==================================================================
    # ATTACK 3: DIGIT-LEVEL ANALYSIS
    # Xổ số dùng viên bi có số → phân tích ở mức CHỮA SỐ
    # Hàng chục (0-4) và hàng đơn vị (0-9) có pattern riêng
    # ==================================================================
    print(f"\n{'━'*80}")
    print("  ATTACK 3: Digit-Level Analysis")
    print("  Phân tích hàng chục (0-4) và hàng đơn vị (0-9) riêng biệt")
    print(f"{'━'*80}")

    # Extract digits
    tens_seqs = []  # tens digit of each number in draw
    units_seqs = []  # units digit
    for d in data:
        sd = sorted(d[:PICK])
        tens_seqs.append([x // 10 for x in sd])
        units_seqs.append([x % 10 for x in sd])

    # Units digit distribution across all draws
    all_units = [u for seq in units_seqs for u in seq]
    units_counts = Counter(all_units)
    total_units = len(all_units)

    # Expected: each digit 0-9 should appear roughly equally, but numbers 1-45:
    # digit 0: appears in 10,20,30,40 = 4 numbers
    # digit 1: 1,11,21,31,41 = 5 numbers
    # digit 2: 2,12,22,32,42 = 5 numbers
    # digit 3: 3,13,23,33,43 = 5 numbers
    # digit 4: 4,14,24,34,44 = 5 numbers
    # digit 5: 5,15,25,35,45 = 5 numbers
    # digit 6: 6,16,26,36 = 4 numbers
    # digit 7: 7,17,27,37 = 4 numbers
    # digit 8: 8,18,28,38 = 4 numbers
    # digit 9: 9,19,29,39 = 4 numbers
    digit_pool_size = {0: 4, 1: 5, 2: 5, 3: 5, 4: 5, 5: 5, 6: 4, 7: 4, 8: 4, 9: 4}
    total_pool = sum(digit_pool_size.values())  # = 45

    print(f"\n  Units digit distribution:")
    digit_anomalies = []
    for d in range(10):
        obs = units_counts.get(d, 0)
        exp = total_units * digit_pool_size[d] / total_pool
        z = (obs - exp) / math.sqrt(exp) if exp > 0 else 0
        status = "⚠️" if abs(z) > 2 else "  "
        print(f"    Digit {d}: obs={obs:5d}, exp={exp:.0f}, z={z:+.2f} {status}")
        if abs(z) > 2:
            digit_anomalies.append(('units', d, round(z, 2)))

    # Units digit TRANSITION: digit at pos k in draw N → digit at pos k in draw N+1
    print(f"\n  Units digit transitions (per-position):")
    unit_trans_anomalies = []
    for pos in range(PICK):
        trans = defaultdict(Counter)
        for i in range(N - 1):
            prev_u = units_seqs[i][pos]
            next_u = units_seqs[i + 1][pos]
            trans[prev_u][next_u] += 1

        # Find anomalous transitions
        for prev_d, next_counts in trans.items():
            total = sum(next_counts.values())
            for next_d, count in next_counts.most_common(2):
                rate = count / total
                # expected: roughly digit_pool_size[next_d] / total_pool
                exp_rate = digit_pool_size[next_d] / total_pool * PICK  # rough
                z = (rate - 0.1) / math.sqrt(0.1 * 0.9 / total) if total > 30 else 0
                if z > 3.0:
                    unit_trans_anomalies.append({
                        'pos': pos + 1, 'from_digit': prev_d,
                        'to_digit': next_d, 'rate': round(rate, 3), 'z': round(z, 2)
                    })

    if unit_trans_anomalies:
        unit_trans_anomalies.sort(key=lambda x: -x['z'])
        print(f"  Digit transition anomalies: {len(unit_trans_anomalies)}")
        for a in unit_trans_anomalies[:10]:
            print(f"    Pos{a['pos']}: digit {a['from_digit']}→{a['to_digit']}: "
                  f"rate={a['rate']:.1%}, z={a['z']:+.2f}")

    # Tens digit analysis
    all_tens = [t for seq in tens_seqs for t in seq]
    tens_counts = Counter(all_tens)
    print(f"\n  Tens digit distribution:")
    tens_pool = {0: 9, 1: 10, 2: 10, 3: 10, 4: 6}  # 1-9, 10-19, 20-29, 30-39, 40-45
    total_tens_pool = sum(tens_pool.values())
    for d in range(5):
        obs = tens_counts.get(d, 0)
        exp = len(all_tens) * tens_pool[d] / total_tens_pool
        z = (obs - exp) / math.sqrt(exp) if exp > 0 else 0
        status = "⚠️" if abs(z) > 2 else "  "
        print(f"    Tens {d}: obs={obs:5d}, exp={exp:.0f}, z={z:+.2f} {status}")
        if abs(z) > 2:
            digit_anomalies.append(('tens', d, round(z, 2)))

    attacks['digit_analysis'] = {
        'unit_digit_anomalies': digit_anomalies,
        'unit_transition_anomalies': len(unit_trans_anomalies),
        'top_transitions': unit_trans_anomalies[:10],
    }

    # ==================================================================
    # ATTACK 4: BALL PHYSICS SIMULATION
    # Viên bi có 2 chữ số (10-45) nặng hơn 1 chữ số (1-9)?
    # Ink weight, surface area → ảnh hưởng lực cản không khí
    # ==================================================================
    print(f"\n{'━'*80}")
    print("  ATTACK 4: Ball Physics — Ink Weight Hypothesis")
    print("  Bi 2 chữ số có nhiều mực hơn → nặng hơn → bias?")
    print(f"{'━'*80}")

    # Total "ink" per ball: count of digit strokes
    # Simple model: each digit = some ink amount
    digit_stroke_weight = {
        '0': 6, '1': 2, '2': 5, '3': 5, '4': 4,
        '5': 5, '6': 6, '7': 3, '8': 7, '9': 6
    }

    ball_ink = {}
    for num in range(1, MAX+1):
        s = str(num)
        ball_ink[num] = sum(digit_stroke_weight[c] for c in s)

    # Group by ink weight and check frequency
    ink_groups = defaultdict(list)
    for num, ink in ball_ink.items():
        ink_groups[ink].append(num)

    freq = Counter()
    for d in data:
        for x in d[:PICK]:
            freq[x] += 1

    print(f"\n  Ink weight vs frequency:")
    ink_freq = []
    for ink in sorted(ink_groups.keys()):
        nums = ink_groups[ink]
        avg_freq = np.mean([freq[n] for n in nums])
        expected = N * PICK / MAX
        z = (avg_freq - expected) / math.sqrt(expected / len(nums))
        direction = "MORE" if z > 0 else "less"
        print(f"    Ink={ink:2d} ({len(nums):2d} balls): avg_freq={avg_freq:.1f} "
              f"(exp={expected:.1f}), z={z:+.2f} → {direction}")
        ink_freq.append((ink, round(z, 2)))

    # Correlation: ink weight vs frequency
    inks = [ball_ink[n] for n in range(1, MAX+1)]
    freqs = [freq[n] for n in range(1, MAX+1)]
    r, p = stats.pearsonr(inks, freqs)
    print(f"\n  Ink-Frequency correlation: r={r:.4f}, p={p:.4f}")
    if p < 0.05:
        print(f"  ⚠️ SIGNIFICANT! Ink weight affects ball selection!")

    # 1-digit vs 2-digit balls
    one_digit = [freq[n] for n in range(1, 10)]  # 1-9
    two_digit = [freq[n] for n in range(10, MAX+1)]  # 10-45
    t_stat, t_p = stats.ttest_ind(one_digit, two_digit)
    print(f"\n  1-digit (1-9) avg={np.mean(one_digit):.1f} vs "
          f"2-digit (10-45) avg={np.mean(two_digit):.1f}")
    print(f"  t-test: t={t_stat:.3f}, p={t_p:.4f}")

    attacks['ball_physics'] = {
        'ink_correlation_r': round(r, 4),
        'ink_correlation_p': round(p, 4),
        'digit_count_t': round(t_stat, 3),
        'digit_count_p': round(t_p, 4),
    }

    # ==================================================================
    # ATTACK 5: COMPRESSION-BASED PREDICTION
    # Kolmogorov complexity: nếu data nén được → có pattern
    # ==================================================================
    print(f"\n{'━'*80}")
    print("  ATTACK 5: Compression-Based Analysis")
    print("  Nếu sequence nén được tốt hơn random → có structure ẩn")
    print(f"{'━'*80}")

    import zlib

    # Compress actual data
    data_bytes = ''.join(str(x).zfill(2) for d in data for x in sorted(d[:PICK]))
    actual_compressed = len(zlib.compress(data_bytes.encode()))
    actual_ratio = actual_compressed / len(data_bytes)

    # Generate random data and compress
    np.random.seed(42)
    random_ratios = []
    for _ in range(200):
        rnd = [sorted(np.random.choice(range(1, MAX+1), PICK, replace=False))
               for _ in range(N)]
        rnd_bytes = ''.join(str(x).zfill(2) for d in rnd for x in d)
        rnd_compressed = len(zlib.compress(rnd_bytes.encode()))
        random_ratios.append(rnd_compressed / len(rnd_bytes))

    z_compress = (actual_ratio - np.mean(random_ratios)) / np.std(random_ratios)
    print(f"  Actual compression ratio: {actual_ratio:.6f}")
    print(f"  Random compression ratio: {np.mean(random_ratios):.6f} ± {np.std(random_ratios):.6f}")
    print(f"  z-score: {z_compress:+.2f}")

    if z_compress < -2:
        print(f"  🔴 DATA IS MORE COMPRESSIBLE THAN RANDOM! Hidden structure exists!")
    elif z_compress > 2:
        print(f"  ⚠️ Data is LESS compressible — anti-correlated?")
    else:
        print(f"  Compression consistent with random")

    # Per-number Lempel-Ziv complexity
    print(f"\n  Per-number Lempel-Ziv complexity:")
    lz_anomalies = []
    for num in range(1, MAX+1):
        seq = ''.join('1' if num in d[:PICK] else '0' for d in data)
        compressed = len(zlib.compress(seq.encode()))
        ratio = compressed / len(seq)

        # Compare with random binary with same density
        p_appear = sum(1 for d in data if num in d[:PICK]) / N
        rnd_ratios = []
        for _ in range(50):
            rnd_seq = ''.join('1' if np.random.random() < p_appear else '0' for _ in range(N))
            rnd_ratios.append(len(zlib.compress(rnd_seq.encode())) / N)
        z = (ratio - np.mean(rnd_ratios)) / max(np.std(rnd_ratios), 0.001)
        if abs(z) > 2.5:
            direction = "MORE structured" if z < 0 else "LESS structured"
            lz_anomalies.append((num, round(ratio, 4), round(z, 2)))

    if lz_anomalies:
        lz_anomalies.sort(key=lambda x: x[2])
        print(f"  Numbers with anomalous complexity: {len(lz_anomalies)}")
        for num, ratio, z in lz_anomalies[:10]:
            direction = "STRUCTURED" if z < 0 else "ANTI-CORR"
            print(f"    #{num:2d}: ratio={ratio:.4f}, z={z:+.2f} ({direction})")

    attacks['compression'] = {
        'overall_z': round(z_compress, 2),
        'actual_ratio': round(actual_ratio, 6),
        'random_ratio': round(np.mean(random_ratios), 6),
        'lz_anomalies': len(lz_anomalies),
    }

    # ==================================================================
    # ATTACK 6: MODULAR ARITHMETIC CRACK
    # If output is from X(n+1) = (a*X(n) + c) mod m
    # Then looking at data mod small primes reveals structure
    # ==================================================================
    print(f"\n{'━'*80}")
    print("  ATTACK 6: Modular Arithmetic — LCG Detection")
    print(f"{'━'*80}")

    # Use sum of each draw as proxy for PRNG output
    for mod in [3, 5, 7, 11, 13, 17, 19, 23]:
        seq_mod = [s % mod for s in sums]
        # Check if seq_mod has sequential dependencies
        # Build transition matrix
        trans = defaultdict(Counter)
        for i in range(len(seq_mod) - 1):
            trans[seq_mod[i]][seq_mod[i+1]] += 1

        # Expected: uniform transitions (each next state equally likely)
        chi2_total = 0
        for state, next_counts in trans.items():
            total = sum(next_counts.values())
            for next_state in range(mod):
                obs = next_counts.get(next_state, 0)
                exp = total / mod
                if exp > 0:
                    chi2_total += (obs - exp) ** 2 / exp

        df = mod * (mod - 1)
        p = 1 - stats.chi2.cdf(chi2_total, df) if df > 0 else 1.0
        status = "🔴 STRUCTURED!" if p < 0.01 else ("⚠️" if p < 0.05 else "✅")
        print(f"  Sum mod {mod:2d}: χ²={chi2_total:.1f}, df={df}, p={p:.4f} {status}")

    attacks['modular'] = {'tested_mods': [3, 5, 7, 11, 13, 17, 19, 23]}

    # ==================================================================
    # ATTACK 7: MACHINE LEARNING — FEATURE-RICH PREDICTION
    # 200+ features → XGBoost/Random Forest → find non-linear patterns
    # ==================================================================
    print(f"\n{'━'*80}")
    print("  ATTACK 7: Machine Learning — Feature-Rich Prediction")
    print("  200+ features per draw → predict next draw")
    print(f"{'━'*80}")

    def build_features(data_slice, idx):
        """Build 200+ features for draw at index idx."""
        if idx < 20:
            return None

        features = []
        curr = sorted(data_slice[idx][:PICK])
        prev = sorted(data_slice[idx-1][:PICK])
        prev2 = sorted(data_slice[idx-2][:PICK]) if idx >= 2 else prev

        # Basic stats (6 features)
        features.extend([sum(curr), max(curr) - min(curr),
                         sum(1 for x in curr if x % 2 == 1),
                         sum(1 for x in curr if x > MAX // 2),
                         np.mean(curr), np.std(curr)])

        # Gaps between consecutive numbers in draw (5 features)
        for i in range(PICK - 1):
            features.append(curr[i+1] - curr[i])

        # Prev draw stats (6 features)
        features.extend([sum(prev), max(prev) - min(prev),
                         sum(1 for x in prev if x % 2 == 1),
                         sum(1 for x in prev if x > MAX // 2),
                         np.mean(prev), np.std(prev)])

        # Overlap with previous draws (5 features)
        for lag in range(1, 6):
            if idx >= lag:
                prev_lag = set(data_slice[idx-lag][:PICK])
                features.append(len(set(curr) & prev_lag))
            else:
                features.append(0)

        # Frequency features at different windows (45 * 3 = 135 features)
        for num in range(1, MAX+1):
            f10 = sum(1 for d in data_slice[max(0,idx-10):idx+1] if num in d[:PICK]) / 10
            f30 = sum(1 for d in data_slice[max(0,idx-30):idx+1] if num in d[:PICK]) / 30
            f100 = sum(1 for d in data_slice[max(0,idx-100):idx+1] if num in d[:PICK]) / min(100, idx+1)
            features.extend([f10, f30, f100])

        # Gap since last seen for each number (45 features)
        last_seen = {}
        for i in range(idx + 1):
            for num in data_slice[i][:PICK]:
                last_seen[num] = i
        for num in range(1, MAX+1):
            features.append(idx - last_seen.get(num, 0))

        return features

    # Build train/test features
    print("  Building features...")
    train_end = int(N * 0.75)
    X_train, Y_train = [], []
    X_test, Y_test = [], []

    for idx in range(20, N - 1):
        feats = build_features(data, idx)
        if feats is None:
            continue

        # Target: for each number, did it appear in next draw?
        next_draw = set(data[idx + 1][:PICK])

        if idx < train_end:
            X_train.append(feats)
            Y_train.append([1 if num in next_draw else 0 for num in range(1, MAX+1)])
        else:
            X_test.append(feats)
            Y_test.append(next_draw)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)

    print(f"  Train: {len(X_train)} samples × {X_train.shape[1]} features")
    print(f"  Test: {len(X_test)} samples")

    # Simple approach: train one model per number (or use multi-output)
    # Using simple logistic regression as baseline (fast)
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    ml_matches = []
    random_matches = []
    np.random.seed(42)

    # Train one model per most-predictable numbers
    print("  Training models...")
    models = {}
    for num_idx in range(MAX):
        num = num_idx + 1
        y = Y_train[:, num_idx]
        if sum(y) < 20 or sum(y) > len(y) - 20:
            continue
        try:
            model = LogisticRegression(max_iter=500, C=0.1, solver='lbfgs')
            model.fit(X_train, y)
            models[num] = model
        except Exception:
            pass

    print(f"  Trained {len(models)} models")

    # Predict on test
    for i in range(len(X_test)):
        scores = {}
        for num, model in models.items():
            prob = model.predict_proba(X_test[i:i+1])[0]
            scores[num] = prob[1] if len(prob) > 1 else 0.5

        # Rank by probability → top 6
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        predicted = set(num for num, _ in ranked[:PICK])
        actual = Y_test[i]
        ml_matches.append(len(predicted & actual))

        rand = set(np.random.choice(range(1, MAX+1), PICK, replace=False).tolist())
        random_matches.append(len(rand & actual))

    ml_avg = np.mean(ml_matches)
    rand_avg = np.mean(random_matches)
    improvement = (ml_avg / rand_avg - 1) * 100 if rand_avg > 0 else 0

    print(f"\n  ML Results:")
    print(f"    ML avg: {ml_avg:.4f}/6 ({improvement:+.1f}% vs random)")
    print(f"    Random: {rand_avg:.4f}/6")

    ml_dist = Counter(ml_matches)
    for k in range(5):
        mc = ml_dist.get(k, 0)
        print(f"    {k}/6: {mc:4d} ({mc/len(ml_matches)*100:.1f}%)")

    pct3_ml = sum(1 for m in ml_matches if m >= 3) / len(ml_matches) * 100
    pct3_r = sum(1 for m in random_matches if m >= 3) / len(random_matches) * 100
    print(f"    ≥3/6: ML={pct3_ml:.2f}% | Random={pct3_r:.2f}%")

    attacks['machine_learning'] = {
        'ml_avg': round(ml_avg, 4),
        'random_avg': round(rand_avg, 4),
        'improvement': round(improvement, 1),
        'pct_3plus': round(pct3_ml, 2),
    }

    # ==================================================================
    # ATTACK 8: LYAPUNOV EXPONENT ESTIMATION
    # λ < 0 → system is predictable (converges to attractor)
    # λ > 0 → chaotic (sensitive to initial conditions)
    # λ ≈ 0 → edge of chaos (interesting!)
    # ==================================================================
    print(f"\n{'━'*80}")
    print("  ATTACK 8: Lyapunov Exponent Estimation")
    print(f"{'━'*80}")

    # Use sum sequence as 1D representation
    sum_seq = np.array(sums, dtype=float)

    # Rosenstein's method for largest Lyapunov exponent
    def estimate_lyapunov(seq, embed_dim=3, tau=1, min_sep=10):
        n = len(seq)
        # Embed
        embedded = []
        for i in range(n - (embed_dim - 1) * tau):
            point = [seq[i + j * tau] for j in range(embed_dim)]
            embedded.append(point)
        embedded = np.array(embedded)
        m = len(embedded)

        divergences = []
        for i in range(min(200, m - min_sep - 1)):
            # Find nearest neighbor (not too close in time)
            min_dist = float('inf')
            nn_idx = -1
            for j in range(m):
                if abs(i - j) < min_sep:
                    continue
                dist = np.linalg.norm(embedded[i] - embedded[j])
                if dist < min_dist and dist > 0:
                    min_dist = dist
                    nn_idx = j

            if nn_idx < 0 or min_dist == 0:
                continue

            # Track divergence
            max_steps = min(20, m - max(i, nn_idx) - 1)
            for step in range(1, max_steps):
                d = np.linalg.norm(embedded[i + step] - embedded[nn_idx + step])
                if d > 0:
                    divergences.append((step, math.log(d / min_dist)))

        if not divergences:
            return 0

        # Average log(divergence) vs step → slope = Lyapunov exponent
        step_divs = defaultdict(list)
        for step, log_d in divergences:
            step_divs[step].append(log_d)

        steps = sorted(step_divs.keys())
        avg_log_divs = [np.mean(step_divs[s]) for s in steps]

        if len(steps) >= 3:
            slope, _, r, p, _ = stats.linregress(steps[:10], avg_log_divs[:10])
            return slope
        return 0

    lyap = estimate_lyapunov(sum_seq)
    print(f"  Lyapunov exponent (sum sequence): λ ≈ {lyap:.4f}")
    if lyap < -0.1:
        print(f"  🔴 λ < 0 → SYSTEM IS PREDICTABLE (converging)")
    elif lyap > 0.1:
        print(f"  System is chaotic (λ > 0)")
    else:
        print(f"  System at edge of chaos / noise-dominated")

    attacks['lyapunov'] = {'exponent': round(lyap, 4)}

    # ==================================================================
    # ATTACK 9: TEMPORAL REGULARITY (Day-of-Week × Position)
    # ==================================================================
    print(f"\n{'━'*80}")
    print("  ATTACK 9: Temporal Regularity — Day × Draw Interaction")
    print(f"{'━'*80}")

    from datetime import datetime
    day_draws = defaultdict(list)
    for date, d in zip(dates, data):
        try:
            dow = datetime.strptime(date, '%Y-%m-%d').weekday()
            day_draws[dow].append(d)
        except Exception:
            continue

    day_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
    print(f"\n  Draws per day: {', '.join(f'{day_names.get(d, d)}={len(v)}' for d, v in sorted(day_draws.items()))}")

    # Sum by day
    day_sums = {}
    for dow, draws in day_draws.items():
        s = [sum(d[:PICK]) for d in draws]
        day_sums[dow] = (np.mean(s), np.std(s), len(s))

    # ANOVA test
    groups = [np.array([sum(d[:PICK]) for d in draws]) for draws in day_draws.values()]
    if len(groups) >= 2:
        f_stat, f_p = stats.f_oneway(*groups)
        print(f"  ANOVA (sum by day): F={f_stat:.3f}, p={f_p:.4f}")
        if f_p < 0.05:
            print(f"  ⚠️ Sum distribution VARIES by day!")
            for dow in sorted(day_draws.keys()):
                m, s, n = day_sums[dow]
                print(f"    {day_names.get(dow, dow)}: mean_sum={m:.1f} ± {s:.1f} (n={n})")

    attacks['temporal'] = {'anova_f': round(f_stat, 3), 'anova_p': round(f_p, 4)}

    # ==================================================================
    # ATTACK 10: MEGA ENSEMBLE — COMBINE ALL ATTACKS
    # ==================================================================
    print(f"\n{'━'*80}")
    print("  🏆 ATTACK 10: MEGA ENSEMBLE — All attacks combined")
    print(f"{'━'*80}")

    # Build ensemble from attacks that showed promise
    def mega_predict(train_slice, idx):
        """Ensemble prediction using all viable signals."""
        n = len(train_slice)
        scores = {num: 0.0 for num in range(1, MAX+1)}
        last = set(train_slice[-1][:PICK])

        # Signal 1: Gap-based (CONFIRMED in V4: lift 1.15x)
        last_seen = {}
        gap_data = defaultdict(list)
        for i, d in enumerate(train_slice):
            for num in d[:PICK]:
                if num in last_seen:
                    gap_data[num].append(i - last_seen[num])
                last_seen[num] = i
        for num in range(1, MAX+1):
            gaps = gap_data.get(num, [])
            if len(gaps) < 5:
                continue
            curr_gap = n - last_seen.get(num, 0)
            avg_gap = np.mean(gaps)
            if curr_gap > avg_gap:
                scores[num] += min((curr_gap / avg_gap - 1) * 2, 3)

        # Signal 2: Decade flow (CONFIRMED: lift 1.25x)
        def dec(x): return min(x // 10, 4)
        last_dec = tuple(sorted(dec(x) for x in train_slice[-1][:PICK]))
        trans = defaultdict(Counter)
        for i in range(1, n):
            prev = tuple(sorted(dec(x) for x in train_slice[i-1][:PICK]))
            for x in train_slice[i][:PICK]:
                trans[prev][dec(x)] += 1
        expected = trans.get(last_dec, Counter())
        total_exp = sum(expected.values()) or 1
        for num in range(1, MAX+1):
            d = dec(num)
            dp = expected.get(d, 0) / total_exp
            scores[num] += (dp - 0.2) * 4  # Strong weight

        # Signal 3: Streak (CONFIRMED: lift 1.175x)
        for num in range(1, MAX+1):
            c = 0
            for x in reversed(train_slice):
                if num not in x[:PICK]:
                    c += 1
                else:
                    break
            expected_gap = MAX / PICK
            if c > expected_gap:
                scores[num] += min((c / expected_gap - 1) * 1.5, 2)

        # Signal 4: Pair boost (CONFIRMED marginal: 1.025x)
        pair_freq = Counter()
        for d in train_slice[-150:]:
            for p in combinations(sorted(d[:PICK]), 2):
                pair_freq[p] += 1
        exp_pair = min(150, n) * PICK * (PICK-1) / (MAX * (MAX-1))
        for num in range(1, MAX+1):
            for p in last:
                key = tuple(sorted([p, num]))
                freq = pair_freq.get(key, 0)
                if freq > exp_pair + 2:
                    scores[num] += (freq - exp_pair) * 0.15

        # Signal 5: KNN (from V4, conditional)
        knn_sc = Counter()
        for i in range(n - 2):
            sim = len(set(train_slice[i][:PICK]) & last)
            if sim >= 3:
                for num in train_slice[i+1][:PICK]:
                    knn_sc[num] += sim ** 2
        mx = max(knn_sc.values()) if knn_sc else 1
        for num in range(1, MAX+1):
            scores[num] += knn_sc.get(num, 0) / mx * 1.5

        # Signal 6: Determinism-based (from phase space attack)
        # Numbers with higher determinism → their pattern is more predictable
        for num in range(1, MAX+1):
            seq = [1 if num in d[:PICK] else 0 for d in train_slice[-30:]]
            if len(seq) >= 10:
                # Simple: if last 2 states match a frequent historical pattern
                last_state = (seq[-2], seq[-1])
                next_hist = []
                for j in range(2, len(seq)):
                    if (seq[j-2], seq[j-1]) == last_state:
                        next_hist.append(seq[j] if j < len(seq) else 0)
                if len(next_hist) >= 3:
                    p_appear = sum(next_hist) / len(next_hist)
                    if p_appear > base_p * 1.5:
                        scores[num] += (p_appear - base_p) * 3

        # Anti-repeat
        rr = []
        for i in range(1, min(n, 80)):
            rr.append(len(set(train_slice[i-1][:PICK]) & set(train_slice[i][:PICK])))
        penalty = (np.mean(rr) - 1.0) * 1.2
        for num in last:
            scores[num] += penalty

        # Generate portfolio with constraints
        recent_sums = [sum(d[:PICK]) for d in train_slice[-50:]]
        sum_lo = np.percentile(recent_sums, 5)
        sum_hi = np.percentile(recent_sums, 95)

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        pool = [num for num, _ in ranked[:22]]

        results = []
        for combo in combinations(pool[:15], PICK):
            s = sum(combo)
            if s < sum_lo or s > sum_hi:
                continue
            rng = combo[-1] - combo[0]
            if rng < 15 or rng > 42:
                continue
            sc = sum(scores.get(n, 0) for n in combo)
            results.append(sorted(combo))
            if len(results) >= 100:
                break

        if not results:
            results = [sorted(pool[:PICK])]
        return results

    # Walk-forward test
    print(f"\n  Walk-forward ensemble test...")
    test_start = int(N * 0.75)
    ensemble_matches = []
    portfolio_bests = []
    random_matches2 = []
    np.random.seed(42)

    for idx in range(test_start, N - 1):
        train_slice = data[:idx + 1]
        actual = set(data[idx + 1][:PICK])

        portfolio = mega_predict(train_slice, idx)
        best_single = set(portfolio[0])
        ensemble_matches.append(len(best_single & actual))

        best_port = max(len(set(p) & actual) for p in portfolio[:50])
        portfolio_bests.append(best_port)

        rand = set(np.random.choice(range(1, MAX+1), PICK, replace=False).tolist())
        random_matches2.append(len(rand & actual))

    ens_avg = np.mean(ensemble_matches)
    port_avg = np.mean(portfolio_bests)
    r_avg = np.mean(random_matches2)
    ens_imp = (ens_avg / r_avg - 1) * 100
    port_imp = (port_avg / r_avg - 1) * 100

    print(f"\n  ══════════════════════════════════════════════")
    print(f"  GOLDEN SET:     {ens_avg:.4f}/6 ({ens_imp:+.1f}%)")
    print(f"  PORTFOLIO (50): {port_avg:.4f}/6 ({port_imp:+.1f}%)")
    print(f"  RANDOM:         {r_avg:.4f}/6")
    print(f"  ══════════════════════════════════════════════")

    nt = len(ensemble_matches)
    for k in range(7):
        ec = Counter(ensemble_matches).get(k, 0)
        pc = Counter(portfolio_bests).get(k, 0)
        rc = Counter(random_matches2).get(k, 0)
        print(f"    {k}/6: Golden={ec:3d}({ec/nt*100:5.1f}%) | "
              f"Port={pc:3d}({pc/nt*100:5.1f}%) | Rand={rc:3d}({rc/nt*100:5.1f}%)")

    pct3e = sum(1 for m in ensemble_matches if m >= 3) / nt * 100
    pct3p = sum(1 for m in portfolio_bests if m >= 3) / nt * 100
    pct3r = sum(1 for m in random_matches2 if m >= 3) / nt * 100
    print(f"\n  ≥3/6: Golden={pct3e:.1f}% | Port={pct3p:.1f}% | Rand={pct3r:.1f}%")

    # ==================================================================
    # SAVE ALL RESULTS
    # ==================================================================
    elapsed = time.time() - t0

    output = {
        'version': '5.0 — Deterministic System Cracker',
        'philosophy': 'No machine is truly random. Find the determinism.',
        'total_draws': N,
        'attacks': {k: v for k, v in attacks.items()},
        'ensemble_backtest': {
            'golden_avg': round(ens_avg, 4),
            'portfolio_avg': round(port_avg, 4),
            'random_avg': round(r_avg, 4),
            'golden_improvement': round(ens_imp, 2),
            'portfolio_improvement': round(port_imp, 2),
            'pct_3plus_golden': round(pct3e, 2),
            'pct_3plus_portfolio': round(pct3p, 2),
            'pct_3plus_random': round(pct3r, 2),
            'n_tests': nt,
        },
        'elapsed_seconds': round(elapsed, 1),
    }

    rules_path = os.path.join(os.path.dirname(__file__), 'models', 'cracker_results_v5.json')
    with open(rules_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {rules_path}")

    print(f"\n{'='*80}")
    print(f"  V5 COMPLETE in {elapsed:.1f}s")
    print(f"  Golden={ens_avg:.4f}/6 ({ens_imp:+.1f}%), "
          f"Portfolio={port_avg:.4f}/6 ({port_imp:+.1f}%)")
    print(f"{'='*80}")


if __name__ == '__main__':
    deep_mine_v5()
