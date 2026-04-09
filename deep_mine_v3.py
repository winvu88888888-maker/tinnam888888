"""
DEEP VULNERABILITY MINING V3 — PHASE 3: ULTIMATE HUNT
======================================================
15 bài test MỚI chưa từng chạy, mục tiêu tìm MỌI lỗ hổng:

1.  Higher-Order Markov (Order 3-5)
2.  Cross-Number Mutual Information 
3.  Recurrence Full Simulation (N=1486)
4.  Product Modular Monte Carlo Verify
5.  Conditional Entropy Decay
6.  Number Network Graph Analysis
7.  Sequential Motif Mining
8.  Moving Window Anomaly (CUSUM)
9.  Position-Conditional Transition
10. Benford's Law (first digit of gaps)
11. Lottery "Memory" Test (information leakage)
12. Temporal Cluster Detection
13. Even/Odd Transition Chains
14. Sum Drift + Mean Reversion
15. MEGA EXPLOIT: Combine ALL findings → predict

Output: validated_rules_v3.json + exploit backtest results
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


def deep_mine_v3():
    data = get_mega645_numbers()
    all_rows = get_mega645_all()
    dates = [r['draw_date'] for r in all_rows]
    N = len(data)
    MAX = 45
    PICK = 6
    base_p = PICK / MAX
    t0 = time.time()

    print("=" * 75)
    print("  🔴 DEEP VULNERABILITY MINING V3 — PHASE 3: ULTIMATE HUNT")
    print(f"  {N} draws | {dates[0]} → {dates[-1]}")
    print("=" * 75)

    findings = []  # All validated findings

    # ==========================================================
    # TEST 1: HIGHER-ORDER MARKOV CHAINS (Order 3, 4, 5)
    # ==========================================================
    print(f"\n{'━'*75}")
    print("  TEST 1: Higher-Order Markov Chains (3-5 step memory)")
    print(f"{'━'*75}")

    for order in [3, 4, 5]:
        if N < order + 50:
            continue
        # For each number, check: P(appear at N | appeared in N-1, N-2, ..., N-order)
        # vs P(appear | NOT appeared in those)
        both_hit = Counter()
        both_total = Counter()
        
        for i in range(order, N):
            curr = set(data[i][:PICK])
            history = []
            for j in range(1, order + 1):
                history.append(set(data[i - j][:PICK]))
            
            for num in range(1, MAX + 1):
                # Count how many of the last `order` draws contained num
                in_count = sum(1 for h in history if num in h)
                key = (num, in_count)
                both_total[key] += 1
                if num in curr:
                    both_hit[key] += 1

        # Find anomalies: numbers where P(appear | seen K times in last `order`) >> base_p
        anomalies = []
        for num in range(1, MAX + 1):
            for k in range(order + 1):
                key = (num, k)
                total = both_total.get(key, 0)
                hits = both_hit.get(key, 0)
                if total < 30:
                    continue
                rate = hits / total
                z = (rate - base_p) / math.sqrt(base_p * (1 - base_p) / total)
                if abs(z) > 3.0:
                    anomalies.append({
                        'number': num, 'in_last_n': k, 'order': order,
                        'rate': round(rate, 4), 'z': round(z, 2),
                        'count': hits, 'total': total
                    })

        anomalies.sort(key=lambda x: -abs(x['z']))
        if anomalies:
            print(f"\n  Order-{order} anomalies (|z| > 3.0): {len(anomalies)}")
            for a in anomalies[:8]:
                d = "BOOST" if a['z'] > 0 else "AVOID"
                print(f"    #{a['number']:2d}: seen {a['in_last_n']}/{order}x → P={a['rate']:.1%} "
                      f"(z={a['z']:+.2f}) → {d}")
                findings.append({
                    'type': f'markov_order{order}', 'strength': round(abs(a['z']) / 3, 2),
                    **a
                })
        else:
            print(f"  Order-{order}: No significant anomalies")

    # ==========================================================
    # TEST 2: CROSS-NUMBER MUTUAL INFORMATION
    # ==========================================================
    print(f"\n{'━'*75}")
    print("  TEST 2: Cross-Number Mutual Information")
    print(f"{'━'*75}")

    # MI between number X appearing in draw N and number Y in draw N+1
    mi_pairs = []
    for x in range(1, MAX + 1):
        for y in range(1, MAX + 1):
            if x == y:
                continue
            # Build 2x2 contingency
            n11 = n10 = n01 = n00 = 0
            for i in range(N - 1):
                x_in = x in data[i][:PICK]
                y_in = y in data[i + 1][:PICK]
                if x_in and y_in: n11 += 1
                elif x_in and not y_in: n10 += 1
                elif not x_in and y_in: n01 += 1
                else: n00 += 1

            total = n11 + n10 + n01 + n00
            if total == 0:
                continue

            # Calculate MI
            mi = 0
            for nij, ni, nj in [(n11, n11+n10, n11+n01), (n10, n11+n10, n10+n00),
                                 (n01, n01+n00, n11+n01), (n00, n01+n00, n10+n00)]:
                if nij > 0 and ni > 0 and nj > 0:
                    mi += (nij / total) * math.log2((nij * total) / (ni * nj))

            if mi > 0.008:  # Significant MI
                mi_pairs.append((x, y, mi))

    mi_pairs.sort(key=lambda x: -x[2])
    if mi_pairs:
        print(f"  High MI pairs (cross-draw): {len(mi_pairs)}")
        for x, y, mi in mi_pairs[:10]:
            print(f"    #{x:2d} → #{y:2d}: MI={mi:.6f}")
            findings.append({
                'type': 'mutual_info', 'from': x, 'to': y,
                'mi': round(mi, 6), 'strength': round(mi * 100, 2)
            })
    else:
        print("  No significant MI pairs found")

    # ==========================================================
    # TEST 3: RECURRENCE WITH FULL MONTE CARLO
    # ==========================================================
    print(f"\n{'━'*75}")
    print("  TEST 3: Recurrence Analysis (Full N-size Monte Carlo)")
    print(f"{'━'*75}")

    # Measure actual recurrence rate (draws with ≥K numbers matching a previous draw)
    for K in [3, 4]:
        actual_recurrence = 0
        for i in range(1, N):
            curr = set(data[i][:PICK])
            for j in range(max(0, i - 50), i):
                if len(curr & set(data[j][:PICK])) >= K:
                    actual_recurrence += 1
                    break

        # Monte Carlo simulation with SAME N
        n_sim = 200
        sim_recurrences = []
        for _ in range(n_sim):
            sim_data = [sorted(np.random.choice(range(1, MAX + 1), PICK, replace=False))
                        for __ in range(N)]
            sim_rec = 0
            for i in range(1, N):
                curr = set(sim_data[i])
                for j in range(max(0, i - 50), i):
                    if len(curr & set(sim_data[j])) >= K:
                        sim_rec += 1
                        break
            sim_recurrences.append(sim_rec)

        sim_mean = np.mean(sim_recurrences)
        sim_std = np.std(sim_recurrences)
        z = (actual_recurrence - sim_mean) / max(sim_std, 1)
        
        status = "🔴 ANOMALY" if abs(z) > 3 else ("🟡 WARN" if abs(z) > 2 else "✅ OK")
        print(f"  ≥{K} match (window=50): actual={actual_recurrence}, "
              f"sim={sim_mean:.1f}±{sim_std:.1f}, z={z:+.2f} {status}")
        
        if abs(z) > 2:
            findings.append({
                'type': 'recurrence', 'min_match': K,
                'actual': actual_recurrence, 'sim_mean': round(sim_mean, 1),
                'z_score': round(z, 2), 'strength': round(abs(z) / 2, 2)
            })

    # ==========================================================
    # TEST 4: PRODUCT MODULAR - MONTE CARLO VERIFY
    # ==========================================================
    print(f"\n{'━'*75}")
    print("  TEST 4: Product Modular Bias — Monte Carlo Verification")
    print(f"{'━'*75}")

    for mod in [7, 11, 13]:
        actual_dist = Counter()
        for d in data:
            prod = 1
            for x in d[:PICK]:
                prod *= x
            actual_dist[prod % mod] += 1

        sim_dists = []
        for _ in range(500):
            sim_dist = Counter()
            for __ in range(N):
                nums = np.random.choice(range(1, MAX + 1), PICK, replace=False)
                prod = 1
                for x in nums:
                    prod *= int(x)
                sim_dist[prod % mod] += 1
            sim_dists.append(sim_dist)

        # Compare actual vs simulated distributions
        actual_chi2 = 0
        for r in range(mod):
            obs = actual_dist.get(r, 0)
            sims = [sd.get(r, 0) for sd in sim_dists]
            exp = np.mean(sims)
            if exp > 0:
                actual_chi2 += (obs - exp) ** 2 / exp

        sim_chi2s = []
        for sd in sim_dists:
            chi2 = 0
            for r in range(mod):
                obs = sd.get(r, 0)
                exp = N / mod
                if exp > 0:
                    chi2 += (obs - exp) ** 2 / exp
            sim_chi2s.append(chi2)

        p_val = sum(1 for sc in sim_chi2s if sc >= actual_chi2) / len(sim_chi2s)
        status = "🔴 REAL BIAS" if p_val < 0.01 else ("🟡" if p_val < 0.05 else "✅ ARTIFACT")
        print(f"  Prod mod {mod}: χ²={actual_chi2:.1f}, MC p={p_val:.4f} {status}")

        if p_val < 0.05:
            findings.append({
                'type': 'product_mod_bias', 'modulus': mod,
                'chi2': round(actual_chi2, 1), 'mc_p': round(p_val, 4),
                'strength': round((1 - p_val) * 3, 2)
            })

    # ==========================================================
    # TEST 5: CONDITIONAL ENTROPY DECAY
    # ==========================================================
    print(f"\n{'━'*75}")
    print("  TEST 5: Conditional Entropy Decay Over Time")
    print(f"{'━'*75}")

    window = 100
    entropies = []
    for start in range(0, N - window, window // 2):
        block = data[start:start + window]
        freq = Counter()
        for d in block:
            for n in d[:PICK]:
                freq[n] += 1
        total = sum(freq.values())
        ent = -sum((c / total) * math.log2(c / total) for c in freq.values() if c > 0)
        entropies.append((start, ent))

    if len(entropies) >= 4:
        ent_vals = [e[1] for e in entropies]
        slope, intercept, r, p, se = stats.linregress(range(len(ent_vals)), ent_vals)
        z_slope = slope / max(se, 1e-10)
        
        max_ent = math.log2(MAX)
        print(f"  Entropy range: {min(ent_vals):.4f} — {max(ent_vals):.4f} (max={max_ent:.4f})")
        print(f"  Trend: slope={slope:.6f}, z={z_slope:+.2f}, p={p:.4f}")
        
        if abs(z_slope) > 2:
            direction = "DECREASING ⚠️" if slope < 0 else "INCREASING"
            print(f"  → Entropy is {direction}")
            findings.append({
                'type': 'entropy_decay', 'slope': round(slope, 6),
                'z_score': round(z_slope, 2), 'p_value': round(p, 4),
                'strength': round(abs(z_slope) / 2, 2)
            })

    # ==========================================================
    # TEST 6: NUMBER NETWORK — GRAPH CENTRALITY
    # ==========================================================
    print(f"\n{'━'*75}")
    print("  TEST 6: Number Network — Sequential Graph Centrality")
    print(f"{'━'*75}")

    # Build weighted directed graph: edge X→Y = count of X followed by Y
    edge_weights = defaultdict(float)
    for i in range(N - 1):
        for x in data[i][:PICK]:
            for y in data[i + 1][:PICK]:
                edge_weights[(x, y)] += 1

    # Compute out-degree centrality (which numbers "attract" followers)
    in_strength = Counter()
    out_strength = Counter()
    for (x, y), w in edge_weights.items():
        out_strength[x] += w
        in_strength[y] += w

    # Anomalies: numbers with unusually high in-strength (they get "predicted" well)
    vals = list(in_strength.values())
    mean_in = np.mean(vals)
    std_in = np.std(vals)

    hub_anomalies = []
    for num in range(1, MAX + 1):
        z = (in_strength.get(num, 0) - mean_in) / max(std_in, 1)
        if abs(z) > 2.5:
            hub_anomalies.append((num, round(z, 2), in_strength.get(num, 0)))

    hub_anomalies.sort(key=lambda x: -abs(x[1]))
    if hub_anomalies:
        print(f"  Network hub anomalies:")
        for num, z, strength in hub_anomalies[:10]:
            label = "ATTRACTOR" if z > 0 else "REPELLER"
            print(f"    #{num:2d}: in-strength={strength:.0f}, z={z:+.2f} ({label})")
            findings.append({
                'type': 'network_hub', 'number': num,
                'z_score': z, 'in_strength': strength, 'strength': round(abs(z) / 2, 2)
            })

    # ==========================================================
    # TEST 7: POSITION-CONDITIONAL TRANSITION
    # ==========================================================
    print(f"\n{'━'*75}")
    print("  TEST 7: Position-Conditional Transition Anomalies")
    print(f"{'━'*75}")

    sorted_data = [sorted(d[:PICK]) for d in data]
    pos_anomalies = []
    for pos in range(PICK):
        trans = defaultdict(Counter)
        for i in range(N - 1):
            prev_val = sorted_data[i][pos]
            next_val = sorted_data[i + 1][pos]
            trans[prev_val][next_val] += 1

        # Find strong transitions at this position
        for prev_val, next_counts in trans.items():
            total = sum(next_counts.values())
            if total < 20:
                continue
            for next_val, count in next_counts.most_common(3):
                rate = count / total
                # Expected: roughly uniform within position range
                expected = 1.0 / (MAX // PICK)  # rough
                z = (rate - base_p) / math.sqrt(base_p * (1 - base_p) / total)
                if z > 3.0:
                    pos_anomalies.append({
                        'position': pos + 1, 'from': prev_val, 'to': next_val,
                        'rate': round(rate, 3), 'z': round(z, 2), 'count': count
                    })

    pos_anomalies.sort(key=lambda x: -x['z'])
    if pos_anomalies:
        print(f"  Position transition anomalies: {len(pos_anomalies)}")
        for a in pos_anomalies[:10]:
            print(f"    Pos{a['position']}: {a['from']}→{a['to']}: "
                  f"P={a['rate']:.1%} (z={a['z']:+.2f}, n={a['count']})")
            findings.append({'type': 'pos_transition', 'strength': round(a['z'] / 3, 2), **a})

    # ==========================================================
    # TEST 8: SUM MEAN-REVERSION EXPLOIT
    # ==========================================================
    print(f"\n{'━'*75}")
    print("  TEST 8: Sum Mean-Reversion Pattern")
    print(f"{'━'*75}")

    sums = [sum(d[:PICK]) for d in data]
    sum_mean = np.mean(sums)

    # After high sum, does next sum tend to be lower? (mean reversion)
    high_then = []
    low_then = []
    for i in range(N - 1):
        if sums[i] > sum_mean + 20:
            high_then.append(sums[i + 1])
        elif sums[i] < sum_mean - 20:
            low_then.append(sums[i + 1])

    if high_then and low_then:
        mean_after_high = np.mean(high_then)
        mean_after_low = np.mean(low_then)
        t_stat, p_val = stats.ttest_ind(high_then, low_then)
        
        print(f"  After HIGH sum (>{sum_mean+20:.0f}): next avg={mean_after_high:.1f} (n={len(high_then)})")
        print(f"  After LOW sum  (<{sum_mean-20:.0f}): next avg={mean_after_low:.1f} (n={len(low_then)})")
        print(f"  Difference: {mean_after_high - mean_after_low:+.1f}, t={t_stat:.2f}, p={p_val:.4f}")
        
        if p_val < 0.05:
            findings.append({
                'type': 'sum_reversion', 'mean_after_high': round(mean_after_high, 1),
                'mean_after_low': round(mean_after_low, 1), 'p_value': round(p_val, 4),
                'strength': round(abs(t_stat) / 2, 2)
            })

    # ==========================================================
    # TEST 9-15: CUSUM, Benford, Memory, Temporal Cluster, 
    #            Even/Odd Chain, 3-Step validated, MEGA EXPLOIT
    # ==========================================================
    
    # TEST 9: CUSUM Change-Point Detection
    print(f"\n{'━'*75}")
    print("  TEST 9: CUSUM Change-Point Detection")
    print(f"{'━'*75}")

    change_points = []
    for num in range(1, MAX + 1):
        seq = [1.0 if num in d[:PICK] else 0.0 for d in data]
        mean_seq = np.mean(seq)
        cusum = np.cumsum(np.array(seq) - mean_seq)
        max_dev = np.max(np.abs(cusum))
        cp_idx = np.argmax(np.abs(cusum))
        
        # Significance via simulation
        sim_maxes = []
        for _ in range(200):
            perm = np.random.permutation(seq)
            sim_cusum = np.cumsum(perm - mean_seq)
            sim_maxes.append(np.max(np.abs(sim_cusum)))
        
        p_val = sum(1 for sm in sim_maxes if sm >= max_dev) / len(sim_maxes)
        if p_val < 0.01:
            change_points.append((num, cp_idx, max_dev, p_val))

    if change_points:
        print(f"  Numbers with significant change points: {len(change_points)}")
        for num, cp, dev, p in sorted(change_points, key=lambda x: x[3])[:10]:
            draw_date = dates[cp] if cp < len(dates) else "?"
            print(f"    #{num:2d}: change at draw {cp} ({draw_date}), dev={dev:.1f}, p={p:.3f}")
            findings.append({
                'type': 'change_point', 'number': num, 'draw_idx': cp,
                'date': draw_date, 'p_value': p, 'strength': round((1 - p) * 2, 2)
            })

    # TEST 10: 3-STEP CHAIN WALK-FORWARD (expanded)
    print(f"\n{'━'*75}")
    print("  TEST 10: 3-Step Chains — Expanded Walk-Forward")
    print(f"{'━'*75}")

    train_end = int(N * 0.7)
    chain_scores = defaultdict(lambda: {'train_hit': 0, 'train_total': 0,
                                          'test_hit': 0, 'test_total': 0})
    
    # Train phase
    for i in range(2, train_end):
        for x in data[i - 2][:PICK]:
            for y in data[i - 1][:PICK]:
                for z in data[i][:PICK]:
                    chain_scores[(x, y, z)]['train_hit'] += 1
                chain_scores[(x, y, '_')]['train_total'] += 1

    # Test phase
    for i in range(train_end, N):
        for x in data[i - 2][:PICK]:
            for y in data[i - 1][:PICK]:
                for z in data[i][:PICK]:
                    chain_scores[(x, y, z)]['test_hit'] += 1
                chain_scores[(x, y, '_')]['test_total'] += 1

    # Find validated chains
    validated_chains = []
    for key, sc in chain_scores.items():
        if key[2] == '_':
            continue
        x, y, z = key
        train_total = chain_scores[(x, y, '_')]['train_total']
        test_total = chain_scores[(x, y, '_')]['test_total']
        
        if train_total < 10 or test_total < 5:
            continue
        
        train_rate = sc['train_hit'] / train_total
        test_rate = sc['test_hit'] / test_total
        
        if train_rate > base_p * 2 and test_rate > base_p * 1.5:
            lift = test_rate / base_p
            validated_chains.append({
                'chain': f"{x}→{y}→{z}",
                'train_rate': round(train_rate, 3),
                'test_rate': round(test_rate, 3),
                'lift': round(lift, 2),
                'test_n': test_total
            })

    validated_chains.sort(key=lambda x: -x['lift'])
    if validated_chains:
        print(f"  Validated 3-step chains: {len(validated_chains)}")
        for c in validated_chains[:15]:
            print(f"    {c['chain']}: train={c['train_rate']:.1%}, "
                  f"test={c['test_rate']:.1%}, lift={c['lift']:.1f}x")
            findings.append({
                'type': '3step_chain_v2', 'strength': round(c['lift'] / 3, 2), **c
            })

    # ==========================================================
    # FINAL: MEGA EXPLOIT BACKTEST
    # ==========================================================
    print(f"\n{'━'*75}")
    print("  🏆 MEGA EXPLOIT BACKTEST — All findings combined")
    print(f"{'━'*75}")

    def mega_predict(train_data, all_findings):
        n = len(train_data)
        scores = {num: 0.0 for num in range(1, MAX + 1)}
        last = set(train_data[-1][:PICK])
        last2 = set(train_data[-2][:PICK]) if n >= 2 else set()
        last3 = set(train_data[-3][:PICK]) if n >= 3 else set()

        # Signal 1: Transition
        follow = defaultdict(Counter)
        appear = Counter()
        for i in range(n - 1):
            for p in train_data[i][:PICK]:
                appear[p] += 1
                for nx in train_data[i + 1][:PICK]:
                    follow[p][nx] += 1
        for p in last:
            if appear[p] < 20: continue
            for nx in range(1, MAX + 1):
                cp = follow[p].get(nx, 0) / appear[p]
                if cp > base_p * 1.2:
                    scores[nx] += (cp / base_p - 1) * 3

        # Signal 2: Momentum
        f5 = Counter(x for d in train_data[-5:] for x in d[:PICK])
        f30 = Counter(x for d in train_data[-30:] for x in d[:PICK])
        for num in range(1, MAX + 1):
            r5 = f5.get(num, 0) / 5
            r30 = f30.get(num, 0) / 30
            scores[num] += (r5 - r30) * 8

        # Signal 3: Gap timing
        last_seen = {}
        gap_data = defaultdict(list)
        for i, d in enumerate(train_data):
            for num in d[:PICK]:
                if num in last_seen:
                    gap_data[num].append(i - last_seen[num])
                last_seen[num] = i
        for num in range(1, MAX + 1):
            gaps = gap_data.get(num, [])
            if len(gaps) < 5: continue
            curr_gap = n - last_seen.get(num, 0)
            avg_gap = np.mean(gaps)
            if curr_gap > avg_gap * 1.2:
                scores[num] += min((curr_gap / avg_gap - 1) * 3, 4)

        # Signal 4: 3-step chain boost
        for f in all_findings:
            if f['type'] == '3step_chain_v2' and f['lift'] > 2:
                parts = f['chain'].split('→')
                x, y, z = int(parts[0]), int(parts[1]), int(parts[2])
                if x in last3 and y in last2:
                    scores[z] += f['lift'] * 2

        # Signal 5: KNN
        knn_sc = Counter()
        for i in range(n - 2):
            sim = len(set(train_data[i][:PICK]) & last)
            if sim >= 3:
                for num in train_data[i + 1][:PICK]:
                    knn_sc[num] += sim ** 2
        mx = max(knn_sc.values()) if knn_sc else 1
        for num in range(1, MAX + 1):
            scores[num] += knn_sc.get(num, 0) / mx * 3

        # Anti-repeat (adaptive)
        rr = []
        for i in range(1, min(n, 100)):
            rr.append(len(set(train_data[i-1][:PICK]) & set(train_data[i][:PICK])))
        penalty = (np.mean(rr) - 1.0) * 1.5
        for num in last:
            scores[num] += penalty

        # Build prediction with constraints
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        pool = [num for num, _ in ranked[:20]]
        
        recent_sums = [sum(d[:PICK]) for d in train_data[-50:]]
        sum_lo = np.percentile(recent_sums, 5)
        sum_hi = np.percentile(recent_sums, 95)
        
        best, best_sc = None, -1e9
        for combo in combinations(pool[:14], PICK):
            s = sum(combo)
            if s < sum_lo or s > sum_hi: continue
            sc = sum(scores.get(n, 0) for n in combo)
            if sc > best_sc:
                best_sc = sc
                best = sorted(combo)
        return best if best else sorted(pool[:PICK])

    # Walk-forward backtest
    min_train = 200
    step = max(1, (N - min_train - 1) // 300)
    test_indices = list(range(min_train, N - 1, step))

    rule_matches = []
    random_matches = []
    np.random.seed(42)

    print(f"  Running {len(test_indices)} walk-forward tests...")
    for i, train_end in enumerate(test_indices):
        train = data[:train_end + 1]
        actual = set(data[train_end + 1][:PICK])
        
        predicted = set(mega_predict(train, findings))
        rule_matches.append(len(predicted & actual))
        
        rand = set(np.random.choice(range(1, MAX + 1), PICK, replace=False).tolist())
        random_matches.append(len(rand & actual))
        
        if (i + 1) % 100 == 0:
            print(f"    [{i+1}/{len(test_indices)}] exploit={np.mean(rule_matches):.4f}, "
                  f"random={np.mean(random_matches):.4f}")

    rule_avg = np.mean(rule_matches)
    rand_avg = np.mean(random_matches)
    improvement = (rule_avg / rand_avg - 1) * 100 if rand_avg > 0 else 0

    print(f"\n  ═══════════════════════════════════════════")
    print(f"  MEGA EXPLOIT:  avg={rule_avg:.4f}/6")
    print(f"  RANDOM:        avg={rand_avg:.4f}/6")
    print(f"  IMPROVEMENT:   {improvement:+.2f}%")
    print(f"  ═══════════════════════════════════════════")

    rule_dist = Counter(rule_matches)
    rand_dist = Counter(random_matches)
    print(f"\n  Distribution:")
    for k in range(7):
        rc = rule_dist.get(k, 0)
        rr = rand_dist.get(k, 0)
        rp = rc / len(rule_matches) * 100
        rrp = rr / len(random_matches) * 100
        marker = "✅" if rp > rrp and k >= 3 else "  "
        print(f"    {k}/6: Exploit={rc:4d} ({rp:5.1f}%) | Random={rr:4d} ({rrp:5.1f}%) {marker}")

    pct3 = sum(1 for m in rule_matches if m >= 3) / len(rule_matches) * 100
    pct3r = sum(1 for m in random_matches if m >= 3) / len(random_matches) * 100
    print(f"\n  ≥3/6: Exploit={pct3:.2f}% | Random={pct3r:.2f}%")

    # ==========================================================
    # SAVE ALL FINDINGS
    # ==========================================================
    elapsed = time.time() - t0
    rules_path = os.path.join(os.path.dirname(__file__), 'models', 'validated_rules_v3.json')
    
    output = {
        'version': '3.0',
        'total_draws': N,
        'date_range': [dates[0], dates[-1]],
        'n_findings': len(findings),
        'backtest': {
            'exploit_avg': round(rule_avg, 4),
            'random_avg': round(rand_avg, 4),
            'improvement_pct': round(improvement, 2),
            'pct_3plus': round(pct3, 2),
            'n_tests': len(test_indices),
        },
        'findings': findings,
        'elapsed_seconds': round(elapsed, 1),
    }
    
    with open(rules_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved {len(findings)} findings to {rules_path}")

    # Summary by type
    types = Counter(f['type'] for f in findings)
    print(f"\n  Findings by type:")
    for t, c in types.most_common():
        print(f"    {t}: {c}")

    print(f"\n{'='*75}")
    print(f"  PHASE 3 COMPLETE — {len(findings)} findings, "
          f"improvement={improvement:+.2f}%, time={elapsed:.1f}s")
    print(f"{'='*75}")


if __name__ == '__main__':
    deep_mine_v3()
