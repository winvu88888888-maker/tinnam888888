"""
DEEP FORENSIC PATTERN DISCOVERY
================================
Phân tích TOÀN BỘ dữ liệu 1486 kỳ Mega 6/45 để tìm:
1. Quy luật lặp lại THỰC SỰ (không phải noise)
2. Bất thường KHÔNG THỂ giải thích bằng ngẫu nhiên
3. Statistical tests nghiêm ngặt (p-value < 0.01)

Dùng: Chi-square, Z-test, Runs test, Serial correlation, etc.
"""
import sys, os, math
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper.data_manager import get_mega645_numbers, get_mega645_all


def analyze():
    data = get_mega645_numbers()
    all_rows = get_mega645_all()
    dates = [r['draw_date'] for r in all_rows]
    N = len(data)
    MAX = 45
    PICK = 6

    print("=" * 75)
    print("  DEEP FORENSIC PATTERN DISCOVERY — Mega 6/45")
    print(f"  {N} draws | {dates[0]} → {dates[-1]}")
    print("=" * 75)

    findings = []  # (importance, category, description)

    # ==========================================================
    # 1. INDIVIDUAL NUMBER FREQUENCY — Chi-Square Test
    # ==========================================================
    print(f"\n{'─'*75}")
    print("  1. INDIVIDUAL NUMBER FREQUENCY ANALYSIS")
    print(f"{'─'*75}")

    freq = Counter()
    for d in data:
        for n in d[:PICK]:
            freq[n] += 1

    total_balls = N * PICK
    expected = total_balls / MAX
    print(f"  Expected frequency per number: {expected:.1f}")
    print(f"  Total balls drawn: {total_balls}")

    chi2_stat = sum((freq.get(n, 0) - expected)**2 / expected for n in range(1, MAX+1))
    chi2_p = 1 - stats.chi2.cdf(chi2_stat, df=MAX-1)
    print(f"\n  Chi-Square test: χ² = {chi2_stat:.2f}, df={MAX-1}, p-value = {chi2_p:.6f}")
    if chi2_p < 0.01:
        print(f"  🚨 SIGNIFICANT (p < 0.01): Number frequencies are NOT uniform!")
        findings.append((10, "Frequency", f"Chi² = {chi2_stat:.2f}, p = {chi2_p:.6f} — frequencies are biased"))
    elif chi2_p < 0.05:
        print(f"  ⚠️ MARGINALLY SIGNIFICANT (p < 0.05)")
        findings.append((7, "Frequency", f"Chi² = {chi2_stat:.2f}, p = {chi2_p:.6f} — marginal bias"))
    else:
        print(f"  ✅ Frequencies are consistent with uniform random (p ≥ 0.05)")

    # Show most/least frequent
    sorted_freq = sorted(freq.items(), key=lambda x: -x[1])
    print(f"\n  Most frequent:  ", end="")
    for n, f in sorted_freq[:5]:
        z = (f - expected) / math.sqrt(expected * (1 - PICK/MAX))
        print(f"#{n}={f}(z={z:+.2f}) ", end="")
    print(f"\n  Least frequent: ", end="")
    for n, f in sorted_freq[-5:]:
        z = (f - expected) / math.sqrt(expected * (1 - PICK/MAX))
        print(f"#{n}={f}(z={z:+.2f}) ", end="")

    # Individual number z-tests
    print(f"\n\n  Numbers with |z| > 2.0 (individually unusual):")
    for n in range(1, MAX+1):
        f = freq.get(n, 0)
        z = (f - expected) / math.sqrt(expected * (1 - PICK/MAX))
        if abs(z) > 2.0:
            direction = "HOT 🔥" if z > 0 else "COLD ❄️"
            print(f"    #{n:2d}: {f} appearances, z = {z:+.3f} ({direction})")
            findings.append((8, "Frequency", f"#{n} z={z:+.3f} ({direction}), freq={f} vs expected {expected:.0f}"))

    # ==========================================================
    # 2. PAIR FREQUENCY — Which pairs appear together too often?
    # ==========================================================
    print(f"\n{'─'*75}")
    print("  2. PAIR CO-OCCURRENCE ANALYSIS")
    print(f"{'─'*75}")

    pair_freq = Counter()
    for d in data:
        for p in combinations(sorted(d[:PICK]), 2):
            pair_freq[p] += 1

    n_pairs = MAX * (MAX - 1) // 2
    expected_pair = N * (PICK * (PICK - 1)) / (MAX * (MAX - 1))
    print(f"  Expected pair frequency: {expected_pair:.2f}")
    print(f"  Total unique pairs possible: {n_pairs}")

    # Chi-square for pairs (top deviations)
    pair_deviations = []
    for a in range(1, MAX+1):
        for b in range(a+1, MAX+1):
            obs = pair_freq.get((a, b), 0)
            z = (obs - expected_pair) / math.sqrt(expected_pair * (1 - PICK*(PICK-1)/(MAX*(MAX-1))))
            if abs(z) > 2.5:
                pair_deviations.append(((a, b), obs, z))

    pair_deviations.sort(key=lambda x: -abs(x[2]))
    if pair_deviations:
        print(f"\n  🚨 {len(pair_deviations)} pairs with |z| > 2.5:")
        for (a, b), obs, z in pair_deviations[:15]:
            direction = "appears TOO OFTEN" if z > 0 else "appears TOO RARELY"
            print(f"    ({a:2d},{b:2d}): {obs}x (expected {expected_pair:.1f}), z={z:+.2f} — {direction}")
            if abs(z) > 3.0:
                findings.append((9, "Pair", f"({a},{b}) z={z:+.2f}, obs={obs} vs exp={expected_pair:.1f}"))
    else:
        print(f"  ✅ No pairs with significant deviation")

    # ==========================================================
    # 3. CONSECUTIVE DRAW REPEAT PATTERNS
    # ==========================================================
    print(f"\n{'─'*75}")
    print("  3. CONSECUTIVE REPEAT ANALYSIS")
    print(f"{'─'*75}")

    repeat_counts = []
    for i in range(1, N):
        prev = set(data[i-1][:PICK])
        curr = set(data[i][:PICK])
        repeat_counts.append(len(prev & curr))

    repeat_dist = Counter(repeat_counts)
    avg_repeat = np.mean(repeat_counts)

    # Expected: hypergeometric distribution
    # P(k repeats) = C(6,k)*C(39,6-k) / C(45,6)
    from scipy.stats import hypergeom
    expected_repeat_dist = {}
    for k in range(PICK+1):
        p = hypergeom.pmf(k, MAX, PICK, PICK)
        expected_repeat_dist[k] = p * (N - 1)

    print(f"  Average repeats per draw: {avg_repeat:.4f}")
    print(f"  Expected (hypergeometric): {sum(k*hypergeom.pmf(k,MAX,PICK,PICK) for k in range(PICK+1)):.4f}")
    print(f"\n  Repeat distribution:")
    chi2_repeat = 0
    for k in range(PICK+1):
        obs = repeat_dist.get(k, 0)
        exp = expected_repeat_dist[k]
        pct_obs = obs / (N-1) * 100
        pct_exp = exp / (N-1) * 100
        ratio = obs / max(exp, 0.01)
        if exp > 5:
            chi2_repeat += (obs - exp)**2 / exp
        print(f"    {k} repeats: {obs:5d} ({pct_obs:5.1f}%) vs expected {exp:6.1f} ({pct_exp:5.1f}%) ratio={ratio:.3f}")
    
    repeat_p = 1 - stats.chi2.cdf(chi2_repeat, df=PICK)
    print(f"\n  Chi-Square: {chi2_repeat:.2f}, p = {repeat_p:.6f}")
    if repeat_p < 0.01:
        print(f"  🚨 SIGNIFICANT: Repeat patterns differ from pure random!")
        findings.append((10, "Repeat", f"Repeat distribution χ²={chi2_repeat:.2f}, p={repeat_p:.6f}"))
    else:
        print(f"  ✅ Repeat pattern consistent with random")

    # Which numbers repeat most often?
    repeat_by_num = Counter()
    repeat_total = Counter()
    for i in range(1, N):
        prev = set(data[i-1][:PICK])
        curr = set(data[i][:PICK])
        for n in prev:
            repeat_total[n] += 1
            if n in curr:
                repeat_by_num[n] += 1
    
    print(f"\n  Numbers that repeat most often (appeared in draw N, then also in N+1):")
    expected_repeat_rate = PICK / MAX  # ~13.3%
    for n in range(1, MAX+1):
        if repeat_total[n] > 0:
            rate = repeat_by_num[n] / repeat_total[n]
            z = (rate - expected_repeat_rate) / math.sqrt(expected_repeat_rate * (1-expected_repeat_rate) / repeat_total[n])
            if abs(z) > 2.0:
                print(f"    #{n:2d}: repeats {repeat_by_num[n]}/{repeat_total[n]} = {rate:.1%} "
                      f"(expected {expected_repeat_rate:.1%}), z={z:+.2f}")
                findings.append((7, "Repeat", f"#{n} repeat rate {rate:.1%} vs {expected_repeat_rate:.1%}, z={z:+.2f}"))

    # ==========================================================
    # 4. GAP ANALYSIS — Are gaps between appearances normal?
    # ==========================================================
    print(f"\n{'─'*75}")
    print("  4. GAP DISTRIBUTION ANALYSIS")
    print(f"{'─'*75}")

    expected_gap = MAX / PICK  # ~7.5
    print(f"  Expected average gap: {expected_gap:.1f} draws")

    gap_anomalies = []
    for n in range(1, MAX+1):
        appearances = [i for i, d in enumerate(data) if n in d[:PICK]]
        if len(appearances) < 10:
            continue
        gaps = [appearances[j+1] - appearances[j] for j in range(len(appearances)-1)]
        mean_gap = np.mean(gaps)
        std_gap = np.std(gaps)
        
        # Test if gap distribution is geometric (expected for random)
        # Coefficient of variation should be close to 1 for geometric
        cv = std_gap / mean_gap if mean_gap > 0 else 0
        
        # Long streaks (max gap)
        max_gap = max(gaps)
        # Expected max gap for geometric distribution
        expected_max = mean_gap * math.log(len(gaps))
        
        if abs(cv - 1.0) > 0.25 or max_gap > expected_max * 1.5:
            gap_anomalies.append((n, mean_gap, std_gap, cv, max_gap, expected_max))

    print(f"\n  Numbers with unusual gap patterns (CV deviation > 0.25 from geometric):")
    for n, mg, sg, cv, mx, emx in sorted(gap_anomalies, key=lambda x: -abs(x[3]-1))[:10]:
        pattern = "TOO REGULAR" if cv < 0.75 else "TOO CLUSTERED" if cv > 1.25 else "UNUSUAL MAX GAP"
        print(f"    #{n:2d}: avg_gap={mg:.1f}, std={sg:.1f}, CV={cv:.3f}, max_gap={mx}, pattern={pattern}")
        if abs(cv - 1.0) > 0.3:
            findings.append((8, "Gap", f"#{n} CV={cv:.3f} ({pattern}), avg_gap={mg:.1f}"))

    # ==========================================================
    # 5. SERIAL CORRELATION (Autocorrelation)
    # ==========================================================
    print(f"\n{'─'*75}")
    print("  5. SERIAL CORRELATION TEST")
    print(f"{'─'*75}")

    # For each number, check if its appearance pattern has autocorrelation
    print(f"\n  Testing autocorrelation at lags 1-5 for all 45 numbers...")
    autocorr_anomalies = []
    for n in range(1, MAX+1):
        seq = np.array([1 if n in d[:PICK] else 0 for d in data])
        seq_centered = seq - np.mean(seq)
        var = np.var(seq)
        if var < 1e-10:
            continue
        for lag in range(1, 6):
            corr = np.sum(seq_centered[:-lag] * seq_centered[lag:]) / (len(seq) * var)
            # Standard error for autocorrelation
            se = 1 / math.sqrt(len(seq))
            z = corr / se
            if abs(z) > 2.5:
                autocorr_anomalies.append((n, lag, corr, z))

    if autocorr_anomalies:
        print(f"\n  🚨 {len(autocorr_anomalies)} significant autocorrelations (|z| > 2.5):")
        autocorr_anomalies.sort(key=lambda x: -abs(x[3]))
        for n, lag, corr, z in autocorr_anomalies[:15]:
            direction = "POSITIVE (clustering)" if corr > 0 else "NEGATIVE (alternating)"
            print(f"    #{n:2d} lag-{lag}: r={corr:+.4f}, z={z:+.2f} — {direction}")
            findings.append((9, "Serial", f"#{n} lag-{lag} autocorr r={corr:+.4f}, z={z:+.2f}"))
    else:
        print(f"  ✅ No significant autocorrelation detected")

    # ==========================================================
    # 6. SUM DISTRIBUTION — Is the sum range anomalous?
    # ==========================================================
    print(f"\n{'─'*75}")
    print("  6. SUM DISTRIBUTION ANALYSIS")
    print(f"{'─'*75}")

    sums = [sum(d[:PICK]) for d in data]
    expected_sum = PICK * (MAX + 1) / 2  # 138
    sum_mean = np.mean(sums)
    sum_std = np.std(sums)
    
    # Expected std for sum of 6 numbers from 1-45 without replacement
    # Variance = PICK * (MAX²-1)/12 * (MAX-PICK)/(MAX-1)
    expected_var = PICK * (MAX**2 - 1) / 12 * (MAX - PICK) / (MAX - 1)
    expected_std = math.sqrt(expected_var)
    
    print(f"  Mean sum: {sum_mean:.2f} (expected: {expected_sum:.2f})")
    print(f"  Std sum:  {sum_std:.2f} (expected: {expected_std:.2f})")
    
    z_mean = (sum_mean - expected_sum) / (expected_std / math.sqrt(N))
    print(f"  Z-test for mean: z = {z_mean:.3f} (p = {2*(1-stats.norm.cdf(abs(z_mean))):.6f})")
    
    if abs(z_mean) > 2.58:
        print(f"  🚨 SIGNIFICANT: Sum mean is biased!")
        findings.append((8, "Sum", f"Mean sum z={z_mean:.3f}, observed={sum_mean:.1f} vs expected={expected_sum:.1f}"))

    # Check for sum trends over time
    window = 50
    sum_trends = []
    for i in range(window, N):
        local_mean = np.mean(sums[i-window:i])
        z = (local_mean - expected_sum) / (expected_std / math.sqrt(window))
        if abs(z) > 2.5:
            sum_trends.append((i, dates[i], local_mean, z))
    
    if sum_trends:
        print(f"\n  ⚠️ {len(sum_trends)} periods with biased sum (sliding window {window}):")
        for idx, dt, lm, z in sum_trends[:5]:
            print(f"    Draw {idx} ({dt}): local_mean={lm:.1f}, z={z:+.2f}")

    # ==========================================================
    # 7. ODD/EVEN DISTRIBUTION
    # ==========================================================
    print(f"\n{'─'*75}")
    print("  7. ODD/EVEN DISTRIBUTION")
    print(f"{'─'*75}")

    odds = [sum(1 for x in d[:PICK] if x % 2 == 1) for d in data]
    odd_dist = Counter(odds)
    
    # Expected: hypergeometric (22 odd, 23 even, pick 6)
    print(f"  Distribution of odd counts in {PICK} numbers:")
    chi2_odd = 0
    for k in range(PICK+1):
        obs = odd_dist.get(k, 0)
        exp = N * hypergeom.pmf(k, MAX, 22, PICK)  # 22 odd numbers in 1-45
        if exp > 0:
            chi2_odd += (obs - exp)**2 / exp
        print(f"    {k} odd: {obs:5d} ({obs/N*100:5.1f}%) vs expected {exp:6.1f} ({exp/N*100:5.1f}%)")
    
    odd_p = 1 - stats.chi2.cdf(chi2_odd, df=PICK)
    print(f"  Chi-Square: {chi2_odd:.2f}, p = {odd_p:.6f}")
    if odd_p < 0.05:
        print(f"  ⚠️ Odd/Even distribution is unusual")
        findings.append((6, "OddEven", f"Odd/Even χ²={chi2_odd:.2f}, p={odd_p:.6f}"))

    # ==========================================================
    # 8. DECADE DISTRIBUTION
    # ==========================================================
    print(f"\n{'─'*75}")
    print("  8. DECADE DISTRIBUTION")  
    print(f"{'─'*75}")

    decade_labels = ['1-9', '10-19', '20-29', '30-39', '40-45']
    decade_sizes = [9, 10, 10, 10, 6]
    decade_draws = defaultdict(int)
    
    for d in data:
        for n in d[:PICK]:
            if n <= 9: decade_draws[0] += 1
            elif n <= 19: decade_draws[1] += 1
            elif n <= 29: decade_draws[2] += 1
            elif n <= 39: decade_draws[3] += 1
            else: decade_draws[4] += 1
    
    print(f"  Decade distribution across all {total_balls} balls:")
    chi2_dec = 0
    for i, (label, size) in enumerate(zip(decade_labels, decade_sizes)):
        obs = decade_draws[i]
        exp = total_balls * size / MAX
        z = (obs - exp) / math.sqrt(exp)
        if exp > 0:
            chi2_dec += (obs - exp)**2 / exp
        print(f"    {label:5s}: {obs:5d} (expected {exp:6.1f}), z={z:+.2f}")
        if abs(z) > 2.5:
            findings.append((7, "Decade", f"Decade {label} z={z:+.2f}"))
    
    dec_p = 1 - stats.chi2.cdf(chi2_dec, df=4)
    print(f"  Chi-Square: {chi2_dec:.2f}, p = {dec_p:.6f}")

    # ==========================================================
    # 9. CONDITIONAL TRANSITION ANOMALIES
    # ==========================================================
    print(f"\n{'─'*75}")
    print("  9. CONDITIONAL TRANSITION ANALYSIS")
    print(f"{'─'*75}")
    print(f"  For each number X: P(Y appears in next draw | X appeared this draw)")

    follow_counts = defaultdict(Counter)
    appear_counts = Counter()
    for i in range(N - 1):
        for p in data[i][:PICK]:
            appear_counts[p] += 1
            for nx in data[i+1][:PICK]:
                follow_counts[p][nx] += 1

    base_p = PICK / MAX
    transition_anomalies = []
    for prev in range(1, MAX+1):
        if appear_counts[prev] < 50:
            continue
        for nxt in range(1, MAX+1):
            obs_count = follow_counts[prev].get(nxt, 0)
            obs_p = obs_count / appear_counts[prev]
            # Expected: base_p
            z = (obs_p - base_p) / math.sqrt(base_p * (1 - base_p) / appear_counts[prev])
            if abs(z) > 3.0:
                transition_anomalies.append((prev, nxt, obs_p, base_p, z, obs_count, appear_counts[prev]))

    transition_anomalies.sort(key=lambda x: -abs(x[4]))
    if transition_anomalies:
        print(f"\n  🚨 {len(transition_anomalies)} transitions with |z| > 3.0:")
        for prev, nxt, op, bp, z, oc, tc in transition_anomalies[:20]:
            direction = "ATTRACTS" if z > 0 else "REPELS"
            print(f"    {prev:2d} → {nxt:2d}: P={op:.3f} vs base {bp:.3f}, "
                  f"z={z:+.2f}, {oc}/{tc} ({direction})")
            findings.append((9, "Transition", f"{prev}→{nxt}: P={op:.3f} vs {bp:.3f}, z={z:+.2f} ({direction})"))
    else:
        print(f"  ✅ No significant transition anomalies")

    # ==========================================================
    # 10. POSITION-SPECIFIC BIAS
    # ==========================================================
    print(f"\n{'─'*75}")
    print("  10. POSITION-SPECIFIC ANALYSIS (sorted positions 1-6)")
    print(f"{'─'*75}")

    sorted_data = [sorted(d[:PICK]) for d in data]
    pos_anomalies = []
    for pos in range(PICK):
        vals = [sd[pos] for sd in sorted_data]
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        
        # Expected mean for position pos in sorted sample of 6 from 1-45
        # Order statistic: E[X_(k)] = k*(MAX+1)/(PICK+1)
        expected_mean_pos = (pos + 1) * (MAX + 1) / (PICK + 1)
        se = (MAX + 1) / (PICK + 1) * math.sqrt((pos+1) * (PICK-pos) / ((PICK+2)))  / math.sqrt(N)
        z = (mean_val - expected_mean_pos) / max(se, 0.01)
        
        print(f"  Position {pos+1}: mean={mean_val:.2f} (expected ~{expected_mean_pos:.1f}), "
              f"std={std_val:.2f}, z={z:+.2f}")
        
        # Check most common values at each position
        val_freq = Counter(vals)
        top_3 = val_freq.most_common(3)
        print(f"           Most common: {', '.join(f'#{v}:{c}x' for v,c in top_3)}")
        
        if abs(z) > 2.5:
            pos_anomalies.append((pos, mean_val, expected_mean_pos, z))
            findings.append((7, "Position", f"Pos {pos+1}: mean={mean_val:.2f} vs {expected_mean_pos:.1f}, z={z:+.2f}"))

    # ==========================================================
    # 11. RUNS TEST — Detect non-random streaks
    # ==========================================================
    print(f"\n{'─'*75}")
    print("  11. RUNS TEST (detect non-random streaks)")
    print(f"{'─'*75}")

    runs_anomalies = []
    for n in range(1, MAX+1):
        seq = [1 if n in d[:PICK] else 0 for d in data]
        n1 = sum(seq)
        n0 = len(seq) - n1
        if n1 < 10 or n0 < 10:
            continue
        
        # Count runs
        runs = 1
        for i in range(1, len(seq)):
            if seq[i] != seq[i-1]:
                runs += 1
        
        # Expected runs
        exp_runs = 1 + 2 * n1 * n0 / (n1 + n0)
        var_runs = 2 * n1 * n0 * (2*n1*n0 - n1 - n0) / ((n1+n0)**2 * (n1+n0-1))
        z = (runs - exp_runs) / math.sqrt(max(var_runs, 0.01))
        
        if abs(z) > 2.5:
            pattern = "TOO FEW runs (clustering)" if z < 0 else "TOO MANY runs (alternating)"
            runs_anomalies.append((n, runs, exp_runs, z, pattern))
    
    if runs_anomalies:
        runs_anomalies.sort(key=lambda x: -abs(x[3]))
        print(f"\n  🚨 {len(runs_anomalies)} numbers with non-random streak patterns:")
        for n, r, er, z, pat in runs_anomalies[:10]:
            print(f"    #{n:2d}: {r} runs (expected {er:.1f}), z={z:+.2f} — {pat}")
            findings.append((8, "Runs", f"#{n} {r} runs vs {er:.1f} expected, z={z:+.2f} ({pat})"))
    else:
        print(f"  ✅ No significant run anomalies")

    # ==========================================================
    # 12. CONSECUTIVE NUMBER PATTERNS
    # ==========================================================
    print(f"\n{'─'*75}")
    print("  12. CONSECUTIVE NUMBER PATTERNS")
    print(f"{'─'*75}")

    consec_counts = []
    for d in data:
        sd = sorted(d[:PICK])
        c = sum(1 for i in range(len(sd)-1) if sd[i+1] - sd[i] == 1)
        consec_counts.append(c)
    
    consec_dist = Counter(consec_counts)
    avg_consec = np.mean(consec_counts)
    
    # Expected: complex, but simulate
    np.random.seed(42)
    sim_consec = []
    for _ in range(100000):
        s = sorted(np.random.choice(range(1, MAX+1), PICK, replace=False))
        c = sum(1 for i in range(len(s)-1) if s[i+1] - s[i] == 1)
        sim_consec.append(c)
    sim_avg = np.mean(sim_consec)
    sim_dist = Counter(sim_consec)
    
    print(f"  Average consecutive pairs per draw: {avg_consec:.4f} (simulated: {sim_avg:.4f})")
    print(f"\n  Distribution:")
    for k in range(max(max(consec_counts), max(sim_consec)) + 1):
        obs = consec_dist.get(k, 0)
        exp = sim_dist.get(k, 0) / 100000 * N
        pct = obs / N * 100
        print(f"    {k} consecutive: {obs:5d} ({pct:5.1f}%) vs expected ~{exp:.1f}")

    z_consec = (avg_consec - sim_avg) / (np.std(consec_counts) / math.sqrt(N))
    print(f"  Z-test: z = {z_consec:.3f}")
    if abs(z_consec) > 2.0:
        findings.append((7, "Consecutive", f"Consecutive pairs z={z_consec:.3f}"))

    # ==========================================================
    # 13. DAY-OF-WEEK BIAS
    # ==========================================================
    print(f"\n{'─'*75}")
    print("  13. DAY-OF-WEEK ANALYSIS")
    print(f"{'─'*75}")

    from datetime import datetime
    day_freq = defaultdict(Counter)  # day -> number freq
    day_count = Counter()
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    for dt, d in zip(dates, data):
        try:
            dow = datetime.strptime(dt, '%Y-%m-%d').weekday()
            day_count[dow] += 1
            for n in d[:PICK]:
                day_freq[dow][n] += 1
        except:
            continue
    
    print(f"  Draw days: {', '.join(f'{day_names[d]}={day_count[d]}' for d in sorted(day_count))}")
    
    # Find numbers with strong day preference
    day_biases = []
    for dow in sorted(day_count):
        if day_count[dow] < 30:
            continue
        for n in range(1, MAX+1):
            day_rate = day_freq[dow].get(n, 0) / day_count[dow]
            overall_rate = freq.get(n, 0) / N
            if overall_rate > 0:
                z = (day_rate - overall_rate) / math.sqrt(overall_rate * (1 - overall_rate) / day_count[dow])
                if abs(z) > 2.5:
                    day_biases.append((n, dow, day_rate, overall_rate, z))
    
    if day_biases:
        day_biases.sort(key=lambda x: -abs(x[4]))
        print(f"\n  🚨 {len(day_biases)} number-day biases with |z| > 2.5:")
        for n, dow, dr, or_, z in day_biases[:10]:
            direction = "HIGHER" if z > 0 else "LOWER"
            print(f"    #{n:2d} on {day_names[dow]}: {dr:.1%} vs overall {or_:.1%}, z={z:+.2f} ({direction})")
            findings.append((7, "DayBias", f"#{n} on {day_names[dow]}: {dr:.1%} vs {or_:.1%}, z={z:+.2f}"))
    else:
        print(f"  ✅ No significant day-of-week biases")

    # ==========================================================
    # 14. BIRTHDAY SPACING TEST
    # ==========================================================
    print(f"\n{'─'*75}")
    print("  14. RANGE (MAX-MIN) ANALYSIS")
    print(f"{'─'*75}")

    ranges = [max(d[:PICK]) - min(d[:PICK]) for d in data]
    range_mean = np.mean(ranges)
    range_std = np.std(ranges)
    
    # Simulate expected
    sim_ranges = []
    for _ in range(100000):
        s = np.random.choice(range(1, MAX+1), PICK, replace=False)
        sim_ranges.append(max(s) - min(s))
    
    sim_range_mean = np.mean(sim_ranges)
    z_range = (range_mean - sim_range_mean) / (np.std(sim_ranges) / math.sqrt(N))
    print(f"  Mean range: {range_mean:.2f} (simulated: {sim_range_mean:.2f})")
    print(f"  Z-test: z = {z_range:.3f}")
    if abs(z_range) > 2.0:
        findings.append((6, "Range", f"Range z={z_range:.3f}"))

    # ==========================================================
    # 15. TRIPLET ANALYSIS
    # ==========================================================
    print(f"\n{'─'*75}")
    print("  15. MOST COMMON TRIPLETS")
    print(f"{'─'*75}")

    trip_freq = Counter()
    for d in data:
        for t in combinations(sorted(d[:PICK]), 3):
            trip_freq[t] += 1

    expected_trip = N * (PICK * (PICK-1) * (PICK-2)) / (MAX * (MAX-1) * (MAX-2))
    print(f"  Expected triplet frequency: {expected_trip:.3f}")
    
    top_trips = trip_freq.most_common(15)
    print(f"\n  Top 15 most frequent triplets:")
    for t, c in top_trips:
        z = (c - expected_trip) / math.sqrt(expected_trip)
        print(f"    ({t[0]:2d},{t[1]:2d},{t[2]:2d}): {c}x (z={z:+.2f})")
        if z > 3.0:
            findings.append((8, "Triplet", f"({t[0]},{t[1]},{t[2]}) appears {c}x, z={z:+.2f}"))

    # ==========================================================
    # FINAL SUMMARY — ALL FINDINGS
    # ==========================================================
    print(f"\n{'='*75}")
    print("  📋 FINAL SUMMARY — ALL ANOMALIES FOUND")
    print(f"{'='*75}")

    findings.sort(key=lambda x: -x[0])
    
    if not findings:
        print("\n  ✅ No significant anomalies found. Data appears random.")
    else:
        print(f"\n  Total anomalies: {len(findings)}")
        print(f"\n  🔴 CRITICAL (importance ≥ 9):")
        for imp, cat, desc in findings:
            if imp >= 9:
                print(f"    [{cat}] {desc}")
        
        print(f"\n  🟡 NOTABLE (importance 7-8):")
        for imp, cat, desc in findings:
            if 7 <= imp <= 8:
                print(f"    [{cat}] {desc}")
        
        print(f"\n  🔵 MINOR (importance < 7):")
        for imp, cat, desc in findings:
            if imp < 7:
                print(f"    [{cat}] {desc}")

    # Category summary
    cat_counts = Counter(cat for _, cat, _ in findings)
    print(f"\n  Anomalies by category:")
    for cat, cnt in cat_counts.most_common():
        print(f"    {cat}: {cnt}")

    print(f"\n{'='*75}")
    print(f"  ANALYSIS COMPLETE — {len(findings)} anomalies detected in {N} draws")
    print(f"{'='*75}")


if __name__ == '__main__':
    analyze()
