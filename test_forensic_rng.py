"""
FORENSIC RNG INVESTIGATION — THÁM TỬ MODE
============================================
Không đoán số. Điều tra HỆ THỐNG.
Tìm lỗ hổng trong RNG / máy quay Vietlott.

Tests:
  1. Chi-square: Mỗi số có xuất hiện đúng tần suất kỳ vọng?
  2. Runs test: Dãy xuất hiện/không xuất hiện có quá đều/lộn xộn?
  3. Serial correlation: Draw liên tiếp có liên quan?
  4. Gap test: Khoảng cách giữa 2 lần xuất hiện có đúng phân bố?
  5. Birthday spacing: Khoảng cách có cluster bất thường?
  6. Position bias: Số nào "thích" slot nào?
  7. Adjacent pair anomaly: Cặp số liền kề có xuất hiện quá nhiều/ít?
  8. Even/odd, High/low deviation
  9. Sum distribution vs theoretical
  10. Day-of-week / temporal bias
  11. Consecutive appearance streaks
  12. Number "heat map" drift over time
"""
import sys, time, os, warnings
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
import numpy as np
from collections import Counter, defaultdict
from scipy import stats
from scraper.data_manager import get_mega645_numbers, get_mega645_all

data = get_mega645_numbers()
all_records = get_mega645_all()
total = len(data)
MAX_NUM = 45
PICK = 6
sorted_draws = [sorted(d[:6]) for d in data]
flat_all = [num for draw in data for num in draw[:6]]

# Try to get dates
try:
    dates = [r.get('draw_date', '') for r in all_records]
except:
    dates = [''] * total

print(f"Data: {total} draws, {len(flat_all)} numbers drawn")
print(f"{'='*90}")
print(f" FORENSIC RNG INVESTIGATION")
print(f" 'Nếu hệ thống có lỗ hổng, data sẽ TỰ NÓI'")
print(f"{'='*90}\n")

FINDINGS = []  # Collect all anomalies

# ================================================================
# TEST 1: CHI-SQUARE — Per-Number Frequency
# H0: All numbers equally likely (P = 6/45 each)
# ================================================================
print(f"  TEST 1: CHI-SQUARE — Number Frequency Bias")
print(f"  {'='*60}\n")

expected_per_num = total * PICK / MAX_NUM
num_counts = Counter(flat_all)
chi2_stat = 0
deviations = {}
for num in range(1, MAX_NUM+1):
    observed = num_counts.get(num, 0)
    deviation = (observed - expected_per_num) / np.sqrt(expected_per_num)
    deviations[num] = deviation
    chi2_stat += (observed - expected_per_num)**2 / expected_per_num

chi2_p = 1 - stats.chi2.cdf(chi2_stat, df=MAX_NUM-1)
print(f"  Expected per number: {expected_per_num:.1f}")
print(f"  Chi-square stat: {chi2_stat:.2f} (df={MAX_NUM-1})")
print(f"  P-value: {chi2_p:.6f}")
if chi2_p < 0.05:
    print(f"  >>> ANOMALY DETECTED! Distribution is NOT uniform (p<0.05)")
    FINDINGS.append(("CHI-SQUARE", f"p={chi2_p:.6f}", "Number distribution NOT uniform"))
else:
    print(f"  Result: Distribution appears uniform (p>0.05)")

# Most/least frequent
sorted_dev = sorted(deviations.items(), key=lambda x: -x[1])
print(f"\n  Most OVER-represented (hot):")
for num, dev in sorted_dev[:5]:
    cnt = num_counts[num]
    print(f"    #{num:>2}: {cnt} times (expected {expected_per_num:.0f}, "
          f"deviation = {dev:+.2f}σ)")

print(f"  Most UNDER-represented (cold):")
for num, dev in sorted_dev[-5:]:
    cnt = num_counts[num]
    print(f"    #{num:>2}: {cnt} times (expected {expected_per_num:.0f}, "
          f"deviation = {dev:+.2f}σ)")

# Per-number chi-square (is any INDIVIDUAL number significantly biased?)
print(f"\n  Individual number significance (|z| > 2 = suspicious):")
suspicious_nums = []
for num, dev in sorted_dev:
    if abs(dev) > 2.0:
        cnt = num_counts[num]
        direction = "HOT" if dev > 0 else "COLD"
        print(f"    #{num:>2}: z={dev:+.2f} ({direction}) — {cnt} vs {expected_per_num:.0f}")
        suspicious_nums.append((num, dev))
        FINDINGS.append(("NUM_BIAS", f"#{num} z={dev:+.2f}", f"{direction} number"))

if not suspicious_nums:
    print(f"    None found — all numbers within ±2σ")

# ================================================================
# TEST 2: RUNS TEST — Sequential Randomness
# For each number: sequence of appear/not-appear → runs test
# ================================================================
print(f"\n  TEST 2: RUNS TEST — Sequential Randomness")
print(f"  {'='*60}\n")

runs_anomalies = []
for num in range(1, MAX_NUM+1):
    # Binary sequence: 1 if num in draw, 0 otherwise
    seq = [1 if num in draw else 0 for draw in data]
    n1 = sum(seq)
    n0 = len(seq) - n1

    if n1 == 0 or n0 == 0: continue

    # Count runs
    runs = 1
    for i in range(1, len(seq)):
        if seq[i] != seq[i-1]:
            runs += 1

    # Expected runs
    exp_runs = 1 + 2*n1*n0 / (n1+n0)
    var_runs = 2*n1*n0*(2*n1*n0 - n1 - n0) / ((n1+n0)**2 * (n1+n0-1))
    if var_runs <= 0: continue
    z_runs = (runs - exp_runs) / np.sqrt(var_runs)

    if abs(z_runs) > 2.5:
        direction = "CLUSTERED" if z_runs < 0 else "ALTERNATING"
        runs_anomalies.append((num, z_runs, direction, runs, exp_runs))

if runs_anomalies:
    runs_anomalies.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"  ANOMALIES (|z| > 2.5):")
    for num, z, direction, actual, expected in runs_anomalies[:10]:
        print(f"    #{num:>2}: z={z:+.2f} ({direction}) — {actual} runs vs {expected:.0f} expected")
        FINDINGS.append(("RUNS", f"#{num} z={z:+.2f}", f"Appearances are {direction}"))
else:
    print(f"  No anomalies — all numbers pass runs test")

# ================================================================
# TEST 3: SERIAL CORRELATION — Consecutive Draw Correlation
# ================================================================
print(f"\n  TEST 3: SERIAL CORRELATION — Draw-to-Draw")
print(f"  {'='*60}\n")

# Test: how many numbers repeat from draw N to N+1?
repeats = [len(set(data[i]) & set(data[i+1])) for i in range(total-1)]
avg_repeat = np.mean(repeats)
expected_repeat = PICK * PICK / MAX_NUM  # ~0.8
print(f"  Avg numbers repeating: {avg_repeat:.3f} (expected ~{expected_repeat:.3f})")
repeat_z = (avg_repeat - expected_repeat) / (np.std(repeats) / np.sqrt(len(repeats)))
print(f"  Z-score: {repeat_z:.3f}")
if abs(repeat_z) > 2:
    direction = "MORE" if repeat_z > 0 else "FEWER"
    print(f"  >>> ANOMALY: {direction} repeats than random!")
    FINDINGS.append(("SERIAL_REPEAT", f"z={repeat_z:.2f}", f"{direction} consecutive repeats"))

# Sum correlation
sums = [sum(d[:6]) for d in data]
sum_corr = np.corrcoef(sums[:-1], sums[1:])[0,1]
print(f"\n  Sum autocorrelation (lag-1): {sum_corr:.4f}")
if abs(sum_corr) > 0.05:
    print(f"  >>> ANOMALY: Consecutive sums are correlated!")
    FINDINGS.append(("SUM_CORR", f"r={sum_corr:.4f}", "Consecutive draw sums correlated"))

# Per-number autocorrelation
print(f"\n  Per-number lag-1 autocorrelation (top anomalies):")
num_autocorr = []
for num in range(1, MAX_NUM+1):
    seq = [1 if num in draw else 0 for draw in data]
    if np.std(seq) == 0: continue
    r = np.corrcoef(seq[:-1], seq[1:])[0,1]
    num_autocorr.append((num, r))

num_autocorr.sort(key=lambda x: abs(x[1]), reverse=True)
for num, r in num_autocorr[:5]:
    marker = " >>> SUSPICIOUS" if abs(r) > 0.05 else ""
    print(f"    #{num:>2}: r={r:+.4f}{marker}")
    if abs(r) > 0.05:
        FINDINGS.append(("NUM_AUTOCORR", f"#{num} r={r:+.4f}", "Draw-to-draw autocorrelation"))

# ================================================================
# TEST 4: GAP DISTRIBUTION — Per Number
# Time between appearances should follow geometric distribution
# ================================================================
print(f"\n  TEST 4: GAP DISTRIBUTION — Per Number")
print(f"  {'='*60}\n")

expected_gap = MAX_NUM / PICK  # ~7.5 draws
all_gaps = []
gap_anomalies = []

for num in range(1, MAX_NUM+1):
    appearances = [i for i, draw in enumerate(data) if num in draw]
    if len(appearances) < 10: continue
    gaps = [appearances[i+1]-appearances[i] for i in range(len(appearances)-1)]
    all_gaps.extend(gaps)

    avg_gap = np.mean(gaps)
    # Expected: geometric(p=6/45)
    exp = MAX_NUM / PICK
    # Test if gap distribution matches geometric
    # KS test against geometric CDF
    p = PICK / MAX_NUM
    ks_stat, ks_p = stats.kstest(gaps, lambda x: 1-(1-p)**x)

    if ks_p < 0.01:  # Highly significant
        gap_anomalies.append((num, avg_gap, ks_p, len(gaps)))

print(f"  Expected avg gap: {expected_gap:.1f} draws")
print(f"  Actual avg gap: {np.mean(all_gaps):.2f} draws")

if gap_anomalies:
    gap_anomalies.sort(key=lambda x: x[2])
    print(f"\n  GAP ANOMALIES (KS test p < 0.01):")
    for num, ag, p, n in gap_anomalies[:10]:
        print(f"    #{num:>2}: avg_gap={ag:.1f}, KS p={p:.6f} (n={n})")
        FINDINGS.append(("GAP", f"#{num} p={p:.6f}", f"Gap distribution abnormal (avg={ag:.1f})"))
else:
    print(f"  No anomalies — all gap distributions match geometric")

# ================================================================
# TEST 5: POSITION BIAS — Does number X prefer position Y?
# ================================================================
print(f"\n  TEST 5: POSITION BIAS — Number-Slot Preference")
print(f"  {'='*60}\n")

pos_counts = defaultdict(Counter)  # pos_counts[pos][num] = count
for sd in sorted_draws:
    for pos in range(6):
        pos_counts[pos][sd[pos]] += 1

# For each number, test if its position distribution is non-uniform
# (given its frequency)
pos_anomalies = []
for num in range(1, MAX_NUM+1):
    pos_dist = [pos_counts[pos].get(num, 0) for pos in range(6)]
    total_appearances = sum(pos_dist)
    if total_appearances < 20: continue

    # Expected: position should match the number's natural range
    # But if the DRAW is random, within-draw ordering is deterministic
    # So this test checks if the number appears at unexpected positions
    # Chi-square within the observed positions
    non_zero = [(p, c) for p, c in enumerate(pos_dist) if c > 0]
    if len(non_zero) <= 1: continue

    # Check if the number appears in unexpected positions
    # E.g., number 45 should ONLY be in positions 1-6 (sorted), mainly pos 5-6
    # A bias would be if it appears in pos it shouldn't
    expected_uniform = total_appearances / len(non_zero)
    chi2 = sum((c - expected_uniform)**2 / expected_uniform for _, c in non_zero)
    p_val = 1 - stats.chi2.cdf(chi2, df=len(non_zero)-1)

    if p_val < 0.001 and len(non_zero) >= 3:
        dominant_pos = max(non_zero, key=lambda x: x[1])
        pos_anomalies.append((num, chi2, p_val, dominant_pos, total_appearances))

if pos_anomalies:
    pos_anomalies.sort(key=lambda x: x[2])
    print(f"  Position bias found (p < 0.001):")
    for num, chi2, p, (dpos, dcnt), tot in pos_anomalies[:10]:
        print(f"    #{num:>2}: chi2={chi2:.1f} p={p:.6f}, "
              f"dominant pos={dpos+1} ({dcnt}/{tot} = {dcnt/tot*100:.0f}%)")
else:
    print(f"  No unexpected position biases")

# ================================================================
# TEST 6: ADJACENT PAIR FREQUENCY
# ================================================================
print(f"\n  TEST 6: ADJACENT PAIR ANOMALY")
print(f"  {'='*60}\n")

pair_counts = Counter()
for draw in data:
    for a, b in zip(sorted(draw[:6]), sorted(draw[:6])[1:]):
        if b - a == 1:
            pair_counts[(a,b)] += 1

# How many adjacent pairs per draw?
adj_per_draw = [sum(1 for a,b in zip(sorted(d[:6]), sorted(d[:6])[1:]) if b-a==1)
                for d in data]
avg_adj = np.mean(adj_per_draw)

# Theoretical: P(adjacent pair in 6 from 45) via simulation
sim_adj = []
for _ in range(50000):
    s = sorted(np.random.choice(range(1, MAX_NUM+1), PICK, replace=False))
    sim_adj.append(sum(1 for a,b in zip(s, s[1:]) if b-a==1))
exp_adj = np.mean(sim_adj)

z_adj = (avg_adj - exp_adj) / (np.std(adj_per_draw) / np.sqrt(total))
print(f"  Avg adjacent pairs per draw: {avg_adj:.3f}")
print(f"  Expected (simulation): {exp_adj:.3f}")
print(f"  Z-score: {z_adj:.2f}")
if abs(z_adj) > 2:
    direction = "MORE" if z_adj > 0 else "FEWER"
    print(f"  >>> ANOMALY: {direction} adjacent pairs than random!")
    FINDINGS.append(("ADJ_PAIRS", f"z={z_adj:.2f}", f"{direction} adjacent pairs"))

# Most common adjacent pairs
print(f"\n  Top adjacent pairs:")
for pair, cnt in pair_counts.most_common(10):
    exp_cnt = total * (PICK/MAX_NUM) * ((PICK-1)/(MAX_NUM-1))  # Approx
    print(f"    {pair}: {cnt} times")

# ================================================================
# TEST 7: EVEN/ODD and HIGH/LOW DISTRIBUTION
# ================================================================
print(f"\n  TEST 7: EVEN/ODD and HIGH/LOW DISTRIBUTION")
print(f"  {'='*60}\n")

even_counts = [sum(1 for n in d[:6] if n % 2 == 0) for d in data]
avg_even = np.mean(even_counts)
exp_even = PICK * 22/45  # 22 even, 23 odd in 1-45
z_even = (avg_even - exp_even) / (np.std(even_counts) / np.sqrt(total))
print(f"  Avg even numbers per draw: {avg_even:.3f} (expected {exp_even:.3f})")
print(f"  Z-score: {z_even:.2f}")
if abs(z_even) > 2:
    FINDINGS.append(("EVEN_ODD", f"z={z_even:.2f}", "Even/odd ratio biased"))

high_counts = [sum(1 for n in d[:6] if n > 22) for d in data]
avg_high = np.mean(high_counts)
exp_high = PICK * 23/45
z_high = (avg_high - exp_high) / (np.std(high_counts) / np.sqrt(total))
print(f"  Avg high(>22) numbers: {avg_high:.3f} (expected {exp_high:.3f})")
print(f"  Z-score: {z_high:.2f}")
if abs(z_high) > 2:
    FINDINGS.append(("HIGH_LOW", f"z={z_high:.2f}", "High/low ratio biased"))

# Even/odd pattern transition
eo_pattern = [sum(1 for n in d[:6] if n % 2 == 0) for d in data]
eo_corr = np.corrcoef(eo_pattern[:-1], eo_pattern[1:])[0,1]
print(f"  Even-count autocorrelation: {eo_corr:.4f}")
if abs(eo_corr) > 0.05:
    FINDINGS.append(("EO_AUTOCORR", f"r={eo_corr:.4f}", "Even/odd pattern autocorrelated"))

# ================================================================
# TEST 8: SUM DISTRIBUTION
# ================================================================
print(f"\n  TEST 8: SUM DISTRIBUTION")
print(f"  {'='*60}\n")

sums = [sum(d[:6]) for d in data]
exp_sum = PICK * (MAX_NUM+1) / 2  # = 138
print(f"  Avg sum: {np.mean(sums):.2f} (expected {exp_sum:.1f})")
print(f"  Std sum: {np.std(sums):.2f}")

# KS test against simulated distribution
sim_sums = [sum(sorted(np.random.choice(range(1,46), 6, replace=False)))
            for _ in range(50000)]
ks_stat, ks_p = stats.ks_2samp(sums, sim_sums)
print(f"  KS test vs simulation: stat={ks_stat:.4f}, p={ks_p:.4f}")
if ks_p < 0.05:
    FINDINGS.append(("SUM_DIST", f"KS p={ks_p:.4f}", "Sum distribution abnormal"))

# ================================================================
# TEST 9: TEMPORAL PATTERNS — Day of Week
# ================================================================
print(f"\n  TEST 9: TEMPORAL PATTERNS")
print(f"  {'='*60}\n")

if dates and dates[0]:
    from datetime import datetime
    day_sums = defaultdict(list)
    day_counts = Counter()
    for i, date_str in enumerate(dates):
        try:
            if isinstance(date_str, str) and date_str:
                dt = datetime.strptime(date_str[:10], '%Y-%m-%d')
                dow = dt.strftime('%A')
                day_counts[dow] += 1
                day_sums[dow].extend(data[i][:6])
        except:
            pass

    if day_counts:
        print(f"  Draw distribution by day:")
        for day, cnt in day_counts.most_common():
            print(f"    {day}: {cnt} draws")

        # Per-day number frequency
        print(f"\n  Per-day number bias (top deviations):")
        overall_freq = Counter(flat_all)
        for day in day_counts:
            day_freq = Counter(day_sums[day])
            n_draws = day_counts[day]
            for num in range(1, MAX_NUM+1):
                obs = day_freq.get(num, 0)
                exp = n_draws * PICK / MAX_NUM
                if exp > 5:
                    z = (obs - exp) / np.sqrt(exp)
                    if abs(z) > 3:
                        print(f"    {day} #{num}: z={z:+.2f} ({obs} vs {exp:.0f})")
                        FINDINGS.append(("DAY_BIAS", f"{day} #{num} z={z:+.2f}",
                                        "Day-specific number bias"))
    else:
        print(f"  No date data available")
else:
    print(f"  No date data available")

# ================================================================
# TEST 10: STREAK ANOMALIES
# ================================================================
print(f"\n  TEST 10: CONSECUTIVE APPEARANCE STREAKS")
print(f"  {'='*60}\n")

# For each number, find longest streak of consecutive draws it appeared in
streak_anomalies = []
for num in range(1, MAX_NUM+1):
    seq = [1 if num in draw else 0 for draw in data]
    max_streak = 0
    curr = 0
    for s in seq:
        if s == 1:
            curr += 1
            max_streak = max(max_streak, curr)
        else:
            curr = 0

    # Expected max streak: P(appear) = 6/45 ≈ 0.133
    # P(streak of k) ≈ n * p^k
    p = PICK / MAX_NUM
    # P(max streak >= k) ≈ 1 - (1-p^k)^n ≈ n*p^k for small p^k
    expected_max = 1
    for k in range(1, 20):
        if total * p**k < 0.5:
            expected_max = k - 1
            break

    if max_streak > expected_max + 2:
        streak_anomalies.append((num, max_streak, expected_max))

if streak_anomalies:
    streak_anomalies.sort(key=lambda x: -x[1])
    print(f"  Expected max streak: ~{expected_max} draws")
    print(f"  ANOMALOUS streaks (> expected + 2):")
    for num, streak, exp in streak_anomalies[:10]:
        print(f"    #{num:>2}: {streak} consecutive draws (expected max ~{exp})")
        FINDINGS.append(("STREAK", f"#{num} streak={streak}", f"Abnormally long streak"))
else:
    print(f"  No anomalous streaks found")

# ================================================================
# TEST 11: BIRTHDAY SPACING TEST
# ================================================================
print(f"\n  TEST 11: BIRTHDAY SPACING")
print(f"  {'='*60}\n")

# Within each draw, check spacing between sorted numbers
all_spacings = []
for sd in sorted_draws:
    spacings = [sd[i+1]-sd[i] for i in range(5)]
    all_spacings.extend(spacings)

# Simulate expected spacing distribution
sim_spacings = []
for _ in range(50000):
    s = sorted(np.random.choice(range(1,46), 6, replace=False))
    sim_spacings.extend([s[i+1]-s[i] for i in range(5)])

ks_stat, ks_p = stats.ks_2samp(all_spacings, sim_spacings)
print(f"  Spacing KS test: stat={ks_stat:.4f}, p={ks_p:.4f}")
if ks_p < 0.05:
    FINDINGS.append(("SPACING", f"KS p={ks_p:.4f}", "Intra-draw spacing abnormal"))

print(f"  Actual spacing distribution: mean={np.mean(all_spacings):.2f}, "
      f"std={np.std(all_spacings):.2f}")
print(f"  Simulated spacing: mean={np.mean(sim_spacings):.2f}, "
      f"std={np.std(sim_spacings):.2f}")

# ================================================================
# TEST 12: TIME-EVOLUTION — Is the RNG drifting?
# ================================================================
print(f"\n  TEST 12: RNG DRIFT OVER TIME")
print(f"  {'='*60}\n")

# Split data into 4 quarters
q_size = total // 4
for metric_name, metric_fn in [
    ("Average sum", lambda d: np.mean([sum(x[:6]) for x in d])),
    ("Avg number", lambda d: np.mean([n for x in d for n in x[:6]])),
    ("Std number", lambda d: np.std([n for x in d for n in x[:6]])),
    ("Avg adjacent pairs", lambda d: np.mean([sum(1 for a,b in zip(sorted(x[:6]),sorted(x[:6])[1:]) if b-a==1) for x in d])),
]:
    quarters = []
    for q in range(4):
        chunk = data[q*q_size:(q+1)*q_size]
        quarters.append(metric_fn(chunk))
    trend = np.polyfit(range(4), quarters, 1)[0]
    vals = " → ".join(f"{v:.2f}" for v in quarters)
    drift = "DRIFTING" if abs(trend) > 0.5 else "stable"
    print(f"  {metric_name}: {vals} (trend={trend:+.3f}) [{drift}]")
    if abs(trend) > 0.5:
        FINDINGS.append(("DRIFT", f"{metric_name} trend={trend:+.3f}", "RNG drifting over time"))

# ================================================================
# TEST 13: MONOBIT TEST (NIST-style)
# ================================================================
print(f"\n  TEST 13: NIST-STYLE MONOBIT")
print(f"  {'='*60}\n")

# Convert draws to binary string: for each draw, each number 1-45 is 0 or 1
bits = []
for draw in data:
    draw_set = set(draw[:6])
    for num in range(1, MAX_NUM+1):
        bits.append(1 if num in draw_set else 0)

# Monobit: count of 1s should be ~n/2
n_bits = len(bits)
n_ones = sum(bits)
exp_ones = n_bits * PICK / MAX_NUM
s_obs = abs(n_ones - exp_ones) / np.sqrt(n_bits * (PICK/MAX_NUM) * (1-PICK/MAX_NUM))
p_mono = 2 * (1 - stats.norm.cdf(s_obs))
print(f"  Total bits: {n_bits}, ones: {n_ones} (expected {exp_ones:.0f})")
print(f"  Monobit z: {s_obs:.4f}, p: {p_mono:.6f}")
if p_mono < 0.01:
    FINDINGS.append(("MONOBIT", f"p={p_mono:.6f}", "NIST monobit test FAILED"))

# ================================================================
# FINAL REPORT: ALL FINDINGS
# ================================================================
print(f"\n{'='*90}")
print(f" FORENSIC FINDINGS — {len(FINDINGS)} anomalies detected")
print(f"{'='*90}\n")

if FINDINGS:
    for i, (test, detail, desc) in enumerate(FINDINGS):
        severity = "🔴" if "FAIL" in desc.upper() or "ANOMAL" in desc.upper() else "🟡"
        print(f"  {severity} [{test}] {desc}")
        print(f"     Detail: {detail}")
        print()

    # Categorize
    critical = [f for f in FINDINGS if any(k in f[0] for k in ['CHI','MONOBIT','SERIAL'])]
    moderate = [f for f in FINDINGS if f not in critical]

    print(f"  Summary:")
    print(f"    Critical anomalies: {len(critical)}")
    print(f"    Moderate anomalies: {len(moderate)}")
    print(f"    Total: {len(FINDINGS)}")

    if critical:
        print(f"\n  >>> CRITICAL FINDINGS — POTENTIALLY EXPLOITABLE:")
        for test, detail, desc in critical:
            print(f"      [{test}] {desc}: {detail}")
else:
    print(f"  NO anomalies detected. RNG appears cryptographically random.")

print(f"\n{'='*90}")
print(f" NEXT STEP: If anomalies found, build targeted exploit predictor")
print(f"{'='*90}")
