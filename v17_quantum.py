"""
V17 QUANTUM — INFORMATION-THEORETIC APPROACH
==============================================
Thay vì dự đoán SỐ → tập trung vào INFORMATION GAIN.

Insight: Xổ số 6/45 = 8,145,060 tổ hợp. Mỗi draw chọn 1.
Nếu ta có bất kỳ EDGE nào → nó phải hiện diện ở mức BIT.

APPROACH: Tính MUTUAL INFORMATION giữa:
  - Bit representation of draws
  - Lag features (multi-step)
  - Conditional entropy H(X_{t}|X_{t-1},...,X_{t-k})

Nếu H(X_t|past) < H(X_t) → CÓ pattern khai thác được.
Nếu H(X_t|past) = H(X_t) → XỔ SỐ LÀ RANDOM, STOP.

Đồng thời test: CONSTRAINT-BASED ELIMINATION
  - Thay vì chọn combo TỐT → LOẠI combo XẤU
  - Nếu ta loại được 80% combos VỚI chỉ 10% false negative
  → Từ 8M combos → 1.6M → cần 1.6M vé thay vì 8M
"""
import sys, os, math, json, time, random
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper.data_manager import get_mega645_numbers

MAX = 45; PICK = 6
random.seed(42); np.random.seed(42)


# ================================================================
# ENTROPY ANALYSIS
# ================================================================
def compute_entropy_analysis(data):
    """Compute conditional entropy to test if there's ANY predictable signal."""
    N = len(data)
    
    print("  🔬 ENTROPY ANALYSIS")
    print("  " + "─"*60)
    
    # 1. Per-number entropy
    # H(number_i in draw_t | number_i in draw_{t-1})
    print("\n  1️⃣  Per-number conditional entropy:")
    
    # Theoretical max: H(X) = -p*log(p) - (1-p)*log(1-p) where p = 6/45 = 0.133
    p = PICK/MAX
    H_max = -(p*math.log2(p) + (1-p)*math.log2(1-p))
    print(f"     H_max (unconditional) = {H_max:.4f} bits")
    
    # Conditional: P(in_t | in_{t-1}) vs P(in_t | out_{t-1})
    in_after_in = 0; in_after_out = 0
    total_in = 0; total_out = 0
    
    for i in range(1, N):
        prev = set(data[i-1][:PICK])
        curr = set(data[i][:PICK])
        for num in range(1, MAX+1):
            if num in prev:
                total_in += 1
                if num in curr: in_after_in += 1
            else:
                total_out += 1
                if num in curr: in_after_out += 1
    
    p_in_in = in_after_in / total_in
    p_in_out = in_after_out / total_out
    
    # Conditional entropy
    def h(p):
        if p <= 0 or p >= 1: return 0
        return -(p*math.log2(p) + (1-p)*math.log2(1-p))
    
    H_cond = (total_in/(total_in+total_out)) * h(p_in_in) + \
             (total_out/(total_in+total_out)) * h(p_in_out)
    
    MI = H_max - H_cond
    
    print(f"     P(in_t | in_{'{t-1}'}) = {p_in_in:.4f} (base={p:.4f})")
    print(f"     P(in_t | out_{'{t-1}'}) = {p_in_out:.4f}")
    print(f"     H(X|Y) = {H_cond:.4f} bits")
    print(f"     Mutual Info I(X;Y) = {MI:.6f} bits")
    print(f"     Info gain: {MI/H_max*100:.3f}%")
    
    # 2. Draw-level properties entropy
    print("\n  2️⃣  Draw SUM transitions:")
    sums = [sum(sorted(d[:PICK])) for d in data]
    sum_bins = [(s-50)//20 for s in sums]  # ~5 bins
    
    trans = defaultdict(Counter)
    for i in range(1, N):
        trans[sum_bins[i-1]][sum_bins[i]] += 1
    
    # Unconditional entropy of sum_bins
    sc = Counter(sum_bins)
    total = sum(sc.values())
    H_sum = -sum((c/total)*math.log2(c/total) for c in sc.values() if c > 0)
    
    # Conditional
    H_sum_cond = 0
    for state, follows in trans.items():
        n_s = sum(follows.values())
        h_s = -sum((c/n_s)*math.log2(c/n_s) for c in follows.values() if c > 0)
        H_sum_cond += (n_s/(N-1)) * h_s
    
    MI_sum = H_sum - H_sum_cond
    print(f"     H(sum_bin) = {H_sum:.4f} bits")
    print(f"     H(sum_bin|prev) = {H_sum_cond:.4f} bits")
    print(f"     MI = {MI_sum:.4f} bits ({MI_sum/H_sum*100:.2f}% of total)")
    
    # 3. Odd/Even
    print("\n  3️⃣  Odd count transitions:")
    odds = [sum(1 for x in d[:PICK] if x%2==1) for d in data]
    
    otrans = defaultdict(Counter)
    for i in range(1, N):
        otrans[odds[i-1]][odds[i]] += 1
    
    oc = Counter(odds); to = sum(oc.values())
    H_odd = -sum((c/to)*math.log2(c/to) for c in oc.values() if c > 0)
    
    H_odd_cond = 0
    for s, fs in otrans.items():
        ns = sum(fs.values())
        hs = -sum((c/ns)*math.log2(c/ns) for c in fs.values() if c > 0)
        H_odd_cond += (ns/(N-1)) * hs
    
    MI_odd = H_odd - H_odd_cond
    print(f"     H(odd_count) = {H_odd:.4f} bits")
    print(f"     H(odd|prev) = {H_odd_cond:.4f} bits")
    print(f"     MI = {MI_odd:.4f} bits ({MI_odd/H_odd*100:.2f}%)")
    
    # 4. Range
    print("\n  4️⃣  Range transitions:")
    ranges = [max(d[:PICK])-min(d[:PICK]) for d in data]
    range_bins = [r//5 for r in ranges]
    
    rtrans = defaultdict(Counter)
    for i in range(1, N):
        rtrans[range_bins[i-1]][range_bins[i]] += 1
    
    rc = Counter(range_bins); tr = sum(rc.values())
    H_range = -sum((c/tr)*math.log2(c/tr) for c in rc.values() if c > 0)
    
    H_range_cond = 0
    for s, fs in rtrans.items():
        ns = sum(fs.values())
        hs = -sum((c/ns)*math.log2(c/ns) for c in fs.values() if c > 0)
        H_range_cond += (ns/(N-1)) * hs
    
    MI_range = H_range - H_range_cond
    print(f"     H(range_bin) = {H_range:.4f} bits")
    print(f"     H(range|prev) = {H_range_cond:.4f} bits")
    print(f"     MI = {MI_range:.4f} bits ({MI_range/H_range*100:.2f}%)")
    
    # 5. Multi-lag analysis
    print("\n  5️⃣  Multi-lag MI for sum:")
    for lag in [1, 2, 3, 5, 10]:
        if lag >= N: break
        lag_trans = defaultdict(Counter)
        for i in range(lag, N):
            lag_trans[sum_bins[i-lag]][sum_bins[i]] += 1
        h_cond = 0
        for s, fs in lag_trans.items():
            ns = sum(fs.values())
            hs = -sum((c/ns)*math.log2(c/ns) for c in fs.values() if c > 0)
            h_cond += (ns/(N-lag)) * hs
        mi = H_sum - h_cond
        print(f"     Lag {lag:2d}: MI = {mi:.4f} bits ({mi/H_sum*100:.2f}%)")
    
    return {
        'per_number_MI': MI,
        'sum_MI': MI_sum,
        'odd_MI': MI_odd,
        'range_MI': MI_range,
        'H_max': H_max,
    }


# ================================================================
# CONSTRAINT-BASED ELIMINATION
# ================================================================
def constraint_elimination_test(data):
    """Test: can we ELIMINATE combos to shrink the search space?"""
    N = len(data)
    WARMUP = 200
    
    print("\n\n  🔬 CONSTRAINT ELIMINATION ANALYSIS")
    print("  " + "─"*60)
    
    # Learn constraints from history, test on future
    n_test = N - WARMUP
    total_combos = math.comb(MAX, PICK)  # 8,145,060
    
    elimination_rates = []
    false_negative_rates = []
    
    for ti in range(n_test):
        te = WARMUP + ti
        actual = sorted(data[te][:PICK])
        recent = data[max(0,te-100):te]
        
        # Compute constraints from recent 100 draws
        sums = [sum(sorted(d[:PICK])) for d in recent]
        ranges = [max(d[:PICK])-min(d[:PICK]) for d in recent]
        odds = [sum(1 for x in d[:PICK] if x%2==1) for d in recent]
        gaps_max = [max(sorted(d[:PICK])[i+1]-sorted(d[:PICK])[i] for i in range(PICK-1)) for d in recent]
        gaps_min = [min(sorted(d[:PICK])[i+1]-sorted(d[:PICK])[i] for i in range(PICK-1)) for d in recent]
        consec = [sum(1 for i in range(PICK-1) if sorted(d[:PICK])[i+1]-sorted(d[:PICK])[i]==1) for d in recent]
        
        # Conservative constraints (2-98 percentile)
        constraints = {
            'sum': (np.percentile(sums, 2), np.percentile(sums, 98)),
            'range': (np.percentile(ranges, 2), np.percentile(ranges, 98)),
            'odd': (np.percentile(odds, 2), np.percentile(odds, 98)),
            'gap_max': (np.percentile(gaps_max, 2), np.percentile(gaps_max, 98)),
            'consec': (0, np.percentile(consec, 98)),
        }
        
        # Check if actual draw passes constraints
        a_sum = sum(actual)
        a_range = actual[-1]-actual[0]
        a_odd = sum(1 for x in actual if x%2==1)
        a_gaps = [actual[i+1]-actual[i] for i in range(PICK-1)]
        a_gmax = max(a_gaps)
        a_consec = sum(1 for g in a_gaps if g==1)
        
        passes = (constraints['sum'][0] <= a_sum <= constraints['sum'][1] and
                  constraints['range'][0] <= a_range <= constraints['range'][1] and
                  constraints['odd'][0] <= a_odd <= constraints['odd'][1] and
                  constraints['gap_max'][0] <= a_gmax <= constraints['gap_max'][1] and
                  constraints['consec'][0] <= a_consec <= constraints['consec'][1])
        
        # Estimate elimination rate (how many combos are eliminated)
        # Sample 10000 random combos and check pass rate
        n_sample = 10000
        pass_count = 0
        for _ in range(n_sample):
            combo = sorted(random.sample(range(1,MAX+1), PICK))
            c_sum = sum(combo)
            c_range = combo[-1]-combo[0]
            c_odd = sum(1 for x in combo if x%2==1)
            c_gaps = [combo[i+1]-combo[i] for i in range(PICK-1)]
            c_gmax = max(c_gaps)
            c_consec = sum(1 for g in c_gaps if g==1)
            
            if (constraints['sum'][0] <= c_sum <= constraints['sum'][1] and
                constraints['range'][0] <= c_range <= constraints['range'][1] and
                constraints['odd'][0] <= c_odd <= constraints['odd'][1] and
                constraints['gap_max'][0] <= c_gmax <= constraints['gap_max'][1] and
                constraints['consec'][0] <= c_consec <= constraints['consec'][1]):
                pass_count += 1
        
        pass_rate = pass_count / n_sample
        elim_rate = 1 - pass_rate
        
        elimination_rates.append(elim_rate)
        false_negative_rates.append(0 if passes else 1)
    
    print(f"    Tests: {n_test}")
    print(f"    Avg elimination rate: {np.mean(elimination_rates)*100:.1f}% of all combos eliminated")
    print(f"    False negative rate: {np.mean(false_negative_rates)*100:.1f}% (actual draw eliminated)")
    
    # This tells us: using constraints, we keep X% of combos.
    # If false negative = 5%, we lose 5% of winning draws.
    # If we keep 20% of combos, we still need 20% × 8M = 1.6M tickets.
    
    kept_pct = 1 - np.mean(elimination_rates)
    fn_pct = np.mean(false_negative_rates)
    surviving_combos = int(total_combos * kept_pct)
    
    print(f"\n    📊 RESULT:")
    print(f"    Total combos C(45,6) = {total_combos:,}")
    print(f"    After constraints: {surviving_combos:,} ({kept_pct*100:.1f}%)")
    print(f"    False negatives: {fn_pct*100:.1f}%")
    print(f"    Effective search space reduction: {(1-kept_pct)*100:.1f}%")
    print(f"    Tickets needed for 100% coverage of kept space: {surviving_combos:,}")
    
    return {
        'elimination_rate': round(np.mean(elimination_rates), 4),
        'false_negative_rate': round(np.mean(false_negative_rates), 4),
        'surviving_combos': surviving_combos,
    }


# ================================================================
# AUTOCORRELATION TEST (is the sequence truly random?)
# ================================================================
def autocorrelation_test(data):
    """Test autocorrelation at multiple lags for each number."""
    N = len(data)
    
    print("\n\n  🔬 AUTOCORRELATION / RANDOMNESS TESTS")
    print("  " + "─"*60)
    
    # Binary time series for each number (1 = appeared, 0 = not)
    series = np.zeros((MAX, N))
    for i, d in enumerate(data):
        for num in d[:PICK]:
            series[num-1, i] = 1
    
    # Test autocorrelation at lags 1-10
    print("\n    Lag | Avg AutoCorr | Significant (#/45) | p<0.01")
    print("    " + "─"*60)
    
    significant_lags = {}
    for lag in range(1, 11):
        corrs = []
        n_sig = 0
        for num in range(MAX):
            s = series[num]
            if len(s) <= lag: continue
            x = s[:-lag]; y = s[lag:]
            if np.std(x) > 0 and np.std(y) > 0:
                r = np.corrcoef(x, y)[0, 1]
                corrs.append(r)
                # Significance test (null: r=0)
                z = r * math.sqrt(N-lag-2) / math.sqrt(1-r*r) if abs(r) < 1 else 0
                p = 2 * (1 - stats.norm.cdf(abs(z)))
                if p < 0.01:
                    n_sig += 1
        
        avg_corr = np.mean(corrs) if corrs else 0
        significant_lags[lag] = n_sig
        bar = '█' * n_sig
        print(f"    {lag:3d} | {avg_corr:+.6f}    | {n_sig:3d}/45 ({n_sig/45*100:.0f}%)     | {bar}")
    
    # Expected under null: 45 * 0.01 = 0.45 numbers would be significant by chance
    print(f"\n    Expected by chance (p<0.01): {45*0.01:.1f} numbers per lag")
    
    # Overall verdict
    max_sig = max(significant_lags.values())
    if max_sig > 5:
        print(f"    ⚡ VERDICT: Some autocorrelation detected! Max {max_sig} significant at some lag.")
    elif max_sig > 2:
        print(f"    ⚠️  VERDICT: Marginal autocorrelation. Max {max_sig} significant.")
    else:
        print(f"    ✅ VERDICT: No significant autocorrelation. Sequence is RANDOM.")
    
    # Runs test (for each number)
    print(f"\n    🎲 RUNS TEST (randomness of each number's appearances):")
    runs_pvals = []
    for num in range(MAX):
        s = series[num]
        runs = 1
        for i in range(1, N):
            if s[i] != s[i-1]: runs += 1
        n1 = int(sum(s)); n0 = N - n1
        if n1 == 0 or n0 == 0: continue
        exp_runs = 2*n1*n0/(n1+n0) + 1
        var_runs = 2*n1*n0*(2*n1*n0-n1-n0) / ((n1+n0)**2 * (n1+n0-1))
        if var_runs <= 0: continue
        z = (runs - exp_runs) / math.sqrt(var_runs)
        p = 2*(1-stats.norm.cdf(abs(z)))
        runs_pvals.append(p)
    
    n_sig_runs = sum(1 for p in runs_pvals if p < 0.01)
    print(f"    Numbers failing runs test (p<0.01): {n_sig_runs}/45")
    print(f"    Expected by chance: {45*0.01:.1f}")
    
    if n_sig_runs > 3:
        print(f"    ⚡ Some numbers show non-random patterns!")
    else:
        print(f"    ✅ All numbers pass randomness test.")
    
    return {
        'max_sig_autocorr': max_sig,
        'runs_test_failures': n_sig_runs,
    }


# ================================================================
# CHI-SQUARE UNIFORMITY
# ================================================================
def uniformity_test(data):
    """Test if numbers are uniformly distributed."""
    N = len(data)
    
    print(f"\n\n  🔬 UNIFORMITY TEST (Chi-Square)")
    print("  " + "─"*60)
    
    freq = Counter()
    for d in data:
        for num in d[:PICK]:
            freq[num] += 1
    
    expected = N * PICK / MAX
    chi2 = sum((freq.get(num,0) - expected)**2 / expected for num in range(1, MAX+1))
    df = MAX - 1
    p_val = 1 - stats.chi2.cdf(chi2, df)
    
    print(f"    Expected freq per number: {expected:.1f}")
    print(f"    Chi-square: {chi2:.2f} (df={df})")
    print(f"    p-value: {p_val:.4f}")
    
    if p_val < 0.01:
        print(f"    ⚡ SIGNIFICANT NON-UNIFORMITY DETECTED!")
        # Find most over/under-represented
        deviations = [(num, freq.get(num,0), (freq.get(num,0)-expected)/math.sqrt(expected)) 
                       for num in range(1, MAX+1)]
        deviations.sort(key=lambda x: -abs(x[2]))
        print(f"    Top deviated numbers:")
        for num, f, z in deviations[:10]:
            direction = "OVER" if z > 0 else "UNDER"
            print(f"      #{num:2d}: {f:5d} ({z:+.2f}σ) {direction}")
    elif p_val < 0.05:
        print(f"    ⚠️  Marginal non-uniformity.")
    else:
        print(f"    ✅ Distribution is uniform (no bias).")
    
    # Window analysis
    print(f"\n    📊 Window-based uniformity:")
    for w in [50, 100, 200, 500]:
        if w > N: break
        fail_count = 0
        for start in range(0, N-w, w//2):
            chunk = data[start:start+w]
            freq_w = Counter()
            for d in chunk:
                for num in d[:PICK]: freq_w[num] += 1
            exp_w = w * PICK / MAX
            chi2_w = sum((freq_w.get(num,0)-exp_w)**2/exp_w for num in range(1,MAX+1))
            p_w = 1 - stats.chi2.cdf(chi2_w, df)
            if p_w < 0.01: fail_count += 1
        total_windows = len(range(0, N-w, w//2))
        print(f"    Window {w:4d}: {fail_count}/{total_windows} windows fail (expected: {total_windows*0.01:.1f})")
    
    return chi2, p_val


def run():
    data = get_mega645_numbers()
    N = len(data)
    t0 = time.time()
    
    print("="*80)
    print("  🔬 V17 QUANTUM — INFORMATION-THEORETIC ANALYSIS")
    print(f"  {N} draws | Testing for ANY exploitable signal")
    print("="*80)
    
    # 1. Entropy analysis
    entropy = compute_entropy_analysis(data)
    
    # 2. Autocorrelation
    autocorr = autocorrelation_test(data)
    
    # 3. Uniformity
    chi2, p_val = uniformity_test(data)
    
    # 4. Constraint elimination
    constraints = constraint_elimination_test(data)
    
    elapsed = time.time() - t0
    
    print(f"\n{'═'*80}")
    print(f"  🎯 V17 QUANTUM — FINAL VERDICT ({elapsed:.0f}s)")
    print(f"{'═'*80}")
    
    print(f"\n  📊 SUMMARY OF ALL TESTS:")
    print(f"  {'Test':<30} | {'Result':>15} | {'Interpretation'}")
    print(f"  {'─'*30} | {'─'*15} | {'─'*30}")
    print(f"  {'Per-number MI':<30} | {entropy['per_number_MI']:.6f} bits | {'TINY signal' if entropy['per_number_MI'] > 0.001 else 'NO signal'}")
    print(f"  {'Sum MI':<30} | {entropy['sum_MI']:.4f} bits   | {'Useful' if entropy['sum_MI'] > 0.05 else 'Negligible'}")
    print(f"  {'Autocorrelation signif.':<30} | {autocorr['max_sig_autocorr']:>15d} | {'Pattern!' if autocorr['max_sig_autocorr'] > 5 else 'Random'}")
    print(f"  {'Runs test failures':<30} | {autocorr['runs_test_failures']:>15d} | {'Non-random!' if autocorr['runs_test_failures'] > 3 else 'Random'}")
    print(f"  {'Chi-square p':<30} | {p_val:>15.4f} | {'Non-uniform!' if p_val < 0.01 else 'Uniform'}")
    print(f"  {'Constraint elim rate':<30} | {constraints['elimination_rate']*100:>14.1f}% | Search space: {constraints['surviving_combos']:,}")
    print(f"  {'Constraint false neg':<30} | {constraints['false_negative_rate']*100:>14.1f}% | Accuracy of elimination")
    
    # ULTIMATE CONCLUSION
    has_signal = (entropy['per_number_MI'] > 0.005 or 
                  autocorr['max_sig_autocorr'] > 5 or
                  autocorr['runs_test_failures'] > 5 or
                  p_val < 0.01)
    
    if has_signal:
        print(f"\n  ⚡ CONCLUSION: EXPLOITABLE SIGNALS EXIST!")
        print(f"  → Further optimization CAN improve 6/6 rate beyond random.")
    else:
        print(f"\n  ✅ CONCLUSION: Lottery is STATISTICALLY RANDOM.")
        print(f"  → No algorithm can consistently predict 6/6 above random chance.")
        print(f"  → Maximum achievable 6/6 rate = N_tickets / C(45,6) = N / 8,145,060")
    
    # Save
    output = {
        'version': '17.0 QUANTUM',
        'entropy': entropy,
        'autocorr': autocorr,
        'chi_square': {'chi2': round(chi2,2), 'p_val': round(p_val,4)},
        'constraints': constraints,
        'has_signal': has_signal,
        'elapsed': round(elapsed,1),
    }
    path = os.path.join(os.path.dirname(__file__), 'models', 'v17_quantum.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {path}")


if __name__ == '__main__':
    run()
