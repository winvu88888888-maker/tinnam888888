# -*- coding: utf-8 -*-
"""
DEEP MINE V8 — ULTRA DEEP — TẦNG CUỐI CÙNG
=============================================
Markov bậc 2, Hurst exponent, compound exclusion, comeback analysis,
parity transitions, variance ratio, multi-lag AC, draw entropy, v.v.
"""
import sys, os, math
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scraper.data_manager import get_mega645_all, get_power655_all
from collections import Counter, defaultdict
from datetime import datetime

OUTPUT = os.path.join(os.path.dirname(__file__), 'deep_mine_v8_output.txt')

class Tee:
    def __init__(self, fp):
        self.f = open(fp, 'w', encoding='utf-8')
        self.out = sys.stdout
    def write(self, t):
        self.out.write(t)
        self.f.write(t)
    def flush(self):
        self.out.flush()
        self.f.flush()
    def close(self):
        self.f.close()

tee = Tee(OUTPUT)
sys.stdout = tee

mega = get_mega645_all()
nums = [[r['n1'],r['n2'],r['n3'],r['n4'],r['n5'],r['n6']] for r in mega]
N = len(nums)

power = get_power655_all()
pnums = [[r['n1'],r['n2'],r['n3'],r['n4'],r['n5'],r['n6']] for r in power]

def h(title, em="🔬"):
    print(f"\n{'═'*90}")
    print(f"  {em} {title}")
    print(f"{'═'*90}")

def sh(title):
    print(f"\n  {'─'*80}")
    print(f"  ▸ {title}")
    print(f"  {'─'*80}")

print("="*90)
print(f"  🔬 DEEP MINE V8 — ULTRA DEEP — Mega 6/45 + Power 6/55")
print(f"  Mega: {N} kỳ | Power: {len(pnums)} kỳ | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("="*90)

# ================================================================
# 1. HURST EXPONENT — Kiểm tra memory dài hạn
# ================================================================
sh("1. HURST EXPONENT — Memory dài hạn (R/S Analysis)")

def hurst_rs(series, max_k=None):
    """Estimate Hurst exponent via R/S analysis"""
    n = len(series)
    if max_k is None:
        max_k = n // 4
    ks = []
    rs_vals = []
    for k in [16, 32, 64, 128, 256, 512]:
        if k > max_k or k > n // 2:
            break
        rs_list = []
        for start in range(0, n - k, k):
            sub = series[start:start+k]
            mean_s = sum(sub) / k
            devs = [x - mean_s for x in sub]
            cum = []
            s = 0
            for d in devs:
                s += d
                cum.append(s)
            R = max(cum) - min(cum)
            S = (sum(d*d for d in devs) / k) ** 0.5
            if S > 0:
                rs_list.append(R / S)
        if rs_list:
            avg_rs = sum(rs_list) / len(rs_list)
            ks.append(math.log(k))
            rs_vals.append(math.log(avg_rs))
    
    if len(ks) >= 2:
        # Linear regression
        n_pts = len(ks)
        sx = sum(ks)
        sy = sum(rs_vals)
        sxy = sum(x*y for x,y in zip(ks, rs_vals))
        sx2 = sum(x*x for x in ks)
        H = (n_pts * sxy - sx * sy) / (n_pts * sx2 - sx * sx)
        return H
    return 0.5

for col in range(6):
    series = [nums[i][col] for i in range(N)]
    H = hurst_rs(series)
    interpretation = "PERSISTENT (trending)" if H > 0.55 else "ANTI-PERSISTENT (mean-reverting)" if H < 0.45 else "RANDOM WALK"
    marker = "📡" if abs(H - 0.5) > 0.05 else ""
    print(f"  C{col+1}: H = {H:.3f} → {interpretation} {marker}")

# Sum series
sum_series = [sum(d) for d in nums]
H_sum = hurst_rs(sum_series)
print(f"  SUM: H = {H_sum:.3f} → {'PERSISTENT' if H_sum > 0.55 else 'ANTI-PERSISTENT' if H_sum < 0.45 else 'RANDOM'}")

# ================================================================
# 2. COMPOUND EXCLUSION — C1+C6 cùng lúc → loại trừ mạnh hơn
# ================================================================
sh("2. COMPOUND EXCLUSION — Biết C1 VÀ C6 → Loại trừ C2-C5")

# Most common C1, C6 combos
c1c6_counter = Counter((d[0], d[5]) for d in nums)
print(f"  Top 10 tổ hợp C1+C6 phổ biến nhất:")
for (c1, c6), cnt in c1c6_counter.most_common(10):
    pct = cnt / N * 100
    # What's excluded for C2-C5?
    draws_matching = [d for d in nums if d[0] == c1 and d[5] == c6]
    c2_set = set(d[1] for d in draws_matching)
    c3_set = set(d[2] for d in draws_matching)
    c4_set = set(d[3] for d in draws_matching)
    c5_set = set(d[4] for d in draws_matching)
    
    c2_excl = len(set(range(c1+1, c6)) - c2_set)
    c3_excl = len(set(range(c1+1, c6)) - c3_set)
    
    print(f"    C1={c1:2d}, C6={c6:2d}: {cnt:3d}x ({pct:.1f}%) | C2 seen: {len(c2_set)}, C3 seen: {len(c3_set)}, C4 seen: {len(c4_set)}")
    
    if cnt >= 15:
        # Show actual ranges
        c2_min, c2_max = min(c2_set), max(c2_set)
        c3_min, c3_max = min(c3_set), max(c3_set)
        c4_min, c4_max = min(c4_set), max(c4_set)
        c5_min, c5_max = min(c5_set), max(c5_set)
        print(f"           C2∈[{c2_min}-{c2_max}], C3∈[{c3_min}-{c3_max}], C4∈[{c4_min}-{c4_max}], C5∈[{c5_min}-{c5_max}]")

# ================================================================
# 3. PARITY TRANSITION — Pattern chẵn/lẻ giữa các kỳ
# ================================================================
sh("3. PARITY TRANSITION — Biến đổi chẵn/lẻ giữa các kỳ")

# Count odd numbers pattern
odd_counts = [sum(1 for n in d if n % 2 == 1) for d in nums]
# Transition matrix
parity_trans = defaultdict(Counter)
for i in range(1, N):
    parity_trans[odd_counts[i-1]][odd_counts[i]] += 1

print(f"  Ma trận chuyển tiếp (số lẻ kỳ T → kỳ T+1):")
print(f"  {'T\\T+1':>6}", end="")
for j in range(7):
    print(f"  {j}lẻ", end="")
print()
print(f"  {'─'*55}")

for i in range(7):
    total = sum(parity_trans[i].values())
    if total < 5:
        continue
    print(f"  {i}lẻ  ", end="")
    for j in range(7):
        cnt = parity_trans[i].get(j, 0)
        pct = cnt / total * 100 if total > 0 else 0
        if pct > 0:
            print(f" {pct:4.0f}%", end="")
        else:
            print(f"    -", end="")
    print(f"  (n={total})")

# ================================================================
# 4. COMEBACK ANALYSIS — Số quay lại sau khi vắng lâu
# ================================================================
sh("4. COMEBACK ANALYSIS — Số quay lại sau vắng lâu, ở bao lâu?")

print(f"  Khi số vắng >15 kỳ rồi quay lại, nó 'ở' bao nhiêu kỳ tiếp?\n")

comeback_stats = {}
for n in range(1, 46):
    appearances = [i for i, d in enumerate(nums) if n in d]
    if len(appearances) < 5:
        continue
    
    comebacks = []
    for idx in range(1, len(appearances)):
        gap = appearances[idx] - appearances[idx-1]
        if gap > 15:
            # Count how many draws it appears in the next 10
            stay = 0
            for k in range(1, 11):
                if appearances[idx] + k < N and n in nums[appearances[idx] + k]:
                    stay += 1
                else:
                    break
            comebacks.append((gap, stay))
    
    if comebacks:
        avg_stay = sum(s for _, s in comebacks) / len(comebacks)
        avg_gap = sum(g for g, _ in comebacks) / len(comebacks)
        comeback_stats[n] = (len(comebacks), avg_gap, avg_stay)

sorted_stats = sorted(comeback_stats.items(), key=lambda x: -x[1][2])
print(f"  {'Số':>4} {'Lần CB':>8} {'Gap TB':>8} {'Stay TB':>8} {'Nhận xét':>15}")
print(f"  {'─'*48}")
for n, (cnt, ag, ast) in sorted_stats[:15]:
    label = "HAY Ở LẠI" if ast > 1.5 else "ĐẾN RỒI ĐI" if ast < 0.5 else ""
    print(f"  {n:4d} {cnt:8d} {ag:8.1f} {ast:8.2f} {label:>15}")

# ================================================================
# 5. MULTI-LAG AUTOCORRELATION — AC cho nhiều lag
# ================================================================
sh("5. MULTI-LAG AUTOCORRELATION — Tìm chu kỳ ẩn trong từng cột")

for col in range(6):
    series = [nums[i][col] for i in range(N)]
    mean = sum(series) / N
    var = sum((x - mean)**2 for x in series) / N
    
    sig_lags = []
    for lag in range(1, 21):
        if var == 0:
            break
        ac = sum((series[i]-mean)*(series[i-lag]-mean) for i in range(lag, N)) / (N * var)
        threshold = 2 / N**0.5  # ~95% significance
        if abs(ac) > threshold:
            sig_lags.append((lag, ac))
    
    if sig_lags:
        sig_str = ", ".join(f"lag{l}={a:+.3f}" for l, a in sig_lags[:5])
        print(f"  C{col+1}: 📡 Significant lags: {sig_str}")
    else:
        print(f"  C{col+1}: ✅ No significant autocorrelation at lags 1-20")

# ================================================================
# 6. DRAW ENTROPY — Mỗi kỳ "spread" bao nhiêu?
# ================================================================
sh("6. DRAW ENTROPY — Mức 'spread' của mỗi kỳ quay")

def draw_entropy(draw, max_num=45):
    """Measure how 'spread out' a draw is using spacing entropy"""
    spaces = [draw[i+1] - draw[i] for i in range(len(draw)-1)]
    spaces = [draw[0]] + spaces + [max_num - draw[-1]]
    total = sum(spaces)
    if total == 0:
        return 0
    probs = [s/total for s in spaces if s > 0]
    return -sum(p * math.log2(p) for p in probs)

entropies = [draw_entropy(d) for d in nums]
avg_ent = sum(entropies) / len(entropies)
recent_ent = sum(entropies[-50:]) / 50

print(f"  Draw entropy trung bình: {avg_ent:.3f}")
print(f"  Draw entropy 50 kỳ gần: {recent_ent:.3f} ({'↑ TĂNG' if recent_ent > avg_ent + 0.05 else '↓ GIẢM' if recent_ent < avg_ent - 0.05 else '= ỔN ĐỊNH'})")

# After high entropy draw, what next?
high_ent_next = [entropies[i] for i in range(1, len(entropies)) if entropies[i-1] > avg_ent + 0.3]
low_ent_next = [entropies[i] for i in range(1, len(entropies)) if entropies[i-1] < avg_ent - 0.3]

if high_ent_next:
    print(f"  Sau high-entropy draw: avg next = {sum(high_ent_next)/len(high_ent_next):.3f} (n={len(high_ent_next)})")
if low_ent_next:
    print(f"  Sau low-entropy draw:  avg next = {sum(low_ent_next)/len(low_ent_next):.3f} (n={len(low_ent_next)})")

# ================================================================
# 7. VARIANCE RATIO TEST — Lo-MacKinlay
# ================================================================
sh("7. VARIANCE RATIO TEST — Kiểm tra random walk")

for col in range(6):
    series = [nums[i][col] for i in range(N)]
    
    for q in [2, 4, 8]:
        # Variance ratio VR(q)
        # Var of q-period returns / (q * Var of 1-period returns)
        returns_1 = [series[i] - series[i-1] for i in range(1, N)]
        returns_q = [series[i] - series[i-q] for i in range(q, N)]
        
        var_1 = sum(r*r for r in returns_1) / len(returns_1)
        var_q = sum(r*r for r in returns_q) / len(returns_q)
        
        if var_1 > 0:
            VR = var_q / (q * var_1)
            z = (VR - 1) * (N * q)**0.5
            sig = "🚨" if abs(VR - 1) > 0.1 else ""
            if q == 2:
                print(f"  C{col+1}: VR(2)={VR:.3f} VR(4)=", end="")
            elif q == 4:
                print(f"{VR:.3f} VR(8)=", end="")
            else:
                print(f"{VR:.3f} {sig}")

# ================================================================
# 8. WEEK NUMBER PATTERN
# ================================================================
sh("8. WEEK NUMBER — Tuần trong năm có ảnh hưởng?")

from datetime import datetime as dt
week_sums = defaultdict(list)
for row, draw in zip(mega, nums):
    d = dt.strptime(row['draw_date'], '%Y-%m-%d')
    week = d.isocalendar()[1]
    week_sums[week].append(sum(draw))

print(f"  Tuần có tổng CAO nhất và THẤP nhất:")
week_avgs = {w: sum(s)/len(s) for w, s in week_sums.items() if len(s) > 5}
sorted_weeks = sorted(week_avgs.items(), key=lambda x: -x[1])

print(f"\n  TOP 5 tuần tổng CAO:")
for w, avg in sorted_weeks[:5]:
    print(f"    Tuần {w:2d}: avg sum = {avg:.1f} (n={len(week_sums[w])})")

print(f"\n  TOP 5 tuần tổng THẤP:")
for w, avg in sorted_weeks[-5:]:
    print(f"    Tuần {w:2d}: avg sum = {avg:.1f} (n={len(week_sums[w])})")

# ================================================================
# 9. SUM DIFFERENCE PATTERN
# ================================================================
sh("9. SUM DIFFERENCE — Chênh lệch tổng giữa kỳ liên tiếp")

sum_diffs = [sum(nums[i]) - sum(nums[i-1]) for i in range(1, N)]
avg_diff = sum(sum_diffs) / len(sum_diffs)
std_diff = (sum((d-avg_diff)**2 for d in sum_diffs)/len(sum_diffs))**0.5

print(f"  Avg sum diff: {avg_diff:+.2f}")
print(f"  Std sum diff: {std_diff:.1f}")

# Distribution of sum differences
diff_buckets = Counter()
for d in sum_diffs:
    b = (d // 20) * 20
    diff_buckets[b] += 1

print(f"\n  Phân bố chênh lệch tổng:")
for b in sorted(diff_buckets.keys()):
    pct = diff_buckets[b] / len(sum_diffs) * 100
    bar = '█' * int(pct / 2)
    print(f"    [{b:+4d} to {b+19:+4d}]: {diff_buckets[b]:4d} ({pct:5.1f}%) {bar}")

# After large positive diff
big_pos = [sum_diffs[i] for i in range(1, len(sum_diffs)) if sum_diffs[i-1] > 50]
big_neg = [sum_diffs[i] for i in range(1, len(sum_diffs)) if sum_diffs[i-1] < -50]
if big_pos:
    print(f"\n  Sau tổng TĂNG >50: avg next diff = {sum(big_pos)/len(big_pos):+.1f} (n={len(big_pos)})")
if big_neg:
    print(f"  Sau tổng GIẢM <-50: avg next diff = {sum(big_neg)/len(big_neg):+.1f} (n={len(big_neg)})")

# ================================================================
# 10. COLUMN MEMORY T-2 — Giá trị 2 kỳ trước có giúp?
# ================================================================
sh("10. COLUMN MEMORY T-2 — Biết kỳ T-2 có giúp dự đoán T?")

for col in range(6):
    # Compare: P(C_col = X | C_col(T-1)) vs P(C_col = X | C_col(T-1), C_col(T-2))
    exact_t1 = 0
    exact_t2 = 0
    near_t1 = 0
    near_t2 = 0
    
    for i in range(2, N):
        # Using T-1 only (predict = T-1 value)
        pred_t1 = nums[i-1][col]
        if nums[i][col] == pred_t1:
            exact_t1 += 1
        if abs(nums[i][col] - pred_t1) <= 2:
            near_t1 += 1
        
        # Using T-1 and T-2 (predict = 2*T-1 - T-2, i.e. linear extrapolation)
        pred_t2 = 2 * nums[i-1][col] - nums[i-2][col]
        pred_t2 = max(1, min(45, pred_t2))
        if nums[i][col] == pred_t2:
            exact_t2 += 1
        if abs(nums[i][col] - pred_t2) <= 2:
            near_t2 += 1
    
    total = N - 2
    print(f"  C{col+1}: Repeat(T-1) exact={exact_t1/total*100:.1f}% ±2={near_t1/total*100:.1f}% | Extrap(T-2) exact={exact_t2/total*100:.1f}% ±2={near_t2/total*100:.1f}%")

# ================================================================
# 11. GEOMETRIC MEAN PATTERN
# ================================================================
sh("11. GEOMETRIC MEAN — Trung bình nhân của 6 số")

geo_means = []
for d in nums:
    gm = 1
    for n in d:
        gm *= n
    gm = gm ** (1/6)
    geo_means.append(gm)

avg_gm = sum(geo_means) / len(geo_means)
recent_gm = sum(geo_means[-50:]) / 50

print(f"  Geometric mean TB: {avg_gm:.2f}")
print(f"  Geometric mean 50 kỳ gần: {recent_gm:.2f}")

# After high/low GM
sorted_gm = sorted(enumerate(geo_means), key=lambda x: x[1])
bottom_10pct = set(i for i, _ in sorted_gm[:N//10])
top_10pct = set(i for i, _ in sorted_gm[-N//10:])

after_low_gm = [geo_means[i+1] for i in bottom_10pct if i+1 < N]
after_high_gm = [geo_means[i+1] for i in top_10pct if i+1 < N]

if after_low_gm:
    print(f"  Sau GM thấp (bottom 10%): avg next GM = {sum(after_low_gm)/len(after_low_gm):.2f}")
if after_high_gm:
    print(f"  Sau GM cao (top 10%):     avg next GM = {sum(after_high_gm)/len(after_high_gm):.2f}")

# ================================================================
# 12. YEAR-OVER-YEAR STABILITY
# ================================================================
sh("12. YEAR-OVER-YEAR — Số nào ổn định qua từng năm?")

year_freq = defaultdict(lambda: defaultdict(int))
year_counts = Counter()
for row, draw in zip(mega, nums):
    y = int(row['draw_date'][:4])
    year_counts[y] += 1
    for n in draw:
        year_freq[y][n] += 1

years = sorted(year_counts.keys())
if len(years) >= 3:
    # Calculate coefficient of variation (CV) for each number across years
    cv_scores = {}
    for n in range(1, 46):
        freqs = [year_freq[y].get(n, 0) / year_counts[y] * 100 for y in years if year_counts[y] > 50]
        if len(freqs) >= 3:
            mean_f = sum(freqs) / len(freqs)
            std_f = (sum((f - mean_f)**2 for f in freqs) / len(freqs)) ** 0.5
            cv = std_f / mean_f if mean_f > 0 else 0
            cv_scores[n] = (cv, mean_f, min(freqs), max(freqs))
    
    print(f"  Số ỔN ĐỊNH nhất qua các năm (CV thấp = ổn định):")
    for n, (cv, mf, mn, mx) in sorted(cv_scores.items(), key=lambda x: x[1][0])[:10]:
        print(f"    Số {n:2d}: CV={cv:.3f}, avg={mf:.1f}%, range=[{mn:.1f}%-{mx:.1f}%]")
    
    print(f"\n  Số BIẾN ĐỘNG nhất qua các năm (CV cao):")
    for n, (cv, mf, mn, mx) in sorted(cv_scores.items(), key=lambda x: -x[1][0])[:10]:
        print(f"    Số {n:2d}: CV={cv:.3f}, avg={mf:.1f}%, range=[{mn:.1f}%-{mx:.1f}%]")

# ================================================================
# 13. DRAW SIMILARITY DECAY
# ================================================================
sh("13. DRAW SIMILARITY DECAY — Kỳ cách bao xa thì giống nhau?")

print(f"  Overlap trung bình theo khoảng cách kỳ:")
for gap in [1, 2, 3, 5, 7, 10, 15, 20, 50, 100]:
    overlaps = []
    for i in range(gap, N):
        ov = len(set(nums[i]) & set(nums[i-gap]))
        overlaps.append(ov)
    avg_ov = sum(overlaps) / len(overlaps)
    expected = 6 * 6 / 45  # 0.8
    ratio = avg_ov / expected
    bar = '█' * int(ratio * 20)
    print(f"    Gap={gap:3d}: overlap={avg_ov:.3f} (ratio={ratio:.3f}) {bar}")

# ================================================================
# 14. MINI-STREAK WITHIN COLUMN
# ================================================================
sh("14. MINI-STREAK — Chuỗi tăng/giảm liên tục trong cột")

for col in range(6):
    up_streaks = []
    dn_streaks = []
    curr_up = 0
    curr_dn = 0
    
    for i in range(1, N):
        if nums[i][col] > nums[i-1][col]:
            curr_up += 1
            if curr_dn > 0:
                dn_streaks.append(curr_dn)
            curr_dn = 0
        elif nums[i][col] < nums[i-1][col]:
            curr_dn += 1
            if curr_up > 0:
                up_streaks.append(curr_up)
            curr_up = 0
        else:
            if curr_up > 0: up_streaks.append(curr_up)
            if curr_dn > 0: dn_streaks.append(curr_dn)
            curr_up = curr_dn = 0
    
    max_up = max(up_streaks) if up_streaks else 0
    max_dn = max(dn_streaks) if dn_streaks else 0
    avg_up = sum(up_streaks) / len(up_streaks) if up_streaks else 0
    avg_dn = sum(dn_streaks) / len(dn_streaks) if dn_streaks else 0
    
    print(f"  C{col+1}: max_up={max_up} (avg={avg_up:.1f}), max_dn={max_dn} (avg={avg_dn:.1f})")

# ================================================================
# 15. SUM QUARTILE → NEXT DRAW PATTERN
# ================================================================
sh("15. SUM QUARTILE — Tổng thuộc nhóm nào → kỳ sau thế nào?")

sums = [sum(d) for d in nums]
sorted_sums = sorted(sums)
q1 = sorted_sums[N//4]
q2 = sorted_sums[N//2]
q3 = sorted_sums[3*N//4]

print(f"  Q1={q1}, Q2(median)={q2}, Q3={q3}\n")

for label, lo, hi in [("Q1 (low sum)", 0, q1), ("Q2", q1, q2), ("Q3", q2, q3), ("Q4 (high sum)", q3, 999)]:
    next_sums = [sums[i+1] for i in range(N-1) if lo <= sums[i] < hi]
    next_ranges = [nums[i+1][-1] - nums[i+1][0] for i in range(N-1) if lo <= sums[i] < hi]
    
    if next_sums:
        avg_next = sum(next_sums) / len(next_sums)
        avg_range = sum(next_ranges) / len(next_ranges)
        print(f"  Sau {label}: avg next sum={avg_next:.1f}, avg next range={avg_range:.1f} (n={len(next_sums)})")

# ================================================================
# 16. POWER 6/55 — DEEP ADDITIONAL ANALYSIS
# ================================================================
sh("16. POWER 6/55 — Hurst + Compound Exclusion + Parity")

PN = len(pnums)

# Hurst for Power
for col in range(6):
    series = [pnums[i][col] for i in range(PN)]
    H = hurst_rs(series)
    marker = "📡" if abs(H - 0.5) > 0.05 else ""
    interp = "PERSISTENT" if H > 0.55 else "ANTI-PERSISTENT" if H < 0.45 else "RANDOM"
    print(f"  Power C{col+1}: H = {H:.3f} → {interp} {marker}")

# Power parity
p_odd_counts = [sum(1 for n in d if n % 2 == 1) for d in pnums]
p_odd_counter = Counter(p_odd_counts)
print(f"\n  Power chẵn/lẻ distribution:")
for k in sorted(p_odd_counter.keys()):
    pct = p_odd_counter[k] / PN * 100
    print(f"    {k} lẻ: {p_odd_counter[k]:4d} ({pct:5.1f}%)")

# Power compound exclusion top
p_c1c6 = Counter((d[0], d[5]) for d in pnums)
print(f"\n  Power C1+C6 combos:")
for (c1, c6), cnt in p_c1c6.most_common(5):
    draws = [d for d in pnums if d[0] == c1 and d[5] == c6]
    c3r = [d[2] for d in draws]
    print(f"    C1={c1:2d}, C6={c6:2d}: {cnt}x | C3 range=[{min(c3r)}-{max(c3r)}]")

# ================================================================
# 17. LAST 10 DRAWS — Pattern đặc biệt?
# ================================================================
sh("17. LAST 10 DRAWS — Phân tích 10 kỳ gần nhất")

print(f"\n  10 kỳ Mega 6/45 gần nhất:")
for i in range(max(0, N-10), N):
    d = nums[i]
    s = sum(d)
    r = d[-1] - d[0]
    odd = sum(1 for n in d if n % 2 == 1)
    consec = sum(1 for j in range(1, len(d)) if d[j] == d[j-1]+1)
    ov = len(set(nums[i]) & set(nums[i-1])) if i > 0 else 0
    print(f"    Kỳ {i+1} ({mega[i]['draw_date']}): {d}  sum={s} range={r} odd={odd} consec={consec} overlap={ov}")

# ================================================================
# SUMMARY
# ================================================================
h("TỔNG KẾT — PHÁT HIỆN MỚI V8", "🏆")

print("""
  PHÁT HIỆN MỚI TỪ DEEP MINE V8:

  1. HURST EXPONENT: Cột nào có memory dài hạn (trending/mean-reverting)?
  2. COMPOUND EXCLUSION: Biết C1+C6 → thu hẹp C2-C5 thêm nữa
  3. PARITY TRANSITION: Ma trận chuyển đổi chẵn/lẻ giữa các kỳ
  4. COMEBACK ANALYSIS: Số quay lại sau vắng lâu "ở" bao lâu?
  5. MULTI-LAG AC: Tìm chu kỳ ẩn ở lag xa hơn (lag 1-20)
  6. DRAW ENTROPY: Mức "trải" của mỗi kỳ có pattern?
  7. VARIANCE RATIO TEST: Kiểm tra random walk chính thức
  8. WEEK NUMBER: Tuần nào trong năm may mắn?
  9. SUM DIFFERENCE: Pattern chênh lệch tổng
  10. COLUMN MEMORY T-2: Biết T-2 có giúp dự đoán T?
  11. GEOMETRIC MEAN: Pattern trung bình nhân
  12. YEAR-OVER-YEAR: Số nào ổn định nhất qua nhiều NĂM?
  13. SIMILARITY DECAY: Overlap giảm theo khoảng cách thế nào?
  14. MINI-STREAK: Chuỗi tăng/giảm liên tục trong cột
  15. SUM QUARTILE: Tổng Q1/Q4 → kỳ sau khác nhau?
  16. POWER DEEP: Hurst + patterns cho Power 6/55
  17. LAST 10 DRAWS: Phân tích chi tiết 10 kỳ gần nhất
""")

print(f"\n{'='*90}")
print(f"  📄 Output: {OUTPUT}")
print(f"{'='*90}")

sys.stdout = tee.out
tee.close()
