# -*- coding: utf-8 -*-
"""
DEEP MINE V9 — EXTREME DEPTH — Mọi khía cạnh còn lại
======================================================
Fourier/spectral, mutual information, gap fitting, number temperature,
draw clustering, chi² per number, internal spacing, block analysis,
modular clock, cross-lag, lag-1 conditional, entropy rate, etc.
"""
import sys, os, math, random
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scraper.data_manager import get_mega645_all, get_power655_all
from collections import Counter, defaultdict
from itertools import combinations

OUTPUT = os.path.join(os.path.dirname(__file__), 'deep_mine_v9_output.txt')

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

def sh(title):
    print(f"\n  {'─'*80}")
    print(f"  ▸ {title}")
    print(f"  {'─'*80}")

print("="*90)
print(f"  🔬 DEEP MINE V9 — EXTREME DEPTH — Mega 6/45 + Power 6/55")
print(f"  Mega: {N} kỳ | Power: {len(pnums)} kỳ")
print("="*90)

# ================================================================
# 1. CHI² PER NUMBER — Số nào lệch nhiều nhất so với kỳ vọng?
# ================================================================
sh("1. CHI² PER NUMBER — Số nào lệch chuẩn nhiều nhất?")

freq = Counter(n for d in nums for n in d)
expected = N * 6 / 45  # ~199.1

print(f"  Kỳ vọng mỗi số: {expected:.1f} lần\n")
print(f"  {'Số':>4} {'Actual':>8} {'Expected':>9} {'Chi²':>7} {'Lệch%':>7} {'Status':>8}")
print(f"  {'─'*48}")

chi_scores = []
for n in range(1, 46):
    actual = freq.get(n, 0)
    chi2 = (actual - expected)**2 / expected
    pct = (actual - expected) / expected * 100
    chi_scores.append((n, actual, chi2, pct))

chi_scores.sort(key=lambda x: -x[2])
for n, actual, chi2, pct in chi_scores[:10]:
    status = "🔥 HIGH" if actual > expected else "❄️ LOW"
    print(f"  {n:4d} {actual:8d} {expected:9.1f} {chi2:7.2f} {pct:+6.1f}% {status}")

print(f"\n  Bottom 5 (gần kỳ vọng nhất):")
for n, actual, chi2, pct in chi_scores[-5:]:
    print(f"  {n:4d} {actual:8d} {expected:9.1f} {chi2:7.2f} {pct:+6.1f}%")

total_chi2 = sum(c for _, _, c, _ in chi_scores)
print(f"\n  Total Chi²(44 df) = {total_chi2:.1f} (threshold ~60 for p=0.05)")

# ================================================================
# 2. GAP DISTRIBUTION FITTING — Gap có theo geometric distribution?
# ================================================================
sh("2. GAP DISTRIBUTION — Gap tuân theo phân bố nào?")

for target in [7, 19, 24, 38, 44]:
    appearances = [i for i, d in enumerate(nums) if target in d]
    gaps = [appearances[i] - appearances[i-1] for i in range(1, len(appearances))]
    
    if len(gaps) < 10:
        continue
    
    avg_gap = sum(gaps) / len(gaps)
    p_geo = 1 / avg_gap  # geometric parameter
    
    # Compare actual gap distribution with geometric
    gap_counter = Counter(gaps)
    max_gap = max(gaps)
    
    # KS-like test: compare CDF
    actual_cdf = []
    geo_cdf = []
    for g in range(1, min(max_gap+1, 30)):
        actual_cnt = sum(1 for x in gaps if x <= g)
        actual_cdf.append(actual_cnt / len(gaps))
        geo_cdf.append(1 - (1 - p_geo)**g)
    
    max_diff = max(abs(a - g) for a, g in zip(actual_cdf, geo_cdf))
    
    # Variance test: geometric has var = (1-p)/p²
    actual_var = sum((g - avg_gap)**2 for g in gaps) / len(gaps)
    geo_var = (1 - p_geo) / (p_geo ** 2)
    var_ratio = actual_var / geo_var if geo_var > 0 else 0
    
    fit = "✅ GOOD FIT" if max_diff < 0.1 else "⚠️ POOR FIT"
    print(f"  Số {target:2d}: avg_gap={avg_gap:.1f}, p={p_geo:.3f}, KS_diff={max_diff:.3f}, var_ratio={var_ratio:.2f} {fit}")

# ================================================================
# 3. INTERNAL SPACING — Khoảng cách TRONG mỗi kỳ quay
# ================================================================
sh("3. INTERNAL SPACING — Khoảng cách giữa các số trong 1 kỳ")

# For each draw, compute spacing pattern
spacing_patterns = []
for d in nums:
    spaces = [d[i+1] - d[i] for i in range(5)]
    spacing_patterns.append(tuple(spaces))

# Most common spacing patterns
sp_counter = Counter(spacing_patterns)
print(f"  Top 10 spacing patterns phổ biến nhất (khoảng cách giữa C1→C2→...→C6):")
for pattern, cnt in sp_counter.most_common(10):
    pct = cnt / N * 100
    print(f"    {pattern}: {cnt}x ({pct:.2f}%)")

# Average spacing per position
print(f"\n  Trung bình khoảng cách:")
for pos in range(5):
    spaces = [nums[i][pos+1] - nums[i][pos] for i in range(N)]
    avg = sum(spaces) / len(spaces)
    mode = Counter(spaces).most_common(1)[0]
    print(f"    C{pos+1}→C{pos+2}: avg={avg:.1f}, mode={mode[0]}({mode[1]}x)")

# Min spacing analysis
min_spaces = [min(d[i+1]-d[i] for i in range(5)) for d in nums]
min_sp_counter = Counter(min_spaces)
print(f"\n  Min spacing (khoảng cách nhỏ nhất trong kỳ):")
for sp in sorted(min_sp_counter.keys())[:8]:
    print(f"    min_gap={sp}: {min_sp_counter[sp]}x ({min_sp_counter[sp]/N*100:.1f}%)")

# ================================================================
# 4. NUMBER TEMPERATURE — Điểm "nóng" theo thời gian
# ================================================================
sh("4. NUMBER TEMPERATURE — Trọng số nóng/lạnh thời gian thực")

def calc_temperature(numbers_list, num, decay=0.95):
    """Calculate temperature: recent appearances weighted more"""
    temp = 0
    n = len(numbers_list)
    for i in range(n-1, max(n-100, -1), -1):
        if num in numbers_list[i]:
            temp += decay ** (n - 1 - i)
    return temp

print(f"  Temperature hiện tại (decay=0.95, 100 kỳ gần nhất):\n")
temps = [(n, calc_temperature(nums, n)) for n in range(1, 46)]
temps.sort(key=lambda x: -x[1])

print(f"  🔥 TOP 10 NÓNG NHẤT:")
for n, t in temps[:10]:
    bar = '█' * int(t * 5)
    print(f"    Số {n:2d}: temp={t:.3f} {bar}")

print(f"\n  ❄️ TOP 10 LẠNH NHẤT:")
for n, t in temps[-10:]:
    bar = '░' * int(t * 5)
    print(f"    Số {n:2d}: temp={t:.3f} {bar}")

# ================================================================
# 5. FOURIER / SPECTRAL ANALYSIS — Tìm chu kỳ trong tổng
# ================================================================
sh("5. FOURIER ANALYSIS — Phát hiện chu kỳ ẩn trong tổng")

sums = [sum(d) for d in nums]
mean_sum = sum(sums) / N

# Simple DFT for key frequencies
print(f"  Phân tích phổ tần số cho chuỗi SUM (N={N}):\n")
top_freqs = []

for period in range(2, 60):
    # Compute power at this frequency
    cos_sum = sum((sums[i] - mean_sum) * math.cos(2 * math.pi * i / period) for i in range(N))
    sin_sum = sum((sums[i] - mean_sum) * math.sin(2 * math.pi * i / period) for i in range(N))
    power = (cos_sum**2 + sin_sum**2) / N
    top_freqs.append((period, power))

top_freqs.sort(key=lambda x: -x[1])
print(f"  {'Period':>7} {'Power':>10} {'Strength':>10}")
print(f"  {'─'*30}")
for period, power in top_freqs[:15]:
    bar = '█' * int(power / top_freqs[0][1] * 20)
    print(f"  {period:7d} {power:10.0f} {bar}")

# Check if top period is significant
avg_power = sum(p for _, p in top_freqs) / len(top_freqs)
max_power = top_freqs[0][1]
sig_ratio = max_power / avg_power
print(f"\n  Max/Avg power ratio: {sig_ratio:.2f}")
if sig_ratio > 3:
    print(f"  📡 SIGNIFICANT PERIODICITY at period={top_freqs[0][0]}!")
else:
    print(f"  ✅ No dominant periodicity detected")

# Per column Fourier
sh("5b. FOURIER PER COLUMN — Chu kỳ ẩn từng cột")
for col in range(6):
    series = [nums[i][col] for i in range(N)]
    mean_s = sum(series) / N
    
    best_period = 2
    best_power = 0
    for period in range(2, 50):
        cos_s = sum((series[i] - mean_s) * math.cos(2*math.pi*i/period) for i in range(N))
        sin_s = sum((series[i] - mean_s) * math.sin(2*math.pi*i/period) for i in range(N))
        pw = (cos_s**2 + sin_s**2) / N
        if pw > best_power:
            best_power = pw
            best_period = period
    
    # Average power
    total_pow = 0
    for period in range(2, 50):
        cos_s = sum((series[i]-mean_s)*math.cos(2*math.pi*i/period) for i in range(N))
        sin_s = sum((series[i]-mean_s)*math.sin(2*math.pi*i/period) for i in range(N))
        total_pow += (cos_s**2 + sin_s**2) / N
    avg_pow = total_pow / 48
    ratio = best_power / avg_pow if avg_pow > 0 else 0
    sig = "📡" if ratio > 2 else ""
    print(f"  C{col+1}: best period={best_period}, power_ratio={ratio:.2f} {sig}")

# ================================================================
# 6. MUTUAL INFORMATION — Mối quan hệ phi tuyến giữa cột
# ================================================================
sh("6. MUTUAL INFORMATION — Quan hệ phi tuyến giữa các cột")

def mutual_info(x_series, y_series, bins=10):
    """Estimate MI between two series"""
    n = len(x_series)
    x_min, x_max = min(x_series), max(x_series)
    y_min, y_max = min(y_series), max(y_series)
    
    x_edges = [x_min + i*(x_max-x_min)/bins for i in range(bins+1)]
    y_edges = [y_min + i*(y_max-y_min)/bins for i in range(bins+1)]
    
    joint = defaultdict(int)
    x_margin = defaultdict(int)
    y_margin = defaultdict(int)
    
    for xi, yi in zip(x_series, y_series):
        xb = min(int((xi - x_min) / (x_max - x_min + 0.001) * bins), bins-1)
        yb = min(int((yi - y_min) / (y_max - y_min + 0.001) * bins), bins-1)
        joint[(xb, yb)] += 1
        x_margin[xb] += 1
        y_margin[yb] += 1
    
    mi = 0
    for (xb, yb), cnt in joint.items():
        p_xy = cnt / n
        p_x = x_margin[xb] / n
        p_y = y_margin[yb] / n
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * math.log2(p_xy / (p_x * p_y))
    return mi

# Same draw MI
print(f"  MI giữa các cột CÙNG kỳ (cao = phụ thuộc mạnh):\n")
for ci in range(6):
    for cj in range(ci+1, 6):
        x = [d[ci] for d in nums]
        y = [d[cj] for d in nums]
        mi = mutual_info(x, y)
        sig = "🔗" if mi > 0.3 else ""
        print(f"    C{ci+1}↔C{cj+1}: MI={mi:.4f} {sig}")

# Cross-draw MI
print(f"\n  MI giữa cột kỳ T và kỳ T+1:")
for col in range(6):
    x = [nums[i][col] for i in range(N-1)]
    y = [nums[i+1][col] for i in range(N-1)]
    mi = mutual_info(x, y)
    print(f"    C{col+1}(T)↔C{col+1}(T+1): MI={mi:.4f}")

# ================================================================
# 7. BLOCK ANALYSIS — Thống kê theo khối 100 kỳ
# ================================================================
sh("7. BLOCK ANALYSIS — So sánh thống kê theo khối 100 kỳ")

block_size = 100
n_blocks = N // block_size
print(f"  {n_blocks} blocks of {block_size} draws each\n")

print(f"  {'Block':>6} {'Sum':>7} {'Range':>7} {'Odd%':>6} {'Consec%':>8} {'Unique':>7}")
print(f"  {'─'*45}")

for b in range(n_blocks):
    block = nums[b*block_size:(b+1)*block_size]
    avg_sum = sum(sum(d) for d in block) / block_size
    avg_range = sum(d[-1]-d[0] for d in block) / block_size
    avg_odd = sum(sum(1 for n in d if n%2==1)/6 for d in block) / block_size * 100
    consec_pct = sum(1 for d in block if any(d[i+1]==d[i]+1 for i in range(5))) / block_size * 100
    unique = len(set(n for d in block for n in d))
    
    print(f"  {b*block_size+1:>4}-{(b+1)*block_size:>4} {avg_sum:6.1f} {avg_range:6.1f} {avg_odd:5.1f}% {consec_pct:7.1f}% {unique:7d}")

# ================================================================
# 8. LAG-1 CONDITIONAL PER NUMBER
# ================================================================
sh("8. LAG-1 CONDITIONAL — Nếu số X ra kỳ T, P(X ra kỳ T+1)?")

print(f"  Top 15 số có P(repeat kỳ liền) cao nhất:\n")
repeat_probs = {}
for n in range(1, 46):
    appearances = [i for i, d in enumerate(nums) if n in d]
    repeats = sum(1 for i in range(len(appearances)-1) if appearances[i+1] == appearances[i]+1)
    p_repeat = repeats / len(appearances) if appearances else 0
    expected_p = freq[n] / N  # baseline probability
    repeat_probs[n] = (p_repeat, expected_p, repeats, len(appearances))

sorted_rp = sorted(repeat_probs.items(), key=lambda x: -x[1][0])
for n, (pr, pe, reps, total) in sorted_rp[:15]:
    ratio = pr / pe if pe > 0 else 0
    print(f"    Số {n:2d}: P(repeat)={pr:.3f}, P(base)={pe:.3f}, ratio={ratio:.2f}x, repeats={reps}/{total}")

# ================================================================
# 9. ODD/EVEN PER POSITION
# ================================================================
sh("9. ODD/EVEN PER POSITION — Vị trí nào ưu tiên chẵn/lẻ?")

for col in range(6):
    odd_cnt = sum(1 for d in nums if d[col] % 2 == 1)
    even_cnt = N - odd_cnt
    odd_pct = odd_cnt / N * 100
    print(f"  C{col+1}: Odd={odd_pct:.1f}%, Even={100-odd_pct:.1f}% {'🔵 ODD' if odd_pct > 55 else '🔴 EVEN' if odd_pct < 45 else ''}")

# ================================================================
# 10. DRAW TYPE CLUSTERING
# ================================================================
sh("10. DRAW TYPE CLUSTERING — Phân loại kỳ quay")

types = Counter()
for d in nums:
    s = sum(d)
    r = d[-1] - d[0]
    odd = sum(1 for n in d if n % 2 == 1)
    consec = sum(1 for i in range(5) if d[i+1] == d[i]+1)
    
    s_type = "LOW" if s < 120 else "HIGH" if s > 155 else "MID"
    r_type = "NAR" if r < 25 else "WIDE" if r > 38 else "MED"
    o_type = f"{odd}L"
    
    draw_type = f"{s_type}_{r_type}_{o_type}"
    types[draw_type] += 1

print(f"  Top 15 draw types:")
for dtype, cnt in types.most_common(15):
    pct = cnt / N * 100
    print(f"    {dtype:20s}: {cnt:4d} ({pct:5.1f}%)")

# ================================================================
# 11. CROSS-LAG ANALYSIS — C_i(T) ↔ C_j(T+1)
# ================================================================
sh("11. CROSS-LAG — Cột kỳ trước → cột khác kỳ sau")

print(f"  Correlation C_i(T) vs C_j(T+1):\n")
for ci in range(6):
    for cj in range(6):
        x = [nums[i][ci] for i in range(N-1)]
        y = [nums[i+1][cj] for i in range(N-1)]
        mx = sum(x) / len(x)
        my = sum(y) / len(y)
        cov = sum((a-mx)*(b-my) for a,b in zip(x,y)) / len(x)
        sx = (sum((a-mx)**2 for a in x)/len(x))**0.5
        sy = (sum((b-my)**2 for b in y)/len(y))**0.5
        r = cov/(sx*sy) if sx*sy > 0 else 0
        if abs(r) > 0.04:
            sig = "📡" if abs(r) > 0.06 else ""
            print(f"    C{ci+1}(T)→C{cj+1}(T+1): r={r:+.4f} {sig}")

# ================================================================
# 12. MODULAR CLOCK — Số xuất hiện theo nhịp modular?
# ================================================================
sh("12. MODULAR CLOCK — Số nào có nhịp xuất hiện theo mod?")

for num in [3, 7, 10, 19, 24, 37, 44]:
    appearances = [i for i, d in enumerate(nums) if num in d]
    
    best_mod = 2
    best_bias = 0
    for mod in range(2, 15):
        mod_dist = Counter(a % mod for a in appearances)
        max_pct = max(mod_dist.values()) / len(appearances)
        expected = 1 / mod
        bias = max_pct - expected
        if bias > best_bias:
            best_bias = bias
            best_mod = mod
            best_residue = mod_dist.most_common(1)[0][0]
    
    sig = "📡 CLOCK" if best_bias > 0.05 else ""
    print(f"  Số {num:2d}: best mod={best_mod}, residue={best_residue}, bias=+{best_bias:.3f} {sig}")

# ================================================================
# 13. SEQUENTIAL RUN TEST PER NUMBER
# ================================================================
sh("13. SEQUENTIAL RUNS — Kiểm tra tính random từng số")

for num in [3, 7, 19, 24, 38, 44]:
    binary = [1 if num in d else 0 for d in nums]
    n1 = sum(binary)
    n0 = N - n1
    
    # Count runs
    runs = 1
    for i in range(1, N):
        if binary[i] != binary[i-1]:
            runs += 1
    
    # Expected runs
    exp_runs = 2*n0*n1/N + 1
    std_runs = ((2*n0*n1*(2*n0*n1 - N)) / (N*N*(N-1)))**0.5 if N > 1 else 1
    z = (runs - exp_runs) / std_runs if std_runs > 0 else 0
    
    status = "🚨 NON-RANDOM" if abs(z) > 2 else "✅ RANDOM"
    print(f"  Số {num:2d}: runs={runs}, expected={exp_runs:.0f}, z={z:+.2f} {status}")

# ================================================================
# 14. ENTROPY RATE — Tốc độ entropy qua thời gian
# ================================================================
sh("14. ENTROPY RATE — Entropy thay đổi theo thời gian")

window = 50
entropies = []
for start in range(0, N - window, window // 2):
    block = nums[start:start+window]
    # Count frequency in block
    bf = Counter(n for d in block for n in d)
    total = sum(bf.values())
    H = -sum(c/total * math.log2(c/total) for c in bf.values() if c > 0)
    max_H = math.log2(45)
    entropies.append((start, H, H/max_H*100))

print(f"  Window={window}, sliding by {window//2}:\n")
print(f"  {'Start':>7} {'Entropy':>9} {'%Max':>6}")
# Show first 5 and last 5
for s, h, p in entropies[:3]:
    print(f"  {s+1:>7} {h:9.4f} {p:5.1f}%")
print(f"  {'...':>7}")
for s, h, p in entropies[-5:]:
    marker = "⚠️" if p < 98 else ""
    print(f"  {s+1:>7} {h:9.4f} {p:5.1f}% {marker}")

min_ent = min(entropies, key=lambda x: x[2])
max_ent = max(entropies, key=lambda x: x[2])
print(f"\n  MIN entropy: draws {min_ent[0]+1}-{min_ent[0]+window} ({min_ent[2]:.1f}%)")
print(f"  MAX entropy: draws {max_ent[0]+1}-{max_ent[0]+window} ({max_ent[2]:.1f}%)")

# ================================================================
# 15. SUM-OF-PAIRS WITHIN DRAW
# ================================================================
sh("15. SUM-OF-PAIRS — Tổng cặp trong kỳ có pattern?")

pair_sums = Counter()
for d in nums:
    for a, b in combinations(d, 2):
        pair_sums[a + b] += 1

expected_per_pair_sum = N * 15  # 15 pairs per draw
print(f"  Top 10 tổng cặp phổ biến nhất:")
for ps, cnt in pair_sums.most_common(10):
    pct = cnt / (N * 15) * 100
    print(f"    Sum={ps:3d}: {cnt:5d}x ({pct:.2f}%)")

# ================================================================
# 16. GOLDBACH-LIKE — Số trong kỳ = tổng 2 số khác?
# ================================================================
sh("16. GOLDBACH-LIKE — Số nào = tổng 2 số khác trong cùng kỳ?")

goldbach_cnt = Counter()
goldbach_draws = 0
for d in nums:
    found = False
    for i, n in enumerate(d):
        for a, b in combinations([x for j, x in enumerate(d) if j != i], 2):
            if a + b == n:
                goldbach_cnt[n] += 1
                found = True
                break
    if found:
        goldbach_draws += 1

print(f"  {goldbach_draws}/{N} kỳ ({goldbach_draws/N*100:.1f}%) có ít nhất 1 số = tổng 2 số khác\n")
print(f"  Top 10 số hay là 'tổng':")
for n, cnt in goldbach_cnt.most_common(10):
    print(f"    Số {n:2d}: {cnt:4d} lần ({cnt/N*100:.1f}%)")

# ================================================================
# 17. POWER 6/55 DEEP — Same analyses
# ================================================================
sh("17. POWER 6/55 — Temperature + Gap + Block")

PN = len(pnums)
p_freq = Counter(n for d in pnums for n in d)

# Temperature
p_temps = [(n, calc_temperature(pnums, n)) for n in range(1, 56)]
p_temps.sort(key=lambda x: -x[1])
print(f"  🔥 Power TOP 5 NÓNG: ", end="")
for n, t in p_temps[:5]:
    print(f"{n}({t:.2f}) ", end="")
print(f"\n  ❄️ Power TOP 5 LẠNH: ", end="")
for n, t in p_temps[-5:]:
    print(f"{n}({t:.2f}) ", end="")
print()

# Power chi² top
p_expected = PN * 6 / 55
p_chi = [(n, p_freq.get(n,0), (p_freq.get(n,0)-p_expected)**2/p_expected) for n in range(1,56)]
p_chi.sort(key=lambda x: -x[2])
print(f"\n  Power Chi² top 5 lệch nhất:")
for n, act, chi in p_chi[:5]:
    print(f"    Số {n:2d}: actual={act}, chi²={chi:.1f} ({'🔥' if act > p_expected else '❄️'})")

# Power Fourier
p_sums = [sum(d) for d in pnums]
p_mean = sum(p_sums) / PN
best_p_period = 2
best_p_power = 0
for period in range(2, 50):
    cs = sum((p_sums[i]-p_mean)*math.cos(2*math.pi*i/period) for i in range(PN))
    ss = sum((p_sums[i]-p_mean)*math.sin(2*math.pi*i/period) for i in range(PN))
    pw = (cs**2 + ss**2) / PN
    if pw > best_p_power:
        best_p_power = pw
        best_p_period = period

print(f"\n  Power Fourier: dominant period={best_p_period}")

# ================================================================
# SUMMARY
# ================================================================
print(f"\n{'═'*90}")
print(f"  🏆 TỔNG KẾT — PHÁT HIỆN MỚI V9")
print(f"{'═'*90}")
print(f"""
  1. CHI² PER NUMBER: Xác định số nào lệch chuẩn NHIỀU NHẤT
  2. GAP FITTING: Gap có tuân theo geometric distribution?
  3. INTERNAL SPACING: Pattern khoảng cách BÊN TRONG mỗi kỳ
  4. NUMBER TEMPERATURE: Chỉ số nóng/lạnh thời gian thực
  5. FOURIER ANALYSIS: Chu kỳ ẩn trong chuỗi tổng & từng cột
  6. MUTUAL INFORMATION: Quan hệ PHI TUYẾN giữa các cột
  7. BLOCK ANALYSIS: Thống kê biến đổi theo khối 100 kỳ
  8. LAG-1 CONDITIONAL: P(repeat) per number
  9. ODD/EVEN PER POSITION: Vị trí nào ưu tiên chẵn vs lẻ
  10. DRAW CLUSTERING: Phân loại kỳ quay theo type
  11. CROSS-LAG: C_i(T) → C_j(T+1) correlation
  12. MODULAR CLOCK: Nhịp xuất hiện theo mod
  13. SEQUENTIAL RUNS: Kiểm tra random per number
  14. ENTROPY RATE: Entropy biến thiên theo thời gian
  15. SUM-OF-PAIRS: Pattern tổng cặp trong kỳ
  16. GOLDBACH-LIKE: Số = tổng 2 số khác trong kỳ
  17. POWER 6/55: Temperature + Chi² + Fourier
""")

print(f"\n{'='*90}")
print(f"  📄 Output: {OUTPUT}")
print(f"{'='*90}")

sys.stdout = tee.out
tee.close()
