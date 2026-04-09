# -*- coding: utf-8 -*-
"""
MEGA FORENSIC ALL-ASPECTS ANALYSIS
===================================
Phân tích MỌI khía cạnh thống kê của Vietlott Mega 6/45 & Power 6/55.
25+ loại phân tích chưa từng được khám phá.
"""
import sys, os, math
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scraper.data_manager import get_mega645_all, get_power655_all
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from itertools import combinations

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), 'mega_forensic_all_output.txt')

class TeeWriter:
    def __init__(self, filepath):
        self.file = open(filepath, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        self.file.close()

tee = TeeWriter(OUTPUT_FILE)
sys.stdout = tee

# ============================================================
# LOAD DATA
# ============================================================
mega_rows = get_mega645_all()
power_rows = get_power655_all()

mega_nums = [[r['n1'],r['n2'],r['n3'],r['n4'],r['n5'],r['n6']] for r in mega_rows]
power_nums = [[r['n1'],r['n2'],r['n3'],r['n4'],r['n5'],r['n6']] for r in power_rows]
mega_dates = [datetime.strptime(r['draw_date'], '%Y-%m-%d') for r in mega_rows]
power_dates = [datetime.strptime(r['draw_date'], '%Y-%m-%d') for r in power_rows]

PRIMES = {2,3,5,7,11,13,17,19,23,29,31,37,41,43}
FIBONACCI = {1,2,3,5,8,13,21,34}

def bar(pct, width=30):
    filled = int(pct / 100 * width)
    return '█' * filled + '░' * (width - filled)

def header(title, emoji="📊"):
    print(f"\n{'═'*90}")
    print(f"  {emoji} {title}")
    print(f"{'═'*90}")

def sub_header(title):
    print(f"\n  {'─'*80}")
    print(f"  ▸ {title}")
    print(f"  {'─'*80}")

def analyze(name, data, dates, max_num, rows_raw):
    header(f"{name} — PHÂN TÍCH TOÀN DIỆN MỌI KHÍA CẠNH", "🔬")
    print(f"  {len(data)} kỳ quay | Range: 1-{max_num}")

    # ============================================================
    # 1. SUM ANALYSIS — Tổng 6 số
    # ============================================================
    sub_header("1. TỔNG 6 SỐ (Sum Analysis)")
    sums = [sum(d) for d in data]
    avg_sum = sum(sums) / len(sums)
    expected_sum = 6 * (max_num + 1) / 2
    min_sum, max_sum_val = min(sums), max(sums)
    
    # Bucket sums into ranges
    bucket_size = 15
    buckets = defaultdict(int)
    for s in sums:
        b = (s // bucket_size) * bucket_size
        buckets[b] += 1
    
    print(f"  Trung bình: {avg_sum:.1f} (kỳ vọng: {expected_sum:.1f})")
    print(f"  Min: {min_sum} | Max: {max_sum_val}")
    print(f"  Std: {(sum((s-avg_sum)**2 for s in sums)/len(sums))**0.5:.1f}")
    print(f"\n  Phân bố tổng (bucket {bucket_size}):")
    for b in sorted(buckets.keys()):
        pct = buckets[b] / len(sums) * 100
        print(f"    {b:3d}-{b+bucket_size-1:3d}: {buckets[b]:4d} ({pct:5.1f}%) |{bar(pct)}")
    
    # Hot sum ranges (last 50 draws)
    recent_sums = sums[-50:]
    recent_avg = sum(recent_sums) / len(recent_sums)
    print(f"\n  📍 Tổng trung bình 50 kỳ gần nhất: {recent_avg:.1f} (vs overall: {avg_sum:.1f})")
    if recent_avg > avg_sum + 5:
        print(f"     → TỔNG ĐANG CAO — Có thể sắp giảm (mean reversion)")
    elif recent_avg < avg_sum - 5:
        print(f"     → TỔNG ĐANG THẤP — Có thể sắp tăng (mean reversion)")
    
    # Sum prediction
    sum_after_high = []
    sum_after_low = []
    median_sum = sorted(sums)[len(sums)//2]
    for i in range(1, len(sums)):
        if sums[i-1] > median_sum + 20:
            sum_after_high.append(sums[i])
        elif sums[i-1] < median_sum - 20:
            sum_after_low.append(sums[i])
    if sum_after_high:
        print(f"  Sau tổng CAO (>{median_sum+20}): avg={sum(sum_after_high)/len(sum_after_high):.1f} (n={len(sum_after_high)})")
    if sum_after_low:
        print(f"  Sau tổng THẤP (<{median_sum-20}): avg={sum(sum_after_low)/len(sum_after_low):.1f} (n={len(sum_after_low)})")

    # ============================================================
    # 2. RANGE (Spread) ANALYSIS
    # ============================================================
    sub_header("2. RANGE / SPREAD (Khoảng trải)")
    ranges = [d[-1] - d[0] for d in data]
    avg_range = sum(ranges) / len(ranges)
    range_counter = Counter(ranges)
    
    print(f"  Range trung bình: {avg_range:.1f}")
    print(f"  Min range: {min(ranges)} | Max range: {max(ranges)}")
    print(f"\n  Top 10 range phổ biến nhất:")
    for r, cnt in range_counter.most_common(10):
        pct = cnt / len(ranges) * 100
        print(f"    Range={r:2d}: {cnt:4d} ({pct:4.1f}%)")

    # ============================================================
    # 3. DAY OF WEEK ANALYSIS
    # ============================================================
    sub_header("3. NGÀY TRONG TUẦN (Day of Week)")
    dow_names = ['Thứ 2','Thứ 3','Thứ 4','Thứ 5','Thứ 6','Thứ 7','Chủ nhật']
    dow_count = Counter(d.weekday() for d in dates)
    dow_sum_avg = defaultdict(list)
    dow_overlap = defaultdict(list)  # overlap with next draw
    
    for i, (d, nums) in enumerate(zip(dates, data)):
        dow_sum_avg[d.weekday()].append(sum(nums))
        if i < len(data) - 1:
            overlap = len(set(nums) & set(data[i+1]))
            dow_overlap[d.weekday()].append(overlap)
    
    print(f"  {'Ngày':<12} {'Số kỳ':>8} {'% kỳ':>8} {'Tổng TB':>10} {'Overlap TB':>12}")
    print(f"  {'─'*55}")
    for dow in range(7):
        cnt = dow_count.get(dow, 0)
        if cnt > 0:
            pct = cnt / len(dates) * 100
            avg_s = sum(dow_sum_avg[dow]) / len(dow_sum_avg[dow])
            avg_o = sum(dow_overlap[dow]) / len(dow_overlap[dow]) if dow_overlap[dow] else 0
            print(f"  {dow_names[dow]:<12} {cnt:8d} {pct:7.1f}% {avg_s:10.1f} {avg_o:12.2f}")

    # ============================================================
    # 4. CONSECUTIVE NUMBERS
    # ============================================================
    sub_header("4. SỐ LIÊN TIẾP (Consecutive Numbers)")
    consec_count = Counter()
    for d in data:
        max_consec = 1
        curr = 1
        for j in range(1, len(d)):
            if d[j] == d[j-1] + 1:
                curr += 1
                max_consec = max(max_consec, curr)
            else:
                curr = 1
        consec_count[max_consec] += 1
    
    # Count how many draws have at least one pair of consecutive
    has_pair = sum(1 for d in data if any(d[j]==d[j-1]+1 for j in range(1,len(d))))
    
    print(f"  Kỳ có ÍT NHẤT 1 cặp liên tiếp: {has_pair}/{len(data)} ({has_pair/len(data)*100:.1f}%)")
    print(f"\n  Phân bố chuỗi liên tiếp dài nhất:")
    for k in sorted(consec_count.keys()):
        pct = consec_count[k] / len(data) * 100
        label = "(không liên tiếp)" if k == 1 else f"({k} số liên tiếp)"
        print(f"    {k}: {consec_count[k]:4d} ({pct:5.1f}%) {label}")
    
    # Which consecutive pairs appear most
    pair_counter = Counter()
    for d in data:
        for j in range(1, len(d)):
            if d[j] == d[j-1] + 1:
                pair_counter[(d[j-1], d[j])] += 1
    
    print(f"\n  Top 10 cặp liên tiếp hay xuất hiện:")
    for (a, b), cnt in pair_counter.most_common(10):
        print(f"    ({a:2d},{b:2d}): {cnt:3d} lần ({cnt/len(data)*100:.1f}%)")

    # ============================================================
    # 5. PRIME NUMBERS
    # ============================================================
    sub_header("5. SỐ NGUYÊN TỐ (Prime Numbers)")
    prime_counts = Counter()
    for d in data:
        n_primes = sum(1 for n in d if n in PRIMES)
        prime_counts[n_primes] += 1
    
    print(f"  Số nguyên tố trong range: {sorted(PRIMES & set(range(1, max_num+1)))}")
    total_primes = len(PRIMES & set(range(1, max_num+1)))
    expected_primes = 6 * total_primes / max_num
    print(f"  Kỳ vọng trung bình: {expected_primes:.2f} số nguyên tố / kỳ")
    actual_avg = sum(k*v for k,v in prime_counts.items()) / len(data)
    print(f"  Thực tế trung bình: {actual_avg:.2f}")
    print(f"\n  Phân bố:")
    for k in sorted(prime_counts.keys()):
        pct = prime_counts[k] / len(data) * 100
        print(f"    {k} nguyên tố: {prime_counts[k]:4d} ({pct:5.1f}%) |{bar(pct)}")

    # ============================================================
    # 6. FIBONACCI NUMBERS
    # ============================================================
    sub_header("6. SỐ FIBONACCI")
    fib_in_range = sorted(FIBONACCI & set(range(1, max_num+1)))
    fib_counts = Counter()
    for d in data:
        n_fib = sum(1 for n in d if n in FIBONACCI)
        fib_counts[n_fib] += 1
    
    print(f"  Fibonacci trong range: {fib_in_range}")
    expected_fib = 6 * len(fib_in_range) / max_num
    actual_fib = sum(k*v for k,v in fib_counts.items()) / len(data)
    print(f"  Kỳ vọng: {expected_fib:.2f} | Thực tế: {actual_fib:.2f}")
    for k in sorted(fib_counts.keys()):
        pct = fib_counts[k] / len(data) * 100
        print(f"    {k} Fibonacci: {fib_counts[k]:4d} ({pct:5.1f}%)")

    # ============================================================
    # 7. DECADE DISTRIBUTION
    # ============================================================
    sub_header("7. THẬP KỶ / NHÓM 10 (Decade Distribution)")
    decade_labels = []
    for start in range(1, max_num+1, 10):
        end = min(start + 9, max_num)
        decade_labels.append((start, end))
    
    decade_counts = defaultdict(int)
    for d in data:
        for n in d:
            for start, end in decade_labels:
                if start <= n <= end:
                    decade_counts[(start, end)] += 1
                    break
    
    total_nums = len(data) * 6
    print(f"  {'Nhóm':<10} {'Đếm':>8} {'%':>8} {'Kỳ vọng%':>10} {'Lệch':>8}")
    print(f"  {'─'*48}")
    for start, end in decade_labels:
        cnt = decade_counts[(start, end)]
        pct = cnt / total_nums * 100
        expected_pct = (end - start + 1) / max_num * 100
        diff = pct - expected_pct
        marker = "⬆️" if diff > 1 else "⬇️" if diff < -1 else "✅"
        print(f"  {start:2d}-{end:2d}     {cnt:8d} {pct:7.1f}% {expected_pct:9.1f}%  {diff:+6.1f}% {marker}")
    
    # Decade by position (last 100 draws)
    print(f"\n  Decades gần đây (100 kỳ cuối) vs toàn bộ:")
    recent = data[-100:]
    for start, end in decade_labels:
        cnt_recent = sum(1 for d in recent for n in d if start <= n <= end)
        cnt_all = decade_counts[(start, end)]
        pct_recent = cnt_recent / (100 * 6) * 100
        pct_all = cnt_all / total_nums * 100
        diff = pct_recent - pct_all
        hot = "🔥" if diff > 2 else "❄️" if diff < -2 else ""
        print(f"    {start:2d}-{end:2d}: recent={pct_recent:5.1f}% all={pct_all:5.1f}% diff={diff:+5.1f}% {hot}")

    # ============================================================
    # 8. TAIL DIGIT (Đuôi số)
    # ============================================================
    sub_header("8. ĐUÔI SỐ (Last Digit / Units)")
    tail_counts = Counter()
    for d in data:
        for n in d:
            tail_counts[n % 10] += 1
    
    print(f"  {'Đuôi':>6} {'Đếm':>8} {'%':>8} {'Kỳ vọng':>8}")
    print(f"  {'─'*35}")
    for digit in range(10):
        cnt = tail_counts[digit]
        pct = cnt / total_nums * 100
        # count how many numbers in range have this tail
        nums_with_tail = sum(1 for n in range(1, max_num+1) if n % 10 == digit)
        expected = nums_with_tail / max_num * 100
        print(f"  {digit:6d} {cnt:8d} {pct:7.1f}% {expected:7.1f}%")
    
    # Tail digit per position
    print(f"\n  Đuôi phổ biến theo vị trí:")
    for pos in range(6):
        pos_tails = Counter(data[i][pos] % 10 for i in range(len(data)))
        top3 = pos_tails.most_common(3)
        top_str = ", ".join(f"{t}({c})" for t, c in top3)
        print(f"    C{pos+1}: {top_str}")

    # ============================================================
    # 9. GAP ANALYSIS (Last seen)
    # ============================================================
    sub_header("9. GAP ANALYSIS — Khoảng cách xuất hiện")
    last_seen = {}
    gaps_per_num = defaultdict(list)
    
    for i, d in enumerate(data):
        for n in d:
            if n in last_seen:
                gap = i - last_seen[n]
                gaps_per_num[n].append(gap)
            last_seen[n] = i
    
    # Current gaps (from last draw)
    current_gaps = {}
    for n in range(1, max_num+1):
        current_gaps[n] = len(data) - 1 - last_seen.get(n, -1)
    
    sorted_gaps = sorted(current_gaps.items(), key=lambda x: -x[1])
    
    print(f"  Top 15 số QUAY HẠN LÂU NHẤT (overdue):")
    for n, gap in sorted_gaps[:15]:
        avg_gap = sum(gaps_per_num[n]) / len(gaps_per_num[n]) if gaps_per_num[n] else 0
        ratio = gap / avg_gap if avg_gap > 0 else 0
        marker = "🚨" if ratio > 3 else "⚠️" if ratio > 2 else ""
        print(f"    Số {n:2d}: {gap:3d} kỳ chưa ra (avg gap: {avg_gap:.1f}, ratio: {ratio:.1f}x) {marker}")
    
    print(f"\n  Top 10 số MỚI XUẤT HIỆN (hot):")
    for n, gap in sorted_gaps[-10:]:
        print(f"    Số {n:2d}: {gap:3d} kỳ trước")
    
    # Average gap per number
    print(f"\n  Top 10 số CÓ GAP TRUNG BÌNH NGẮN NHẤT (hay xuất hiện):")
    avg_gaps = {n: sum(g)/len(g) for n, g in gaps_per_num.items() if len(g) > 5}
    for n, ag in sorted(avg_gaps.items(), key=lambda x: x[1])[:10]:
        print(f"    Số {n:2d}: avg gap = {ag:.1f} kỳ (xuất hiện {len(gaps_per_num[n])+1} lần)")

    # ============================================================
    # 10. OVERLAP ANALYSIS (Trùng lặp giữa các kỳ)
    # ============================================================
    sub_header("10. TRÙNG SỐ GIỮA CÁC KỲ (Overlap)")
    overlaps = []
    for i in range(1, len(data)):
        overlap = len(set(data[i]) & set(data[i-1]))
        overlaps.append(overlap)
    
    overlap_counter = Counter(overlaps)
    avg_overlap = sum(overlaps) / len(overlaps)
    expected_overlap = 6 * 6 / max_num
    
    print(f"  Overlap trung bình: {avg_overlap:.3f} (kỳ vọng: {expected_overlap:.3f})")
    print(f"\n  Phân bố overlap (kỳ i vs kỳ i-1):")
    for k in sorted(overlap_counter.keys()):
        pct = overlap_counter[k] / len(overlaps) * 100
        print(f"    {k} trùng: {overlap_counter[k]:4d} ({pct:5.1f}%) |{bar(pct)}")
    
    # Overlap pattern: after high overlap, what happens?
    after_high_overlap = []
    after_zero_overlap = []
    for i in range(1, len(overlaps)):
        if overlaps[i-1] >= 3:
            after_high_overlap.append(overlaps[i])
        elif overlaps[i-1] == 0:
            after_zero_overlap.append(overlaps[i])
    
    if after_high_overlap:
        print(f"\n  Sau overlap ≥3: avg overlap tiếp = {sum(after_high_overlap)/len(after_high_overlap):.2f} (n={len(after_high_overlap)})")
    if after_zero_overlap:
        print(f"  Sau overlap =0: avg overlap tiếp = {sum(after_zero_overlap)/len(after_zero_overlap):.2f} (n={len(after_zero_overlap)})")

    # ============================================================
    # 11. MIRROR NUMBERS (Số gương: 12-21, 13-31, etc.)
    # ============================================================
    sub_header("11. SỐ GƯƠNG (Mirror Numbers)")
    mirror_pairs = []
    for n in range(10, max_num+1):
        d1, d2 = n // 10, n % 10
        if d1 != d2 and d2 != 0:
            mirror = d2 * 10 + d1
            if mirror <= max_num and mirror != n and (mirror, n) not in mirror_pairs:
                mirror_pairs.append((n, mirror))
    
    print(f"  {'Cặp gương':<15} {'Cùng kỳ':>8} {'Tỷ lệ':>8} {'Kỳ vọng':>8}")
    print(f"  {'─'*42}")
    for a, b in mirror_pairs[:15]:
        together = sum(1 for d in data if a in d and b in d)
        pct = together / len(data) * 100
        # expected: P(both in 6 from max_num)
        exp = 6/max_num * 5/(max_num-1) * len(data)
        exp_pct = exp / len(data) * 100
        marker = "⬆️" if pct > exp_pct * 1.5 else ""
        print(f"  ({a:2d},{b:2d})       {together:8d} {pct:7.2f}% {exp_pct:7.2f}% {marker}")

    # ============================================================
    # 12. SUM OF DIGITS (Tổng các chữ số)
    # ============================================================
    sub_header("12. TỔNG CÁC CHỮ SỐ (Digital Root)")
    def digit_sum(nums):
        return sum(sum(int(c) for c in str(n)) for n in nums)
    
    def digital_root(n):
        while n >= 10:
            n = sum(int(c) for c in str(n))
        return n
    
    dsums = [digit_sum(d) for d in data]
    droots = [digital_root(sum(d)) for d in data]
    
    droot_counter = Counter(droots)
    print(f"  Digital root phân bố (of sum):")
    for k in sorted(droot_counter.keys()):
        pct = droot_counter[k] / len(data) * 100
        print(f"    Root {k}: {droot_counter[k]:4d} ({pct:5.1f}%) |{bar(pct, 20)}")

    # ============================================================
    # 13. SPACING BETWEEN NUMBERS
    # ============================================================
    sub_header("13. KHOẢNG CÁCH GIỮA CÁC SỐ (Spacing)")
    spacings_all = []
    for d in data:
        for j in range(1, len(d)):
            spacings_all.append(d[j] - d[j-1])
    
    spacing_counter = Counter(spacings_all)
    avg_spacing = sum(spacings_all) / len(spacings_all)
    
    print(f"  Khoảng cách trung bình: {avg_spacing:.2f}")
    print(f"\n  Top 15 khoảng cách phổ biến:")
    for sp, cnt in spacing_counter.most_common(15):
        pct = cnt / len(spacings_all) * 100
        print(f"    Δ={sp:2d}: {cnt:5d} ({pct:5.1f}%) |{bar(pct, 25)}")

    # ============================================================
    # 14. HOT/COLD STREAKS
    # ============================================================
    sub_header("14. CHUỖI NÓNG/LẠNH (Hot/Cold Streaks)")
    # For each number, find max consecutive draws where it appeared / didn't appear
    for label, target_nums in [("SỐ HAY STREAK", range(1, max_num+1))]:
        print(f"\n  Top 10 streak NÓNG (xuất hiện liên tục):")
        hot_streaks = {}
        for n in target_nums:
            max_streak = 0
            curr_streak = 0
            for d in data:
                if n in d:
                    curr_streak += 1
                    max_streak = max(max_streak, curr_streak)
                else:
                    curr_streak = 0
            hot_streaks[n] = max_streak
        
        for n, s in sorted(hot_streaks.items(), key=lambda x: -x[1])[:10]:
            print(f"    Số {n:2d}: {s} kỳ liên tiếp")
        
        print(f"\n  Top 10 streak LẠNH (vắng mặt liên tục):")
        cold_streaks = {}
        for n in target_nums:
            max_streak = 0
            curr_streak = 0
            for d in data:
                if n not in d:
                    curr_streak += 1
                    max_streak = max(max_streak, curr_streak)
                else:
                    curr_streak = 0
            cold_streaks[n] = max_streak
        
        for n, s in sorted(cold_streaks.items(), key=lambda x: -x[1])[:10]:
            print(f"    Số {n:2d}: {s} kỳ vắng mặt liên tiếp")

    # ============================================================
    # 15. TRIPLET FREQUENCY
    # ============================================================
    sub_header("15. BỘ BA THƯỜNG XUYÊN (Triplet Analysis)")
    triplet_counter = Counter()
    for d in data:
        for combo in combinations(d, 3):
            triplet_counter[combo] += 1
    
    print(f"  Tổng loại bộ ba: {len(triplet_counter)}")
    expected_triplet = len(data) * math.comb(6,3) / math.comb(max_num,3)
    print(f"  Kỳ vọng mỗi bộ ba: {expected_triplet:.3f}")
    print(f"\n  Top 15 bộ ba xuất hiện nhiều nhất:")
    for trip, cnt in triplet_counter.most_common(15):
        ratio = cnt / expected_triplet
        print(f"    {trip}: {cnt:3d}x (ratio: {ratio:.1f}x)")

    # ============================================================
    # 16. SEASONS / MONTHS
    # ============================================================
    sub_header("16. MÙA / THÁNG (Seasonal Patterns)")
    month_avg_sum = defaultdict(list)
    month_counts = Counter()
    for date, nums in zip(dates, data):
        month_avg_sum[date.month].append(sum(nums))
        month_counts[date.month] += 1
    
    month_names = {1:'Tháng 1',2:'Tháng 2',3:'Tháng 3',4:'Tháng 4',
                   5:'Tháng 5',6:'Tháng 6',7:'Tháng 7',8:'Tháng 8',
                   9:'Tháng 9',10:'Tháng 10',11:'Tháng 11',12:'Tháng 12'}
    
    print(f"  {'Tháng':<12} {'Kỳ quay':>8} {'Sum TB':>8} {'Lệch':>8}")
    print(f"  {'─'*40}")
    overall_avg = avg_sum
    for m in range(1, 13):
        if month_avg_sum[m]:
            avg_m = sum(month_avg_sum[m]) / len(month_avg_sum[m])
            diff = avg_m - overall_avg
            marker = "⬆️" if diff > 3 else "⬇️" if diff < -3 else ""
            print(f"  {month_names[m]:<12} {month_counts[m]:8d} {avg_m:8.1f} {diff:+7.1f} {marker}")

    # ============================================================
    # 17. POSITION CORRELATION
    # ============================================================
    sub_header("17. TƯƠNG QUAN GIỮA CÁC VỊ TRÍ (Position Correlation)")
    # Pearson correlation between C_i and C_j across draws
    print(f"  Ma trận tương quan (Pearson r) giữa các cột:")
    col_data = [[data[i][p] for i in range(len(data))] for p in range(6)]
    
    print(f"        {'C1':>8}{'C2':>8}{'C3':>8}{'C4':>8}{'C5':>8}{'C6':>8}")
    for i in range(6):
        row = f"  C{i+1}  "
        for j in range(6):
            if i == j:
                row += f"{'1.000':>8}"
            else:
                n = len(col_data[i])
                mx = sum(col_data[i]) / n
                my = sum(col_data[j]) / n
                cov = sum((col_data[i][k]-mx)*(col_data[j][k]-my) for k in range(n)) / n
                sx = (sum((x-mx)**2 for x in col_data[i]) / n) ** 0.5
                sy = (sum((y-my)**2 for y in col_data[j]) / n) ** 0.5
                r = cov / (sx * sy) if sx * sy > 0 else 0
                row += f"{r:8.3f}"
        print(row)

    # ============================================================
    # 18. SAME DRAW REPEAT CHECK
    # ============================================================
    sub_header("18. KỲ TRÙNG HOÀN TOÀN (Same Draw Repeat)")
    draw_set = Counter(tuple(d) for d in data)
    repeats = {k: v for k, v in draw_set.items() if v > 1}
    if repeats:
        print(f"  🚨 {len(repeats)} bộ số xuất hiện hơn 1 lần:")
        for combo, cnt in sorted(repeats.items(), key=lambda x: -x[1]):
            print(f"    {combo} → {cnt} lần")
    else:
        print(f"  ✅ Không có bộ số nào trùng hoàn toàn — Đúng kỳ vọng random")

    # ============================================================
    # 19. EDGE NUMBERS (Biên: 1-5 và max-4 → max)
    # ============================================================
    sub_header("19. SỐ BIÊN (Edge Numbers)")
    low_edge = set(range(1, 6))
    high_edge = set(range(max_num - 4, max_num + 1))
    
    edge_count = Counter()
    for d in data:
        n_low = sum(1 for n in d if n in low_edge)
        n_high = sum(1 for n in d if n in high_edge)
        edge_count[(n_low, n_high)] += 1
    
    print(f"  Biên thấp: {sorted(low_edge)} | Biên cao: {sorted(high_edge)}")
    print(f"\n  {'(Low,High)':<15} {'Lần':>6} {'%':>7}")
    print(f"  {'─'*30}")
    for (lo, hi), cnt in sorted(edge_count.items(), key=lambda x: -x[1])[:12]:
        pct = cnt / len(data) * 100
        print(f"  ({lo},{hi})         {cnt:6d} {pct:6.1f}%")

    # ============================================================
    # 20. QUADRANT ANALYSIS (Chia 4 phần đều)
    # ============================================================
    sub_header("20. PHẦN TƯ (Quadrant Analysis)")
    q_size = max_num // 4
    quadrants = [(1, q_size), (q_size+1, 2*q_size), (2*q_size+1, 3*q_size), (3*q_size+1, max_num)]
    
    quad_patterns = Counter()
    for d in data:
        pattern = tuple(sum(1 for n in d if lo <= n <= hi) for lo, hi in quadrants)
        quad_patterns[pattern] += 1
    
    print(f"  4 phần: {quadrants}")
    print(f"\n  Top 15 patterns (Q1,Q2,Q3,Q4):")
    for pat, cnt in quad_patterns.most_common(15):
        pct = cnt / len(data) * 100
        print(f"    {pat}: {cnt:4d} ({pct:4.1f}%)")

    # ============================================================
    # 21. BIRTHDAY PARADOX — Collision Analysis
    # ============================================================
    sub_header("21. PHÂN TÍCH TRÙNG SỐ (Collision Analysis)")
    # How often do specific numbers appear together?
    pair_freq = Counter()
    for d in data:
        for a, b in combinations(d, 2):
            pair_freq[(a, b)] += 1
    
    expected_pair = len(data) * math.comb(6,2) / math.comb(max_num,2)
    
    print(f"  Kỳ vọng mỗi cặp: {expected_pair:.2f}")
    print(f"\n  Top 15 cặp số HAY ĐI CÙNG nhau:")
    for (a, b), cnt in pair_freq.most_common(15):
        ratio = cnt / expected_pair
        marker = "🔥" if ratio > 2 else "⬆️" if ratio > 1.5 else ""
        print(f"    ({a:2d},{b:2d}): {cnt:3d}x (ratio: {ratio:.1f}x) {marker}")
    
    print(f"\n  Top 10 cặp HIẾM KHI ĐI CÙNG:")
    for (a, b), cnt in pair_freq.most_common()[:-11:-1]:
        ratio = cnt / expected_pair
        print(f"    ({a:2d},{b:2d}): {cnt:3d}x (ratio: {ratio:.1f}x)")

    # ============================================================
    # 22. NUMBER FOLLOWING PATTERN (A appears → what's next draw?)
    # ============================================================
    sub_header("22. SỐ TIẾP THEO (Following Pattern)")
    print(f"  Khi số X xuất hiện kỳ T, số nào hay xuất hiện kỳ T+1?")
    
    following = defaultdict(Counter)
    for i in range(len(data)-1):
        for n in data[i]:
            for m in data[i+1]:
                following[n][m] += 1
    
    # Find strongest following signals
    strong_follows = []
    for n in range(1, max_num+1):
        if n in following:
            total = sum(following[n].values())
            for m, cnt in following[n].most_common(3):
                expected = total * 6/max_num
                ratio = cnt / expected if expected > 0 else 0
                if ratio > 1.5 and cnt > 15:
                    strong_follows.append((n, m, cnt, ratio))
    
    strong_follows.sort(key=lambda x: -x[3])
    print(f"\n  Top 20 'following signals' mạnh nhất (ratio > 1.5x):")
    for n, m, cnt, ratio in strong_follows[:20]:
        print(f"    Sau {n:2d} → {m:2d}: {cnt:3d}x (ratio: {ratio:.2f}x)")

    # ============================================================  
    # 23. YEARLY TREND — Số nào đang "nóng" gần đây?
    # ============================================================
    sub_header("23. XU HƯỚNG HIỆN TẠI — Số nóng/lạnh gần đây")
    recent_50 = data[-50:]
    recent_100 = data[-100:]
    all_freq = Counter(n for d in data for n in d)
    recent_freq = Counter(n for d in recent_50 for n in d)
    
    print(f"  Tần suất so sánh (50 kỳ gần nhất vs toàn bộ):")
    print(f"  {'Số':>4} {'All%':>7} {'R50%':>7} {'Lệch':>7} {'Trend':>10}")
    print(f"  {'─'*40}")
    
    trends = []
    for n in range(1, max_num+1):
        all_pct = all_freq[n] / (len(data) * 6) * 100
        rec_pct = recent_freq.get(n, 0) / (50 * 6) * 100
        diff = rec_pct - all_pct
        trends.append((n, all_pct, rec_pct, diff))
    
    trends.sort(key=lambda x: -x[3])
    print(f"\n  🔥 TOP 10 SỐ ĐANG NÓNG:")
    for n, ap, rp, d in trends[:10]:
        print(f"     Số {n:2d}: all={ap:.1f}% recent={rp:.1f}% diff={d:+.1f}% 🔥")
    
    print(f"\n  ❄️ TOP 10 SỐ ĐANG LẠNH:")
    for n, ap, rp, d in trends[-10:]:
        print(f"     Số {n:2d}: all={ap:.1f}% recent={rp:.1f}% diff={d:+.1f}% ❄️")

    # ============================================================
    # 24. MATHEMATICAL PATTERNS
    # ============================================================
    sub_header("24. PATTERNS TOÁN HỌC (Sum Mod, Product Patterns)")
    
    # Sum modulo patterns
    for modulo in [3, 5, 7, 9]:
        mod_counter = Counter(sum(d) % modulo for d in data)
        print(f"\n  Sum mod {modulo}:")
        for k in sorted(mod_counter.keys()):
            pct = mod_counter[k] / len(data) * 100
            expected = 100 / modulo
            diff = pct - expected
            print(f"    ≡{k}: {mod_counter[k]:4d} ({pct:5.1f}%) [kv: {expected:.1f}%] {'+' if diff > 0 else ''}{diff:.1f}%")
    
    # All-odd, all-even draws
    all_odd = sum(1 for d in data if all(n % 2 == 1 for n in d))
    all_even = sum(1 for d in data if all(n % 2 == 0 for n in d))
    print(f"\n  All-odd draws: {all_odd} ({all_odd/len(data)*100:.2f}%)")
    print(f"  All-even draws: {all_even} ({all_even/len(data)*100:.2f}%)")

    # ============================================================
    # 25. JACKPOT CORRELATION (if available)
    # ============================================================
    sub_header("25. PHÂN TÍCH BỔ SUNG — Jackpot & Special")
    
    # Longest gap without 3+ match to previous
    no_overlap_streak = 0
    max_no_overlap = 0
    for i in range(1, len(data)):
        if len(set(data[i]) & set(data[i-1])) < 1:
            no_overlap_streak += 1
            max_no_overlap = max(max_no_overlap, no_overlap_streak)
        else:
            no_overlap_streak = 0
    print(f"  Chuỗi dài nhất KHÔNG TRÙNG NHAU: {max_no_overlap} kỳ liên tiếp")
    
    # Number appearance frequency (global)
    print(f"\n  BẢNG TẦN SUẤT TOÀN BỘ (sorted by freq):")
    freq_list = sorted(all_freq.items(), key=lambda x: -x[1])
    for i, (n, cnt) in enumerate(freq_list):
        pct = cnt / len(data) * 100
        expected = len(data) * 6 / max_num
        ratio = cnt / expected
        hot_cold = "🔥" if ratio > 1.1 else "❄️" if ratio < 0.9 else ""
        if i < 10 or i >= len(freq_list) - 5:
            print(f"    #{i+1:2d} Số {n:2d}: {cnt:4d}x ({pct:5.1f}%) ratio={ratio:.2f}x {hot_cold}")
        elif i == 10:
            print(f"    {'... (truncated middle) ...':>50}")

    # ============================================================
    # 26. REPEAT-AFTER-N ANALYSIS
    # ============================================================
    sub_header("26. LẶP SAU N KỲ (Repeat-After-N)")
    print(f"  Khi số X xuất hiện kỳ T, xác suất X xuất hiện lại kỳ T+N?")
    
    for lag in [1, 2, 3, 5, 7, 10]:
        repeats = 0
        total_checks = 0
        for i in range(len(data) - lag):
            s1 = set(data[i])
            s2 = set(data[i + lag])
            repeats += len(s1 & s2)
            total_checks += 6
        pct = repeats / total_checks * 100
        expected = 6 / max_num * 100
        print(f"    Lag={lag:2d}: repeat%={pct:.2f}% (kv: {expected:.2f}%) ratio={pct/expected:.2f}x")

    # ============================================================
    # 27. WINNING PATTERN TEMPLATE
    # ============================================================
    sub_header("27. TEMPLATE THẮNG — Pattern phổ biến nhất")
    
    # Pattern based on modulo 5 distribution
    mod5_patterns = Counter()
    for d in data:
        pat = tuple(sorted(Counter(n % 5 for n in d).values(), reverse=True))
        mod5_patterns[pat] += 1
    
    print(f"  Phân bố mod5 pattern (top 10):")
    for pat, cnt in mod5_patterns.most_common(10):
        pct = cnt / len(data) * 100
        print(f"    {pat}: {cnt:4d} ({pct:4.1f}%)")
    
    # Odd count pattern
    odd_counter = Counter(sum(1 for n in d if n % 2 == 1) for d in data)
    print(f"\n  Phân bố số lẻ:")
    for k in sorted(odd_counter.keys()):
        pct = odd_counter[k] / len(data) * 100
        print(f"    {k} lẻ: {odd_counter[k]:4d} ({pct:5.1f}%) |{bar(pct)}")

    print(f"\n{'═'*90}")
    print(f"  ✅ HOÀN THÀNH PHÂN TÍCH {name}")
    print(f"{'═'*90}\n")


# ============================================================
# RUN ANALYSIS
# ============================================================
print("="*90)
print("  🔬 MEGA FORENSIC ALL-ASPECTS — PHÂN TÍCH MỌI KHÍA CẠNH")
print(f"  📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*90)

analyze("MEGA 6/45", mega_nums, mega_dates, 45, mega_rows)
analyze("POWER 6/55", power_nums, power_dates, 55, power_rows)

print(f"\n{'='*90}")
print(f"  📄 Output saved to: {OUTPUT_FILE}")
print(f"{'='*90}")

sys.stdout = tee.stdout
tee.close()
