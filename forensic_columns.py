"""
FORENSIC TOÀN DIỆN 6 CỘT — Mega 6/45 + Power 6/55
====================================================
Phân tích 10 chiều: tần suất, xu hướng, chẵn/lẻ, lớn/nhỏ,
cụm số, loại trừ, xác suất cố định vs ngẫu nhiên.
Chạy cho CẢ HAI lottery types.
"""
import sys, os, math, json
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper.data_manager import get_mega645_numbers, get_power655_numbers


def analyze_lottery(name, raw_data, MAX_NUM):
    """Run full forensic on one lottery type."""
    # Power 6/55 has 7 elements (6 + bonus), only take first 6
    data = [d[:6] for d in raw_data]
    N = len(data)
    all_sorted = [sorted(d) for d in data]
    
    if N < 50:
        print(f"  ⚠️ {name}: Chỉ có {N} kỳ — không đủ dữ liệu!")
        return
    
    print(f"\n{'█'*110}")
    print(f"  🔬 {name} — FORENSIC TOÀN DIỆN 6 CỘT")
    print(f"  {N} kỳ quay | Range: 1-{MAX_NUM} | C({MAX_NUM},6) = {math.comb(MAX_NUM,6):,}")
    print(f"{'█'*110}")
    
    # ═══════════════════════════════════════════════════════
    # 1. TẦN SUẤT — Số hay về nhất mỗi cột
    # ═══════════════════════════════════════════════════════
    print(f"\n{'━'*110}")
    print(f"  📊 1. TẦN SUẤT — SỐ HAY VỀ MỖI CỘT")
    print(f"{'━'*110}")
    
    freq_data = {}
    for ci in range(6):
        col = [row[ci] for row in all_sorted]
        freq = Counter(col)
        unique_vals = sorted(set(col))
        expected = N / len(unique_vals)
        chi2 = sum((freq[v] - expected)**2 / expected for v in unique_vals)
        df = len(unique_vals) - 1
        is_biased = chi2 > 1.5 * df
        
        freq_data[ci] = {
            'top10': freq.most_common(10),
            'chi2': chi2, 'df': df, 'biased': is_biased,
            'unique': len(unique_vals), 'expected': expected,
        }
        
        bias_label = "⚠️ LỆCH" if is_biased else "✅ ĐỀU"
        print(f"\n  C{ci+1} [{len(unique_vals)} giá trị, kv={expected:.1f}] Chi²={chi2:.1f}/{df} → {bias_label}")
        top_str = " ".join(f"{v}({c})" for v, c in freq.most_common(10))
        print(f"    Top 10: {top_str}")

    # ═══════════════════════════════════════════════════════
    # 2. XU HƯỚNG TĂNG/GIẢM
    # ═══════════════════════════════════════════════════════
    print(f"\n{'━'*110}")
    print(f"  📊 2. XU HƯỚNG TĂNG / GIẢM / GIỮ")
    print(f"{'━'*110}\n")
    print(f"  {'Cột':>4} │ {'Tăng%':>7} {'Giảm%':>7} {'Giữ%':>7} │ {'AvgΔ':>6} │ {'Sau↑':>6} {'Sau↓':>6} │ Pattern")
    print(f"  {'─'*4} │ {'─'*7} {'─'*7} {'─'*7} │ {'─'*6} │ {'─'*6} {'─'*6} │ {'─'*20}")
    
    delta_data = {}
    for ci in range(6):
        col = [row[ci] for row in all_sorted]
        deltas = [col[j+1] - col[j] for j in range(N-1)]
        ups = sum(1 for d in deltas if d > 0)
        downs = sum(1 for d in deltas if d < 0)
        stays = sum(1 for d in deltas if d == 0)
        after_up = [deltas[j+1] for j in range(len(deltas)-1) if deltas[j] > 0]
        after_down = [deltas[j+1] for j in range(len(deltas)-1) if deltas[j] < 0]
        au = np.mean(after_up) if after_up else 0
        ad = np.mean(after_down) if after_down else 0
        
        if au < -0.3 and ad > 0.3: pattern = "🔄 MEAN REVERSION"
        elif au > 0.3 and ad < -0.3: pattern = "📈 MOMENTUM"
        else: pattern = "➖ TRUNG TÍNH"
        
        delta_data[ci] = {'au': au, 'ad': ad, 'pattern': pattern}
        print(f"  C{ci+1:>2} │ {ups/(N-1)*100:>6.1f}% {downs/(N-1)*100:>6.1f}% {stays/(N-1)*100:>6.1f}% │ "
              f"{np.mean(deltas):>+5.2f} │ {au:>+5.2f} {ad:>+5.2f} │ {pattern}")

    # ═══════════════════════════════════════════════════════
    # 3. CHẴN/LẺ
    # ═══════════════════════════════════════════════════════
    print(f"\n{'━'*110}")
    print(f"  📊 3. CHẴN / LẺ mỗi cột + toàn dãy")
    print(f"{'━'*110}\n")
    print(f"  {'Cột':>4} │ {'Chẵn%':>7} {'Lẻ%':>7} │ {'C→C':>7} {'C→L':>7} {'L→C':>7} {'L→L':>7} │ Bias")
    print(f"  {'─'*4} │ {'─'*7} {'─'*7} │ {'─'*7} {'─'*7} {'─'*7} {'─'*7} │ {'─'*10}")
    
    for ci in range(6):
        col = [row[ci] for row in all_sorted]
        even = sum(1 for v in col if v % 2 == 0)
        ee, el, le, ll = 0, 0, 0, 0
        for j in range(N-1):
            c, n = col[j] % 2 == 0, col[j+1] % 2 == 0
            if c and n: ee += 1
            elif c and not n: el += 1
            elif not c and n: le += 1
            else: ll += 1
        t = N - 1
        bias = "CÂN BẰNG" if abs(even/N*100 - 50) < 5 else ("CHẴN ⚠️" if even/N > 0.55 else "LẺ ⚠️")
        print(f"  C{ci+1:>2} │ {even/N*100:>6.1f}% {(N-even)/N*100:>6.1f}% │ {ee/t*100:>6.1f}% {el/t*100:>6.1f}% {le/t*100:>6.1f}% {ll/t*100:>6.1f}% │ {bias}")

    # Whole-draw
    print(f"\n  Phân bố chẵn/lẻ TOÀN DÃY:")
    eo_patterns = Counter(sum(1 for v in row if v % 2 == 0) for row in all_sorted)
    for ne in range(7):
        cnt = eo_patterns.get(ne, 0)
        expected = N * math.comb(MAX_NUM//2, ne) * math.comb((MAX_NUM+1)//2, 6-ne) / math.comb(MAX_NUM, 6) if ne <= MAX_NUM//2 and 6-ne <= (MAX_NUM+1)//2 else 0
        ratio = cnt / expected if expected > 0 else 0
        bar = '█' * int(cnt/N*100)
        print(f"    {ne}C{6-ne}L: {cnt:>5} ({cnt/N*100:>5.1f}%)  kv={expected:>5.0f} ({ratio:>.2f}x) {bar}")

    # ═══════════════════════════════════════════════════════
    # 4. LỚN/NHỎ
    # ═══════════════════════════════════════════════════════
    print(f"\n{'━'*110}")
    print(f"  📊 4. LỚN / NHỎ (>{MAX_NUM//2} = lớn)")
    print(f"{'━'*110}\n")
    mid = MAX_NUM / 2
    print(f"  {'Cột':>4} │ {'Median':>7} │ {f'>={int(mid+1)}%':>8} {f'<={int(mid)}%':>8} │ Nhận xét")
    print(f"  {'─'*4} │ {'─'*7} │ {'─'*8} {'─'*8} │ {'─'*15}")
    for ci in range(6):
        col = [row[ci] for row in all_sorted]
        med = np.median(col)
        big = sum(1 for v in col if v > mid) / N * 100
        small = sum(1 for v in col if v <= mid) / N * 100
        note = "CÂN BẰNG" if abs(big - small) < 10 else ("LỚN" if big > small else "NHỎ")
        print(f"  C{ci+1:>2} │ {med:>7.0f} │ {big:>7.1f}% {small:>7.1f}% │ {note}")

    # Whole-draw big/small distribution
    print(f"\n  Phân bố lớn/nhỏ TOÀN DÃY:")
    bs_patterns = Counter(sum(1 for v in row if v > mid) for row in all_sorted)
    for nb in range(7):
        cnt = bs_patterns.get(nb, 0)
        bar = '█' * int(cnt/N*100)
        print(f"    {nb}L{6-nb}N: {cnt:>5} ({cnt/N*100:>5.1f}%) {bar}")

    # ═══════════════════════════════════════════════════════
    # 5. CỤM SỐ — Cross-column pairs
    # ═══════════════════════════════════════════════════════
    print(f"\n{'━'*110}")
    print(f"  📊 5. CỤM SỐ — Cặp (Ci,Cj) xuất hiện bất thường nhiều")
    print(f"{'━'*110}")
    
    all_clusters = []
    for ci, cj in combinations(range(6), 2):
        pairs = Counter((row[ci], row[cj]) for row in all_sorted)
        u_i = len(set(row[ci] for row in all_sorted))
        u_j = len(set(row[cj] for row in all_sorted))
        exp = N / (u_i * u_j)
        for pair, cnt in pairs.most_common(3):
            ratio = cnt / max(0.1, exp)
            if ratio > 2:
                all_clusters.append((ci, cj, pair, cnt, ratio))
    
    all_clusters.sort(key=lambda x: -x[4])
    if all_clusters:
        print(f"\n  🔥 TOP CỤM (>2x kỳ vọng):")
        for rank, (ci, cj, pair, cnt, ratio) in enumerate(all_clusters[:20]):
            print(f"    {rank+1:>2}. C{ci+1}={pair[0]:>2}, C{cj+1}={pair[1]:>2} → {cnt:>3}x ({ratio:.1f}x expected)")
    else:
        print(f"\n  Không có cụm nào >2x kỳ vọng.")

    # ═══════════════════════════════════════════════════════
    # 6. LOẠI TRỪ
    # ═══════════════════════════════════════════════════════
    print(f"\n{'━'*110}")
    print(f"  📊 6. LOẠI TRỪ — Khi Ci=X, giá trị nào KHÔNG xuất hiện ở Cj?")
    print(f"{'━'*110}")
    
    exclusion_found = 0
    for ci in range(6):
        col = [row[ci] for row in all_sorted]
        top3_vals = [v for v, _ in Counter(col).most_common(3)]
        for v in top3_vals:
            matching = [row for row in all_sorted if row[ci] == v]
            if len(matching) < 15: continue
            for cj in range(6):
                if cj == ci: continue
                other_all = set(row[cj] for row in all_sorted)
                other_v = set(row[cj] for row in matching)
                never = other_all - other_v
                if len(never) >= 3 and len(never) < len(other_all) * 0.5:
                    print(f"  C{ci+1}={v} (n={len(matching)}): C{cj+1} không có {sorted(never)[:10]}")
                    exclusion_found += 1
    if exclusion_found == 0:
        print(f"  Không có quy luật loại trừ đáng kể.")

    # ═══════════════════════════════════════════════════════
    # 7. ỔN ĐỊNH THEO THỜI GIAN
    # ═══════════════════════════════════════════════════════
    print(f"\n{'━'*110}")
    print(f"  📊 7. ỔN ĐỊNH — Top 5 giá trị qua 4 giai đoạn (cố định hay thay đổi?)")
    print(f"{'━'*110}\n")
    
    q_size = N // 4
    for ci in range(6):
        col = [row[ci] for row in all_sorted]
        quarters = [col[i*q_size:(i+1)*q_size] for i in range(4)]
        q_tops = [set(v for v, _ in Counter(q).most_common(5)) for q in quarters]
        common = q_tops[0] & q_tops[1] & q_tops[2] & q_tops[3]
        all_u = q_tops[0] | q_tops[1] | q_tops[2] | q_tops[3]
        stab = len(common) / max(1, len(all_u)) * 100
        
        label = "🟢 CỐ ĐỊNH" if stab >= 40 else ("🟡 BÁN CĐ" if stab >= 20 else "🔴 THAY ĐỔI")
        q_str = " | ".join(str(sorted(qt)) for qt in q_tops)
        print(f"  C{ci+1}: {label} ({stab:.0f}%) — Luôn top5: {sorted(common)}")
        print(f"    Q1-Q4: {q_str}")

    # ═══════════════════════════════════════════════════════
    # 8. KIỂM ĐỊNH NGẪU NHIÊN
    # ═══════════════════════════════════════════════════════
    print(f"\n{'━'*110}")
    print(f"  📊 8. KIỂM ĐỊNH NGẪU NHIÊN — Runs test + Serial correlation + Autocorrelation")
    print(f"{'━'*110}\n")
    
    print(f"  {'Cột':>4} │ {'Z-runs':>7} {'Runs?':>7} │ {'Serial r':>9} {'Serial?':>8} │ {'AC lag1':>8} {'AC lag2':>8} │ KẾT LUẬN")
    print(f"  {'─'*4} │ {'─'*7} {'─'*7} │ {'─'*9} {'─'*8} │ {'─'*8} {'─'*8} │ {'─'*18}")
    
    random_results = {}
    for ci in range(6):
        col = [row[ci] for row in all_sorted]
        med = np.median(col)
        binary = [1 if v > med else 0 for v in col]
        runs = 1 + sum(1 for j in range(1, len(binary)) if binary[j] != binary[j-1])
        n1, n0 = sum(binary), len(binary) - sum(binary)
        er = 1 + 2*n1*n0/(n1+n0)
        sr = math.sqrt(2*n1*n0*(2*n1*n0-n1-n0)/((n1+n0)**2*(n1+n0-1))) if n1+n0 > 1 else 1
        z = (runs - er) / sr if sr > 0 else 0
        
        sc = np.corrcoef(col[:-1], col[1:])[0, 1]
        ac1 = np.corrcoef(col[:-1], col[1:])[0, 1]
        ac2 = np.corrcoef(col[:-2], col[2:])[0, 1] if N > 2 else 0
        
        r_runs = abs(z) < 1.96
        r_serial = abs(sc) < 0.06
        conclusion = "✅ RANDOM" if r_runs and r_serial else "⚠️ CÓ PATTERN"
        
        random_results[ci] = {'z': z, 'sc': sc, 'random': r_runs and r_serial}
        runs_icon = '✅' if r_runs else '⚠️'
        serial_icon = '✅' if r_serial else '⚠️'
        print(f"  C{ci+1:>2} │ {z:>+6.2f} {runs_icon:>7} │ {sc:>+8.4f} {serial_icon:>8} │ "
              f"{ac1:>+7.4f} {ac2:>+7.4f} │ {conclusion}")

    # ═══════════════════════════════════════════════════════
    # 9. TỔNG HỢP
    # ═══════════════════════════════════════════════════════
    print(f"\n{'━'*110}")
    print(f"  ⭐ 9. TỔNG HỢP — NGẪU NHIÊN vs KHÔNG NGẪU NHIÊN")
    print(f"{'━'*110}\n")
    
    print(f"  {'Cột':>4} │ {'Freq':>8} │ {'Xu hướng':>18} │ {'Chẵn/Lẻ':>8} │ {'Ổn định':>10} │ {'Random?':>10} │ {'KẾT LUẬN':>18}")
    print(f"  {'─'*4} │ {'─'*8} │ {'─'*18} │ {'─'*8} │ {'─'*10} │ {'─'*10} │ {'─'*18}")
    
    for ci in range(6):
        col = [row[ci] for row in all_sorted]
        fd = freq_data[ci]
        dd = delta_data[ci]
        even_pct = sum(1 for v in col if v % 2 == 0) / N * 100
        
        # Stability
        q_size_l = N // 4
        quarters_l = [col[i*q_size_l:(i+1)*q_size_l] for i in range(4)]
        q_tops_l = [set(v for v, _ in Counter(q).most_common(5)) for q in quarters_l]
        common_l = q_tops_l[0] & q_tops_l[1] & q_tops_l[2] & q_tops_l[3]
        all_u_l = q_tops_l[0] | q_tops_l[1] | q_tops_l[2] | q_tops_l[3]
        stab = len(common_l) / max(1, len(all_u_l)) * 100
        
        fb = "⚠️ LỆCH" if fd['biased'] else "✅ ĐỀU"
        trend = dd['pattern']
        eo = "CÂN BẰNG" if abs(even_pct - 50) < 5 else ("CHẴN⚠️" if even_pct > 55 else "LẺ⚠️")
        stable = "🟢 CĐ" if stab >= 40 else ("🟡" if stab >= 20 else "🔴 ĐỔI")
        rand = "✅ RND" if random_results[ci]['random'] else "⚠️ PTRN"
        
        exploit_score = sum([
            fd['biased'],
            abs(dd['au']) > 0.3 or abs(dd['ad']) > 0.3,
            stab >= 40,
            not random_results[ci]['random'],
        ])
        concl = "🟢 KHAI THÁC" if exploit_score >= 3 else ("🟡 YẾU" if exploit_score >= 2 else "🔴 RANDOM")
        
        print(f"  C{ci+1:>2} │ {fb:>8} │ {trend:>18} │ {eo:>8} │ {stable:>10} │ {rand:>10} │ {concl:>18}")

    # ═══════════════════════════════════════════════════════
    # 10. BẢNG PHÁT HIỆN CHI TIẾT
    # ═══════════════════════════════════════════════════════
    print(f"\n{'━'*110}")
    print(f"  📋 10. BẢNG PHÁT HIỆN — Gì ngẫu nhiên, gì KHÔNG?")
    print(f"{'━'*110}\n")
    
    findings = []
    for ci in range(6):
        col = [row[ci] for row in all_sorted]
        fd = freq_data[ci]
        dd = delta_data[ci]
        
        # Check frequency bias
        if fd['biased']:
            top3 = [(v, c/N*100) for v, c in fd['top10'][:3]]
            t3s = ", ".join(f"{v}={p:.1f}%" for v, p in top3)
            findings.append(('🔴 KO RANDOM', f'C{ci+1}', 'Tần suất phân bố lệch', t3s, fd['chi2']))
        else:
            findings.append(('🟢 RANDOM', f'C{ci+1}', 'Tần suất đều', f'Chi²={fd["chi2"]:.1f}', fd['chi2']))
        
        # Check mean reversion
        if dd['au'] < -0.5 and dd['ad'] > 0.5:
            findings.append(('🔴 KO RANDOM', f'C{ci+1}', 'Mean reversion mạnh', f'Sau↑={dd["au"]:+.2f}, Sau↓={dd["ad"]:+.2f}', abs(dd['au'])+abs(dd['ad'])))
        elif abs(dd['au']) < 0.15 and abs(dd['ad']) < 0.15:
            findings.append(('🟢 RANDOM', f'C{ci+1}', 'Xu hướng trung tính', f'Sau↑={dd["au"]:+.2f}', 0))
        
        # Even/odd bias
        even_pct = sum(1 for v in col if v % 2 == 0) / N * 100
        if abs(even_pct - 50) > 5:
            findings.append(('🔴 KO RANDOM', f'C{ci+1}', f'Chẵn/lẻ lệch', f'Chẵn={even_pct:.1f}%', abs(even_pct-50)))
        
        # Serial correlation
        sc = random_results[ci]['sc']
        if abs(sc) >= 0.06:
            findings.append(('🔴 KO RANDOM', f'C{ci+1}', 'Serial correlation', f'r={sc:+.4f}', abs(sc)))
        else:
            findings.append(('🟢 RANDOM', f'C{ci+1}', 'Không serial correlation', f'r={sc:+.4f}', abs(sc)))
    
    # Sort: non-random first, then by strength
    findings.sort(key=lambda x: (0 if 'KO RANDOM' in x[0] else 1, -x[4]))
    
    print(f"  {'Loại':<16} │ {'Cột':>4} │ {'Bằng chứng':<30} │ {'Chi tiết'}")
    print(f"  {'─'*16} │ {'─'*4} │ {'─'*30} │ {'─'*40}")
    for cat, col, evidence, detail, _ in findings:
        print(f"  {cat:<16} │ {col:>4} │ {evidence:<30} │ {detail}")
    
    # Summary counts
    non_random = sum(1 for f in findings if 'KO RANDOM' in f[0])
    random_cnt = sum(1 for f in findings if 'RANDOM' in f[0] and 'KO' not in f[0])
    print(f"\n  📊 TỔNG KẾT: {non_random} phát hiện KHÔNG NGẪU NHIÊN | {random_cnt} phát hiện NGẪU NHIÊN")
    ratio = non_random / max(1, non_random + random_cnt) * 100
    if ratio > 60:
        print(f"  → {ratio:.0f}% signal không random — CÓ CƠ SỞ KHAI THÁC (nhưng yếu)")
    elif ratio > 40:
        print(f"  → {ratio:.0f}% signal — HỖN HỢP random + pattern yếu")
    else:
        print(f"  → {ratio:.0f}% signal — GẦN NHƯ HOÀN TOÀN RANDOM")
    
    print(f"\n{'█'*110}\n")


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Loading data...")
    
    mega = get_mega645_numbers()
    print(f"  Mega 6/45: {len(mega)} kỳ")
    
    power = get_power655_numbers()
    print(f"  Power 6/55: {len(power)} kỳ")
    
    # Run for both
    analyze_lottery("MEGA 6/45", mega, 45)
    analyze_lottery("POWER 6/55", power, 55)
    
    print("=" * 110)
    print("  ✅ HOÀN THÀNH — Forensic cả Mega 6/45 và Power 6/55")
    print("=" * 110)
