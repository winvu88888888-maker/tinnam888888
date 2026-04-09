"""
PHÂN TÍCH SÂU 6 CỘT — Mega 6/45
==================================
Tìm cột dễ đoán nhất, khó đoán nhất, và các yếu tố căn cứ đánh số.

Kết quả sorted ascending: [C1, C2, C3, C4, C5, C6]
C1 = số nhỏ nhất, C6 = số lớn nhất
"""
import sys, os, math, time
import numpy as np
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper.data_manager import get_mega645_numbers

data = get_mega645_numbers()
N = len(data)
WARMUP = 200
n_test = N - WARMUP

# Prepare columns
columns = [[] for _ in range(6)]
for d in data:
    s = sorted(d[:6])
    for i in range(6):
        columns[i].append(s[i])

print("=" * 100)
print("  🔬 PHÂN TÍCH SÂU 6 CỘT — Mega 6/45")
print(f"  {N} kỳ quay | Backtest: {n_test} kỳ (warmup={WARMUP})")
print("=" * 100)

# ═══════════════════════════════════════
# A. THỐNG KÊ CƠ BẢN
# ═══════════════════════════════════════
print(f"\n{'━'*100}")
print(f"  📊 A. THỐNG KÊ CƠ BẢN MỖI CỘT")
print(f"{'━'*100}\n")

print(f"  {'Cột':>4} │ {'Min':>4} {'Max':>4} {'TB':>6} {'Median':>7} {'StdDev':>7} │ "
      f"{'Range':>6} │ {'Unique':>7} │ {'Entropy':>8} │ {'Độ khó':>10}")

print(f"  {'─'*4} │ {'─'*4} {'─'*4} {'─'*6} {'─'*7} {'─'*7} │ {'─'*6} │ {'─'*7} │ {'─'*8} │ {'─'*10}")

col_stats = []
for i in range(6):
    col = columns[i]
    mn, mx = min(col), max(col)
    avg = np.mean(col)
    med = np.median(col)
    std = np.std(col)
    unique = len(set(col))
    rng = mx - mn + 1
    
    # Entropy (higher = harder to predict)
    freq = Counter(col)
    probs = [c/N for c in freq.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    max_entropy = math.log2(unique)
    entropy_ratio = entropy / max_entropy if max_entropy > 0 else 0
    
    col_stats.append({
        'idx': i, 'min': mn, 'max': mx, 'avg': avg, 'med': med,
        'std': std, 'unique': unique, 'range': rng, 'entropy': entropy,
        'entropy_ratio': entropy_ratio
    })
    
    difficulty = "★" * int(entropy_ratio * 5 + 0.5)
    print(f"  C{i+1:>2} │ {mn:>4} {mx:>4} {avg:>6.1f} {med:>7.1f} {std:>7.1f} │ "
          f"{rng:>6} │ {unique:>7} │ {entropy:>7.2f}b │ {difficulty:>10}")

# Rank by difficulty
print(f"\n  📌 Xếp hạng từ DỄ → KHÓ (theo entropy):")
ranked = sorted(col_stats, key=lambda x: x['entropy'])
for rank, cs in enumerate(ranked):
    label = "🟢 DỄ NHẤT" if rank == 0 else ("🔴 KHÓ NHẤT" if rank == 5 else "")
    print(f"    {rank+1}. C{cs['idx']+1} — entropy={cs['entropy']:.2f}b, "
          f"range={cs['range']}, unique={cs['unique']} {label}")

# ═══════════════════════════════════════
# B. PHÂN BỐ TẦN SUẤT CHI TIẾT
# ═══════════════════════════════════════
print(f"\n{'━'*100}")
print(f"  📊 B. PHÂN BỐ TẦN SUẤT — Mỗi cột xuất hiện số nào nhiều nhất?")
print(f"{'━'*100}")

for i in range(6):
    col = columns[i]
    freq = Counter(col).most_common()
    top10 = freq[:10]
    total = N
    
    # Concentration: top 5 values cover how much?
    top5_pct = sum(c for _, c in freq[:5]) / total * 100
    top10_pct = sum(c for _, c in freq[:10]) / total * 100
    
    print(f"\n  C{i+1} — Top 10 giá trị (top5 phủ {top5_pct:.1f}%, top10 phủ {top10_pct:.1f}%):")
    row = "    "
    for val, cnt in top10:
        pct = cnt/total*100
        bar = '█' * int(pct / 2)
        row_item = f"{val:>3}({pct:.1f}%){bar}"
        print(f"    {val:>3}: {cnt:>4} lần ({pct:>5.1f}%)  {'█' * int(pct)}")

# ═══════════════════════════════════════
# C. KHOẢNG CHẠY THỰC TẾ (90% RANGE)
# ═══════════════════════════════════════
print(f"\n{'━'*100}")
print(f"  📊 C. KHOẢNG CHẠY THỰC TẾ — Mỗi cột thường nằm trong dãy nào?")
print(f"{'━'*100}\n")

print(f"  {'Cột':>4} │ {'80% range':>14} │ {'90% range':>14} │ {'95% range':>14} │ "
      f"{'Độ rộng 90%':>12} │ {'Đánh giá':>12}")
print(f"  {'─'*4} │ {'─'*14} │ {'─'*14} │ {'─'*14} │ {'─'*12} │ {'─'*12}")

range_data = []
for i in range(6):
    col = columns[i]
    p80 = (int(np.percentile(col, 10)), int(np.percentile(col, 90)))
    p90 = (int(np.percentile(col, 5)), int(np.percentile(col, 95)))
    p95 = (int(np.percentile(col, 2.5)), int(np.percentile(col, 97.5)))
    
    width90 = p90[1] - p90[0] + 1
    range_data.append({'idx': i, 'p90': p90, 'width90': width90})
    
    label = "🟢 HẸP" if width90 <= 15 else ("🔴 RỘNG" if width90 >= 25 else "🟡 VỪA")
    
    print(f"  C{i+1:>2} │ [{p80[0]:>3} - {p80[1]:>3}]   │ [{p90[0]:>3} - {p90[1]:>3}]   │ "
          f"[{p95[0]:>3} - {p95[1]:>3}]   │ {width90:>12} │ {label:>12}")

print(f"\n  📌 Xếp hạng dãy HẸP → RỘNG (90% range):")
ranked_range = sorted(range_data, key=lambda x: x['width90'])
for rank, rd in enumerate(ranked_range):
    label = "🟢 HẸP NHẤT → DỄ ĐOÁN NHẤT" if rank == 0 else ("🔴 RỘNG NHẤT → KHÓ ĐOÁN NHẤT" if rank == 5 else "")
    print(f"    {rank+1}. C{rd['idx']+1} — width={rd['width90']} số, "
          f"range=[{rd['p90'][0]}-{rd['p90'][1]}] {label}")

# ═══════════════════════════════════════
# D. BACKTEST ĐOÁN TỪNG CỘT (NHIỀU PHƯƠNG PHÁP)
# ═══════════════════════════════════════
print(f"\n{'━'*100}")
print(f"  📊 D. BACKTEST — Đoán chính xác từng cột ({n_test} kỳ)")
print(f"{'━'*100}\n")

methods = {
    'Mode (1 số)': {},       # Đoán mode
    'Top-3 (3 số)': {},      # 1 trong 3 hay nhất
    'Top-5 (5 số)': {},      # 1 trong 5 hay nhất
    'Median±1 (3 số)': {},   # Median ± 1
    'Median±2 (5 số)': {},   # Median ± 2
    'Last-seen mode': {},    # Mode từ 30 kỳ gần nhất
    '90% range': {},         # Nằm trong khoảng 90%
    'Repeat (=last)': {},    # Lặp lại giá trị kỳ trước
}

for method in methods:
    methods[method] = {i: 0 for i in range(6)}

for ti in range(n_test):
    te = WARMUP + ti
    actual = sorted(data[te][:6])
    
    train_cols = [[] for _ in range(6)]
    for d in data[te-200:te]:
        s = sorted(d[:6])
        for j in range(6):
            train_cols[j].append(s[j])
    
    prev = sorted(data[te-1][:6])
    
    for i in range(6):
        col = train_cols[i]
        recent = col[-30:]
        
        # Mode
        mode_v = Counter(col).most_common(1)[0][0]
        if actual[i] == mode_v:
            methods['Mode (1 số)'][i] += 1
        
        # Top-3
        top3 = set(v for v, _ in Counter(col).most_common(3))
        if actual[i] in top3:
            methods['Top-3 (3 số)'][i] += 1
        
        # Top-5
        top5 = set(v for v, _ in Counter(col).most_common(5))
        if actual[i] in top5:
            methods['Top-5 (5 số)'][i] += 1
        
        # Median±1
        med = int(np.median(col))
        if abs(actual[i] - med) <= 1:
            methods['Median±1 (3 số)'][i] += 1
        
        # Median±2
        if abs(actual[i] - med) <= 2:
            methods['Median±2 (5 số)'][i] += 1
        
        # Recent mode (30 kỳ)
        recent_mode = Counter(recent).most_common(1)[0][0]
        if actual[i] == recent_mode:
            methods['Last-seen mode'][i] += 1
        
        # 90% range
        lo90 = int(np.percentile(col, 5))
        hi90 = int(np.percentile(col, 95))
        if lo90 <= actual[i] <= hi90:
            methods['90% range'][i] += 1
        
        # Repeat
        if actual[i] == prev[i]:
            methods['Repeat (=last)'][i] += 1

# Print results
print(f"  {'Phương pháp':<22} │", end='')
for i in range(6):
    print(f"  C{i+1:>1}   ", end='')
print(f" │ {'TB':>6}")
print(f"  {'─'*22} │{'─'*48} │ {'─'*6}")

best_method_per_col = {}
for method, results in methods.items():
    print(f"  {method:<22} │", end='')
    avg_pct = 0
    for i in range(6):
        pct = results[i] / n_test * 100
        avg_pct += pct
        marker = ''
        print(f" {pct:>5.1f}%", end='')
    avg_pct /= 6
    print(f" │ {avg_pct:>5.1f}%")

# ═══════════════════════════════════════
# E. TÍNH ỔN ĐỊNH (STREAKS)
# ═══════════════════════════════════════
print(f"\n{'━'*100}")
print(f"  📊 E. TÍNH ỔN ĐỊNH — Cột nào hay lặp lại?")
print(f"{'━'*100}\n")

print(f"  {'Cột':>4} │ {'Repeat rate':>12} │ {'Δ≤1 rate':>10} │ {'Δ≤2 rate':>10} │ {'Δ≤3 rate':>10} │ "
      f"{'Max streak':>11} │ {'TB streak':>10}")
print(f"  {'─'*4} │ {'─'*12} │ {'─'*10} │ {'─'*10} │ {'─'*10} │ {'─'*11} │ {'─'*10}")

stability_scores = []
for i in range(6):
    col = columns[i]
    
    # Repeat rate (exact same value)
    repeats = sum(1 for j in range(1, N) if col[j] == col[j-1])
    repeat_rate = repeats / (N-1) * 100
    
    # Delta ≤ k rates
    d1 = sum(1 for j in range(1, N) if abs(col[j] - col[j-1]) <= 1) / (N-1) * 100
    d2 = sum(1 for j in range(1, N) if abs(col[j] - col[j-1]) <= 2) / (N-1) * 100
    d3 = sum(1 for j in range(1, N) if abs(col[j] - col[j-1]) <= 3) / (N-1) * 100
    
    # Streak analysis (consecutive same value)
    max_streak = 0
    streaks = []
    cur_streak = 1
    for j in range(1, N):
        if col[j] == col[j-1]:
            cur_streak += 1
        else:
            if cur_streak > 1:
                streaks.append(cur_streak)
            cur_streak = 1
    if cur_streak > 1:
        streaks.append(cur_streak)
    max_streak = max(streaks) if streaks else 0
    avg_streak = np.mean(streaks) if streaks else 0
    
    stability = repeat_rate + d2/2  # composite score
    stability_scores.append({'idx': i, 'stability': stability, 'repeat': repeat_rate, 'd2': d2})
    
    print(f"  C{i+1:>2} │ {repeat_rate:>11.1f}% │ {d1:>9.1f}% │ {d2:>9.1f}% │ {d3:>9.1f}% │ "
          f"{max_streak:>11} │ {avg_streak:>9.1f}")

print(f"\n  📌 Xếp hạng ỔN ĐỊNH nhất (hay repeat/gần nhau):")
ranked_stab = sorted(stability_scores, key=lambda x: -x['stability'])
for rank, ss in enumerate(ranked_stab):
    label = "🟢 ỔN ĐỊNH NHẤT" if rank == 0 else ("🔴 BIẾN ĐỘNG NHẤT" if rank == 5 else "")
    print(f"    {rank+1}. C{ss['idx']+1} — repeat={ss['repeat']:.1f}%, Δ≤2={ss['d2']:.1f}% {label}")

# ═══════════════════════════════════════
# F. MARKOV TRANSITION — Cột nào có pattern?
# ═══════════════════════════════════════
print(f"\n{'━'*100}")
print(f"  📊 F. MARKOV — Cột nào có xu hướng chuyển đổi rõ ràng?")
print(f"{'━'*100}\n")

for i in range(6):
    col = columns[i]
    # Build transition matrix
    transitions = defaultdict(Counter)
    for j in range(1, N):
        transitions[col[j-1]][col[j]] += 1
    
    # For each state, what's the max probability of transitioning to a specific next state?
    max_probs = []
    for state, next_counts in transitions.items():
        total = sum(next_counts.values())
        max_next = next_counts.most_common(1)[0]
        max_probs.append(max_next[1] / total)
    
    avg_max_prob = np.mean(max_probs) * 100
    max_max_prob = max(max_probs) * 100
    
    # Predictability: if we always guess the most likely next value, how often are we right?
    correct = 0
    for j in range(WARMUP, N):
        prev_val = col[j-1]
        if prev_val in transitions:
            predicted = transitions[prev_val].most_common(1)[0][0]
            if col[j] == predicted:
                correct += 1
    markov_accuracy = correct / n_test * 100
    
    print(f"  C{i+1}: Markov accuracy = {markov_accuracy:.1f}%, "
          f"avg max transition prob = {avg_max_prob:.1f}%, "
          f"max = {max_max_prob:.1f}%")

# ═══════════════════════════════════════
# G. TỔNG HỢP — BẢNG XẾP HẠNG TOÀN DIỆN
# ═══════════════════════════════════════
print(f"\n{'━'*100}")
print(f"  ⭐ G. TỔNG HỢP — BẢNG XẾP HẠNG DỄ → KHÓ")
print(f"{'━'*100}\n")

# Calculate composite scores for each column
final_scores = []
for i in range(6):
    cs = col_stats[i]
    rd = range_data[i]
    ss = stability_scores[i]
    
    # Mode accuracy
    mode_acc = methods['Mode (1 số)'][i] / n_test * 100
    top5_acc = methods['Top-5 (5 số)'][i] / n_test * 100
    
    # Composite: lower entropy + narrower range + higher repeat + higher mode accuracy = easier
    score = (
        (1 - cs['entropy_ratio']) * 25 +  # Lower entropy = easier
        (1 - rd['width90'] / 35) * 25 +    # Narrower range = easier  
        ss['repeat'] * 1.0 +               # Higher repeat rate = easier
        mode_acc * 1.5 +                    # Higher mode accuracy = easier
        top5_acc * 0.5                      # Higher top5 accuracy = easier
    )
    
    final_scores.append({
        'idx': i,
        'score': score,
        'entropy': cs['entropy'],
        'range': rd['width90'], 
        'repeat': ss['repeat'],
        'mode_acc': mode_acc,
        'top5_acc': top5_acc,
        'p90': rd['p90'],
        'avg': cs['avg'],
        'std': cs['std'],
    })

final_ranked = sorted(final_scores, key=lambda x: -x['score'])

print(f"  {'Hạng':>4} {'Cột':>4} │ {'Score':>6} │ {'Entropy':>8} │ {'90% Range':>12} │ "
      f"{'Width':>6} │ {'Repeat%':>8} │ {'Mode%':>6} │ {'Top5%':>6} │ {'StdDev':>7} │ {'Đánh giá'}")
print(f"  {'─'*4} {'─'*4} │ {'─'*6} │ {'─'*8} │ {'─'*12} │ {'─'*6} │ {'─'*8} │ {'─'*6} │ {'─'*6} │ {'─'*7} │ {'─'*15}")

for rank, fs in enumerate(final_ranked):
    i = fs['idx']
    if rank == 0:
        label = "🟢 DỄ NHẤT"
    elif rank == 1:
        label = "🟢 DỄ"
    elif rank <= 3:
        label = "🟡 VỪA"
    elif rank == 4:
        label = "🔴 KHÓ"
    else:
        label = "🔴 KHÓ NHẤT"
    
    print(f"  {rank+1:>4} C{i+1:>2} │ {fs['score']:>6.1f} │ {fs['entropy']:>7.2f}b │ "
          f"[{fs['p90'][0]:>3}-{fs['p90'][1]:>3}]    │ {fs['range']:>6} │ {fs['repeat']:>7.1f}% │ "
          f"{fs['mode_acc']:>5.1f}% │ {fs['top5_acc']:>5.1f}% │ {fs['std']:>7.1f} │ {label}")

# ═══════════════════════════════════════
# H. CHIẾN LƯỢC GỢI Ý — 10 VÉ / KỲ
# ═══════════════════════════════════════
print(f"\n{'━'*100}")
print(f"  💡 H. CHIẾN LƯỢC 10 VÉ / KỲ — Dựa trên phân tích")
print(f"{'━'*100}")

# Categorize columns
easy = final_ranked[0]  # Easiest
easy2 = final_ranked[1]
hard = final_ranked[-1]  # Hardest

print(f"""
  Dựa trên phân tích {N} kỳ quay:

  ┌────────────────────────────────────────────────────────────────────────┐
  │ CỘT DỄ NHẤT: C{easy['idx']+1}                                                     │
  │   • Entropy: {easy['entropy']:.2f}b (thấp nhất)                                   │
  │   • 90% range: [{easy['p90'][0]}-{easy['p90'][1]}] (chỉ {easy['range']} số)                             │
  │   • Repeat rate: {easy['repeat']:.1f}% (hay lặp lại)                             │
  │   • Mode accuracy: {easy['mode_acc']:.1f}% (đoán đúng bằng mode)                   │
  │   • Top 5 accuracy: {easy['top5_acc']:.1f}% (nằm trong top 5 hay nhất)              │
  │                                                                        │
  │ CỘT KHÓ NHẤT: C{hard['idx']+1}                                                    │
  │   • Entropy: {hard['entropy']:.2f}b (cao nhất)                                    │
  │   • 90% range: [{hard['p90'][0]}-{hard['p90'][1]}] ({hard['range']} số)                                │
  │   • Repeat rate: {hard['repeat']:.1f}%                                            │
  │   • Mode accuracy: {hard['mode_acc']:.1f}%                                          │
  └────────────────────────────────────────────────────────────────────────┘
""")

# Strategy
print(f"  📌 CHIẾN LƯỢC GỢI Ý VỚI 10 VÉ:")
print(f"""
  1. CỐ ĐỊNH cột DỄ (C{easy['idx']+1}, C{easy2['idx']+1}):
     • C{easy['idx']+1}: Chọn Mode hoặc Top-3 → đúng {methods['Top-3 (3 số)'][easy['idx']]/n_test*100:.0f}% thời gian
     • C{easy2['idx']+1}: Chọn Mode hoặc Top-3 → đúng {methods['Top-3 (3 số)'][easy2['idx']]/n_test*100:.0f}% thời gian
     → Fix 2 cột dễ = giảm không gian tìm kiếm

  2. MỞ RỘNG cột KHÓ:
     • Dùng 90% range cho cột khó → cover nhiều giá trị hơn
     • Spread 10 vé ra cover nhiều tổ hợp cột khó

  3. CHIẾN THUẬT CỤ THỂ VỚI 10 VÉ:
""")

# Actually calculate: given the best strategy, how many matches can we get?
# Fix easy columns, vary hard columns
match_4plus = 0
match_3plus = 0
match_2plus = 0

for ti in range(n_test):
    te = WARMUP + ti
    actual = sorted(data[te][:6])
    
    train_cols = [[] for _ in range(6)]
    for d in data[te-200:te]:
        s = sorted(d[:6])
        for j in range(6):
            train_cols[j].append(s[j])
    
    # Strategy: for each column, pick top-K most likely values
    # Then generate 10 combos covering them
    predicted_ranges = []
    for j in range(6):
        col = train_cols[j]
        freq = Counter(col)
        # For easy columns: top 3, for hard: top 7
        k = 3 if col_stats[j]['entropy_ratio'] < 0.9 else 5
        top_vals = sorted([v for v, _ in freq.most_common(k)])
        predicted_ranges.append(set(top_vals))
    
    # Check: how many columns have actual value in predicted range?
    matches_in_range = sum(1 for j in range(6) if actual[j] in predicted_ranges[j])
    if matches_in_range >= 4: match_4plus += 1
    if matches_in_range >= 3: match_3plus += 1
    if matches_in_range >= 2: match_2plus += 1

print(f"     Nếu dùng Top-K per column (K=3 dễ, K=5 khó):")
print(f"     • Cả 6 số nằm trong range: phải tính combo")
print(f"     • ≥ 4 cột nằm trong predicted range: {match_4plus}/{n_test} = {match_4plus/n_test*100:.1f}%")
print(f"     • ≥ 3 cột nằm trong predicted range: {match_3plus}/{n_test} = {match_3plus/n_test*100:.1f}%")
print(f"     • ≥ 2 cột nằm trong predicted range: {match_2plus}/{n_test} = {match_2plus/n_test*100:.1f}%")

# Final practical analysis: 10 tickets strategy backtest
print(f"\n  📌 BACKTEST: 10 vé tốt nhất mỗi kỳ → trúng bao nhiêu?")

match_dist = Counter()
for ti in range(n_test):
    te = WARMUP + ti
    actual = set(data[te][:6])
    actual_sorted = sorted(data[te][:6])
    
    train_cols = [[] for _ in range(6)]
    for d in data[te-200:te]:
        s = sorted(d[:6])
        for j in range(6):
            train_cols[j].append(s[j])
    
    # Generate 10 best combos
    # Strategy: fix mode for easy cols, vary top values for hard cols
    candidates = []
    for j in range(6):
        col = train_cols[j]
        freq = Counter(col)
        candidates.append([v for v, _ in freq.most_common(5)])
    
    # Generate combos: take top-1 for each column as base, then vary
    combos = set()
    # Base combo: mode of each column
    base = tuple(candidates[j][0] for j in range(6))
    if len(set(base)) == 6 and all(1 <= x <= 45 for x in base):
        combos.add(tuple(sorted(base)))
    
    # Vary each column one at a time
    for j in range(6):
        for alt_idx in range(1, min(3, len(candidates[j]))):
            combo = list(base)
            combo[j] = candidates[j][alt_idx]
            combo_sorted = tuple(sorted(set(combo)))
            if len(combo_sorted) == 6 and all(1 <= x <= 45 for x in combo_sorted):
                combos.add(combo_sorted)
    
    # Also try: vary the 2 hardest columns together
    hard_cols = [fs['idx'] for fs in final_ranked[-2:]]
    for a in range(min(3, len(candidates[hard_cols[0]]))):
        for b in range(min(3, len(candidates[hard_cols[1]]))):
            combo = list(base)
            combo[hard_cols[0]] = candidates[hard_cols[0]][a]
            combo[hard_cols[1]] = candidates[hard_cols[1]][b]
            combo_sorted = tuple(sorted(set(combo)))
            if len(combo_sorted) == 6 and all(1 <= x <= 45 for x in combo_sorted):
                combos.add(combo_sorted)
    
    # Take best 10
    tickets = list(combos)[:10]
    
    best_match = max((len(actual & set(t)) for t in tickets), default=0)
    match_dist[best_match] += 1

print(f"\n     Kết quả với 10 vé thông minh (mode + variation):")
print(f"     {'Trúng':>8} │ {'Số kỳ':>8} │ {'Tỷ lệ':>8}")
print(f"     {'─'*8} │ {'─'*8} │ {'─'*8}")
for k in range(7):
    cnt = match_dist.get(k, 0)
    pct = cnt / n_test * 100
    label = {0: '', 1: '', 2: '', 3: 'Good', 4: 'Very Good', 5: 'EXCELLENT', 6: 'JACKPOT!'}[k]
    print(f"     {k}/6      │ {cnt:>8} │ {pct:>7.2f}%  {label}")

cum_3 = sum(match_dist.get(k, 0) for k in range(3, 7))
cum_4 = sum(match_dist.get(k, 0) for k in range(4, 7))
print(f"\n     ≥3/6: {cum_3}/{n_test} = {cum_3/n_test*100:.1f}%")
print(f"     ≥4/6: {cum_4}/{n_test} = {cum_4/n_test*100:.1f}%")

print(f"\n{'═'*100}")
