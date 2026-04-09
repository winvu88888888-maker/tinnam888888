"""
V9 BLOCK PUZZLE ANALYSIS — Mỗi kỳ = 1 mảnh ghép LEGO

Ý tưởng: Mỗi kỳ quay = 1 khối có 6 vị trí (sorted).
Phân tích NGƯỢC xem:
  1. POSITION TRANSITIONS: Vị trí 1 của khối N → Vị trí 1 của khối N+1?
     (Ví dụ: nếu vị trí 1 = 3 → vị trí 1 tiếp theo thường ở range nào?)
  2. BRIDGE NUMBERS: Số nào đóng vai trò "cầu nối" giữa 2 khối?
     (Số lặp lại, hoặc số ở vị trí "chuyển tiếp")
  3. BLOCK SIGNATURE: "Hình dạng" mỗi khối = pattern decade
     (VD: [2,15,23,31,38,44] → shape "1-10-20-30-30-40" → decade [1,1,1,1,1,1])
  4. SHAPE FLOW: Hình dạng khối N → Hình dạng khối N+1 có mẫu không?
  5. GAP PATTERN: Khoảng cách giữa các vị trí trong 1 khối
  6. HEAD/TAIL LINKAGE: Đầu số & cuối số khối N → khối N+1
"""
import sys, os, time
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter, defaultdict
from scraper.data_manager import get_mega645_numbers

data = get_mega645_numbers()
total = len(data)
print(f"Data: {total} draws\n")

# Sort each draw for positional analysis
sorted_data = [sorted(d[:6]) for d in data]

# ============================================
# 1. POSITION-BY-POSITION TRANSITIONS
# ============================================
print(f"{'='*70}")
print(f" 1. POSITIONAL TRANSITIONS (pos[i] of draw N → pos[i] of draw N+1)")
print(f"{'='*70}\n")

for pos in range(6):
    values = [sd[pos] for sd in sorted_data]
    transitions = [(values[i], values[i+1]) for i in range(len(values)-1)]
    diffs = [v2-v1 for v1,v2 in transitions]
    
    print(f"  Position {pos+1} (avg={np.mean(values):.1f}, std={np.std(values):.1f}):")
    print(f"    Diff (next-prev): mean={np.mean(diffs):+.1f}, std={np.std(diffs):.1f}")
    print(f"    P10={int(np.percentile(diffs,10)):+d}, P25={int(np.percentile(diffs,25)):+d}, "
          f"P50={int(np.percentile(diffs,50)):+d}, P75={int(np.percentile(diffs,75)):+d}, "
          f"P90={int(np.percentile(diffs,90)):+d}")
    
    # Conditional: given this position's value, what's next?
    buckets = defaultdict(list)
    for v1, v2 in transitions:
        buckets[v1].append(v2)
    # Show top transitions for common values
    common_vals = sorted(Counter(values).most_common(5))
    
    # How predictable is this position?
    # If we predict pos[i] stays in range [prev-3, prev+3], how often right?
    in_range = sum(1 for d in diffs if -3 <= d <= 3)
    print(f"    Within ±3 of previous: {in_range}/{len(diffs)} = {in_range/len(diffs)*100:.1f}%")
    print()

# ============================================
# 2. BRIDGE NUMBERS (cầu nối)
# ============================================
print(f"{'='*70}")
print(f" 2. BRIDGE NUMBERS — Cầu nối giữa các khối")
print(f"{'='*70}\n")

# Which numbers most often appear in CONSECUTIVE draws (bridge role)?
bridge_count = Counter()
bridge_position = defaultdict(list)  # num → positions it bridges from/to

for i in range(total-1):
    s1, s2 = set(sorted_data[i]), set(sorted_data[i+1])
    bridges = s1 & s2  # Numbers in BOTH draws
    for b in bridges:
        bridge_count[b] += 1
        pos1 = sorted_data[i].index(b)
        pos2 = sorted_data[i+1].index(b)
        bridge_position[b].append((pos1, pos2))

print(f"  Most frequent bridge numbers (appear in consecutive draws):")
for num, cnt in bridge_count.most_common(15):
    rate = cnt / (total-1) * 100
    avg_pos = np.mean([p1 for p1,p2 in bridge_position[num]])
    print(f"    #{num:2d}: bridges {cnt} times ({rate:.1f}%), avg position={avg_pos:.1f}")

# Bridge position transitions
print(f"\n  Bridge position flow (which pos → which pos):")
pos_flow = Counter()
for num, positions in bridge_position.items():
    for p1, p2 in positions:
        pos_flow[(p1, p2)] += 1
for (p1, p2), cnt in pos_flow.most_common(10):
    print(f"    Pos {p1+1} → Pos {p2+1}: {cnt} times ({cnt/sum(pos_flow.values())*100:.1f}%)")

# ============================================
# 3. BLOCK SIGNATURES (hình dạng mỗi khối)
# ============================================
print(f"\n{'='*70}")
print(f" 3. BLOCK SIGNATURES — Hình dạng mỗi khối")
print(f"{'='*70}\n")

def get_decade(n):
    if n <= 9: return 0
    elif n <= 19: return 1
    elif n <= 29: return 2
    elif n <= 39: return 3
    else: return 4

def get_signature(draw):
    """Block signature = tuple of decades for each position."""
    return tuple(get_decade(n) for n in sorted(draw[:6]))

signatures = [get_signature(d) for d in data]
sig_count = Counter(signatures)

print(f"  Total unique signatures: {len(sig_count)}")
print(f"  Top 15 most common block shapes:")
for sig, cnt in sig_count.most_common(15):
    dec_names = ['0x','1x','2x','3x','4x']
    shape = '-'.join(dec_names[s] for s in sig)
    pct = cnt / total * 100
    print(f"    [{shape}]: {cnt} times ({pct:.1f}%)")

# ============================================
# 4. SIGNATURE FLOW — Shape N → Shape N+1
# ============================================
print(f"\n{'='*70}")
print(f" 4. SIGNATURE FLOW — Hình dạng khối N → N+1")
print(f"{'='*70}\n")

sig_transitions = Counter()
for i in range(total-1):
    sig_transitions[(signatures[i], signatures[i+1])] += 1

print(f"  Total unique transitions: {len(sig_transitions)}")
print(f"  Top 10 most common shape flows:")
dec_names = ['0x','1x','2x','3x','4x']
for (s1, s2), cnt in sig_transitions.most_common(10):
    sh1 = '-'.join(dec_names[s] for s in s1)
    sh2 = '-'.join(dec_names[s] for s in s2)
    pct = cnt / (total-1) * 100
    print(f"    [{sh1}] → [{sh2}]: {cnt} ({pct:.1f}%)")

# Most predictable transitions: given shape X, what's most likely next shape?
print(f"\n  Best predictable shape (highest conditional P for next shape):")
best_pred = []
for sig, cnt in sig_count.most_common(30):
    if cnt < 10: continue
    next_sigs = Counter()
    for i in range(total-1):
        if signatures[i] == sig:
            next_sigs[signatures[i+1]] += 1
    top_next, top_cnt = next_sigs.most_common(1)[0]
    total_from = sum(next_sigs.values())
    cond_p = top_cnt / total_from
    best_pred.append((sig, top_next, cond_p, top_cnt, total_from))

best_pred.sort(key=lambda x: -x[2])
for sig, next_sig, cond_p, cnt, total_from in best_pred[:10]:
    sh1 = '-'.join(dec_names[s] for s in sig)
    sh2 = '-'.join(dec_names[s] for s in next_sig)
    print(f"    [{sh1}] → [{sh2}]: {cond_p*100:.1f}% ({cnt}/{total_from})")

# ============================================
# 5. GAP PATTERN within blocks
# ============================================
print(f"\n{'='*70}")
print(f" 5. INTERNAL GAPS — Khoảng cách trong mỗi khối")
print(f"{'='*70}\n")

gaps_by_pos = defaultdict(list)  # gap between pos[i] and pos[i+1]
for sd in sorted_data:
    for i in range(5):
        gaps_by_pos[i].append(sd[i+1] - sd[i])

print(f"  Gap between consecutive positions:")
for i in range(5):
    g = gaps_by_pos[i]
    print(f"    Gap {i+1}-{i+2}: mean={np.mean(g):.1f}, median={np.median(g):.0f}, "
          f"P10={int(np.percentile(g,10))}, P90={int(np.percentile(g,90))}")

# Gap transitions
print(f"\n  Gap pattern flow (total gap of block N → block N+1):")
total_gaps = [max(sd)-min(sd) for sd in sorted_data]
gap_diffs = [total_gaps[i+1]-total_gaps[i] for i in range(len(total_gaps)-1)]
print(f"    Total range: mean={np.mean(total_gaps):.1f}, std={np.std(total_gaps):.1f}")
print(f"    Range change: mean={np.mean(gap_diffs):+.1f}, std={np.std(gap_diffs):.1f}")

# ============================================
# 6. HEAD/TAIL LINKAGE (đầu số/cuối số)
# ============================================
print(f"\n{'='*70}")
print(f" 6. HEAD/TAIL LINKAGE — Đầu số & Cuối số")
print(f"{'='*70}\n")

heads = [sd[0] for sd in sorted_data]
tails = [sd[5] for sd in sorted_data]

# Head N → Head N+1
hh_diffs = [heads[i+1]-heads[i] for i in range(len(heads)-1)]
tt_diffs = [tails[i+1]-tails[i] for i in range(len(tails)-1)]

print(f"  HEAD (smallest number):")
print(f"    mean={np.mean(heads):.1f}, range=[{min(heads)},{max(heads)}]")
print(f"    Head→Head diff: mean={np.mean(hh_diffs):+.1f}, P25={int(np.percentile(hh_diffs,25)):+d}, "
      f"P75={int(np.percentile(hh_diffs,75)):+d}")
print(f"    Within ±2: {sum(1 for d in hh_diffs if -2 <= d <= 2)}/{len(hh_diffs)} "
      f"= {sum(1 for d in hh_diffs if -2 <= d <= 2)/len(hh_diffs)*100:.1f}%")

print(f"\n  TAIL (largest number):")
print(f"    mean={np.mean(tails):.1f}, range=[{min(tails)},{max(tails)}]")
print(f"    Tail→Tail diff: mean={np.mean(tt_diffs):+.1f}, P25={int(np.percentile(tt_diffs,25)):+d}, "
      f"P75={int(np.percentile(tt_diffs,75)):+d}")
print(f"    Within ±3: {sum(1 for d in tt_diffs if -3 <= d <= 3)}/{len(tt_diffs)} "
      f"= {sum(1 for d in tt_diffs if -3 <= d <= 3)/len(tt_diffs)*100:.1f}%")

# Tail-to-Head bridge (tail of N → head of N+1)
th_vals = [(tails[i], heads[i+1]) for i in range(total-1)]
print(f"\n  TAIL→HEAD bridge (end of block → start of next):")
print(f"    Diff (head_next - tail_prev): mean={np.mean([h-t for t,h in th_vals]):+.1f}")
print(f"    Always negative (reset): {sum(1 for t,h in th_vals if h < t)}/{len(th_vals)} "
      f"= {sum(1 for t,h in th_vals if h < t)/len(th_vals)*100:.1f}%")

# ============================================
# 7. POSITIONAL PREDICTION POWER
# ============================================
print(f"\n{'='*70}")
print(f" 7. POSITIONAL PREDICTION — Can we predict next block from current?")
print(f"{'='*70}\n")

# For each position, if we use [prev_pos - 5, prev_pos + 5] as range,
# what % of time does the actual next value fall in this range?
for pos in range(6):
    values = [sd[pos] for sd in sorted_data]
    correct_5 = 0
    correct_3 = 0
    correct_same_decade = 0
    for i in range(len(values)-1):
        v1, v2 = values[i], values[i+1]
        if abs(v2-v1) <= 5: correct_5 += 1
        if abs(v2-v1) <= 3: correct_3 += 1
        if get_decade(v1) == get_decade(v2): correct_same_decade += 1
    n = len(values)-1
    print(f"  Pos {pos+1}: within ±3={correct_3/n*100:.1f}%, ±5={correct_5/n*100:.1f}%, "
          f"same_decade={correct_same_decade/n*100:.1f}%")

# ============================================
# 8. BLOCK LINKING KEY — Which number connects blocks?
# ============================================
print(f"\n{'='*70}")
print(f" 8. BLOCK LINKING KEY — Mỗi con số có 'sứ mệnh' riêng")
print(f"{'='*70}\n")

# For each number 1-45: what's its role? Bridge? Head? Tail? Internal?
for num in range(1, 46):
    n_bridge = bridge_count.get(num, 0)
    n_head = sum(1 for sd in sorted_data if sd[0] == num)
    n_tail = sum(1 for sd in sorted_data if sd[5] == num)
    n_appear = sum(1 for d in data if num in d)
    n_internal = n_appear - n_head - n_tail
    role = 'HEAD' if n_head > n_tail and n_head > n_internal/4 else \
           'TAIL' if n_tail > n_head and n_tail > n_internal/4 else \
           'BRIDGE' if n_bridge > n_appear * 0.08 else 'INTERNAL'
    if num <= 10 or num >= 40 or n_bridge > 15:
        print(f"  #{num:2d}: appear={n_appear}, head={n_head}, tail={n_tail}, "
              f"bridge={n_bridge}, role={role}")

print(f"\n{'='*70}")
print(f" CONCLUSION — Key patterns for V9 engine")
print(f"{'='*70}")
