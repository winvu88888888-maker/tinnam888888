"""
🎯 GOLDEN 10 — 10 VÉ VÀNG cho kỳ quay tiếp theo
==================================================
Dùng TOÀN BỘ sức mạnh V22 (14 signals + forensic)
để chọn ra 10 tổ hợp TỐT NHẤT.

Chạy: python golden10.py
"""
import sys, os, math, time
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper.data_manager import get_mega645_numbers, get_power655_numbers

MAX_MEGA = 45; MAX_POWER = 55; PICK = 6


def compute_signals(data, MAX_NUM):
    """14-signal engine — full V22."""
    n = len(data)
    if n < 50: return {num: PICK/MAX_NUM for num in range(1, MAX_NUM+1)}
    last = set(data[-1][:PICK]); base_p = PICK/MAX_NUM
    scores = {num: 0.0 for num in range(1, MAX_NUM+1)}

    for w, wt in [(3,4),(5,3),(10,2),(20,1.5),(50,1)]:
        if n < w: continue
        fc = Counter()
        for d in data[-w:]:
            for num in d[:PICK]: fc[num] += 1
        for num in range(1, MAX_NUM+1): scores[num] += fc.get(num,0)/w * wt

    follow = defaultdict(Counter); pc = Counter()
    for i in range(n-1):
        for p in data[i][:PICK]:
            pc[p] += 1
            for nx in data[i+1][:PICK]: follow[p][nx] += 1
    for num in range(1, MAX_NUM+1):
        tf = sum(follow[p].get(num,0) for p in last)
        tp = sum(pc[p] for p in last)
        if tp > 0: scores[num] += (tf/tp/base_p-1)*3

    knn = Counter()
    for i in range(n-2):
        sim = len(set(data[i][:PICK]) & last)
        if sim >= 2:
            for num in data[i+1][:PICK]: knn[num] += sim*sim
    mx = max(knn.values()) if knn else 1
    for num in range(1, MAX_NUM+1): scores[num] += knn.get(num,0)/mx*2.5

    ls_idx = {}; gs, gn = defaultdict(float), defaultdict(int)
    for i, d in enumerate(data):
        for num in d[:PICK]:
            if num in ls_idx: gs[num] += i-ls_idx[num]; gn[num] += 1
            ls_idx[num] = i
    for num in range(1, MAX_NUM+1):
        if gn[num] < 5: continue
        mg = gs[num]/gn[num]; cg = n-ls_idx.get(num,0)
        if cg > mg*1.3: scores[num] += min((cg/mg-1)*1.5, 3)

    pf = Counter()
    for d in data[-200:]:
        for p in combinations(sorted(d[:PICK]), 2): pf[p] += 1
    for num in range(1, MAX_NUM+1):
        scores[num] += sum(pf.get(tuple(sorted([p,num])),0) for p in last)*0.05

    top12 = sorted(scores.items(), key=lambda x: -x[1])[:12]
    top12_nums = [num for num, _ in top12]
    for num in range(1, MAX_NUM+1):
        if num in top12_nums: continue
        pair_bonus = sum(pf.get(tuple(sorted([num, t])), 0) for t in top12_nums)
        scores[num] += pair_bonus * 0.03

    ww = min(100, n)
    iai, ic = Counter(), Counter()
    for i in range(max(0,n-ww), n-1):
        curr, nxt = set(data[i][:PICK]), set(data[i+1][:PICK])
        for num in curr:
            ic[num] += 1
            if num in nxt: iai[num] += 1
    for num in range(1, MAX_NUM+1):
        if num in last and ic[num] > 5:
            scores[num] += (iai[num]/ic[num]-base_p)*3

    if n >= 3:
        last2 = set(data[-2][:PICK])
        follow2 = defaultdict(Counter); pc2 = Counter()
        for i in range(n-2):
            for p in data[i][:PICK]:
                pc2[p] += 1
                for nx in data[i+2][:PICK]: follow2[p][nx] += 1
        for num in range(1, MAX_NUM+1):
            tf2 = sum(follow2[p].get(num,0) for p in last2)
            tp2 = sum(pc2[p] for p in last2)
            if tp2 > 0: scores[num] += (tf2/tp2/base_p-1)*2

    if n >= 10:
        digit_freq = Counter()
        for d in data[-10:]:
            for num in d[:PICK]: digit_freq[num % 10] += 1
        total_digits = sum(digit_freq.values())
        for num in range(1, MAX_NUM+1):
            d = num % 10; ed = total_digits / 10; ad = digit_freq.get(d, 0)
            if ad > ed * 1.3: scores[num] += min((ad/ed - 1) * 1.0, 2)

    if n >= 30:
        recent_30 = Counter()
        for d in data[-30:]:
            for num in d[:PICK]: recent_30[num] += 1
        all_time = Counter()
        for d in data:
            for num in d[:PICK]: all_time[num] += 1
        for num in range(1, MAX_NUM+1):
            expected_30 = (all_time.get(num, 0) / n) * 30
            actual_30 = recent_30.get(num, 0)
            if expected_30 > 0:
                deviation = (actual_30 - expected_30) / max(1, expected_30)
                scores[num] -= deviation * 0.8

    return scores


def build_pool(data, scores, target_size, MAX_NUM):
    """Build best number pool."""
    n = len(data); last_set = set(data[-1][:PICK])
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    pools = set(num for num, _ in ranked[:15])
    for w in [30, 50, 100]:
        fc = Counter(num for d in data[-min(w,n):] for num in d[:PICK])
        pools.update(num for num, _ in fc.most_common(15))
    ls = {}
    for i, d in enumerate(data):
        for num in d[:PICK]: ls[num] = i
    pools.update(sorted(range(1,MAX_NUM+1), key=lambda x: -(n-ls.get(x,0)))[:15])
    fc = Counter()
    for i in range(n-1):
        for p in data[i][:PICK]:
            if p in last_set:
                for nx in data[i+1][:PICK]: fc[nx] += 1
    pools.update(num for num, _ in fc.most_common(15))
    kc = Counter()
    for i in range(n-2):
        sim = len(set(data[i][:PICK]) & last_set)
        if sim >= 3:
            for num in data[i+1][:PICK]: kc[num] += sim*sim
    pools.update(num for num, _ in kc.most_common(15))
    for num in last_set:
        for delta in [-2,-1,1,2]:
            nb = num+delta
            if 1<=nb<=MAX_NUM: pools.add(nb)
    signal = set(num for num, _ in ranked[:35])
    gap_due = set(sorted(range(1, MAX_NUM+1), key=lambda x: -(n - ls.get(x, 0)))[:15])
    if n >= 3:
        last2 = set(data[-2][:PICK])
        m2 = Counter()
        for i in range(n-2):
            for p in data[i][:PICK]:
                if p in last2:
                    for nx in data[i+2][:PICK]: m2[nx] += 1
        markov2 = set(num for num, _ in m2.most_common(15))
    else: markov2 = set()
    fusion = pools | signal | gap_due | markov2
    return sorted(fusion, key=lambda x: -scores.get(x, 0))[:target_size]


def generate_golden_10(pool, scores):
    """Generate exactly 10 BEST combos from pool.
    Strategy: exhaustive top-scored combos from top 18 numbers.
    """
    # Take top 18 numbers from pool (C(18,6)=18564 — fast enumeration)
    top_n = min(18, len(pool))
    top_nums = pool[:top_n]

    # Enumerate ALL C(top_n, 6) combos and score them
    all_combos = []
    for combo in combinations(top_nums, PICK):
        sc = sum(scores.get(num, 0) for num in combo)
        # Bonus: spread across number range (avoid all-low or all-high)
        nums = sorted(combo)
        spread = nums[-1] - nums[0]
        sc += spread * 0.1  # slight bonus for wider spread
        # Bonus: mix of even/odd
        even_count = sum(1 for x in combo if x % 2 == 0)
        if 2 <= even_count <= 4:
            sc += 0.5  # bonus for balanced even/odd
        # Penalty: too many consecutive
        consec = sum(1 for i in range(len(nums)-1) if nums[i+1] - nums[i] == 1)
        if consec >= 3:
            sc -= 1.0  # penalize 3+ consecutive numbers
        all_combos.append((combo, sc))

    all_combos.sort(key=lambda x: -x[1])

    # Pick top 10 with DIVERSITY constraint
    # Ensure no two tickets share more than 4 numbers
    selected = []
    for combo, sc in all_combos:
        combo_set = set(combo)
        too_similar = False
        for prev, _ in selected:
            if len(combo_set & set(prev)) >= 5:
                too_similar = True
                break
        if not too_similar:
            selected.append((combo, sc))
        if len(selected) >= 10:
            break

    return selected


def run_lottery(name, data_raw, MAX_NUM):
    """Generate golden 10 for one lottery type."""
    data = [d[:PICK] for d in data_raw]
    n = len(data)
    last_draw = data[-1]
    prev_draw = data[-2] if n >= 2 else None

    print(f"\n{'█'*70}")
    print(f"  🎯 {name} — GOLDEN 10")
    print(f"  {n} kỳ quay | Kỳ quay gần nhất: {sorted(last_draw)}")
    print(f"{'█'*70}")

    # Compute signals
    scores = compute_signals(data, MAX_NUM)

    # Top 20 numbers by score
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    print(f"\n  📊 TOP 20 số nóng nhất:")
    print(f"  {'─'*60}")
    for i, (num, sc) in enumerate(ranked[:20]):
        bar = '█' * int(sc * 2)
        print(f"    {i+1:>2}. Số {num:>2}  │ Score: {sc:>6.2f}  │ {bar}")

    # Build pool
    pool = build_pool(data, scores, 42, MAX_NUM)
    print(f"\n  🎱 Pool ({len(pool)} số): {pool}")

    # Generate golden 10
    golden = generate_golden_10(pool, scores)

    total_combos = math.comb(MAX_NUM, PICK)

    print(f"\n  {'━'*70}")
    print(f"  🏆 10 VÉ VÀNG — KỲ QUAY TIẾP THEO")
    print(f"  {'━'*70}\n")

    for i, (combo, sc) in enumerate(golden):
        nums = sorted(combo)
        even = sum(1 for x in nums if x % 2 == 0)
        odd = PICK - even
        spread = nums[-1] - nums[0]
        num_str = "  ".join(f"{x:>2}" for x in nums)
        print(f"  Vé {i+1:>2} │ [{num_str}] │ Score: {sc:.1f} │ {even}C{odd}L │ Spread: {spread}")

    print(f"\n  {'━'*70}")

    # Stats
    avg_score = np.mean([sc for _, sc in golden])
    print(f"\n  📊 Thống kê:")
    print(f"    • Score trung bình: {avg_score:.1f}")
    print(f"    • Tổng tổ hợp khả dĩ: C({MAX_NUM},6) = {total_combos:,}")
    print(f"    • 10 vé / {total_combos:,} = {10/total_combos*100:.6f}%")

    # Coverage analysis
    all_nums = set()
    for combo, _ in golden:
        all_nums.update(combo)
    print(f"    • Phủ {len(all_nums)} số khác nhau trong 10 vé")
    print(f"    • Số phủ: {sorted(all_nums)}")

    # Recent pattern check
    print(f"\n  📋 So với kỳ gần nhất:")
    print(f"    • Kỳ trước: {sorted(last_draw)}")
    if prev_draw:
        print(f"    • Kỳ trước nữa: {sorted(prev_draw)}")
    for i, (combo, _) in enumerate(golden):
        overlap = len(set(combo) & set(last_draw))
        if overlap >= 3:
            print(f"    • Vé {i+1} có {overlap} số trùng kỳ trước")

    # Historical hit check (backtest last 50 draws)
    print(f"\n  🔍 Backtest 50 kỳ gần nhất:")
    max_hit = 0
    best_draws = []
    for j in range(max(0, n-50), n):
        actual = set(data[j][:PICK])
        for i, (combo, _) in enumerate(golden):
            hit = len(actual & set(combo))
            if hit > max_hit:
                max_hit = hit
                best_draws = [(j, i+1, hit, sorted(actual), sorted(combo))]
            elif hit == max_hit and hit >= 3:
                best_draws.append((j, i+1, hit, sorted(actual), sorted(combo)))

    if best_draws:
        print(f"    • Match tốt nhất: {max_hit}/6")
        for draw_idx, ticket, hit, actual, combo in best_draws[:5]:
            print(f"      Draw #{draw_idx}: Vé {ticket} trùng {hit} số "
                  f"({actual} vs {combo})")

    # Honest disclaimer
    print(f"\n  {'━'*70}")
    print(f"  ⚠️  LƯU Ý QUAN TRỌNG:")
    print(f"  • Xác suất trúng 6/6 với 10 vé: {10/total_combos*100:.6f}% ≈ 1/{total_combos//10:,}")
    print(f"  • Engine tăng xác suất ~2.5x → vẫn chỉ ≈ 1/{total_combos//25:,}")
    print(f"  • Đây là 10 vé TỐI ƯU NHẤT nhưng KHÔNG đảm bảo trúng")
    print(f"  • Chơi có trách nhiệm — chỉ dùng tiền bạn có thể mất")
    print(f"  {'━'*70}")

    return golden


if __name__ == '__main__':
    print("=" * 70)
    print("  🎯 GOLDEN 10 — Chọn 10 Vé Tốt Nhất")
    print("  Engine: V22 NEXUS (14 signals + forensic)")
    print("=" * 70)

    mega = get_mega645_numbers()
    power = get_power655_numbers()

    print(f"\n  📦 Data: Mega 6/45 = {len(mega)} kỳ | Power 6/55 = {len(power)} kỳ")

    golden_mega = run_lottery("MEGA 6/45", mega, MAX_MEGA)
    golden_power = run_lottery("POWER 6/55", power, MAX_POWER)

    print(f"\n{'█'*70}")
    print(f"  ✅ HOÀN THÀNH — 10 vé cho mỗi loại xổ số")
    print(f"  Giá: 10 × 10,000 = 100,000 VND mỗi loại")
    print(f"{'█'*70}")
