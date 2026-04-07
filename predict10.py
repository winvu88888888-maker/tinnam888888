"""
🎯 PREDICT 10 — Dự đoán 10 vé CHUẨN XÁC cho kỳ quay tiếp theo
================================================================
✅ DETERMINISTIC: Chạy bao nhiêu lần cũng ra ĐÚNG kết quả
✅ CACHE: Lưu file JSON — load lại không tính lại
✅ THEO NGÀY: Mỗi ngày quay có 1 bộ dự đoán duy nhất

Chạy: python predict10.py
"""
import sys, os, math, json, hashlib
from datetime import datetime, timedelta
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper.data_manager import get_mega645_numbers, get_power655_numbers

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predictions')
os.makedirs(CACHE_DIR, exist_ok=True)

MAX_MEGA = 45; MAX_POWER = 55; PICK = 6


# ═══════════════════════════════════════
# NGÀY QUAY VIETLOTT
# ═══════════════════════════════════════
# Mega 6/45: Thứ 4 (2), Thứ 6 (4), Chủ nhật (6)
# Power 6/55: Thứ 3 (1), Thứ 5 (3), Thứ 7 (5)

MEGA_DAYS = {2, 4, 6}   # Wed=2, Fri=4, Sun=6
POWER_DAYS = {1, 3, 5}  # Tue=1, Thu=3, Sat=5


def get_next_draw_date(draw_days, now=None):
    """Tìm ngày quay tiếp theo."""
    if now is None:
        now = datetime.now()
    
    # Nếu hôm nay là ngày quay và trước 18:30 → hôm nay
    if now.weekday() in draw_days and now.hour < 18:
        return now.date()
    
    # Tìm ngày quay tiếp theo
    for i in range(1, 8):
        candidate = now + timedelta(days=i)
        if candidate.weekday() in draw_days:
            return candidate.date()
    
    return (now + timedelta(days=1)).date()


def date_to_str(d):
    """Format ngày."""
    weekdays = ['Thứ 2', 'Thứ 3', 'Thứ 4', 'Thứ 5', 'Thứ 6', 'Thứ 7', 'Chủ nhật']
    return f"{weekdays[d.weekday()]} {d.strftime('%d/%m/%Y')}"


# ═══════════════════════════════════════
# 14-SIGNAL ENGINE (V22 NEXUS — 100% deterministic)
# ═══════════════════════════════════════

def compute_signals(data, MAX_NUM):
    """14-signal scoring — hoàn toàn deterministic, không random."""
    n = len(data)
    if n < 50: return {num: PICK/MAX_NUM for num in range(1, MAX_NUM+1)}
    last = set(data[-1][:PICK]); base_p = PICK/MAX_NUM
    scores = {num: 0.0 for num in range(1, MAX_NUM+1)}

    # Signal 1-5: Multi-window frequency
    for w, wt in [(3,4),(5,3),(10,2),(20,1.5),(50,1)]:
        if n < w: continue
        fc = Counter()
        for d in data[-w:]:
            for num in d[:PICK]: fc[num] += 1
        for num in range(1, MAX_NUM+1): scores[num] += fc.get(num,0)/w * wt

    # Signal 6: Markov-1 follow
    follow = defaultdict(Counter); pc = Counter()
    for i in range(n-1):
        for p in data[i][:PICK]:
            pc[p] += 1
            for nx in data[i+1][:PICK]: follow[p][nx] += 1
    for num in range(1, MAX_NUM+1):
        tf = sum(follow[p].get(num,0) for p in last)
        tp = sum(pc[p] for p in last)
        if tp > 0: scores[num] += (tf/tp/base_p-1)*3

    # Signal 7: KNN pattern
    knn = Counter()
    for i in range(n-2):
        sim = len(set(data[i][:PICK]) & last)
        if sim >= 2:
            for num in data[i+1][:PICK]: knn[num] += sim*sim
    mx = max(knn.values()) if knn else 1
    for num in range(1, MAX_NUM+1): scores[num] += knn.get(num,0)/mx*2.5

    # Signal 8: Gap-due
    ls_idx = {}; gs, gn = defaultdict(float), defaultdict(int)
    for i, d in enumerate(data):
        for num in d[:PICK]:
            if num in ls_idx: gs[num] += i-ls_idx[num]; gn[num] += 1
            ls_idx[num] = i
    for num in range(1, MAX_NUM+1):
        if gn[num] < 5: continue
        mg = gs[num]/gn[num]; cg = n-ls_idx.get(num,0)
        if cg > mg*1.3: scores[num] += min((cg/mg-1)*1.5, 3)

    # Signal 9: Pair frequency
    pf = Counter()
    for d in data[-200:]:
        for p in combinations(sorted(d[:PICK]), 2): pf[p] += 1
    for num in range(1, MAX_NUM+1):
        scores[num] += sum(pf.get(tuple(sorted([p,num])),0) for p in last)*0.05

    # Signal 10: Pair-boost for non-top12
    top12 = sorted(scores.items(), key=lambda x: -x[1])[:12]
    top12_nums = [num for num, _ in top12]
    for num in range(1, MAX_NUM+1):
        if num in top12_nums: continue
        pair_bonus = sum(pf.get(tuple(sorted([num, t])), 0) for t in top12_nums)
        scores[num] += pair_bonus * 0.03

    # Signal 11: Repeat-adjacency
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

    # Signal 12: Markov-2
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

    # Signal 13: Digit frequency
    if n >= 10:
        digit_freq = Counter()
        for d in data[-10:]:
            for num in d[:PICK]: digit_freq[num % 10] += 1
        total_digits = sum(digit_freq.values())
        for num in range(1, MAX_NUM+1):
            d = num % 10; ed = total_digits / 10; ad = digit_freq.get(d, 0)
            if ad > ed * 1.3: scores[num] += min((ad/ed - 1) * 1.0, 2)

    # Signal 14: Mean-reversion
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
    """Build pool — deterministic."""
    n = len(data); last_set = set(data[-1][:PICK])
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))  # secondary sort by num for stability
    pools = set(num for num, _ in ranked[:15])
    for w in [30, 50, 100]:
        fc = Counter(num for d in data[-min(w,n):] for num in d[:PICK])
        pools.update(num for num, _ in fc.most_common(15))
    ls = {}
    for i, d in enumerate(data):
        for num in d[:PICK]: ls[num] = i
    pools.update(sorted(range(1,MAX_NUM+1), key=lambda x: (-(n-ls.get(x,0)), x))[:15])
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
    gap_due = set(sorted(range(1, MAX_NUM+1), key=lambda x: (-(n - ls.get(x, 0)), x))[:15])
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
    return sorted(fusion, key=lambda x: (-scores.get(x, 0), x))[:target_size]


def generate_10_tickets(pool, scores):
    """Generate 10 BEST combos — 100% deterministic.
    Dùng exhaustive enumeration, KHÔNG có random.
    """
    top_n = min(18, len(pool))
    top_nums = pool[:top_n]

    all_combos = []
    for combo in combinations(top_nums, PICK):
        sc = sum(scores.get(num, 0) for num in combo)
        nums = sorted(combo)
        spread = nums[-1] - nums[0]
        sc += spread * 0.1
        even_count = sum(1 for x in combo if x % 2 == 0)
        if 2 <= even_count <= 4: sc += 0.5
        consec = sum(1 for i in range(len(nums)-1) if nums[i+1] - nums[i] == 1)
        if consec >= 3: sc -= 1.0
        all_combos.append((tuple(sorted(combo)), round(sc, 4)))

    # Sort deterministic: by score DESC, then by tuple ASC (for tiebreaker)
    all_combos.sort(key=lambda x: (-x[1], x[0]))

    # Pick top 10 with diversity (no 2 tickets share 5+ numbers)
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


def data_fingerprint(data):
    """Tạo fingerprint từ data để detect thay đổi."""
    last5 = data[-5:] if len(data) >= 5 else data
    raw = str(last5)
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def get_cache_path(lottery_type, draw_date, data_hash):
    """Get cache file path."""
    date_str = draw_date.strftime('%Y%m%d')
    return os.path.join(CACHE_DIR, f'{lottery_type}_{date_str}_{data_hash}.json')


def save_prediction(cache_path, prediction):
    """Lưu dự đoán vào cache."""
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(prediction, f, indent=2, ensure_ascii=False)


def load_prediction(cache_path):
    """Load dự đoán từ cache."""
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def predict_lottery(name, lottery_type, data_raw, MAX_NUM, draw_days):
    """Predict 10 tickets for next draw — deterministic + cached."""
    data = [d[:PICK] for d in data_raw]
    n = len(data)
    
    # Xác định ngày quay tiếp theo
    draw_date = get_next_draw_date(draw_days)
    data_hash = data_fingerprint(data)
    cache_path = get_cache_path(lottery_type, draw_date, data_hash)
    
    print(f"\n{'█'*70}")
    print(f"  🎯 {name} — DỰ ĐOÁN 10 VÉ")
    print(f"  📅 Kỳ quay: {date_to_str(draw_date)}")
    print(f"  📊 Dữ liệu: {n} kỳ | Hash: {data_hash}")
    print(f"{'█'*70}")
    
    # Check cache — nếu đã dự đoán rồi, trả về kết quả cũ
    cached = load_prediction(cache_path)
    if cached:
        print(f"\n  ✅ ĐÃ CÓ DỰ ĐOÁN — Load từ cache")
        print(f"  📁 {cache_path}")
        print(f"  ⏰ Tạo lúc: {cached['created_at']}")
        tickets = cached['tickets']
    else:
        print(f"\n  🔄 Tính toán dự đoán mới...")
        
        # Compute signals
        scores = compute_signals(data, MAX_NUM)
        
        # Build pool
        pool = build_pool(data, scores, 42, MAX_NUM)
        
        # Generate 10 tickets
        tickets_raw = generate_10_tickets(pool, scores)
        
        # Format for cache
        tickets = []
        for combo, sc in tickets_raw:
            tickets.append({
                'numbers': list(combo),
                'score': sc
            })
        
        # Top numbers info
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:20]
        top_nums = [{'num': num, 'score': round(sc, 2)} for num, sc in ranked]
        
        # Save to cache
        prediction = {
            'lottery': name,
            'draw_date': str(draw_date),
            'draw_date_display': date_to_str(draw_date),
            'data_draws': n,
            'data_hash': data_hash,
            'last_draw': sorted(data[-1]),
            'prev_draw': sorted(data[-2]) if n >= 2 else [],
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'engine': 'V22 NEXUS (14 signals + forensic)',
            'tickets': tickets,
            'top_numbers': top_nums,
            'pool_size': len(pool),
        }
        save_prediction(cache_path, prediction)
        print(f"  💾 Đã lưu: {cache_path}")
    
    # Display results
    print(f"\n  Kỳ trước: {sorted(data[-1])}")
    if n >= 2:
        print(f"  Kỳ trước nữa: {sorted(data[-2])}")
    
    print(f"\n  ┌─{'─'*56}─┐")
    print(f"  │ {'🏆 10 VÉ DỰ ĐOÁN — ' + date_to_str(draw_date):^56} │")
    print(f"  ├─{'─'*56}─┤")
    
    all_nums = set()
    for i, t in enumerate(tickets):
        nums = t['numbers']
        all_nums.update(nums)
        sc = t['score']
        num_str = " — ".join(f"{x:>2}" for x in nums)
        even = sum(1 for x in nums if x % 2 == 0)
        print(f"  │  Vé {i+1:>2}:  {num_str}   (score: {sc:.1f}){'':>4} │")
    
    print(f"  └─{'─'*56}─┘")
    
    # Stats
    total = math.comb(MAX_NUM, PICK)
    print(f"\n  📊 Phủ {len(all_nums)} số: {sorted(all_nums)}")
    print(f"  📊 Xác suất: 10/{total:,} ≈ 1/{total//10:,}")
    
    print(f"\n  ⚠️  Kết quả CỐ ĐỊNH — chạy lại sẽ cho CÙNG dãy số")
    print(f"  ⚠️  Chỉ thay đổi khi có DATA MỚI (kỳ quay mới)")
    
    return tickets


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════

if __name__ == '__main__':
    now = datetime.now()
    
    print("=" * 70)
    print("  🎯 PREDICT 10 — Dự Đoán Cố Định Theo Ngày")
    print(f"  ⏰ Thời gian: {now.strftime('%H:%M:%S %d/%m/%Y')}")
    print("  ✅ Deterministic: Chạy lại => CÙng kết quả")
    print("  ✅ Cache: Lưu file JSON, không tính lại")
    print("  Engine: V22 NEXUS (14 signals)")
    print("=" * 70)

    mega = get_mega645_numbers()
    power = get_power655_numbers()

    print(f"\n  📦 Mega 6/45: {len(mega)} kỳ | Power 6/55: {len(power)} kỳ")

    predict_lottery("MEGA 6/45", "mega645", mega, MAX_MEGA, MEGA_DAYS)
    predict_lottery("POWER 6/55", "power655", power, MAX_POWER, POWER_DAYS)

    print(f"\n{'█'*70}")
    print(f"  ✅ HOÀN THÀNH")
    print(f"  📁 Cache: {CACHE_DIR}")
    print(f"  🔄 Chạy lại: python predict10.py")
    print(f"  💡 Kết quả sẽ KHÔNG thay đổi cho cùng kỳ quay + cùng data")
    print(f"{'█'*70}")
