"""
V11 — PHÂN TÍCH XỔ SỐ TRUYỀN THỐNG: MỖI CỘT RIÊNG BIỆT
==========================================================
Phương pháp 100% per-column (6 cột sorted):
  Col1=nhỏ nhất, Col6=lớn nhất

8 PHÂN TÍCH:
1. MÔ HÌNH KHỐI: chia 45 số thành 5 khối (1-9, 10-18, 19-27, 28-36, 37-45)
2. TĂNG GIẢM mỗi cột: cột N tăng/giảm/giữ so kỳ trước → predict next
3. LỚN NHỎ mỗi cột: giá trị > median cột → predict pattern
4. CHẴN LẺ mỗi cột: predict chẵn/lẻ pattern cho từng cột
5. KHOẢNG GIÁ TRỊ mỗi cột: range [min, max] xác suất cao nhất
6. SỐ ĐI THEO NHAU (companion): khi A ở cột i → B thường ở cột j
7. SỐ LOẠI TRỪ (exclusion): khi A → B KHÔNG BAO GIỜ xuất hiện
8. PER-COLUMN MARKOV: transition matrix riêng từng cột
"""
import sys, os, math, json, time
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper.data_manager import get_mega645_numbers, get_mega645_all


def run_v11():
    data = get_mega645_numbers()
    all_rows = get_mega645_all()
    N = len(data)
    MAX = 45
    PICK = 6
    t0 = time.time()

    # Convert to sorted columns
    sorted_data = [sorted(d[:PICK]) for d in data]

    print("=" * 80)
    print("  🎯 V11 — PHÂN TÍCH TRUYỀN THỐNG: MỖI CỘT RIÊNG BIỆT")
    print(f"  {N} draws | 6 cột × 8 phân tích = 48 signals")
    print("=" * 80)

    # ================================================================
    # 1. MÔ HÌNH KHỐI — Block Pattern
    # ================================================================
    print(f"\n{'━'*80}")
    print("  1. MÔ HÌNH KHỐI — 5 khối: (1-9)(10-18)(19-27)(28-36)(37-45)")
    print(f"{'━'*80}")

    def block(x):
        return min((x - 1) // 9, 4)

    block_names = ['K1(1-9)', 'K2(10-18)', 'K3(19-27)', 'K4(28-36)', 'K5(37-45)']

    # Count block patterns
    block_patterns = Counter()
    for d in sorted_data:
        bp = tuple(block(x) for x in d)
        block_patterns[bp] += 1

    print(f"\n  Top 15 block patterns (trong {N} kỳ):")
    for bp, count in block_patterns.most_common(15):
        pct = count / N * 100
        bp_str = '-'.join(block_names[b][0:2] for b in bp)
        print(f"    {bp_str}: {count:4d} ({pct:5.2f}%)")

    # Block transition: given current block pattern → predict next
    block_trans = defaultdict(Counter)
    for i in range(N - 1):
        curr_bp = tuple(block(x) for x in sorted_data[i])
        next_bp = tuple(block(x) for x in sorted_data[i + 1])
        block_trans[curr_bp][next_bp] += 1

    # ================================================================
    # 2. TĂNG GIẢM MỖI CỘT
    # ================================================================
    print(f"\n{'━'*80}")
    print("  2. TĂNG GIẢM MỖI CỘT — Per-column up/down/same")
    print(f"{'━'*80}")

    col_trends = {col: {'up': 0, 'down': 0, 'same': 0} for col in range(PICK)}
    trend_trans = {col: defaultdict(Counter) for col in range(PICK)}

    for i in range(1, N):
        for col in range(PICK):
            diff = sorted_data[i][col] - sorted_data[i-1][col]
            if diff > 0:
                trend = 'up'
            elif diff < 0:
                trend = 'down'
            else:
                trend = 'same'
            col_trends[col][trend] += 1

            if i >= 2:
                prev_diff = sorted_data[i-1][col] - sorted_data[i-2][col]
                prev_trend = 'up' if prev_diff > 0 else ('down' if prev_diff < 0 else 'same')
                trend_trans[col][prev_trend][trend] += 1

    print(f"\n  {'Col':>4} | {'Up':>8} | {'Down':>8} | {'Same':>8} | Best next after UP | Best next after DOWN")
    for col in range(PICK):
        n_total = N - 1
        up_pct = col_trends[col]['up'] / n_total * 100
        dn_pct = col_trends[col]['down'] / n_total * 100
        sm_pct = col_trends[col]['same'] / n_total * 100

        # After UP → what's most likely next?
        up_next = trend_trans[col].get('up', Counter())
        up_total = sum(up_next.values()) or 1
        dn_next = trend_trans[col].get('down', Counter())
        dn_total = sum(dn_next.values()) or 1

        best_after_up = max(up_next, key=up_next.get) if up_next else '?'
        best_after_dn = max(dn_next, key=dn_next.get) if dn_next else '?'
        aup_pct = up_next[best_after_up] / up_total * 100 if up_next else 0
        adn_pct = dn_next[best_after_dn] / dn_total * 100 if dn_next else 0

        print(f"  Col{col+1} | {up_pct:6.1f}% | {dn_pct:6.1f}% | {sm_pct:6.1f}% | "
              f"→{best_after_up}({aup_pct:.0f}%) | →{best_after_dn}({adn_pct:.0f}%)")

    # ================================================================
    # 3. LỚN NHỎ MỖI CỘT
    # ================================================================
    print(f"\n{'━'*80}")
    print("  3. LỚN NHỎ MỖI CỘT — Giá trị > Median cột = Lớn")
    print(f"{'━'*80}")

    col_medians = []
    for col in range(PICK):
        vals = [sorted_data[i][col] for i in range(N)]
        med = np.median(vals)
        col_medians.append(med)

    big_small_trans = {col: defaultdict(Counter) for col in range(PICK)}
    for i in range(1, N):
        for col in range(PICK):
            prev_bs = 'B' if sorted_data[i-1][col] > col_medians[col] else 'S'
            curr_bs = 'B' if sorted_data[i][col] > col_medians[col] else 'S'
            big_small_trans[col][prev_bs][curr_bs] += 1

    print(f"\n  {'Col':>4} | {'Median':>7} | Sau Lớn→? | Sau Nhỏ→?")
    for col in range(PICK):
        after_B = big_small_trans[col].get('B', Counter())
        after_S = big_small_trans[col].get('S', Counter())
        tB = sum(after_B.values()) or 1
        tS = sum(after_S.values()) or 1

        b_to_b = after_B.get('B', 0) / tB * 100
        b_to_s = after_B.get('S', 0) / tB * 100
        s_to_b = after_S.get('B', 0) / tS * 100
        s_to_s = after_S.get('S', 0) / tS * 100

        print(f"  Col{col+1} | {col_medians[col]:>5.0f}   | "
              f"B→B:{b_to_b:.0f}% B→S:{b_to_s:.0f}% | "
              f"S→B:{s_to_b:.0f}% S→S:{s_to_s:.0f}%")

    # ================================================================
    # 4. CHẴN LẺ MỖI CỘT
    # ================================================================
    print(f"\n{'━'*80}")
    print("  4. CHẴN LẺ MỖI CỘT — Per-column even/odd transition")
    print(f"{'━'*80}")

    eo_trans = {col: defaultdict(Counter) for col in range(PICK)}
    for i in range(1, N):
        for col in range(PICK):
            prev_eo = 'E' if sorted_data[i-1][col] % 2 == 0 else 'O'
            curr_eo = 'E' if sorted_data[i][col] % 2 == 0 else 'O'
            eo_trans[col][prev_eo][curr_eo] += 1

    # Also: full 6-col EO pattern
    eo_patterns = Counter()
    for d in sorted_data:
        pat = tuple('E' if x % 2 == 0 else 'O' for x in d)
        eo_patterns[pat] += 1

    print(f"\n  Per-column Even/Odd transitions:")
    print(f"  {'Col':>4} | Sau Chẵn→? | Sau Lẻ→?")
    for col in range(PICK):
        after_E = eo_trans[col].get('E', Counter())
        after_O = eo_trans[col].get('O', Counter())
        tE = sum(after_E.values()) or 1
        tO = sum(after_O.values()) or 1
        print(f"  Col{col+1} | E→E:{after_E.get('E',0)/tE*100:.0f}% E→O:{after_E.get('O',0)/tE*100:.0f}% | "
              f"O→E:{after_O.get('E',0)/tO*100:.0f}% O→O:{after_O.get('O',0)/tO*100:.0f}%")

    print(f"\n  Top 10 EO patterns:")
    for pat, count in eo_patterns.most_common(10):
        print(f"    {''.join(pat)}: {count:4d} ({count/N*100:.1f}%)")

    eo_pattern_trans = defaultdict(Counter)
    for i in range(N-1):
        curr = tuple('E' if x % 2 == 0 else 'O' for x in sorted_data[i])
        nxt = tuple('E' if x % 2 == 0 else 'O' for x in sorted_data[i+1])
        eo_pattern_trans[curr][nxt] += 1

    # ================================================================
    # 5. KHOẢNG GIÁ TRỊ MỖI CỘT — Value Range
    # ================================================================
    print(f"\n{'━'*80}")
    print("  5. KHOẢNG GIÁ TRỊ MỖI CỘT — Phạm vi xuất hiện nhiều nhất")
    print(f"{'━'*80}")

    col_ranges = {}
    for col in range(PICK):
        vals = [sorted_data[i][col] for i in range(N)]
        p5, p25, p50, p75, p95 = np.percentile(vals, [5, 25, 50, 75, 95])
        col_ranges[col] = {
            'p5': int(p5), 'p25': int(p25), 'p50': int(p50),
            'p75': int(p75), 'p95': int(p95),
            'mean': round(np.mean(vals), 1),
            'top_values': Counter(vals).most_common(5),
        }
        print(f"\n  Col{col+1}: range [{int(min(vals))}-{int(max(vals))}], "
              f"90% CI: [{int(p5)}-{int(p95)}], "
              f"50% CI: [{int(p25)}-{int(p75)}], median={int(p50)}")
        top_str = ', '.join(f"{v}({c})" for v, c in col_ranges[col]['top_values'])
        print(f"         Top values: {top_str}")

    # ================================================================
    # 6. SỐ ĐI THEO NHAU (Companion Numbers)
    # ================================================================
    print(f"\n{'━'*80}")
    print("  6. SỐ ĐI THEO NHAU — Companion per column")
    print(f"{'━'*80}")

    # For each column pair (colA → colB): when value X in colA → what's most common in colB?
    companion = {(ca, cb): defaultdict(Counter) for ca in range(PICK) for cb in range(PICK) if ca != cb}

    for d in sorted_data:
        for ca in range(PICK):
            for cb in range(PICK):
                if ca != cb:
                    companion[(ca, cb)][d[ca]][d[cb]] += 1

    # Cross-draw companions: when X in colA draw N → Y in colB draw N+1
    cross_companion = {(ca, cb): defaultdict(Counter) for ca in range(PICK) for cb in range(PICK)}
    for i in range(N - 1):
        for ca in range(PICK):
            for cb in range(PICK):
                cross_companion[(ca, cb)][sorted_data[i][ca]][sorted_data[i+1][cb]] += 1

    # Find strongest cross-draw companions
    print(f"\n  Strongest cross-draw companions (Col_A kỳ N → Col_B kỳ N+1):")
    strong_companions = []
    for (ca, cb), trans in cross_companion.items():
        for val_a, next_counts in trans.items():
            total = sum(next_counts.values())
            if total < 30:
                continue
            for val_b, count in next_counts.most_common(1):
                rate = count / total
                # Expected rate depends on column
                col_freq = Counter(sorted_data[i][cb] for i in range(N))
                exp_rate = col_freq.get(val_b, 0) / N
                if exp_rate > 0 and rate > exp_rate * 1.8 and rate > 0.05:
                    z = (rate - exp_rate) / max(math.sqrt(exp_rate * (1 - exp_rate) / total), 0.01)
                    if z > 2.5:
                        strong_companions.append({
                            'from_col': ca + 1, 'from_val': val_a,
                            'to_col': cb + 1, 'to_val': val_b,
                            'rate': round(rate, 3), 'expected': round(exp_rate, 3),
                            'z': round(z, 2), 'count': count,
                        })

    strong_companions.sort(key=lambda x: -x['z'])
    for sc in strong_companions[:15]:
        print(f"    Col{sc['from_col']}={sc['from_val']:2d} → Col{sc['to_col']}={sc['to_val']:2d}: "
              f"rate={sc['rate']:.1%} (exp={sc['expected']:.1%}), z={sc['z']:+.1f}, n={sc['count']}")

    # ================================================================
    # 7. SỐ LOẠI TRỪ (Exclusion Numbers)
    # ================================================================
    print(f"\n{'━'*80}")
    print("  7. SỐ LOẠI TRỪ — Numbers that NEVER appear together")
    print(f"{'━'*80}")

    # Within same draw: pairs that appear much less than expected
    co_occur = defaultdict(int)
    appear = Counter()
    for d in sorted_data:
        for x in d:
            appear[x] += 1
        for a, b in combinations(d, 2):
            co_occur[(min(a, b), max(a, b))] += 1

    exclusion_pairs = []
    for a in range(1, MAX + 1):
        for b in range(a + 1, MAX + 1):
            if appear[a] < 50 or appear[b] < 50:
                continue
            obs = co_occur.get((a, b), 0)
            exp = appear[a] * appear[b] * PICK * (PICK - 1) / (N * MAX * (MAX - 1))
            if exp > 3:
                z = (obs - exp) / math.sqrt(max(exp, 1))
                if z < -2.5:
                    exclusion_pairs.append({
                        'pair': (a, b), 'observed': obs,
                        'expected': round(exp, 1), 'z': round(z, 2),
                    })

    exclusion_pairs.sort(key=lambda x: x['z'])
    print(f"\n  Pairs that EXCLUDE each other (appear less than expected):")
    print(f"  Found: {len(exclusion_pairs)} pairs (z < -2.5)")
    for ep in exclusion_pairs[:15]:
        print(f"    ({ep['pair'][0]:2d},{ep['pair'][1]:2d}): obs={ep['observed']}, "
              f"exp={ep['expected']}, z={ep['z']}")

    # Cross-draw exclusion: if A in draw N → B NEVER in draw N+1
    cross_exclude = []
    for a in range(1, MAX + 1):
        for b in range(1, MAX + 1):
            a_count = sum(1 for d in sorted_data[:-1] if a in d)
            if a_count < 50:
                continue
            follow_count = sum(1 for i in range(N - 1)
                               if a in sorted_data[i] and b in sorted_data[i + 1])
            exp = a_count * PICK / MAX
            if exp > 5:
                z = (follow_count - exp) / math.sqrt(max(exp, 1))
                if z < -3:
                    cross_exclude.append({
                        'a': a, 'b': b, 'follow': follow_count,
                        'expected': round(exp, 1), 'z': round(z, 2),
                    })

    cross_exclude.sort(key=lambda x: x['z'])
    print(f"\n  Cross-draw exclusion (A kỳ N → B KHÔNG kỳ N+1):")
    print(f"  Found: {len(cross_exclude)} pairs (z < -3)")
    for ce in cross_exclude[:10]:
        print(f"    {ce['a']:2d} → NOT {ce['b']:2d}: follow={ce['follow']}, "
              f"exp={ce['expected']}, z={ce['z']}")

    # ================================================================
    # 8. PER-COLUMN MARKOV — Predict next value per column
    # ================================================================
    print(f"\n{'━'*80}")
    print("  8. PER-COLUMN MARKOV — Transition matrix per column")
    print(f"{'━'*80}")

    col_markov = {}
    for col in range(PICK):
        trans = defaultdict(Counter)
        for i in range(N - 1):
            prev = sorted_data[i][col]
            nxt = sorted_data[i + 1][col]
            trans[prev][nxt] += 1
        col_markov[col] = trans

        # Find strongest transitions for this column
        best = []
        for prev_val, next_counts in trans.items():
            total = sum(next_counts.values())
            if total < 10:
                continue
            for nxt_val, count in next_counts.most_common(3):
                rate = count / total
                best.append((prev_val, nxt_val, rate, count, total))

        best.sort(key=lambda x: -x[2])
        print(f"\n  Col{col+1} — Top 5 transitions:")
        for prev_v, nxt_v, rate, cnt, tot in best[:5]:
            print(f"    {prev_v:2d} → {nxt_v:2d}: {rate:.1%} ({cnt}/{tot})")

    # ================================================================
    # MEGA PREDICTOR: Combine ALL 8 analyses
    # ================================================================
    print(f"\n{'━'*80}")
    print("  🏆 MEGA PREDICTOR — Walk-Forward Backtest")
    print(f"{'━'*80}")

    min_train = 200
    test_start = min_train
    n_test = N - test_start - 1
    port_sizes = [1, 10, 50, 100, 500]
    results = {ps: [] for ps in port_sizes}
    random_results = []
    np.random.seed(42)

    for ti in range(test_start, N - 1):
        train = sorted_data[:ti + 1]
        actual = set(sorted_data[ti + 1])
        last = train[-1]

        # Score per number based on ALL 8 analyses
        scores = {num: 0.0 for num in range(1, MAX + 1)}

        # Signal 1: Block model — predict next block pattern
        curr_bp = tuple(block(x) for x in last)
        local_bt = defaultdict(Counter)
        for i in range(len(train) - 1):
            bp = tuple(block(x) for x in train[i])
            next_bp = tuple(block(x) for x in train[i + 1])
            local_bt[bp][next_bp] += 1
        if curr_bp in local_bt:
            total = sum(local_bt[curr_bp].values())
            for next_bp, count in local_bt[curr_bp].most_common(3):
                weight = count / total
                for col_idx, b in enumerate(next_bp):
                    lo = b * 9 + 1
                    hi = min(lo + 8, MAX)
                    for num in range(lo, hi + 1):
                        scores[num] += weight * 2

        # Signal 2: Up/Down per column
        for col in range(PICK):
            if ti >= 1:
                diff = train[-1][col] - train[-2][col]
                curr_trend = 'up' if diff > 0 else ('down' if diff < 0 else 'same')

                local_tt = defaultdict(Counter)
                for i in range(2, len(train)):
                    d = train[i-1][col] - train[i-2][col]
                    t = 'up' if d > 0 else ('down' if d < 0 else 'same')
                    d2 = train[i][col] - train[i-1][col]
                    t2 = 'up' if d2 > 0 else ('down' if d2 < 0 else 'same')
                    local_tt[t][t2] += 1

                next_trends = local_tt.get(curr_trend, Counter())
                total_tt = sum(next_trends.values()) or 1
                best_next = max(next_trends, key=next_trends.get) if next_trends else 'up'

                curr_val = last[col]
                if best_next == 'up':
                    for num in range(curr_val + 1, min(curr_val + 10, MAX + 1)):
                        scores[num] += 1.5
                elif best_next == 'down':
                    for num in range(max(1, curr_val - 10), curr_val):
                        scores[num] += 1.5

        # Signal 3: Big/Small per column
        for col in range(PICK):
            vals = [train[i][col] for i in range(len(train))]
            med = np.median(vals)
            prev_bs = 'B' if last[col] > med else 'S'

            local_bs = defaultdict(Counter)
            for i in range(1, len(train)):
                p_bs = 'B' if train[i-1][col] > med else 'S'
                c_bs = 'B' if train[i][col] > med else 'S'
                local_bs[p_bs][c_bs] += 1

            after = local_bs.get(prev_bs, Counter())
            total_bs = sum(after.values()) or 1
            p_big = after.get('B', 0) / total_bs
            if p_big > 0.55:
                for num in range(int(med) + 1, MAX + 1):
                    scores[num] += 0.5
            elif p_big < 0.45:
                for num in range(1, int(med) + 1):
                    scores[num] += 0.5

        # Signal 4: Even/Odd per column
        for col in range(PICK):
            prev_eo = 'E' if last[col] % 2 == 0 else 'O'
            local_eo = defaultdict(Counter)
            for i in range(1, len(train)):
                p_eo = 'E' if train[i-1][col] % 2 == 0 else 'O'
                c_eo = 'E' if train[i][col] % 2 == 0 else 'O'
                local_eo[p_eo][c_eo] += 1
            after = local_eo.get(prev_eo, Counter())
            total_eo = sum(after.values()) or 1
            p_even = after.get('E', 0) / total_eo
            if p_even > 0.55:
                for num in range(2, MAX + 1, 2):
                    scores[num] += 0.5
            elif p_even < 0.45:
                for num in range(1, MAX + 1, 2):
                    scores[num] += 0.5

        # Signal 5: Column value range
        for col in range(PICK):
            vals = [train[i][col] for i in range(len(train))]
            p10, p90 = np.percentile(vals, [10, 90])
            for num in range(int(p10), int(p90) + 1):
                scores[num] += 1

        # Signal 6: Cross-draw companions
        for ca in range(PICK):
            for cb in range(PICK):
                val_a = last[ca]
                local_comp = defaultdict(Counter)
                for i in range(len(train) - 1):
                    if train[i][ca] == val_a:
                        local_comp[val_a][train[i + 1][cb]] += 1
                if val_a in local_comp:
                    total_c = sum(local_comp[val_a].values()) or 1
                    for val_b, cnt in local_comp[val_a].most_common(3):
                        rate = cnt / total_c
                        scores[val_b] += rate * 3

        # Signal 7: Exclusion — penalize excluded pairs
        for a in set(last):
            for b in range(1, MAX + 1):
                follow_cnt = sum(1 for i in range(len(train) - 1)
                                  if a in train[i] and b in train[i + 1])
                a_cnt = sum(1 for i in range(len(train) - 1) if a in train[i])
                if a_cnt > 30:
                    rate = follow_cnt / a_cnt
                    if rate < PICK / MAX * 0.5:
                        scores[b] -= 1

        # Signal 8: Per-column Markov
        for col in range(PICK):
            val = last[col]
            local_mk = defaultdict(Counter)
            for i in range(len(train) - 1):
                local_mk[train[i][col]][train[i + 1][col]] += 1
            if val in local_mk:
                total_mk = sum(local_mk[val].values()) or 1
                for nxt_val, cnt in local_mk[val].most_common(5):
                    scores[nxt_val] += (cnt / total_mk) * 4

        # Generate portfolio
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        pool = [num for num, _ in ranked[:22]]

        portfolio = []
        seen = set()
        for combo in combinations(pool[:16], PICK):
            s = sum(combo)
            if s < 60 or s > 210:
                continue
            r = combo[-1] - combo[0]
            if r < 15 or r > 43:
                continue
            odds = sum(1 for x in combo if x % 2 == 1)
            if odds < 1 or odds > 5:
                continue
            sc = sum(scores.get(n, 0) for n in combo)
            key = tuple(combo)
            if key not in seen:
                seen.add(key)
                portfolio.append((sorted(combo), sc))
            if len(portfolio) >= max(port_sizes) * 2:
                break

        portfolio.sort(key=lambda x: -x[1])

        for ps in port_sizes:
            port = [p[0] for p in portfolio[:ps]]
            if port:
                best = max(len(set(p) & actual) for p in port)
            else:
                best = 0
            results[ps].append(best)

        rand = set(np.random.choice(range(1, MAX + 1), PICK, replace=False).tolist())
        random_results.append(len(rand & actual))

        if (ti - test_start + 1) % 200 == 0:
            print(f"    [{ti - test_start + 1}/{n_test}] "
                  f"port500 avg={np.mean(results[500]):.3f}")

    # ================================================================
    # RESULTS
    # ================================================================
    elapsed = time.time() - t0
    rand_avg = np.mean(random_results)

    print(f"\n{'='*80}")
    print(f"  📊 V11 RESULTS — {n_test} tests")
    print(f"{'='*80}")

    print(f"\n  {'Port':>6} | {'Avg':>6} | {'≥3':>6} | {'≥4':>6} | {'≥5':>6} | {'6/6':>6} | {'vs Rand':>8}")
    for ps in port_sizes:
        bm = results[ps]
        avg = np.mean(bm)
        imp = (avg / max(rand_avg, 0.01) - 1) * 100
        print(f"  {ps:6d} | {avg:.3f} | "
              f"{sum(1 for m in bm if m>=3)/n_test*100:5.1f}% | "
              f"{sum(1 for m in bm if m>=4)/n_test*100:5.1f}% | "
              f"{sum(1 for m in bm if m>=5)/n_test*100:5.1f}% | "
              f"{sum(1 for m in bm if m>=6)/n_test*100:5.2f}% | "
              f"{imp:+6.1f}%")

    # Distribution
    bm_max = results[max(port_sizes)]
    dist = Counter(bm_max)
    print(f"\n  Distribution (Port {max(port_sizes)}):")
    for k in range(7):
        c = dist.get(k, 0)
        bar = '█' * int(c / n_test * 100)
        print(f"    {k}/6: {c:4d} ({c/n_test*100:5.2f}%) {bar}")

    # Save
    output = {
        'version': '11.0 — Traditional Column Analysis',
        'n_draws': N, 'n_test': n_test,
        'col_ranges': {f'col{c+1}': col_ranges[c] for c in range(PICK)},
        'n_companion': len(strong_companions),
        'n_exclusion': len(exclusion_pairs),
        'n_cross_exclude': len(cross_exclude),
        'results': {},
        'elapsed': round(elapsed, 1),
    }
    for ps in port_sizes:
        bm = results[ps]
        output['results'][f'port_{ps}'] = {
            'avg': round(np.mean(bm), 4),
            'pct_3': round(sum(1 for m in bm if m>=3)/n_test*100, 2),
            'pct_4': round(sum(1 for m in bm if m>=4)/n_test*100, 2),
            'pct_5': round(sum(1 for m in bm if m>=5)/n_test*100, 2),
            'pct_6': round(sum(1 for m in bm if m>=6)/n_test*100, 2),
        }
    path = os.path.join(os.path.dirname(__file__), 'models', 'v11_results.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to {path}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'='*80}")


if __name__ == '__main__':
    run_v11()
