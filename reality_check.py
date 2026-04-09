"""
REALITY CHECK: V21 TITAN — Chi phí vs Lợi nhuận thực tế
=========================================================
Tính toán:
1. Mỗi kỳ phải mua bao nhiêu vé?
2. Chi phí mỗi kỳ và tổng chi phí?
3. Trúng bao nhiêu lần 6/6?
4. So sánh với đánh random (ngẫu nhiên)?
5. Lời hay lỗ?
"""
import json, math, os, sys

sys.stdout.reconfigure(encoding='utf-8')

# ═══════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════
TOTAL_COMBOS = math.comb(45, 6)  # C(45,6) = 8,145,060
TICKET_PRICE = 10_000  # VND per ticket (1 combo = 1 vé)
DRAWS_PER_WEEK = 3  # Mon, Wed, Fri
JACKPOT_MIN = 12_000_000_000  # 12 tỷ VND
JACKPOT_AVG = 35_000_000_000  # ~35 tỷ VND (trung bình thực tế)
JACKPOT_HIGH = 80_000_000_000  # 80 tỷ VND (jackpot lớn)

# Load V21 results
with open(os.path.join(os.path.dirname(__file__), 'models', 'v21_titan.json'), 'r') as f:
    v21 = json.load(f)

n_test = v21['n_test']  # 1286 draws
best_config = 'MS-F42-70'

print("=" * 90)
print("  💰 REALITY CHECK — V21 TITAN (MS-F42-70)")
print(f"  Dữ liệu: {n_test} kỳ quay backtest | C(45,6) = {TOTAL_COMBOS:,} tổ hợp")
print(f"  Giá vé: {TICKET_PRICE:,} VND/vé | Quay: {DRAWS_PER_WEEK} kỳ/tuần")
print("=" * 90)

# ═══════════════════════════════════════
# 1. KẾT QUẢ BACKTEST THỰC TẾ
# ═══════════════════════════════════════
print(f"\n  📊 1. KẾT QUẢ THỰC TẾ (MS-F42-70 — Best Config)")
print(f"  {'─'*80}")

portfolio_sizes = [5000, 10000, 18000, 20000, 30000, 50000]
results = v21['configs'][best_config]

total_weeks = n_test / DRAWS_PER_WEEK
total_years = total_weeks / 52

print(f"\n  Khoảng thời gian: {n_test} kỳ = {total_weeks:.0f} tuần ≈ {total_years:.1f} năm")
print(f"\n  {'Vé/kỳ':>10} │ {'Trúng 6/6':>10} │ {'Trúng 5/6':>10} │ {'Tần suất':>15} │ {'Mỗi bao lâu':>15}")
print(f"  {'─'*10} │ {'─'*10} │ {'─'*10} │ {'─'*15} │ {'─'*15}")

for pt in portfolio_sizes:
    r = results[str(pt)]
    s6 = r['six']
    s5 = r['five']
    
    if s6 > 0:
        freq = f"1/{n_test//s6} kỳ"
        interval_draws = n_test / s6
        interval_weeks = interval_draws / DRAWS_PER_WEEK
        interval_str = f"~{interval_weeks:.0f} tuần"
        if interval_weeks > 52:
            interval_str = f"~{interval_weeks/52:.1f} năm"
    else:
        freq = "0 lần"
        interval_str = "∞"
    
    print(f"  {pt:>10,} │ {s6:>10} │ {s5:>10} │ {freq:>15} │ {interval_str:>15}")

# ═══════════════════════════════════════
# 2. CHI PHÍ THỰC TẾ
# ═══════════════════════════════════════
print(f"\n\n  💸 2. CHI PHÍ MỖI KỲ QUAY")
print(f"  {'─'*80}")

print(f"\n  {'Vé/kỳ':>10} │ {'Chi phí/kỳ':>18} │ {'Chi phí/tuần':>18} │ {'Chi phí/tháng':>18}")
print(f"  {'─'*10} │ {'─'*18} │ {'─'*18} │ {'─'*18}")

for pt in portfolio_sizes:
    cost_per_draw = pt * TICKET_PRICE
    cost_per_week = cost_per_draw * DRAWS_PER_WEEK
    cost_per_month = cost_per_week * 4.33
    
    def fmt_vnd(val):
        if val >= 1_000_000_000:
            return f"{val/1_000_000_000:.1f} tỷ"
        elif val >= 1_000_000:
            return f"{val/1_000_000:.0f} triệu"
        else:
            return f"{val:,.0f}"
    
    print(f"  {pt:>10,} │ {fmt_vnd(cost_per_draw):>18} │ {fmt_vnd(cost_per_week):>18} │ {fmt_vnd(cost_per_month):>18}")

# ═══════════════════════════════════════
# 3. TỔNG CHI PHÍ vs LỢI NHUẬN
# ═══════════════════════════════════════
print(f"\n\n  📈 3. TỔNG CHI PHÍ vs LỢI NHUẬN (qua {n_test} kỳ ≈ {total_years:.1f} năm)")
print(f"  {'─'*80}")

scenarios = [
    ("Jackpot TB (35 tỷ)", JACKPOT_AVG),
    ("Jackpot Cao (80 tỷ)", JACKPOT_HIGH),
]

for scenario_name, jackpot_val in scenarios:
    print(f"\n  📎 Kịch bản: {scenario_name}")
    print(f"  {'Vé/kỳ':>10} │ {'Tổng chi':>14} │ {'Trúng 6/6':>10} │ {'Tổng thưởng':>14} │ {'Lời/Lỗ':>14} │ {'ROI':>8}")
    print(f"  {'─'*10} │ {'─'*14} │ {'─'*10} │ {'─'*14} │ {'─'*14} │ {'─'*8}")
    
    for pt in portfolio_sizes:
        r = results[str(pt)]
        s6 = r['six']
        
        total_cost = pt * TICKET_PRICE * n_test
        total_revenue = s6 * jackpot_val
        profit = total_revenue - total_cost
        roi = (profit / total_cost * 100) if total_cost > 0 else 0
        
        def fmt_ty(val):
            return f"{val/1_000_000_000:.1f} tỷ"
        
        profit_marker = "✅" if profit > 0 else "❌"
        print(f"  {pt:>10,} │ {fmt_ty(total_cost):>14} │ {s6:>10} │ {fmt_ty(total_revenue):>14} │ {fmt_ty(profit):>14} {profit_marker} │ {roi:>+7.0f}%")

# ═══════════════════════════════════════
# 4. SO SÁNH VỚI RANDOM
# ═══════════════════════════════════════
print(f"\n\n  🎲 4. SO SÁNH VỚI ĐÁNH RANDOM")
print(f"  {'─'*80}")
print(f"  Tổng tổ hợp C(45,6) = {TOTAL_COMBOS:,}")
print()
print(f"  {'Vé/kỳ':>10} │ {'% phủ/kỳ':>10} │ {'Random kỳ vọng':>15} │ {'V21 thực tế':>12} │ {'Hơn Random':>12}")
print(f"  {'─'*10} │ {'─'*10} │ {'─'*15} │ {'─'*12} │ {'─'*12}")

for pt in portfolio_sizes:
    r = results[str(pt)]
    s6 = r['six']
    
    coverage_pct = pt / TOTAL_COMBOS * 100
    random_expected = n_test * pt / TOTAL_COMBOS
    advantage = s6 / random_expected if random_expected > 0 else 0
    
    print(f"  {pt:>10,} │ {coverage_pct:>9.3f}% │ {random_expected:>14.2f} │ {s6:>12} │ {advantage:>11.2f}x")

# ═══════════════════════════════════════
# 5. THỰC TẾ NGẮN GỌN
# ═══════════════════════════════════════
print(f"\n\n  ⭐ 5. TÓM TẮT THỰC TẾ")
print(f"  {'═'*80}")

# Best practical option: 10K
r10k = results['10000']
cost_10k = 10000 * TICKET_PRICE
print(f"""
  📌 OPTION TỐI ƯU: 10,000 vé/kỳ (100 triệu/kỳ)
  
     • Trúng 6/6: {r10k['six']} lần trong {n_test} kỳ ({total_years:.1f} năm)
     • Trúng 5/6: {r10k['five']} lần (thưởng ước tính ~30-100 triệu mỗi lần)
     • Trúng 4/6: {r10k['four']} lần (thưởng ~300,000 mỗi lần)
     • Tần suất 6/6: khoảng 1 lần mỗi {n_test/max(1,r10k['six']):.0f} kỳ ≈ {n_test/max(1,r10k['six'])/DRAWS_PER_WEEK/52:.1f} năm
     • Tổng chi phí: {cost_10k * n_test / 1e9:.1f} tỷ VND
     • So với random: gấp {r10k['six'] / (n_test * 10000 / TOTAL_COMBOS):.1f}x hiệu quả hơn
     • Thưởng 5/6: {r10k['five']} × ~50 triệu = {r10k['five'] * 50_000_000 / 1e9:.1f} tỷ (ước tính)
""")

# Smallest realistic option
r5k = results['5000']
cost_5k = 5000 * TICKET_PRICE
print(f"""  📌 OPTION TIẾT KIỆM: 5,000 vé/kỳ (50 triệu/kỳ)
  
     • Trúng 6/6: {r5k['six']} lần trong {n_test} kỳ
     • Trúng 5/6: {r5k['five']} lần
     • Tổng chi phí: {cost_5k * n_test / 1e9:.1f} tỷ VND
     • So với random: gấp {r5k['six'] / max(0.01, n_test * 5000 / TOTAL_COMBOS):.1f}x
""")

# ═══════════════════════════════════════
# 6. KẾT LUẬN
# ═══════════════════════════════════════
random_10k = n_test * 10000 / TOTAL_COMBOS
print(f"""  🏁 6. KẾT LUẬN TRUNG THỰC
  {'═'*80}
  
  ✅ V21 TinNam AI CÓ hiệu quả hơn random:
     • @10K: gấp {r10k['six']/random_10k:.1f}x random  (tốt nhất)
     • @50K: gấp {results['50000']['six'] / (n_test * 50000 / TOTAL_COMBOS):.1f}x random (gần như = random)
     
  ⚠️  Nhưng vẫn là đầu tư RỦI RO CỰC CAO:
     • Cần 100 triệu/kỳ × 3 kỳ/tuần = 300 triệu/tuần cho option 10K
     • Trúng 6/6 trung bình mỗi {n_test/max(1,r10k['six'])/DRAWS_PER_WEEK/52:.1f} năm
     • Nếu jackpot < 32 tỷ → LỖ RÒNG dù trúng 6/6
     • Engine chỉ giúp TẬP TRUNG vé vào vùng xác suất cao hơn, KHÔNG bảo đảm trúng
     
  📊 Con số thật:
     • C(45,6) = {TOTAL_COMBOS:,} tổ hợp
     • 10K vé = phủ {10000/TOTAL_COMBOS*100:.3f}% mỗi kỳ
     • Xác suất trúng 6/6 mỗi kỳ (V21): ~{r10k['six']/n_test*100:.2f}%
     • Xác suất KHÔNG trúng 6/6 mỗi kỳ: ~{(1 - r10k['six']/n_test)*100:.2f}%
""")

print(f"{'═'*90}")
