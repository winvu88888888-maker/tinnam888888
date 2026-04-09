# -*- coding: utf-8 -*-
"""
FORENSIC RULES MODULE — Toàn bộ 58+ quy luật & lỗ hổng
========================================================
Module hiển thị trên Streamlit UI với nút toggle.
"""

def get_all_rules():
    """Return all forensic rules as structured data for rendering."""
    
    critical_rules = [
        {
            "id": 49, "tier": "★★★★★", "version": "V9",
            "title": "FOURIER — CHU KỲ 36 KỲ",
            "icon": "📡",
            "description": "Tổng Mega có chu kỳ ~36 kỳ (12 tuần ≈ 3 tháng). C3 mạnh nhất (power ratio 6.20x). Đây KHÔNG nên xảy ra nếu RNG random.",
            "data": [
                ("SUM", "36 kỳ", "4.66x", "📡 SIGNIFICANT"),
                ("C3", "36 kỳ", "6.20x", "📡 STRONGEST"),
                ("C4", "36 kỳ", "5.13x", "📡"),
                ("C1", "14 kỳ", "4.63x", "📡"),
                ("C2", "28 kỳ", "4.26x", "📡"),
                ("C5", "37 kỳ", "3.54x", "📡"),
                ("C6", "28 kỳ", "3.13x", "📡"),
            ],
            "headers": ["Series", "Period", "Power Ratio", "Status"],
            "exploit": "Theo dõi chu kỳ 36 kỳ để dự đoán xu hướng tổng tăng/giảm."
        },
        {
            "id": 37, "tier": "★★★★★", "version": "V8",
            "title": "HURST — 5/6 CỘT CÓ MEMORY",
            "icon": "🧠",
            "description": "Hurst exponent H>0.55 cho 5/6 cột Mega → Có memory dài hạn, KHÔNG phải random walk.",
            "data": [
                ("C1", "0.653", "PERSISTENT", "📡"),
                ("C2", "0.630", "PERSISTENT", "📡"),
                ("C3", "0.620", "PERSISTENT", "📡"),
                ("C4", "0.544", "Random", ""),
                ("C5", "0.571", "PERSISTENT", "📡"),
                ("C6", "0.567", "PERSISTENT", "📡"),
            ],
            "headers": ["Cột", "H", "Kết luận", ""],
            "exploit": "Trending strategy hoạt động — xu hướng có tính bền."
        },
        {
            "id": 38, "tier": "★★★★★", "version": "V8",
            "title": "VARIANCE RATIO — BÁC BỎ RANDOM WALK",
            "icon": "📊",
            "description": "TẤT CẢ 6 cột có VR << 1 → Bác bỏ random walk. Hệ thống mean-reverting ngắn hạn + trending dài hạn.",
            "data": [
                ("C1", "0.487", "0.245", "0.125", "🚨"),
                ("C2", "0.518", "0.256", "0.134", "🚨"),
                ("C3", "0.479", "0.257", "0.128", "🚨"),
                ("C4", "0.508", "0.267", "0.140", "🚨"),
                ("C5", "0.500", "0.257", "0.136", "🚨"),
                ("C6", "0.537", "0.265", "0.138", "🚨"),
            ],
            "headers": ["Cột", "VR(2)", "VR(4)", "VR(8)", ""],
            "exploit": "Mean reversion là chiến lược DỰ ĐOÁN TỐT NHẤT."
        },
        {
            "id": 30, "tier": "★★★★★", "version": "V7",
            "title": "POWER 6/55 — TRÙNG 6/6 HOÀN TOÀN",
            "icon": "🚨",
            "description": "Power 6/55 có 2 kỳ (647 vs 993) ra TRÙNG HOÀN TOÀN 6/6 cùng bộ [8,19,27,34,46,51]. Xác suất tự nhiên ~0.003%.",
            "data": [("647", "993", "[8,19,27,34,46,51]", "6/6 TRÙNG")],
            "headers": ["Kỳ A", "Kỳ B", "Bộ số", "Match"],
            "exploit": "Lỗ hổng RNG nghiêm trọng hoặc lỗi dữ liệu."
        },
        {
            "id": 34, "tier": "★★★★★", "version": "V7",
            "title": "PRODUCT MOD — BIAS CỰC MẠNH",
            "icon": "🔢",
            "description": "Tích 6 số KHÔNG PHÂN BỐ ĐỀU theo bất kỳ modulo nào → RNG có cấu trúc ẩn.",
            "data": [
                ("mod 6", "5,582.5", "🚨 BIAS CỰC MẠNH"),
                ("mod 10", "7,174.0", "🚨 BIAS CỰC MẠNH"),
                ("mod 12", "11,511.3", "🚨 BIAS CỰC MẠNH"),
            ],
            "headers": ["Test", "Chi²", "Kết quả"],
            "exploit": "Dấu vân tay thuật toán RNG."
        },
    ]
    
    high_rules = [
        {
            "id": 25, "tier": "★★★★", "version": "V7",
            "title": "DELTA >2σ — ĐẢO CHIỀU CỰC MẠNH",
            "icon": "↩️",
            "description": "Khi cột nhảy >2σ, kỳ sau LUÔN đảo chiều. C1 sau nhảy lên: -15.8. 70% đảo chiều!",
            "data": [
                ("C1", "-15.8 (n=44)", "+3.9 (n=46)", "🚨"),
                ("C3", "-12.9 (n=27)", "+11.7 (n=40)", "🚨"),
                ("C5", "-6.8 (n=37)", "+15.6 (n=34)", "🚨"),
            ],
            "headers": ["Cột", "Sau nhảy LÊN", "Sau nhảy XUỐNG", ""],
            "exploit": "Khi cột nhảy bất thường → bet ngược chiều."
        },
        {
            "id": 39, "tier": "★★★★", "version": "V8",
            "title": "COMPOUND EXCLUSION — C1+C6 → THU HẸP",
            "icon": "🎯",
            "description": "Biết C1=1, C6=45 → C5 chỉ ∈ [30-44] → Giảm 60% không gian tìm kiếm!",
            "data": [
                ("1", "44", "[3-21]", "[6-36]", "[13-43]"),
                ("1", "45", "[4-33]", "[7-38]", "[30-44]"),
                ("5", "45", "[6-24]", "[8-33]", "[24-44]"),
            ],
            "headers": ["C1", "C6", "C2 range", "C3 range", "C5 range"],
            "exploit": "Fix C1+C6 → loại 40-60% combinations cho C2-C5."
        },
        {
            "id": 4, "tier": "★★★★", "version": "V1",
            "title": "EXCLUSION RULES — LOẠI TRỪ 20-40%",
            "icon": "🚫",
            "description": "Biết C1 → loại 8 giá trị C2. Biết any column → loại 20-40% không gian tìm kiếm.",
            "data": [],
            "headers": [],
            "exploit": "Exclusion rules giảm không gian từ 8M xuống ~5M combinations."
        },
        {
            "id": 43, "tier": "★★★★", "version": "V8",
            "title": "SUM DIFF >50 → MEAN REVERT -36",
            "icon": "📉",
            "description": "Sau tổng TĂNG >50: avg next diff = -36.3. Sau GIẢM <-50: +34.8.",
            "data": [
                ("Tổng TĂNG >50", "-36.3", "181"),
                ("Tổng GIẢM <-50", "+34.8", "168"),
                ("Overall", "-0.02", "—"),
            ],
            "headers": ["Điều kiện", "Avg next diff", "n"],
            "exploit": "Tổng tăng mạnh → kỳ sau chọn số nhỏ. Giảm mạnh → chọn số lớn."
        },
        {
            "id": 26, "tier": "★★★★", "version": "V7",
            "title": "BENFORD'S LAW — VI PHẠM",
            "icon": "📐",
            "description": "Chi² = 18.11 >> 7.8 threshold. Số bắt đầu bằng 3 xuất hiện GẤP ĐÔI kỳ vọng Benford.",
            "data": [
                ("1", "24.3%", "30.1%", "-5.8%"),
                ("2", "24.9%", "17.6%", "+7.3%"),
                ("3", "23.9%", "12.5%", "+11.4%"),
                ("4", "15.6%", "9.7%", "+5.9%"),
            ],
            "headers": ["Digit", "Actual", "Benford", "Lệch"],
            "exploit": "RNG range-limited bias → Ưu tiên số 30-39."
        },
        {
            "id": 54, "tier": "★★★★", "version": "V9",
            "title": "SỐ 24 NON-RANDOM — RUNS TEST",
            "icon": "🎰",
            "description": "Runs test z=+2.10 → Chuỗi xuất hiện/vắng mặt của số 24 KHÔNG RANDOM.",
            "data": [("24", "395", "375", "+2.10", "🚨 NON-RANDOM")],
            "headers": ["Số", "Runs", "Expected", "z-score", ""],
            "exploit": "Số 24 có pattern xuất hiện → theo dõi chu kỳ."
        },
    ]
    
    medium_rules = [
        {
            "id": 24, "tier": "★★★", "version": "V7",
            "title": "CONDITIONAL C1+C2 → C3",
            "icon": "🔮",
            "description": "C1≤6, C2∈[10-14] → C3=15 (32.4% vs 4% random = 8x!)",
            "exploit": "C3 ≈ C2 + 3~5 khi C1 nhỏ."
        },
        {
            "id": 27, "tier": "★★★", "version": "V7",
            "title": "AFFINITY NETWORK — SỐ HÚT/ĐẨY",
            "icon": "🔗",
            "description": "7↔44 (1.7x), 24↔37 (1.7x), 10↔22 (1.6x). 1 và 2 đẩy nhau (0.5x).",
            "exploit": "Chọn 7 → kèm 44. Chọn 24 → kèm 37."
        },
        {
            "id": 28, "tier": "★★★", "version": "V7",
            "title": "NEVER-TOGETHER — 25 BỘ BA KHÔNG BAO GIỜ CÙNG",
            "icon": "🚷",
            "description": "19+37+28 và 19+7+28 CHƯA BAO GIỜ cùng kỳ dù rất phổ biến.",
            "exploit": "TRÁNH chọn 3 số từ nhóm cấm."
        },
        {
            "id": 29, "tier": "★★★", "version": "V7",
            "title": "VOLATILITY CLUSTERING (GARCH)",
            "icon": "📊",
            "description": "AC(1)=+0.21 → Kỳ biến động hay đi kèm nhau. Sau calm: sum giảm xuống 133.1.",
            "exploit": "Kỳ biến động → kỳ sau cũng biến động → chọn range rộng."
        },
        {
            "id": 50, "tier": "★★★", "version": "V9",
            "title": "MUTUAL INFORMATION C3↔C4 = 0.783",
            "icon": "🔗",
            "description": "Cột liền nhau phụ thuộc phi tuyến rất mạnh. Biết C3 → dự đoán C4 tốt hơn.",
            "exploit": "Sử dụng MI để chain predict: C1→C2→C3→C4→C5→C6."
        },
        {
            "id": 52, "tier": "★★★", "version": "V9",
            "title": "CROSS-LAG: C6(T) → C5(T+1)",
            "icon": "⏩",
            "description": "C5(T)→C6(T+1): r=+0.085. C6 kỳ trước ảnh hưởng C4,C5,C6 kỳ sau.",
            "exploit": "C6 kỳ trước cao → C5,C6 kỳ sau cũng cao."
        },
        {
            "id": 53, "tier": "★★★", "version": "V9",
            "title": "MODULAR CLOCK",
            "icon": "⏰",
            "description": "Số 24 hay ra kỳ chẵn (+7.1%). Số 3 hay ra mỗi 6 kỳ. Số 19 ≡5(mod6).",
            "exploit": "Xác định kỳ hiện tại mod 6 → chọn số phù hợp."
        },
        {
            "id": 40, "tier": "★★★", "version": "V8",
            "title": "PARITY — 3 LẺ LUÔN PHỔ BIẾN NHẤT",
            "icon": "⚖️",
            "description": "Bất kể kỳ trước, 3 lẻ luôn phổ biến nhất (32-43%). Sau 5 lẻ → giảm mạnh về 2-3.",
            "exploit": "LUÔN chọn 3 lẻ + 3 chẵn. Sau 5 lẻ → chọn nhiều chẵn hơn."
        },
        {
            "id": 44, "tier": "★★★", "version": "V8",
            "title": "YEAR-OVER-YEAR: SỐ 8, 24 ỔN ĐỊNH NHẤT",
            "icon": "📅",
            "description": "Số 8 (CV=0.096) và 24 (CV=0.099) ổn định nhất qua mọi năm. Số 17, 42 biến động nhất.",
            "exploit": "Ưu tiên 8, 24 — đáng tin cậy. Tránh 17, 42 — rủi ro."
        },
        {
            "id": 1, "tier": "★★★", "version": "V1",
            "title": "C1/C6 STABILITY",
            "icon": "🔒",
            "description": "C1 ∈ {1-10} 90%. C6 ∈ {37-45} 90%. Fix 2 cột → đúng ~34%.",
            "exploit": "C1 luôn chọn 1-10, C6 luôn chọn 37-45."
        },
        {
            "id": 5, "tier": "★★★", "version": "V1",
            "title": "LIÊN TIẾP >50%",
            "icon": "🔢",
            "description": "51.4% kỳ có ít nhất 1 cặp liên tiếp. Mode spacing = 1 cho C1→C2, C2→C3, C3→C4.",
            "exploit": "LUÔN chọn ít nhất 1 cặp liên tiếp."
        },
        {
            "id": 41, "tier": "★★★", "version": "V8",
            "title": "COMEBACK — ĐẾN RỒI ĐI",
            "icon": "👋",
            "description": "Khi số vắng >15 kỳ quay lại: stay TB = 0.18-0.35. Tất cả đều 'đến rồi đi'.",
            "exploit": "Số quá hạn quay lại → chọn 1-2 kỳ rồi bỏ."
        },
    ]
    
    info_rules = [
        {
            "id": 55, "tier": "★★", "version": "V9",
            "title": "GOLDBACH-LIKE — 52.8% KỲ",
            "icon": "➕",
            "description": "52.8% kỳ có ≥1 số = tổng 2 số khác. Số 40-45 hay là 'tổng'.",
        },
        {
            "id": 56, "tier": "★★", "version": "V9",
            "title": "DRAW TYPE MID_MED_3L = 10.5%",
            "icon": "📋",
            "description": "Type phổ biến: Tổng trung bình + Range vừa + 3 lẻ → 10.5%.",
        },
        {
            "id": 31, "tier": "★★", "version": "V7",
            "title": "CROSS-DRAW PAIRS",
            "icon": "↔️",
            "description": "5→24 (1.44x), 7→8/44 (1.40x). Kỳ trước có 5 → kỳ sau chọn 24.",
        },
        {
            "id": 33, "tier": "★★", "version": "V8",
            "title": "NUMBER FAMILIES",
            "icon": "👪",
            "description": "STABLE: 2,8,9,24,36,45. FALLING: 6,40. VOLATILE: phần lớn.",
        },
        {
            "id": 32, "tier": "★★", "version": "V8",
            "title": "MOMENTUM INDICATOR",
            "icon": "🏎️",
            "description": "Sau momentum thấp (<-40): avg next sum = 135.5 (thấp hơn 2.1 vs avg).",
        },
        {
            "id": 35, "tier": "★★", "version": "V7",
            "title": "DOUBLE APPEARANCE",
            "icon": "♻️",
            "description": "Số 32 repeat 2 kỳ liên tiếp 34% nhiều hơn kỳ vọng.",
        },
        {
            "id": 42, "tier": "★★", "version": "V8",
            "title": "MULTI-LAG AC",
            "icon": "📡",
            "description": "C1: lag 9,16. C4: lag 8 đảo chiều. C5: lag 12 đảo chiều.",
        },
        {
            "id": 45, "tier": "★★", "version": "V8",
            "title": "TUẦN TRONG NĂM",
            "icon": "📆",
            "description": "Tuần 30: sum cao nhất (151.6). Tuần 14: thấp nhất (128.5).",
        },
    ]
    
    return {
        "critical": critical_rules,
        "high": high_rules,
        "medium": medium_rules,
        "info": info_rules,
        "stats": {
            "total_rules": 58,
            "critical": len(critical_rules),
            "high": len(high_rules),
            "medium": len(medium_rules),
            "info": len(info_rules),
            "versions": "V1→V9",
            "aspects": "~70 khía cạnh",
        }
    }


def render_forensic_rules(st):
    """Render the complete forensic rules panel in Streamlit."""
    
    rules = get_all_rules()
    stats = rules["stats"]
    
    # Header
    st.markdown(f"""
    <div class="glass-card" style="border-color:#8b5cf6; text-align:center; padding:32px;">
        <div style="font-size:2rem;margin-bottom:8px;">🔬</div>
        <div style="font-size:1.6rem;font-weight:900;
            background:linear-gradient(135deg,#6366f1,#8b5cf6,#ec4899);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            FORENSIC MASTER RULES
        </div>
        <div style="color:#94a3b8;font-size:0.9rem;margin:8px 0;">
            Toàn bộ quy luật & lỗ hổng RNG — Phân tích {stats['aspects']}
        </div>
        <div style="display:flex;justify-content:center;gap:16px;flex-wrap:wrap;margin-top:16px;">
            <div style="padding:8px 20px;background:rgba(239,68,68,0.15);border:1px solid rgba(239,68,68,0.3);border-radius:50px;">
                <span style="color:#ef4444;font-weight:900;font-family:JetBrains Mono;">{stats['critical']}</span>
                <span style="color:#94a3b8;font-size:0.8rem;"> CRITICAL</span>
            </div>
            <div style="padding:8px 20px;background:rgba(245,158,11,0.15);border:1px solid rgba(245,158,11,0.3);border-radius:50px;">
                <span style="color:#f59e0b;font-weight:900;font-family:JetBrains Mono;">{stats['high']}</span>
                <span style="color:#94a3b8;font-size:0.8rem;"> HIGH</span>
            </div>
            <div style="padding:8px 20px;background:rgba(34,197,94,0.15);border:1px solid rgba(34,197,94,0.3);border-radius:50px;">
                <span style="color:#22c55e;font-weight:900;font-family:JetBrains Mono;">{stats['medium']}</span>
                <span style="color:#94a3b8;font-size:0.8rem;"> MEDIUM</span>
            </div>
            <div style="padding:8px 20px;background:rgba(99,102,241,0.15);border:1px solid rgba(99,102,241,0.3);border-radius:50px;">
                <span style="color:#6366f1;font-weight:900;font-family:JetBrains Mono;">{stats['info']}</span>
                <span style="color:#94a3b8;font-size:0.8rem;"> INFO</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # CRITICAL TIER
    _render_tier(st, "🔴 CRITICAL — Lỗ hổng RNG nghiêm trọng", rules["critical"], "#ef4444")
    
    # HIGH TIER
    _render_tier(st, "🟠 HIGH — Quy luật khai thác mạnh", rules["high"], "#f59e0b")
    
    # MEDIUM TIER 
    _render_tier(st, "🟢 MEDIUM — Quy luật hỗ trợ", rules["medium"], "#22c55e")
    
    # INFO TIER
    _render_tier(st, "🔵 INFO — Thông tin bổ sung", rules["info"], "#6366f1")


def _render_tier(st, title, rules_list, color):
    """Render a tier of rules."""
    
    st.markdown(f"""
    <div style="padding:12px 20px;background:linear-gradient(135deg,{color}22,transparent);
        border-left:4px solid {color};border-radius:0 12px 12px 0;margin:24px 0 16px;
        font-weight:800;font-size:1.1rem;color:{color};">
        {title}
    </div>
    """, unsafe_allow_html=True)
    
    for rule in rules_list:
        data_html = ""
        if rule.get("data") and rule.get("headers"):
            headers = rule["headers"]
            th_html = "".join(f'<th style="padding:8px 12px;color:#06b6d4;font-weight:700;font-size:0.75rem;border-bottom:1px solid rgba(255,255,255,0.1);text-align:center;">{h}</th>' for h in headers)
            rows_html = ""
            for row in rule["data"]:
                cells = "".join(f'<td style="padding:6px 12px;color:#e2e8f0;font-family:JetBrains Mono,monospace;font-size:0.75rem;border-bottom:1px solid rgba(255,255,255,0.05);text-align:center;">{c}</td>' for c in row)
                rows_html += f"<tr>{cells}</tr>"
            data_html = f"""
            <div style="overflow-x:auto;margin:12px 0;border-radius:8px;background:rgba(0,0,0,0.2);">
                <table style="width:100%;border-collapse:collapse;">
                    <thead><tr>{th_html}</tr></thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </div>"""
        
        exploit_html = ""
        if rule.get("exploit"):
            exploit_html = f"""
            <div style="margin-top:10px;padding:8px 14px;background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.2);border-radius:8px;">
                <span style="color:#f59e0b;font-weight:700;font-size:0.75rem;">💡 KHAI THÁC: </span>
                <span style="color:#fcd34d;font-size:0.8rem;">{rule['exploit']}</span>
            </div>"""
        
        st.markdown(f"""
        <div style="background:rgba(17,24,39,0.6);border:1px solid rgba(255,255,255,0.08);border-radius:14px;
            padding:20px;margin-bottom:12px;border-left:3px solid {color};
            transition:all 0.3s;position:relative;overflow:hidden;">
            <div style="position:absolute;top:10px;right:14px;font-size:0.65rem;color:#64748b;font-family:JetBrains Mono;">
                #{rule['id']} | {rule.get('tier','')} | {rule.get('version','')}
            </div>
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                <span style="font-size:1.3rem;">{rule['icon']}</span>
                <span style="font-weight:800;font-size:1rem;color:#e2e8f0;">{rule['title']}</span>
            </div>
            <div style="color:#94a3b8;font-size:0.85rem;line-height:1.5;">
                {rule['description']}
            </div>
            {data_html}
            {exploit_html}
        </div>
        """, unsafe_allow_html=True)
