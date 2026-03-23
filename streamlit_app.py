"""
📊 TinNam AI - Data Analysis Platform V16
Premium dark-themed UI with 70+ prediction models + Constraint Engine + Multi-Set Portfolio.
Deploy: streamlit run streamlit_app.py
"""
import streamlit as st
import sys
import os
import time

# Setup path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from scraper.data_manager import (
    init_db, get_mega645_all, get_power655_all,
    get_mega645_numbers, get_power655_numbers,
    get_count, get_latest_date, get_recent
)

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="📊 TinNam AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================
# PASSWORD PROTECTION
# ============================================
def check_password():
    """Simple password gate."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    # Centered login form
    st.markdown("""
    <style>
        .login-container {
            max-width: 400px;
            margin: 15vh auto;
            padding: 40px;
            background: rgba(17, 24, 39, 0.95);
            border-radius: 20px;
            border: 1px solid rgba(99, 102, 241, 0.3);
            box-shadow: 0 0 60px rgba(99, 102, 241, 0.15);
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("## 📊 TinNam AI - Phân Tích Dữ Liệu")
        st.markdown("*Nhập mật khẩu để truy cập*")
        password = st.text_input("Mật khẩu", type="password", key="pw_input")
        if st.button("🔓 Đăng Nhập", use_container_width=True):
            if password == "1991":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ Sai mật khẩu!")
    return False


# ============================================
# CUSTOM CSS - Premium Dark Theme
# ============================================
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ---- Global ---- */
    .stApp {
        background: #0a0e1a;
        font-family: 'Inter', sans-serif;
    }
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background:
            radial-gradient(ellipse at 20% 50%, rgba(99, 102, 241, 0.08) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 20%, rgba(139, 92, 246, 0.06) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 80%, rgba(236, 72, 153, 0.04) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }

    /* Hide Streamlit defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container { padding-top: 1rem; max-width: 1400px; }

    /* ---- Title ---- */
    .main-title {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        letter-spacing: -1px;
        margin-bottom: 4px;
    }
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 16px;
    }

    /* ---- Stat Badges ---- */
    .stat-row {
        display: flex;
        justify-content: center;
        gap: 20px;
        flex-wrap: wrap;
        margin: 16px 0 24px;
    }
    .stat-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 18px;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 50px;
        font-size: 0.85rem;
        color: #94a3b8;
    }
    .stat-badge .val {
        color: #f59e0b;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ---- Cards ---- */
    .glass-card {
        background: rgba(17, 24, 39, 0.8);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.2);
        box-shadow: 0 0 30px rgba(99, 102, 241, 0.15);
    }
    .card-title-row {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        color: #f1f5f9;
    }

    /* ---- Data Balls ---- */
    .ball-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        justify-content: center;
        margin: 16px 0;
    }
    .data-ball {
        width: 58px; height: 58px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 1.3rem;
        font-family: 'JetBrains Mono', monospace;
        color: white;
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #ec4899);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
        animation: ballPop 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
    }
    .data-ball.master {
        width: 68px; height: 68px;
        font-size: 1.6rem;
        background: linear-gradient(135deg, #f43f5e, #7c3aed);
        box-shadow: 0 6px 20px rgba(244, 63, 94, 0.4);
    }
    .data-ball.small {
        width: 40px; height: 40px;
        font-size: 0.9rem;
    }
    .data-ball.bonus {
        background: linear-gradient(135deg, #f59e0b, #ef4444);
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);
    }
    @keyframes ballPop {
        0% { transform: scale(0); opacity: 0; }
        50% { transform: scale(1.2); }
        100% { transform: scale(1); opacity: 1; }
    }

    /* ---- Result Card ---- */
    .result-card {
        background: linear-gradient(135deg, rgba(34,197,94,0.04), rgba(244,63,94,0.04));
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 16px;
        padding: 28px;
        margin: 20px 0;
        text-align: center;
        box-shadow: 0 0 50px rgba(34, 197, 94, 0.15);
    }
    .confidence-score {
        font-size: 2.2rem;
        font-weight: 900;
    }
    .metric-row {
        display: flex;
        justify-content: center;
        gap: 28px;
        flex-wrap: wrap;
        margin-top: 16px;
    }
    .metric-item {
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 900;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 2px;
    }

    /* ---- Strategy Table ---- */
    .strat-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.88rem;
    }
    .strat-table th {
        background: rgba(99, 102, 241, 0.1);
        padding: 12px 14px;
        text-align: left;
        font-weight: 700;
        color: #06b6d4;
        border-bottom: 2px solid rgba(255,255,255,0.1);
    }
    .strat-table td {
        padding: 10px 14px;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        color: #e2e8f0;
    }
    .strat-table tr:hover td {
        background: rgba(99, 102, 241, 0.05);
    }
    .good { color: #22c55e; font-weight: 700; }
    .bad { color: #ef4444; font-weight: 700; }

    /* ---- Confidence Bar ---- */
    .conf-bar-wrap {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 5px;
    }
    .conf-bar-bg {
        flex: 1;
        height: 10px;
        background: rgba(255,255,255,0.05);
        border-radius: 5px;
        overflow: hidden;
    }
    .conf-bar {
        height: 100%;
        border-radius: 5px;
        transition: width 0.5s ease;
    }

    /* ---- History Table ---- */
    .hist-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.88rem;
    }
    .hist-table th {
        background: rgba(99, 102, 241, 0.1);
        padding: 12px 14px;
        text-align: left;
        font-weight: 700;
        color: #06b6d4;
        border-bottom: 2px solid rgba(255,255,255,0.1);
        white-space: nowrap;
    }
    .hist-table td {
        padding: 10px 14px;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        color: #e2e8f0;
        white-space: nowrap;
    }
    .hist-table tr:hover td {
        background: rgba(99, 102, 241, 0.05);
    }
    .date-cell {
        color: #94a3b8;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }
    .jackpot-cell {
        color: #f59e0b;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ---- Streamlit Overrides ---- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        justify-content: center;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 14px 36px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 50px;
        color: #94a3b8;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #ec4899) !important;
        color: white !important;
        border-color: transparent !important;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
    }
    div[data-testid="stExpander"] {
        background: rgba(17, 24, 39, 0.6);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
    }
    .stButton > button {
        border-radius: 50px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
    }

    /* Master predict button */
    div[data-testid="stVerticalBlock"] > div:has(> div > .stButton > button[kind="primary"]) .stButton > button {
        background: linear-gradient(135deg, #f43f5e, #7c3aed) !important;
        color: white !important;
        font-size: 1.2rem !important;
        padding: 16px 40px !important;
        box-shadow: 0 6px 25px rgba(244, 63, 94, 0.4) !important;
    }

    /* ---- Match Balls ---- */
    .match-ball {
        width: 36px; height: 36px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85rem;
        font-family: 'JetBrains Mono', monospace;
        margin: 2px;
        color: white;
        background: linear-gradient(135deg, #22c55e, #16a34a);
        box-shadow: 0 2px 8px rgba(34, 197, 94, 0.4);
    }
    .nomatch-ball {
        width: 36px; height: 36px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85rem;
        font-family: 'JetBrains Mono', monospace;
        margin: 2px;
        color: #64748b;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .pred-ball {
        width: 36px; height: 36px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85rem;
        font-family: 'JetBrains Mono', monospace;
        margin: 2px;
        color: white;
    }
    .pred-ball.hit {
        background: linear-gradient(135deg, #22c55e, #16a34a);
        box-shadow: 0 2px 8px rgba(34, 197, 94, 0.4);
    }
    .pred-ball.miss {
        background: rgba(239, 68, 68, 0.2);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: #f87171;
    }

    /* ---- Footer ---- */
    .footer-text {
        text-align: center;
        color: #64748b;
        font-size: 0.8rem;
        padding: 30px 0 16px;
    }
    .footer-text .warn {
        color: #ec4899;
        font-size: 0.75rem;
        margin-top: 6px;
    }
</style>
"""


# ============================================
# HELPER FUNCTIONS
# ============================================
def render_balls(numbers, css_class="data-ball"):
    """Render data balls as HTML."""
    return " ".join(
        f'<span class="{css_class}">{str(n).zfill(2)}</span>'
        for n in numbers
    )


def render_master_result(data):
    """Render master prediction result."""
    numbers = data.get("numbers", [])
    conf = data.get("confidence", {})
    bt = data.get("backtest", {})
    method = data.get("method", "")

    conf_score = conf.get("score", 0)
    conf_color = "#22c55e" if conf.get("level") == "high" else "#f59e0b" if conf.get("level") == "medium" else "#ef4444"

    balls_html = render_balls(numbers, "data-ball master")

    improvement = bt.get("improvement", 0)
    imp_color = "#22c55e" if improvement > 0 else "#ef4444"
    imp_sign = "+" if improvement > 0 else ""

    html = f"""
    <div class="result-card">
        <div style="font-size:0.9rem;color:#94a3b8;margin-bottom:8px;">Kỳ tiếp theo</div>
        <div style="font-size:1.1rem;font-weight:700;color:#22c55e;margin-bottom:12px;">🎯 DỰ ĐOÁN CHÍNH XÁC</div>
        <div class="ball-row">{balls_html}</div>
        <div class="metric-row">
            <div class="metric-item">
                <div class="metric-value" style="color:{conf_color};">{conf_score}%</div>
                <div class="metric-label">Confidence</div>
            </div>
            <div class="metric-item">
                <div class="metric-value" style="color:#6366f1;">{bt.get('avg', 0)}/6</div>
                <div class="metric-label">TB Backtest</div>
            </div>
            <div class="metric-item">
                <div class="metric-value" style="color:#f59e0b;">{bt.get('max', 0)}/6</div>
                <div class="metric-label">Max</div>
            </div>
            <div class="metric-item">
                <div class="metric-value" style="color:{imp_color};">{imp_sign}{improvement}%</div>
                <div class="metric-label">vs Random</div>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

    # V16 Method Selection Card
    ens_info = data.get("ensemble_info", {})
    if ens_info:
        st.markdown(f"""
        <div class="glass-card" style="border-color:#8b5cf6;">
            <div class="card-title-row">🧠 V16 Method Auto-Selection</div>
            <div style="display:flex;justify-content:center;gap:16px;flex-wrap:wrap;">
                <div style="text-align:center;padding:8px 14px;background:rgba(255,255,255,0.03);border-radius:10px;">
                    <div style="font-size:1.1rem;font-weight:800;color:#6366f1;font-family:JetBrains Mono,monospace;">{ens_info.get('base_avg', 0)}/6</div>
                    <div style="font-size:0.65rem;color:#64748b;">Signal V16</div>
                </div>
                <div style="text-align:center;padding:8px 14px;background:rgba(255,255,255,0.03);border-radius:10px;">
                    <div style="font-size:1.1rem;font-weight:800;color:#22c55e;font-family:JetBrains Mono,monospace;">{ens_info.get('ensemble_avg', 0)}/6</div>
                    <div style="font-size:0.65rem;color:#64748b;">Ensemble V16</div>
                </div>
                <div style="text-align:center;padding:8px 14px;background:rgba(255,255,255,0.03);border-radius:10px;">
                    <div style="font-size:1.1rem;font-weight:800;color:#f59e0b;font-family:JetBrains Mono,monospace;">{ens_info.get('constraint_avg', 0)}/6</div>
                    <div style="font-size:0.65rem;color:#64748b;">Constraint AI</div>
                </div>
                <div style="text-align:center;padding:8px 14px;background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.3);border-radius:10px;">
                    <div style="font-size:0.85rem;font-weight:700;color:#22c55e;">✅ {ens_info.get('chosen', '')}</div>
                    <div style="font-size:0.65rem;color:#64748b;">Best Method</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # V16 Portfolio Display
    portfolio = data.get("portfolio", [])
    if portfolio:
        portfolio_html = '<div class="glass-card" style="border-color:#f59e0b;">'
        portfolio_html += '<div class="card-title-row">🎯 Multi-Set Portfolio (%d bộ số)</div>' % len(portfolio)
        portfolio_html += '<div style="font-size:0.75rem;color:#94a3b8;text-align:center;margin-bottom:12px;">Mỗi bộ được tối ưu hóa độc lập - đa dạng hóa cơ hội trúng</div>'
        for idx, pset in enumerate(portfolio):
            badge = '⭐' if idx == 0 else f'#{idx+1}'
            balls = ''.join(f'<span class="ball">{n}</span>' for n in pset)
            portfolio_html += f'<div style="display:flex;align-items:center;gap:10px;margin:6px 0;padding:8px 12px;background:rgba(255,255,255,0.02);border-radius:10px;border-left:3px solid {"#f59e0b" if idx==0 else "#334155"};">'
            portfolio_html += f'<span style="font-size:0.8rem;min-width:28px;color:{"#f59e0b" if idx==0 else "#64748b"};">{badge}</span>'
            portfolio_html += f'<div>{balls}</div>'
            portfolio_html += '</div>'
        portfolio_html += '</div>'
        st.markdown(portfolio_html, unsafe_allow_html=True)

    # Backtest details
    if bt.get("tests"):
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-title-row">🧪 Kết Quả Backtest ({bt.get('tests', 0)} kỳ)</div>
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:12px;">
                <div style="text-align:center;padding:12px;background:rgba(255,255,255,0.03);border-radius:10px;">
                    <div style="font-size:1.3rem;font-weight:800;color:#22c55e;">{bt.get('match_3plus', 0)}</div>
                    <div style="font-size:0.7rem;color:#64748b;">Trúng 3+ số</div>
                </div>
                <div style="text-align:center;padding:12px;background:rgba(255,255,255,0.03);border-radius:10px;">
                    <div style="font-size:1.3rem;font-weight:800;color:#f59e0b;">{bt.get('match_4plus', 0)}</div>
                    <div style="font-size:0.7rem;color:#64748b;">Trúng 4+ số</div>
                </div>
                <div style="text-align:center;padding:12px;background:rgba(255,255,255,0.03);border-radius:10px;">
                    <div style="font-size:1.3rem;font-weight:800;color:#f43f5e;">{bt.get('match_5plus', 0)}</div>
                    <div style="font-size:0.7rem;color:#64748b;">Trúng 5+ số</div>
                </div>
                <div style="text-align:center;padding:12px;background:rgba(255,255,255,0.03);border-radius:10px;">
                    <div style="font-size:1.3rem;font-weight:800;color:#6366f1;">{bt.get('random_expected', 0)}</div>
                    <div style="font-size:0.7rem;color:#64748b;">Random TB</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Distribution
    dist = bt.get("distribution", {})
    if dist:
        dist_html = ""
        for k, v in dist.items():
            pct = (v / max(bt.get("tests", 1), 1) * 100)
            dist_html += f'<div style="padding:6px 12px;background:rgba(255,255,255,0.03);border-radius:8px;text-align:center;"><div style="font-weight:700;color:#e2e8f0;">{k} số</div><div style="font-size:0.7rem;color:#64748b;">{v}x ({pct:.1f}%)</div></div>'
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-title-row">📊 Phân Bố Số Trùng</div>
            <div style="display:flex;gap:8px;flex-wrap:wrap;">{dist_html}</div>
        </div>
        """, unsafe_allow_html=True)

    # Score details
    score_details = data.get("score_details", [])
    if score_details:
        bars_html = ""
        for s in score_details:
            is_sel = s.get("selected", False)
            bg_style = "background:linear-gradient(135deg,#f43f5e,#7c3aed);box-shadow:0 2px 10px rgba(244,63,94,0.3);" if is_sel else ""
            bar_bg = "linear-gradient(90deg,#f43f5e,#7c3aed)" if is_sel else "linear-gradient(90deg,#6366f1,#0ea5e9)"
            weight = "700" if is_sel else "400"
            bars_html += f"""
            <div class="conf-bar-wrap">
                <span class="data-ball small" style="{bg_style}">{str(s['number']).zfill(2)}</span>
                <div class="conf-bar-bg"><div class="conf-bar" style="width:{s.get('confidence',0)}%;background:{bar_bg};"></div></div>
                <div style="font-size:0.75rem;min-width:55px;text-align:right;color:#64748b;font-weight:{weight};">{s.get('score',0)} pts</div>
            </div>"""
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-title-row">📊 Điểm Tin Cậy (Top 15 số)</div>
            {bars_html}
        </div>
        """, unsafe_allow_html=True)

    if method:
        st.markdown(f'<div style="text-align:center;font-size:0.75rem;color:#64748b;margin-top:8px;">{method}</div>', unsafe_allow_html=True)


def render_dan_result(dan_data, ver):
    """Render dàn prediction result."""
    if not isinstance(dan_data, dict):
        st.warning(f"⚠️ Dữ liệu dàn {ver} không hợp lệ. Vui lòng bấm lại nút tạo dàn.")
        return
    cands = dan_data.get("candidates", [])
    total = dan_data.get("total", 0)
    info = dan_data.get("info", "")
    combos = dan_data.get("combos", [])
    if not cands:
        st.warning(f"⚠️ Dàn {ver} không có dữ liệu candidates. Vui lòng bấm lại nút tạo dàn.")
        return

    ver_label = "V1 - ĐẦY ĐỦ" if ver == "v1" else "V2 - TỐI ƯU"
    ver_icon = "📊" if ver == "v1" else "⚡"
    ver_color = "#6366f1" if ver == "v1" else "#22c55e"
    ver_desc = "Block + Direction + S/L" if ver == "v1" else "Block + Direction + S/L + Gap + Sum"

    # Candidates per column
    cols_html = ""
    for p, nums in enumerate(cands):
        balls = ""
        for n in nums:
            balls += f'<span style="display:inline-flex;align-items:center;justify-content:center;width:38px;height:38px;border-radius:50%;background:linear-gradient(135deg,{ver_color},#8b5cf6);color:white;font-weight:700;font-size:0.9rem;font-family:JetBrains Mono,monospace;margin:3px;">{str(n).zfill(2)}</span>'
        cols_html += f'''
        <div style="margin-bottom:12px;">
            <div style="font-size:0.85rem;font-weight:700;color:#94a3b8;margin-bottom:6px;">Cột {p+1} ({len(nums)} số)</div>
            <div style="display:flex;flex-wrap:wrap;gap:2px;">{balls}</div>
        </div>'''

    # Cost estimate
    cost = total * 10000
    if cost >= 1_000_000_000:
        cost_str = f"{cost/1_000_000_000:.1f} tỷ"
    elif cost >= 1_000_000:
        cost_str = f"{cost/1_000_000:.0f} triệu"
    else:
        cost_str = f"{cost/1000:.0f}K"

    # Sample combos
    sample_html = ""
    show_n = min(10, len(combos))
    for i in range(show_n):
        c = combos[i]
        row_balls = " ".join(
            f'<span style="display:inline-flex;align-items:center;justify-content:center;width:30px;height:30px;border-radius:50%;background:rgba(99,102,241,0.2);border:1px solid rgba(99,102,241,0.4);color:#e2e8f0;font-weight:600;font-size:0.75rem;font-family:JetBrains Mono,monospace;">{str(n).zfill(2)}</span>'
            for n in c
        )
        sample_html += f'<div style="margin:4px 0;padding:6px 10px;background:rgba(0,0,0,0.2);border-radius:8px;display:flex;align-items:center;gap:6px;"><span style="color:#64748b;font-size:0.7rem;min-width:24px;">#{i+1}</span>{row_balls}</div>'

    st.markdown(f"""
    <div class="glass-card" style="border-color:{ver_color};box-shadow:0 0 40px {ver_color}22;">
        <div class="card-title-row">{ver_icon} DÀN {ver_label}</div>
        <div style="display:flex;justify-content:center;gap:24px;margin-bottom:16px;flex-wrap:wrap;">
            <div style="text-align:center;padding:12px 20px;background:rgba(255,255,255,0.03);border-radius:12px;">
                <div style="font-size:1.8rem;font-weight:900;color:{ver_color};font-family:JetBrains Mono,monospace;">{total:,}</div>
                <div style="font-size:0.75rem;color:#64748b;">Tổng dãy số</div>
            </div>
            <div style="text-align:center;padding:12px 20px;background:rgba(255,255,255,0.03);border-radius:12px;">
                <div style="font-size:1.8rem;font-weight:900;color:#f59e0b;font-family:JetBrains Mono,monospace;">{cost_str}</div>
                <div style="font-size:0.75rem;color:#64748b;">Chi phí (10K/vé)</div>
            </div>
        </div>
        <div style="font-size:0.8rem;color:#64748b;text-align:center;margin-bottom:16px;">Bộ lọc: {ver_desc}</div>
        {cols_html}
    </div>
    """, unsafe_allow_html=True)

    if sample_html:
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-title-row">🎫 Mẫu Dãy Số (10/{total:,})</div>
            {sample_html}
        </div>
        """, unsafe_allow_html=True)


def render_history_table(rows, lottery_type):
    """Render history table as HTML."""
    is_power = lottery_type == "power"

    body = ""
    for idx, row in enumerate(rows):
        numbers = [row['n1'], row['n2'], row['n3'], row['n4'], row['n5'], row['n6']]
        balls = ""
        for n in numbers:
            balls += f'<span style="display:inline-flex;align-items:center;justify-content:center;width:36px;height:36px;border-radius:50%;background:linear-gradient(135deg,#6366f1,#8b5cf6);color:white;font-weight:700;font-size:0.85rem;font-family:JetBrains Mono,monospace;margin:2px;">{str(n).zfill(2)}</span>'
        bonus_html = ""
        if is_power:
            bn = row.get('bonus', 0)
            bonus_html = f'<td style="padding:8px;"><span style="display:inline-flex;align-items:center;justify-content:center;width:36px;height:36px;border-radius:50%;background:linear-gradient(135deg,#f59e0b,#ef4444);color:white;font-weight:700;font-size:0.85rem;font-family:JetBrains Mono,monospace;">{str(bn).zfill(2)}</span></td>'
        jackpot = row.get('jackpot', '-')
        if jackpot and len(str(jackpot)) > 15:
            jackpot = str(jackpot).split('≈')[-1].strip() if '≈' in str(jackpot) else str(jackpot)[-15:]
        body += f'''<tr>
            <td style="color:#64748b;padding:8px;text-align:center;">{idx + 1}</td>
            <td style="color:#94a3b8;padding:8px;font-family:JetBrains Mono,monospace;font-size:0.85rem;white-space:nowrap;">{row.get('draw_date', '')}</td>
            <td style="padding:8px;"><div style="display:flex;gap:4px;flex-wrap:wrap;">{balls}</div></td>
            {bonus_html}
            <td style="color:#f59e0b;font-weight:600;padding:8px;font-family:JetBrains Mono,monospace;font-size:0.85rem;">{jackpot}</td>
        </tr>'''

    header = '<th style="background:rgba(99,102,241,0.1);padding:12px;color:#06b6d4;font-weight:700;border-bottom:2px solid rgba(255,255,255,0.1);">#</th>'
    header += '<th style="background:rgba(99,102,241,0.1);padding:12px;color:#06b6d4;font-weight:700;border-bottom:2px solid rgba(255,255,255,0.1);">Ngày</th>'
    header += '<th style="background:rgba(99,102,241,0.1);padding:12px;color:#06b6d4;font-weight:700;border-bottom:2px solid rgba(255,255,255,0.1);">Kết Quả</th>'
    if is_power:
        header += '<th style="background:rgba(99,102,241,0.1);padding:12px;color:#06b6d4;font-weight:700;border-bottom:2px solid rgba(255,255,255,0.1);">Số ĐB</th>'
    header += '<th style="background:rgba(99,102,241,0.1);padding:12px;color:#06b6d4;font-weight:700;border-bottom:2px solid rgba(255,255,255,0.1);">Jackpot</th>'

    html_content = f"""
    <div class="glass-card">
        <div class="card-title-row">📋 Lịch Sử Kết Quả {'Bộ B (6/55)' if is_power else 'Bộ A (6/45)'}</div>
        <div style="overflow-x:auto;border-radius:10px;">
            <table style="width:100%;border-collapse:collapse;">
                <thead><tr>{header}</tr></thead>
                <tbody>{body}</tbody>
            </table>
        </div>
    </div>
    """
    try:
        st.html(html_content)
    except AttributeError:
        st.markdown(html_content, unsafe_allow_html=True)


def render_phase_result(data, phase_name, phase_icon, phase_color):
    """Generic renderer for phase analysis results."""
    v = data.get("verdict", {})
    score = v.get("score", 0)
    best = v.get("best_strategy", {})
    verdict_text = v.get("verdict", "")

    # Verdict card
    st.markdown(f"""
    <div class="glass-card" style="border-color:{phase_color};box-shadow:0 0 50px {phase_color}44;">
        <div class="card-title-row">{phase_icon} {phase_name} - KẾT QUẢ</div>
        <div style="text-align:center;margin:16px 0;">
            <div style="font-size:1.3rem;font-weight:700;color:{phase_color};">Best: {best.get('name', 'N/A')}</div>
            <div style="font-size:2.5rem;font-weight:900;color:#22c55e;margin:8px 0;">{best.get('avg', 0)}/6</div>
            <div style="font-size:1.1rem;font-weight:700;color:{'#22c55e' if best.get('improvement', 0) > 0 else '#ef4444'};">
                {"+" if best.get('improvement', 0) > 0 else ""}{best.get('improvement', 0)}% so với random
            </div>
            <div style="font-size:0.85rem;color:#94a3b8;margin-top:6px;">"{verdict_text}"</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Next prediction
    np_data = data.get("next_prediction", {})
    numbers = np_data.get("numbers") or np_data.get("primary", [])
    if numbers:
        balls_html = render_balls(numbers, "data-ball master")
        st.markdown(f"""
        <div class="glass-card" style="border-color:#22c55e;box-shadow:0 0 30px rgba(34,197,94,0.2);">
            <div class="card-title-row">🔮 DỰ ĐOÁN KỲ TIẾP</div>
            <div class="ball-row">{balls_html}</div>
            <div style="text-align:center;font-size:0.8rem;color:#64748b;">{np_data.get('method', '')}</div>
        </div>
        """, unsafe_allow_html=True)

    # Strategy ranking
    ranking = v.get("strategy_ranking", [])
    if ranking:
        rows_html = ""
        for i, s in enumerate(ranking):
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else str(i + 1)
            imp = s.get("improvement", 0)
            imp_cls = "good" if imp > 0 else "bad"
            bg = f"background:rgba(34,197,94,0.08);" if imp > 10 else ""
            rows_html += f"""<tr style="{bg}">
                <td>{medal}</td><td><strong>{s.get('name', '')}</strong></td>
                <td><strong>{s.get('avg', '')}</strong></td><td>{s.get('max', '-')}/6</td>
                <td>{s.get('match_3plus', 0)}</td><td>{s.get('match_4plus', 0)}</td>
                <td><span class="{imp_cls}">{"+" if imp > 0 else ""}{imp}%</span></td>
            </tr>"""
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-title-row">🏆 Xếp Hạng Chiến Lược (Walk-Forward Backtest)</div>
            <div style="overflow-x:auto;">
                <table class="strat-table">
                    <thead><tr><th>#</th><th>Chiến Lược</th><th>TB/6</th><th>Max</th><th>3+</th><th>4+</th><th>vs Random</th></tr></thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Evidence
    evidence = v.get("evidence", [])
    if evidence:
        ev_html = ""
        for e in evidence:
            c = "#22c55e" if e.startswith("+") else "#ef4444"
            ev_html += f'<div style="color:{c};">{e}</div>'
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-title-row">📝 Tổng Kết</div>
            <div style="font-family:monospace;font-size:0.78rem;line-height:1.8;">{ev_html}</div>
        </div>
        """, unsafe_allow_html=True)


def render_stats(analysis, lottery_type):
    """Render statistics analysis."""
    a = analysis
    max_num = 45 if lottery_type == "mega" else 55

    # Summary stats
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:16px;margin-bottom:20px;">
        <div class="glass-card" style="text-align:center;padding:16px;">
            <div style="font-size:1.8rem;font-weight:900;background:linear-gradient(135deg,#6366f1,#8b5cf6,#ec4899);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-family:'JetBrains Mono',monospace;">{a['total_draws']}</div>
            <div style="font-size:0.8rem;color:#64748b;">Tổng Số Kỳ</div>
        </div>
        <div class="glass-card" style="text-align:center;padding:16px;">
            <div style="font-size:1.8rem;font-weight:900;background:linear-gradient(135deg,#6366f1,#8b5cf6,#ec4899);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-family:'JetBrains Mono',monospace;">{a['avg_sum']}</div>
            <div style="font-size:0.8rem;color:#64748b;">Tổng TB</div>
        </div>
        <div class="glass-card" style="text-align:center;padding:16px;">
            <div style="font-size:1.8rem;font-weight:900;background:linear-gradient(135deg,#6366f1,#8b5cf6,#ec4899);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-family:'JetBrains Mono',monospace;">{a['avg_odd_count']}</div>
            <div style="font-size:0.8rem;color:#64748b;">TB Số Lẻ</div>
        </div>
        <div class="glass-card" style="text-align:center;padding:16px;">
            <div style="font-size:1.8rem;font-weight:900;background:linear-gradient(135deg,#6366f1,#8b5cf6,#ec4899);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-family:'JetBrains Mono',monospace;">{a['sum_range'][0]}-{a['sum_range'][1]}</div>
            <div style="font-size:0.8rem;color:#64748b;">Range Tổng</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Hot & Cold numbers
    col1, col2 = st.columns(2)
    with col1:
        hot_html = ""
        for h in a.get("hot_numbers", []):
            pct = h["count"] / max(a["total_draws"], 1) * 100
            hot_html += f"""<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                <span style="font-family:'JetBrains Mono',monospace;font-weight:700;font-size:0.85rem;min-width:28px;text-align:right;color:#94a3b8;">{str(h['number']).zfill(2)}</span>
                <div style="flex:1;height:24px;background:rgba(255,255,255,0.03);border-radius:4px;overflow:hidden;">
                    <div style="height:100%;width:{min(pct*4, 100)}%;background:linear-gradient(135deg,#f59e0b,#ef4444);border-radius:4px;"></div>
                </div>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#64748b;min-width:40px;text-align:right;">{h['count']}x</span>
            </div>"""
        st.markdown(f"""<div class="glass-card">
            <div class="card-title-row">🔥 Số Nóng</div>{hot_html}
        </div>""", unsafe_allow_html=True)

    with col2:
        cold_html = ""
        for c in a.get("cold_numbers", []):
            pct = c["count"] / max(a["total_draws"], 1) * 100
            cold_html += f"""<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                <span style="font-family:'JetBrains Mono',monospace;font-weight:700;font-size:0.85rem;min-width:28px;text-align:right;color:#94a3b8;">{str(c['number']).zfill(2)}</span>
                <div style="flex:1;height:24px;background:rgba(255,255,255,0.03);border-radius:4px;overflow:hidden;">
                    <div style="height:100%;width:{min(pct*4, 100)}%;background:linear-gradient(135deg,#3b82f6,#6366f1);border-radius:4px;"></div>
                </div>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#64748b;min-width:40px;text-align:right;">{c['count']}x</span>
            </div>"""
        st.markdown(f"""<div class="glass-card">
            <div class="card-title-row">❄️ Số Lạnh</div>{cold_html}
        </div>""", unsafe_allow_html=True)

    # Overdue numbers
    overdue = a.get("overdue_numbers", [])
    if overdue:
        overdue_html = ""
        for o in overdue:
            overdue_html += f"""<div style="text-align:center;margin:8px;">
                <span class="data-ball">{str(o['number']).zfill(2)}</span>
                <div style="font-size:0.75rem;color:#64748b;margin-top:4px;">{o['gap']} kỳ</div>
            </div>"""
        st.markdown(f"""<div class="glass-card">
            <div class="card-title-row">⏰ Số Quá Hạn (Lâu chưa xuất hiện)</div>
            <div style="display:flex;flex-wrap:wrap;justify-content:center;">{overdue_html}</div>
        </div>""", unsafe_allow_html=True)

    # Frequency grid
    all_freq = a.get("all_frequency", {})
    if all_freq:
        counts = [f["count"] for f in all_freq.values()]
        max_c = max(counts) if counts else 1
        min_c = min(counts) if counts else 0
        th_hot = max_c - (max_c - min_c) * 0.2
        th_cold = min_c + (max_c - min_c) * 0.2

        grid_html = ""
        for n in range(1, max_num + 1):
            f = all_freq.get(str(n)) or all_freq.get(n)
            if not f:
                continue
            cls = ""
            bg_style = "background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);"
            if f["count"] >= th_hot:
                bg_style = "background:rgba(236,72,153,0.15);border:1px solid rgba(236,72,153,0.3);"
                cls = "color:#ec4899;"
            elif f["count"] <= th_cold:
                bg_style = "background:rgba(6,182,212,0.15);border:1px solid rgba(6,182,212,0.3);"
                cls = "color:#06b6d4;"
            grid_html += f"""<div style="padding:8px 4px;border-radius:8px;text-align:center;font-family:'JetBrains Mono',monospace;font-weight:700;font-size:0.85rem;{bg_style}{cls}cursor:pointer;">
                {str(n).zfill(2)}
                <span style="display:block;font-size:0.65rem;color:#64748b;font-weight:400;margin-top:2px;">{f['count']}</span>
            </div>"""

        st.markdown(f"""<div class="glass-card">
            <div class="card-title-row">🔢 Bảng Tần Suất Toàn Bộ</div>
            <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(50px,1fr));gap:8px;">{grid_html}</div>
        </div>""", unsafe_allow_html=True)


# ============================================
# BACKTEST RESULTS RENDERER
# ============================================
def render_backtest_results(bt_results, lottery_type):
    """Render backtest results with ranking, distribution, and detail view."""
    if 'error' in bt_results:
        st.error(f"❌ {bt_results['error']}")
        return

    models = bt_results.get('models', [])
    best = bt_results.get('best_model', {})
    total_iters = bt_results.get('total_iterations', 0)

    if not models:
        st.warning("Không có kết quả.")
        return

    # ---- Summary Card ----
    best_name = best.get('model', 'N/A')
    best_avg = best.get('avg_matches', 0)
    best_max = best.get('best_score', 0)
    best_3plus = best.get('at_least_3', 0)

    st.markdown(f"""
    <div class="result-card" style="border-color:rgba(99,102,241,0.3);box-shadow:0 0 50px rgba(99,102,241,0.15);">
        <div style="font-size:0.9rem;color:#94a3b8;margin-bottom:8px;">🧪 Backtest: {total_iters} kỳ kiểm tra</div>
        <div style="font-size:1.2rem;font-weight:700;color:#22c55e;margin-bottom:12px;">🏆 Phương pháp tốt nhất: {best_name}</div>
        <div class="metric-row">
            <div class="metric-item">
                <div class="metric-value" style="color:#6366f1;">{best_avg:.2f}/6</div>
                <div class="metric-label">TB Trùng</div>
            </div>
            <div class="metric-item">
                <div class="metric-value" style="color:#f59e0b;">{best_max}/6</div>
                <div class="metric-label">Max Trùng</div>
            </div>
            <div class="metric-item">
                <div class="metric-value" style="color:#22c55e;">{best_3plus}%</div>
                <div class="metric-label">Trùng 3+ số</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Ranking Table ----
    rows_html = ""
    random_model = next((x for x in models if x['model'] == 'Random Baseline'), None)
    random_avg = random_model['avg_matches'] if random_model else 0.8

    for m in models:
        rank = m.get('rank', 0)
        medal = ["🥇", "🥈", "🥉"][rank - 1] if rank <= 3 else str(rank)
        name = m.get('model', '')
        avg = m.get('avg_matches', 0)
        best_s = m.get('best_score', 0)
        a3 = m.get('at_least_3', 0)
        a2 = m.get('at_least_2', 0)

        improvement = ((avg - random_avg) / random_avg * 100) if random_avg > 0 else 0
        imp_cls = "good" if improvement > 0 else "bad"
        imp_sign = "+" if improvement > 0 else ""

        bg = "background:rgba(34,197,94,0.06);" if rank <= 3 else ""
        is_random = "opacity:0.5;" if name == 'Random Baseline' else ""

        rows_html += f"""<tr style="{bg}{is_random}">
            <td style="padding:10px 14px;border-bottom:1px solid rgba(255,255,255,0.06);color:#e2e8f0;text-align:center;">{medal}</td>
            <td style="padding:10px 14px;border-bottom:1px solid rgba(255,255,255,0.06);color:#e2e8f0;font-weight:600;">{name}</td>
            <td style="padding:10px 14px;border-bottom:1px solid rgba(255,255,255,0.06);color:#6366f1;font-weight:700;font-family:JetBrains Mono,monospace;">{avg:.3f}</td>
            <td style="padding:10px 14px;border-bottom:1px solid rgba(255,255,255,0.06);color:#f59e0b;font-weight:700;">{best_s}/6</td>
            <td style="padding:10px 14px;border-bottom:1px solid rgba(255,255,255,0.06);color:#22c55e;">{a3}%</td>
            <td style="padding:10px 14px;border-bottom:1px solid rgba(255,255,255,0.06);color:#94a3b8;">{a2}%</td>
            <td style="padding:10px 14px;border-bottom:1px solid rgba(255,255,255,0.06);"><span class="{imp_cls}">{imp_sign}{improvement:.1f}%</span></td>
        </tr>"""

    th_style = 'style="background:rgba(99,102,241,0.1);padding:12px 14px;color:#06b6d4;font-weight:700;border-bottom:2px solid rgba(255,255,255,0.1);white-space:nowrap;"'
    st.markdown(f"""
    <div class="glass-card">
        <div class="card-title-row">🏆 Xếp Hạng Phương Pháp Dự Đoán</div>
        <div style="overflow-x:auto;border-radius:10px;">
            <table style="width:100%;border-collapse:collapse;">
                <thead><tr>
                    <th {th_style}>#</th>
                    <th {th_style}>Phương Pháp</th>
                    <th {th_style}>TB/6</th>
                    <th {th_style}>Max</th>
                    <th {th_style}>3+</th>
                    <th {th_style}>2+</th>
                    <th {th_style}>vs Random</th>
                </tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Distribution Grid ----
    if models and models[0].get('match_distribution'):
        best_dist = models[0]['match_distribution']
        total_tests = models[0].get('total_tests', 1)
        dist_html = ""
        colors = ['#ef4444', '#f97316', '#f59e0b', '#22c55e', '#06b6d4', '#6366f1', '#ec4899']
        for i in range(7):
            count = int(best_dist.get(str(i), 0))
            pct = count / total_tests * 100 if total_tests > 0 else 0
            color = colors[i]
            dist_html += f'''<div style="text-align:center;padding:12px 8px;background:rgba(255,255,255,0.03);border-radius:10px;border:1px solid {color}33;">
                <div style="font-size:1.3rem;font-weight:800;color:{color};font-family:JetBrains Mono,monospace;">{count}</div>
                <div style="font-size:0.7rem;color:#64748b;margin-top:2px;">{pct:.1f}%</div>
                <div style="font-size:0.75rem;font-weight:600;color:#94a3b8;margin-top:4px;">Trùng {i}</div>
            </div>'''

        st.markdown(f"""
        <div class="glass-card">
            <div class="card-title-row">📊 Phân Bố Trùng Số — {models[0].get('model', '')}</div>
            <div style="display:grid;grid-template-columns:repeat(7,1fr);gap:10px;">{dist_html}</div>
        </div>
        """, unsafe_allow_html=True)

    # ---- Detail View: Recent Tests ----
    recent = models[0].get('recent_results', []) if models else []
    if recent:
        detail_rows = ""
        for idx, test in enumerate(reversed(recent)):
            predicted = set(test.get('predicted', []))
            actual = set(test.get('actual', []))
            matches = test.get('matches', 0)

            # Actual balls with match highlight
            actual_balls = ""
            for n in sorted(actual):
                if n in predicted:
                    actual_balls += f'<span class="match-ball">{str(n).zfill(2)}</span>'
                else:
                    actual_balls += f'<span class="nomatch-ball">{str(n).zfill(2)}</span>'

            # Predicted balls with hit/miss
            pred_balls = ""
            for n in sorted(predicted):
                if n in actual:
                    pred_balls += f'<span class="pred-ball hit">{str(n).zfill(2)}</span>'
                else:
                    pred_balls += f'<span class="pred-ball miss">{str(n).zfill(2)}</span>'

            match_color = '#22c55e' if matches >= 3 else '#f59e0b' if matches >= 2 else '#ef4444'

            detail_rows += f'''<tr>
                <td style="padding:8px;border-bottom:1px solid rgba(255,255,255,0.06);color:#64748b;text-align:center;">{test.get('draw_index', '')}</td>
                <td style="padding:8px;border-bottom:1px solid rgba(255,255,255,0.06);"><div style="display:flex;gap:2px;flex-wrap:wrap;">{actual_balls}</div></td>
                <td style="padding:8px;border-bottom:1px solid rgba(255,255,255,0.06);"><div style="display:flex;gap:2px;flex-wrap:wrap;">{pred_balls}</div></td>
                <td style="padding:8px;border-bottom:1px solid rgba(255,255,255,0.06);text-align:center;font-weight:800;font-size:1.1rem;color:{match_color};font-family:JetBrains Mono,monospace;">{matches}/6</td>
            </tr>'''

        dth = 'style="background:rgba(99,102,241,0.1);padding:10px 14px;color:#06b6d4;font-weight:700;border-bottom:2px solid rgba(255,255,255,0.1);"'
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-title-row">🔍 Chi Tiết Các Kỳ Gần Nhất — {models[0].get('model', '')}</div>
            <div style="overflow-x:auto;border-radius:10px;">
                <table style="width:100%;border-collapse:collapse;">
                    <thead><tr>
                        <th {dth}>Kỳ</th>
                        <th {dth}>Kết Quả Thực</th>
                        <th {dth}>Dự Đoán</th>
                        <th {dth}>Trùng</th>
                    </tr></thead>
                    <tbody>{detail_rows}</tbody>
                </table>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================
# DATA TAB CONTENT
# ============================================
def render_lottery_tab(lottery_type):
    """Render a full data analysis tab."""
    type_name = "Bộ A (6/45)" if lottery_type == "mega" else "Bộ B (6/55)"
    max_num = 45 if lottery_type == "mega" else 55
    pick = 6

    # ---- MASTER PREDICTION ----
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    if st.button(f"🎯 DỰ ĐOÁN KỲ TIẾP THEO", key=f"master_{lottery_type}", type="primary", use_container_width=True):
        with st.spinner("🎯 AI V16 đang phân tích: Constraint Engine + N-gram + Ensemble + Portfolio... Vui lòng chờ 3-6 phút."):
            try:
                from models.master_predictor import MasterPredictor
                if lottery_type == "mega":
                    data = get_mega645_numbers()
                else:
                    data = get_power655_numbers()
                predictor = MasterPredictor(max_num, pick)
                result = predictor.predict(data)
                st.session_state[f"master_result_{lottery_type}"] = result
            except Exception as e:
                st.error(f"❌ Lỗi: {e}")

    # Show master result if exists
    if f"master_result_{lottery_type}" in st.session_state:
        render_master_result(st.session_state[f"master_result_{lottery_type}"])

    st.markdown("</div>", unsafe_allow_html=True)

    # ---- DÀN PREDICTION ----
    dan_col1, dan_col2 = st.columns(2)
    with dan_col1:
        if st.button("📊 DÀN V1 - ĐẦY ĐỦ", key=f"dan_v1_{lottery_type}", use_container_width=True):
            with st.spinner("📊 Đang tạo dàn V1..."):
                try:
                    from models.dan_predictor import predict_dan
                    nums = get_mega645_numbers() if lottery_type == "mega" else get_power655_numbers()
                    is_mega = (lottery_type == "mega")
                    cands, combos, info = predict_dan(nums, max_num, pick, is_mega, version="v1")
                    st.session_state[f"dan_result_v1_{lottery_type}"] = {
                        "candidates": cands, "combos": combos[:200], "info": info,
                        "total": len(combos)
                    }
                except Exception as e:
                    st.error(f"❌ {e}")
    with dan_col2:
        if st.button("⚡ DÀN V2 - TỐI ƯU", key=f"dan_v2_{lottery_type}", use_container_width=True):
            with st.spinner("⚡ Đang tạo dàn V2 tối ưu..."):
                try:
                    from models.dan_predictor import predict_dan
                    nums = get_mega645_numbers() if lottery_type == "mega" else get_power655_numbers()
                    is_mega = (lottery_type == "mega")
                    cands, combos, info = predict_dan(nums, max_num, pick, is_mega, version="v2")
                    st.session_state[f"dan_result_v2_{lottery_type}"] = {
                        "candidates": cands, "combos": combos[:200], "info": info,
                        "total": len(combos)
                    }
                except Exception as e:
                    st.error(f"❌ {e}")

    # Show dàn results
    for ver in ["v1", "v2"]:
        skey = f"dan_result_{ver}_{lottery_type}"
        if skey in st.session_state:
            dan_data = st.session_state[skey]
            if isinstance(dan_data, dict) and "candidates" in dan_data:
                render_dan_result(dan_data, ver)

    # ---- BACKTEST: KIỂM TRA PHƯƠNG PHÁP ----
    with st.expander("🧪 Kiểm Tra Độ Chính Xác Các Phương Pháp (Backtest các kỳ trước)..."):
        bt_col1, bt_col2 = st.columns([3, 1])
        with bt_col1:
            bt_tests = st.selectbox(
                "Số kỳ kiểm tra",
                [50, 100, 200, 500],
                index=1,
                key=f"bt_tests_{lottery_type}",
                help="Số kỳ quá khứ để kiểm tra. Nhiều hơn = chính xác hơn nhưng chạy lâu hơn."
            )
        with bt_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            run_bt = st.button(
                "🚀 Chạy Backtest",
                key=f"run_bt_{lottery_type}",
                use_container_width=True
            )

        if run_bt:
            with st.spinner(f"🧪 Đang chạy backtest {bt_tests} kỳ... Vui lòng chờ 1-3 phút."):
                try:
                    from models.backtester import BacktestEngine
                    if lottery_type == "mega":
                        data = get_mega645_numbers()
                        bt_engine = BacktestEngine(45, 6)
                    else:
                        data = get_power655_numbers()
                        # Power data has 7 elements (6 + bonus), slice to 6
                        data = [d[:6] for d in data]
                        bt_engine = BacktestEngine(55, 6)

                    step_size = max(1, (len(data) - 50) // bt_tests)
                    result = bt_engine.run_backtest(
                        data,
                        start_from=50,
                        step=step_size,
                        max_tests=bt_tests
                    )
                    st.session_state[f"backtest_result_{lottery_type}"] = result
                except Exception as e:
                    st.error(f"❌ Lỗi backtest: {e}")

        # Show stored backtest result
        if f"backtest_result_{lottery_type}" in st.session_state:
            render_backtest_results(
                st.session_state[f"backtest_result_{lottery_type}"],
                lottery_type
            )

    # ---- PHASE TOOLS (collapsed) ----
    with st.expander("🛠️ Công cụ phân tích chi tiết (Phase 1-7)..."):
        phase_cols = st.columns(4)

        # Phase buttons
        phases = [
            ("🔮 Dự Đoán Cơ Bản", "predict", "models.ensemble_model", "EnsembleModel"),
            ("📈 Thống Kê", "stats", None, None),
            ("🔓 PRNG Cracker", "crack", "models.prng_cracker", "PRNGCracker"),
            ("📅 Temporal", "temporal", "models.temporal_analyzer", "DeepTemporalAnalyzer"),
            ("🚀 Phase 2", "phase2", "models.phase2_cracker", "Phase2Cracker"),
            ("🔍 Phase 3", "phase3", "models.phase3_forensic", "ForensicAnalyzer"),
            ("🎯 Phase 4", "phase4", "models.phase4_exploit", "ExploitEngine"),
            ("🏆 Phase 5", "phase5", "models.phase5_ultra", "UltraOptimizer"),
            ("🧠 Phase 6", "phase6", "models.phase6_deep", "DeepIntelligenceEngine"),
            ("👑 Phase 7", "phase7", "models.phase7_ultimate", "UltimatePredictor"),
        ]

        cols = st.columns(5)
        for i, (label, key, module, cls_name) in enumerate(phases):
            with cols[i % 5]:
                if st.button(label, key=f"{key}_{lottery_type}", use_container_width=True):
                    if key == "stats":
                        # Stats uses different flow
                        with st.spinner("Đang phân tích..."):
                            try:
                                from models.ensemble_model import EnsembleModel
                                if lottery_type == "mega":
                                    nums = get_mega645_numbers()
                                else:
                                    nums = get_power655_numbers()
                                model = EnsembleModel(max_num, pick)
                                model.fit(nums, train_deep=False)
                                analysis = model.get_analysis()
                                st.session_state[f"stats_result_{lottery_type}"] = analysis
                            except Exception as e:
                                st.error(f"❌ {e}")
                    elif key == "predict":
                        with st.spinner("Đang dự đoán..."):
                            try:
                                from models.ensemble_model import EnsembleModel
                                if lottery_type == "mega":
                                    nums = get_mega645_numbers()
                                else:
                                    nums = get_power655_numbers()
                                model = EnsembleModel(max_num, pick)
                                model.fit(nums, train_deep=False)
                                results = model.predict_all_models(nums, n_sets=3)
                                st.session_state[f"predict_result_{lottery_type}"] = {
                                    "models": results,
                                    "total_draws": len(nums),
                                    "training_info": model.training_info
                                }
                            except Exception as e:
                                st.error(f"❌ {e}")
                    else:
                        with st.spinner(f"⏳ {label} đang chạy... Vui lòng chờ 2-5 phút."):
                            try:
                                mod = __import__(module, fromlist=[cls_name])
                                EngineClass = getattr(mod, cls_name)

                                if lottery_type == "mega":
                                    nums = get_mega645_numbers()
                                else:
                                    nums = get_power655_numbers()

                                if key == "temporal":
                                    if lottery_type == "mega":
                                        full_data = get_mega645_all()
                                    else:
                                        full_data = get_power655_all()
                                    engine = EngineClass(max_num, pick)
                                    result = engine.analyze(full_data)
                                elif key == "phase2":
                                    cross = get_power655_numbers() if lottery_type == "mega" else get_mega645_numbers()
                                    cross = [d[:6] for d in cross]
                                    engine = EngineClass(max_num, pick)
                                    result = engine.analyze(nums, cross_data=cross)
                                else:
                                    engine = EngineClass(max_num, pick)
                                    result = engine.analyze(nums)

                                st.session_state[f"{key}_result_{lottery_type}"] = result
                            except Exception as e:
                                st.error(f"❌ {e}")

        # Render stored results
        if f"predict_result_{lottery_type}" in st.session_state:
            pred_data = st.session_state[f"predict_result_{lottery_type}"]
            for key, model_data in pred_data.get("models", {}).items():
                balls_html = ""
                for pred_set in model_data.get("predictions", []):
                    balls = render_balls(pred_set, "data-ball small")
                    balls_html += f'<div style="margin:6px 0;padding:8px 12px;background:rgba(0,0,0,0.2);border-radius:10px;"><div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">{balls}</div></div>'
                status_text = "✅ Trained" if model_data.get("status") == "trained" else "⚡ Fast"
                st.markdown(f"""<div class="glass-card">
                    <div class="card-title-row">{model_data.get('icon', '🔮')} {model_data.get('name', key)} <span style="font-size:0.75rem;padding:3px 10px;border-radius:50px;background:rgba(16,185,129,0.15);color:#10b981;margin-left:auto;">{status_text}</span></div>
                    {balls_html}
                </div>""", unsafe_allow_html=True)

        if f"stats_result_{lottery_type}" in st.session_state:
            render_stats(st.session_state[f"stats_result_{lottery_type}"], lottery_type)

        for phase_key, phase_label, phase_icon, phase_color in [
            ("crack", "PRNG CRACKER", "🔓", "#dc2626"),
            ("temporal", "TEMPORAL ANALYZER", "📅", "#6366f1"),
            ("phase2", "PHASE 2 CRACKER", "🚀", "#7c3aed"),
            ("phase3", "PHASE 3 FORENSIC", "🔍", "#059669"),
            ("phase4", "PHASE 4 EXPLOIT", "🎯", "#ea580c"),
            ("phase5", "PHASE 5 ULTRA", "🏆", "#d4af37"),
            ("phase6", "PHASE 6 DEEP INTELLIGENCE", "🧠", "#6366f1"),
            ("phase7", "PHASE 7 ULTIMATE", "👑", "#f43f5e"),
        ]:
            result_key = f"{phase_key}_result_{lottery_type}"
            if result_key in st.session_state:
                render_phase_result(
                    st.session_state[result_key],
                    phase_label, phase_icon, phase_color
                )

    # ---- HISTORY TABLE ----
    limit = st.selectbox(
        "Số kỳ hiển thị",
        [20, 50, 200, 9999],
        format_func=lambda x: "Tất cả" if x == 9999 else str(x),
        key=f"limit_{lottery_type}"
    )
    rows = get_recent(lottery_type, limit)
    if rows:
        render_history_table(rows, lottery_type)
    else:
        st.info("📭 Chưa có dữ liệu. Hãy cập nhật dữ liệu trước!")


# ============================================
# MAIN APP
# ============================================
def main():
    if not check_password():
        return

    # Inject CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ---- AUTO-UPDATE DATA ON FIRST LOAD ----
    if "auto_update_done" not in st.session_state:
        st.session_state.auto_update_done = False
        st.session_state.auto_update_result = None

    if not st.session_state.auto_update_done:
        with st.spinner("🔄 Đang kiểm tra & cập nhật dữ liệu mới nhất..."):
            try:
                from scraper.auto_updater import auto_update_data
                result = auto_update_data()
                st.session_state.auto_update_result = result
                st.session_state.auto_update_done = True
                if result['status'] == 'updated' and (result['mega_new'] > 0 or result['power_new'] > 0):
                    st.rerun()  # Rerun to show fresh data in header
            except Exception as e:
                st.session_state.auto_update_result = {
                    'status': 'error',
                    'message': f'⚠️ Auto-update error: {str(e)[:100]}',
                }
                st.session_state.auto_update_done = True

    # Show auto-update status notification
    update_result = st.session_state.get("auto_update_result")
    if update_result:
        status = update_result.get('status', '')
        msg = update_result.get('message', '')
        if status == 'updated' and (update_result.get('mega_new', 0) > 0 or update_result.get('power_new', 0) > 0):
            st.success(f"🔄 {msg}")
        elif status == 'error':
            st.warning(f"{msg}")
        else:
            st.info(f"📊 {msg}")

    # ---- HEADER ----
    mega_count = get_count("mega")
    power_count = get_count("power")
    mega_latest = get_latest_date("mega")

    st.markdown('<div class="main-title">📊 TinNam AI - Phân Tích Dữ Liệu</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">V16 — Constraint Engine + N-gram Mining + Multi-Set Portfolio | Maximum Accuracy AI</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-badge">📊 Bộ A (6/45): <span class="val">{mega_count}</span> kỳ</div>
        <div class="stat-badge">⚡ Bộ B (6/55): <span class="val">{power_count}</span> kỳ</div>
        <div class="stat-badge">🕐 Cập nhật: <span class="val">{mega_latest or 'Chưa có'}</span></div>
    </div>
    """, unsafe_allow_html=True)

    # ---- TABS ----
    tab_mega, tab_power = st.tabs(["🟢 Bộ A (6/45)", "🔴 Bộ B (6/55)"])

    with tab_mega:
        render_lottery_tab("mega")

    with tab_power:
        render_lottery_tab("power")

    # ---- SIDEBAR ----
    with st.sidebar:
        st.markdown("### ⚙️ Cài Đặt")
        if st.button("🔄 Cập Nhật Dữ Liệu (Thủ Công)", use_container_width=True):
            with st.spinner("Đang thu thập dữ liệu mới..."):
                try:
                    from scraper.scraper import scrape_all
                    scrape_all()
                    st.success(f"✅ Mega: {get_count('mega')} | Power: {get_count('power')}")
                    # Reset auto-update state so next rerun shows fresh counts
                    st.session_state.auto_update_done = False
                    st.session_state.auto_update_result = None
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ {e}")

        if st.button("🚪 Đăng Xuất", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()

    # ---- FOOTER ----
    st.markdown("""
    <div class="footer-text">
        📊 TinNam AI - Phân Tích Dữ Liệu © 2026 | Hệ thống phân tích dữ liệu đa chiều
        <div class="warn">⚠️ Lưu ý: Dự đoán dựa trên phân tích dữ liệu lịch sử, chỉ mang tính tham khảo.</div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
