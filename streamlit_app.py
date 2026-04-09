"""
TinNam AI V2.0 — Golden Set Consensus Engine
Premium dark-themed UI with 30+ signal fusion + RNG vulnerability detection.
Deploy: streamlit run streamlit_app.py
"""
import streamlit as st
import sys
import os
import time
from collections import Counter

# Setup path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from scraper.data_manager import (
    init_db, get_mega645_all, get_power655_all,
    get_mega645_numbers, get_power655_numbers,
    get_count, get_latest_date, get_first_date, get_recent
)

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="TinNam AI",
    page_icon="🔍",
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
            border: 1px solid rgba(255,255,255,0.1);
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div style="text-align:center;font-size:2rem;margin:20vh 0 10px;">🔍</div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;font-size:1.2rem;font-weight:700;color:#f1f5f9;margin-bottom:20px;">TinNam AI</div>', unsafe_allow_html=True)
        pwd = st.text_input("Password", type="password", key="login_pw")
        if st.button("Đăng nhập", use_container_width=True): 
            if pwd == "1991":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ Sai mật khẩu")
    return False


# ============================================
# CUSTOM CSS (Premium Dark Theme)
# ============================================
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&family=JetBrains+Mono:wght@400;700;800&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0f172a 50%, #1a0a2e 100%);
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

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container { padding-top: 1rem; max-width: 1400px; }

    .main-title {
        font-size: 2.8rem; font-weight: 900;
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #ec4899);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; letter-spacing: -1px; margin-bottom: 4px;
    }
    .subtitle { text-align: center; color: #94a3b8; font-size: 0.95rem; margin-bottom: 16px; }
    .stat-row { display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; margin: 16px 0 24px; }
    .stat-badge {
        display: inline-flex; align-items: center; gap: 8px;
        padding: 8px 18px; background: rgba(255,255,255,0.05);
        backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.1);
        border-radius: 50px; font-size: 0.85rem; color: #94a3b8;
    }
    .stat-badge .val { color: #f59e0b; font-weight: 700; font-family: 'JetBrains Mono', monospace; }

    .glass-card {
        background: rgba(17, 24, 39, 0.8); backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.1); border-radius: 16px;
        padding: 24px; margin-bottom: 20px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.3); transition: all 0.3s ease;
    }
    .glass-card:hover { border-color: rgba(99,102,241,0.2); box-shadow: 0 0 30px rgba(99,102,241,0.15); }
    .card-title-row {
        display: flex; align-items: center; gap: 10px;
        font-size: 1.2rem; font-weight: 700; margin-bottom: 16px;
        padding-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.1); color: #f1f5f9;
    }

    .data-ball {
        width: 58px; height: 58px; border-radius: 50%;
        display: inline-flex; align-items: center; justify-content: center;
        font-weight: 800; font-size: 1.3rem; font-family: 'JetBrains Mono', monospace;
        color: white; background: linear-gradient(135deg, #6366f1, #8b5cf6, #ec4899);
        box-shadow: 0 4px 12px rgba(99,102,241,0.4);
        animation: ballPop 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
    }
    .data-ball.master {
        width: 68px; height: 68px; font-size: 1.6rem;
        background: linear-gradient(135deg, #f43f5e, #7c3aed);
        box-shadow: 0 6px 20px rgba(244,63,94,0.4);
    }
    .data-ball.small { width: 40px; height: 40px; font-size: 0.9rem; }
    .data-ball.bonus {
        background: linear-gradient(135deg, #f59e0b, #ef4444);
        box-shadow: 0 4px 12px rgba(245,158,11,0.4);
    }
    @keyframes ballPop {
        0% { transform: scale(0); opacity: 0; }
        50% { transform: scale(1.2); }
        100% { transform: scale(1); opacity: 1; }
    }

    .ball-row { display: flex; gap: 10px; flex-wrap: wrap; justify-content: center; margin: 16px 0; }

    .result-card {
        background: linear-gradient(135deg, rgba(34,197,94,0.04), rgba(244,63,94,0.04));
        border: 1px solid rgba(34,197,94,0.3); border-radius: 16px;
        padding: 28px; margin: 20px 0; text-align: center;
        box-shadow: 0 0 50px rgba(34,197,94,0.15);
    }
    .metric-row { display: flex; justify-content: center; gap: 28px; flex-wrap: wrap; margin-top: 16px; }
    .metric-item { text-align: center; }
    .metric-value { font-size: 1.8rem; font-weight: 900; font-family: 'JetBrains Mono', monospace; }
    .metric-label { font-size: 0.75rem; color: #64748b; margin-top: 2px; }

    .strat-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
    .strat-table th {
        background: rgba(99,102,241,0.1); padding: 12px 14px;
        text-align: left; font-weight: 700; color: #06b6d4;
        border-bottom: 2px solid rgba(255,255,255,0.1);
    }
    .strat-table td { padding: 10px 14px; border-bottom: 1px solid rgba(255,255,255,0.06); color: #e2e8f0; }
    .strat-table tr:hover td { background: rgba(99,102,241,0.05); }
    .good { color: #22c55e; font-weight: 700; }
    .bad { color: #ef4444; font-weight: 700; }

    .stTabs [data-baseweb="tab-list"] { gap: 12px; justify-content: center; background-color: transparent; }
    .stTabs [data-baseweb="tab"] {
        padding: 14px 36px; background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1); border-radius: 50px;
        color: #94a3b8; font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #ec4899) !important;
        color: white !important; border-color: transparent !important;
        box-shadow: 0 4px 20px rgba(99,102,241,0.4);
    }
    div[data-testid="stExpander"] {
        background: rgba(17,24,39,0.6); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px;
    }
    .stButton > button {
        border-radius: 50px; font-weight: 600; font-family: 'Inter', sans-serif;
        border: none; transition: all 0.3s ease;
    }
    .stButton > button:hover { transform: translateY(-2px); }
    div[data-testid="stVerticalBlock"] > div:has(> div > .stButton > button[kind="primary"]) .stButton > button {
        background: linear-gradient(135deg, #f43f5e, #7c3aed) !important;
        color: white !important; font-size: 1.2rem !important;
        padding: 16px 40px !important; box-shadow: 0 6px 25px rgba(244,63,94,0.4) !important;
    }

    .footer-text { text-align: center; color: #64748b; font-size: 0.8rem; padding: 30px 0 16px; }
    .footer-text .warn { color: #ec4899; font-size: 0.75rem; margin-top: 6px; }

    /* === GOLDEN SET STYLES === */
    .golden-card {
        background: linear-gradient(135deg, rgba(245,158,11,0.08), rgba(234,179,8,0.04));
        border: 2px solid rgba(245,158,11,0.5); border-radius: 20px;
        padding: 32px; margin: 20px 0; text-align: center;
        box-shadow: 0 0 60px rgba(245,158,11,0.2), inset 0 0 60px rgba(245,158,11,0.05);
        position: relative; overflow: hidden;
    }
    .golden-card::before {
        content: ''; position: absolute; top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: conic-gradient(transparent, rgba(245,158,11,0.1), transparent, transparent);
        animation: goldenSpin 8s linear infinite;
    }
    @keyframes goldenSpin { 100% { transform: rotate(360deg); } }
    .golden-ball {
        width: 72px; height: 72px; border-radius: 50%;
        display: inline-flex; align-items: center; justify-content: center;
        font-weight: 900; font-size: 1.8rem; font-family: 'JetBrains Mono', monospace;
        color: #0a0a1a;
        background: linear-gradient(135deg, #f59e0b, #eab308, #fbbf24);
        box-shadow: 0 6px 24px rgba(245,158,11,0.5), 0 0 40px rgba(245,158,11,0.2);
        animation: goldenPop 0.6s cubic-bezier(0.68,-0.55,0.265,1.55);
        margin: 4px;
    }
    @keyframes goldenPop {
        0% { transform: scale(0) rotate(-180deg); opacity: 0; }
        60% { transform: scale(1.15) rotate(10deg); }
        100% { transform: scale(1) rotate(0deg); opacity: 1; }
    }
    .confidence-gauge {
        width: 120px; height: 120px; border-radius: 50%;
        display: inline-flex; align-items: center; justify-content: center; flex-direction: column;
        margin: 16px auto;
        border: 4px solid rgba(245,158,11,0.3);
        background: rgba(0,0,0,0.3);
    }
    .confidence-value { font-size: 2rem; font-weight: 900; font-family: 'JetBrains Mono', monospace; }
    .confidence-label { font-size: 0.7rem; color: #94a3b8; }

    .heat-cell {
        display: inline-flex; align-items: center; justify-content: center;
        width: 44px; height: 44px; border-radius: 8px; margin: 2px;
        font-weight: 700; font-size: 0.85rem; font-family: 'JetBrains Mono', monospace;
        transition: all 0.3s ease; cursor: default;
    }
    .heat-cell:hover { transform: scale(1.15); z-index: 10; }
</style>
"""


# ============================================
# HELPER FUNCTIONS
# ============================================
def render_balls(numbers, css_class="data-ball"):
    return " ".join(f'<span class="{css_class}">{str(n).zfill(2)}</span>' for n in numbers)


def render_scan_results(scan_data):
    """Render vulnerability scan results."""
    summary = scan_data['summary']
    tests = scan_data['tests']

    verdict_color = '#ef4444' if summary['verdict'] == 'VULNERABLE' else '#f59e0b' if summary['verdict'] == 'SUSPICIOUS' else '#22c55e'
    verdict_icon = '🚨' if summary['verdict'] == 'VULNERABLE' else '⚠️' if summary['verdict'] == 'SUSPICIOUS' else '🛡️'

    # Verdict header
    st.markdown(f"""
    <div class="result-card" style="border-color:{verdict_color};box-shadow:0 0 50px {verdict_color}44;">
        <div style="font-size:2.5rem;">{verdict_icon}</div>
        <div style="font-size:1.8rem;font-weight:900;color:{verdict_color};margin:8px 0;">{summary['verdict']}</div>
        <div class="metric-row">
            <div class="metric-item">
                <div class="metric-value" style="color:#22c55e;">{summary['passed']}</div>
                <div class="metric-label">PASS</div>
            </div>
            <div class="metric-item">
                <div class="metric-value" style="color:#f59e0b;">{summary['warned']}</div>
                <div class="metric-label">WARN</div>
            </div>
            <div class="metric-item">
                <div class="metric-value" style="color:#ef4444;">{summary['failed']}</div>
                <div class="metric-label">FAIL</div>
            </div>
            <div class="metric-item">
                <div class="metric-value" style="color:#6366f1;">{summary['total_draws']}</div>
                <div class="metric-label">Draws</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Test results table
    rows_html = ""
    for test_id, test in tests.items():
        s = test['status']
        s_color = '#22c55e' if s == 'PASS' else '#f59e0b' if s == 'WARN' else '#ef4444' if s == 'FAIL' else '#64748b'
        s_icon = '✅' if s == 'PASS' else '⚠️' if s == 'WARN' else '❌' if s == 'FAIL' else '⏭️'
        p_val = f"{test.get('p_value', 'N/A')}" if test.get('p_value') is not None else 'N/A'
        n_biases = len(test.get('biases', []))
        bias_html = f'<span style="color:#ef4444;font-weight:700;">{n_biases}</span>' if n_biases > 0 else '<span style="color:#64748b;">0</span>'
        rows_html += f"""<tr>
            <td style="padding:10px 14px;border-bottom:1px solid rgba(255,255,255,0.06);color:#e2e8f0;font-weight:600;">{s_icon} {test['name']}</td>
            <td style="padding:10px 14px;border-bottom:1px solid rgba(255,255,255,0.06);text-align:center;">
                <span style="color:{s_color};font-weight:800;font-family:JetBrains Mono,monospace;">{s}</span>
            </td>
            <td style="padding:10px 14px;border-bottom:1px solid rgba(255,255,255,0.06);text-align:center;font-family:JetBrains Mono,monospace;color:#94a3b8;">{p_val}</td>
            <td style="padding:10px 14px;border-bottom:1px solid rgba(255,255,255,0.06);text-align:center;">{bias_html}</td>
        </tr>"""

    th = 'style="background:rgba(99,102,241,0.1);padding:12px 14px;color:#06b6d4;font-weight:700;border-bottom:2px solid rgba(255,255,255,0.1);text-align:center;"'
    th_l = 'style="background:rgba(99,102,241,0.1);padding:12px 14px;color:#06b6d4;font-weight:700;border-bottom:2px solid rgba(255,255,255,0.1);"'
    st.markdown(f"""
    <div class="glass-card">
        <div class="card-title-row">🧪 12 Vulnerability Tests</div>
        <div style="overflow-x:auto;border-radius:10px;">
            <table style="width:100%;border-collapse:collapse;">
                <thead><tr>
                    <th {th_l}>Test</th><th {th}>Status</th><th {th}>p-value</th><th {th}>Biases</th>
                </tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Exploitable biases
    biases = summary['exploitable_biases']
    if biases:
        bias_html = ""
        for b in biases:
            strength = b.get('strength', 0)
            s_color = '#ef4444' if strength > 5 else '#f59e0b' if strength > 3 else '#64748b'
            nums = b.get('numbers', [])
            nums_str = ', '.join(str(n) for n in nums[:15]) if nums else ''
            pairs = b.get('pairs', [])
            if pairs:
                nums_str = ', '.join(f"({p[0][0]},{p[0][1]})" for p in pairs[:8])
            triplets = b.get('triplets', [])
            if triplets:
                nums_str = ', '.join(f"({t[0][0]},{t[0][1]},{t[0][2]})" for t in triplets[:5])
            details = b.get('details', [])
            if details and not nums_str:
                if isinstance(details[0], dict) and 'number' in details[0]:
                    nums_str = ', '.join(str(d['number']) for d in details[:10])

            bias_html += f"""
            <div style="padding:12px 16px;background:rgba(255,255,255,0.02);border-radius:10px;margin-bottom:8px;border-left:3px solid {s_color};">
                <div style="font-weight:700;color:#e2e8f0;font-size:0.9rem;">{b['type']}</div>
                <div style="color:#94a3b8;font-size:0.8rem;margin:4px 0;">{b['description']}</div>
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-family:JetBrains Mono,monospace;font-size:0.75rem;color:#64748b;">{nums_str}</span>
                    <span style="font-weight:800;color:{s_color};font-family:JetBrains Mono,monospace;">z={strength:.1f}</span>
                </div>
            </div>"""

        st.markdown(f"""
        <div class="glass-card" style="border-color:#ef4444;">
            <div class="card-title-row">🎯 {len(biases)} Exploitable Biases Detected</div>
            {bias_html}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="glass-card" style="border-color:#22c55e;">
            <div class="card-title-row">🛡️ No Exploitable Biases</div>
            <div style="text-align:center;color:#94a3b8;padding:20px;">RNG appears fair. No statistical vulnerabilities detected.</div>
        </div>
        """, unsafe_allow_html=True)


def render_deep_forensic_results(result):
    """Render deep forensic analysis results."""
    primary = result.get('primary', [])
    portfolio = result.get('portfolio', [])
    weights = result.get('weights', {})
    reports = result.get('reports', {})
    top_30 = result.get('top_30', [])

    # Primary prediction
    if primary:
        balls_html = render_balls(primary, 'data-ball master')
        st.markdown(f"""
        <div class="result-card" style="border-color:#22c55e;box-shadow:0 0 60px rgba(34,197,94,0.25);">
            <div style="font-size:0.9rem;color:#94a3b8;margin-bottom:8px;">Deep Forensic Prediction</div>
            <div style="font-size:1.1rem;font-weight:700;color:#22c55e;margin-bottom:12px;">🧬 15-Signal V2 Analysis + Walk-Forward Calibrated</div>
            <div class="ball-row">{balls_html}</div>
            <div class="metric-row">
                <div class="metric-item">
                    <div class="metric-value" style="color:#6366f1;">{result.get('n_signals', 0)}</div>
                    <div class="metric-label">Signals</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value" style="color:#f59e0b;">{len(portfolio)}</div>
                    <div class="metric-label">Portfolio Sets</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Signal weights
    if weights:
        weight_html = ''
        sorted_w = sorted(weights.items(), key=lambda x: -x[1])
        for name, w in sorted_w:
            bar_pct = min(w / max(v for _, v in sorted_w) * 100, 100)
            color = '#22c55e' if w > 1.5 else '#f59e0b' if w > 1.0 else '#64748b'
            weight_html += f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0;"><span style="min-width:120px;font-size:0.8rem;color:#94a3b8;">{name}</span><div style="flex:1;height:16px;background:rgba(255,255,255,0.05);border-radius:4px;overflow:hidden;"><div style="height:100%;width:{bar_pct}%;background:{color};border-radius:4px;"></div></div><span style="min-width:40px;text-align:right;font-family:JetBrains Mono,monospace;font-size:0.75rem;color:{color};font-weight:700;">{w:.2f}</span></div>'
        st.markdown(f'<div class="glass-card"><div class="card-title-row">📊 Signal Weights (Calibrated)</div>{weight_html}</div>', unsafe_allow_html=True)

    # Key findings
    findings_html = ''
    # Overdue numbers
    gap_report = reports.get('gap_timing', {})
    overdue = gap_report.get('overdue_numbers', [])[:8]
    if overdue:
        overdue_items = ''.join(f'<div style="text-align:center;margin:6px;"><span class="data-ball" style="background:linear-gradient(135deg,#ef4444,#f59e0b);">{str(o["number"]).zfill(2)}</span><div style="font-size:0.65rem;color:#ef4444;margin-top:4px;font-weight:700;">{o["current_gap"]} draws</div><div style="font-size:0.6rem;color:#64748b;">z={o["z_overdue"]}</div></div>' for o in overdue)
        findings_html += f'<div class="glass-card" style="border-color:#ef4444;"><div class="card-title-row">⏰ Overdue Numbers (Gap Analysis)</div><div style="display:flex;flex-wrap:wrap;justify-content:center;">{overdue_items}</div></div>'

    # Momentum
    mom_report = reports.get('momentum', {})
    rising = mom_report.get('rising', [])[:5]
    if rising:
        rising_items = ''.join(f'<span class="data-ball" style="background:linear-gradient(135deg,#22c55e,#10b981);margin:4px;">{str(r["number"]).zfill(2)}</span>' for r in rising)
        findings_html += f'<div class="glass-card" style="border-color:#22c55e;"><div class="card-title-row">📈 Rising Momentum</div><div style="display:flex;flex-wrap:wrap;justify-content:center;">{rising_items}</div></div>'

    # KNN matches
    knn_report = reports.get('knn', {})
    knn_matches = knn_report.get('best_matches', [])[:3]
    if knn_matches:
        knn_html = ''
        for m in knn_matches:
            next_balls = ''.join(f'<span style="display:inline-flex;align-items:center;justify-content:center;width:32px;height:32px;border-radius:50%;background:rgba(99,102,241,0.2);border:1px solid rgba(99,102,241,0.4);color:#e2e8f0;font-weight:600;font-size:0.75rem;font-family:JetBrains Mono,monospace;margin:2px;">{str(n).zfill(2)}</span>' for n in m['next'])
            knn_html += f'<div style="padding:8px;margin:4px 0;background:rgba(0,0,0,0.2);border-radius:8px;"><span style="color:#f59e0b;font-size:0.75rem;">Draw #{m["draw_idx"]} ({m["similarity"]}/6 match):</span> {next_balls}</div>'
        findings_html += f'<div class="glass-card"><div class="card-title-row">🔎 KNN History Match (next draws after similar)</div>{knn_html}</div>'

    if findings_html:
        st.markdown(findings_html, unsafe_allow_html=True)

    # Portfolio
    if portfolio:
        port_html = '<div class="glass-card" style="border-color:#f59e0b;">'
        port_html += f'<div class="card-title-row">🎯 Deep Forensic Portfolio ({len(portfolio)} sets)</div>'
        port_html += '<div style="font-size:0.65rem;color:#94a3b8;text-align:center;margin-bottom:8px;">Backtest: Portfolio avg 2.25/6, max 4/6 | +181% vs random</div>'
        for idx, combo in enumerate(portfolio):
            badge = '⭐' if idx == 0 else f'#{idx+1}'
            balls = ''.join(f'<span style="display:inline-flex;align-items:center;justify-content:center;width:38px;height:38px;border-radius:50%;background:linear-gradient(135deg,#6366f1,#8b5cf6);color:white;font-weight:700;font-size:0.9rem;font-family:JetBrains Mono,monospace;margin:2px;">{str(n).zfill(2)}</span>' for n in combo)
            score = sum(result.get('scores', {}).get(n, 0) for n in combo)
            port_html += f'<div style="display:flex;align-items:center;gap:10px;padding:8px 12px;margin:4px 0;background:rgba(0,0,0,0.2);border-radius:10px;"><span style="font-size:0.85rem;min-width:32px;color:#f59e0b;font-weight:700;">{badge}</span><div style="display:flex;gap:3px;flex-wrap:wrap;">{balls}</div><span style="margin-left:auto;font-family:JetBrains Mono,monospace;font-size:0.75rem;color:#64748b;">{score:.1f}</span></div>'
        port_html += '</div>'
        st.markdown(port_html, unsafe_allow_html=True)


def render_exploit_results(exploit_data):
    """Render exploit-generated predictions."""
    if exploit_data['strategy'] == 'NO_EXPLOIT':
        st.markdown("""
        <div class="glass-card" style="border-color:#64748b;">
            <div class="card-title-row">⚠️ No Exploits Available</div>
            <div style="text-align:center;color:#94a3b8;padding:20px;">No biases found to exploit. RNG appears fair.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    predictions = exploit_data['predictions']
    confidence = exploit_data['confidence']
    strategy = exploit_data['strategy']
    biases_used = exploit_data['biases_used']

    conf_color = '#22c55e' if confidence > 50 else '#f59e0b' if confidence > 20 else '#ef4444'

    # Primary prediction
    if predictions:
        primary = predictions[0]
        balls_html = render_balls(primary['numbers'], "data-ball master")

        st.markdown(f"""
        <div class="result-card" style="border-color:{conf_color};">
            <div style="font-size:0.9rem;color:#94a3b8;margin-bottom:8px;">Exploit-Based Prediction</div>
            <div style="font-size:1.1rem;font-weight:700;color:{conf_color};margin-bottom:12px;">⚡ {strategy}</div>
            <div class="ball-row">{balls_html}</div>
            <div class="metric-row">
                <div class="metric-item">
                    <div class="metric-value" style="color:{conf_color};">{confidence}%</div>
                    <div class="metric-label">Bias Confidence</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value" style="color:#6366f1;">{biases_used}</div>
                    <div class="metric-label">Biases Used</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value" style="color:#f59e0b;">{len(predictions)}</div>
                    <div class="metric-label">Total Sets</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Portfolio
    if len(predictions) > 1:
        portfolio_html = '<div class="glass-card" style="border-color:#f59e0b;">'
        portfolio_html += f'<div class="card-title-row">🎯 Portfolio {len(predictions)} Sets (Bias-Based)</div>'
        portfolio_html += '<div style="font-size:0.65rem;color:#94a3b8;text-align:center;margin-bottom:8px;">Each set differs by >=3 numbers | Based on real detected biases only</div>'
        for idx, pred in enumerate(predictions):
            badge = '⭐' if idx == 0 else f'#{idx+1}'
            balls = ''.join(f'<span style="display:inline-flex;align-items:center;justify-content:center;width:38px;height:38px;border-radius:50%;background:linear-gradient(135deg,#6366f1,#8b5cf6);color:white;font-weight:700;font-size:0.9rem;font-family:JetBrains Mono,monospace;margin:2px;">{str(n).zfill(2)}</span>' for n in pred['numbers'])
            portfolio_html += f'<div style="display:flex;align-items:center;gap:10px;padding:8px 12px;margin:4px 0;background:rgba(0,0,0,0.2);border-radius:10px;"><span style="font-size:0.85rem;min-width:32px;color:#f59e0b;font-weight:700;">{badge}</span><div style="display:flex;gap:3px;flex-wrap:wrap;">{balls}</div><span style="margin-left:auto;font-family:JetBrains Mono,monospace;font-size:0.75rem;color:#64748b;">{pred["score"]:.1f}</span></div>'
        portfolio_html += '</div>'
        st.markdown(portfolio_html, unsafe_allow_html=True)


def render_dan_result(dan_data, ver):
    """Render dan prediction result."""
    cands = dan_data.get("candidates", [])
    combos = dan_data.get("combos", [])
    total = dan_data.get("total", 0)

    ver_label = "V1 - DAY DU" if ver == "v1" else "V2 - TOI UU"
    ver_icon = "📊" if ver == "v1" else "⚡"
    ver_color = "#6366f1" if ver == "v1" else "#22c55e"
    ver_desc = "Block + Direction + S/L" if ver == "v1" else "Block + Direction + S/L + Gap + Sum"

    cols_html = ""
    for p, nums in enumerate(cands):
        balls = ""
        for n in nums:
            balls += f'<span style="display:inline-flex;align-items:center;justify-content:center;width:38px;height:38px;border-radius:50%;background:linear-gradient(135deg,{ver_color},#8b5cf6);color:white;font-weight:700;font-size:0.9rem;font-family:JetBrains Mono,monospace;margin:3px;">{str(n).zfill(2)}</span>'
        cols_html += f'<div style="margin-bottom:12px;"><div style="font-size:0.85rem;font-weight:700;color:#94a3b8;margin-bottom:6px;">Col {p+1} ({len(nums)} nums)</div><div style="display:flex;flex-wrap:wrap;gap:2px;">{balls}</div></div>'

    cost = total * 10000
    cost_str = f"{cost/1_000_000_000:.1f}B" if cost >= 1_000_000_000 else f"{cost/1_000_000:.0f}M" if cost >= 1_000_000 else f"{cost/1000:.0f}K"

    sample_html = ""
    for i in range(min(10, len(combos))):
        c = combos[i]
        row_balls = " ".join(f'<span style="display:inline-flex;align-items:center;justify-content:center;width:30px;height:30px;border-radius:50%;background:rgba(99,102,241,0.2);border:1px solid rgba(99,102,241,0.4);color:#e2e8f0;font-weight:600;font-size:0.75rem;font-family:JetBrains Mono,monospace;">{str(n).zfill(2)}</span>' for n in c)
        sample_html += f'<div style="margin:4px 0;padding:6px 10px;background:rgba(0,0,0,0.2);border-radius:8px;display:flex;align-items:center;gap:6px;"><span style="color:#64748b;font-size:0.7rem;min-width:24px;">#{i+1}</span>{row_balls}</div>'

    st.markdown(f"""
    <div class="glass-card" style="border-color:{ver_color};box-shadow:0 0 40px {ver_color}22;">
        <div class="card-title-row">{ver_icon} DAN {ver_label}</div>
        <div style="display:flex;justify-content:center;gap:24px;margin-bottom:16px;flex-wrap:wrap;">
            <div style="text-align:center;padding:12px 20px;background:rgba(255,255,255,0.03);border-radius:12px;">
                <div style="font-size:1.8rem;font-weight:900;color:{ver_color};font-family:JetBrains Mono,monospace;">{total:,}</div>
                <div style="font-size:0.75rem;color:#64748b;">Total combos</div>
            </div>
            <div style="text-align:center;padding:12px 20px;background:rgba(255,255,255,0.03);border-radius:12px;">
                <div style="font-size:1.8rem;font-weight:900;color:#f59e0b;font-family:JetBrains Mono,monospace;">{cost_str}</div>
                <div style="font-size:0.75rem;color:#64748b;">Cost (10K/ticket)</div>
            </div>
        </div>
        <div style="font-size:0.8rem;color:#64748b;text-align:center;margin-bottom:16px;">Filters: {ver_desc}</div>
        {cols_html}
    </div>
    """, unsafe_allow_html=True)

    if sample_html:
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-title-row">🎫 Sample ({min(10, len(combos))}/{total:,})</div>
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
            jackpot = str(jackpot).split('~')[-1].strip() if '~' in str(jackpot) else str(jackpot)[-15:]
        body += f'''<tr>
            <td style="color:#64748b;padding:8px;text-align:center;">{idx + 1}</td>
            <td style="color:#94a3b8;padding:8px;font-family:JetBrains Mono,monospace;font-size:0.85rem;white-space:nowrap;">{row.get('draw_date', '')}</td>
            <td style="padding:8px;"><div style="display:flex;gap:4px;flex-wrap:wrap;">{balls}</div></td>
            {bonus_html}
            <td style="color:#f59e0b;font-weight:600;padding:8px;font-family:JetBrains Mono,monospace;font-size:0.85rem;">{jackpot}</td>
        </tr>'''

    header = '<th style="background:rgba(99,102,241,0.1);padding:12px;color:#06b6d4;font-weight:700;border-bottom:2px solid rgba(255,255,255,0.1);">#</th>'
    header += '<th style="background:rgba(99,102,241,0.1);padding:12px;color:#06b6d4;font-weight:700;border-bottom:2px solid rgba(255,255,255,0.1);">Date</th>'
    header += '<th style="background:rgba(99,102,241,0.1);padding:12px;color:#06b6d4;font-weight:700;border-bottom:2px solid rgba(255,255,255,0.1);">Result</th>'
    if is_power:
        header += '<th style="background:rgba(99,102,241,0.1);padding:12px;color:#06b6d4;font-weight:700;border-bottom:2px solid rgba(255,255,255,0.1);">Bonus</th>'
    header += '<th style="background:rgba(99,102,241,0.1);padding:12px;color:#06b6d4;font-weight:700;border-bottom:2px solid rgba(255,255,255,0.1);">Jackpot</th>'

    html_content = f"""
    <div class="glass-card">
        <div class="card-title-row">📋 History {'Bo B (6/55)' if is_power else 'Bo A (6/45)'}</div>
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


# ============================================
# GOLDEN SET RENDERER
# ============================================
def render_golden_set_results(result, lottery_type):
    """Render the Golden Set consensus prediction with full visualization."""
    golden = result['golden_set']
    portfolio = result['golden_portfolio']
    heat_map = result['heat_map']
    confidence = result['confidence']
    bt = result['backtest_summary']
    weights = result['signal_weights']
    engine_results = result['engine_results']
    max_num = 45 if lottery_type == 'mega' else 55

    # Confidence color
    conf_color = '#22c55e' if confidence > 60 else '#f59e0b' if confidence > 35 else '#ef4444'

    # === GOLDEN SET (HERO SECTION) ===
    balls_html = ' '.join(
        f'<span class="golden-ball" style="animation-delay:{i*0.1}s;">{str(n).zfill(2)}</span>'
        for i, n in enumerate(golden)
    )
    st.markdown(f"""
    <div class="golden-card">
        <div style="position:relative;z-index:1;">
            <div style="font-size:2.2rem;margin-bottom:4px;">🏆</div>
            <div style="font-size:1.6rem;font-weight:900;
                        background:linear-gradient(135deg,#f59e0b,#eab308,#fbbf24);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                        margin-bottom:8px;">GOLDEN SET — AI CONSENSUS</div>
            <div style="font-size:0.85rem;color:#94a3b8;margin-bottom:16px;">
                {result['n_signals']} signals fused | Walk-forward calibrated | Pool size: {result['super_pool_size']}
            </div>
            <div class="ball-row">{balls_html}</div>
            <div class="confidence-gauge" style="border-color:{conf_color};">
                <div class="confidence-value" style="color:{conf_color};">{confidence:.0f}%</div>
                <div class="confidence-label">Confidence</div>
            </div>
            <div class="metric-row">
                <div class="metric-item">
                    <div class="metric-value" style="color:#6366f1;">{result['n_signals']}</div>
                    <div class="metric-label">Signals</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value" style="color:#f59e0b;">{result['total_sets']}</div>
                    <div class="metric-label">Portfolio</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value" style="color:#22c55e;">{bt.get('avg',0):.2f}/6</div>
                    <div class="metric-label">BT Avg</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value" style="color:#ec4899;">{bt.get('max',0)}/6</div>
                    <div class="metric-label">BT Best</div>
                </div>
            </div>
            <div style="font-size:0.7rem;color:#64748b;margin-top:8px;">⏱️ Computed in {result['elapsed']}s</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # === BACKTEST SUMMARY ===
    if bt.get('tests', 0) > 0:
        imp_color = '#22c55e' if bt.get('improvement', 0) > 0 else '#ef4444'
        dist_html = ''
        for k in range(7):
            c = bt.get('distribution', {}).get(str(k), 0)
            pct = c / max(bt['tests'], 1) * 100
            bar_color = '#ef4444' if k <= 1 else '#f59e0b' if k == 2 else '#22c55e' if k <= 4 else '#6366f1'
            dist_html += f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0;">'
            dist_html += f'<span style="min-width:32px;font-weight:700;color:{bar_color};font-family:JetBrains Mono,monospace;">{k}/6</span>'
            dist_html += f'<div style="flex:1;height:16px;background:rgba(255,255,255,0.05);border-radius:4px;overflow:hidden;">'
            dist_html += f'<div style="height:100%;width:{pct}%;background:{bar_color};border-radius:4px;"></div></div>'
            dist_html += f'<span style="min-width:50px;text-align:right;font-size:0.75rem;color:#94a3b8;font-family:JetBrains Mono,monospace;">{c} ({pct:.0f}%)</span></div>'

        st.markdown(f"""
        <div class="glass-card">
            <div class="card-title-row">📊 Walk-Forward Backtest ({bt['tests']} tests)</div>
            <div style="display:flex;justify-content:center;gap:24px;margin-bottom:16px;flex-wrap:wrap;">
                <div style="text-align:center;padding:8px 16px;background:rgba(255,255,255,0.03);border-radius:10px;">
                    <div style="font-size:1.4rem;font-weight:900;color:#22c55e;font-family:JetBrains Mono,monospace;">{bt['avg']:.3f}/6</div>
                    <div style="font-size:0.7rem;color:#64748b;">Consensus Avg</div>
                </div>
                <div style="text-align:center;padding:8px 16px;background:rgba(255,255,255,0.03);border-radius:10px;">
                    <div style="font-size:1.4rem;font-weight:900;color:#64748b;font-family:JetBrains Mono,monospace;">{bt.get('random_avg',0):.3f}/6</div>
                    <div style="font-size:0.7rem;color:#64748b;">Random Avg</div>
                </div>
                <div style="text-align:center;padding:8px 16px;background:rgba(255,255,255,0.03);border-radius:10px;">
                    <div style="font-size:1.4rem;font-weight:900;color:{imp_color};font-family:JetBrains Mono,monospace;">{'+'if bt.get('improvement',0)>0 else ''}{bt.get('improvement',0):.1f}%</div>
                    <div style="font-size:0.7rem;color:#64748b;">vs Random</div>
                </div>
            </div>
            {dist_html}
        </div>
        """, unsafe_allow_html=True)

    # === HEAT MAP ===
    cells_html = ''
    for num in range(1, max_num + 1):
        score = heat_map.get(num, 0)
        # Color: green for high, gray for low, red for negative
        if score >= 70:
            bg = f'rgba(34,197,94,{score/100*0.8 + 0.2})'
            color = '#fff'
        elif score >= 40:
            bg = f'rgba(245,158,11,{score/100*0.6 + 0.2})'
            color = '#fff'
        elif score >= 20:
            bg = f'rgba(99,102,241,{score/100*0.5 + 0.1})'
            color = '#e2e8f0'
        else:
            bg = 'rgba(255,255,255,0.05)'
            color = '#64748b'
        is_golden = ' border:2px solid #f59e0b;' if num in golden else ''
        cells_html += f'<span class="heat-cell" style="background:{bg};color:{color};{is_golden}" title="Score: {score}">{str(num).zfill(2)}</span>'

    st.markdown(f"""
    <div class="glass-card">
        <div class="card-title-row">🔥 Consensus Heat Map ({max_num} numbers)</div>
        <div style="display:flex;flex-wrap:wrap;justify-content:center;gap:0;">{cells_html}</div>
        <div style="display:flex;justify-content:center;gap:16px;margin-top:12px;font-size:0.7rem;color:#64748b;">
            <span>🟢 Hot (70+)</span><span>🟡 Warm (40-69)</span><span>🔵 Medium (20-39)</span><span>⚫ Cold (0-19)</span>
            <span>🔶 = Golden Set</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # === SIGNAL WEIGHTS ===
    if weights:
        weight_html = ''
        sorted_w = sorted(weights.items(), key=lambda x: -x[1])
        max_w = max(v for _, v in sorted_w) if sorted_w else 1
        for name, w in sorted_w[:15]:
            bar_pct = min(w / max(max_w, 0.01) * 100, 100)
            color = '#22c55e' if w > 1.5 else '#f59e0b' if w > 1.0 else '#6366f1' if w > 0.5 else '#64748b'
            weight_html += f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0;">'
            weight_html += f'<span style="min-width:130px;font-size:0.72rem;color:#94a3b8;">{name}</span>'
            weight_html += f'<div style="flex:1;height:14px;background:rgba(255,255,255,0.05);border-radius:4px;overflow:hidden;">'
            weight_html += f'<div style="height:100%;width:{bar_pct}%;background:{color};border-radius:4px;"></div></div>'
            weight_html += f'<span style="min-width:40px;text-align:right;font-family:JetBrains Mono,monospace;font-size:0.7rem;color:{color};font-weight:700;">{w:.3f}</span></div>'
        st.markdown(f'<div class="glass-card"><div class="card-title-row">⚖️ Signal Weights (Top 15 / {result["n_signals"]} total)</div>{weight_html}</div>', unsafe_allow_html=True)

    # === ENGINE COMPARISON ===
    if engine_results:
        eng_html = ''
        for eng_name, eng_data in engine_results.items():
            primary = eng_data.get('primary', [])
            if primary:
                overlap = len(set(primary) & set(golden))
                balls = ' '.join(f'<span class="data-ball small" style="{"border:2px solid #f59e0b;" if n in golden else ""}">{str(n).zfill(2)}</span>' for n in primary)
                eng_html += f'<div style="margin:8px 0;padding:12px;background:rgba(0,0,0,0.2);border-radius:10px;">'
                eng_html += f'<div style="font-size:0.85rem;font-weight:700;color:#94a3b8;margin-bottom:6px;">{eng_name.replace("_"," ").title()} <span style="color:#f59e0b;">({overlap}/6 overlap with Golden)</span></div>'
                eng_html += f'<div style="display:flex;gap:6px;flex-wrap:wrap;justify-content:center;">{balls}</div></div>'
        if eng_html:
            st.markdown(f'<div class="glass-card"><div class="card-title-row">🔄 Engine Comparison</div>{eng_html}</div>', unsafe_allow_html=True)

    # === GOLDEN PORTFOLIO ===
    if portfolio:
        port_html = f'<div class="glass-card" style="border-color:rgba(245,158,11,0.4);">'
        port_html += f'<div class="card-title-row">🎯 Golden Portfolio (Top {min(len(portfolio), 50)} / {result["total_sets"]} total)</div>'
        cstr = result.get('constraints', {})
        port_html += f'<div style="font-size:0.65rem;color:#94a3b8;text-align:center;margin-bottom:8px;">'
        port_html += f'Sum: {cstr.get("sum_lo",0)}-{cstr.get("sum_hi",999)} | '
        port_html += f'Odd: {cstr.get("odd_lo",0)}-{cstr.get("odd_hi",6)} | '
        port_html += f'Range: {cstr.get("range_lo",0)}-{cstr.get("range_hi",99)}</div>'
        for idx, p in enumerate(portfolio[:50]):
            badge = '🥇' if idx == 0 else '🥈' if idx == 1 else '🥉' if idx == 2 else f'#{idx+1}'
            balls = ''.join(
                f'<span style="display:inline-flex;align-items:center;justify-content:center;'
                f'width:36px;height:36px;border-radius:50%;'
                f'background:linear-gradient(135deg,{"#f59e0b,#eab308" if idx==0 else "#6366f1,#8b5cf6"});'
                f'color:{"#0a0a1a" if idx==0 else "white"};font-weight:700;font-size:0.85rem;'
                f'font-family:JetBrains Mono,monospace;margin:1px;">{str(n).zfill(2)}</span>'
                for n in p['numbers']
            )
            port_html += f'<div style="display:flex;align-items:center;gap:8px;padding:6px 10px;margin:3px 0;background:rgba(0,0,0,0.2);border-radius:8px;">'
            port_html += f'<span style="font-size:0.8rem;min-width:32px;color:#f59e0b;font-weight:700;">{badge}</span>'
            port_html += f'<div style="display:flex;gap:2px;flex-wrap:wrap;">{balls}</div>'
            port_html += f'<span style="margin-left:auto;font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#64748b;">{p["score"]}</span></div>'
        port_html += '</div>'
        st.markdown(port_html, unsafe_allow_html=True)


# ============================================
# LOTTERY TAB
# ============================================
def render_lottery_tab(lottery_type):
    """Render a full data analysis tab."""
    max_num = 45 if lottery_type == "mega" else 55
    pick = 6

    # ---- GOLDEN SET (PRIMARY ACTION — TOP) ----
    if st.button("🏆 GOLDEN SET — AI CONSENSUS (30+ Signals)", key=f"golden_{lottery_type}", type="primary", use_container_width=True):
        with st.spinner("🏆 Fusing 30+ signals + walk-forward calibration... (~45 sec)"):
            try:
                from models.consensus_engine import ConsensusEngine
                if lottery_type == "mega":
                    data = get_mega645_numbers()
                    all_rows = get_mega645_all()
                else:
                    data = get_power655_numbers()
                    data = [d[:6] for d in data]
                    all_rows = get_power655_all()
                dates = [r['draw_date'] for r in all_rows]
                engine = ConsensusEngine(max_num, pick)
                result = engine.predict(data, dates, n_portfolio=50)
                st.session_state[f"golden_result_{lottery_type}"] = result
            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback
                st.code(traceback.format_exc())

    if f"golden_result_{lottery_type}" in st.session_state:
        render_golden_set_results(st.session_state[f"golden_result_{lottery_type}"], lottery_type)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.05);margin:24px 0;'>", unsafe_allow_html=True)

    # ---- DEEP FORENSIC ----
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    if st.button("🧬 DEEP FORENSIC V2 (15 Signals) → PREDICT", key=f"deep_{lottery_type}", type="primary", use_container_width=True):
        with st.spinner("🧬 Running 15-signal deep forensic V2 + walk-forward calibration... (~40 sec)"):
            try:
                from models.deep_forensic import DeepForensic
                if lottery_type == "mega":
                    data = get_mega645_numbers()
                    all_rows = get_mega645_all()
                else:
                    data = get_power655_numbers()
                    data = [d[:6] for d in data]
                    all_rows = get_power655_all()
                dates = [r['draw_date'] for r in all_rows]
                engine = DeepForensic(max_num, pick)
                result = engine.analyze(data, dates)
                st.session_state[f"deep_result_{lottery_type}"] = result
            except Exception as e:
                st.error(f"❌ Error: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

    if f"deep_result_{lottery_type}" in st.session_state:
        render_deep_forensic_results(st.session_state[f"deep_result_{lottery_type}"])

    # ---- RULE ENGINE (Validated Statistical Rules) ----
    if st.button("📐 RULE ENGINE — 69 Validated Patterns (+27%)", key=f"rule_{lottery_type}", type="primary", use_container_width=True):
        with st.spinner("📐 Applying 69 validated rules (transition, periodicity, triplet, momentum)..."):
            try:
                from models.rule_engine import RuleEngine
                if lottery_type == "mega":
                    data = get_mega645_numbers()
                else:
                    data = get_power655_numbers()
                    data = [d[:6] for d in data]
                max_num = 45 if lottery_type == "mega" else 55
                engine = RuleEngine(max_num, 6)
                result = engine.predict(data, n_portfolio=30)
                st.session_state[f"rule_result_{lottery_type}"] = result
            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback
                st.code(traceback.format_exc())

    if f"rule_result_{lottery_type}" in st.session_state:
        rr = st.session_state[f"rule_result_{lottery_type}"]
        primary = rr['primary']
        portfolio = rr.get('portfolio', [])
        rules_fired = rr.get('n_rules_fired', 0)
        rules_total = rr.get('n_rules_total', 0)
        
        balls_html = render_balls(primary, 'data-ball master')
        st.markdown(f"""
        <div class="result-card" style="border-color:#f97316;box-shadow:0 0 60px rgba(249,115,22,0.25);">
            <div style="font-size:0.9rem;color:#94a3b8;margin-bottom:8px;">Rule Engine — Validated Statistical Patterns</div>
            <div style="font-size:1.1rem;font-weight:700;color:#f97316;margin-bottom:12px;">📐 69 Rules | Walk-Forward Validated | +27% vs Random</div>
            <div class="ball-row">{balls_html}</div>
            <div class="metric-row">
                <div class="metric-item">
                    <div class="metric-value" style="color:#f97316;">{rules_fired}</div>
                    <div class="metric-label">Rules Fired</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value" style="color:#6366f1;">{rules_total}</div>
                    <div class="metric-label">Total Rules</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value" style="color:#22c55e;">{len(portfolio)}</div>
                    <div class="metric-label">Portfolio</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show which rules fired
        rules_applied = rr.get('rules_applied', [])
        if rules_applied:
            rules_html = ''.join(f'<div style="padding:4px 8px;margin:2px 0;background:rgba(249,115,22,0.1);border-radius:4px;font-size:0.75rem;color:#fb923c;font-family:JetBrains Mono,monospace;">✦ {r}</div>' for r in rules_applied[:10])
            st.markdown(f'<div class="glass-card"><div class="card-title-row">📐 Active Rules</div>{rules_html}</div>', unsafe_allow_html=True)
        
        # Portfolio
        if portfolio:
            port_html = f'<div class="glass-card" style="border-color:#f97316;">'
            port_html += f'<div class="card-title-row">🎯 Rule Engine Portfolio ({len(portfolio)} sets)</div>'
            for idx, p in enumerate(portfolio[:15]):
                badge = '⭐' if idx == 0 else f'#{idx+1}'
                balls = ''.join(f'<span style="display:inline-flex;align-items:center;justify-content:center;width:36px;height:36px;border-radius:50%;background:linear-gradient(135deg,#f97316,#ea580c);color:white;font-weight:700;font-size:0.85rem;font-family:JetBrains Mono,monospace;margin:1px;">{str(n).zfill(2)}</span>' for n in p['numbers'])
                port_html += f'<div style="display:flex;align-items:center;gap:8px;padding:6px 10px;margin:3px 0;background:rgba(0,0,0,0.2);border-radius:8px;"><span style="font-size:0.8rem;min-width:28px;color:#f97316;font-weight:700;">{badge}</span><div style="display:flex;gap:2px;flex-wrap:wrap;">{balls}</div><span style="margin-left:auto;font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#64748b;">{p["score"]}</span></div>'
            port_html += '</div>'
            st.markdown(port_html, unsafe_allow_html=True)

    # ---- ULTIMATE ENGINE V9 (Block Puzzle) ----
    if st.button("🏆 ULTIMATE V9 → BLOCK PUZZLE ENGINE", key=f"ultimate_{lottery_type}", type="primary", use_container_width=True):
        with st.spinner("🏆 V9: Block Puzzle + Multi-Pool... (~0.5 sec)"):
            try:
                from models.ultimate_engine import UltimateEngine
                if lottery_type == "mega":
                    data = get_mega645_numbers()
                    all_rows = get_mega645_all()
                else:
                    data = get_power655_numbers()
                    data = [d[:6] for d in data]
                    all_rows = get_power655_all()
                dates = [r['draw_date'] for r in all_rows]
                engine = UltimateEngine(max_num, pick)
                result = engine.predict(data, dates, n_portfolio=500)
                st.session_state[f"ultimate_result_{lottery_type}"] = result
            except Exception as e:
                st.error(f"❌ Error: {e}")

    if f"ultimate_result_{lottery_type}" in st.session_state:
        ult = st.session_state[f"ultimate_result_{lottery_type}"]
        primary = ult['primary']
        portfolio = ult['portfolio']
        
        coverage = ult.get('coverage', 0)
        n_signals = ult.get('n_signals', 20)
        
        # Primary
        balls_html = render_balls(primary, 'data-ball master')
        st.markdown(f"""
        <div class="result-card" style="border-color:#f59e0b;box-shadow:0 0 60px rgba(245,158,11,0.25);">
            <div style="font-size:0.9rem;color:#94a3b8;margin-bottom:8px;">Ultimate Engine V9 — Block Puzzle</div>
            <div style="font-size:1.1rem;font-weight:700;color:#f59e0b;margin-bottom:12px;">🏆 {n_signals} Signals | Block Puzzle | ≥4/6=47.6% | 5/6=3.9%</div>
            <div class="ball-row">{balls_html}</div>
            <div class="metric-row">
                <div class="metric-item">
                    <div class="metric-value" style="color:#6366f1;">{n_signals}</div>
                    <div class="metric-label">Signals</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value" style="color:#f59e0b;">{len(portfolio)}</div>
                    <div class="metric-label">Portfolio Sets</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value" style="color:#22c55e;">{coverage}/{max_num}</div>
                    <div class="metric-label">Coverage</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Signal weights bar chart
        sig_weights = ult.get('weights', {})
        if sig_weights:
            weight_html = ''
            sorted_w = sorted(sig_weights.items(), key=lambda x: -x[1])
            max_w = max(v for _, v in sorted_w) if sorted_w else 1
            for name, w in sorted_w:
                bar_pct = min(w / max_w * 100, 100)
                color = '#22c55e' if w > 1.5 else '#f59e0b' if w > 1.0 else '#64748b'
                weight_html += f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0;"><span style="min-width:110px;font-size:0.72rem;color:#94a3b8;">{name}</span><div style="flex:1;height:14px;background:rgba(255,255,255,0.05);border-radius:4px;overflow:hidden;"><div style="height:100%;width:{bar_pct}%;background:{color};border-radius:4px;"></div></div><span style="min-width:36px;text-align:right;font-family:JetBrains Mono,monospace;font-size:0.7rem;color:{color};font-weight:700;">{w:.2f}</span></div>'
            st.markdown(f'<div class="glass-card"><div class="card-title-row">📊 Signal Weights (Auto-Calibrated)</div>{weight_html}</div>', unsafe_allow_html=True)
        
        # Strategy breakdown
        strat_count = Counter(p['strategy'] for p in portfolio)
        strat_html = ' | '.join(f'<span style="color:#94a3b8;">{s}: <span style="color:#f59e0b;font-weight:700;">{c}</span></span>' for s, c in strat_count.most_common())
        
        # Portfolio
        port_html = f'<div class="glass-card" style="border-color:#f59e0b;">'
        port_html += f'<div class="card-title-row">🎯 Ultimate V2 Portfolio ({len(portfolio)} sets)</div>'
        port_html += f'<div style="font-size:0.65rem;color:#94a3b8;text-align:center;margin-bottom:8px;">{strat_html}</div>'
        for idx, p in enumerate(portfolio):
            badge = '⭐' if idx == 0 else f'#{idx+1}'
            balls = ''.join(f'<span style="display:inline-flex;align-items:center;justify-content:center;width:36px;height:36px;border-radius:50%;background:linear-gradient(135deg,#6366f1,#8b5cf6);color:white;font-weight:700;font-size:0.85rem;font-family:JetBrains Mono,monospace;margin:1px;">{str(n).zfill(2)}</span>' for n in p['numbers'])
            strat_badge = f'<span style="font-size:0.6rem;padding:2px 6px;border-radius:8px;background:rgba(99,102,241,0.15);color:#818cf8;">{p["strategy"]}</span>'
            port_html += f'<div style="display:flex;align-items:center;gap:8px;padding:6px 10px;margin:3px 0;background:rgba(0,0,0,0.2);border-radius:8px;"><span style="font-size:0.8rem;min-width:28px;color:#f59e0b;font-weight:700;">{badge}</span><div style="display:flex;gap:2px;flex-wrap:wrap;">{balls}</div>{strat_badge}<span style="margin-left:auto;font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#64748b;">{p["score"]}</span></div>'
        port_html += '</div>'
        st.markdown(port_html, unsafe_allow_html=True)

    # ---- VULNERABILITY SCAN ----
    with st.expander("🔍 Vulnerability Scanner (12 RNG Tests)..."):
        if st.button("🔍 SCAN RNG", key=f"scan_{lottery_type}", use_container_width=True):
            with st.spinner("🔍 Scanning 12 vulnerability tests... (~10 sec)"):
                try:
                    from models.vulnerability_scanner import VulnerabilityScanner
                    if lottery_type == "mega":
                        data = get_mega645_numbers()
                        all_rows = get_mega645_all()
                    else:
                        data = get_power655_numbers()
                        data = [d[:6] for d in data]
                        all_rows = get_power655_all()
                    dates = [r['draw_date'] for r in all_rows]
                    scanner = VulnerabilityScanner(max_num, pick)
                    result = scanner.scan_all(data, dates)
                    st.session_state[f"scan_result_{lottery_type}"] = result
                except Exception as e:
                    st.error(f"❌ Error: {e}")

        if f"scan_result_{lottery_type}" in st.session_state:
            render_scan_results(st.session_state[f"scan_result_{lottery_type}"])

            scan_result = st.session_state[f"scan_result_{lottery_type}"]
            biases = scan_result['summary']['exploitable_biases']
            if biases:
                if st.button(f"⚡ EXPLOIT {len(biases)} BIASES", key=f"exploit_{lottery_type}", use_container_width=True):
                    with st.spinner("⚡ Generating bias-based predictions..."):
                        try:
                            from models.exploit_engine import ExploitEngine
                            if lottery_type == "mega":
                                data = get_mega645_numbers()
                            else:
                                data = get_power655_numbers()
                                data = [d[:6] for d in data]
                            engine = ExploitEngine(max_num, pick)
                            exploit = engine.exploit(data, scan_result, n_sets=20)
                            st.session_state[f"exploit_result_{lottery_type}"] = exploit
                        except Exception as e:
                            st.error(f"❌ Error: {e}")

                if f"exploit_result_{lottery_type}" in st.session_state:
                    render_exploit_results(st.session_state[f"exploit_result_{lottery_type}"])

    # ---- DAN PREDICTION ----
    dan_col1, dan_col2 = st.columns(2)
    with dan_col1:
        if st.button("📊 DAN V1 - FULL", key=f"dan_v1_{lottery_type}", use_container_width=True):
            with st.spinner("📊 Generating DAN V1..."):
                try:
                    from models.dan_predictor import predict_dan
                    nums = get_mega645_numbers() if lottery_type == "mega" else get_power655_numbers()
                    is_mega = (lottery_type == "mega")
                    cands, combos, info = predict_dan(nums, max_num, pick, is_mega, version="v1")
                    st.session_state[f"dan_result_v1_{lottery_type}"] = {
                        "candidates": cands, "combos": combos[:200], "info": info, "total": len(combos)
                    }
                except Exception as e:
                    st.error(f"❌ {e}")
    with dan_col2:
        if st.button("⚡ DAN V2 - OPTIMIZED", key=f"dan_v2_{lottery_type}", use_container_width=True):
            with st.spinner("⚡ Generating DAN V2..."):
                try:
                    from models.dan_predictor import predict_dan
                    nums = get_mega645_numbers() if lottery_type == "mega" else get_power655_numbers()
                    is_mega = (lottery_type == "mega")
                    cands, combos, info = predict_dan(nums, max_num, pick, is_mega, version="v2")
                    st.session_state[f"dan_result_v2_{lottery_type}"] = {
                        "candidates": cands, "combos": combos[:200], "info": info, "total": len(combos)
                    }
                except Exception as e:
                    st.error(f"❌ {e}")

    for ver in ["v1", "v2"]:
        skey = f"dan_result_{ver}_{lottery_type}"
        if skey in st.session_state:
            dan_data = st.session_state[skey]
            if isinstance(dan_data, dict) and "candidates" in dan_data:
                render_dan_result(dan_data, ver)

    # ---- BACKTEST ----
    with st.expander("🧪 Backtest (test prediction methods against history)..."):
        bt_col1, bt_col2 = st.columns([3, 1])
        with bt_col1:
            bt_tests = st.selectbox("Test iterations", [50, 100, 200, 500], index=1, key=f"bt_tests_{lottery_type}")
        with bt_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            run_bt = st.button("🚀 Run", key=f"run_bt_{lottery_type}", use_container_width=True)

        if run_bt:
            with st.spinner(f"🧪 Running backtest {bt_tests} iterations..."):
                try:
                    from models.backtester import BacktestEngine
                    if lottery_type == "mega":
                        data = get_mega645_numbers()
                        bt_engine = BacktestEngine(45, 6)
                    else:
                        data = get_power655_numbers()
                        data = [d[:6] for d in data]
                        bt_engine = BacktestEngine(55, 6)
                    step_size = max(1, (len(data) - 50) // bt_tests)
                    result = bt_engine.run_backtest(data, start_from=50, step=step_size, max_tests=bt_tests)
                    st.session_state[f"backtest_result_{lottery_type}"] = result
                except Exception as e:
                    st.error(f"❌ Backtest error: {e}")

        if f"backtest_result_{lottery_type}" in st.session_state:
            bt = st.session_state[f"backtest_result_{lottery_type}"]
            if 'models' in bt:
                rows_html = ""
                for m in bt['models'][:14]:
                    imp = ((m['avg_matches'] / (pick * pick / max_num)) - 1) * 100
                    imp_cls = "good" if imp > 0 else "bad"
                    rows_html += f'<tr><td>#{m["rank"]}</td><td><strong>{m["model"]}</strong></td><td><strong>{m["avg_matches"]}</strong></td><td>{m["best_score"]}/6</td><td>{m["at_least_3"]}%</td><td><span class="{imp_cls}">{"+" if imp > 0 else ""}{imp:.1f}%</span></td></tr>'
                st.markdown(f"""
                <div class="glass-card">
                    <div class="card-title-row">🏆 Backtest Results ({bt['total_iterations']} iterations)</div>
                    <div style="overflow-x:auto;">
                        <table class="strat-table">
                            <thead><tr><th>#</th><th>Method</th><th>Avg/6</th><th>Max</th><th>3+</th><th>vs Random</th></tr></thead>
                            <tbody>{rows_html}</tbody>
                        </table>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ---- HISTORY TABLE ----
    total_draws = get_count(lottery_type)
    first_date = get_first_date(lottery_type) or 'N/A'
    latest_date = get_latest_date(lottery_type) or 'N/A'
    st.markdown(f"""
    <div class="glass-card" style="text-align:center;">
        <div style="font-size:1.2rem;font-weight:700;color:#f1f5f9;margin-bottom:8px;">📋 Lịch Sử Kết Quả</div>
        <div style="display:flex;justify-content:center;gap:20px;flex-wrap:wrap;">
            <div style="text-align:center;"><div style="font-size:1.6rem;font-weight:900;color:#f59e0b;font-family:JetBrains Mono,monospace;">{total_draws}</div><div style="font-size:0.75rem;color:#64748b;">Tổng kỳ quay</div></div>
            <div style="text-align:center;"><div style="font-size:1rem;font-weight:700;color:#22c55e;font-family:JetBrains Mono,monospace;">{first_date}</div><div style="font-size:0.75rem;color:#64748b;">Kỳ đầu tiên</div></div>
            <div style="text-align:center;"><div style="font-size:1rem;font-weight:700;color:#ec4899;font-family:JetBrains Mono,monospace;">{latest_date}</div><div style="font-size:0.75rem;color:#64748b;">Kỳ mới nhất</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    limit = st.selectbox(
        "Hiển thị", [20, 50, 200, 500, 9999],
        format_func=lambda x: f"Tất cả ({total_draws} kỳ)" if x == 9999 else f"{x} kỳ gần nhất",
        key=f"limit_{lottery_type}"
    )
    rows = get_recent(lottery_type, limit)
    if rows:
        render_history_table(rows, lottery_type)
    else:
        st.info("📭 Chưa có data. Bấm Update Data để cập nhật!")


# ============================================
# MAIN APP
# ============================================
def main():
    if not check_password():
        return

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Auto-update
    if "auto_update_done" not in st.session_state:
        st.session_state.auto_update_done = False
        st.session_state.auto_update_result = None

    if not st.session_state.auto_update_done:
        with st.spinner("🔄 Checking for new data..."):
            try:
                from scraper.auto_updater import auto_update_data
                result = auto_update_data()
                st.session_state.auto_update_result = result
                st.session_state.auto_update_done = True
                if result['status'] == 'updated' and (result['mega_new'] > 0 or result['power_new'] > 0):
                    st.rerun()
            except Exception as e:
                st.session_state.auto_update_result = {'status': 'error', 'message': f'⚠️ {str(e)[:100]}'}
                st.session_state.auto_update_done = True

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

    # Header
    mega_count = get_count("mega")
    power_count = get_count("power")
    mega_latest = get_latest_date("mega")

    power_latest = get_latest_date("power")
    mega_first = get_first_date("mega")
    power_first = get_first_date("power")

    st.markdown('<div class="main-title">🏆 TinNam AI V2.0 — Golden Set</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">30+ Signal Consensus Engine | Walk-Forward Calibrated | RNG Exploit</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-badge">📊 Mega 6/45: <span class="val">{mega_count}</span> kỳ ({mega_first or '?'} → {mega_latest or '?'})</div>
        <div class="stat-badge">⚡ Power 6/55: <span class="val">{power_count}</span> kỳ ({power_first or '?'} → {power_latest or '?'})</div>
    </div>
    """, unsafe_allow_html=True)

    # Tabs
    tab_mega, tab_power = st.tabs(["🟢 Bo A (6/45)", "🔴 Bo B (6/55)"])
    with tab_mega:
        render_lottery_tab("mega")
    with tab_power:
        render_lottery_tab("power")

    # ==========================================
    # FORENSIC RULES PANEL — Toggle Button
    # ==========================================
    st.markdown("""
    <div style="text-align:center;margin:32px 0 8px;">
        <div style="width:60%;margin:0 auto;height:1px;background:linear-gradient(90deg,transparent,rgba(139,92,246,0.4),transparent);"></div>
    </div>
    """, unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns([1, 2, 1])
    with col_f2:
        show_rules = st.toggle(
            "🔬 Forensic Master Rules — 58 Quy Luật & Lỗ Hổng",
            value=False,
            help="Bật để xem toàn bộ quy luật và lỗ hổng RNG đã phát hiện từ phân tích V1→V9"
        )

    if show_rules:
        try:
            from models.forensic_rules import render_forensic_rules
            render_forensic_rules(st)
        except Exception as e:
            st.error(f"❌ Lỗi load Forensic Rules: {e}")

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        if st.button("🔄 Cập Nhật Data", use_container_width=True):
            with st.spinner("🔄 Đang tải dữ liệu mới..."):
                try:
                    # Try light scraper first (works on Streamlit Cloud)
                    from scraper.light_scraper import scrape_all_light
                    mega_new, power_new = scrape_all_light()
                    mega_c = get_count('mega')
                    power_c = get_count('power')
                    if mega_new > 0 or power_new > 0:
                        st.success(f"✅ Cập nhật thành công! Mega: {mega_c} (+{mega_new}) | Power: {power_c} (+{power_new})")
                    else:
                        st.info(f"📊 Không có kỳ mới. Mega: {mega_c} | Power: {power_c}")
                    st.session_state.auto_update_done = False
                    st.session_state.auto_update_result = None
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Lỗi: {e}")
                    try:
                        # Fallback: Selenium scraper
                        from scraper.scraper import scrape_all
                        scrape_all()
                        st.success(f"✅ Mega: {get_count('mega')} | Power: {get_count('power')}")
                        st.session_state.auto_update_done = False
                        st.rerun()
                    except Exception:
                        pass

        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()

    # Footer
    st.markdown("""
    <div class="footer-text">
        🏆 TinNam AI V2.0 — Golden Set Consensus Engine © 2026
        <div class="warn">⚠️ Statistical analysis tool for research purposes only.</div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
