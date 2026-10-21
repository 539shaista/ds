"""
Probability Distribution Simulator
Simulate and plot Normal, Binomial, and Poisson distributions using built-in functions.
Aesthetic: Modern crisp blue-and-white theme matching the reference dashboard.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Probability Distribution Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== QUERY PARAMS SYNC ====================
if "dist" in st.query_params:
    q = str(st.query_params["dist"]).lower()
    if "norm" in q:
        st.session_state["dist_type"] = "Normal (Gaussian)"
    elif "binom" in q:
        st.session_state["dist_type"] = "Binomial"
    elif "poiss" in q:
        st.session_state["dist_type"] = "Poisson"

if "nav" in st.query_params:
    q_nav = str(st.query_params["nav"]).lower()
    if "comp" in q_nav:
        st.session_state["nav_mode"] = "Compare All Distributions"
    elif "sim" in q_nav:
        st.session_state["nav_mode"] = "Distribution Simulator"

# ==================== SESSION STATE DEFAULTS ====================
if "nav_mode" not in st.session_state:
    st.session_state["nav_mode"] = "Distribution Simulator"
if "dist_type" not in st.session_state:
    st.session_state["dist_type"] = "Normal (Gaussian)"
if "mu" not in st.session_state:
    st.session_state["mu"] = 0.0
if "sigma" not in st.session_state:
    st.session_state["sigma"] = 1.0
if "n_trials" not in st.session_state:
    st.session_state["n_trials"] = 20
if "p_prob" not in st.session_state:
    st.session_state["p_prob"] = 0.50
if "lam" not in st.session_state:
    st.session_state["lam"] = 4.0
if "sample_size" not in st.session_state:
    st.session_state["sample_size"] = 5000
if "seed" not in st.session_state:
    st.session_state["seed"] = 42
if "bins" not in st.session_state:
    st.session_state["bins"] = 40
if "sim_run_count" not in st.session_state:
    st.session_state["sim_run_count"] = 0


# ==================== CUSTOM CSS (BLUE-AND-WHITE THEME) ====================
st.markdown("""
<style>
    :root {
        --primary-color: #0878F9 !important;
    }

    /* 1. Hide Streamlit Top Header & Deploy Controls cleanly */
    header[data-testid="stHeader"] {
        display: none !important;
        visibility: hidden !important;
        height: 0px !important;
    }
    #MainMenu, footer, .stDeployButton, [data-testid="stToolbar"], [data-testid="stDecoration"] {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* 2. Global Page Background - Soft Modern Blue (#EDF4FC) */
    .stApp {
        background-color: #EDF4FC !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #14265C !important;
    }
    
    /* Main block container padding */
    .block-container {
        padding-top: 1.1rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }
    
    /* 3. Sidebar - Soft Blue Background (#E2EDF9) */
    section[data-testid="stSidebar"] {
        width: 310px !important;
        min-width: 310px !important;
        background-color: #E2EDF9 !important;
        border-right: 1px solid #D8E5F3 !important;
    }
    
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.1rem !important;
        padding-left: 1.1rem !important;
        padding-right: 1.1rem !important;
    }
    
    /* Sidebar Section Headings */
    .sidebar-heading {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.84rem;
        font-weight: 700;
        color: #14265C !important;
        margin-top: 0.85rem;
        margin-bottom: 0.35rem;
    }
    
    /* Number Inputs in Sidebar */
    div[data-testid="stNumberInput"] {
        margin-bottom: 0.35rem;
    }
    div[data-testid="stNumberInput"] label {
        font-size: 0.79rem !important;
        font-weight: 600 !important;
        color: #405778 !important;
        margin-bottom: 0.15rem !important;
    }
    div[data-testid="stNumberInput"] input {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: #14265C !important;
        background-color: #FFFFFF !important;
        border: 1px solid #D8E5F3 !important;
        border-radius: 6px !important;
    }
    
    /* 4. Action Buttons (Both Primary & Secondary) */
    /* Primary Action: Bright Blue (#0878F9) with hover (#0666D6) */
    button[kind="primary"],
    button[data-testid="baseButton-primary"],
    .stButton > button[kind="primary"],
    div[data-testid="stSidebar"] div.stButton > button[kind="primary"],
    div.stButton > button[data-testid="baseButton-primary"] {
        background-color: #0878F9 !important;
        background: #0878F9 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        border: 1px solid #0878F9 !important;
        border-radius: 8px !important;
        padding: 0.55rem 1rem !important;
        box-shadow: 0 1px 3px rgba(8, 120, 249, 0.25) !important;
        transition: all 0.15s ease !important;
    }
    button[kind="primary"]:hover,
    button[data-testid="baseButton-primary"]:hover,
    .stButton > button[kind="primary"]:hover,
    div[data-testid="stSidebar"] div.stButton > button[kind="primary"]:hover,
    div.stButton > button[data-testid="baseButton-primary"]:hover {
        background-color: #0666D6 !important;
        background: #0666D6 !important;
        border-color: #0666D6 !important;
        color: #FFFFFF !important;
        box-shadow: 0 2px 6px rgba(8, 120, 249, 0.35) !important;
    }
    button[kind="primary"]:focus,
    button[data-testid="baseButton-primary"]:focus,
    button[kind="primary"]:active,
    button[data-testid="baseButton-primary"]:active {
        background-color: #0666D6 !important;
        background: #0666D6 !important;
        border-color: #0666D6 !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 0 2px rgba(8, 120, 249, 0.3) !important;
    }
    
    /* Secondary Action: Outlined White (#FFFFFF) with subtle border (#D8E5F3) and navy text */
    button[kind="secondary"],
    button[data-testid="baseButton-secondary"],
    .stButton > button[kind="secondary"],
    div[data-testid="stSidebar"] div.stButton > button[kind="secondary"],
    div.stButton > button[data-testid="baseButton-secondary"] {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        color: #14265C !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        border: 1px solid #D8E5F3 !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.15s ease !important;
    }
    button[kind="secondary"]:hover,
    button[data-testid="baseButton-secondary"]:hover,
    .stButton > button[kind="secondary"]:hover,
    div[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover,
    div.stButton > button[data-testid="baseButton-secondary"]:hover {
        background-color: #EDF4FC !important;
        background: #EDF4FC !important;
        border-color: #B0C4DB !important;
        color: #14265C !important;
    }

    /* 5. Main Header Layout - Natural & Spacious without About Card */
    .header-box {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 0.95rem;
    }
    
    .header-title-text {
        font-size: 1.58rem;
        font-weight: 800;
        color: #14265C !important;
        letter-spacing: -0.4px;
        margin: 0;
        line-height: 1.2;
    }
    
    .header-subtitle-text {
        font-size: 0.85rem;
        color: #405778 !important;
        margin-top: 3px;
        margin-bottom: 0;
    }
    
    /* 6. Distribution Selection Cards - Clean Modern Style */
    .dist-card-item {
        background: #FFFFFF !important;
        border: 1px solid #D8E5F3 !important;
        border-radius: 12px;
        padding: 12px 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        transition: all 0.15s ease;
        text-decoration: none !important;
        cursor: pointer;
        min-height: 72px;
    }
    .dist-card-item:hover {
        border-color: #0878F9 !important;
        transform: translateY(-1px);
        box-shadow: 0 2px 6px rgba(8, 120, 249, 0.12);
    }
    
    /* Active Cards - Specific per distribution */
    .dist-card-active-norm {
        background: #EAF3FF !important;
        border: 2px solid #0878F9 !important;
        box-shadow: 0 2px 6px rgba(8, 120, 249, 0.12) !important;
    }
    .dist-card-active-binom {
        background: #ECFDF5 !important;
        border: 2px solid #10B981 !important;
        box-shadow: 0 2px 6px rgba(16, 185, 129, 0.12) !important;
    }
    .dist-card-active-poiss {
        background: #F5F3FF !important;
        border: 2px solid #8B5CF6 !important;
        box-shadow: 0 2px 6px rgba(139, 92, 246, 0.12) !important;
    }
    
    .dist-card-heading {
        font-size: 0.94rem;
        font-weight: 700;
        color: #14265C !important;
        margin: 0;
        line-height: 1.2;
    }
    
    .dist-card-sub {
        font-size: 0.74rem;
        color: #405778;
        margin: 2px 0 0 0;
    }
    
    .dist-card-tag {
        font-size: 0.74rem;
        color: #405778;
        margin: 0;
    }
    
    /* 7. Statistical Summary Metric Cards - Consistent Height & Alignment */
    .stat-card-container {
        background: #FFFFFF !important;
        border: 1px solid #D8E5F3 !important;
        border-radius: 10px;
        padding: 10px 14px;
        display: flex;
        align-items: center;
        gap: 12px;
        min-height: 76px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    
    .stat-circle-icon {
        width: 38px;
        height: 38px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.95rem;
        font-weight: 700;
        flex-shrink: 0;
    }
    
    .stat-text-col {
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .stat-text-label {
        font-size: 0.73rem;
        font-weight: 600;
        color: #405778;
        margin-bottom: 2px;
        line-height: 1.1;
    }
    
    .stat-text-val {
        font-size: 1.25rem;
        font-weight: 800;
        color: #14265C !important;
        line-height: 1.15;
        letter-spacing: -0.3px;
    }
    
    .delta-pill-red {
        font-size: 0.69rem;
        font-weight: 700;
        color: #E53935;
        margin-top: 2px;
        line-height: 1;
    }
    
    .delta-pill-green {
        font-size: 0.69rem;
        font-weight: 700;
        color: #10B981;
        margin-top: 2px;
        line-height: 1;
    }
    
    /* 8. Main Chart Card - Crisp White (#FFFFFF) */
    .chart-box-container {
        background: #FFFFFF !important;
        border: 1px solid #D8E5F3 !important;
        border-radius: 12px;
        padding: 14px 18px 10px 18px;
        margin-top: 0.85rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    
    .chart-box-head {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 1.02rem;
        font-weight: 800;
        color: #14265C !important;
    }
    
    /* Segmented Control in chart card - Blue Active, White Inactive */
    div[data-testid="stSegmentedControl"] {
        background-color: transparent !important;
        border: none !important;
        padding: 0 !important;
        gap: 6px !important;
    }
    div[data-testid="stSegmentedControl"] button {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        color: #405778 !important;
        border: 1px solid #D8E5F3 !important;
        border-radius: 6px !important;
        font-size: 0.76rem !important;
        font-weight: 600 !important;
        padding: 4px 12px !important;
        white-space: nowrap !important;
        transition: all 0.15s ease !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-checked="true"] {
        background-color: #0878F9 !important;
        background: #0878F9 !important;
        color: #FFFFFF !important;
        border: 1px solid #0878F9 !important;
        font-weight: 700 !important;
        box-shadow: 0 1px 3px rgba(8, 120, 249, 0.25) !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-checked="true"]:hover {
        background-color: #0666D6 !important;
        background: #0666D6 !important;
        border-color: #0666D6 !important;
        color: #FFFFFF !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-checked="false"]:hover {
        background-color: #EDF4FC !important;
        border-color: #B0C4DB !important;
        color: #14265C !important;
    }

    /* 9. Bottom Info Cards - Crisp White (#FFFFFF) */
    .bottom-info-box {
        background: #FFFFFF !important;
        border: 1px solid #D8E5F3 !important;
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        height: 100%;
    }
    
    .bottom-info-title {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.95rem;
        font-weight: 800;
        color: #14265C !important;
        margin-bottom: 8px;
    }
    
    .badge-latex {
        background: #EAF3FF !important;
        color: #0878F9 !important;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 6px;
        border: 1px solid #CBE0F8 !important;
    }
    
    .prop-row {
        display: flex;
        align-items: flex-start;
        gap: 8px;
        font-size: 0.81rem;
        color: #405778 !important;
        margin-bottom: 5px;
        line-height: 1.35;
    }
    .prop-dot {
        color: #0878F9 !important;
        font-size: 1rem;
        line-height: 1;
        margin-top: -1px;
    }
</style>
""", unsafe_allow_html=True)


# ==================== SIDEBAR ====================
with st.sidebar:
    # 1. Header with App Logo
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:1.0rem; padding-top:0.2rem;">
        <svg width="26" height="26" viewBox="0 0 32 32" fill="none">
            <rect x="2" y="10" width="5.5" height="18" rx="2.5" fill="#F97316"/>
            <rect x="9.5" y="4" width="5.5" height="24" rx="2.5" fill="#0878F9"/>
            <rect x="17" y="14" width="5.5" height="14" rx="2.5" fill="#EAB308"/>
            <rect x="24.5" y="7" width="5.5" height="21" rx="2.5" fill="#10B981"/>
        </svg>
        <div style="font-size:0.95rem; font-weight:800; color:#14265C; line-height:1.15;">
            Probability<br>Distribution Simulator
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Navigation buttons (Distribution Simulator vs Compare All Distributions)
    if st.session_state["nav_mode"] == "Distribution Simulator":
        st.button("📊 Distribution Simulator", type="primary", use_container_width=True, key="side_nav_sim")
        if st.button("📊 Compare All Distributions", type="secondary", use_container_width=True, key="side_nav_comp"):
            st.session_state["nav_mode"] = "Compare All Distributions"
            st.rerun()
    else:
        if st.button("📊 Distribution Simulator", type="secondary", use_container_width=True, key="side_nav_sim"):
            st.session_state["nav_mode"] = "Distribution Simulator"
            st.rerun()
        st.button("📊 Compare All Distributions", type="primary", use_container_width=True, key="side_nav_comp")

    # 3. Parameters (directly following navigation, no gap)
    st.markdown('<div class="sidebar-heading" style="margin-top:0.9rem;">⚙️ Distribution Parameters</div>', unsafe_allow_html=True)
    if st.session_state["dist_type"] == "Normal (Gaussian)":
        st.session_state["mu"] = st.number_input(
            "Mean (μ)",
            value=float(st.session_state["mu"]),
            step=0.1,
            format="%.1f",
            key="side_in_mu"
        )
        st.session_state["sigma"] = st.number_input(
            "Standard Deviation (σ)",
            value=float(max(0.01, st.session_state["sigma"])),
            min_value=0.01,
            step=0.1,
            format="%.1f",
            key="side_in_sigma"
        )
    elif st.session_state["dist_type"] == "Binomial":
        st.session_state["n_trials"] = st.number_input(
            "Number of Trials (n)",
            value=int(max(1, st.session_state["n_trials"])),
            min_value=1,
            step=1,
            key="side_in_n"
        )
        st.session_state["p_prob"] = st.number_input(
            "Success Probability (p)",
            value=float(min(1.0, max(0.0, st.session_state["p_prob"]))),
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            format="%.2f",
            key="side_in_p"
        )
    elif st.session_state["dist_type"] == "Poisson":
        st.session_state["lam"] = st.number_input(
            "Rate Parameter (λ)",
            value=float(max(0.1, st.session_state["lam"])),
            min_value=0.1,
            step=0.5,
            format="%.1f",
            key="side_in_lam"
        )

    # 4. Simulation Settings
    st.markdown('<div class="sidebar-heading" style="margin-top:0.7rem;">🎛️ Simulation Settings</div>', unsafe_allow_html=True)
    st.session_state["sample_size"] = st.number_input(
        "Sample Size (N)",
        value=int(st.session_state["sample_size"]),
        min_value=100,
        max_value=100000,
        step=500,
        key="side_in_N"
    )
    st.session_state["seed"] = st.number_input(
        "Random Seed (optional)",
        value=int(st.session_state["seed"]),
        min_value=0,
        max_value=999999,
        step=1,
        key="side_in_seed"
    )
    
    if st.session_state["dist_type"] == "Normal (Gaussian)":
        st.session_state["bins"] = st.number_input(
            "Histogram Bins",
            value=int(st.session_state["bins"]),
            min_value=5,
            max_value=150,
            step=5,
            key="side_in_bins"
        )

    # 5. Prominent Action Buttons
    st.markdown("<div style='margin-top: 1.0rem;'></div>", unsafe_allow_html=True)
    if st.button("▶ Run Simulation", type="primary", use_container_width=True, key="side_btn_run"):
        st.session_state["sim_run_count"] += 1
        st.rerun()

    if st.button("🔄 Reset", type="secondary", use_container_width=True, key="side_btn_reset"):
        st.session_state["dist_type"] = "Normal (Gaussian)"
        st.session_state["mu"] = 0.0
        st.session_state["sigma"] = 1.0
        st.session_state["n_trials"] = 20
        st.session_state["p_prob"] = 0.50
        st.session_state["lam"] = 4.0
        st.session_state["sample_size"] = 5000
        st.session_state["seed"] = 42
        st.session_state["bins"] = 40
        st.session_state["nav_mode"] = "Distribution Simulator"
        st.session_state["sim_run_count"] = 0
        st.query_params["dist"] = "Normal"
        st.rerun()


# ==================== MAIN HEADER (NO ABOUT CARD, NATURAL SPACING) ====================
st.markdown("""
<div class="header-box">
    <svg width="34" height="34" viewBox="0 0 32 32" fill="none" style="flex-shrink:0;">
        <rect x="2" y="10" width="5.5" height="18" rx="2.5" fill="#F97316"/>
        <rect x="9.5" y="4" width="5.5" height="24" rx="2.5" fill="#0878F9"/>
        <rect x="17" y="14" width="5.5" height="14" rx="2.5" fill="#EAB308"/>
        <rect x="24.5" y="7" width="5.5" height="21" rx="2.5" fill="#10B981"/>
    </svg>
    <div>
        <h1 class="header-title-text">Probability Distribution Simulator</h1>
        <p class="header-subtitle-text">Simulate and plot Normal, Binomial, and Poisson distributions using built-in functions.</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ==================== VIEW 1: DISTRIBUTION SIMULATOR ====================
if st.session_state["nav_mode"] == "Distribution Simulator":
    
    # ---------------- 1. Distribution Selection Cards (3 in 1 row) ----------------
    c_card1, c_card2, c_card3 = st.columns(3)
    current_dist = st.session_state["dist_type"]
    
    with c_card1:
        is_active = (current_dist == "Normal (Gaussian)")
        cls_name = "dist-card-item dist-card-active-norm" if is_active else "dist-card-item"
        radio_html = '<div style="width:16px; height:16px; border-radius:50%; border:2px solid #0878F9; display:flex; align-items:center; justify-content:center; flex-shrink:0;"><div style="width:8px; height:8px; border-radius:50%; background:#0878F9;"></div></div>' if is_active else '<div style="width:16px; height:16px; border-radius:50%; border:1.5px solid #B0C4DB; flex-shrink:0;"></div>'
        st.markdown(f"""
        <a href="?dist=Normal" target="_self" style="text-decoration:none; color:inherit; display:block;">
            <div class="{cls_name}">
                <div style="display:flex; align-items:center; gap:12px;">
                    <div style="background:#EAF3FF; border-radius:8px; width:44px; height:38px; display:flex; align-items:center; justify-content:center; flex-shrink:0;">
                        <svg width="34" height="26" viewBox="0 0 50 36" fill="none">
                            <path d="M 2 32 Q 18 32 25 6 Q 32 32 48 32" stroke="#0878F9" stroke-width="3" fill="none" stroke-linecap="round"/>
                        </svg>
                    </div>
                    <div>
                        <div class="dist-card-heading">Normal (Gaussian)</div>
                        <div class="dist-card-sub">Continuous distribution</div>
                        <div class="dist-card-tag">(μ, σ)</div>
                    </div>
                </div>
                {radio_html}
            </div>
        </a>
        """, unsafe_allow_html=True)

    with c_card2:
        is_active = (current_dist == "Binomial")
        cls_name = "dist-card-item dist-card-active-binom" if is_active else "dist-card-item"
        radio_html = '<div style="width:16px; height:16px; border-radius:50%; border:2px solid #10B981; display:flex; align-items:center; justify-content:center; flex-shrink:0;"><div style="width:8px; height:8px; border-radius:50%; background:#10B981;"></div></div>' if is_active else '<div style="width:16px; height:16px; border-radius:50%; border:1.5px solid #B0C4DB; flex-shrink:0;"></div>'
        st.markdown(f"""
        <a href="?dist=Binomial" target="_self" style="text-decoration:none; color:inherit; display:block;">
            <div class="{cls_name}">
                <div style="display:flex; align-items:center; gap:12px;">
                    <div style="background:#ECFDF5; border-radius:8px; width:44px; height:38px; display:flex; align-items:center; justify-content:center; flex-shrink:0;">
                        <svg width="26" height="22" viewBox="0 0 36 32" fill="none">
                            <rect x="2" y="16" width="5.5" height="14" rx="1.5" fill="#10B981"/>
                            <rect x="11" y="8" width="5.5" height="22" rx="1.5" fill="#10B981"/>
                            <rect x="20" y="2" width="5.5" height="28" rx="1.5" fill="#10B981"/>
                            <rect x="29" y="12" width="5.5" height="18" rx="1.5" fill="#10B981"/>
                        </svg>
                    </div>
                    <div>
                        <div class="dist-card-heading">Binomial</div>
                        <div class="dist-card-sub">Discrete distribution</div>
                        <div class="dist-card-tag">(n, p)</div>
                    </div>
                </div>
                {radio_html}
            </div>
        </a>
        """, unsafe_allow_html=True)

    with c_card3:
        is_active = (current_dist == "Poisson")
        cls_name = "dist-card-item dist-card-active-poiss" if is_active else "dist-card-item"
        radio_html = '<div style="width:16px; height:16px; border-radius:50%; border:2px solid #8B5CF6; display:flex; align-items:center; justify-content:center; flex-shrink:0;"><div style="width:8px; height:8px; border-radius:50%; background:#8B5CF6;"></div></div>' if is_active else '<div style="width:16px; height:16px; border-radius:50%; border:1.5px solid #B0C4DB; flex-shrink:0;"></div>'
        st.markdown(f"""
        <a href="?dist=Poisson" target="_self" style="text-decoration:none; color:inherit; display:block;">
            <div class="{cls_name}">
                <div style="display:flex; align-items:center; gap:12px;">
                    <div style="background:#F5F3FF; border-radius:8px; width:44px; height:38px; display:flex; align-items:center; justify-content:center; flex-shrink:0;">
                        <svg width="26" height="22" viewBox="0 0 36 32" fill="none">
                            <rect x="2" y="22" width="5.5" height="8" rx="1.5" fill="#8B5CF6"/>
                            <rect x="11" y="4" width="5.5" height="26" rx="1.5" fill="#8B5CF6"/>
                            <rect x="20" y="10" width="5.5" height="20" rx="1.5" fill="#8B5CF6"/>
                            <rect x="29" y="18" width="5.5" height="12" rx="1.5" fill="#8B5CF6"/>
                        </svg>
                    </div>
                    <div>
                        <div class="dist-card-heading">Poisson</div>
                        <div class="dist-card-sub">Discrete distribution</div>
                        <div class="dist-card-tag">(λ)</div>
                    </div>
                </div>
                {radio_html}
            </div>
        </a>
        """, unsafe_allow_html=True)

    # ---------------- 2. Monte-Carlo Simulation & Moments ----------------
    seed_base = int(st.session_state["seed"]) if st.session_state["seed"] is not None else 42
    effective_seed = seed_base + st.session_state["sim_run_count"]
    rng = np.random.default_rng(effective_seed)
    N = int(st.session_state["sample_size"])

    if current_dist == "Normal (Gaussian)":
        mu = float(st.session_state["mu"])
        sigma = float(st.session_state["sigma"])
        samples = rng.normal(loc=mu, scale=sigma, size=N)
        theo_mean = mu
        theo_var = sigma ** 2
        chart_title = f"Normal Distribution: μ = {mu:.1f}, σ = {sigma:.1f} (N = {N:,})"
        view_tabs = ["Histogram + Probability Density Function", "Cumulative Distribution Function", "Q-Q Plot"]
        
    elif current_dist == "Binomial":
        n = int(st.session_state["n_trials"])
        p = float(st.session_state["p_prob"])
        samples = rng.binomial(n=n, p=p, size=N)
        theo_mean = n * p
        theo_var = n * p * (1.0 - p)
        chart_title = f"Binomial Distribution: n = {n}, p = {p:.2f} (N = {N:,})"
        view_tabs = ["Histogram + Probability Mass Function", "Cumulative Distribution Function", "Box Plot"]
        
    else:  # Poisson
        lam = float(st.session_state["lam"])
        samples = rng.poisson(lam=lam, size=N)
        theo_mean = lam
        theo_var = lam
        chart_title = f"Poisson Distribution: λ = {lam:.1f} (N = {N:,})"
        view_tabs = ["Histogram + Probability Mass Function", "Cumulative Distribution Function", "Box Plot"]

    sample_mean = float(np.mean(samples))
    sample_var = float(np.var(samples, ddof=1))
    
    # Accurate error deltas
    diff_mean = sample_mean - theo_mean
    diff_var = sample_var - theo_var
    
    if abs(theo_mean) > 1e-5:
        pct_m = (diff_mean / abs(theo_mean)) * 100
        delta_m_str = f"▲ = {diff_mean:+.4f} ({pct_m:+.2f}%)"
    else:
        delta_m_str = f"▲ = {diff_mean:+.4f}"
        
    if abs(theo_var) > 1e-5:
        pct_v = (diff_var / abs(theo_var)) * 100
        delta_v_str = f"▲ = {diff_var:+.4f} ({pct_v:+.2f}%)"
    else:
        delta_v_str = f"▲ = {diff_var:+.4f}"

    delta_m_class = "delta-pill-red" if abs(diff_mean) > 0.05 else "delta-pill-green"
    delta_v_class = "delta-pill-red" if abs(diff_var) > 0.05 else "delta-pill-green"

    # ---------------- 3. Statistical Summary Cards (4 in a row, uniform height) ----------------
    st.markdown("<div style='margin-top: 0.75rem;'></div>", unsafe_allow_html=True)
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        st.markdown(f"""
        <div class="stat-card-container">
            <div class="stat-circle-icon" style="background:#FEE2E2; color:#E53935;">μ</div>
            <div class="stat-text-col">
                <div class="stat-text-label">Theoretical Mean</div>
                <div class="stat-text-val">{theo_mean:.4f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m_col2:
        st.markdown(f"""
        <div class="stat-card-container">
            <div class="stat-circle-icon" style="background:#E0EDFF; color:#0878F9;">σ²</div>
            <div class="stat-text-col">
                <div class="stat-text-label">Theoretical Variance</div>
                <div class="stat-text-val">{theo_var:.4f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m_col3:
        st.markdown(f"""
        <div class="stat-card-container">
            <div class="stat-circle-icon" style="background:#DCFCE7; color:#10B981;">📊</div>
            <div class="stat-text-col">
                <div class="stat-text-label">Sample Mean</div>
                <div class="stat-text-val">{sample_mean:.4f}</div>
                <div class="{delta_m_class}">{delta_m_str}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m_col4:
        st.markdown(f"""
        <div class="stat-card-container">
            <div class="stat-circle-icon" style="background:#EDE9FE; color:#8B5CF6;">📊</div>
            <div class="stat-text-col">
                <div class="stat-text-label">Sample Variance</div>
                <div class="stat-text-val">{sample_var:.4f}</div>
                <div class="{delta_v_class}">{delta_v_str}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---------------- 4. Main Visualization Panel ----------------
    st.markdown('<div class="chart-box-container">', unsafe_allow_html=True)
    
    # Title & View Segmented Switcher
    head_c1, head_c2 = st.columns([1.6, 2.4])
    with head_c1:
        st.markdown(f"""
        <div class="chart-box-head">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" style="flex-shrink:0;">
                <rect x="2" y="10" width="4.5" height="12" rx="1.5" fill="#0878F9"/>
                <rect x="9.5" y="4" width="4.5" height="18" rx="1.5" fill="#0878F9"/>
                <rect x="17" y="7" width="4.5" height="15" rx="1.5" fill="#0878F9"/>
            </svg>
            <span>{chart_title}</span>
        </div>
        """, unsafe_allow_html=True)
        
    with head_c2:
        selected_tab = st.segmented_control(
            "Plot View",
            options=view_tabs,
            default=view_tabs[0],
            label_visibility="collapsed",
            key="chart_view_segmented"
        )
        if selected_tab is None or selected_tab not in view_tabs:
            selected_tab = view_tabs[0]

    # Matplotlib Figure with Crisp White Background & Cool Blue-Gray Gridlines
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    fig, ax = plt.subplots(figsize=(10.5, 4.2), dpi=140)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.grid(True, linestyle='--', color='#D8E5F3', alpha=0.85, zorder=1)

    # TAB 1: HISTOGRAM + PROBABILITY DENSITY / MASS FUNCTION
    if selected_tab in [
        "Histogram + Probability Density Function",
        "Histogram + Probability Mass Function",
        "Histogram + PDF",
        "Histogram + PMF"
    ]:
        if current_dist == "Normal (Gaussian)":
            nbins = int(st.session_state["bins"])
            count, b_edges, _ = ax.hist(
                samples, bins=nbins, density=True, alpha=0.65,
                color='#70B4FF', edgecolor='#0878F9', linewidth=1.1,
                label=f'Simulated Histogram (N = {N:,})', zorder=2
            )
            # Theoretical PDF overlay
            x_min = min(b_edges[0], mu - 4.0 * sigma)
            x_max = max(b_edges[-1], mu + 4.0 * sigma)
            x_line = np.linspace(x_min, x_max, 500)
            pdf_line = stats.norm.pdf(x_line, loc=mu, scale=sigma)
            ax.plot(x_line, pdf_line, color='#E53935', linewidth=2.3,
                    label=f'Theoretical Probability Density Function $\\mathcal{{N}}({mu:.1f}, {sigma:.1f}^2)$', zorder=3)
            # Dashed Mean line
            ax.axvline(mu, color='#14265C', linestyle='--', linewidth=1.5,
                       label=f'Mean (μ = {mu:.1f})', zorder=4)
            ax.set_xlabel('Value (x)', fontsize=10, fontweight='bold', color='#14265C')
            ax.set_ylabel('Probability Density', fontsize=10, fontweight='bold', color='#14265C')
            ax.set_xlim(x_min, x_max)
            
        elif current_dist == "Binomial":
            k_min = 0
            k_max = n
            if n > 35:
                k_min = max(0, int(theo_mean - 4.0 * np.sqrt(theo_var)))
                k_max = min(n, int(theo_mean + 4.0 * np.sqrt(theo_var)))
                
            k_vals = np.arange(k_min, k_max + 1)
            counts = np.bincount(samples, minlength=n + 1)
            emp_freq = counts[k_min:k_max + 1] / N
            theo_pmf = stats.binom.pmf(k_vals, n=n, p=p)
            
            bar_w = 0.38
            ax.bar(k_vals - bar_w/2, emp_freq, width=bar_w, alpha=0.75,
                   color='#10B981', edgecolor='#059669', linewidth=1.0,
                   label=f'Simulated Relative Freq (N = {N:,})', zorder=2)
            ax.bar(k_vals + bar_w/2, theo_pmf, width=bar_w, alpha=0.75,
                   color='#F59E0B', edgecolor='#D97706', linewidth=1.0,
                   label=f'Theoretical Probability Mass Function $B({n}, {p:.2f})$', zorder=2)
            ax.axvline(theo_mean, color='#14265C', linestyle='--', linewidth=1.5,
                       label=f'Mean (np = {theo_mean:.2f})', zorder=4)
            ax.set_xlabel('Successes (k)', fontsize=10, fontweight='bold', color='#14265C')
            ax.set_ylabel('Probability P(X = k)', fontsize=10, fontweight='bold', color='#14265C')
            ax.set_xticks(k_vals[::max(1, len(k_vals) // 15)])
            
        else:  # Poisson
            k_min = max(0, int(lam - 4.0 * np.sqrt(lam))) if lam > 20 else 0
            k_max = max(int(np.max(samples)), int(lam + 4.0 * np.sqrt(lam)))
            k_vals = np.arange(k_min, k_max + 1)
            counts = np.bincount(samples, minlength=k_max + 1)
            emp_freq = counts[k_min:k_max + 1] / N
            theo_pmf = stats.poisson.pmf(k_vals, mu=lam)
            
            bar_w = 0.38
            ax.bar(k_vals - bar_w/2, emp_freq, width=bar_w, alpha=0.75,
                   color='#8B5CF6', edgecolor='#7C3AED', linewidth=1.0,
                   label=f'Simulated Relative Freq (N = {N:,})', zorder=2)
            ax.bar(k_vals + bar_w/2, theo_pmf, width=bar_w, alpha=0.75,
                   color='#EC4899', edgecolor='#BE185D', linewidth=1.0,
                   label=f'Theoretical Probability Mass Function Poisson(λ = {lam:.1f})', zorder=2)
            ax.axvline(lam, color='#14265C', linestyle='--', linewidth=1.5,
                       label=f'Mean (λ = {lam:.1f})', zorder=4)
            ax.set_xlabel('Event Count (k)', fontsize=10, fontweight='bold', color='#14265C')
            ax.set_ylabel('Probability P(X = k)', fontsize=10, fontweight='bold', color='#14265C')
            ax.set_xticks(k_vals[::max(1, len(k_vals) // 15)])

    # TAB 2: CUMULATIVE DISTRIBUTION FUNCTION
    elif selected_tab == "Cumulative Distribution Function":
        sorted_x = np.sort(samples)
        ecdf_y = np.arange(1, N + 1) / N
        ax.step(sorted_x, ecdf_y, where='post', color='#0878F9', lw=2.2, label=f'Empirical CDF (N = {N:,})', zorder=2)
        
        if current_dist == "Normal (Gaussian)":
            x_line = np.linspace(sorted_x[0], sorted_x[-1], 400)
            theo_cdf = stats.norm.cdf(x_line, loc=mu, scale=sigma)
            ax.plot(x_line, theo_cdf, color='#E53935', linestyle='--', lw=2.2, label='Theoretical CDF', zorder=3)
            ax.set_xlabel('Value (x)', fontsize=10, fontweight='bold', color='#14265C')
        elif current_dist == "Binomial":
            k_vals = np.arange(0, n + 1)
            theo_cdf = stats.binom.cdf(k_vals, n=n, p=p)
            ax.step(k_vals, theo_cdf, where='post', color='#E53935', linestyle='--', lw=2.0, label='Theoretical CDF', zorder=3)
            ax.set_xlabel('Successes (k)', fontsize=10, fontweight='bold', color='#14265C')
        else:
            k_vals = np.arange(0, int(np.max(samples)) + 1)
            theo_cdf = stats.poisson.cdf(k_vals, mu=lam)
            ax.step(k_vals, theo_cdf, where='post', color='#E53935', linestyle='--', lw=2.0, label='Theoretical CDF', zorder=3)
            ax.set_xlabel('Event Count (k)', fontsize=10, fontweight='bold', color='#14265C')
            
        ax.set_ylabel('Cumulative Probability P(X ≤ x)', fontsize=10, fontweight='bold', color='#14265C')
        ax.set_ylim(-0.02, 1.05)

    # TAB 3: Q-Q PLOT / DIAGNOSTIC
    elif selected_tab == "Q-Q Plot":
        osm, osr = stats.probplot(samples, dist="norm")
        ax.plot(osm[0], osm[1], 'o', color='#0878F9', alpha=0.55, ms=4.5, label='Sample Quantiles', zorder=2)
        ax.plot(osm[0], osm[0] * osr[0] + osr[1], color='#E53935', lw=2.2, label=f'Theoretical Fit Line (R² = {osr[2]**2:.4f})', zorder=3)
        ax.set_xlabel('Theoretical Normal Quantiles', fontsize=10, fontweight='bold', color='#14265C')
        ax.set_ylabel('Sample Quantiles', fontsize=10, fontweight='bold', color='#14265C')

    else:  # Box plot for discrete view
        accent_color = '#10B981' if current_dist == "Binomial" else '#8B5CF6'
        accent_bg = '#ECFDF5' if current_dist == "Binomial" else '#F5F3FF'
        ax.boxplot(
            samples, orientation='horizontal', patch_artist=True,
            boxprops=dict(facecolor=accent_bg, color=accent_color, lw=1.5),
            whiskerprops=dict(color=accent_color, lw=1.5),
            capprops=dict(color=accent_color, lw=1.5),
            medianprops=dict(color='#E53935', lw=2.0),
            flierprops=dict(marker='o', markerfacecolor='#F59E0B', alpha=0.6, markersize=5)
        )
        ax.set_xlabel('Observations', fontsize=10, fontweight='bold', color='#14265C')
        ax.set_yticks([])

    # Legend & Axis Cleanliness
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(handles=handles, labels=labels, frameon=True, facecolor='#FFFFFF', edgecolor='#D8E5F3', fontsize=9, loc='upper right')
    ax.tick_params(colors='#14265C', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#D8E5F3')
        
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- 5. Bottom Cards: Formula & Key Properties ----------------
    st.markdown("<div style='margin-top: 0.95rem;'></div>", unsafe_allow_html=True)
    b_col1, b_col2 = st.columns(2)

    with b_col1:
        st.markdown("""
        <div class="bottom-info-box">
            <div class="bottom-info-title">
                <span style="display:flex; align-items:center; gap:6px;">
                    <span style="color:#0878F9;">📖</span> Distribution Formula
                </span>
                <span class="badge-latex">LaTeX</span>
            </div>
        """, unsafe_allow_html=True)

        if current_dist == "Normal (Gaussian)":
            st.latex(r"f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)")
            st.markdown("""
            <p style="font-size:0.79rem; color:#405778; line-height:1.45; margin-top:8px;">
                where <strong>μ</strong> is the mean and <strong>σ</strong> is the standard deviation.<br>
                The normal distribution is a continuous probability distribution that is symmetric and bell-shaped.
            </p>
            """, unsafe_allow_html=True)
        elif current_dist == "Binomial":
            st.latex(r"P(X = k) = \binom{n}{k} p^k (1 - p)^{n - k}")
            st.markdown("""
            <p style="font-size:0.79rem; color:#405778; line-height:1.45; margin-top:8px;">
                where <strong>n</strong> is the number of trials, <strong>p</strong> is the probability of success,
                and <strong>k</strong> is the number of successes.
            </p>
            """, unsafe_allow_html=True)
        else:  # Poisson
            st.latex(r"P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}")
            st.markdown("""
            <p style="font-size:0.79rem; color:#405778; line-height:1.45; margin-top:8px;">
                where <strong>λ</strong> is the rate parameter (expected occurrences) and <strong>k</strong>
                is the event count.
            </p>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with b_col2:
        st.markdown("""
        <div class="bottom-info-box">
            <div class="bottom-info-title">
                <span style="display:flex; align-items:center; gap:6px;">
                    <span>💡</span> Key Properties
                </span>
            </div>
        """, unsafe_allow_html=True)

        if current_dist == "Normal (Gaussian)":
            props_list = [
                "Continuous probability distribution",
                "Bell-shaped and symmetric",
                "Defined by parameters μ (mean) and σ (standard deviation)",
                "Mean = μ and Variance = σ²",
                "Total area under the curve = 1",
                "Commonly used in natural and social sciences"
            ]
        elif current_dist == "Binomial":
            props_list = [
                "Discrete probability distribution",
                "Models successes in a fixed number of independent trials",
                "Each trial has the same success probability",
                "Mean = np",
                "Variance = np(1 − p)",
                "Symmetric when p = 0.5, skewed when p ≠ 0.5"
            ]
        else:  # Poisson
            props_list = [
                "Discrete probability distribution",
                "Models event counts over a fixed interval under appropriate assumptions",
                "Defined by rate parameter λ",
                "Mean = λ",
                "Variance = λ",
                "Commonly used for modeling arrival processes and rare events"
            ]

        for p_item in props_list:
            st.markdown(f"""
            <div class="prop-row">
                <span class="prop-dot">•</span>
                <span>{p_item}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# ==================== VIEW 2: COMPARE ALL DISTRIBUTIONS ====================
else:
    st.markdown("""
    <div style="background:#FFFFFF; border:1px solid #D8E5F3; border-radius:12px; padding:16px 20px; margin-bottom:1rem; box-shadow:0 1px 3px rgba(0,0,0,0.02);">
        <h3 style="font-size:1.12rem; font-weight:800; color:#14265C; margin:0 0 3px 0;">📑 Compare All Distributions</h3>
        <p style="font-size:0.82rem; color:#405778; margin:0;">
            Side-by-side comparative visualization for Normal, Binomial, and Poisson distributions generated using current simulation settings.
        </p>
    </div>
    """, unsafe_allow_html=True)

    N_c = int(st.session_state["sample_size"])
    seed_c = int(st.session_state["seed"]) if st.session_state["seed"] is not None else 42
    rng_c = np.random.default_rng(seed_c)

    mu_val, sig_val = float(st.session_state["mu"]), float(st.session_state["sigma"])
    n_val, p_val = int(st.session_state["n_trials"]), float(st.session_state["p_prob"])
    lam_val = float(st.session_state["lam"])

    norm_data = rng_c.normal(loc=mu_val, scale=sig_val, size=N_c)
    binom_data = rng_c.binomial(n=n_val, p=p_val, size=N_c)
    poiss_data = rng_c.poisson(lam=lam_val, size=N_c)

    # 3-Panel Side-by-Side Plots with Blue Theme
    fig_comp, axes_c = plt.subplots(1, 3, figsize=(15.5, 4.3), dpi=130)
    fig_comp.patch.set_facecolor('#FFFFFF')

    # 1. Normal
    axes_c[0].set_facecolor('#FFFFFF')
    axes_c[0].grid(True, linestyle='--', color='#D8E5F3', alpha=0.85)
    _, bins_norm, _ = axes_c[0].hist(norm_data, bins=25, density=True, alpha=0.6, color='#70B4FF', edgecolor='#0878F9')
    x_grid = np.linspace(bins_norm[0], bins_norm[-1], 250)
    axes_c[0].plot(x_grid, stats.norm.pdf(x_grid, loc=mu_val, scale=sig_val), color='#E53935', lw=2.2, label=f'Probability Density Function $\\mathcal{{N}}({mu_val:.1f}, {sig_val:.1f}^2)$')
    axes_c[0].set_title(f'Normal: μ={mu_val:.1f}, σ={sig_val:.1f}', fontsize=11, fontweight='bold', color='#14265C')
    axes_c[0].set_xlabel('Value (x)', fontsize=9, fontweight='bold', color='#14265C')
    axes_c[0].set_ylabel('Probability Density', fontsize=9, fontweight='bold', color='#14265C')
    axes_c[0].legend(fontsize=8, loc='upper right', facecolor='#FFFFFF', edgecolor='#D8E5F3')

    # 2. Binomial
    axes_c[1].set_facecolor('#FFFFFF')
    axes_c[1].grid(True, linestyle='--', color='#D8E5F3', alpha=0.85, axis='y')
    kb = np.arange(0, n_val + 1)
    counts_b = np.bincount(binom_data, minlength=n_val + 1)[:n_val + 1]
    axes_c[1].bar(kb - 0.2, counts_b / N_c, width=0.4, color='#10B981', label='Sample Freq')
    axes_c[1].bar(kb + 0.2, stats.binom.pmf(kb, n=n_val, p=p_val), width=0.4, color='#F59E0B', label=f'Probability Mass Function B({n_val}, {p_val:.2f})')
    axes_c[1].set_title(f'Binomial: n={n_val}, p={p_val:.2f}', fontsize=11, fontweight='bold', color='#14265C')
    axes_c[1].set_xlabel('Successes (k)', fontsize=9, fontweight='bold', color='#14265C')
    axes_c[1].set_ylabel('Probability', fontsize=9, fontweight='bold', color='#14265C')
    axes_c[1].set_xticks(kb[::max(1, len(kb) // 10)])
    axes_c[1].legend(fontsize=8, loc='upper right', facecolor='#FFFFFF', edgecolor='#D8E5F3')

    # 3. Poisson
    axes_c[2].set_facecolor('#FFFFFF')
    axes_c[2].grid(True, linestyle='--', color='#D8E5F3', alpha=0.85, axis='y')
    max_kp = max(int(np.max(poiss_data)), int(lam_val + 4 * np.sqrt(lam_val)))
    kp = np.arange(0, max_kp + 1)
    counts_p = np.bincount(poiss_data, minlength=max_kp + 1)[:max_kp + 1]
    axes_c[2].bar(kp - 0.2, counts_p / N_c, width=0.4, color='#8954F6', label='Sample Freq')
    axes_c[2].bar(kp + 0.2, stats.poisson.pmf(kp, mu=lam_val), width=0.4, color='#EC4899', label=f'Probability Mass Function Poisson(λ={lam_val:.1f})')
    axes_c[2].set_title(f'Poisson: λ={lam_val:.1f}', fontsize=11, fontweight='bold', color='#14265C')
    axes_c[2].set_xlabel('Event Count (k)', fontsize=9, fontweight='bold', color='#14265C')
    axes_c[2].set_ylabel('Probability', fontsize=9, fontweight='bold', color='#14264F')
    axes_c[2].set_xticks(kp[::max(1, len(kp) // 10)])
    axes_c[2].legend(fontsize=8, loc='upper right', facecolor='#FFFFFF', edgecolor='#D8E5F3')

    for ax_it in axes_c:
        ax_it.tick_params(colors='#14265C', labelsize=8.5)
        for spine in ax_it.spines.values():
            spine.set_color('#D8E5F3')
            
    fig_comp.tight_layout()
    st.pyplot(fig_comp)
    plt.close(fig_comp)

    # Comparison Summary Table
    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    table_data = [
        {
            "Distribution": "Normal (Gaussian)",
            "Type": "Continuous",
            "Parameters": f"μ = {mu_val:.1f}, σ = {sig_val:.1f}",
            "Theoretical Mean": f"{mu_val:.4f}",
            "Sample Mean": f"{np.mean(norm_data):.4f}",
            "Theoretical Variance": f"{sig_val**2:.4f}",
            "Sample Variance": f"{np.var(norm_data, ddof=1):.4f}",
            "Mean Error (Δ)": f"{np.mean(norm_data) - mu_val:+.4f}"
        },
        {
            "Distribution": "Binomial",
            "Type": "Discrete",
            "Parameters": f"n = {n_val}, p = {p_val:.2f}",
            "Theoretical Mean": f"{n_val * p_val:.4f}",
            "Sample Mean": f"{np.mean(binom_data):.4f}",
            "Theoretical Variance": f"{n_val * p_val * (1 - p_val):.4f}",
            "Sample Variance": f"{np.var(binom_data, ddof=1):.4f}",
            "Mean Error (Δ)": f"{np.mean(binom_data) - n_val * p_val:+.4f}"
        },
        {
            "Distribution": "Poisson",
            "Type": "Discrete",
            "Parameters": f"λ = {lam_val:.1f}",
            "Theoretical Mean": f"{lam_val:.4f}",
            "Sample Mean": f"{np.mean(poiss_data):.4f}",
            "Theoretical Variance": f"{lam_val:.4f}",
            "Sample Variance": f"{np.var(poiss_data, ddof=1):.4f}",
            "Mean Error (Δ)": f"{np.mean(poiss_data) - lam_val:+.4f}"
        }
    ]
    st.dataframe(table_data, width='stretch', hide_index=True)
