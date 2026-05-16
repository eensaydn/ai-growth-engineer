"""ikas Growth Dashboard.

Editorial, Stripe/Linear style dashboard for AI-driven user segmentation.
Deployed on Streamlit Community Cloud.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ============================================================================
# CONFIG
# ============================================================================
st.set_page_config(
    page_title="ikas · Growth Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "ikas AI Growth Engineer case study"},
)

ROOT = Path(__file__).resolve().parent
PREDICTIONS_CSV = ROOT / "outputs" / "user_predictions.csv"
SUMMARY_CSV = ROOT / "outputs" / "segment_summary.csv"
IMPORTANCES_CSV = ROOT / "outputs" / "feature_importances.csv"
METADATA_JSON = ROOT / "models" / "model_metadata.json"

# Editorial palette. Neutrals + one ikas-yellow accent for branding.
INK = "#0a0a0a"
INK_2 = "#1f2937"
INK_MUTED = "#6b7280"
INK_FAINT = "#9ca3af"
LINE = "#e5e7eb"
LINE_FAINT = "#f3f4f6"
SURFACE = "#ffffff"
SURFACE_SOFT = "#fafafa"
YELLOW = "#FFE600"
YELLOW_SOFT = "rgba(255, 230, 0, 0.10)"

# Restrained, semantic segment palette.
SEGMENT_COLORS = {
    "High Value":       "#047857",  # emerald-700
    "Medium Value":     "#4b5563",  # gray-600
    "Churn Risk":       "#b91c1c",  # red-700
    "Growth Potential": "#b45309",  # amber-700
}

# ikas-inspired inline SVG lockup (lightning bolt + wordmark on yellow pill).
IKAS_LOGO_SVG = """
<svg viewBox="0 0 120 32" xmlns="http://www.w3.org/2000/svg" aria-label="ikas">
  <rect width="120" height="32" rx="8" fill="#FFE600"/>
  <path d="M22 6 L15 18 L19 18 L16 26 L25 14 L21 14 L24 6 Z" fill="#0a0a0a" stroke="none"/>
  <text x="38" y="22" font-family="Inter, system-ui, sans-serif"
        font-weight="800" font-size="17" fill="#0a0a0a"
        letter-spacing="-0.02em">ikas</text>
</svg>
"""

PLOT_FONT = dict(
    family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif",
    color=INK_2, size=12,
)


def style_fig(fig, height: int = 320):
    fig.update_layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=PLOT_FONT,
        margin=dict(t=8, b=8, l=8, r=8),
        height=height,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.22,
            xanchor="center", x=0.5,
            font=dict(size=11, color=INK_MUTED),
            bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(
            bgcolor=INK, bordercolor=INK,
            font=dict(family="Inter", color="white", size=12),
        ),
    )
    fig.update_xaxes(
        gridcolor=LINE_FAINT, linecolor=LINE, zerolinecolor=LINE,
        tickfont=dict(size=11, color=INK_MUTED),
        title_font=dict(size=12, color=INK_MUTED),
    )
    fig.update_yaxes(
        gridcolor=LINE_FAINT, linecolor=LINE, zerolinecolor=LINE,
        tickfont=dict(size=11, color=INK_MUTED),
        title_font=dict(size=12, color=INK_MUTED),
    )
    return fig


# ============================================================================
# CSS  (typography, layout, KPI cards, tabs, sidebar, background)
# ============================================================================
st.markdown(
    f"""
    <style>
    @import url('https://rsms.me/inter/inter.css');

    html, body, [class*="css"], [data-testid="stAppViewContainer"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
        font-feature-settings: 'cv11', 'ss01', 'liga';
        color: {INK};
    }}

    /* Subtle branded ambience: faint yellow glow at the top of the page,
       fading to clean white. Inspired by Vercel/Linear marketing pages. */
    [data-testid="stAppViewContainer"] > .main {{
        background:
            radial-gradient(ellipse 70% 40% at 50% -10%, {YELLOW_SOFT}, transparent 70%),
            {SURFACE};
    }}

    .block-container {{
        padding-top: 2.5rem;
        padding-bottom: 4rem;
        max-width: 1320px;
    }}

    /* Typography scale: 11 / 12 / 14 / 16 / 24 / 30 px. No other sizes. */
    h1, h2, h3, h4 {{ color: {INK}; letter-spacing: -0.022em; }}
    h1 {{ font-size: 1.875rem; font-weight: 700; line-height: 1.15; margin: 0; }}
    h2 {{ font-size: 1.125rem; font-weight: 600; margin: 1.75rem 0 0.5rem; }}

    /* KPI numbers: tabular-nums so columns of digits align cleanly. */
    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 14px;
        margin: 24px 0 8px;
    }}
    .kpi {{
        background: {SURFACE};
        border: 1px solid {LINE};
        border-radius: 12px;
        padding: 1.25rem 1.25rem 1.1rem;
        position: relative;
        transition: border-color 0.15s ease;
    }}
    .kpi:hover {{ border-color: #d1d5db; }}
    .kpi .label {{
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {INK_MUTED};
        margin-bottom: 0.6rem;
    }}
    .kpi .value {{
        font-size: 1.875rem;
        font-weight: 700;
        color: {INK};
        line-height: 1;
        font-variant-numeric: tabular-nums;
        letter-spacing: -0.02em;
    }}
    .kpi .meta {{
        font-size: 0.8rem;
        color: {INK_FAINT};
        margin-top: 0.55rem;
        font-variant-numeric: tabular-nums;
    }}
    .kpi.green   {{ border-top: 3px solid {SEGMENT_COLORS["High Value"]}; }}
    .kpi.red     {{ border-top: 3px solid {SEGMENT_COLORS["Churn Risk"]}; }}
    .kpi.amber   {{ border-top: 3px solid {SEGMENT_COLORS["Growth Potential"]}; }}
    .kpi.slate   {{ border-top: 3px solid {SEGMENT_COLORS["Medium Value"]}; }}

    /* Section header above each chart. */
    .section-title {{
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {INK_MUTED};
        margin: 1.75rem 0 0.6rem;
    }}
    .section-sub {{
        font-size: 0.82rem;
        color: {INK_FAINT};
        margin-top: -0.4rem;
        margin-bottom: 0.6rem;
    }}

    /* Tabs: minimal underline (Linear style). */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        border-bottom: 1px solid {LINE};
        margin-bottom: 1rem;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent !important;
        font-size: 14px;
        font-weight: 500;
        color: {INK_MUTED};
        padding: 10px 14px;
        border-radius: 6px 6px 0 0;
        transition: color 0.15s ease;
    }}
    .stTabs [data-baseweb="tab"]:hover {{ color: {INK}; }}
    .stTabs [aria-selected="true"] {{
        color: {INK} !important;
        font-weight: 600;
        border-bottom: 2px solid {INK} !important;
        margin-bottom: -1px;
    }}

    /* Sidebar: clean, no clutter. */
    section[data-testid="stSidebar"] {{
        background: {SURFACE_SOFT};
        border-right: 1px solid {LINE};
    }}
    section[data-testid="stSidebar"] .block-container {{
        padding-top: 1.5rem;
    }}
    section[data-testid="stSidebar"] .sidebar-label {{
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {INK_MUTED};
        margin: 1rem 0 0.5rem;
    }}
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {{
        border-color: {LINE};
        border-radius: 8px;
        background: {SURFACE};
    }}
    section[data-testid="stSidebar"] [data-baseweb="select"] > div:hover {{
        border-color: #d1d5db;
    }}

    /* Buttons. */
    .stButton button, .stDownloadButton button {{
        background: {INK};
        color: white;
        border: 1px solid {INK};
        border-radius: 8px;
        padding: 0.45rem 1rem;
        font-weight: 500;
        font-size: 0.85rem;
        transition: background 0.15s ease, border-color 0.15s ease;
        box-shadow: none;
    }}
    .stButton button:hover, .stDownloadButton button:hover {{
        background: {INK_2};
        color: white;
        border-color: {INK_2};
    }}
    .stButton button:focus, .stDownloadButton button:focus {{ box-shadow: none; }}

    /* DataFrame. */
    [data-testid="stDataFrame"] {{
        border: 1px solid {LINE};
        border-radius: 10px;
        overflow: hidden;
    }}

    /* Streamlit's default metric, used in Model tab. */
    [data-testid="stMetric"] {{
        background: {SURFACE};
        border: 1px solid {LINE};
        border-radius: 10px;
        padding: 1rem 1.1rem;
    }}
    [data-testid="stMetricLabel"] {{
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {INK_MUTED} !important;
    }}
    [data-testid="stMetricValue"] {{
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: {INK} !important;
        font-variant-numeric: tabular-nums;
    }}

    /* Alerts and dividers. */
    .stAlert {{ border-radius: 10px; border: 1px solid {LINE}; }}
    hr {{ border-color: {LINE}; margin: 1.5rem 0; }}

    /* Hide noise. */
    #MainMenu, footer {{ visibility: hidden; }}
    [data-testid="stToolbar"] {{ display: none; }}

    /* Logo lockup sizing. */
    .ikas-logo {{ display: inline-block; height: 28px; vertical-align: middle; }}
    .header-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        margin-bottom: 0.25rem;
    }}
    .header-meta {{
        color: {INK_FAINT};
        font-size: 0.84rem;
        font-variant-numeric: tabular-nums;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# DATA
# ============================================================================
@st.cache_data(ttl=60)
def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_CSV.exists():
        st.error("Predictions not found. Run `python run_pipeline.py` first.")
        st.stop()
    return pd.read_csv(PREDICTIONS_CSV)


@st.cache_data(ttl=60)
def load_summary() -> pd.DataFrame:
    return pd.read_csv(SUMMARY_CSV) if SUMMARY_CSV.exists() else pd.DataFrame()


@st.cache_data(ttl=60)
def load_importances() -> pd.DataFrame:
    return pd.read_csv(IMPORTANCES_CSV) if IMPORTANCES_CSV.exists() else pd.DataFrame()


@st.cache_data(ttl=60)
def load_metadata() -> dict:
    return json.loads(METADATA_JSON.read_text()) if METADATA_JSON.exists() else {}


df = load_predictions()
summary = load_summary()
importances = load_importances()
meta = load_metadata()
mtime = datetime.fromtimestamp(PREDICTIONS_CSV.stat().st_mtime)


# ============================================================================
# HEADER  (ikas logo, title, refresh)
# ============================================================================
head_left, head_right = st.columns([6, 1])
with head_left:
    st.markdown(
        f"""
        <div class="header-row">
            <div style="display:flex; align-items:center; gap:14px;">
                <div class="ikas-logo">{IKAS_LOGO_SVG}</div>
                <h1>Growth Dashboard</h1>
            </div>
        </div>
        <div class="header-meta">
            AI-driven user segmentation. Last refresh {mtime.strftime("%b %d, %Y at %H:%M")}.
        </div>
        """,
        unsafe_allow_html=True,
    )
with head_right:
    st.write("")
    if st.button("Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ============================================================================
# SIDEBAR  (logo lockup, then filters only)
# ============================================================================
with st.sidebar:
    st.markdown(
        f'<div style="margin: 0.25rem 0 1.5rem 0;">{IKAS_LOGO_SVG}</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-label">Segment</div>', unsafe_allow_html=True)
    f_segment = st.multiselect("seg", sorted(df["segment"].unique()),
                               placeholder="All segments", label_visibility="collapsed")
    st.markdown('<div class="sidebar-label">Plan</div>', unsafe_allow_html=True)
    f_plan = st.multiselect("plan", sorted(df["plan_type"].unique()),
                            placeholder="All plans", label_visibility="collapsed")
    st.markdown('<div class="sidebar-label">Country</div>', unsafe_allow_html=True)
    f_country = st.multiselect("cnt", sorted(df["country"].unique()),
                               placeholder="All countries", label_visibility="collapsed")
    st.markdown('<div class="sidebar-label">Industry</div>', unsafe_allow_html=True)
    f_industry = st.multiselect("ind", sorted(df["industry"].unique()),
                                placeholder="All industries", label_visibility="collapsed")
    st.markdown('<div class="sidebar-label">Device</div>', unsafe_allow_html=True)
    f_device = st.multiselect("dev", sorted(df["device_type"].unique()),
                              placeholder="All devices", label_visibility="collapsed")
    st.markdown('<div class="sidebar-label">Behavior cluster</div>', unsafe_allow_html=True)
    f_cluster = st.multiselect("clu", sorted(df["behavior_cluster_name"].unique()),
                               placeholder="All clusters", label_visibility="collapsed")

    st.markdown(
        f"""
        <div style="margin-top: 2rem; padding-top: 1rem; border-top: 1px solid {LINE};
                    font-size: 0.78rem; color: {INK_FAINT};">
            <a href="https://github.com/eensaydn/ai-growth-engineer"
               style="color: {INK_MUTED}; text-decoration: none;">
                Source on GitHub ↗
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

filt = df.copy()
if f_segment:  filt = filt[filt["segment"].isin(f_segment)]
if f_plan:     filt = filt[filt["plan_type"].isin(f_plan)]
if f_country:  filt = filt[filt["country"].isin(f_country)]
if f_industry: filt = filt[filt["industry"].isin(f_industry)]
if f_device:   filt = filt[filt["device_type"].isin(f_device)]
if f_cluster:  filt = filt[filt["behavior_cluster_name"].isin(f_cluster)]


# ============================================================================
# KPI CARDS
# ============================================================================
total = len(filt)
hv = int((filt["segment"] == "High Value").sum())
cr = int((filt["segment"] == "Churn Risk").sum())
gp = int((filt["segment"] == "Growth Potential").sum())


def pct(n):
    return f"{(n / total * 100):.0f}%" if total else "0%"


def kpi(label, value, meta_text, klass=""):
    return (
        f'<div class="kpi {klass}">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'<div class="meta">{meta_text}</div>'
        '</div>'
    )


kpi_html = (
    '<div class="kpi-grid">'
    + kpi("Total users", f"{total}",
          f"of {len(df)} segmented" if total < len(df) else "all segmented")
    + kpi("High Value",       f"{hv}", f"{pct(hv)} of filtered", "green")
    + kpi("Churn Risk",       f"{cr}", f"{pct(cr)} of filtered", "red")
    + kpi("Growth Potential", f"{gp}", f"{pct(gp)} of filtered", "amber")
    + '</div>'
)
st.markdown(kpi_html, unsafe_allow_html=True)


# ============================================================================
# TABS
# ============================================================================
tab_overview, tab_users, tab_model, tab_about = st.tabs(
    ["Overview", "Users", "Model", "About"]
)

# ---------- OVERVIEW ----------
with tab_overview:
    if filt.empty:
        st.info("No users match the current filters.")
    else:
        a, b = st.columns([1, 1])

        with a:
            st.markdown('<div class="section-title">Segment distribution</div>',
                        unsafe_allow_html=True)
            counts = filt["segment"].value_counts().reset_index()
            counts.columns = ["segment", "n"]
            pie = px.pie(counts, names="segment", values="n", color="segment",
                         color_discrete_map=SEGMENT_COLORS, hole=0.66)
            pie.update_traces(
                textinfo="value", textfont_size=15, textfont_color="white",
                marker=dict(line=dict(color=SURFACE, width=3)),
                hovertemplate="<b>%{label}</b><br>%{value} users (%{percent})<extra></extra>",
            )
            style_fig(pie, height=320)
            st.plotly_chart(pie, use_container_width=True,
                            config={"displayModeBar": False})

        with b:
            st.markdown('<div class="section-title">Segments by plan</div>',
                        unsafe_allow_html=True)
            sp = filt.groupby(["plan_type", "segment"]).size().reset_index(name="n")
            bar = px.bar(sp, x="plan_type", y="n", color="segment",
                         color_discrete_map=SEGMENT_COLORS, barmode="stack")
            bar.update_traces(marker_line_width=0,
                              hovertemplate="<b>%{x}</b> &middot; %{fullData.name}<br>%{y} users<extra></extra>")
            style_fig(bar, height=320)
            bar.update_xaxes(title=None, categoryorder="array",
                             categoryarray=["Free", "Pro", "Business"])
            bar.update_yaxes(title=None)
            st.plotly_chart(bar, use_container_width=True,
                            config={"displayModeBar": False})

        st.markdown('<div class="section-title">Checkouts vs payment likelihood</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="section-sub">Point size reflects total payments. '
            f'Hover for user detail.</div>',
            unsafe_allow_html=True,
        )
        sca = px.scatter(
            filt, x="n_checkout_start", y="payment_likelihood",
            color="segment", color_discrete_map=SEGMENT_COLORS,
            size="n_payment_success", size_max=22,
            hover_data={
                "user_id": True, "plan_type": True,
                "country": True, "industry": True,
                "n_payment_success": True,
                "payment_likelihood": ":.2f",
                "n_checkout_start": True,
            },
        )
        sca.update_traces(marker=dict(line=dict(width=0.5,
                                                color="rgba(255,255,255,0.85)")))
        style_fig(sca, height=400)
        sca.update_xaxes(title="Checkout starts")
        sca.update_yaxes(title="Payment likelihood",
                         tickformat=".0%", range=[0, 1])
        st.plotly_chart(sca, use_container_width=True,
                        config={"displayModeBar": False})

        st.markdown(
            '<div class="section-title">Payment likelihood by country and industry</div>',
            unsafe_allow_html=True,
        )
        pivot = filt.pivot_table(values="payment_likelihood", index="country",
                                 columns="industry", aggfunc="mean")
        if not pivot.empty:
            hm = px.imshow(
                pivot,
                color_continuous_scale=[
                    (0.0, "#fef2f2"), (0.5, "#fef3c7"), (1.0, "#dcfce7"),
                ],
                aspect="auto", text_auto=".2f", zmin=0, zmax=1,
                labels=dict(color="Likelihood"),
            )
            hm.update_xaxes(side="bottom", title=None,
                            tickfont=dict(color=INK_MUTED))
            hm.update_yaxes(title=None, tickfont=dict(color=INK_MUTED))
            hm.update_traces(textfont=dict(size=11, color=INK))
            style_fig(hm, height=320)
            hm.update_layout(coloraxis_colorbar=dict(
                tickfont=dict(color=INK_MUTED, size=11),
                outlinewidth=0, thickness=12,
            ))
            st.plotly_chart(hm, use_container_width=True,
                            config={"displayModeBar": False})


# ---------- USERS ----------
with tab_users:
    st.markdown(
        '<div class="section-title">Predictions table</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="section-sub">{len(filt)} users. Sortable. Filterable. '
        f'Exportable.</div>',
        unsafe_allow_html=True,
    )

    display_cols = [
        "user_id", "plan_type", "country", "industry", "device_type",
        "segment", "behavior_cluster_name",
        "payment_likelihood", "churn_risk", "growth_potential",
        "n_payment_success", "n_checkout_start", "n_login", "last_event_day",
        "segment_reason",
    ]
    show = filt[display_cols].copy().round({
        "payment_likelihood": 3, "churn_risk": 3,
        "growth_potential": 3, "last_event_day": 1,
    })
    show["user_id"] = show["user_id"].str[:8] + "…"

    st.dataframe(
        show, use_container_width=True, height=520, hide_index=True,
        column_config={
            "user_id":              st.column_config.TextColumn("User", width="small"),
            "plan_type":            st.column_config.TextColumn("Plan", width="small"),
            "country":              st.column_config.TextColumn("Country", width="small"),
            "industry":             st.column_config.TextColumn("Industry", width="small"),
            "device_type":          st.column_config.TextColumn("Device", width="small"),
            "segment":              st.column_config.TextColumn("Segment", width="small"),
            "behavior_cluster_name": st.column_config.TextColumn("Cluster", width="small"),
            "payment_likelihood":   st.column_config.ProgressColumn(
                "Payment likelihood", min_value=0, max_value=1,
                format="%.2f", width="medium"),
            "churn_risk":           st.column_config.ProgressColumn(
                "Churn risk", min_value=0, max_value=1,
                format="%.2f", width="medium"),
            "growth_potential":     st.column_config.ProgressColumn(
                "Growth potential", min_value=0, max_value=1,
                format="%.2f", width="medium"),
            "n_payment_success":    st.column_config.NumberColumn("Payments", width="small"),
            "n_checkout_start":     st.column_config.NumberColumn("Checkouts", width="small"),
            "n_login":              st.column_config.NumberColumn("Logins", width="small"),
            "last_event_day":       st.column_config.NumberColumn("Last seen (d)", width="small"),
            "segment_reason":       st.column_config.TextColumn("Reason", width="large"),
        },
    )

    st.download_button(
        "Download CSV",
        data=filt[display_cols].to_csv(index=False).encode(),
        file_name=f"ikas_users_{datetime.now():%Y%m%d_%H%M}.csv",
        mime="text/csv",
    )

    st.markdown('<div class="section-title">User lookup</div>',
                unsafe_allow_html=True)
    sel = st.selectbox(
        "Search user ID",
        options=[""] + filt["user_id"].tolist(),
        label_visibility="collapsed",
        placeholder="Type or pick a user ID",
    )
    if sel:
        rec = filt[filt["user_id"] == sel].iloc[0]
        seg_color = SEGMENT_COLORS.get(rec["segment"], INK)
        st.markdown(
            f"""
            <div style='border:1px solid {LINE}; border-radius:12px; padding:1.25rem;
                        background:{SURFACE}; border-left:4px solid {seg_color};
                        margin:0.75rem 0 1rem;'>
                <div style='font-size:0.7rem; color:{INK_MUTED}; font-weight:600;
                            letter-spacing:0.08em; text-transform:uppercase;
                            margin-bottom:6px;'>
                    {rec['segment']}
                </div>
                <div style='font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
                            color:{INK}; font-size:0.9rem; margin-bottom:8px;'>
                    {rec['user_id']}
                </div>
                <div style='color:{INK_MUTED}; font-size:0.85rem;'>
                    {rec['segment_reason']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Payment likelihood", f"{rec['payment_likelihood']:.1%}")
        m2.metric("Churn risk",         f"{rec['churn_risk']:.1%}")
        m3.metric("Growth potential",   f"{rec['growth_potential']:.1%}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Plan",           rec["plan_type"])
        m2.metric("Total payments", int(rec["n_payment_success"]))
        m3.metric("Last activity",  f"{rec['last_event_day']:.1f}d ago")
        with st.expander("Full record"):
            st.json(rec.to_dict())


# ---------- MODEL ----------
with tab_model:
    cv = meta.get("classifier", {}).get("metrics_cv", {})
    if cv:
        st.markdown(
            '<div class="section-title">Random Forest, 5-fold cross-validation</div>',
            unsafe_allow_html=True,
        )
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy",  f"{cv['accuracy']:.3f}")
        m2.metric("ROC-AUC",   f"{cv['roc_auc']:.3f}")
        m3.metric("F1",        f"{cv['f1']:.3f}")
        m4.metric("Precision", f"{cv['precision']:.3f}")
        m5.metric("Recall",    f"{cv['recall']:.3f}")
        st.markdown(
            f'<div class="section-sub" style="margin-top:8px;">'
            "Target: <code>is_high_payer</code> (top tertile by payment count). "
            "Payment-derived features excluded to prevent leakage."
            "</div>",
            unsafe_allow_html=True,
        )

    if not importances.empty:
        st.markdown('<div class="section-title">Top feature importances</div>',
                    unsafe_allow_html=True)
        imp = importances.head(10).sort_values("importance")
        ibar = px.bar(imp, x="importance", y="feature", orientation="h",
                      color_discrete_sequence=[INK])
        ibar.update_traces(marker_line_width=0,
                           hovertemplate="<b>%{y}</b><br>%{x:.3f}<extra></extra>")
        style_fig(ibar, height=340)
        ibar.update_xaxes(title=None, tickformat=".0%")
        ibar.update_yaxes(title=None)
        st.plotly_chart(ibar, use_container_width=True,
                        config={"displayModeBar": False})

    clu = meta.get("clusterer", {})
    if clu:
        st.markdown('<div class="section-title">Behavioral clusters (K-Means, k=4)</div>',
                    unsafe_allow_html=True)
        cluster_df = pd.DataFrame(clu.get("centroids_original_space", []))
        if not cluster_df.empty:
            order = [
                "cluster_name", "n_login", "n_checkout_start", "n_payment_success",
                "n_feature_click", "n_trial_extension", "activity_trend",
                "last_event_day", "checkout_to_payment_rate",
            ]
            cluster_df = cluster_df[[c for c in order if c in cluster_df.columns]]
            st.dataframe(cluster_df, use_container_width=True, hide_index=True)
        sizes = clu.get("cluster_sizes", {})
        st.markdown(
            f'<div class="section-sub" style="margin-top:8px;">'
            f"Silhouette @ k=4: <b>{clu.get('silhouette_k4', 0):.3f}</b> &middot; "
            f"Sizes: {', '.join(f'{k}={v}' for k, v in sizes.items())}"
            "</div>",
            unsafe_allow_html=True,
        )

    if not summary.empty:
        st.markdown('<div class="section-title">Segment summary</div>',
                    unsafe_allow_html=True)
        summary_display = summary.copy()
        summary_display.columns = [c.replace("_", " ").title()
                                   for c in summary_display.columns]
        st.dataframe(summary_display, use_container_width=True, hide_index=True)


# ---------- ABOUT ----------
with tab_about:
    auc = cv.get("roc_auc", 0) if cv else 0

    st.markdown('<div class="section-title">Methodology</div>',
                unsafe_allow_html=True)
    st.markdown(
        """
For each user the pipeline predicts three signals (**payment likelihood**,
**churn risk**, **upsell potential**) and routes the user to one of four
business segments via priority-ordered rules.

The approach is deliberately **hybrid**: a probabilistic classifier where labels
exist, unsupervised clustering for behavioral structure, and transparent
rule-based scores where ground-truth labels do not.

**1. Feature engineering.** 26 per-user features from a 28-day event log: event
counts, conversion ratios, recency, last-7-vs-first-7 activity trend, behavioral
patterns, and one-hot encoded profile attributes.

**2. Random Forest classifier.** `is_high_payer` target (top tertile by payment
count). Stratified 5-fold CV. Payment-derived features are explicitly excluded
from the feature matrix to prevent target leakage.

**3. K-Means (k=4).** Eight curated behavioral features, StandardScaler-normalized.
Cluster names assigned post-hoc from centroid magnitudes.

**4. Rule-based scores.** `churn_risk` weights recency, decay and
trial-extension intensity. `growth_potential` multiplies an engagement composite
by a plan-type multiplier (Business = 0 because there is no upsell target).

**5. Segmentation.** Priority order:
`Churn Risk → Growth Potential → High Value → Medium Value` (fallback). A
high-paying user with churn signals must be flagged for retention, not loyalty
rewards.
""",
    )

    st.markdown('<div class="section-title">Critical data finding</div>',
                unsafe_allow_html=True)
    st.markdown(
        """
99 of 100 users had at least one `payment_success` event. A naive *ever-paid*
target is degenerate. We reframed the target to *high payer* (top tertile by
payment count), yielding a balanced 41/59 split and a more business-meaningful
signal.
"""
    )

    st.markdown('<div class="section-title">Honest limitations</div>',
                unsafe_allow_html=True)
    st.markdown(
        f"""
- **N = 100.** 5-fold CV metrics have ±5pp run-to-run variance. ROC-AUC of
  {auc:.3f} is modest but in line with the sample size.
- The 28-day window is short relative to typical churn cohorts. `activity_trend`
  (last-7 vs first-7) is our best proxy.
- Rule weights are hand-tuned. With labeled outcomes they could be learned.
- Synthetic-looking data. The model should be revalidated on real production data.
"""
    )

    st.markdown('<div class="section-title">Data overview</div>',
                unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Users",     len(df))
    d2.metric("Plans",     df["plan_type"].nunique())
    d3.metric("Countries", df["country"].nunique())
    d4.metric("Industries", df["industry"].nunique())

    st.markdown(
        f"""
<div style='color:{INK_MUTED}; font-size:0.85rem; margin-top:1.5rem;
            line-height:1.7;'>
<b>Stack.</b> Python, pandas, scikit-learn, Plotly, Streamlit, FastAPI<br>
<b>Repo.</b> <a href='https://github.com/eensaydn/ai-growth-engineer'
   style='color:{INK}; text-decoration: underline;'>github.com/eensaydn/ai-growth-engineer</a><br>
<b>Re-train.</b> <code>python run_pipeline.py</code> (about 3 seconds)<br>
<b>API.</b> <code>python api.py</code>. Swagger UI at <code>/docs</code>.
</div>
""",
        unsafe_allow_html=True,
    )
