"""ikas — Growth Engineer Dashboard (Streamlit).

A minimal, editorial-style dashboard for AI-driven user segmentation.
Deployed on Streamlit Community Cloud.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================================
# CONFIG
# ============================================================================
st.set_page_config(
    page_title="ikas · Growth Dashboard",
    page_icon="https://ikas.com/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "ikas — AI Growth Engineer case study"},
)

ROOT = Path(__file__).resolve().parent
PREDICTIONS_CSV = ROOT / "outputs" / "user_predictions.csv"
SUMMARY_CSV = ROOT / "outputs" / "segment_summary.csv"
IMPORTANCES_CSV = ROOT / "outputs" / "feature_importances.csv"
METADATA_JSON = ROOT / "models" / "model_metadata.json"

# Design tokens — muted, editorial palette
INK = "#0f172a"          # text primary (near-black, slight blue)
INK_MUTED = "#64748b"    # text secondary
INK_FAINT = "#94a3b8"    # text tertiary / labels
LINE = "#e2e8f0"         # borders
LINE_FAINT = "#f1f5f9"   # gridlines
SURFACE = "#ffffff"
SURFACE_SOFT = "#f8fafc"
YELLOW = "#FFE600"       # ikas brand, used very sparingly

SEGMENT_COLORS = {
    "High Value":       "#059669",  # emerald-600
    "Medium Value":     "#475569",  # slate-600
    "Churn Risk":       "#dc2626",  # red-600
    "Growth Potential": "#d97706",  # amber-600
}

# Plotly defaults — re-used by every chart so they feel like one app, not seven
PLOT_FONT = dict(family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif",
                 color=INK, size=12)

def style_fig(fig, height: int = 360):
    fig.update_layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=PLOT_FONT,
        margin=dict(t=10, b=10, l=10, r=10),
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2,
                    xanchor="center", x=0.5, font=dict(size=11)),
    )
    fig.update_xaxes(gridcolor=LINE_FAINT, linecolor=LINE, zerolinecolor=LINE,
                     tickfont=dict(size=11, color=INK_MUTED), title_font=dict(size=12, color=INK_MUTED))
    fig.update_yaxes(gridcolor=LINE_FAINT, linecolor=LINE, zerolinecolor=LINE,
                     tickfont=dict(size=11, color=INK_MUTED), title_font=dict(size=12, color=INK_MUTED))
    return fig


# ============================================================================
# CSS — typography, KPI cards, tabs, sidebar polish
# ============================================================================
st.markdown(
    f"""
    <style>
    /* ----------------- Typography & base ----------------- */
    @import url('https://rsms.me/inter/inter.css');

    html, body, [class*="css"], [data-testid="stAppViewContainer"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
        font-feature-settings: 'cv11', 'ss01';
    }}

    /* Page padding */
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1320px;
    }}

    h1, h2, h3, h4 {{ color: {INK}; letter-spacing: -0.02em; }}
    h1 {{ font-size: 1.875rem; font-weight: 700; margin-bottom: 0.25rem; }}
    h2 {{ font-size: 1.25rem; font-weight: 600; margin-top: 1.5rem; }}
    h3 {{ font-size: 1rem; font-weight: 600; }}

    /* Smooth out Streamlit's default red/blue accents */
    [data-testid="stHeader"] {{ background: transparent; }}
    [data-testid="stToolbar"] {{ background: transparent; }}

    /* ----------------- KPI cards (custom) ----------------- */
    .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 8px 0 24px; }}
    .kpi {{
        background: {SURFACE};
        border: 1px solid {LINE};
        border-radius: 10px;
        padding: 1.25rem 1.25rem 1.1rem;
        border-top: 3px solid {INK};
        transition: border-color 0.2s ease;
    }}
    .kpi .label {{
        font-size: 0.7rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.08em;
        color: {INK_FAINT}; margin-bottom: 0.5rem;
    }}
    .kpi .value {{ font-size: 2rem; font-weight: 700; color: {INK}; line-height: 1; }}
    .kpi .meta {{ font-size: 0.78rem; color: {INK_MUTED}; margin-top: 0.4rem; }}
    .kpi.green   {{ border-top-color: {SEGMENT_COLORS["High Value"]}; }}
    .kpi.red     {{ border-top-color: {SEGMENT_COLORS["Churn Risk"]}; }}
    .kpi.amber   {{ border-top-color: {SEGMENT_COLORS["Growth Potential"]}; }}
    .kpi.slate   {{ border-top-color: {SEGMENT_COLORS["Medium Value"]}; }}

    /* ----------------- Section header ----------------- */
    .section-title {{
        font-size: 0.72rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.1em;
        color: {INK_FAINT}; margin: 2rem 0 0.5rem;
    }}

    /* ----------------- Tabs (subtle underline accent) ----------------- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px; border-bottom: 1px solid {LINE};
        margin-bottom: 1.25rem;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent !important;
        font-size: 14px; font-weight: 500; color: {INK_MUTED};
        padding: 10px 16px; border-radius: 6px 6px 0 0;
        transition: color 0.15s ease;
    }}
    .stTabs [data-baseweb="tab"]:hover {{ color: {INK}; }}
    .stTabs [aria-selected="true"] {{
        color: {INK} !important; font-weight: 600;
        border-bottom: 2px solid {INK} !important; margin-bottom: -1px;
    }}

    /* ----------------- Sidebar ----------------- */
    section[data-testid="stSidebar"] {{
        background: {SURFACE_SOFT}; border-right: 1px solid {LINE};
    }}
    section[data-testid="stSidebar"] h2 {{
        font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.1em; color: {INK_FAINT}; margin-top: 0.5rem;
    }}
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {{
        border-color: {LINE}; border-radius: 6px;
    }}

    /* ----------------- Buttons ----------------- */
    .stButton button, .stDownloadButton button {{
        background: {INK}; color: white; border: none;
        border-radius: 6px; padding: 0.4rem 1rem; font-weight: 500;
        font-size: 0.85rem; transition: background 0.15s ease;
    }}
    .stButton button:hover, .stDownloadButton button:hover {{ background: #1e293b; color: white; }}
    .stButton button:focus, .stDownloadButton button:focus {{ box-shadow: none; }}

    /* ----------------- DataFrame ----------------- */
    [data-testid="stDataFrame"] {{
        border: 1px solid {LINE}; border-radius: 8px; overflow: hidden;
    }}

    /* ----------------- Metric (Streamlit's default, used in Model tab) ----------------- */
    [data-testid="stMetric"] {{
        background: {SURFACE}; border: 1px solid {LINE}; border-radius: 8px;
        padding: 0.9rem 1rem;
    }}
    [data-testid="stMetricLabel"] {{
        font-size: 0.7rem !important; font-weight: 600 !important;
        text-transform: uppercase; letter-spacing: 0.08em;
        color: {INK_FAINT} !important;
    }}
    [data-testid="stMetricValue"] {{ font-size: 1.6rem !important; color: {INK} !important; }}

    /* ----------------- Misc cleanup ----------------- */
    .stAlert {{ border-radius: 8px; border: 1px solid {LINE}; }}
    hr {{ border-color: {LINE}; margin: 1.5rem 0; }}
    #MainMenu, footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# DATA LOADING
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
# HEADER
# ============================================================================
h1, h2 = st.columns([6, 1])
with h1:
    st.markdown(
        f"<h1 style='margin-bottom:4px'>Growth Dashboard</h1>"
        f"<div style='color:{INK_MUTED}; font-size:0.95rem'>"
        f"AI-driven user segmentation · Last refresh "
        f"{mtime.strftime('%b %d, %Y · %H:%M')}</div>",
        unsafe_allow_html=True,
    )
with h2:
    st.write("")
    if st.button("Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ============================================================================
# SIDEBAR — FILTERS
# ============================================================================
with st.sidebar:
    st.markdown("## Filters")

    f_segment = st.multiselect("Segment", sorted(df["segment"].unique()), placeholder="All segments")
    f_plan = st.multiselect("Plan", sorted(df["plan_type"].unique()), placeholder="All plans")
    f_country = st.multiselect("Country", sorted(df["country"].unique()), placeholder="All countries")
    f_industry = st.multiselect("Industry", sorted(df["industry"].unique()), placeholder="All industries")
    f_device = st.multiselect("Device", sorted(df["device_type"].unique()), placeholder="All devices")
    f_cluster = st.multiselect("Behavior cluster", sorted(df["behavior_cluster_name"].unique()), placeholder="All clusters")

    st.markdown("---")
    st.markdown("## About")
    st.markdown(
        f"<div style='font-size:0.82rem; color:{INK_MUTED}; line-height:1.55'>"
        "Hybrid pipeline — Random Forest, K-Means and rule-based scores "
        "drive four business segments. Full methodology in the About tab."
        "</div>",
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
mv = int((filt["segment"] == "Medium Value").sum())
cr = int((filt["segment"] == "Churn Risk").sum())
gp = int((filt["segment"] == "Growth Potential").sum())

def pct(n): return f"{(n/total*100):.0f}%" if total else "—"

def kpi(label: str, value, meta_text: str, klass: str = "") -> str:
    return (
        f'<div class="kpi {klass}">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'<div class="meta">{meta_text}</div>'
        '</div>'
    )

kpi_html = (
    '<div class="kpi-grid">'
    + kpi("Total users", f"{total}", f"of {len(df)} segmented" if total < len(df) else "all segmented")
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

# ---------------- OVERVIEW ----------------
with tab_overview:
    if filt.empty:
        st.info("No users match the current filters. Try clearing some.")
    else:
        a, b = st.columns([1, 1])

        with a:
            st.markdown('<div class="section-title">Segment distribution</div>', unsafe_allow_html=True)
            counts = filt["segment"].value_counts().reset_index()
            counts.columns = ["segment", "n"]
            pie = px.pie(counts, names="segment", values="n", color="segment",
                         color_discrete_map=SEGMENT_COLORS, hole=0.62)
            pie.update_traces(textinfo="value", textfont_size=14, textfont_color=INK,
                              marker=dict(line=dict(color=SURFACE, width=2)))
            style_fig(pie, height=340)
            st.plotly_chart(pie, use_container_width=True, config={"displayModeBar": False})

        with b:
            st.markdown('<div class="section-title">Segment by plan</div>', unsafe_allow_html=True)
            sp = filt.groupby(["plan_type", "segment"]).size().reset_index(name="n")
            bar = px.bar(sp, x="plan_type", y="n", color="segment",
                         color_discrete_map=SEGMENT_COLORS, barmode="stack")
            bar.update_traces(marker_line_width=0)
            style_fig(bar, height=340)
            bar.update_xaxes(title=None, categoryorder="array",
                             categoryarray=["Free", "Pro", "Business"])
            bar.update_yaxes(title=None)
            st.plotly_chart(bar, use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="section-title">Funnel intent — checkouts vs payment likelihood</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:0.82rem; color:{INK_MUTED}; margin-bottom:-8px'>"
            "Point size reflects total payments. Hover for user detail.</div>",
            unsafe_allow_html=True,
        )
        sca = px.scatter(filt, x="n_checkout_start", y="payment_likelihood",
                         color="segment", color_discrete_map=SEGMENT_COLORS,
                         size="n_payment_success", size_max=22,
                         hover_data={"user_id": True, "plan_type": True,
                                     "country": True, "industry": True,
                                     "n_payment_success": True,
                                     "payment_likelihood": ":.2f",
                                     "n_checkout_start": True})
        sca.update_traces(marker=dict(line=dict(width=0.5, color="rgba(255,255,255,0.8)")))
        style_fig(sca, height=420)
        sca.update_xaxes(title="Checkout starts (count)")
        sca.update_yaxes(title="Payment likelihood", tickformat=".0%", range=[0, 1])
        st.plotly_chart(sca, use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="section-title">Heatmap — payment likelihood by country & industry</div>',
                    unsafe_allow_html=True)
        pivot = filt.pivot_table(values="payment_likelihood", index="country",
                                 columns="industry", aggfunc="mean")
        if not pivot.empty:
            hm = px.imshow(pivot, color_continuous_scale=[(0, "#fef2f2"),
                                                          (0.5, "#fef3c7"),
                                                          (1, "#dcfce7")],
                           aspect="auto", text_auto=".2f", zmin=0, zmax=1,
                           labels=dict(color="Likelihood"))
            hm.update_xaxes(side="bottom", title=None, tickfont=dict(color=INK_MUTED))
            hm.update_yaxes(title=None, tickfont=dict(color=INK_MUTED))
            hm.update_traces(textfont=dict(size=11, color=INK))
            style_fig(hm, height=320)
            st.plotly_chart(hm, use_container_width=True, config={"displayModeBar": False})

# ---------------- USERS ----------------
with tab_users:
    st.markdown('<div class="section-title">Predictions</div>', unsafe_allow_html=True)
    st.markdown(
        f"<div style='color:{INK_MUTED}; font-size:0.88rem; margin-bottom:8px'>"
        f"{len(filt)} users · sortable · filterable · exportable</div>",
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
            "payment_likelihood":   st.column_config.ProgressColumn("Payment likelihood",
                                        min_value=0, max_value=1, format="%.2f", width="medium"),
            "churn_risk":           st.column_config.ProgressColumn("Churn risk",
                                        min_value=0, max_value=1, format="%.2f", width="medium"),
            "growth_potential":     st.column_config.ProgressColumn("Growth potential",
                                        min_value=0, max_value=1, format="%.2f", width="medium"),
            "n_payment_success":    st.column_config.NumberColumn("Payments", width="small"),
            "n_checkout_start":     st.column_config.NumberColumn("Checkouts", width="small"),
            "n_login":              st.column_config.NumberColumn("Logins", width="small"),
            "last_event_day":       st.column_config.NumberColumn("Last seen (d)", width="small"),
            "segment_reason":       st.column_config.TextColumn("Reason", width="large"),
        },
    )

    st.download_button(
        "Download as CSV",
        data=filt[display_cols].to_csv(index=False).encode(),
        file_name=f"ikas_users_{datetime.now():%Y%m%d_%H%M}.csv",
        mime="text/csv",
    )

    st.markdown('<div class="section-title">User detail</div>', unsafe_allow_html=True)
    sel = st.selectbox("Look up a user by ID", options=[""] + filt["user_id"].tolist(),
                       label_visibility="collapsed", placeholder="Search user ID")
    if sel:
        rec = filt[filt["user_id"] == sel].iloc[0]
        seg_color = SEGMENT_COLORS.get(rec["segment"], INK)
        st.markdown(
            f"""<div style='border:1px solid {LINE}; border-radius:10px; padding:1.25rem;
                background:{SURFACE}; border-left:4px solid {seg_color}; margin:8px 0 16px'>
            <div style='font-size:0.72rem; color:{INK_FAINT}; font-weight:600;
                letter-spacing:0.08em; text-transform:uppercase; margin-bottom:4px'>
                {rec['segment']}</div>
            <div style='font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
                color:{INK}; font-size:0.9rem; margin-bottom:8px'>{rec['user_id']}</div>
            <div style='color:{INK_MUTED}; font-size:0.85rem'>{rec['segment_reason']}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Payment likelihood", f"{rec['payment_likelihood']:.1%}")
        m2.metric("Churn risk", f"{rec['churn_risk']:.1%}")
        m3.metric("Growth potential", f"{rec['growth_potential']:.1%}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Plan", rec["plan_type"])
        m2.metric("Total payments", int(rec["n_payment_success"]))
        m3.metric("Last activity", f"{rec['last_event_day']:.1f}d ago")
        with st.expander("Full record"):
            st.json(rec.to_dict())

# ---------------- MODEL ----------------
with tab_model:
    cv = meta.get("classifier", {}).get("metrics_cv", {})
    if cv:
        st.markdown('<div class="section-title">Random Forest · 5-fold stratified CV</div>',
                    unsafe_allow_html=True)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy",  f"{cv['accuracy']:.3f}")
        m2.metric("ROC-AUC",   f"{cv['roc_auc']:.3f}")
        m3.metric("F1",        f"{cv['f1']:.3f}")
        m4.metric("Precision", f"{cv['precision']:.3f}")
        m5.metric("Recall",    f"{cv['recall']:.3f}")
        st.markdown(
            f"<div style='color:{INK_MUTED}; font-size:0.83rem; margin-top:4px'>"
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
        ibar.update_traces(marker_line_width=0)
        style_fig(ibar, height=340)
        ibar.update_xaxes(title=None, tickformat=".0%")
        ibar.update_yaxes(title=None)
        st.plotly_chart(ibar, use_container_width=True, config={"displayModeBar": False})

    clu = meta.get("clusterer", {})
    if clu:
        st.markdown('<div class="section-title">Behavioral clusters (K-Means, k=4)</div>',
                    unsafe_allow_html=True)
        cluster_df = pd.DataFrame(clu.get("centroids_original_space", []))
        if not cluster_df.empty:
            display_order = [
                "cluster_name", "n_login", "n_checkout_start", "n_payment_success",
                "n_feature_click", "n_trial_extension", "activity_trend",
                "last_event_day", "checkout_to_payment_rate",
            ]
            cluster_df = cluster_df[[c for c in display_order if c in cluster_df.columns]]
            st.dataframe(cluster_df, use_container_width=True, hide_index=True)
        sizes = clu.get("cluster_sizes", {})
        st.markdown(
            f"<div style='color:{INK_MUTED}; font-size:0.83rem; margin-top:4px'>"
            f"Silhouette @ k=4: <b>{clu.get('silhouette_k4', 0):.3f}</b> · "
            f"Sizes: {', '.join(f'{k}={v}' for k, v in sizes.items())}"
            "</div>",
            unsafe_allow_html=True,
        )

    if not summary.empty:
        st.markdown('<div class="section-title">Segment summary</div>',
                    unsafe_allow_html=True)
        summary_display = summary.copy()
        summary_display.columns = [c.replace("_", " ").title() for c in summary_display.columns]
        st.dataframe(summary_display, use_container_width=True, hide_index=True)

# ---------------- ABOUT ----------------
with tab_about:
    st.markdown(
        f"""
<div style='max-width:780px'>

### Methodology

For each user we predict three signals — **payment likelihood**, **churn risk**, **upsell potential**
— then route the user to one of four business segments via priority-ordered rules.

The pipeline is intentionally **hybrid**: a probabilistic classifier where labels exist,
unsupervised clustering for behavioral structure, and transparent rule-based scores
where ground-truth labels do not.

**1.&nbsp;&nbsp;Feature engineering**&nbsp;&nbsp;26 per-user features from a 28-day event log —
event counts, conversion ratios, recency, last-7-vs-first-7 activity trend, behavioral
patterns, and one-hot encoded profile attributes.

**2.&nbsp;&nbsp;Random Forest classifier**&nbsp;&nbsp;`is_high_payer` target (top tertile by payment
count). Stratified 5-fold CV. Payment-derived features explicitly excluded from the
feature matrix to prevent target leakage.

**3.&nbsp;&nbsp;K-Means (k=4)**&nbsp;&nbsp;eight curated behavioral features, StandardScaler-normalized.
Cluster names assigned post-hoc from centroid magnitudes.

**4.&nbsp;&nbsp;Rule-based scores**&nbsp;&nbsp;`churn_risk` weights recency, decay, trial-extension
intensity; `growth_potential` multiplies an engagement composite by a plan-type
multiplier (Business = 0 since there is no upsell target).

**5.&nbsp;&nbsp;Segmentation**&nbsp;&nbsp;priority order is
`Churn Risk → Growth Potential → High Value → Medium Value` (fallback). A high-paying
user with churn signals must be flagged for retention, not loyalty rewards.

### Critical data finding

99 of 100 users had at least one `payment_success` event. A naive *ever-paid* target is
degenerate. We reframed the target to *high payer* (top tertile by payment count) — a
balanced 41/59 split and a more business-meaningful signal.

### Honest limitations

- **N = 100** → 5-fold CV metrics have ±5pp run-to-run variance. ROC-AUC of {cv.get('roc_auc', 0):.3f}
  is modest but in line with the sample size.
- The 28-day window is short relative to typical churn cohorts. `activity_trend`
  (last-7 vs first-7) is our best proxy.
- Rule weights are hand-tuned. With labeled outcomes they could be learned.
- Synthetic-looking data — model should be revalidated on real production data.

</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Data overview</div>', unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Users",     len(df))
    d2.metric("Plans",     df["plan_type"].nunique())
    d3.metric("Countries", df["country"].nunique())
    d4.metric("Industries", df["industry"].nunique())

    st.markdown(
        f"""
<div style='color:{INK_MUTED}; font-size:0.85rem; margin-top:1.5rem; line-height:1.6'>
<b>Stack</b> — Python · pandas · scikit-learn · Plotly · Streamlit · FastAPI<br>
<b>Repo</b> — <a href='https://github.com/eensaydn/ai-growth-engineer' style='color:{INK}'>github.com/eensaydn/ai-growth-engineer</a><br>
<b>Re-train</b> — <code>python run_pipeline.py</code> (≈ 3s)<br>
<b>API</b> — <code>python api.py</code> · Swagger UI at <code>/docs</code>
</div>
""",
        unsafe_allow_html=True,
    )
