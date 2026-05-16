"""ikas — Growth Engineer Dashboard (Streamlit).

Public deployment target: Streamlit Community Cloud.
Run locally:  streamlit run streamlit_app.py
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
    page_title="ikas — Growth Engineer Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
PREDICTIONS_CSV = ROOT / "outputs" / "user_predictions.csv"
SUMMARY_CSV = ROOT / "outputs" / "segment_summary.csv"
IMPORTANCES_CSV = ROOT / "outputs" / "feature_importances.csv"
METADATA_JSON = ROOT / "models" / "model_metadata.json"

# ikas-style yellow accent
PRIMARY = "#FFE600"
SEGMENT_COLORS = {
    "High Value": "#2ECC71",
    "Medium Value": "#3498DB",
    "Churn Risk": "#E74C3C",
    "Growth Potential": "#F39C12",
}

# Custom CSS — left-border accent on KPI cards + tab spacing
st.markdown(
    f"""
    <style>
        [data-testid="stMetric"] {{
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid {PRIMARY};
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 20px; }}
        .stTabs [data-baseweb="tab"] {{ font-size: 15px; padding: 8px 16px; }}
        h1, h2, h3 {{ color: #1a1a1a; }}
        section[data-testid="stSidebar"] h2 {{ padding-top: 0; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# DATA LOADING (cached)
# ============================================================================
@st.cache_data(ttl=60)
def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_CSV.exists():
        st.error(
            "`outputs/user_predictions.csv` bulunamadı. "
            "Önce `python run_pipeline.py` çalıştırın."
        )
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
hcol1, hcol2 = st.columns([4, 1])
with hcol1:
    st.title("⚡ ikas — Growth Engineer Dashboard")
    st.caption(
        f"AI-powered user segmentation & churn/payment prediction · "
        f"Last refresh: {mtime.strftime('%Y-%m-%d %H:%M')}"
    )
with hcol2:
    st.write("")
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.divider()

# ============================================================================
# SIDEBAR — FILTERS
# ============================================================================
with st.sidebar:
    st.header("🔍 Filters")
    f_country = st.multiselect("Country", sorted(df["country"].unique()))
    f_industry = st.multiselect("Industry", sorted(df["industry"].unique()))
    f_plan = st.multiselect("Plan", sorted(df["plan_type"].unique()))
    f_device = st.multiselect("Device", sorted(df["device_type"].unique()))
    f_segment = st.multiselect("Segment", sorted(df["segment"].unique()))
    f_cluster = st.multiselect("Behavior cluster", sorted(df["behavior_cluster_name"].unique()))

    st.divider()
    st.markdown("**About**")
    st.caption(
        "Hybrid pipeline: Random Forest (payment) + K-Means (behavior) + rule-based "
        "scores (churn / growth) → 4 business segments. Full methodology in the "
        "About tab."
    )

# Apply filters
filt = df.copy()
if f_country:  filt = filt[filt["country"].isin(f_country)]
if f_industry: filt = filt[filt["industry"].isin(f_industry)]
if f_plan:     filt = filt[filt["plan_type"].isin(f_plan)]
if f_device:   filt = filt[filt["device_type"].isin(f_device)]
if f_segment:  filt = filt[filt["segment"].isin(f_segment)]
if f_cluster:  filt = filt[filt["behavior_cluster_name"].isin(f_cluster)]

# ============================================================================
# KPI ROW
# ============================================================================
total = len(filt)
hv = int((filt["segment"] == "High Value").sum())
mv = int((filt["segment"] == "Medium Value").sum())
cr = int((filt["segment"] == "Churn Risk").sum())
gp = int((filt["segment"] == "Growth Potential").sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Users", f"{total}",
          f"of {len(df)} total" if total < len(df) else None)
c2.metric("🟢 High Value", f"{hv}", f"{hv/total*100:.0f}%" if total else "—")
c3.metric("🔴 Churn Risk", f"{cr}", f"{cr/total*100:.0f}%" if total else "—")
c4.metric("🟠 Growth Potential", f"{gp}", f"{gp/total*100:.0f}%" if total else "—")

# ============================================================================
# TABS
# ============================================================================
tab_overview, tab_users, tab_model, tab_about = st.tabs([
    "📊 Overview", "👥 Users", "🤖 Model", "ℹ️ About",
])

# ---------- OVERVIEW ----------
with tab_overview:
    if filt.empty:
        st.info("Mevcut filtrelerle sonuç yok. Lütfen filtreleri sıfırlayın.")
    else:
        ca, cb = st.columns(2)
        with ca:
            st.subheader("Segment distribution")
            pie = px.pie(filt, names="segment", color="segment",
                         color_discrete_map=SEGMENT_COLORS, hole=0.45)
            pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=380)
            st.plotly_chart(pie, use_container_width=True)

        with cb:
            st.subheader("Segment × Plan type")
            sp = filt.groupby(["plan_type", "segment"]).size().reset_index(name="n")
            bar = px.bar(sp, x="plan_type", y="n", color="segment",
                         color_discrete_map=SEGMENT_COLORS, barmode="stack")
            bar.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=380,
                              xaxis_title="Plan", yaxis_title="Users")
            st.plotly_chart(bar, use_container_width=True)

        st.divider()
        st.subheader("Checkouts vs Payment likelihood (size = total payments)")
        sca = px.scatter(filt, x="n_checkout_start", y="payment_likelihood",
                         color="segment", color_discrete_map=SEGMENT_COLORS,
                         size="n_payment_success",
                         hover_data=["user_id", "plan_type", "country", "industry"])
        sca.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=420,
                          xaxis_title="Checkout starts", yaxis_title="Payment likelihood")
        st.plotly_chart(sca, use_container_width=True)

        st.divider()
        st.subheader("Average payment likelihood — country × industry")
        pivot = filt.pivot_table(
            values="payment_likelihood", index="country",
            columns="industry", aggfunc="mean",
        )
        if not pivot.empty:
            hm = px.imshow(pivot, color_continuous_scale="RdYlGn", aspect="auto",
                           text_auto=".2f", zmin=0, zmax=1,
                           labels=dict(color="Avg payment likelihood"))
            hm.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=380)
            st.plotly_chart(hm, use_container_width=True)

# ---------- USERS ----------
with tab_users:
    st.subheader(f"User predictions — {len(filt)} users")

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

    st.dataframe(
        show, use_container_width=True, height=500,
        column_config={
            "payment_likelihood": st.column_config.ProgressColumn(
                "Payment likelihood", min_value=0, max_value=1, format="%.2f"),
            "churn_risk": st.column_config.ProgressColumn(
                "Churn risk", min_value=0, max_value=1, format="%.2f"),
            "growth_potential": st.column_config.ProgressColumn(
                "Growth potential", min_value=0, max_value=1, format="%.2f"),
        },
    )

    st.download_button(
        "⬇️ Download filtered as CSV",
        data=show.to_csv(index=False).encode(),
        file_name=f"ikas_users_{datetime.now():%Y%m%d_%H%M}.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("User detail")
    sel = st.selectbox(
        "Select a user_id", options=[""] + filt["user_id"].tolist(),
    )
    if sel:
        rec = filt[filt["user_id"] == sel].iloc[0]
        d1, d2, d3 = st.columns(3)
        d1.metric("Segment", rec["segment"])
        d2.metric("Total payments", int(rec["n_payment_success"]))
        d3.metric("Last activity", f"{rec['last_event_day']:.1f} days ago")

        s1, s2, s3 = st.columns(3)
        s1.metric("Payment likelihood", f"{rec['payment_likelihood']:.1%}")
        s2.metric("Churn risk", f"{rec['churn_risk']:.1%}")
        s3.metric("Growth potential", f"{rec['growth_potential']:.1%}")

        st.info(f"**Segment reason**: {rec['segment_reason']}")
        with st.expander("View full record"):
            st.json(rec.to_dict())

# ---------- MODEL ----------
with tab_model:
    st.subheader("Random Forest — 5-fold cross-validation metrics")
    cv = meta.get("classifier", {}).get("metrics_cv", {})
    if cv:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{cv['accuracy']:.3f}")
        m2.metric("ROC-AUC", f"{cv['roc_auc']:.3f}")
        m3.metric("F1", f"{cv['f1']:.3f}")
        m4.metric("Precision", f"{cv['precision']:.3f}")
        m5.metric("Recall", f"{cv['recall']:.3f}")
    st.caption(
        "Target: `is_high_payer` (≥ 67th percentile by payment count). "
        "Payment-derived columns excluded to prevent leakage."
    )

    st.divider()
    st.subheader("Top 10 feature importances")
    if not importances.empty:
        imp = importances.head(10).sort_values("importance")
        bar = px.bar(imp, x="importance", y="feature", orientation="h")
        bar.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=380,
                          xaxis_title="Importance", yaxis_title="")
        st.plotly_chart(bar, use_container_width=True)

    st.divider()
    st.subheader("K-Means behavioral clusters (k=4)")
    clu = meta.get("clusterer", {})
    if clu:
        cluster_df = pd.DataFrame(clu.get("centroids_original_space", []))
        if not cluster_df.empty:
            display_order = [
                "cluster_name", "n_login", "n_checkout_start", "n_payment_success",
                "n_feature_click", "n_trial_extension", "activity_trend",
                "last_event_day", "checkout_to_payment_rate",
            ]
            cluster_df = cluster_df[[c for c in display_order if c in cluster_df.columns]]
            st.dataframe(cluster_df, use_container_width=True, hide_index=True)
        st.caption(
            f"Silhouette score @ k=4: {clu.get('silhouette_k4', 0):.3f} · "
            f"Cluster sizes: {clu.get('cluster_sizes', {})}"
        )

    st.divider()
    st.subheader("Segment summary")
    if not summary.empty:
        st.dataframe(summary, use_container_width=True, hide_index=True)

# ---------- ABOUT ----------
with tab_about:
    st.subheader("Methodology")
    st.markdown(
        """
**Goal**: For each user, predict (1) payment likelihood, (2) churn risk, (3) upsell
potential — then assign one of four business segments.

**Hybrid pipeline** (classical ML + transparent rules):

1. **Feature engineering** — 26 numeric features per user from a 28-day event log:
   event counts, conversion ratios, recency, activity trend (last 7 vs first 7 days),
   behavioral patterns, plus one-hot encoded profile data.
2. **Random Forest classifier** for `is_high_payer` (top tertile by payment count).
   5-fold stratified CV. Payment-derived features excluded to prevent leakage.
3. **K-Means (k=4)** on 8 curated behavioral features (StandardScaler-normalized) for
   unsupervised behavioral clusters.
4. **Rule-based scores** for churn risk and growth potential (hand-tuned weights,
   transparent and inspectable — no churn labels exist in the data so this is
   defensible).
5. **Priority-ordered segmentation**:
   `Churn Risk > Growth Potential > High Value > Medium Value`.
"""
    )

    st.subheader("Critical data finding")
    st.warning(
        "**99/100 users had at least one `payment_success` event.** "
        "A naive 'ever paid' binary target is degenerate (99% positive class). "
        "We reframed the target to `is_high_payer` (top tertile by payment count) "
        "yielding a balanced 41/59 split — and a more business-meaningful signal."
    )

    st.subheader("Honest limitations")
    st.markdown(
        """
- **N = 100** → 5-fold CV metrics have ±5pp variance run-to-run.
- **28-day window** is shorter than typical churn cohorts (30–90 days). The
  `activity_trend` (last-7 vs first-7) feature is our best churn proxy.
- **Rule-based weights** are hand-tuned. With labeled churn/upsell outcomes they
  could be learned via logistic regression.
- **Synthetic-looking data** — event spacing is regular and per-user counts cluster
  tightly (18–38). Model should be revalidated on real production data.
- **Cluster names** are heuristic post-hoc labels; the centroid table is the
  primary reference.
"""
    )

    st.divider()
    st.subheader("Data summary")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Users", len(df))
    d2.metric("Plans", df["plan_type"].nunique())
    d3.metric("Countries", df["country"].nunique())
    d4.metric("Industries", df["industry"].nunique())

    st.divider()
    st.markdown(
        """
**How to run locally**:
```bash
git clone <this-repo>
cd ikas
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py        # retrain models + generate outputs (~3s)
streamlit run streamlit_app.py
```

**Other entry points**:
- `python api.py` — FastAPI REST service (port 8000, `/docs` for Swagger UI)
- `python dashboard.py` — Plotly Dash equivalent (legacy)
"""
    )
