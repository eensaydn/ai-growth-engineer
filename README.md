# ikas Growth Engineer Case Study

> **Live demo**: <https://ai-growth-engineer-emss3fhfrukhzt8bmzfday.streamlit.app>
> **Stack**: Python · pandas · scikit-learn · Plotly · Streamlit · FastAPI

AI driven user segmentation for ikas. Given two CSVs (100 user profiles + 2,817 events over 28 days) the pipeline produces, for each user:

- **payment likelihood** (Random Forest probability),
- **churn risk** (rule based score, 0 to 1),
- **growth / upsell potential** (rule based score, 0 to 1),
- a **behavior cluster** (K-Means, k=4),
- and a final **business segment** out of *High Value · Medium Value · Churn Risk · Growth Potential*.

The deliverables are a trained pipeline, a CSV / JSON of predictions, a Streamlit dashboard deployed to Streamlit Community Cloud, and a FastAPI REST service that exposes the same data for downstream integrations.

---

## Quick start

```bash
git clone https://github.com/eensaydn/ai-growth-engineer.git
cd ai-growth-engineer
python -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt          # full ML stack for local runs
python run_pipeline.py                        # retrain + regenerate everything (~3s)
streamlit run streamlit_app.py                # local dashboard at :8501
```

For Streamlit Cloud deployment the lighter `requirements.txt` is used (only `pandas`, `numpy`, `plotly`, `streamlit`). Python 3.11 is pinned via `.python-version` and `runtime.txt`.

---

## Live deliverables

| Surface | URL / command | Purpose |
| --- | --- | --- |
| Dashboard | <https://ai-growth-engineer-emss3fhfrukhzt8bmzfday.streamlit.app> | Real time segmentation view |
| REST API | `python api.py` then `/docs` | 10 JSON endpoints + Swagger UI |
| Repo | <https://github.com/eensaydn/ai-growth-engineer> | Source, models, outputs |
| Plotly Dash (legacy) | `python dashboard.py` | Alternative dashboard in pure Python |

The trained models (`models/*.joblib`), the precomputed predictions (`outputs/user_predictions.csv` and `.json`) and the case study PDF are committed for one click reproducibility.

---

## Architecture

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                       src/                              │
                    │  data_loader → feature_engineering → labeling           │
                    │     → train_classifier (RF, 5-fold CV)                  │
                    │     → train_clusterer  (K-Means, k=4)                   │
                    │     → segmentation     (priority rules)                 │
                    │     → predict.py       (orchestrator)                   │
                    └────────────────────────┬────────────────────────────────┘
                                             │ writes
                                             ▼
                    outputs/user_predictions.csv  +  models/*.joblib
                                             │ read by
                                ┌────────────┼────────────┐
                                ▼            ▼            ▼
                       streamlit_app.py   dashboard.py   api.py
                       (Streamlit Cloud)  (Plotly Dash)  (FastAPI + Swagger)
```

A single `run_pipeline.py` re-runs the entire chain idempotently. The three frontends read the same artifacts.

---

## Key results

5-fold stratified cross validation of the high-payer classifier:

| Metric | Value |
| --- | --- |
| Accuracy | 0.600 |
| ROC-AUC | 0.615 |
| F1 | 0.487 |
| Precision | 0.514 |
| Recall | 0.463 |

Final segment distribution:

| Segment | n | Avg payment likelihood | Avg payments | Top plan | Top country | Top industry |
| --- | --- | --- | --- | --- | --- | --- |
| High Value | 16 | 0.719 | 7.75 | Business | UK | Healthcare |
| Growth Potential | 23 | 0.466 | 5.52 | Free | USA | SaaS |
| Churn Risk | 31 | 0.463 | 4.87 | Free | France | Healthcare |
| Medium Value | 30 | 0.274 | 3.63 | Pro | Germany | Healthcare |

Top three Random Forest feature importances: `trial_extension_intensity`, `n_trial_extension`, `feature_click_rate`. No payment derived feature appears in the top ten, confirming the leakage guards.

---

## Methodology

### 1. Critical data finding

99 of 100 users have at least one `payment_success` event. A naive "ever paid" target is degenerate. We reframed the target to **`is_high_payer`** (top tertile by payment count), yielding a balanced 41 / 59 split and a more business meaningful signal.

### 2. Feature engineering (26 per-user features)

Built in `src/feature_engineering.py` from the event log alone. Categories:

- **Event counts** (7): `n_login`, `n_logout`, `n_checkout_start`, `n_payment_success`, `n_trial_extension`, `n_feature_click`, `n_total_events`.
- **Conversion ratios** (4): `checkout_to_payment_rate`, `login_to_payment_rate`, `payment_per_event`, `feature_click_rate`.
- **Recency** (6) anchored on `REFERENCE_DATE = max(timestamp) + 1 day` (not `datetime.now()`).
- **Activity trend** (3): `events_last_7d`, `events_first_7d`, `activity_trend = (last7 − first7) / max(first7, 1)`.
- **Behavioral patterns** (3): `avg_events_per_active_day`, `login_logout_ratio`, `trial_extension_intensity`.
- **Profile**: `plan_ordinal` (Free=0, Pro=1, Business=2), `device_is_desktop`, one hot encoded country and industry.

### 3. Hybrid ML pipeline

- **Random Forest classifier** for `is_high_payer`. `n_estimators=300`, `max_depth=6`, `min_samples_leaf=3`, `class_weight="balanced"`. Stratified 5-fold CV (not hold-out) because a 20-sample test split is too noisy on N=100. Final model retrained on all 100 samples. **Leakage guard**: 5 payment derived columns explicitly excluded from the feature matrix.
- **K-Means (k=4)** on 8 curated behavioral features (StandardScaler normalized). Validated with elbow + silhouette score (0.173). Cluster names assigned post hoc from centroid magnitudes.
- **Rule based churn risk** (0 to 1): weighted blend of recency, decay, trial-extension intensity.
- **Rule based growth potential** (0 to 1): engagement composite × plan multiplier (Free=1.0, Pro=0.7, Business=0.0).

### 4. Priority ordered segmentation

```
if churn_risk >= 0.65:                                              → Churn Risk
elif growth_potential >= 0.40 and plan in (Free, Pro):              → Growth Potential
elif payment_likelihood >= 0.55 and n_payment >= median(payments):  → High Value
else:                                                                → Medium Value
```

The priority order is deliberate. A high-paying user with churn signals must be flagged for retention, not for loyalty rewards.

---

## Honest limitations

1. **N = 100.** Cross-validation metrics carry ±5pp run-to-run variance. ROC-AUC of 0.615 is modest but in line with the sample size.
2. **28 day window.** Shorter than typical churn cohorts (30 to 90 days). `activity_trend` (last 7 vs first 7) is the best available proxy.
3. **Rule weights are hand tuned.** With labeled churn outcomes they could be learned through logistic regression.
4. **Synthetic looking data.** Event spacing is unusually regular and event counts cluster in a narrow 18-38 range. The pipeline should be revalidated on real production data.
5. **Cluster names are heuristic.** The centroid table in `model_metadata.json` is the primary reference. The labels are convenience UI annotations.
6. **"Real time" is snapshot refresh.** Streamlit reloads the CSV on a Refresh click. For genuine streaming, the pipeline would need to wrap an event consumer.

---

## Repository layout

```
ai-growth-engineer/
├── data/
│   ├── raw/                                Read only inputs
│   └── processed/user_features.csv         Feature engineering output
├── models/                                 Joblib serialized models + metadata
├── outputs/                                CSV, JSON, PNG deliverables
├── src/
│   ├── config.py            Paths, hyperparameters, thresholds, leakage list
│   ├── data_loader.py       CSV loading + validation
│   ├── feature_engineering.py
│   ├── labeling.py          Target + rule based scores
│   ├── train_classifier.py  RF + stratified 5-fold CV + leakage guard
│   ├── train_clusterer.py   KMeans + elbow + PCA + post hoc names
│   ├── segmentation.py      Priority rules
│   └── predict.py           End to end orchestrator
├── streamlit_app.py         Primary dashboard (deployed)
├── dashboard.py             Plotly Dash alternative
├── api.py                   FastAPI REST service with CORS for Retool / n8n / CRM
├── run_pipeline.py          One command retrain
├── start_services.sh        Boots api.py + cloudflared tunnel
├── stop_services.sh         Stops the above
├── requirements.txt         Streamlit Cloud only (lean)
├── requirements-dev.txt     Full local stack
├── .python-version          3.11
├── runtime.txt              python-3.11
├── RETOOL_SETUP.md          Optional Retool integration guide
└── README.md
```

---

## REST API surface

Run `python api.py` then visit `/docs` for the Swagger UI.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Liveness + last refresh timestamp |
| GET | `/dashboard-data` | One shot bundle for any dashboard frontend |
| GET | `/predictions` | Filterable users (segment, country, industry, plan, device) |
| GET | `/predictions/{user_id}` | Single user detail |
| GET | `/segments/counts` | Counts per segment |
| GET | `/segments/summary` | Aggregate table per segment |
| GET | `/feature-importances` | Random Forest top features |
| GET | `/filters` | Distinct values for every filterable column |
| GET | `/metadata` | Model metadata (CV scores, cluster info, hyperparameters) |
| POST | `/refresh` | Re-run the full pipeline (~3 seconds) |

CORS is open (`*`) so Retool, n8n, internal frontends or scripts can call it without proxying. For production a single Retool subdomain or signed JWT would replace the wildcard.

---

## Reproduction

```bash
python run_pipeline.py          # writes outputs/, models/, and prints CV scores
streamlit run streamlit_app.py  # local dashboard
python api.py                   # local FastAPI on :8000
./start_services.sh             # API + public cloudflared tunnel together
```

Reruns are idempotent: same input data produces identical output (every randomness is seeded with `random_state=42`).
