"""Plotly Dash real-time dashboard for ikas user segments.

Run with:  python dashboard.py
Opens at:  http://127.0.0.1:8050
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, dash_table, dcc, html

from src.config import (
    FEATURE_IMPORTANCES_CSV,
    PREDICTIONS_CSV,
    SEGMENT_SUMMARY_CSV,
)

SEGMENT_COLORS = {
    "High Value": "#2ECC71",
    "Medium Value": "#3498DB",
    "Churn Risk": "#E74C3C",
    "Growth Potential": "#F39C12",
}


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    preds = pd.read_csv(PREDICTIONS_CSV)
    summary = pd.read_csv(SEGMENT_SUMMARY_CSV)
    importances = pd.read_csv(FEATURE_IMPORTANCES_CSV).head(10)
    mtime = datetime.fromtimestamp(Path(PREDICTIONS_CSV).stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return preds, summary, importances, mtime


def _kpi_card(title: str, value, color: str = "primary") -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.H6(title, className="text-muted mb-1"),
            html.H3(value, className=f"text-{color} mb-0"),
        ]),
        className="shadow-sm",
    )


def _filter_options(df: pd.DataFrame, col: str) -> list[dict]:
    return [{"label": v, "value": v} for v in sorted(df[col].dropna().unique())]


app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    title="ikas — Growth Engineer Dashboard",
)

preds_init, summary_init, importances_init, mtime_init = _load()

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("ikas — Growth Engineer Dashboard", className="mt-3 mb-0"),
            html.P([
                "Real-time user segmentation & prediction · ",
                html.Span(id="mtime-display", children=f"data refreshed: {mtime_init}"),
            ], className="text-muted"),
        ], md=10),
        dbc.Col([
            dbc.Button("Refresh data", id="refresh-btn", color="primary", className="mt-3 float-end"),
        ], md=2),
    ]),
    html.Hr(),

    dbc.Row(id="kpi-row", className="mb-4"),

    dbc.Card([
        dbc.CardHeader(html.B("Filters")),
        dbc.CardBody(dbc.Row([
            dbc.Col([html.Label("Country"), dcc.Dropdown(id="f-country", multi=True,
                options=_filter_options(preds_init, "country"))], md=2),
            dbc.Col([html.Label("Industry"), dcc.Dropdown(id="f-industry", multi=True,
                options=_filter_options(preds_init, "industry"))], md=2),
            dbc.Col([html.Label("Plan"), dcc.Dropdown(id="f-plan", multi=True,
                options=_filter_options(preds_init, "plan_type"))], md=2),
            dbc.Col([html.Label("Device"), dcc.Dropdown(id="f-device", multi=True,
                options=_filter_options(preds_init, "device_type"))], md=2),
            dbc.Col([html.Label("Segment"), dcc.Dropdown(id="f-segment", multi=True,
                options=_filter_options(preds_init, "segment"))], md=2),
            dbc.Col([html.Label("Behavior cluster"), dcc.Dropdown(id="f-cluster", multi=True,
                options=_filter_options(preds_init, "behavior_cluster_name"))], md=2),
        ])),
    ], className="mb-4 shadow-sm"),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.B("Segment distribution")),
            dbc.CardBody(dcc.Graph(id="pie-segments", config={"displayModeBar": False})),
        ], className="shadow-sm"), md=6),
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.B("Segment × Plan type")),
            dbc.CardBody(dcc.Graph(id="bar-seg-plan", config={"displayModeBar": False})),
        ], className="shadow-sm"), md=6),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.B("Checkouts → payment likelihood (colored by segment)")),
            dbc.CardBody(dcc.Graph(id="scatter-checkout-pay", config={"displayModeBar": False})),
        ], className="shadow-sm"), md=7),
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.B("Top-10 feature importances (RF)")),
            dbc.CardBody(dcc.Graph(id="bar-importances", config={"displayModeBar": False})),
        ], className="shadow-sm"), md=5),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.B("Avg. payment likelihood — country × industry")),
            dbc.CardBody(dcc.Graph(id="heatmap-country-industry", config={"displayModeBar": False})),
        ], className="shadow-sm"), md=12),
    ], className="mb-4"),

    dbc.Card([
        dbc.CardHeader(html.B("Users — sortable, filterable, exportable")),
        dbc.CardBody(dash_table.DataTable(
            id="user-table",
            columns=[],
            data=[],
            page_size=15,
            sort_action="native",
            filter_action="native",
            export_format="csv",
            export_headers="display",
            style_cell={"fontSize": 13, "fontFamily": "system-ui", "padding": "6px"},
            style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
            style_data_conditional=[
                {"if": {"filter_query": f'{{segment}} = "{seg}"'}, "backgroundColor": color + "20"}
                for seg, color in SEGMENT_COLORS.items()
            ],
        )),
    ], className="shadow-sm mb-5"),

    dcc.Store(id="data-store"),
], fluid=True)


def _filter(df: pd.DataFrame, country, industry, plan, device, segment, cluster) -> pd.DataFrame:
    if country: df = df[df["country"].isin(country)]
    if industry: df = df[df["industry"].isin(industry)]
    if plan: df = df[df["plan_type"].isin(plan)]
    if device: df = df[df["device_type"].isin(device)]
    if segment: df = df[df["segment"].isin(segment)]
    if cluster: df = df[df["behavior_cluster_name"].isin(cluster)]
    return df


@app.callback(
    Output("data-store", "data"),
    Output("mtime-display", "children"),
    Input("refresh-btn", "n_clicks"),
)
def reload_data(_):
    preds, _summary, _imps, mtime = _load()
    return preds.to_dict("records"), f"data refreshed: {mtime}"


@app.callback(
    Output("kpi-row", "children"),
    Output("pie-segments", "figure"),
    Output("bar-seg-plan", "figure"),
    Output("scatter-checkout-pay", "figure"),
    Output("bar-importances", "figure"),
    Output("heatmap-country-industry", "figure"),
    Output("user-table", "data"),
    Output("user-table", "columns"),
    Input("data-store", "data"),
    Input("f-country", "value"),
    Input("f-industry", "value"),
    Input("f-plan", "value"),
    Input("f-device", "value"),
    Input("f-segment", "value"),
    Input("f-cluster", "value"),
)
def update(data, country, industry, plan, device, segment, cluster):
    df = pd.DataFrame(data) if data else preds_init.copy()
    df = _filter(df, country, industry, plan, device, segment, cluster)

    # KPI cards
    total = len(df)
    seg_counts = df["segment"].value_counts()
    kpis = dbc.Row([
        dbc.Col(_kpi_card("Total users", total, "primary"), md=3),
        dbc.Col(_kpi_card("High Value", int(seg_counts.get("High Value", 0)), "success"), md=3),
        dbc.Col(_kpi_card("Churn Risk", int(seg_counts.get("Churn Risk", 0)), "danger"), md=3),
        dbc.Col(_kpi_card("Growth Potential", int(seg_counts.get("Growth Potential", 0)), "warning"), md=3),
    ]).children

    # Pie
    pie = px.pie(df, names="segment", color="segment", color_discrete_map=SEGMENT_COLORS, hole=0.45)
    pie.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)

    # Stacked bar segment × plan
    if not df.empty:
        sp = df.groupby(["plan_type", "segment"]).size().reset_index(name="n")
        bar = px.bar(sp, x="plan_type", y="n", color="segment",
                     color_discrete_map=SEGMENT_COLORS, barmode="stack")
    else:
        bar = go.Figure()
    bar.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320,
                      xaxis_title="Plan", yaxis_title="Users")

    # Scatter: n_checkout_start vs payment_likelihood
    if not df.empty:
        sca = px.scatter(df, x="n_checkout_start", y="payment_likelihood",
                         color="segment", color_discrete_map=SEGMENT_COLORS,
                         size="n_payment_success", hover_data=["user_id", "plan_type", "country"])
    else:
        sca = go.Figure()
    sca.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=400,
                      xaxis_title="Checkout starts", yaxis_title="Payment likelihood")

    # Feature importances
    imp = pd.read_csv(FEATURE_IMPORTANCES_CSV).head(10).sort_values("importance")
    fi_fig = px.bar(imp, x="importance", y="feature", orientation="h")
    fi_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=400,
                         yaxis_title="", xaxis_title="Importance")

    # Heatmap country × industry
    if not df.empty:
        pivot = df.pivot_table(values="payment_likelihood", index="country",
                               columns="industry", aggfunc="mean")
        hm = px.imshow(pivot, color_continuous_scale="RdYlGn", aspect="auto",
                       text_auto=".2f", labels=dict(color="Avg payment likelihood"))
    else:
        hm = go.Figure()
    hm.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=400)

    # User table
    display_cols = [
        "user_id", "plan_type", "country", "device_type", "industry",
        "segment", "behavior_cluster_name",
        "payment_likelihood", "churn_risk", "growth_potential",
        "n_payment_success", "n_checkout_start", "n_login", "last_event_day",
    ]
    table_df = df[display_cols].copy()
    table_df["user_id"] = table_df["user_id"].str[:8]
    table_df = table_df.round({"payment_likelihood": 3, "churn_risk": 3,
                               "growth_potential": 3, "last_event_day": 1})
    columns = [{"name": c.replace("_", " ").title(), "id": c} for c in display_cols]

    return kpis, pie, bar, sca, fi_fig, hm, table_df.to_dict("records"), columns


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
