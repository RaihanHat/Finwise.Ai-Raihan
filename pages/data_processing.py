import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from models.preprocessing import run_pipeline
from utils.calculations import load_state
from utils.constants import AMBER, STEEL
from utils.helpers import ai_panel, empty_state, kpi_card, money, no_data_gate, page_header, pct, pipeline_flow, section, style_fig, term


def _simulate_missing(raw, seed=3):
    rng = np.random.default_rng(seed)
    r = raw.copy()
    r.loc[rng.random(len(r)) < 0.06, "amount"] = np.nan
    r.loc[rng.random(len(r)) < 0.08, "description"] = np.nan
    return r


def render():
    page_header("Data Processing", "Raw transactions pass through a real, reusable preprocessing pipeline before reaching any model.", "03 · Pipeline")
    state = load_state()
    if no_data_gate(state):
        return
    demo = st.toggle("Demo: simulate missing values on a copy (not saved)", help="Randomly blanks ~6% of amounts and ~8% of descriptions in an in-memory copy so you can watch imputation work.")
    raw = _simulate_missing(state["raw"]) if demo else state["raw"]
    pipe = run_pipeline(raw, state["settings"]["opening_balance"])
    if not pipe["ready"]:
        empty_state("No usable rows after cleaning.", "Check that your transactions have a date, type and category.")
        return
    pipeline_flow(["RAW DATA", "Missing Value Handling", "Encoding", "Normalization", "Feature Engineering", "PROCESSED DATA"])
    ai_panel("Data health", f"<div><b>{len(pipe['clean'])}</b> transactions ready for analysis · {int(pipe['missing']['missing'].sum())} missing value(s) detected"
             f"{' (simulated)' if demo else ''}.</div>", "Class 7 pipeline")

    section("1 · Raw data & missing values", "Missing-value detection per column (is_discretionary counts expenses only).")
    a, b = st.columns([3, 2])
    a.dataframe(pipe["raw"].head(15), width="stretch", hide_index=True)
    b.dataframe(pipe["missing"], width="stretch", hide_index=True)
    st.caption(f"Raw shape: {pipe['raw'].shape[0]} rows × {pipe['raw'].shape[1]} columns")

    section("2 · Missing value handling", "Amounts → category mean · descriptions → category default · discretionary flag → auto-tag.")
    st.dataframe(pd.DataFrame(pipe["log"]), width="stretch", hide_index=True)
    imp = pipe["clean"][pipe["clean"]["amount_imputed"] | pipe["clean"]["description_imputed"]]
    if len(imp):
        st.caption("Rows changed by imputation (after):")
        st.dataframe(imp[["id", "date", "category", "amount", "description", "amount_imputed", "description_imputed"]].head(15), width="stretch", hide_index=True)

    section("3 · Categorical encoding", "One-hot for type & category, plus target encoding (mean amount per category).")
    enc = pipe["encoded"]
    onehot = [c for c in enc.columns if c.startswith(("type_", "cat_"))]
    st.markdown(f"{len(onehot)} one-hot features + 1 target-encoded feature: " + ", ".join(f"`{c}`" for c in onehot) + ", `category_target_enc`")
    st.dataframe(enc.head(10), width="stretch", hide_index=True)

    section("4 · MinMax normalization", "Each numeric feature scaled to [0, 1] with scikit-learn MinMaxScaler (used by K-Means and KNN).")
    num_cols = list(pipe["ranges"]["feature"])
    before = pipe["features"][[c for c in num_cols if c in pipe["features"].columns]].head(8).reset_index(drop=True)
    before["category_target_enc"] = enc["category_target_enc"].head(8).values
    a, b = st.columns(2)
    a.markdown("**Before**")
    a.dataframe(before.round(2), width="stretch", hide_index=True)
    b.markdown("**After (normalized)**")
    b.dataframe(pipe["normalized"].head(8).round(3).reset_index(drop=True), width="stretch", hide_index=True)
    st.dataframe(pipe["ranges"].round(2), width="stretch", hide_index=True)
    fig = go.Figure(go.Box(y=pipe["normalized"]["amount_norm"], name="amount (normalized)", marker_color=STEEL))
    st.plotly_chart(style_fig(fig, 260, False), width="stretch")

    section("5 · Feature engineering & derived metrics")
    feat = pipe["features"]
    st.dataframe(feat[["id", "date", "category", "amount", "signed_amount", "month_key", "day_of_week", "is_weekend", "days_to_month_end", "discretionary", "essential"]].head(12),
                 width="stretch", hide_index=True)
    m = pipe["metrics"]
    st.markdown(f"Derived metrics for the latest month (**{m['month_key']}**):")
    c = st.columns(3)
    with c[0]:
        kpi_card("Savings Ratio", pct(m["savings_ratio"]), f"({money(m['net_income'], 0)} − {money(m['expenses'], 0)}) / {money(m['net_income'], 0)}", tip="Net income excludes Loan proceeds.")
    with c[1]:
        kpi_card("Monthly Burn Rate", money(m["burn_rate"]), "7-day rolling avg daily spend × 30")
    with c[2]:
        kpi_card("Discretionary Ratio", pct(m["discretionary_ratio"]), f"{money(m['discretionary'], 0)} / {money(m['expenses'], 0)}")
    st.markdown("Monthly metrics (all months):")
    st.dataframe(pipe["monthly"].round(3), width="stretch", hide_index=True)

    section("PROCESSED DATA")
    out = feat.join(pipe["normalized"]).drop(columns=["running_balance_delta"])
    st.dataframe(out.head(25), width="stretch", hide_index=True)
    st.download_button("Download processed data (CSV)", out.to_csv(index=False), "finwise_processed.csv", "text/csv")
    st.markdown("Glossary: " + term("MinMax") + " · " + term("Savings Ratio") + " · " + term("Burn Rate") + " · " + term("Discretionary Ratio"), unsafe_allow_html=True)
