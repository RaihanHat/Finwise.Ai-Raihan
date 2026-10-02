import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import calculations as calc
from utils.constants import AMBER, BRICK, CATEGORY_COLORS, JADE, STEEL
from utils.helpers import ai_panel, chip, empty_state, kpi_card, money, no_data_gate, page_header, pct, section, style_fig, tone_chip


def render():
    page_header("Monthly Overview", "Live view of your transactions, derived metrics and financial health.", "01 · Dashboard")
    state = calc.load_state()
    if no_data_gate(state):
        return
    months = state["months"]
    c1, c2 = st.columns([1, 3])
    month = c1.selectbox("Month", months[::-1], index=0, help="Metrics below are computed for the selected month.")
    mm = calc.month_metrics(state, month)
    prev = calc.previous_month(state, month)
    clean, monthly = state["pipe"]["clean"], state["pipe"]["monthly"]
    balance = calc.total_balance(state)
    delta = (mm["net_savings"] - prev["net_savings"]) if prev else None
    with c2:
        st.markdown(f'<div class="fw-kpi-label">Current balance</div><div class="fw-kpi-value" style="font-size:2.1rem">{html.escape(money(balance))}</div>'
                    f'<div class="fw-kpi-sub">{"Net savings vs previous month: " + money(delta) if delta is not None else "No previous month to compare"}</div>', unsafe_allow_html=True)

    s = state["settings"]
    sr, dr = mm["savings_ratio"], mm["discretionary_ratio"]
    cols = st.columns(3)
    with cols[0]:
        kpi_card("Monthly Income", money(mm["income"]), f"{mm['tx_count']} transactions in month")
    with cols[1]:
        kpi_card("Monthly Expenses", money(mm["expenses"]), f"Essential {money(mm['essential'], 0)}")
    with cols[2]:
        kpi_card("Remaining Balance", money(balance), "All-time: opening + income - expenses", "jade" if balance >= 0 else "brick")
    cols = st.columns(3)
    with cols[0]:
        kpi_card("Savings Ratio", pct(sr), f"Healthy ≥ {pct(s['thr_savings_healthy'], 0)} · min {pct(s['thr_savings_min'], 0)}",
                 None if sr is None else "jade" if sr >= s["thr_savings_healthy"] else "amber" if sr >= s["thr_savings_min"] else "brick",
                 "(Net income − expenses) / net income. Loan proceeds are excluded from net income.")
    with cols[1]:
        kpi_card("Burn Rate (30d)", money(mm["burn_rate"]), "Rolling 7-day average × 30", tip="Rolling 7-day average daily spend extrapolated to a 30-day window.")
    with cols[2]:
        kpi_card("Discretionary Ratio", pct(dr), f"Limit {pct(s['thr_discretionary_max'], 0)} of expenses",
                 None if dr is None else "jade" if dr <= s["thr_discretionary_max"] else "brick")
    if mm["month_key"] == months[-1]:
        last = clean["date"].max()
        if last < (last + pd.offsets.MonthEnd(0)).normalize():
            st.caption(f"Latest month is in progress (data up to {last:%d %b %Y}); ratios are partial-month values.")

    section("Charts")
    a, b = st.columns(2)
    fig = go.Figure([go.Bar(name="Income", x=monthly["month_key"], y=monthly["income"], marker_color=JADE),
                     go.Bar(name="Expense", x=monthly["month_key"], y=monthly["expenses"], marker_color=BRICK)])
    fig.update_layout(barmode="group", title="Income vs Expense")
    a.plotly_chart(style_fig(fig), width="stretch")
    cat = calc.category_totals(clean, "Expense", month)
    fig = go.Figure(go.Bar(x=cat["amount"], y=cat["category"], orientation="h", marker_color=[CATEGORY_COLORS.get(c, STEEL) for c in cat["category"]],
                           text=[f"{p:.0f}%" for p in cat["pct"]], textposition="outside"))
    fig.update_layout(title=f"Expense by Category · {month}", yaxis=dict(autorange="reversed"))
    b.plotly_chart(style_fig(fig, legend=False), width="stretch")
    a, b = st.columns(2)
    fig = go.Figure([go.Scatter(name="Expenses", x=monthly["month_key"], y=monthly["expenses"], mode="lines+markers", line=dict(color=AMBER, width=3)),
                     go.Scatter(name="Net savings", x=monthly["month_key"], y=monthly["net_savings"], mode="lines+markers", line=dict(color=STEEL, width=3, dash="dot"))])
    fig.update_layout(title="Monthly Spending Trend")
    a.plotly_chart(style_fig(fig), width="stretch")
    fig = go.Figure([go.Bar(name="Essential", x=monthly["month_key"], y=monthly["essential"], marker_color=STEEL),
                     go.Bar(name="Discretionary", x=monthly["month_key"], y=monthly["discretionary"], marker_color=AMBER)])
    fig.update_layout(barmode="stack", title="Essential vs Discretionary Spending")
    b.plotly_chart(style_fig(fig), width="stretch")

    section("Financial Health", "Each check is a rule from the Rule-Based Advisor evaluated on the selected month.")
    metrics, rules = calc.rules_for(state, month)
    warn = [r for r in rules if r["kind"] == "warning" and r["status"] != "Not evaluable"]
    ok = [r for r in warn if not r["triggered"]]
    score = len(ok) / len(warn) * 100 if warn else None
    a, b = st.columns([1, 2])
    with a:
        if score is None:
            empty_state("Insufficient data for a health score.")
        else:
            g = go.Figure(go.Indicator(mode="gauge+number", value=score, number=dict(suffix="%", font=dict(family="JetBrains Mono")),
                          gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=JADE if score >= 80 else AMBER if score >= 50 else BRICK))))
            g.update_layout(title="Rule checks passed")
            st.plotly_chart(style_fig(g, 230, False), width="stretch")
    with b:
        for r in rules:
            if r["status"] == "Not evaluable":
                continue
            tone = "Satisfied" if (not r["triggered"] and r["kind"] == "warning") or (r["triggered"] and r["kind"] == "positive") else "Violated" if r["priority"] == "High" else "At limit"
            st.markdown(f'{tone_chip(tone)} &nbsp; **{r["rule_id"]} · {html.escape(r["name"])}** — <span class="fw-mono">{r["current"]}</span> '
                        f'<span style="color:#8B949E">(threshold {r["threshold"]})</span>', unsafe_allow_html=True)
    high = [r for r in rules if r["triggered"] and r["kind"] == "warning"]
    summary = "On track: no warning rules triggered." if not high else "Watch: " + "; ".join(r["name"].lower() for r in high[:3]) + "."
    ai_panel("Monthly snapshot", f"<div>{html.escape(summary)}</div>", "Rule engine")
    csp = st.session_state.get("csp_result")
    if csp and csp.get("status") == "infeasible":
        st.error(f"Budget alert (CSP): {csp.get('reason')}")
