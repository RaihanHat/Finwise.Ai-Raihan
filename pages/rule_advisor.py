import html

import streamlit as st

from models.rule_engine import PRIORITY_ORDER, RULES
from utils.calculations import load_state, rules_for
from utils.helpers import ai_panel, kpi_card, money, no_data_gate, page_header, section, tone_chip


def render():
    page_header("Rule-Based Advisor", "Transparent IF-THEN rules evaluated on your actual financial metrics (rules live in models/rule_engine.py).", "07 · Rules")
    state = load_state()
    if no_data_gate(state):
        return
    month = st.selectbox("Month", state["months"][::-1], index=0)
    metrics, results = rules_for(state, month)
    trig = sorted([r for r in results if r["triggered"]], key=lambda r: -PRIORITY_ORDER[r["priority"]])
    na = [r for r in results if r["status"] == "Not evaluable"]
    c = st.columns(3)
    with c[0]:
        kpi_card("Rules evaluated", f"{len(results) - len(na)} / {len(results)}", "R01–R06")
    with c[1]:
        kpi_card("Triggered", str(len(trig)), "recommendations below", "amber" if trig else "jade")
    with c[2]:
        kpi_card("Emergency target", money((state["settings"]["emergency_target_months"] * state["essential_avg"]), 0), f"{state['settings']['emergency_target_months']:g} months of essentials")

    section("Recommendations", "Sorted by priority.")
    if not trig:
        ai_panel("All clear", "<div>No rule was triggered for this month.</div>", "Rule engine")
    for r in trig:
        st.markdown(
            f'<div class="fw-card" style="margin-bottom:.7rem"><div><b class="fw-mono">RULE {r["rule_id"]}</b> &nbsp;{tone_chip(r["status"])} &nbsp;'
            f'{tone_chip(r["priority"]) if r["priority"] != "Info" else ""}<span class="fw-chip muted">Priority: {r["priority"]}</span></div>'
            f'<div style="margin-top:.5rem"><span class="fw-kpi-label">Condition</span><br>{html.escape(r["condition"])}</div>'
            f'<div style="margin-top:.4rem"><span class="fw-kpi-label">Current</span> <span class="fw-mono">{r["current"]}</span> &nbsp; '
            f'<span class="fw-kpi-label">Threshold</span> <span class="fw-mono">{r["threshold"]}</span></div>'
            f'<div style="margin-top:.4rem"><span class="fw-kpi-label">Recommendation</span><br>{html.escape(r["recommendation"])}</div></div>', unsafe_allow_html=True)

    section("Rule details", "Every rule, whether triggered or not, with the reasoning behind it.")
    st.dataframe([{"Rule ID": r["rule_id"], "Name": r["name"], "Condition": r["condition"], "Current": r["current"], "Threshold": r["threshold"],
                   "Result": r["status"], "Priority": r["priority"], "Recommendation": r["recommendation"]} for r in results], width="stretch", hide_index=True)
    for r, rule in zip(results, RULES):
        with st.expander(f"{r['rule_id']} · {r['name']} — {r['status']}"):
            st.markdown(f"**IF** `{r['condition']}` **THEN** {rule.recommendation.format(thr=r['threshold'])}\n\n"
                        f"- Why: {r['explanation']}\n- Metric `{r['metric']}` currently **{r['current']}** vs threshold **{r['threshold']}** → **{r['status']}**\n"
                        f"- Type: {r['kind']} rule · priority {r['priority']}")
    with st.expander("Metric inputs used"):
        st.json({k: (None if v is None else round(v, 4)) for k, v in metrics.items()})
        st.caption("Debt payment and emergency reserve come from Settings; all other metrics are computed from your transactions.")
