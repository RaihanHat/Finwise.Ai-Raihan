import html

import plotly.graph_objects as go
import streamlit as st

from models.budget_csp import CSPInputs, VARS, WEIGHTS, solve
from utils.calculations import average_category_expenses, load_state
from utils.constants import AMBER, BRICK, CURRENCY, JADE, STEEL
from utils.helpers import ai_panel, empty_state, kpi_card, money, no_data_gate, page_header, section, style_fig, term, tone_chip


def render():
    page_header("Budget Optimizer", "A Constraint Satisfaction Problem solved by recursive backtracking + forward checking + MRV.", "06 · Budget")
    state = load_state()
    if no_data_gate(state):
        return
    clean, s = state["pipe"]["clean"], state["settings"]
    avg = average_category_expenses(clean)
    monthly = state["pipe"]["monthly"]
    n = st.session_state.get("csp_reset", 0)
    income_d = float(monthly["net_income"].mean())
    disc_d = float(monthly["discretionary"].mean())
    ess = state["essential_avg"]
    em_gap = max(s["emergency_target_months"] * ess - s["emergency_reserve"], 0.0)
    st.caption(f"Defaults are averages over {len(monthly)} month(s) of your transactions and your Settings; edit freely.")
    with st.form("csp_form"):
        st.markdown("**Inputs**")
        a, b, c = st.columns(3)
        income = a.number_input(f"Monthly income ({CURRENCY})", min_value=0.0, value=round(income_d, 0), step=1000.0, key=f"i{n}")
        housing = b.number_input("Minimum housing", min_value=0.0, value=round(avg.get("Housing", 0.0), 0), step=500.0, key=f"h{n}")
        debt = c.number_input("Debt payment (minimum)", min_value=0.0, value=float(s["monthly_debt_payment"]), step=500.0, key=f"d{n}")
        a, b, c = st.columns(3)
        food = a.number_input("Food", min_value=0.0, value=round(avg.get("Food", 0.0) - 0, 0), step=500.0, key=f"f{n}")
        transport = b.number_input("Transportation", min_value=0.0, value=round(avg.get("Transportation", 0.0), 0), step=250.0, key=f"t{n}")
        health = c.number_input("Health", min_value=0.0, value=round(avg.get("Health", 0.0), 0), step=250.0, key=f"he{n}")
        a, b, c = st.columns(3)
        edu = a.number_input("Education", min_value=0.0, value=round(avg.get("Education", 0.0), 0), step=500.0, key=f"e{n}")
        emergency = b.number_input("Emergency reserve (target contribution)", min_value=0.0, value=round(em_gap, 0), step=1000.0, key=f"em{n}", help="Prefilled: gap between your emergency target and current reserve (Settings).")
        disc = c.number_input("Discretionary spending", min_value=0.0, value=round(disc_d, 0), step=500.0, key=f"di{n}")
        a, b, c = st.columns(3)
        savings = a.number_input("Savings target", min_value=0.0, value=round(s["savings_target_ratio"] * income_d, 0), step=1000.0, key=f"s{n}")
        floor = b.slider("Minimum acceptable share of essentials", 0.3, 1.0, 0.8, 0.05, help="Food/transport/health/education may be trimmed down to this share of the requested amount.")
        cap = c.slider("Discretionary cap (soft constraint)", 0.05, 0.5, 0.20, 0.01, help="Soft constraint S3: discretionary spending below this share of net income.")
        run = st.form_submit_button("Solve budget (CSP)", type="primary")
    if run:
        inp = CSPInputs(income, housing, food, transport, health, edu, debt, emergency, disc, savings, floor, cap)
        with st.spinner("Searching allocations…"):
            st.session_state["csp_result"] = solve(inp)
    if st.button("Reset inputs to data defaults"):
        st.session_state["csp_reset"] = n + 1
        st.session_state.pop("csp_result", None)
        st.rerun()
    res = st.session_state.get("csp_result")
    if not res:
        empty_state("Set your inputs and press 'Solve budget'.", "The solver will search for a feasible allocation that satisfies all hard constraints.")
        return
    if res["status"] == "invalid":
        for e in res["errors"]:
            st.error(e)
        return
    inp = res["inputs"]

    section("Constraints")
    cons = res["constraints"]
    st.markdown("<br>".join(f"{tone_chip(c['status'])} &nbsp;<b>{c['id']}</b> <span style='color:#8B949E'>({c['type']})</span> {html.escape(c['constraint'])} — <span class='fw-mono'>{html.escape(c['detail'])}</span>" for c in cons), unsafe_allow_html=True)

    section("Search status")
    st_ = res["stats"]
    status_label = {"optimal": "Search complete · optimal solution found", "truncated": "Node limit reached · best solution so far", "infeasible": "Infeasible · no allocation satisfies the hard constraints"}[res["status"]]
    st.markdown(tone_chip({"optimal": "Satisfied", "truncated": "At limit", "infeasible": "Violated"}[res["status"]]) + f" &nbsp; {status_label}", unsafe_allow_html=True)
    c = st.columns(5)
    for col, (lab, val) in zip(c, [("Variables", res["n_variables"]), ("Nodes visited", f"{st_['nodes']:,}"), ("Backtracks", f"{st_['backtracks']:,}"),
                                   ("FC prunings", f"{st_['fc_prunings']:,}"), ("Feasible solutions", f"{st_['solutions']:,}")]):
        with col:
            kpi_card(lab, str(val))

    if not res["feasible"]:
        short = res.get("shortfall")
        st.error(f"This month's fixed costs exceed income by {money(short, 0)}. Reduce a goal target or log additional income." if short else res.get("reason", "Infeasible."))
        st.markdown(f"**Why:** {res.get('reason', '')}")
        with st.expander("Search trace"):
            st.write(res["trace"])
        return

    section("Optimized allocation")
    a = res["allocation"]
    req = res["requested"]
    rows = [{"Category": v, "Requested / minimum": req[v], "Allocated": a[v], "Share of income": a[v] / inp.income} for v in VARS]
    st.dataframe(rows, width="stretch", hide_index=True, column_config={
        "Requested / minimum": st.column_config.NumberColumn(format=f"{CURRENCY}%.0f"), "Allocated": st.column_config.NumberColumn(format=f"{CURRENCY}%.0f"),
        "Share of income": st.column_config.NumberColumn(format="%.1f%%", help="Fraction × 100 shown as percent")})
    c = st.columns(3)
    with c[0]:
        kpi_card("Total allocation", money(res["total"]), f"of income {money(inp.income, 0)}")
    with c[1]:
        kpi_card("Remaining amount", money(res["remaining"]), "unallocated (domain step rounding)" if res["remaining"] > 0 else "fully allocated", "jade")
    ok = all(x["status"] != "Violated" for x in cons)
    with c[2]:
        kpi_card("Constraint status", "All hard constraints met" if ok else "Violation", "3 hard · 3 soft checked", "jade" if ok else "brick")
    fig = go.Figure([go.Bar(name="Requested / minimum", x=VARS, y=[req[v] for v in VARS], marker_color="#39414d"),
                     go.Bar(name="Allocated", x=VARS, y=[a[v] for v in VARS], marker_color=STEEL)])
    fig.update_layout(barmode="group", title="Requested vs allocated")
    st.plotly_chart(style_fig(fig), width="stretch")

    section("Why this solution was selected")
    ai_panel("CSP reasoning", "".join(f"<div>• {html.escape(x)}</div>" for x in res["explanation"]), "Backtracking · FC · MRV")
    with st.expander("Domains, objective weights and search trace"):
        st.markdown(f"Domain step {money(res['step'], 0)} · objective = Σ weight × allocation / requested (higher is better).")
        st.dataframe([{"Variable": v, "Floor": res["floors"][v], "Requested": req[v], "Domain size": res["domain_sizes"][v], "Soft weight": WEIGHTS[v]} for v in VARS], width="stretch", hide_index=True)
        st.markdown("**Search trace (first events)**")
        st.code("\n".join(res["trace"][:40]))
    st.markdown(" · ".join(term(t) for t in ("CSP", "Backtracking", "Forward Checking", "MRV")), unsafe_allow_html=True)
