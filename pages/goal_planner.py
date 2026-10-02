import html

import plotly.graph_objects as go
import streamlit as st

from models.goal_astar import astar_plan
from utils.calculations import load_state
from utils.constants import AMBER, BRICK, CURRENCY, JADE, STEEL
from utils.helpers import ai_panel, empty_state, kpi_card, money, no_data_gate, page_header, section, style_fig, term

LEVEL_COLOR = {"Low": JADE, "Medium": AMBER, "High": BRICK}


def render():
    page_header("Goal Planner", "A* search finds a low-disruption month-by-month savings path to your goal.", "08 · Goals")
    state = load_state()
    if no_data_gate(state):
        return
    m = state["pipe"]["monthly"]
    comfy = max(float(m["net_savings"].mean()), 0.0)
    cut = float(m["discretionary"].mean())
    st.caption(f"Defaults come from your data: average monthly net savings {money(comfy, 0)} (no-cut saving) and average discretionary spending {money(cut, 0)} (what could be cut).")
    with st.form("goal_form"):
        a, b, c = st.columns(3)
        name = a.text_input("Goal name", placeholder="e.g. Laptop")
        target = b.number_input(f"Target price ({CURRENCY})", min_value=0.0, value=0.0, step=1000.0)
        current = c.number_input(f"Current savings ({CURRENCY})", min_value=0.0, value=0.0, step=1000.0)
        a, b, c = st.columns(3)
        max_save = a.number_input(f"Maximum monthly saving ({CURRENCY})", min_value=0.0, value=round(comfy + cut, 0), step=500.0)
        comfortable = b.number_input(f"Comfortable saving, no lifestyle cut ({CURRENCY})", min_value=0.0, value=round(comfy, 0), step=500.0, help="Saving that needs no lifestyle disruption (default = your average net savings).")
        use_r = c.checkbox("Apply lifestyle restriction")
        restr = c.number_input(f"Max extra monthly lifestyle cut ({CURRENCY})", min_value=0.0, value=round(cut / 2, 0), step=500.0, help="Optional: caps how far above the comfortable saving you will go.")
        lam = st.slider("Lifestyle disruption weight (λ)", 0.0, 5.0, 1.0, 0.1, help="Penalty for months that need bigger lifestyle cuts: penalty = λ × (extra cut fraction)². Higher λ → A* prefers gentler, longer paths.")
        run = st.form_submit_button("Run A* search", type="primary")
    if run:
        if not name.strip():
            st.error("Goal name is required.")
        else:
            with st.spinner("Running A*…"):
                st.session_state["astar_result"] = astar_plan(name.strip(), target, current, max_save, comfortable, restr if use_r else None, lam)
    res = st.session_state.get("astar_result")
    if not res:
        empty_state("Enter a goal above and run A*.", "You will get the explored states, the chosen path and a roadmap timeline.")
        return
    if res["status"] == "invalid":
        for e in res["errors"]:
            st.error(e)
        return
    init = res["initial"]
    section("Initial state")
    st.markdown(f"Month 0 · savings **{money(init['savings'], 0)}** · remaining **{money(res['target'] - init['savings'], 0)}** · "
                f"h(start) = **{init['h'] if init['h'] is not None else 'n/a'}** month(s) · saving levels {', '.join(money(x, 0) for x in res['levels']) or 'none'}")
    if not res["found"]:
        st.error(res["message"])
        return
    if res["status"] == "done":
        st.success(res["message"])
        return
    st.success(res["message"])
    c = st.columns(4)
    with c[0]:
        kpi_card("Months to goal", str(res["months"]), f"achieved {res['goal_month']}", "jade")
    with c[1]:
        kpi_card("Nodes explored", f"{res['nodes_explored']:,}", f"{res['nodes_generated']:,} generated")
    with c[2]:
        kpi_card("Final cost g(goal)", f"{res['total_cost']:.2f}", f"{res['months']} months + {res['total_penalty']:.2f} disruption")
    with c[3]:
        kpi_card("Initial h(n)", str(init["h"]), "ceil(remaining / max saving)")

    section("Roadmap")
    path = res["path"]
    cards = "".join(f'<div class="m"><div class="n">{p["month_no"]:02d} · {p["month"]}</div><div class="fw-mono" style="margin-top:.2rem">{CURRENCY}{p["saving"]:,.0f}</div>'
                    f'<div class="fw-kpi-sub">Σ {CURRENCY}{p["cumulative"]:,.0f}</div><div style="margin-top:.3rem"><span style="color:{LEVEL_COLOR[p["level"]]}">●</span> <span class="fw-kpi-sub">{p["level"]} disruption</span></div></div>' for p in path)
    st.markdown(f'<div class="fw-tl">{cards}</div>', unsafe_allow_html=True)
    months = [p["month"] for p in path]
    fig = go.Figure([go.Bar(x=months, y=[p["saving"] for p in path], name="Monthly saving", marker_color=[LEVEL_COLOR[p["level"]] for p in path], yaxis="y2", opacity=.55),
                     go.Scatter(x=months, y=[p["cumulative"] for p in path], name="Accumulated savings", mode="lines+markers", line=dict(color=STEEL, width=3))])
    fig.add_hline(y=res["target"], line_dash="dash", line_color=AMBER, annotation_text=f"Target {money(res['target'], 0)}")
    fig.update_layout(title="Savings timeline", yaxis=dict(title="Accumulated"), yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Monthly saving"))
    st.plotly_chart(style_fig(fig, 380), width="stretch")

    section("Selected path")
    st.dataframe([{"Month #": p["month_no"], "Month": p["month"], "Monthly saving": p["saving"], "Accumulated": p["cumulative"], "Remaining target": p["remaining"],
                   "Disruption": p["level"], "g(n)": round(p["g"], 3), "h(n)": p["h"], "f(n)": round(p["f"], 3)} for p in path], width="stretch", hide_index=True)
    st.markdown(f"Goal achievement month: **{res['goal_month']}**")

    section("Search details")
    ex = res["explored"]
    a, b = st.columns(2)
    with a:
        st.markdown(f"**Explored states** ({len(ex)} expanded; first 40 shown)")
        st.dataframe(ex[:40], width="stretch", hide_index=True)
    with b:
        sc = go.Figure(go.Scatter(x=[e["month"] for e in ex], y=[e["savings"] for e in ex], mode="markers", marker=dict(size=7, color=[e["f"] for e in ex], colorscale="Viridis", colorbar=dict(title="f(n)"))))
        sc.update_layout(title="Expanded states (month, savings) coloured by f(n)", xaxis_title="Month", yaxis_title="Savings")
        st.plotly_chart(style_fig(sc, 330, False), width="stretch")
    ai_panel("Heuristic check", f"<div>h(n) = ⌈(target − savings) / {money(res['cap'], 0)}⌉ never overestimates the remaining months, so A* returns the lowest-cost path. "
             f"Verified on the selected path: {'h(n) ≤ true remaining cost at every step ✔' if res['heuristic_admissible_on_path'] else 'violation detected ✖'}.</div>", "A* · admissible")
    st.markdown(" · ".join(term(t) for t in ("A*", "Heuristic")), unsafe_allow_html=True)
