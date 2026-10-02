import html

import streamlit as st

from models.guidance import build_report
from models.spending_clustering import fit_profile
from utils.calculations import load_state, rules_for, total_balance
from utils.helpers import ai_panel, chip, empty_state, kpi_card, money, no_data_gate, page_header, section

SRC_TONE = {"ML prediction": "steel", "K-Means (ML)": "steel", "Rule-based": "amber", "CSP optimization": "jade", "A* planning": "brick", "Financial metric": "muted"}


def _src(s):
    return chip(s, SRC_TONE.get(s, "muted"))


def _markdown(rep) -> str:
    L = [f"# FinWise AI Financial Guidance Report ({rep['month']})", "", "## Financial Summary"]
    L += [f"- {r['label']}: {r['value']} [{r['source']}]" for r in rep["summary"]]
    L += ["", "## Spending Profile", rep["profile"].get("profile", rep["profile"].get("reason", "")) + (f" - {rep['profile'].get('evidence', '')}" if rep["profile"]["available"] else "")]
    L += ["", "## Budget Recommendations"]
    L += ([f"- {k}: {v:,.0f}" for k, v in rep["budget"]["allocation"].items()] if rep["budget"]["available"] else [rep["budget"].get("reason", "")])
    L += ["", "## Purchase Risk"]
    p = rep["purchase"]
    L += ([f"{p['item']}: LR={p['lr']['predicted']} ({p['lr']['probabilities']})" + (f"; KNN={p['knn']['predicted']}" if p["knn"] else "") + " [synthetic-data model]"] if p["available"] else [p["reason"]])
    L += ["", "## Saving Recommendations"] + [f"- [{r['rule_id']}] {r['recommendation']}" for r in rep["saving_recs"]]
    L += ["", "## Goal Roadmap"]
    g = rep["goal"]
    L += ([f"{g['plan']['goal']}: {g['plan']['months']} months, finish {g['plan']['goal_month']}"] if g["available"] else [g["reason"]])
    L += ["", "## Top 3 Actions"] + [f"{i}. {a['action']} [{a['source']}] - {a['evidence']}" for i, a in enumerate(rep["top_actions"], 1)]
    return "\n".join(L)


def render():
    page_header("Integrated Financial Guidance", "Every recommendation below is traced to a real model, rule, metric or optimisation result.", "09 · Guidance")
    state = load_state()
    if no_data_gate(state):
        return
    month = state["months"][-1]
    mm = {k: v for k, v in state["pipe"]["monthly"].iloc[-1].items()}
    mm = {k: (None if v != v else v) for k, v in mm.items()}
    _, rules = rules_for(state, month)
    prof = fit_profile(state["pipe"]["clean"], seed=int(state["settings"]["random_seed"]))
    rep = build_report(mm, month, total_balance(state), prof, st.session_state.get("purchase_result"), st.session_state.get("csp_result"), rules, st.session_state.get("astar_result"))
    st.markdown(" ".join([_src("ML prediction"), _src("Rule-based"), _src("CSP optimization"), _src("A* planning"), _src("Financial metric")]) + " &nbsp; <span style='color:#8B949E;font-size:.8rem'>source labels</span>", unsafe_allow_html=True)

    section(f"Financial summary · {month}")
    cols = st.columns(4)
    for i, r in enumerate(rep["summary"]):
        with cols[i % 4]:
            kpi_card(r["label"], r["value"], r["source"])

    section("Spending profile")
    p = rep["profile"]
    if p["available"]:
        ai_panel(p["profile"], f"<div>{html.escape(p['evidence'])}</div><div>{html.escape(p['drift'])}</div>", p["source"])
    else:
        empty_state(p["reason"])

    section("Budget recommendations")
    b = rep["budget"]
    if b["available"]:
        st.dataframe([{"Category": k, "Allocation": round(v, 2)} for k, v in b["allocation"].items()], width="stretch", hide_index=True)
        st.markdown(_src(b["source"]) + f" &nbsp; {html.escape(b['evidence'])} Remaining unallocated: {money(b['remaining'])}.", unsafe_allow_html=True)
    elif b.get("infeasible"):
        st.error("CSP infeasible: " + b["reason"])
    else:
        empty_state(b["reason"])

    section("Purchase risk")
    pr = rep["purchase"]
    if pr["available"]:
        for r in [pr["lr"]] + ([pr["knn"]] if pr["knn"] else []):
            role = "primary" if r["model"] == "Logistic Regression" else "comparison"
            st.markdown(f"{_src('ML prediction')} **{html.escape(pr['item'])}** ({money(pr['price'])}) · {r['model']} ({role}): **{r['predicted']}** · " +
                        ", ".join(f"{k} {v:.0%}" for k, v in r["probabilities"].items()) + f" · risk {r['risk_level']}", unsafe_allow_html=True)
        st.caption(pr["note"] + ("" if pr["agree"] else " The models disagree - treat the result with extra caution."))
    else:
        empty_state(pr["reason"])

    section("Saving recommendations")
    if rep["saving_recs"]:
        for r in rep["saving_recs"]:
            st.markdown(f"{_src('Rule-based')} **{r['rule_id']}** · {html.escape(r['recommendation'])} <span style='color:#8B949E'>({html.escape(r['condition'])}; current {r['current']})</span>", unsafe_allow_html=True)
    else:
        st.success("No saving rule was triggered.")

    section("Goal roadmap")
    g = rep["goal"]
    if g["available"]:
        pl = g["plan"]
        st.markdown(f"{_src(g['source'])} **{html.escape(pl['goal'])}** — {money(pl['target'], 0)} in **{pl['months']} month(s)**, finishing **{pl['goal_month']}**. "
                    f"First month saving {money(pl['path'][0]['saving'], 0)}; A* explored {pl['nodes_explored']} states.", unsafe_allow_html=True)
        st.dataframe([{"Month": x["month"], "Saving": x["saving"], "Accumulated": x["cumulative"], "Disruption": x["level"]} for x in pl["path"]], width="stretch", hide_index=True)
    else:
        empty_state(g["reason"])

    section("Top 3 actions")
    if rep["top_actions"]:
        for i, a in enumerate(rep["top_actions"], 1):
            st.markdown(f'<div class="fw-card" style="margin-bottom:.6rem"><span class="fw-chip steel">{i}</span> &nbsp; <b>{html.escape(a["action"])}</b><br>'
                        f'{_src(a["source"])} <span style="color:#8B949E;font-size:.85rem">Evidence: {html.escape(a["evidence"])}</span></div>', unsafe_allow_html=True)
    else:
        empty_state("No actions yet.", "Run the Purchase Advisor, Budget Optimizer or Goal Planner to add model-based actions.")
    st.download_button("Download report (Markdown)", _markdown(rep), "finwise_guidance.md", "text/markdown")
