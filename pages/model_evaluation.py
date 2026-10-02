import plotly.graph_objects as go
import streamlit as st

from models import spending_clustering as sc
from pages.purchase_advisor import get_bundle, show_metrics
from utils.calculations import load_state
from utils.helpers import disclaimer, empty_state, kpi_card, page_header, section, style_fig, term
from models import purchase_classifier as pc
from utils.constants import AMBER, BRICK, JADE, STEEL

INSUFFICIENT = "Insufficient data for reliable evaluation."


def render():
    page_header("Model Evaluation", "Metrics computed live for every model. Nothing here is hard-coded.", "11 · Evaluation")
    state = load_state()
    s = state["settings"]

    section("Logistic Regression & KNN", "Trained on SYNTHETIC / DEMO data (labelled).")
    disclaimer(pc.SYNTHETIC_NOTICE)
    try:
        bundle = get_bundle(int(s["synthetic_samples"]), int(s["random_seed"]), 5)
        for n in bundle["notes"]:
            st.warning(n)
        if bundle["metrics"]["Logistic Regression"]["n_test"] < 10:
            st.warning(INSUFFICIENT)
        show_metrics(bundle, "_eval")
    except ValueError as e:
        empty_state(INSUFFICIENT, str(e))

    section("K-Means spending profile", "Fitted on your own transactions.")
    if not state["ready"]:
        empty_state(INSUFFICIENT, "No transactions available.")
    else:
        prof = sc.fit_profile(state["pipe"]["clean"], seed=int(s["random_seed"]))
        if not prof["ready"]:
            empty_state(INSUFFICIENT, prof["reason"])
        else:
            c = st.columns(3)
            with c[0]:
                kpi_card("Inertia", f"{prof['inertia']:.3f}", "lower = tighter")
            with c[1]:
                kpi_card("Silhouette", f"{prof['silhouette']:.3f}" if prof["silhouette"] is not None else "n/a", "−1 … 1")
            with c[2]:
                kpi_card("Windows", str(prof["n_windows"]), prof["window_desc"])
            col = {"Disciplined / Frugal": JADE, "Balanced / Moderate": AMBER, "Impulsive / High-Risk": BRICK}
            a, b = st.columns(2)
            f1 = go.Figure(go.Bar(x=list(prof["distribution"]), y=list(prof["distribution"].values()), marker_color=[col[k] for k in prof["distribution"]]))
            f1.update_layout(title="Cluster distribution")
            a.plotly_chart(style_fig(f1, 300, False), width="stretch")
            w = prof["windows_df"]
            f2 = go.Figure()
            for name, color in col.items():
                m = (w["profile"] == name).values
                if m.any():
                    f2.add_trace(go.Scatter(x=prof["coords"][m, 0], y=prof["coords"][m, 1], mode="markers", name=name, marker=dict(color=color, size=8)))
            f2.add_trace(go.Scatter(x=prof["centroid_xy"][:, 0], y=prof["centroid_xy"][:, 1], mode="markers", name="Centroids", marker=dict(symbol="x", size=12, color="white")))
            f2.update_layout(title="Cluster visualization (PCA)")
            b.plotly_chart(style_fig(f2, 300), width="stretch")
            if prof["n_windows"] < 12:
                st.caption("Few windows - clustering metrics should be read cautiously.")

    section("CSP Budget Optimizer")
    csp = st.session_state.get("csp_result")
    if not csp or csp.get("status") == "invalid":
        empty_state(INSUFFICIENT, "Run the Budget Optimizer first.")
    else:
        stt = csp["stats"]
        c = st.columns(4)
        with c[0]:
            kpi_card("Constraints", str(len(csp["constraints"])), "3 hard · 3 soft")
        with c[1]:
            kpi_card("Status", "Feasible" if csp["feasible"] else "Infeasible", csp["status"], "jade" if csp["feasible"] else "brick")
        with c[2]:
            kpi_card("Nodes / backtracks", f"{stt['nodes']:,} / {stt['backtracks']:,}", f"FC prunings {stt['fc_prunings']:,}")
        with c[3]:
            kpi_card("Solutions found", f"{stt['solutions']:,}", f"bound prunes {stt['bound_prunes']:,}")
        st.dataframe([{"ID": x["id"], "Type": x["type"], "Constraint": x["constraint"], "Status": x["status"], "Detail": x["detail"]} for x in csp["constraints"]], width="stretch", hide_index=True)

    section("A* Goal Planner")
    a_ = st.session_state.get("astar_result")
    if not a_ or not a_.get("found") or a_.get("status") == "done":
        empty_state(INSUFFICIENT, "Run the Goal Planner first." if not a_ else a_.get("message", ""))
    else:
        c = st.columns(4)
        with c[0]:
            kpi_card("Nodes explored", f"{a_['nodes_explored']:,}", f"{a_['nodes_generated']:,} generated")
        with c[1]:
            kpi_card("Path length", f"{len(a_['path'])} months", a_["goal_month"])
        with c[2]:
            kpi_card("Final cost", f"{a_['total_cost']:.3f}", "g(goal)")
        with c[3]:
            kpi_card("Admissible h", "Yes" if a_["heuristic_admissible_on_path"] else "No", "checked on path", "jade" if a_["heuristic_admissible_on_path"] else "brick")
        fig = go.Figure([go.Scatter(x=[p["month_no"] for p in a_["path"]], y=[p["h"] for p in a_["path"]], name="h(n)", mode="lines+markers", line=dict(color=STEEL)),
                         go.Scatter(x=[p["month_no"] for p in a_["path"]], y=[p["true_remaining_cost"] for p in a_["path"]], name="True remaining cost", mode="lines+markers", line=dict(color=AMBER, dash="dot"))])
        fig.update_layout(title="Heuristic h(n) vs true remaining cost along the path", xaxis_title="Month")
        st.plotly_chart(style_fig(fig, 300), width="stretch")
        st.dataframe([{"Month": p["month"], "g(n)": round(p["g"], 3), "h(n)": p["h"], "f(n)": round(p["f"], 3)} for p in a_["path"]], width="stretch", hide_index=True)
    st.markdown(" · ".join(term(t) for t in ("Precision", "Recall", "F1", "Confusion Matrix", "Silhouette", "Inertia")), unsafe_allow_html=True)
