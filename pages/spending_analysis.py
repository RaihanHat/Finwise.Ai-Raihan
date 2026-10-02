import plotly.graph_objects as go
import streamlit as st

from models import spending_clustering as sc
from utils.calculations import category_totals, load_state
from utils.constants import AMBER, BRICK, CATEGORY_COLORS, JADE, STEEL
from utils.helpers import ai_panel, chip, empty_state, kpi_card, money, no_data_gate, page_header, pct, section, style_fig, term

PROFILE_COLOR = {"Disciplined / Frugal": JADE, "Balanced / Moderate": AMBER, "Impulsive / High-Risk": BRICK}
PROFILE_TONE = {"Disciplined / Frugal": "jade", "Balanced / Moderate": "amber", "Impulsive / High-Risk": "brick"}


def _fmt(f, v):
    return pct(v) if f in ("savings_ratio", "discretionary_ratio") else f"{v:,.1f}" if f == "tx_frequency" else money(v, 0)


def render():
    page_header("Spending Analysis", "K-Means groups your spending windows into behavioural profiles; the latest window is assigned by the trained model.", "04 · Analysis")
    state = load_state()
    if no_data_gate(state):
        return
    clean = state["pipe"]["clean"]
    with st.spinner("Fitting K-Means…"):
        prof = sc.fit_profile(clean, seed=int(state["settings"]["random_seed"]))
    st.session_state["profile_result"] = prof
    if not prof["ready"]:
        empty_state(prof["reason"], "Keep logging transactions, or load the demo sample data in Settings.")
        st.caption("Insufficient data for reliable clustering.")
        return

    cur, drift = prof["current_profile"], prof["drift"]
    ai_panel("Spending profile", f'<div>{chip(cur, PROFILE_TONE[cur])} &nbsp; assigned by <b>KMeans.predict</b> on the latest window '
             f'(cluster {prof["current_cluster"]}).</div>', "K-Means · k=3")
    c = st.columns(4)
    with c[0]:
        kpi_card("Current profile", cur, f"Cluster {prof['current_cluster']}", PROFILE_TONE[cur])
    with c[1]:
        kpi_card("Analysis windows", str(prof["n_windows"]), prof["window_desc"])
    with c[2]:
        kpi_card("Silhouette", f"{prof['silhouette']:.3f}" if prof["silhouette"] is not None else "n/a", "cluster separation")
    with c[3]:
        kpi_card("Inertia", f"{prof['inertia']:.3f}", "within-cluster sum of squares")

    section("Cluster visualization", "2-D PCA projection of the MinMax-scaled features. ★ = your latest window.")
    w = prof["windows_df"]
    fig = go.Figure()
    for name in sc.PROFILE_NAMES:
        mask = (w["profile"] == name).values
        if mask.any():
            fig.add_trace(go.Scatter(x=prof["coords"][mask, 0], y=prof["coords"][mask, 1], mode="markers", name=name, marker=dict(size=9, color=PROFILE_COLOR[name], opacity=.8),
                                     text=w.loc[mask, "window_end"].dt.strftime("%Y-%m-%d"), hovertemplate="window ending %{text}<extra>" + name + "</extra>"))
    fig.add_trace(go.Scatter(x=prof["centroid_xy"][:, 0], y=prof["centroid_xy"][:, 1], mode="markers", name="Centroids", marker=dict(symbol="x", size=14, color="white", line=dict(width=2))))
    fig.add_trace(go.Scatter(x=[prof["current_xy"][0]], y=[prof["current_xy"][1]], mode="markers", name="You (latest)", marker=dict(symbol="star", size=20, color="#F5F3EE", line=dict(width=2, color=PROFILE_COLOR[cur]))))
    fig.update_layout(xaxis_title=f"PC1 ({prof['pca_var'][0]:.0%} var)", yaxis_title=f"PC2 ({prof['pca_var'][1]:.0%} var)")
    st.plotly_chart(style_fig(fig, 420), width="stretch")

    section("Features used by K-Means", "Six features per window; MinMax-scaled before clustering.")
    feats = prof["current_features"]
    rows = [{"Feature": sc.FEATURE_LABELS[f], "Your value": _fmt(f, feats[f]), "Scaled [0,1]": round(prof["current_scaled"][f], 3)} for f in sc.FEATURES]
    a, b = st.columns(2)
    a.dataframe(rows, width="stretch", hide_index=True)
    dist = prof["distances"]
    fig = go.Figure(go.Bar(x=list(dist.values()), y=list(dist), orientation="h", marker_color=[PROFILE_COLOR[k] for k in dist]))
    fig.update_layout(title="Distance to each centroid (scaled space)")
    b.plotly_chart(style_fig(fig, 240, False), width="stretch")

    section("Cluster characteristics", "Centroids converted back to original units. Names come from ranking the centroids' composite risk score.")
    cen = prof["centroids"].copy()
    disp = cen.copy()
    for f in sc.FEATURES:
        disp[f] = [_fmt(f, v) for v in cen[f]]
    disp = disp.rename(columns={**sc.FEATURE_LABELS, "windows": "Windows", "risk_score": "Risk score"})
    disp["Risk score"] = disp["Risk score"].round(3)
    st.dataframe(disp, width="stretch", hide_index=True)
    dist_fig = go.Figure(go.Bar(x=list(prof["distribution"]), y=list(prof["distribution"].values()), marker_color=[PROFILE_COLOR[k] for k in prof["distribution"]]))
    dist_fig.update_layout(title="Windows per profile")
    st.plotly_chart(style_fig(dist_fig, 240, False), width="stretch")

    section("Risk drift")
    if drift["available"]:
        tone = "brick" if drift["direction"] == "toward High-Risk" else "jade" if drift["direction"] == "toward Disciplined" else "steel"
        st.markdown(f'{chip("Drift: " + drift["direction"], tone)} &nbsp; recent mean risk rank {drift["recent"]:.2f} vs previous {drift["previous"]:.2f} '
                    f'(last {drift["m"]} windows vs the {drift["m"]} before; rank 0=Disciplined, 2=High-Risk).', unsafe_allow_html=True)
        tl = go.Figure(go.Scatter(x=w["window_end"], y=w["risk_rank"], mode="lines+markers", line=dict(color=AMBER, shape="hv"), text=w["profile"]))
        tl.update_layout(title="Profile over time (0 = Disciplined, 2 = High-Risk)", yaxis=dict(tickvals=[0, 1, 2]))
        st.plotly_chart(style_fig(tl, 260, False), width="stretch")
    else:
        empty_state("Risk drift needs at least 8 analysis windows.", f"{prof['n_windows']} available.")

    section("Why this profile?")
    cc = prof["cluster_of_name"][cur]
    cent_s = prof["centroids_scaled"].set_index("profile").loc[cur]
    diffs = sorted(((sc.FEATURE_LABELS[f], prof["current_scaled"][f] - cent_s[f]) for f in sc.FEATURES), key=lambda t: abs(t[1]))
    st.markdown(f"- Your latest window is closest to the **{cur}** centroid (distance {dist[cur]:.3f}); other centroids are at "
                + ", ".join(f"{k}: {v:.3f}" for k, v in dist.items() if k != cur) + ".\n"
                f"- Closest-matching features: {', '.join(d[0] for d in diffs[:2])}; largest gap: {diffs[-1][0]} ({diffs[-1][1]:+.2f} scaled).")
    with st.expander("Model information"):
        st.markdown(f"- Algorithm: {term('K-Means')} (scikit-learn), k=3, n_init=10, random_state={prof['seed']}\n"
                    f"- Scaling: {term('MinMax')} fitted on all windows\n- Samples: {prof['n_windows']} overlapping {prof['window_desc']}\n"
                    f"- {term('Inertia')}: {prof['inertia']:.3f} · {term('Silhouette')}: {prof['silhouette']:.3f}\n"
                    "- Cluster names are derived from the centroid composite risk (low savings, high discretionary ratio, spend variance, burn rate); assignment of the current window uses `KMeans.predict`, not if/else rules.\n"
                    "- Limitation: windows overlap and come from one user, so clusters describe *your own* spending regimes, not a population.", unsafe_allow_html=True)
