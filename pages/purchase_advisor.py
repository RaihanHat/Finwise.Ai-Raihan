import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from models import purchase_classifier as pc
from utils import data_loader as dl
from utils.calculations import load_state, total_balance
from utils.constants import AMBER, BRICK, CURRENCY, JADE, PURCHASE_CATEGORIES, STEEL
from utils.helpers import ai_panel, confusion_heatmap, disclaimer, empty_state, money, page_header, pct, section, style_fig, term, tone_chip

TONE_COLOR = {"Approved": JADE, "Caution": AMBER, "Rejected": BRICK}
LR, KNN = "Logistic Regression", "KNN"


@st.cache_resource(show_spinner=False)
def get_bundle(n: int, seed: int, k: int):
    return pc.train_models(n=n, seed=seed, k=k)


def metrics_table(bundle) -> pd.DataFrame:
    rows = []
    for name, role in ((LR, "Primary"), (KNN, "Comparison / supporting")):
        m = bundle["metrics"][name]
        rows.append({"Model": name, "Role": role, "Accuracy": m["accuracy"], "Precision (weighted)": m["precision"], "Recall (weighted)": m["recall"], "F1 (weighted)": m["f1"], "Test samples": m["n_test"]})
    return pd.DataFrame(rows)


def show_metrics(bundle, key=""):
    st.dataframe(metrics_table(bundle).style.format({c: "{:.3f}" for c in ["Accuracy", "Precision (weighted)", "Recall (weighted)", "F1 (weighted)"]}), width="stretch", hide_index=True)
    a, b = st.columns(2)
    a.plotly_chart(confusion_heatmap(bundle["metrics"][LR]["confusion"], pc.LABELS, "Logistic Regression · confusion matrix"), width="stretch", key=f"cm_lr{key}")
    b.plotly_chart(confusion_heatmap(bundle["metrics"][KNN]["confusion"], pc.LABELS, f"KNN (k={bundle['k']}) · confusion matrix"), width="stretch", key=f"cm_knn{key}")
    with st.expander("Per-class precision / recall / F1"):
        a, b = st.columns(2)
        a.markdown("**Logistic Regression**")
        a.dataframe(bundle["metrics"][LR]["per_class"].round(3), width="stretch", hide_index=True)
        b.markdown("**KNN**")
        b.dataframe(bundle["metrics"][KNN]["per_class"].round(3), width="stretch", hide_index=True)


def _result_block(res: dict, bundle, feats: dict, primary: bool):
    role = "Primary model" if primary else "Comparison / supporting model"
    st.markdown(f"#### {res['model']} &nbsp; <span style='font-size:.75rem;color:#8B949E'>{role}</span>", unsafe_allow_html=True)
    pred = res["predicted"]
    st.markdown(f'{tone_chip(pred)} &nbsp; predicted class &nbsp;·&nbsp; risk level {tone_chip(res["risk_level"])} &nbsp;<span class="fw-mono" style="color:#8B949E">(expected risk {res["expected_risk"]:.2f})</span>', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, lab in zip(cols, pc.LABELS):
        p = res["probabilities"][lab]
        col.markdown(f'<div class="fw-card"><div class="fw-kpi-label">{lab}</div><div class="fw-prob" style="color:{TONE_COLOR[lab]}">{p * 100:.1f}%</div></div>', unsafe_allow_html=True)
    if res["model"] == LR:
        ex = pc.explain_lr(bundle, feats, pred)
        fig = go.Figure(go.Bar(x=ex["contribution"], y=ex["feature"], orientation="h", marker_color=[BRICK if v > 0 else JADE for v in ex["contribution"]]))
        fig.update_layout(title=f"Feature contribution to '{pred}' (coefficient × standardised value)", yaxis=dict(autorange="reversed"))
        st.plotly_chart(style_fig(fig, 260, False), width="stretch", key="lr_contrib")
        top = ex.reindex(ex["contribution"].abs().sort_values(ascending=False).index).iloc[0]
        st.markdown(f"**Explanation:** the strongest driver of the *{pred}* score is **{top['feature']}** "
                    f"(value {top['value']:.2f}, contribution {top['contribution']:+.2f} log-odds). Bars to the right push toward *{pred}*, bars to the left push away from it.")
    else:
        nb = pc.neighbours(bundle, feats)
        votes = nb["label"].value_counts().to_dict()
        st.markdown(f"**Explanation:** the {len(nb)} nearest synthetic training examples (MinMax-scaled distance) voted " + ", ".join(f"{k}: {v}" for k, v in votes.items()) + ".")
        st.dataframe(nb.rename(columns=pc.FEATURE_LABELS).round(3), width="stretch", hide_index=True)


def render():
    page_header("Purchase Advisor", "Can I buy this? A supervised classifier estimates class probabilities for a proposed purchase.", "05 · Purchase Advisor")
    state = load_state()
    s = state["settings"]
    disclaimer(pc.SYNTHETIC_NOTICE + " Educational demo - do not use for real-world financial decisions.")
    try:
        with st.spinner("Training Logistic Regression and KNN on synthetic data…"):
            bundle = get_bundle(int(s["synthetic_samples"]), int(s["random_seed"]), 5)
    except ValueError as e:
        empty_state(str(e), "Increase 'Synthetic samples' in Settings (≥ 30).")
        return
    for n in bundle["notes"]:
        st.warning(n)

    default_bal = max(total_balance(state), 0.0) if state["ready"] else 0.0
    section("Proposed purchase")
    with st.form("purchase_form"):
        a, b, c = st.columns(3)
        item = a.text_input("Product name", placeholder="e.g. Laptop")
        price = b.number_input(f"Price ({CURRENCY})", min_value=0.0, value=0.0, step=500.0, format="%.2f")
        balance = c.number_input(f"Current liquid balance ({CURRENCY})", value=float(round(default_bal, 2)), step=500.0, format="%.2f", help="Prefilled from your transactions; editable.")
        a, b, c = st.columns(3)
        days = a.number_input("Days remaining in month", min_value=0, max_value=31, value=dl.days_left_in_month())
        urg = b.slider("Goal urgency (1-5)", 1, 5, 3, help="How urgent are your current savings goals? 5 = very urgent.")
        cat = c.selectbox("Category", PURCHASE_CATEGORIES, help="Category impact weight: 0 = essential, 1 = highly discretionary.")
        view = st.radio("Model view", ["Logistic Regression (primary)", "KNN (comparison)", "Compare both"], index=2, horizontal=True)
        go_btn = st.form_submit_button("Evaluate purchase", type="primary")
    if go_btn:
        errs = []
        if not item.strip():
            errs.append("Product name is required.")
        if price <= 0:
            errs.append("Price must be greater than 0.")
        if balance <= 0:
            st.warning("Balance is zero or negative: the cost/balance ratio is treated as extremely high (5.0).")
        if errs:
            for e in errs:
                st.error(e)
        else:
            feats = pc.build_features(price, balance, days, urg, cat)
            st.session_state["purchase_result"] = {"item": item.strip(), "price": price, "balance": balance, "days": days, "urgency": urg, "category": cat, "feats": feats,
                                                   "lr": pc.predict(bundle, LR, feats), "knn": pc.predict(bundle, KNN, feats), "view": view}
    res = st.session_state.get("purchase_result")
    if res:
        section(f"Verdict · {res['item']} ({money(res['price'])})")
        feats = res["feats"]
        st.markdown("**Input features**")
        st.dataframe(pd.DataFrame({"Feature": [pc.FEATURE_LABELS[f] for f in pc.FEATURES], "Value": [round(feats[f], 4) for f in pc.FEATURES]}), width="stretch", hide_index=True)
        view = res["view"] if not go_btn else view
        if view.startswith("Logistic") or view.startswith("Compare"):
            _result_block(res["lr"], bundle, feats, True)
        if view.startswith("KNN") or view.startswith("Compare"):
            _result_block(res["knn"], bundle, feats, False)
        if view.startswith("Compare"):
            cmp = pd.DataFrame([{"Model": r["model"], "Predicted": r["predicted"], **{f"P({k})": f"{v:.1%}" for k, v in r["probabilities"].items()}, "Risk level": r["risk_level"]} for r in (res["lr"], res["knn"])])
            st.markdown("**Prediction comparison**")
            st.dataframe(cmp, width="stretch", hide_index=True)
            if res["lr"]["predicted"] != res["knn"]["predicted"]:
                st.info("The two models disagree. Logistic Regression is the primary model; KNN is a supporting check based on similar synthetic examples.")
        if res["lr"]["predicted"] != "Approved":
            wi = pc.what_if(bundle, feats, res["price"], res["balance"], res["days"], res["urgency"], res["category"])
            tips = []
            if wi["price_for_approved"] is not None:
                tips.append(f"Reduce the price to about {money(wi['price_for_approved'], 0)} (−{money(res['price'] - wi['price_for_approved'], 0)}) for the model to predict *Approved*.")
            if wi["balance_for_approved"] is not None and wi["balance_for_approved"] > max(res["balance"], 1):
                tips.append(f"Or wait until your liquid balance is about {money(wi['balance_for_approved'], 0)}.")
            ai_panel("If Caution / Rejected: model-driven alternatives", "<br>".join(html.escape(t) for t in tips) if tips else "No price or balance change within the tested range flips the prediction.", "What-if · LR")
    else:
        empty_state("Enter a purchase above and press Evaluate.", "The model returns class probabilities, a risk level and an explanation.")

    section("Model evaluation", "Logistic Regression is the PRIMARY classifier; KNN (MinMax-normalised) is the COMPARISON / supporting classifier. Metrics are on a held-out SYNTHETIC test set.")
    show_metrics(bundle)
    st.caption(f"Synthetic training set: {bundle['n']} samples · class counts {bundle['class_counts']} · 75/25 train/test split · {term('Precision')}, {term('Recall')}, {term('F1')} are weighted averages.")
    with st.expander("Synthetic/demo training data preview"):
        st.dataframe(bundle["data"].head(20).round(3), width="stretch", hide_index=True)
