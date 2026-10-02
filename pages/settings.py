import streamlit as st

from utils import data_loader as dl
from utils.constants import CURRENCY, DEFAULT_SETTINGS
from utils.helpers import page_header, section


def render():
    page_header("Settings", "Thresholds and personal inputs used by the rule engine, budget defaults and models.", "10 · Settings")
    s = dl.load_settings()
    with st.form("settings"):
        section("Personal financial inputs")
        a, b, c = st.columns(3)
        s["opening_balance"] = a.number_input(f"Opening balance ({CURRENCY})", value=float(s["opening_balance"]), step=1000.0, help="Balance before the first logged transaction.")
        s["monthly_debt_payment"] = b.number_input(f"Monthly debt payment ({CURRENCY})", min_value=0.0, value=float(s["monthly_debt_payment"]), step=500.0)
        s["emergency_reserve"] = c.number_input(f"Current emergency reserve ({CURRENCY})", min_value=0.0, value=float(s["emergency_reserve"]), step=1000.0)
        a, b = st.columns(2)
        s["emergency_target_months"] = a.number_input("Emergency target (months of essential expenses)", min_value=0.5, max_value=24.0, value=float(s["emergency_target_months"]), step=0.5)
        s["savings_target_ratio"] = b.slider("Savings target (share of income)", 0.0, 0.6, float(s["savings_target_ratio"]), 0.01)
        section("Rule thresholds")
        a, b, c = st.columns(3)
        s["thr_savings_min"] = a.slider("R01 min savings ratio", 0.0, 0.5, float(s["thr_savings_min"]), 0.01)
        s["thr_savings_healthy"] = b.slider("R06 healthy savings ratio", 0.05, 0.8, float(s["thr_savings_healthy"]), 0.01)
        s["thr_discretionary_max"] = c.slider("R02 max discretionary ratio", 0.05, 0.8, float(s["thr_discretionary_max"]), 0.01)
        a, b = st.columns(2)
        s["thr_burn_income_max"] = a.slider("R03 max burn rate / income", 0.3, 2.0, float(s["thr_burn_income_max"]), 0.05)
        s["thr_debt_burden_max"] = b.slider("R05 max debt payment / income", 0.05, 0.8, float(s["thr_debt_burden_max"]), 0.01)
        section("Synthetic training data (Purchase Advisor)")
        a, b = st.columns(2)
        s["synthetic_samples"] = int(a.number_input("Synthetic samples", min_value=30, max_value=5000, value=int(s["synthetic_samples"]), step=50, help="Size of the SYNTHETIC/DEMO dataset used to train Logistic Regression and KNN."))
        s["random_seed"] = int(b.number_input("Random seed", min_value=0, value=int(s["random_seed"]), step=1))
        if st.form_submit_button("Save settings", type="primary"):
            if s["thr_savings_healthy"] < s["thr_savings_min"]:
                st.error("Healthy savings threshold must be ≥ the minimum savings threshold.")
            else:
                dl.save_settings(s)
                st.success("Settings saved. All modules will use the new values.")
    section("Data management")
    a, b, c = st.columns(3)
    raw = dl.load_transactions()
    a.download_button("Export transactions CSV", raw.assign(date=raw["date"].dt.strftime("%Y-%m-%d")).to_csv(index=False), "finwise_transactions.csv", "text/csv", disabled=raw.empty)
    if b.button("Load demo sample data", help="Replaces current transactions with the bundled DEMO dataset (data/sample_data.csv)."):
        st.session_state["flash_settings"] = f"Loaded {dl.load_sample()} demo transactions."
        st.rerun()
    with c:
        sure = st.checkbox("I understand this deletes all transactions")
        if st.button("Clear all transactions", disabled=not sure):
            dl.clear_all()
            st.session_state["flash_settings"] = "All transactions cleared."
            st.rerun()
    if "flash_settings" in st.session_state:
        st.success(st.session_state.pop("flash_settings"))
    if st.button("Reset settings to defaults"):
        dl.save_settings(dict(DEFAULT_SETTINGS))
        st.rerun()
