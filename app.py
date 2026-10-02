"""FinWise AI - Intelligent Personal Financial Optimizer & Purchase Advisor.
Run:  streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="FinWise AI", page_icon=":material/insights:", layout="wide", initial_sidebar_state="expanded")

from pages import (budget_optimizer, dashboard, data_processing, financial_guidance, goal_planner, model_evaluation,  # noqa: E402
                   purchase_advisor, rule_advisor, settings, spending_analysis, transactions)
from utils.constants import ROOT  # noqa: E402
from utils.helpers import inject_css  # noqa: E402

inject_css()
try:
    st.logo(str(ROOT / "assets" / "logo.svg"), size="large")
except Exception:  # older Streamlit versions
    pass

PAGES = [
    st.Page(dashboard.render, title="Dashboard", icon=":material/dashboard:", url_path="dashboard", default=True),
    st.Page(transactions.render, title="Income & Expenses", icon=":material/receipt_long:", url_path="transactions"),
    st.Page(data_processing.render, title="Data Processing", icon=":material/tune:", url_path="data-processing"),
    st.Page(spending_analysis.render, title="Spending Analysis", icon=":material/bubble_chart:", url_path="spending-analysis"),
    st.Page(purchase_advisor.render, title="Purchase Advisor", icon=":material/shopping_cart:", url_path="purchase-advisor"),
    st.Page(budget_optimizer.render, title="Budget Optimizer", icon=":material/account_tree:", url_path="budget-optimizer"),
    st.Page(rule_advisor.render, title="Rule-Based Advisor", icon=":material/rule:", url_path="rule-advisor"),
    st.Page(goal_planner.render, title="Goal Planner", icon=":material/flag:", url_path="goal-planner"),
    st.Page(financial_guidance.render, title="Financial Guidance", icon=":material/lightbulb:", url_path="financial-guidance"),
    st.Page(model_evaluation.render, title="Model Evaluation", icon=":material/analytics:", url_path="model-evaluation"),
    st.Page(settings.render, title="Settings", icon=":material/settings:", url_path="settings"),
]
nav = st.navigation(PAGES)
with st.sidebar:
    st.caption("FinWise AI · Group 10 · CSE 3812 AI Lab\n\nEducational prototype. Not financial advice.")
nav.run()
