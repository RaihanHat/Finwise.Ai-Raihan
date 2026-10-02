"""Shared constants (no Streamlit imports so models stay UI-independent)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TRANSACTIONS_PATH = DATA_DIR / "transactions.csv"
SAMPLE_PATH = DATA_DIR / "sample_data.csv"
SETTINGS_PATH = DATA_DIR / "settings.json"

CURRENCY = "৳"

INCOME_CATEGORIES = ["Salary", "Freelance", "Investment", "Gift", "Loan", "Debt Repaid"]
EXPENSE_CATEGORIES = ["Housing", "Food", "Transportation", "Entertainment", "Health", "Education"]
ESSENTIAL_CATEGORIES = ["Housing", "Food", "Transportation", "Health", "Education"]
# Loan proceeds are borrowed money, so they are excluded from "net income" used in ratios.
NON_INCOME_FOR_RATIOS = ["Loan"]

DISCRETIONARY_CATEGORIES = {"Entertainment"}
DINING_KEYWORDS = ("dining", "restaurant", "cafe", "café", "takeaway", "fast food", "pizza", "burger")

COLUMNS = ["id", "date", "type", "category", "amount", "description", "is_discretionary"]
UPLOAD_COLUMNS = ["date", "type", "category", "amount", "description"]

# Category Impact Weight used by the Purchase Advisor (0 = essential, 1 = highly discretionary).
CATEGORY_WEIGHT = {
    "Health": 0.10, "Housing": 0.10, "Education": 0.20, "Transportation": 0.35,
    "Food": 0.40, "Shopping": 0.85, "Entertainment": 0.90,
}
PURCHASE_CATEGORIES = list(CATEGORY_WEIGHT.keys())

# Design tokens (frontend spec)
INK, INK2, BONE = "#0D1117", "#161B22", "#F5F3EE"
STEEL, AMBER, JADE, BRICK = "#3D5A80", "#EE964B", "#4C9F70", "#C1445E"
MUTED = "#8B949E"

CATEGORY_COLORS = {
    "Housing": "#5B7DB1", "Food": "#EE964B", "Transportation": "#8F7BC6",
    "Entertainment": "#C1445E", "Health": "#4C9F70", "Education": "#D4B24C",
    "Salary": "#4C9F70", "Freelance": "#5BA8A0", "Investment": "#7FB069",
    "Gift": "#D68FB0", "Loan": "#B08968", "Debt Repaid": "#9AA5B1", "Shopping": "#E07A5F",
}

DEFAULT_SETTINGS = {
    "opening_balance": 0.0,
    "monthly_debt_payment": 0.0,
    "emergency_reserve": 0.0,
    "emergency_target_months": 3.0,
    "savings_target_ratio": 0.20,
    "thr_savings_min": 0.10,
    "thr_savings_healthy": 0.20,
    "thr_discretionary_max": 0.20,
    "thr_burn_income_max": 0.90,
    "thr_debt_burden_max": 0.30,
    "synthetic_samples": 800,
    "random_seed": 42,
}
