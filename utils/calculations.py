"""UI-independent calculations shared by pages: one place that turns the central CSV into model-ready state."""
import pandas as pd

from models.preprocessing import run_pipeline
from models import rule_engine
from utils.data_loader import load_settings, load_transactions
from utils.constants import ESSENTIAL_CATEGORIES


def load_state() -> dict:
    """Single source of truth: raw CSV -> preprocessing pipeline -> metrics (+ settings)."""
    raw = load_transactions()
    settings = load_settings()
    pipe = run_pipeline(raw, settings["opening_balance"])
    state = {"raw": raw, "settings": settings, "pipe": pipe, "ready": pipe["ready"]}
    if pipe["ready"]:
        m = pipe["monthly"]
        state["months"] = list(m["month_key"])
        state["essential_avg"] = float(m["essential"].mean())
    return state


def month_metrics(state: dict, month_key: str | None = None) -> dict:
    m = state["pipe"]["monthly"]
    row = m.iloc[-1] if month_key is None else m[m["month_key"] == month_key].iloc[0]
    return {k: (None if pd.isna(v) else v) for k, v in row.items()}


def previous_month(state: dict, month_key: str) -> dict | None:
    months = state["months"]
    i = months.index(month_key)
    return month_metrics(state, months[i - 1]) if i > 0 else None


def category_totals(clean: pd.DataFrame, tx_type: str, month_key: str | None = None) -> pd.DataFrame:
    df = clean[clean["type"] == tx_type]
    if month_key:
        df = df[df["date"].dt.strftime("%Y-%m") == month_key]
    out = df.groupby("category")["amount"].sum().sort_values(ascending=False).reset_index()
    out["pct"] = out["amount"] / out["amount"].sum() * 100 if len(out) else []
    return out


def average_category_expenses(clean: pd.DataFrame) -> dict:
    """Average monthly expense per category across the months present in the data."""
    exp = clean[clean["type"] == "Expense"].copy()
    if exp.empty:
        return {}
    n = exp["date"].dt.strftime("%Y-%m").nunique()
    return (exp.groupby("category")["amount"].sum() / n).to_dict()


def rules_for(state: dict, month_key: str | None = None) -> tuple[dict, list[dict]]:
    mm = month_metrics(state, month_key)
    metrics = rule_engine.build_metrics(mm, state["settings"], state["essential_avg"])
    return metrics, rule_engine.evaluate(metrics, state["settings"], state["essential_avg"])


def total_balance(state: dict) -> float:
    return state["pipe"]["balance"] if state["ready"] else state["settings"]["opening_balance"]
