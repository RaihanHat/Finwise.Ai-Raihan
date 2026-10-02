"""Reusable preprocessing + feature-engineering pipeline (no Streamlit dependency)."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils.constants import ESSENTIAL_CATEGORIES, NON_INCOME_FOR_RATIOS
from utils.data_loader import auto_discretionary

NUMERIC_FEATURES = ["amount", "signed_amount", "day_of_month", "day_of_week", "month", "days_to_month_end", "category_target_enc"]


# ------------------------------------------------------------------ 1. missing values
def detect_missing(raw: pd.DataFrame) -> pd.DataFrame:
    cols = ["date", "type", "category", "amount", "description", "is_discretionary"]
    rep = pd.DataFrame({"column": cols, "missing": [int(raw[c].isna().sum()) if c in raw else len(raw) for c in cols]})
    # is_discretionary is legitimately empty for income rows
    if "is_discretionary" in raw:
        exp_missing = int(raw.loc[raw["type"] == "Expense", "is_discretionary"].isna().sum())
        rep.loc[rep["column"] == "is_discretionary", "missing"] = exp_missing
    rep["pct"] = (rep["missing"] / max(len(raw), 1) * 100).round(2)
    return rep


def handle_missing(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Drop structurally unusable rows, impute amount (category mean) and description (category default)."""
    df = raw.copy()
    log: list[dict] = []
    bad = df["date"].isna() | df["type"].isna() | df["category"].isna()
    if bad.any():
        log.append({"step": "Drop rows without date/type/category", "rows": int(bad.sum()), "method": "row removal"})
        df = df[~bad]
    miss_amt = df["amount"].isna()
    df["amount_imputed"] = False
    if miss_amt.any():
        cat_mean = df.groupby(["type", "category"])["amount"].transform("mean")
        type_mean = df.groupby("type")["amount"].transform("mean")
        fill = cat_mean.fillna(type_mean)
        df.loc[miss_amt, "amount"] = fill[miss_amt]
        df.loc[miss_amt, "amount_imputed"] = True
        still = df["amount"].isna()
        if still.any():
            df = df[~still]
            log.append({"step": "Drop rows with amount that cannot be imputed", "rows": int(still.sum()), "method": "row removal"})
        log.append({"step": "Impute missing amount", "rows": int(miss_amt.sum()), "method": "category mean (fallback: type mean)"})
    miss_desc = df["description"].isna() | (df["description"].astype(str).str.strip() == "")
    df["description_imputed"] = miss_desc
    if miss_desc.any():
        df.loc[miss_desc, "description"] = df.loc[miss_desc, "category"].astype(str) + " (default)"
        log.append({"step": "Fill missing description", "rows": int(miss_desc.sum()), "method": "category default text"})
    miss_flag = (df["type"] == "Expense") & df["is_discretionary"].isna()
    if miss_flag.any():
        df.loc[miss_flag, "is_discretionary"] = [auto_discretionary(t, c, d) for t, c, d in
                                                  zip(df.loc[miss_flag, "type"], df.loc[miss_flag, "category"], df.loc[miss_flag, "description"])]
        log.append({"step": "Auto-tag missing discretionary flag", "rows": int(miss_flag.sum()), "method": "category / keyword rule"})
    if not log:
        log.append({"step": "No missing values found", "rows": 0, "method": "none required"})
    return df.sort_values(["date", "id"]).reset_index(drop=True), log


# ------------------------------------------------------------------ 2. feature engineering
def engineer_features(clean: pd.DataFrame) -> pd.DataFrame:
    df = clean.copy()
    df["month_key"] = df["date"].dt.strftime("%Y-%m")
    df["day_of_month"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["days_to_month_end"] = df["date"].dt.days_in_month - df["day_of_month"]
    df["signed_amount"] = np.where(df["type"] == "Income", df["amount"], -df["amount"])
    df["discretionary"] = ((df["type"] == "Expense") & (df["is_discretionary"] == 1)).astype(int)
    df["essential"] = ((df["type"] == "Expense") & df["category"].isin(ESSENTIAL_CATEGORIES) & (df["discretionary"] == 0)).astype(int)
    df["running_balance_delta"] = df["signed_amount"].cumsum()
    return df


# ------------------------------------------------------------------ 3. encoding
def encode_categoricals(feat: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode type & category, plus target encoding (mean amount per category)."""
    oh = pd.get_dummies(feat[["type", "category"]], prefix=["type", "cat"], dtype=int)
    enc = feat[["id", "date"]].join(oh)
    enc["category_target_enc"] = feat.groupby("category")["amount"].transform("mean")
    return enc


# ------------------------------------------------------------------ 4. normalisation
def normalize(feat: pd.DataFrame, enc: pd.DataFrame):
    num = feat[["amount", "signed_amount", "day_of_month", "day_of_week", "month", "days_to_month_end"]].copy()
    num["category_target_enc"] = enc["category_target_enc"].values
    scaler = MinMaxScaler()
    arr = scaler.fit_transform(num.values) if len(num) else np.empty((0, num.shape[1]))
    norm = pd.DataFrame(arr, columns=[c + "_norm" for c in num.columns], index=num.index)
    ranges = pd.DataFrame({"feature": num.columns, "min": scaler.data_min_, "max": scaler.data_max_}) if len(num) else pd.DataFrame()
    return norm, scaler, ranges


# ------------------------------------------------------------------ 5. derived metrics
def daily_expense_series(clean: pd.DataFrame, end=None) -> pd.Series:
    exp = clean[clean["type"] == "Expense"]
    if clean.empty:
        return pd.Series(dtype=float)
    start = clean["date"].min().normalize()
    end = pd.Timestamp(end).normalize() if end is not None else clean["date"].max().normalize()
    idx = pd.date_range(start, max(end, start), freq="D")
    return exp.groupby(exp["date"].dt.normalize())["amount"].sum().reindex(idx, fill_value=0.0)


def burn_rate(clean: pd.DataFrame, as_of=None) -> float:
    """Rolling 7-day average daily spend (ending at as_of) extrapolated to 30 days."""
    daily = daily_expense_series(clean, as_of)
    if daily.empty:
        return 0.0
    if as_of is not None:
        daily = daily[daily.index <= pd.Timestamp(as_of).normalize()]
    return float(daily.rolling(7, min_periods=1).mean().iloc[-1] * 30)


def monthly_metrics(clean: pd.DataFrame) -> pd.DataFrame:
    """One row per month with income/expense totals and the three derived ratios."""
    cols = ["month_key", "income", "net_income", "expenses", "essential", "discretionary", "net_savings",
            "savings_ratio", "discretionary_ratio", "burn_rate", "tx_count"]
    if clean.empty:
        return pd.DataFrame(columns=cols)
    df = clean.copy()
    df["month_key"] = df["date"].dt.strftime("%Y-%m")
    last_date = df["date"].max()
    rows = []
    for mk, g in df.groupby("month_key"):
        inc = g.loc[g["type"] == "Income", "amount"].sum()
        net_inc = g.loc[(g["type"] == "Income") & ~g["category"].isin(NON_INCOME_FOR_RATIOS), "amount"].sum()
        ex = g[g["type"] == "Expense"]
        exp_total = ex["amount"].sum()
        disc = ex.loc[ex["is_discretionary"] == 1, "amount"].sum()
        month_end = (pd.Period(mk).end_time).normalize()
        rows.append({
            "month_key": mk, "income": inc, "net_income": net_inc, "expenses": exp_total,
            "essential": exp_total - disc, "discretionary": disc, "net_savings": inc - exp_total,
            "savings_ratio": (net_inc - exp_total) / net_inc if net_inc > 0 else np.nan,
            "discretionary_ratio": disc / exp_total if exp_total > 0 else np.nan,
            "burn_rate": burn_rate(clean, min(month_end, last_date)), "tx_count": len(g)})
    return pd.DataFrame(rows, columns=cols)


def current_metrics(monthly: pd.DataFrame, month_key: str | None = None) -> dict:
    if monthly.empty:
        return {}
    row = monthly.iloc[-1] if month_key is None else monthly[monthly["month_key"] == month_key].iloc[0]
    return {k: (None if pd.isna(v) else (float(v) if k != "month_key" else v)) for k, v in row.items()}


# ------------------------------------------------------------------ pipeline
def run_pipeline(raw: pd.DataFrame, balance_opening: float = 0.0) -> dict:
    """RAW -> missing values -> encoding -> normalisation -> feature engineering -> PROCESSED."""
    result = {"raw": raw, "ready": False}
    if raw is None or raw.empty:
        return result
    missing = detect_missing(raw)
    clean, log = handle_missing(raw)
    if clean.empty:
        result.update(missing=missing, log=log)
        return result
    feat = engineer_features(clean)
    enc = encode_categoricals(feat)
    norm, scaler, ranges = normalize(feat, enc)
    monthly = monthly_metrics(clean)
    result.update(ready=True, missing=missing, log=log, clean=clean, features=feat, encoded=enc,
                  normalized=norm, scaler=scaler, ranges=ranges, monthly=monthly,
                  metrics=current_metrics(monthly),
                  balance=float(balance_opening + clean.loc[clean["type"] == "Income", "amount"].sum()
                                - clean.loc[clean["type"] == "Expense", "amount"].sum()))
    return result
