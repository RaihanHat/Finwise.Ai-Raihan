"""Central transaction store. Every page reads/writes the same CSV (data/transactions.csv)."""
import json
import shutil
from datetime import date

import numpy as np
import pandas as pd

from utils.constants import (COLUMNS, DEFAULT_SETTINGS, DINING_KEYWORDS, DISCRETIONARY_CATEGORIES,
                             EXPENSE_CATEGORIES, INCOME_CATEGORIES, SAMPLE_PATH, SETTINGS_PATH,
                             TRANSACTIONS_PATH, UPLOAD_COLUMNS)


def auto_discretionary(tx_type, category, description="") -> float:
    """Auto-tag: Entertainment (and dining-out food) is discretionary. Income -> NaN."""
    if str(tx_type).lower() != "expense":
        return np.nan
    if category in DISCRETIONARY_CATEGORIES:
        return 1.0
    desc = str(description).lower() if isinstance(description, str) else ""
    if category == "Food" and any(k in desc for k in DINING_KEYWORDS):
        return 1.0
    return 0.0


def _empty() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="object") for c in COLUMNS})
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = df["amount"].astype(float)
    df["id"] = df["id"].astype("Int64")
    df["is_discretionary"] = df["is_discretionary"].astype(float)
    return df


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["type"] = df["type"].astype("object").where(df["type"].isna(), df["type"].astype(str).str.strip().str.title())
    df["category"] = df["category"].astype("object").where(df["category"].isna(), df["category"].astype(str).str.strip())
    df["description"] = df["description"].astype("object")
    df["is_discretionary"] = pd.to_numeric(df["is_discretionary"], errors="coerce")
    ids = pd.to_numeric(df["id"], errors="coerce")
    if ids.isna().any() or ids.duplicated().any():
        ids = pd.Series(np.arange(1, len(df) + 1), index=df.index)
    df["id"] = ids.astype(int)
    return df.reset_index(drop=True)


def load_transactions() -> pd.DataFrame:
    """Raw stored data (missing values deliberately preserved for the preprocessing pipeline)."""
    if not TRANSACTIONS_PATH.exists() or TRANSACTIONS_PATH.stat().st_size == 0:
        return _empty()
    try:
        df = pd.read_csv(TRANSACTIONS_PATH)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return _empty()
    if df.empty:
        return _empty()
    return _normalise(df).sort_values(["date", "id"], ascending=[False, False]).reset_index(drop=True)


def save_transactions(df: pd.DataFrame) -> None:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out[COLUMNS].to_csv(TRANSACTIONS_PATH, index=False)


def _next_id(df: pd.DataFrame) -> int:
    return int(df["id"].max()) + 1 if len(df) else 1


def add_transaction(tx_date, tx_type, category, amount, description, is_discretionary=None) -> int:
    df = load_transactions()
    if is_discretionary is None:
        is_discretionary = auto_discretionary(tx_type, category, description)
    new_id = _next_id(df)
    row = {"id": new_id, "date": pd.Timestamp(tx_date), "type": tx_type, "category": category,
           "amount": float(amount), "description": description.strip(),
           "is_discretionary": is_discretionary if tx_type == "Expense" else np.nan}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True) if len(df) else pd.DataFrame([row])
    save_transactions(_normalise(df))
    return new_id


def update_transaction(tx_id, tx_date, tx_type, category, amount, description, is_discretionary=None) -> bool:
    df = load_transactions()
    idx = df.index[df["id"] == tx_id]
    if len(idx) == 0:
        return False
    i = idx[0]
    if is_discretionary is None:
        is_discretionary = auto_discretionary(tx_type, category, description)
    df.loc[i, ["date", "type", "category", "amount", "description"]] = [
        pd.Timestamp(tx_date), tx_type, category, float(amount), description.strip()]
    df.loc[i, "is_discretionary"] = is_discretionary if tx_type == "Expense" else np.nan
    save_transactions(df)
    return True


def delete_transaction(tx_id) -> bool:
    df = load_transactions()
    keep = df[df["id"] != tx_id]
    if len(keep) == len(df):
        return False
    save_transactions(keep)
    return True


def validate_upload(file) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Parse an uploaded CSV (date,type,category,amount,description).
    Returns (valid_rows, rejected_rows_with_reason, info). Blank amount/description rows are kept
    (NaN) so the preprocessing pipeline can impute them; structurally invalid rows are rejected."""
    info = {"ok": False, "error": None, "n_rows": 0}
    try:
        raw = pd.read_csv(file)
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"Could not read CSV: {exc}"
        return _empty(), pd.DataFrame(), info
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    missing = [c for c in UPLOAD_COLUMNS if c not in raw.columns]
    if missing:
        info["error"] = f"Missing required column(s): {', '.join(missing)}. Expected: {', '.join(UPLOAD_COLUMNS)}."
        return _empty(), pd.DataFrame(), info
    info["n_rows"] = len(raw)
    if raw.empty:
        info["error"] = "The CSV file has a header but no rows."
        return _empty(), pd.DataFrame(), info

    df = raw[UPLOAD_COLUMNS].copy()
    df["type"] = df["type"].astype(str).str.strip().str.title()
    df["category"] = df["category"].astype(str).str.strip().str.title().replace({"Debt repaid": "Debt Repaid"})
    df["category"] = df["category"].replace({"Debt Repaid": "Debt Repaid"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    reasons = pd.Series("", index=df.index)
    reasons[df["date"].isna()] += "invalid date; "
    reasons[~df["type"].isin(["Income", "Expense"])] += "type must be Income/Expense; "
    valid_cat = df.apply(lambda r: r["category"] in (INCOME_CATEGORIES if r["type"] == "Income" else EXPENSE_CATEGORIES), axis=1)
    reasons[~valid_cat] += "category not valid for type; "
    reasons[raw["amount"].notna() & df["amount"].isna()] += "amount not numeric; "
    reasons[df["amount"] < 0] += "negative amount; "
    bad = reasons != ""
    rejected = df[bad].assign(reason=reasons[bad])
    good = df[~bad].copy()
    good["is_discretionary"] = [auto_discretionary(t, c, d) for t, c, d in zip(good["type"], good["category"], good["description"])]
    good["description"] = good["description"].astype("object").where(good["description"].notna(), np.nan)
    good["id"] = np.nan
    info["ok"] = True
    return good[["id", "date", "type", "category", "amount", "description", "is_discretionary"]], rejected, info


def import_transactions(valid: pd.DataFrame, mode: str = "append", skip_duplicates: bool = True) -> dict:
    """mode: 'append' or 'replace'. Returns counts."""
    existing = load_transactions() if mode == "append" else _empty()
    new = valid.copy()
    skipped = 0
    if skip_duplicates and len(existing) and len(new):
        key = lambda d: d["date"].dt.strftime("%Y-%m-%d") + "|" + d["type"].astype(str) + "|" + d["category"].astype(str) + "|" + d["amount"].round(2).astype(str) + "|" + d["description"].fillna("").astype(str)  # noqa: E731
        dup = key(new).isin(set(key(existing)))
        skipped = int(dup.sum())
        new = new[~dup]
    combined = pd.concat([existing, new], ignore_index=True) if len(existing) else new.copy()
    combined = combined.reset_index(drop=True)
    start = _next_id(existing)
    old_ids = combined["id"].copy()
    combined["id"] = np.where(old_ids.notna(), old_ids, np.nan)
    n_old = len(existing)
    combined.loc[n_old:, "id"] = np.arange(start, start + len(combined) - n_old)
    save_transactions(_normalise(combined))
    return {"added": len(new), "skipped_duplicates": skipped, "total": len(combined)}


def load_sample() -> int:
    if SAMPLE_PATH.exists():
        shutil.copy(SAMPLE_PATH, TRANSACTIONS_PATH)
    return len(load_transactions())


def clear_all() -> None:
    save_transactions(_empty())


# ---------------------------------------------------------------- settings
def load_settings() -> dict:
    s = dict(DEFAULT_SETTINGS)
    if SETTINGS_PATH.exists():
        try:
            s.update(json.loads(SETTINGS_PATH.read_text()))
        except (json.JSONDecodeError, OSError):
            pass
    return s


def save_settings(settings: dict) -> None:
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2))


def days_left_in_month(today: date | None = None) -> int:
    today = today or date.today()
    end = (pd.Timestamp(today) + pd.offsets.MonthEnd(0)).date()
    return (end - today).days
