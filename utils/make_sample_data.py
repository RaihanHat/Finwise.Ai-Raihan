"""Generates data/sample_data.csv (demo transactions, Jun-Sep 2026). Run: python -m utils.make_sample_data
This file is DEMO data so the app is usable on first launch; replace it via the Income & Expenses page."""
import numpy as np
import pandas as pd

from utils.constants import SAMPLE_PATH, TRANSACTIONS_PATH
from utils.data_loader import auto_discretionary


def build(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    months = pd.period_range("2026-06", "2026-09", freq="M")
    for mi, m in enumerate(months):
        d = lambda day: pd.Timestamp(year=m.year, month=m.month, day=min(day, m.days_in_month))  # noqa: E731
        rows.append((d(1), "Income", "Salary", 65000, "Monthly salary"))
        if mi in (0, 2, 3):
            rows.append((d(int(rng.integers(10, 25))), "Income", "Freelance", int(rng.integers(6, 14)) * 1000, "Freelance project payment"))
        rows.append((d(15), "Income", "Investment", int(rng.integers(8, 14)) * 100, "Dividend / interest"))
        if mi == 1:
            rows.append((d(12), "Income", "Gift", 5000, "Birthday gift"))
        if mi == 2:
            rows.append((d(20), "Income", "Debt Repaid", 4000, "Friend repaid loan"))
        if mi == 0:
            rows.append((d(3), "Income", "Loan", 10000, "Short-term personal loan"))
        rows.append((d(5), "Expense", "Housing", 18000, "House rent"))
        rows.append((d(7), "Expense", "Housing", int(rng.integers(22, 30)) * 100, "Utilities (electricity, gas, internet)"))
        for day in (3, 10, 17, 24):
            rows.append((d(day + int(rng.integers(0, 2))), "Expense", "Food", int(rng.integers(18, 28)) * 100, "Groceries"))
        n_dine = 2 + mi
        for _ in range(n_dine):
            rows.append((d(int(rng.integers(2, 29))), "Expense", "Food", int(rng.integers(4, 9)) * 100 + 50 * mi * 2, "Dining out - restaurant"))
        for _ in range(8):
            rows.append((d(int(rng.integers(1, 29))), "Expense", "Transportation", int(rng.integers(8, 25)) * 10, "Ride / bus fare"))
        rows.append((d(int(rng.integers(8, 20))), "Expense", "Transportation", int(rng.integers(12, 20)) * 100, "Fuel / monthly pass"))
        n_ent = 3 + 2 * mi
        for _ in range(n_ent):
            rows.append((d(int(rng.integers(1, 29))), "Expense", "Entertainment", int(rng.integers(5, 16)) * 100 + 400 * mi, rng.choice(["Movie night", "Streaming subscription", "Gaming", "Concert / outing", "Weekend trip"])))
        rows.append((d(int(rng.integers(5, 25))), "Expense", "Health", int(rng.integers(8, 20)) * 100, "Pharmacy / checkup"))
        rows.append((d(10), "Expense", "Education", 4500, "Course / books"))
    df = pd.DataFrame(rows, columns=["date", "type", "category", "amount", "description"])
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df.sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    sample = build()
    sample.to_csv(SAMPLE_PATH, index=False)
    out = sample.copy()
    out.insert(0, "id", range(1, len(out) + 1))
    out["is_discretionary"] = [auto_discretionary(t, c, de) for t, c, de in zip(out["type"], out["category"], out["description"])]
    out.to_csv(TRANSACTIONS_PATH, index=False)
    print(len(sample), "rows written")
