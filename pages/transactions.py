from datetime import date

import pandas as pd
import streamlit as st

from utils import data_loader as dl
from utils.calculations import load_state
from utils.constants import CURRENCY, EXPENSE_CATEGORIES, INCOME_CATEGORIES, UPLOAD_COLUMNS
from utils.helpers import empty_state, kpi_card, money, page_header, section

DISC_OPTIONS = ["Auto (by category)", "Yes - discretionary", "No - essential"]


def _disc_value(choice, tx_type, category, description):
    if tx_type != "Expense":
        return None
    return {DISC_OPTIONS[1]: 1.0, DISC_OPTIONS[2]: 0.0}.get(choice, dl.auto_discretionary(tx_type, category, description))


def _validate(amount, description):
    errs = []
    if amount is None or amount <= 0:
        errs.append("Amount must be greater than 0.")
    if not description or not description.strip():
        errs.append("Description is required.")
    return errs


def render():
    page_header("Income & Expenses", "Add, edit, delete and import transactions. Every module reads this same dataset.", "02 · Transactions")
    if "flash" in st.session_state:
        st.success(st.session_state.pop("flash"))
    state = load_state()
    raw = state["raw"]
    income = raw.loc[raw["type"] == "Income", "amount"].sum()
    expense = raw.loc[raw["type"] == "Expense", "amount"].sum()
    balance = state["settings"]["opening_balance"] + income - expense
    c = st.columns(3)
    with c[0]:
        kpi_card("Total Income", money(income), f"{(raw['type'] == 'Income').sum()} transactions", "jade")
    with c[1]:
        kpi_card("Total Expenses", money(expense), f"{(raw['type'] == 'Expense').sum()} transactions")
    with c[2]:
        kpi_card("Current Balance", money(balance), "opening balance + income − expenses", "jade" if balance >= 0 else "brick")

    t_list, t_add, t_edit, t_csv = st.tabs(["Transactions", "Add", "Edit / Delete", "Import CSV"])

    with t_list:
        if raw.empty:
            empty_state("No transactions logged yet. Add your first one to start tracking.", "Use the Add tab, import a CSV, or load demo data in Settings.")
        else:
            f1, f2, f3, f4 = st.columns([2, 1, 2, 2])
            q = f1.text_input("Search", placeholder="description or category", key="tx_q")
            ftype = f2.selectbox("Type", ["All", "Income", "Expense"], key="tx_type")
            cats = sorted(raw["category"].dropna().unique())
            fcat = f3.multiselect("Category", cats, key="tx_cat")
            dmin, dmax = raw["date"].min().date(), raw["date"].max().date()
            frange = f4.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax, key="tx_dates")
            view = raw.copy()
            if q:
                mask = view["description"].fillna("").str.contains(q, case=False, regex=False) | view["category"].fillna("").str.contains(q, case=False, regex=False)
                view = view[mask]
            if ftype != "All":
                view = view[view["type"] == ftype]
            if fcat:
                view = view[view["category"].isin(fcat)]
            if isinstance(frange, tuple) and len(frange) == 2:
                view = view[(view["date"] >= pd.Timestamp(frange[0])) & (view["date"] <= pd.Timestamp(frange[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))]
            vi, ve = view.loc[view["type"] == "Income", "amount"].sum(), view.loc[view["type"] == "Expense", "amount"].sum()
            st.caption(f"{len(view)} of {len(raw)} transactions · filtered income {money(vi)} · filtered expenses {money(ve)} · net {money(vi - ve)}")
            show = view.rename(columns={"id": "ID", "date": "Date", "type": "Type", "category": "Category", "amount": "Amount", "description": "Description", "is_discretionary": "Discretionary"}).copy()
            show["Discretionary"] = show["Discretionary"].map({1.0: "Yes", 0.0: "No"}).fillna("")
            st.dataframe(show, width="stretch", hide_index=True, height=420,
                         column_config={"Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                                        "Amount": st.column_config.NumberColumn(format=f"{CURRENCY}%.2f")})

    with t_add:
        tx_type = st.radio("Transaction type", ["Income", "Expense"], horizontal=True, key="add_type")
        cats = INCOME_CATEGORIES if tx_type == "Income" else EXPENSE_CATEGORIES
        with st.form("add_form", clear_on_submit=True):
            a, b = st.columns(2)
            d = a.date_input("Date", value=date.today())
            amt = b.number_input(f"Amount ({CURRENCY})", min_value=0.0, step=100.0, format="%.2f")
            cat = a.selectbox("Category", cats)
            desc = b.text_input("Description")
            disc = st.selectbox("Discretionary tag", DISC_OPTIONS, help="Only used for expenses. 'Auto' tags Entertainment and dining-out as discretionary.") if tx_type == "Expense" else DISC_OPTIONS[0]
            if st.form_submit_button("Add transaction", type="primary"):
                errs = _validate(amt, desc)
                if errs:
                    for e in errs:
                        st.error(e)
                else:
                    dl.add_transaction(d, tx_type, cat, amt, desc, _disc_value(disc, tx_type, cat, desc))
                    st.session_state["flash"] = f"Added {tx_type.lower()} of {money(amt)} ({cat})."
                    st.rerun()

    with t_edit:
        if raw.empty:
            empty_state("Nothing to edit yet.", "Add a transaction first.")
        else:
            labels = {int(r.id): f"#{int(r.id)} · {r.date:%Y-%m-%d} · {r.type} · {r.category} · {money(r.amount) if pd.notna(r.amount) else 'n/a'} · {r.description if isinstance(r.description, str) else ''}"
                      for r in raw.itertuples() if pd.notna(r.date)}
            if not labels:
                empty_state("No editable rows (all rows have invalid dates).")
            else:
                tid = st.selectbox("Select a transaction", list(labels), format_func=lambda i: labels[i])
                row = raw[raw["id"] == tid].iloc[0]
                et = st.radio("Type", ["Income", "Expense"], index=0 if row["type"] == "Income" else 1, horizontal=True, key=f"et_{tid}")
                cats = INCOME_CATEGORIES if et == "Income" else EXPENSE_CATEGORIES
                with st.form(f"edit_form_{tid}"):
                    a, b = st.columns(2)
                    d = a.date_input("Date", value=row["date"].date(), key=f"ed_{tid}")
                    amt = b.number_input(f"Amount ({CURRENCY})", min_value=0.0, step=100.0, format="%.2f", value=float(row["amount"]) if pd.notna(row["amount"]) else 0.0, key=f"ea_{tid}")
                    cat = a.selectbox("Category", cats, index=cats.index(row["category"]) if row["category"] in cats else 0, key=f"ec_{tid}")
                    desc = b.text_input("Description", value=row["description"] if isinstance(row["description"], str) else "", key=f"edd_{tid}")
                    disc = st.selectbox("Discretionary tag", DISC_OPTIONS, key=f"edi_{tid}") if et == "Expense" else DISC_OPTIONS[0]
                    u, dele = st.columns(2)
                    save = u.form_submit_button("Save changes", type="primary")
                    confirm = dele.checkbox("Confirm delete", key=f"cd_{tid}")
                    remove = dele.form_submit_button("Delete transaction")
                    if save:
                        errs = _validate(amt, desc)
                        if errs:
                            for e in errs:
                                st.error(e)
                        else:
                            dl.update_transaction(tid, d, et, cat, amt, desc, _disc_value(disc, et, cat, desc))
                            st.session_state["flash"] = f"Updated transaction #{tid}."
                            st.rerun()
                    if remove:
                        if confirm:
                            dl.delete_transaction(tid)
                            st.session_state["flash"] = f"Deleted transaction #{tid}."
                            st.rerun()
                        else:
                            st.warning("Tick 'Confirm delete' first.")

    with t_csv:
        st.markdown(f"CSV format: `{','.join(UPLOAD_COLUMNS)}` — type is `Income`/`Expense`; dates like `2026-09-15`.")
        st.download_button("Download CSV template", "date,type,category,amount,description\n2026-09-01,Income,Salary,65000,Monthly salary\n2026-09-05,Expense,Housing,18000,House rent\n",
                           "finwise_template.csv", "text/csv")
        up = st.file_uploader("Upload transactions CSV", type=["csv"], key=f"up_{st.session_state.get('up_n', 0)}")
        if up is not None:
            valid, rejected, info = dl.validate_upload(up)
            if not info["ok"]:
                st.error(info["error"])
            else:
                st.info(f"{info['n_rows']} rows read · {len(valid)} valid · {len(rejected)} rejected.")
                if len(rejected):
                    with st.expander("Rejected rows"):
                        st.dataframe(rejected, width="stretch", hide_index=True)
                if len(valid):
                    miss = int(valid["amount"].isna().sum() + valid["description"].isna().sum())
                    if miss:
                        st.warning(f"{miss} missing amount/description value(s) were kept and will be imputed by the Data Processing pipeline.")
                    st.dataframe(valid.drop(columns=["id"]).head(20), width="stretch", hide_index=True)
                    mode = st.radio("Import mode", ["Append to existing", "Replace all data"], horizontal=True)
                    skip = st.checkbox("Skip duplicates", value=True)
                    if st.button("Import valid rows", type="primary"):
                        res = dl.import_transactions(valid, "append" if mode.startswith("Append") else "replace", skip)
                        st.session_state["up_n"] = st.session_state.get("up_n", 0) + 1
                        st.session_state["flash"] = f"Imported {res['added']} rows ({res['skipped_duplicates']} duplicates skipped). Dataset now has {res['total']} transactions."
                        st.rerun()
