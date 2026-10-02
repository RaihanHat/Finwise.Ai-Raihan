"""End-to-end smoke test: drives every page with Streamlit's AppTest.  Run: python tests/e2e_test.py
Back up data/ first - the test adds/edits/deletes a transaction (it restores the files at the end)."""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest  # noqa: E402

DATA = ROOT / "data"
backup = ROOT / ".data_backup"
shutil.copytree(DATA, backup, dirs_exist_ok=True)

SCRIPT = """
import sys, importlib, streamlit as st
sys.path.insert(0, r'%s')
st.session_state.setdefault('__p', 'dashboard')
importlib.import_module('pages.' + st.session_state['__p']).render()
""" % ROOT

fails = []


def check(at, label):
    errs = [str(e.value) for e in at.exception]
    print(("FAIL " if errs else "ok   ") + label, errs[:1] if errs else "")
    if errs:
        fails.append(label)


def goto(at, page):
    at.session_state["__p"] = page
    at.run()
    check(at, f"page:{page}")


try:
    at = AppTest.from_string(SCRIPT, default_timeout=120)
    at.run()
    check(at, "page:dashboard")
    for p in ["transactions", "data_processing", "spending_analysis", "purchase_advisor", "budget_optimizer", "rule_advisor", "goal_planner", "financial_guidance", "model_evaluation", "settings"]:
        goto(at, p)

    # --- purchase advisor
    goto(at, "purchase_advisor")
    at.text_input[0].set_value("Laptop"); at.number_input[0].set_value(60000.0)
    [b for b in at.button if b.label == "Evaluate purchase"][0].click().run(); check(at, "purchase evaluate")
    assert at.session_state["purchase_result"]["lr"]["predicted"] in ("Approved", "Caution", "Rejected")
    # --- budget
    goto(at, "budget_optimizer")
    [b for b in at.button if b.label == "Solve budget (CSP)"][0].click().run(); check(at, "csp solve")
    print("   csp status:", at.session_state["csp_result"]["status"])
    # --- goal
    goto(at, "goal_planner")
    at.text_input[0].set_value("Phone"); at.number_input[0].set_value(80000.0)
    [b for b in at.button if b.label == "Run A* search"][0].click().run(); check(at, "astar run")
    print("   astar:", at.session_state["astar_result"].get("message"))
    for p in ["financial_guidance", "model_evaluation", "rule_advisor", "dashboard"]:
        goto(at, p)

    # --- transaction CRUD
    from utils import data_loader as dl
    n0 = len(dl.load_transactions())
    nid = dl.add_transaction("2026-09-30", "Expense", "Food", 123.0, "Test lunch")
    assert len(dl.load_transactions()) == n0 + 1
    assert dl.update_transaction(nid, "2026-09-30", "Expense", "Entertainment", 150.0, "Test movie")
    assert dl.load_transactions().query("id == @nid").iloc[0]["is_discretionary"] == 1.0
    assert dl.delete_transaction(nid) and len(dl.load_transactions()) == n0
    goto(at, "transactions")

    # --- empty data
    dl.clear_all()
    for p in ["dashboard", "transactions", "data_processing", "spending_analysis", "budget_optimizer", "rule_advisor", "goal_planner", "financial_guidance", "model_evaluation", "purchase_advisor"]:
        goto(at, p)
finally:
    shutil.rmtree(DATA); shutil.copytree(backup, DATA); shutil.rmtree(backup)
print("\nFAILED:" if fails else "\nALL PAGES OK", fails)
sys.exit(1 if fails else 0)
