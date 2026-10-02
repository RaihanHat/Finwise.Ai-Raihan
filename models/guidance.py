"""Integrated Financial Guidance: assembles a traceable report from every module's REAL output."""
from models.rule_engine import PRIORITY_ORDER

SRC = {"ml": "ML prediction", "rule": "Rule-based", "csp": "CSP optimization", "astar": "A* planning", "metric": "Financial metric", "kmeans": "K-Means (ML)"}


def _money(x):
    return f"৳{x:,.0f}"


def build_report(metrics: dict, month_key: str, balance: float, profile: dict | None, purchase: dict | None,
                 csp: dict | None, rules: list[dict], astar: dict | None) -> dict:
    rep = {"month": month_key}
    pct = lambda v: "n/a" if v is None else f"{v * 100:.1f}%"  # noqa: E731
    rep["summary"] = [
        {"label": "Monthly income", "value": _money(metrics.get("income", 0)), "source": SRC["metric"]},
        {"label": "Monthly expenses", "value": _money(metrics.get("expenses", 0)), "source": SRC["metric"]},
        {"label": "Net savings", "value": _money(metrics.get("net_savings", 0)), "source": SRC["metric"]},
        {"label": "Current balance", "value": _money(balance), "source": SRC["metric"]},
        {"label": "Savings ratio", "value": pct(metrics.get("savings_ratio")), "source": SRC["metric"]},
        {"label": "Burn rate (30d)", "value": _money(metrics.get("burn_rate", 0)), "source": SRC["metric"]},
        {"label": "Discretionary ratio", "value": pct(metrics.get("discretionary_ratio")), "source": SRC["metric"]},
    ]
    # spending profile
    if profile and profile.get("ready"):
        d = profile["drift"]
        rep["profile"] = {"available": True, "profile": profile["current_profile"], "source": SRC["kmeans"],
                          "evidence": f"Latest {profile['window_desc'].split(',')[0]} assigned to cluster {profile['current_cluster']} by K-Means (k=3, {profile['n_windows']} windows, silhouette={profile['silhouette']:.2f}).",
                          "drift": (f"Risk drift: {d['direction']} (delta {d['delta']:+.2f})" if d.get("available") else "Risk drift: not enough history.")}
    else:
        rep["profile"] = {"available": False, "reason": (profile or {}).get("reason", "Spending profile not available.")}
    # budget
    if csp and csp.get("feasible"):
        a = csp["allocation"]
        rep["budget"] = {"available": True, "status": csp["status"], "allocation": a, "remaining": csp["remaining"], "source": SRC["csp"],
                         "constraints": csp["constraints"], "evidence": f"Backtracking+FC+MRV explored {csp['stats']['nodes']:,} nodes, {csp['stats']['solutions']:,} feasible solutions; best soft-constraint score {csp['score']:.2f}."}
    elif csp and csp.get("status") == "infeasible":
        rep["budget"] = {"available": False, "infeasible": True, "reason": csp.get("reason"), "source": SRC["csp"]}
    else:
        rep["budget"] = {"available": False, "reason": "Budget Optimizer has not been run. Open 'Budget Optimizer' and press Solve."}
    # purchase
    if purchase:
        lr, knn = purchase["lr"], purchase.get("knn")
        rep["purchase"] = {"available": True, "item": purchase["item"], "price": purchase["price"], "lr": lr, "knn": knn, "source": SRC["ml"],
                           "agree": (knn is None or knn["predicted"] == lr["predicted"]),
                           "note": "Trained on SYNTHETIC/DEMO data - not a real-world financial prediction."}
    else:
        rep["purchase"] = {"available": False, "reason": "No purchase evaluated yet. Open 'Purchase Advisor' to evaluate one."}
    # saving recommendations (rules)
    rep["rules"] = rules
    rep["saving_recs"] = [r for r in rules if r["triggered"]] + [r for r in rules if r["kind"] == "positive" and r["triggered"] and r not in []]
    seen, uniq = set(), []
    for r in rep["saving_recs"]:
        if r["rule_id"] not in seen:
            seen.add(r["rule_id"]); uniq.append(r)
    rep["saving_recs"] = uniq
    # goal
    if astar and astar.get("found"):
        rep["goal"] = {"available": True, "plan": astar, "source": SRC["astar"]}
    elif astar:
        rep["goal"] = {"available": False, "reason": astar.get("message", "Goal plan not found."), "source": SRC["astar"]}
    else:
        rep["goal"] = {"available": False, "reason": "Goal Planner has not been run. Open 'Goal Planner' and run A*."}
    # top-3 actions
    cands = []
    for r in rules:
        if r["triggered"] and r["kind"] == "warning":
            cands.append((PRIORITY_ORDER[r["priority"]] * 10 + 5, r["recommendation"], SRC["rule"], f"Rule {r['rule_id']}: {r['condition']} (current {r['current']})"))
    if purchase and purchase["lr"]["predicted"] in ("Rejected", "Caution"):
        lr = purchase["lr"]
        score = 28 if lr["predicted"] == "Rejected" else 18
        cands.append((score, f"Postpone or reduce the purchase '{purchase['item']}' ({lr['predicted']} by Logistic Regression, P={lr['probabilities'][lr['predicted']]:.0%}).",
                      SRC["ml"], f"Logistic Regression probabilities: " + ", ".join(f"{k} {v:.0%}" for k, v in lr["probabilities"].items())))
    if profile and profile.get("ready") and profile["drift"].get("available") and profile["drift"]["direction"] == "toward High-Risk":
        cands.append((25, "Spending is drifting toward the High-Risk profile: tighten discretionary limits this month.", SRC["kmeans"], profile["drift"]["direction"]))
    elif profile and profile.get("ready") and profile["current_rank"] == 2:
        cands.append((24, "Your current spending pattern matches the Impulsive / High-Risk cluster: review non-essential purchases.", SRC["kmeans"], f"Cluster {profile['current_cluster']}"))
    if csp and csp.get("status") == "infeasible":
        cands.append((35, "Budget is infeasible: " + (csp.get("reason") or "constraints cannot be met") + " Reduce requests or increase income.", SRC["csp"], "Backtracking search found no feasible allocation"))
    elif csp and csp.get("feasible"):
        short = [c for c in csp["constraints"] if c["type"] == "Soft" and c["status"] != "Satisfied"]
        if short:
            cands.append((15, f"Follow the optimised budget; soft targets not fully met: {', '.join(c['constraint'] for c in short)}.", SRC["csp"], short[0]["detail"]))
        else:
            cands.append((8, "Follow the optimised budget allocation: all hard and soft constraints satisfied.", SRC["csp"], f"{csp['stats']['solutions']:,} feasible solutions compared"))
    if astar and astar.get("found") and astar.get("months", 0) > 0:
        hi = [p for p in astar["path"] if p["level"] == "High"]
        cands.append((12 + (3 if hi else 0), f"Save {_money(astar['path'][0]['saving'])} per month toward '{astar['goal']}' to finish in {astar['months']} month(s) ({astar['goal_month']}).",
                      SRC["astar"], f"A* explored {astar['nodes_explored']} states; total cost {astar['total_cost']:.2f}"))
    for r in rules:
        if r["triggered"] and r["kind"] == "positive":
            cands.append((3, r["recommendation"], SRC["rule"], f"Rule {r['rule_id']}: {r['condition']} (current {r['current']})"))
    cands.sort(key=lambda c: -c[0])
    rep["top_actions"] = [{"action": a, "source": s, "evidence": e} for _, a, s, e in cands[:3]]
    return rep
