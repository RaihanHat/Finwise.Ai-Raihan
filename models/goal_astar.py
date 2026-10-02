"""Goal Planner using A* search over monthly savings states.

State   : (month index, accumulated savings)
Action  : save one of several discrete monthly levels (comfortable .. maximum / lifestyle-restricted cap)
g(n)    : months elapsed + lifestyle-disruption penalty   (penalty = lambda * x^2, x = extra cut fraction in [0,1])
h(n)    : ceil((target - savings) / max monthly saving)   -> admissible and consistent
Open set: heapq priority queue ordered by f = g + h.
"""
import heapq
import itertools
import math
from datetime import date

import pandas as pd


def _h(savings: float, target: float, max_save: float) -> int:
    rem = target - savings
    return 0 if rem <= 1e-9 else int(math.ceil(round(rem / max_save, 9)))


def saving_levels(comfortable: float, cap: float, n_levels: int = 5) -> list[float]:
    if cap <= 0:
        return []
    comfortable = min(max(comfortable, 0.0), cap)
    if comfortable <= 0:
        raw = [cap * (i + 1) / n_levels for i in range(n_levels)]
    else:
        raw = [comfortable + (cap - comfortable) * i / max(n_levels - 1, 1) for i in range(n_levels)]
    return sorted(set(round(x, 2) for x in raw if x > 0))


def astar_plan(goal_name: str, target: float, current: float, max_save: float, comfortable: float = 0.0,
               restriction: float | None = None, lam: float = 1.0, n_levels: int = 5, max_months: int = 120,
               node_limit: int = 60_000, start: date | None = None) -> dict:
    errs = []
    if target is None or target <= 0:
        errs.append("Target price must be greater than 0.")
    if current is None or current < 0:
        errs.append("Current savings cannot be negative.")
    if max_save is None or max_save < 0:
        errs.append("Maximum monthly saving cannot be negative.")
    if errs:
        return {"status": "invalid", "found": False, "errors": errs}
    start = start or date.today()
    cap = max_save if restriction is None else min(max_save, comfortable + restriction)
    cap = max(cap, 0.0)
    levels = saving_levels(comfortable, cap, n_levels)
    comfortable_eff = min(max(comfortable, 0.0), cap)
    init = {"month": 0, "savings": float(current), "g": 0.0, "h": _h(current, target, cap) if cap > 0 else None}
    base = {"goal": goal_name, "target": target, "current": current, "cap": cap, "levels": levels, "lambda": lam,
            "comfortable": comfortable_eff, "initial": init, "restriction": restriction}
    if current >= target:
        return {**base, "status": "done", "found": True, "months": 0, "path": [], "explored": [], "nodes_explored": 0, "nodes_generated": 0,
                "total_cost": 0.0, "message": "Goal already reached with current savings."}
    if not levels:
        return {**base, "status": "no_path", "found": False, "explored": [], "nodes_explored": 0, "nodes_generated": 0,
                "message": "Target isn't reachable: monthly saving capacity is 0. Raise monthly savings or relax the lifestyle restriction."}

    def penalty(s):
        x = (s - comfortable_eff) / (cap - comfortable_eff) if cap > comfortable_eff else 0.0
        return lam * max(0.0, min(1.0, x)) ** 2, max(0.0, min(1.0, x))

    counter = itertools.count()
    start_state = (0, round(float(current), 2))
    open_heap = [(init["h"], init["h"], next(counter), start_state, 0.0)]
    best_g = {start_state: 0.0}
    parent: dict = {}
    closed: set = set()
    explored, generated = [], 1
    goal_state = None
    while open_heap:
        f, h_, _, state, g = heapq.heappop(open_heap)
        if state in closed or g > best_g.get(state, math.inf) + 1e-12:
            continue
        closed.add(state)
        m, sv = state
        explored.append({"order": len(explored) + 1, "month": m, "savings": sv, "g": round(g, 3), "h": h_, "f": round(f, 3)})
        if sv >= target - 1e-9:
            goal_state = state
            break
        if len(explored) >= node_limit or m >= max_months:
            if len(explored) >= node_limit:
                break
            continue
        for s in levels:
            take = min(s, target - sv)               # last month only saves what is still needed
            nsv = round(sv + take, 2)
            pen, x = penalty(take)
            ng = g + 1 + pen
            ns = (m + 1, nsv)
            if ng < best_g.get(ns, math.inf) - 1e-12:
                best_g[ns] = ng
                parent[ns] = (state, take, pen, x)
                nh = _h(nsv, target, cap)
                heapq.heappush(open_heap, (ng + nh, nh, next(counter), ns, ng))
                generated += 1
    if goal_state is None:
        return {**base, "status": "no_path", "found": False, "explored": explored, "nodes_explored": len(explored), "nodes_generated": generated,
                "message": f"Target isn't reachable within {max_months} months (or the node limit) at the allowed saving rate. Extend the date or raise monthly savings."}
    # reconstruct
    steps, s = [], goal_state
    while s in parent:
        ps, take, pen, x = parent[s]
        steps.append((s, ps, take, pen, x))
        s = ps
    steps.reverse()
    path, total_pen = [], 0.0
    first_day = pd.Timestamp(start).replace(day=1)
    for (st, ps, take, pen, x) in steps:
        total_pen += pen
        label = (first_day + pd.DateOffset(months=st[0] - 1)).strftime("%b %Y")
        path.append({"month_no": st[0], "month": label, "saving": take, "cumulative": st[1], "remaining": max(target - st[1], 0.0),
                     "disruption": x, "penalty": pen, "level": "High" if x > 0.66 else "Medium" if x > 0.33 else "Low",
                     "g": best_g[st], "h": _h(st[1], target, cap), "f": best_g[st] + _h(st[1], target, cap)})
    n_months = goal_state[0]
    # admissibility check along the optimal path: h(n) <= true remaining cost
    final_cost = best_g[goal_state]
    for p in path:
        p["true_remaining_cost"] = final_cost - p["g"]
    h_ok = _h(current, target, cap) <= final_cost + 1e-9 and all(p["h"] <= p["true_remaining_cost"] + 1e-9 for p in path)
    return {**base, "status": "found", "found": True, "months": n_months, "path": path, "explored": explored, "nodes_explored": len(explored),
            "nodes_generated": generated, "total_cost": final_cost, "total_penalty": total_pen,
            "goal_month": path[-1]["month"], "heuristic_admissible_on_path": bool(h_ok),
            "message": f"Goal '{goal_name}' reachable in {n_months} month(s) - {path[-1]['month']}."}
