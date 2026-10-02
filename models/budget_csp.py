"""Budget Optimizer as a real Constraint Satisfaction Problem.

Variables : Housing, Food, Transportation, Health, Education, Debt, Emergency, Discretionary, Savings
Domains   : discretised currency amounts between each variable's floor and requested amount
Hard      : (C1) sum(allocation) <= income  (C2) Housing >= housing minimum  (C3) Debt >= minimum debt payment
Soft      : (S1) maximise emergency reserve (S2) increase savings (S3) discretionary <= 20% of net income
Search    : recursive backtracking + forward checking + MRV, with branch-and-bound on the soft objective
            (values tried best-first, LP-relaxation upper bound prunes branches that cannot beat the incumbent).
"""
from dataclasses import dataclass, field

VARS = ["Housing", "Food", "Transportation", "Health", "Education", "Debt", "Emergency", "Discretionary", "Savings"]
# soft-constraint utility weight per variable (value / requested * weight)
WEIGHTS = {"Housing": 0.0, "Food": 2.5, "Transportation": 2.5, "Health": 2.5, "Education": 2.5, "Debt": 0.0,
           "Emergency": 2.0, "Discretionary": 0.5, "Savings": 1.5}
NICE_STEPS = [10, 50, 100, 250, 500, 1000, 2500, 5000, 10000]


@dataclass
class CSPInputs:
    income: float
    housing_min: float
    food: float
    transport: float
    health: float
    education: float
    debt: float
    emergency: float
    discretionary: float
    savings_target: float
    essential_floor_pct: float = 0.80
    disc_cap_pct: float = 0.20

    def validate(self) -> list[str]:
        errs = []
        for k, v in self.__dict__.items():
            if v is None or v != v:
                errs.append(f"{k} is missing.")
            elif v < 0:
                errs.append(f"{k} cannot be negative.")
        if self.income is not None and self.income <= 0:
            errs.append("Monthly income must be greater than 0.")
        return errs


@dataclass
class _Stats:
    nodes: int = 0
    backtracks: int = 0
    fc_prunings: int = 0
    fc_wipeouts: int = 0
    bound_prunes: int = 0
    solutions: int = 0
    truncated: bool = False


def _requested(inp: CSPInputs) -> dict:
    return {"Housing": inp.housing_min, "Food": inp.food, "Transportation": inp.transport, "Health": inp.health,
            "Education": inp.education, "Debt": inp.debt, "Emergency": inp.emergency,
            "Discretionary": min(inp.discretionary, inp.disc_cap_pct * inp.income), "Savings": inp.savings_target}


def _floors(inp: CSPInputs, req: dict) -> dict:
    f = {v: 0.0 for v in VARS}
    f["Housing"], f["Debt"] = inp.housing_min, inp.debt
    for v in ("Food", "Transportation", "Health", "Education"):
        f[v] = req[v] * inp.essential_floor_pct
    return f


def _choose_step(req, floors) -> float:
    rng = max([req[v] - floors[v] for v in VARS] + [1])
    for s in NICE_STEPS:
        if rng / s <= 20:
            return float(s)
    return float(NICE_STEPS[-1])


def _build_domains(req, floors, step) -> dict:
    dom = {}
    for v in VARS:
        lo, hi = floors[v], max(req[v], floors[v])
        vals, x = [], lo
        while x < hi - 1e-9:
            vals.append(round(x, 2))
            x += step
        vals.append(round(hi, 2))
        dom[v] = sorted(set(vals))
    return dom


def solve(inp: CSPInputs, node_limit: int = 300_000) -> dict:
    errs = inp.validate()
    if errs:
        return {"status": "invalid", "feasible": False, "errors": errs}
    income = inp.income
    req = _requested(inp)
    floors = _floors(inp, req)
    step = _choose_step(req, floors)
    domains = _build_domains(req, floors, step)
    stats = _Stats()
    trace: list[str] = []
    info = {"inputs": inp, "requested": req, "floors": floors, "step": step,
            "domain_sizes": {v: len(d) for v, d in domains.items()}, "n_variables": len(VARS),
            "disc_cap": inp.disc_cap_pct * income, "disc_trimmed": inp.discretionary > inp.disc_cap_pct * income + 1e-9}

    # ---- root consistency check (initial forward check) --------------------------------
    floor_sum = sum(floors.values())
    hard_sum = inp.housing_min + inp.debt
    if floor_sum > income + 1e-9:
        if hard_sum > income + 1e-9:
            reason = (f"Hard constraints C2+C3 cannot both hold: housing minimum + minimum debt = {hard_sum:,.0f} "
                      f"exceeds income {income:,.0f} by {hard_sum - income:,.0f}.")
        else:
            reason = (f"Even with essentials cut to their {inp.essential_floor_pct:.0%} minimum, mandatory costs total "
                      f"{floor_sum:,.0f}, exceeding income {income:,.0f} by {floor_sum - income:,.0f}.")
        return {**info, "status": "infeasible", "feasible": False, "shortfall": floor_sum - income, "reason": reason,
                "stats": stats.__dict__, "trace": ["Root forward check: sum of minimum values > income -> a domain would be wiped out."],
                "constraints": _constraint_table(inp, None, req), "explanation": [reason,
                "Suggested fix: lower essential requests, reduce debt/housing obligations, or log additional income."]}

    dens = {v: (WEIGHTS[v] / req[v] if req[v] > 0 else 0.0) for v in VARS}

    def utility(v, x):
        return WEIGHTS[v] * (x / req[v]) if req[v] > 0 else 0.0

    def upper_bound(assign, doms, remaining):
        un = [v for v in VARS if v not in assign]
        base = sum(utility(v, doms[v][0]) for v in un)
        budget = remaining - sum(doms[v][0] for v in un)
        for v in sorted(un, key=lambda z: -dens[z]):
            if budget <= 0:
                break
            add = min(budget, doms[v][-1] - doms[v][0])
            base += dens[v] * add
            budget -= add
        return base

    best = {"score": -1.0, "assign": None}

    def forward_check(assign, doms, remaining):
        un = [v for v in VARS if v not in assign]
        mins = {v: doms[v][0] for v in un}
        total_min = sum(mins.values())
        for v in un:
            limit = remaining - (total_min - mins[v])
            keep = [x for x in doms[v] if x <= limit + 1e-9]
            stats.fc_prunings += len(doms[v]) - len(keep)
            if not keep:
                stats.fc_wipeouts += 1
                return None
            doms[v] = keep
        return doms

    def mrv(assign, doms):
        un = [v for v in VARS if v not in assign]
        return min(un, key=lambda z: (len(doms[z]), VARS.index(z)))

    def recurse(assign, doms, remaining, score):
        if stats.truncated:
            return
        stats.nodes += 1
        if stats.nodes > node_limit:
            stats.truncated = True
            return
        if len(assign) == len(VARS):
            stats.solutions += 1
            if score > best["score"] + 1e-12:
                best.update(score=score, assign=dict(assign))
                if len(trace) < 40:
                    trace.append(f"New best solution #{stats.solutions}: score={score:.3f}, remaining={remaining:,.0f}")
            return
        if best["assign"] is not None and score + upper_bound(assign, doms, remaining) <= best["score"] + 1e-9:
            stats.bound_prunes += 1
            return
        var = mrv(assign, doms)
        if len(trace) < 40 and stats.nodes < 60:
            trace.append(f"Node {stats.nodes}: MRV picks '{var}' (|domain|={len(doms[var])}), remaining budget {remaining:,.0f}")
        for val in sorted(doms[var], reverse=True):
            new = {k: list(d) for k, d in doms.items()}
            new[var] = [val]
            assign[var] = val
            if forward_check(assign, new, remaining - val) is not None:
                recurse(assign, new, remaining - val, score + utility(var, val))
            else:
                stats.backtracks += 1
            del assign[var]
            if stats.truncated:
                return

    recurse({}, domains, income, 0.0)
    if best["assign"] is None:
        return {**info, "status": "infeasible", "feasible": False, "stats": stats.__dict__, "trace": trace,
                "reason": "Search exhausted without a feasible allocation.", "constraints": _constraint_table(inp, None, req),
                "explanation": ["Search exhausted all branches; no allocation satisfies the hard constraints."]}
    alloc = best["assign"]
    total = sum(alloc.values())
    explanation = _explain(inp, alloc, req, floors, stats, best["score"], step, info)
    return {**info, "status": "truncated" if stats.truncated else "optimal", "feasible": True, "allocation": alloc,
            "total": total, "remaining": income - total, "score": best["score"], "stats": stats.__dict__, "trace": trace,
            "constraints": _constraint_table(inp, alloc, req), "explanation": explanation}


def _constraint_table(inp: CSPInputs, alloc, req) -> list[dict]:
    inc = inp.income
    rows = []

    def add(cid, kind, text, ok, atlimit, detail):
        if ok is None:
            status = "Unknown"
        elif not ok:
            status = "At limit" if (kind == "Soft" and alloc is not None and cid != "S3") else "Violated"  # soft targets: income exhausted
        else:
            status = "At limit" if atlimit else "Satisfied"
        rows.append({"id": cid, "type": kind, "constraint": text, "status": status, "detail": detail})

    if alloc is None:
        for cid, text in [("C1", "Total allocation <= monthly income"), ("C2", "Housing >= mandatory minimum"), ("C3", "Debt >= minimum payment")]:
            add(cid, "Hard", text, False, False, "No feasible assignment found")
        return rows
    tot = sum(alloc.values())
    add("C1", "Hard", "Total allocation <= monthly income", tot <= inc + 1e-6, abs(tot - inc) < 1e-6, f"{tot:,.0f} vs {inc:,.0f}")
    add("C2", "Hard", "Housing >= mandatory minimum", alloc["Housing"] >= inp.housing_min - 1e-6, abs(alloc["Housing"] - inp.housing_min) < 1e-6,
        f"{alloc['Housing']:,.0f} vs min {inp.housing_min:,.0f}")
    add("C3", "Hard", "Debt >= minimum debt payment", alloc["Debt"] >= inp.debt - 1e-6, abs(alloc["Debt"] - inp.debt) < 1e-6,
        f"{alloc['Debt']:,.0f} vs min {inp.debt:,.0f}")
    em_ratio = alloc["Emergency"] / inp.emergency if inp.emergency > 0 else 1.0
    add("S1", "Soft", "Maximise emergency reserve", em_ratio >= 0.999, False, f"{alloc['Emergency']:,.0f} of target {inp.emergency:,.0f} ({em_ratio:.0%})")
    sv_ratio = alloc["Savings"] / inp.savings_target if inp.savings_target > 0 else 1.0
    add("S2", "Soft", "Increase savings (reach target)", sv_ratio >= 0.999, False, f"{alloc['Savings']:,.0f} of target {inp.savings_target:,.0f} ({sv_ratio:.0%})")
    cap = inp.disc_cap_pct * inc
    add("S3", "Soft", f"Discretionary <= {inp.disc_cap_pct:.0%} of net income", alloc["Discretionary"] <= cap + 1e-6, abs(alloc["Discretionary"] - cap) < 1e-6,
        f"{alloc['Discretionary']:,.0f} vs cap {cap:,.0f}")
    return rows


def _explain(inp, alloc, req, floors, stats, score, step, info) -> list[str]:
    ex = [f"Search evaluated {stats.nodes:,} nodes and found {stats.solutions:,} complete feasible allocation(s); this one has the highest soft-constraint score ({score:.3f})."
          if not stats.truncated else
          f"Node limit reached after {stats.nodes:,} nodes; this is the best of {stats.solutions:,} feasible allocation(s) found (not proven optimal)."]
    ex.append(f"MRV picked the most-constrained variable first (Housing and Debt have a single feasible value). Forward checking pruned {stats.fc_prunings:,} domain values "
              f"and caused {stats.backtracks:,} early backtracks; bound pruning cut {stats.bound_prunes:,} branches.")
    reduced = [v for v in ("Food", "Transportation", "Health", "Education") if alloc[v] < req[v] - 1e-6]
    if reduced:
        ex.append("Reduced (but still above the minimum acceptable level): " + ", ".join(f"{v} {alloc[v]:,.0f} (requested {req[v]:,.0f})" for v in reduced) + ". Income was not enough for every request, so the solver traded these for higher-weight goals.")
    else:
        ex.append("All essential categories were fully funded.")
    if alloc["Emergency"] >= inp.emergency - 1e-6:
        ex.append("Emergency reserve target fully met (soft constraint S1).")
    else:
        ex.append(f"Emergency reserve only {alloc['Emergency']:,.0f} of {inp.emergency:,.0f}: income is exhausted (maximised subject to the hard constraints).")
    if alloc["Savings"] < inp.savings_target - 1e-6:
        ex.append(f"Savings {alloc['Savings']:,.0f} below target {inp.savings_target:,.0f} (income is the binding resource).")
    if info["disc_trimmed"]:
        ex.append(f"Requested discretionary spending exceeded the {inp.disc_cap_pct:.0%} cap, so its domain was capped at {info['disc_cap']:,.0f} (soft constraint S3).")
    ex.append(f"Domains were discretised in steps of {step:,.0f}.")
    return ex
