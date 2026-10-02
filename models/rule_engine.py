"""Modular IF-THEN rule engine. Pure Python: no UI code. Each rule reads ONE metric from a metrics dict
and compares it to a configurable threshold."""
from dataclasses import dataclass
from typing import Callable

PRIORITY_ORDER = {"High": 3, "Medium": 2, "Low": 1, "Info": 0}


@dataclass(frozen=True)
class Rule:
    rule_id: str
    name: str
    metric: str                 # key in metrics dict
    op: str                     # '<' or '>' or '>='
    threshold_key: str          # key in config (thresholds)
    fmt: str                    # 'pct' or 'money'
    kind: str                   # 'warning' | 'positive'
    priority: str
    recommendation: str
    explanation: str
    threshold_fn: Callable[[dict, dict], float] | None = None   # optional derived threshold


def _fmt(v, fmt):
    if v is None:
        return "n/a"
    return f"{v * 100:.1f}%" if fmt == "pct" else f"৳{v:,.0f}"


def _cmp(v, op, t):
    return v < t if op == "<" else v > t if op == ">" else v >= t


RULES = [
    Rule("R01", "Low savings ratio", "savings_ratio", "<", "thr_savings_min", "pct", "warning", "High",
         "Increase savings: aim to set aside at least {thr} of net income (e.g. automate a transfer on payday).",
         "Savings ratio = (net income - expenses) / net income. Below the minimum, little buffer is left for shocks."),
    Rule("R02", "High discretionary spending", "discretionary_ratio", ">", "thr_discretionary_max", "pct", "warning", "Medium",
         "Reduce discretionary spending (entertainment / dining out) to below {thr} of total expenses.",
         "Discretionary ratio = discretionary expenses / total expenses."),
    Rule("R03", "Dangerous burn rate", "burn_income_ratio", ">", "thr_burn_income_max", "pct", "warning", "High",
         "High-spending warning: the 7-day burn rate projects spending above {thr} of monthly income. Pause non-essential purchases.",
         "Burn rate = rolling 7-day average daily spend x 30, compared with net monthly income."),
    Rule("R04", "Emergency reserve below target", "emergency_reserve", "<", "emergency_target", "money", "warning", "High",
         "Increase emergency savings until the reserve reaches {thr} (target months x average essential spending).",
         "Emergency target = configured months of average essential monthly expenses."),
    Rule("R05", "Heavy debt burden", "debt_ratio", ">", "thr_debt_burden_max", "pct", "warning", "High",
         "Debt burden warning: debt payments exceed {thr} of income. Avoid new credit and consider restructuring or prepaying the costliest debt.",
         "Debt ratio = monthly debt payment / net monthly income."),
    Rule("R06", "Healthy savings ratio", "savings_ratio", ">=", "thr_savings_healthy", "pct", "positive", "Info",
         "Positive: your savings ratio is healthy. Keep it up and consider directing the surplus to goals or investments.",
         "Savings ratio at or above the healthy threshold."),
]


def build_metrics(monthly_row: dict, settings: dict, essential_monthly_avg: float | None) -> dict:
    """Assemble the metrics dict the rules read from real data + user settings."""
    inc = monthly_row.get("net_income")
    burn = monthly_row.get("burn_rate")
    debt = settings.get("monthly_debt_payment", 0.0)
    return {
        "savings_ratio": monthly_row.get("savings_ratio"),
        "discretionary_ratio": monthly_row.get("discretionary_ratio"),
        "burn_income_ratio": (burn / inc) if (burn is not None and inc) else None,
        "emergency_reserve": settings.get("emergency_reserve"),
        "debt_ratio": (debt / inc) if inc else None,
    }


def evaluate(metrics: dict, settings: dict, essential_monthly_avg: float | None) -> list[dict]:
    cfg = dict(settings)
    cfg["emergency_target"] = (settings.get("emergency_target_months", 3.0) * essential_monthly_avg) if essential_monthly_avg is not None else None
    results = []
    for r in RULES:
        value, thr = metrics.get(r.metric), cfg.get(r.threshold_key)
        base = {"rule_id": r.rule_id, "name": r.name, "condition": f"{r.metric.replace('_', ' ').title()} {r.op} {_fmt(thr, r.fmt)}",
                "current": _fmt(value, r.fmt), "threshold": _fmt(thr, r.fmt), "kind": r.kind, "priority": r.priority,
                "explanation": r.explanation, "metric": r.metric, "value": value, "threshold_value": thr}
        if value is None or thr is None:
            results.append({**base, "status": "Not evaluable", "triggered": False, "recommendation": "Not enough data to evaluate this rule (see Settings / add transactions)."})
            continue
        trig = _cmp(value, r.op, thr)
        results.append({**base, "status": "Triggered" if trig else "Not triggered", "triggered": bool(trig),
                        "recommendation": r.recommendation.format(thr=_fmt(thr, r.fmt)) if trig else "No action needed."})
    return results


def triggered_sorted(results: list[dict]) -> list[dict]:
    return sorted([r for r in results if r["triggered"]], key=lambda r: -PRIORITY_ORDER[r["priority"]])
