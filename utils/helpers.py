"""Streamlit UI helpers: theme CSS, cards, chips, tooltips, Plotly styling."""
import html

import plotly.graph_objects as go
import streamlit as st

from utils.constants import AMBER, BONE, BRICK, CURRENCY, INK, INK2, JADE, MUTED, STEEL

GLOSSARY = {
    "K-Means": "Unsupervised clustering: groups similar spending windows around learned centroids by minimising within-cluster distance (inertia).",
    "Silhouette": "Cluster separation score from -1 to 1. Higher = tighter, better-separated clusters.",
    "Inertia": "Sum of squared distances from points to their cluster centroid. Lower = tighter clusters.",
    "Logistic Regression": "Supervised classifier that models class probabilities with a softmax over a weighted sum of the features.",
    "KNN": "K-Nearest Neighbours: classifies by majority (distance-weighted) vote of the k most similar training examples.",
    "MinMax": "Scales each feature to [0,1] so no feature dominates distance-based models.",
    "CSP": "Constraint Satisfaction Problem: find variable values that satisfy all hard constraints while optimising soft ones.",
    "Backtracking": "Depth-first search that assigns one variable at a time and undoes (backtracks) on dead ends.",
    "Forward Checking": "After each assignment, removes inconsistent values from the domains of unassigned variables; backtracks early if a domain is empty.",
    "MRV": "Minimum Remaining Values: always assign next the variable with the fewest legal values left (fail-first).",
    "A*": "Best-first search ordered by f(n) = g(n) + h(n); optimal when h never overestimates.",
    "Heuristic": "h(n) estimates remaining cost. Admissible = never overestimates; consistent = h(n) <= step cost + h(n').",
    "Savings Ratio": "(Net income - expenses) / net income.",
    "Burn Rate": "Rolling 7-day average daily spend extrapolated to 30 days.",
    "Discretionary Ratio": "Discretionary (non-essential) expenses / total expenses.",
    "Precision": "Of the items predicted as a class, the share that were truly that class.",
    "Recall": "Of the items that truly are a class, the share the model found.",
    "F1": "Harmonic mean of precision and recall.",
    "Confusion Matrix": "Rows = actual class, columns = predicted class.",
}


def term(name: str, label: str | None = None) -> str:
    return f'<span class="fw-term" title="{html.escape(GLOSSARY.get(name, name))}">{html.escape(label or name)}</span>'


def money(x, dec=2) -> str:
    return "n/a" if x is None else f"{CURRENCY}{x:,.{dec}f}"


def pct(x, dec=1) -> str:
    return "n/a" if x is None else f"{x * 100:.{dec}f}%"


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'Inter', system-ui, sans-serif; }}
h1, h2, h3, h4 {{ font-family: 'Space Grotesk', 'Inter', sans-serif !important; letter-spacing: -0.01em; }}
h1 {{ font-size: 2rem !important; font-weight: 700 !important; }}
.fw-mono, .fw-num {{ font-family: 'JetBrains Mono', ui-monospace, monospace; font-variant-numeric: tabular-nums; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
section[data-testid="stSidebar"] {{ background: {INK2}; border-right: 1px solid #222a35; }}
.fw-brand {{ font-family:'Space Grotesk'; font-weight:700; font-size:1.35rem; color:{BONE}; }}
.fw-brand span {{ color:{AMBER}; }}
.fw-sub {{ color:{MUTED}; font-size:.85rem; margin-top:-.3rem; }}
.fw-eyebrow {{ font-family:'JetBrains Mono'; font-size:.72rem; color:{AMBER}; text-transform:uppercase; letter-spacing:.08em; }}
.fw-card {{ background:{INK2}; border:1px solid #232b36; border-radius:12px; padding:1rem 1.15rem; height:100%; }}
.fw-kpi-label {{ color:{MUTED}; font-size:.78rem; text-transform:uppercase; letter-spacing:.05em; }}
.fw-kpi-value {{ font-family:'JetBrains Mono'; font-variant-numeric:tabular-nums; font-size:1.55rem; font-weight:500; margin:.15rem 0; color:{BONE}; }}
.fw-kpi-sub {{ font-size:.78rem; color:{MUTED}; font-family:'JetBrains Mono'; }}
.fw-chip {{ display:inline-block; font-family:'JetBrains Mono'; font-size:.75rem; padding:.18rem .6rem; border-radius:6px; border:1px solid; white-space:nowrap; }}
.fw-chip.jade {{ color:{JADE}; border-color:{JADE}; background:rgba(76,159,112,.10); }}
.fw-chip.amber {{ color:{AMBER}; border-color:{AMBER}; background:rgba(238,150,75,.10); }}
.fw-chip.brick {{ color:{BRICK}; border-color:{BRICK}; background:rgba(193,68,94,.10); }}
.fw-chip.steel {{ color:#8FB0DB; border-color:{STEEL}; background:rgba(61,90,128,.20); }}
.fw-chip.muted {{ color:{MUTED}; border-color:#39414d; }}
.fw-ai {{ background:rgba(61,90,128,.14); border:1px solid {STEEL}; border-radius:10px; padding:.9rem 1.1rem; position:relative; margin:.5rem 0 1rem; }}
.fw-ai .tag {{ position:absolute; top:.55rem; right:.8rem; font-family:'JetBrains Mono'; font-size:.68rem; color:#8FB0DB; }}
.fw-ai .ttl {{ font-weight:600; font-size:.9rem; margin-bottom:.25rem; }}
.fw-term {{ border-bottom:1px dotted {AMBER}; cursor:help; }}
.fw-empty {{ border:1px dashed #39414d; border-radius:12px; padding:1.6rem; text-align:center; color:{MUTED}; }}
.fw-flow {{ display:flex; flex-wrap:wrap; gap:.4rem; align-items:center; margin:.4rem 0 1rem; }}
.fw-flow .st {{ background:{INK2}; border:1px solid {STEEL}; padding:.35rem .7rem; border-radius:8px; font-family:'JetBrains Mono'; font-size:.75rem; }}
.fw-flow .ar {{ color:{AMBER}; }}
.fw-prob {{ font-family:'JetBrains Mono'; font-size:2.6rem; font-weight:500; }}
.fw-warn {{ font-size:.78rem; color:{MUTED}; border-left:3px solid {AMBER}; padding:.3rem .7rem; margin:.4rem 0; }}
.fw-tl {{ display:flex; gap:.6rem; overflow-x:auto; padding-bottom:.5rem; }}
.fw-tl .m {{ min-width:118px; background:{INK2}; border:1px solid #232b36; border-radius:10px; padding:.55rem .7rem; }}
.fw-tl .m .n {{ font-family:'JetBrains Mono'; font-size:.68rem; color:{AMBER}; }}
div[data-testid="stMetric"] {{ background:{INK2}; border:1px solid #232b36; border-radius:10px; padding:.6rem .9rem; }}
@media (max-width: 768px) {{ h1 {{ font-size:1.5rem !important; }} .fw-kpi-value {{ font-size:1.25rem; }} .fw-prob {{ font-size:2rem; }} }}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str, eyebrow: str = ""):
    if eyebrow:
        st.markdown(f'<div class="fw-eyebrow">{html.escape(eyebrow)}</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(subtitle)


def section(title: str, sub: str | None = None):
    st.markdown(f"### {title}")
    if sub:
        st.caption(sub)


def chip(text: str, tone: str = "steel") -> str:
    return f'<span class="fw-chip {tone}">{html.escape(text)}</span>'


VERDICT_TONE = {"Approved": "jade", "Caution": "amber", "Rejected": "brick", "Satisfied": "jade", "At limit": "amber", "Violated": "brick",
                "Triggered": "amber", "Not triggered": "muted", "Not evaluable": "muted", "Low": "jade", "Medium": "amber", "High": "brick"}


def tone_chip(text: str) -> str:
    return chip(text, VERDICT_TONE.get(text, "steel"))


def kpi_card(label: str, value: str, sub: str = "", tone: str | None = None, tip: str | None = None):
    color = {"jade": JADE, "amber": AMBER, "brick": BRICK}.get(tone or "", BONE)
    tt = f' title="{html.escape(tip)}"' if tip else ""
    st.markdown(f'<div class="fw-card"{tt}><div class="fw-kpi-label">{html.escape(label)}</div>'
                f'<div class="fw-kpi-value" style="color:{color}">{html.escape(value)}</div>'
                f'<div class="fw-kpi-sub">{html.escape(sub) if sub else "&nbsp;"}</div></div>', unsafe_allow_html=True)


def ai_panel(title: str, body_html: str, tag: str = "AI"):
    st.markdown(f'<div class="fw-ai"><span class="tag">{html.escape(tag)}</span><div class="ttl">{html.escape(title)}</div>{body_html}</div>', unsafe_allow_html=True)


def empty_state(message: str, hint: str = ""):
    st.markdown(f'<div class="fw-empty"><div style="font-size:1.05rem;color:{BONE}">{html.escape(message)}</div>'
                f'<div style="margin-top:.3rem">{html.escape(hint)}</div></div>', unsafe_allow_html=True)


def pipeline_flow(steps: list[str]):
    st.markdown('<div class="fw-flow">' + '<span class="ar">→</span>'.join(f'<span class="st">{html.escape(s)}</span>' for s in steps) + "</div>", unsafe_allow_html=True)


def disclaimer(text: str):
    st.markdown(f'<div class="fw-warn">{html.escape(text)}</div>', unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 340, legend: bool = True) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif", color=BONE, size=12), showlegend=legend,
                      title=dict(font=dict(family="Space Grotesk, sans-serif", size=15), x=0.01),
                      legend=dict(orientation="h", y=-0.2, font=dict(size=11)),
                      colorway=[STEEL, AMBER, JADE, BRICK, "#8F7BC6", "#5BA8A0"])
    fig.update_xaxes(gridcolor="#222a35", zerolinecolor="#222a35")
    fig.update_yaxes(gridcolor="#222a35", zerolinecolor="#222a35")
    return fig


def no_data_gate(state: dict) -> bool:
    """Show an empty state and return True if there are no usable transactions."""
    if not state["ready"]:
        empty_state("No transactions logged yet. Add your first one to start tracking.",
                    "Go to 'Income & Expenses' to add a transaction, upload a CSV, or load the demo sample data.")
        return True
    return False


def confusion_heatmap(cm, labels, title: str) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=cm, x=labels, y=labels, colorscale=[[0, INK2], [1, STEEL]], text=cm, texttemplate="%{text}", showscale=False,
                               hovertemplate="actual %{y}, predicted %{x}: %{z}<extra></extra>"))
    fig.update_layout(title=title, xaxis_title="Predicted", yaxis_title="Actual", yaxis=dict(autorange="reversed"))
    return style_fig(fig, 320, False)
