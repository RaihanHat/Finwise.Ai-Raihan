"""K-Means spending-profile model.

Because one user = one trajectory, the model clusters the user's own trailing 30-day spending
windows (stepped every 3 days). The CURRENT profile is the cluster the most recent window belongs to,
obtained with KMeans.predict (not if/else rules). Cluster NAMES are assigned from the learned
centroids' composite risk score (ranked low -> high)."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

from models.preprocessing import daily_expense_series
from utils.constants import NON_INCOME_FOR_RATIOS

FEATURES = ["tx_frequency", "avg_tx_amount", "savings_ratio", "discretionary_ratio", "spending_variance", "burn_rate"]
FEATURE_LABELS = {"tx_frequency": "Transactions / week", "avg_tx_amount": "Avg expense amount",
                  "savings_ratio": "Savings ratio", "discretionary_ratio": "Discretionary ratio",
                  "spending_variance": "Daily spend std-dev", "burn_rate": "Burn rate (30d)"}
PROFILE_NAMES = ["Disciplined / Frugal", "Balanced / Moderate", "Impulsive / High-Risk"]
WINDOW_DAYS, STEP_DAYS, MIN_DAYS, MIN_WINDOWS, K = 30, 3, 14, 6, 3


def build_windows(clean: pd.DataFrame, window_days=WINDOW_DAYS, step_days=STEP_DAYS, min_days=MIN_DAYS) -> pd.DataFrame:
    if clean.empty:
        return pd.DataFrame(columns=["window_start", "window_end"] + FEATURES)
    first, last = clean["date"].min().normalize(), clean["date"].max().normalize()
    daily = daily_expense_series(clean)
    ends, e = [], last
    while e >= first + pd.Timedelta(days=min_days - 1):
        ends.append(e)
        e -= pd.Timedelta(days=step_days)
    rows = []
    for end in reversed(ends):
        start = max(first, end - pd.Timedelta(days=window_days - 1))
        g = clean[(clean["date"] >= start) & (clean["date"] <= end + pd.Timedelta(hours=23, minutes=59))]
        exp = g[g["type"] == "Expense"]
        inc = g.loc[(g["type"] == "Income") & ~g["category"].isin(NON_INCOME_FOR_RATIOS), "amount"].sum()
        exp_total = exp["amount"].sum()
        disc = exp.loc[exp["is_discretionary"] == 1, "amount"].sum()
        d = daily[(daily.index >= start) & (daily.index <= end)]
        days = (end - start).days + 1
        sr = (inc - exp_total) / inc if inc > 0 else (-1.0 if exp_total > 0 else 0.0)
        rows.append({
            "window_start": start, "window_end": end,
            "tx_frequency": len(exp) / (days / 7),
            "avg_tx_amount": exp["amount"].mean() if len(exp) else 0.0,
            "savings_ratio": float(np.clip(sr, -1, 1)),
            "discretionary_ratio": disc / exp_total if exp_total > 0 else 0.0,
            "spending_variance": float(d.std(ddof=0)) if len(d) else 0.0,
            "burn_rate": float(d.tail(7).mean() * 30) if len(d) else 0.0})
    return pd.DataFrame(rows)


def _risk(scaled: np.ndarray) -> np.ndarray:
    """Composite risk of (scaled) points: low savings, high discretionary, high variance, high burn."""
    i = {f: FEATURES.index(f) for f in FEATURES}
    return np.mean([1 - scaled[:, i["savings_ratio"]], scaled[:, i["discretionary_ratio"]],
                    scaled[:, i["spending_variance"]], scaled[:, i["burn_rate"]]], axis=0)


def fit_profile(clean: pd.DataFrame, seed: int = 42) -> dict:
    wins = build_windows(clean)
    if clean.empty:
        return {"ready": False, "reason": "No transactions available."}
    span = (clean["date"].max() - clean["date"].min()).days + 1
    if len(wins) < MIN_WINDOWS:
        need = MIN_DAYS + STEP_DAYS * (MIN_WINDOWS - 1) - span
        return {"ready": False, "windows": len(wins), "reason": f"Spending profile needs more history: {len(wins)} analysis windows built, "
                f"{MIN_WINDOWS} required. About {max(need, 1)} more day(s) of transactions required."}
    X = wins[FEATURES].to_numpy(float)
    if len(np.unique(np.round(X, 6), axis=0)) < K:
        return {"ready": False, "windows": len(wins), "reason": "Spending behaviour is too uniform to form 3 distinct clusters."}
    scaler = MinMaxScaler().fit(X)
    Xs = scaler.transform(X)
    km = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(Xs)
    cent_risk = _risk(km.cluster_centers_)
    order = np.argsort(cent_risk)                       # low -> high risk
    name_of = {int(c): PROFILE_NAMES[r] for r, c in enumerate(order)}
    rank_of = {int(c): r for r, c in enumerate(order)}
    wins = wins.assign(cluster=km.labels_)
    wins["profile"] = wins["cluster"].map(name_of)
    wins["risk_rank"] = wins["cluster"].map(rank_of)
    wins["risk_score"] = _risk(Xs)
    cur = Xs[-1:]
    dists = np.linalg.norm(km.cluster_centers_ - cur, axis=1)
    cur_cluster = int(km.predict(cur)[0])
    pca = PCA(n_components=2, random_state=seed).fit(Xs)
    coords, cent_xy, cur_xy = pca.transform(Xs), pca.transform(km.cluster_centers_), pca.transform(cur)[0]
    try:
        sil = float(silhouette_score(Xs, km.labels_))
    except ValueError:
        sil = None
    centroids = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=FEATURES)
    centroids.insert(0, "profile", [name_of[i] for i in range(K)])
    centroids["windows"] = np.bincount(km.labels_, minlength=K)
    centroids["risk_score"] = cent_risk
    centroids = centroids.sort_values("risk_score").reset_index(drop=True)
    # risk drift: recent vs preceding windows
    drift = {"available": False}
    n = len(wins)
    if n >= 8:
        m = max(2, n // 4)
        recent, prev = wins["risk_rank"].iloc[-m:].mean(), wins["risk_rank"].iloc[-2 * m:-m].mean()
        delta = float(recent - prev)
        drift = {"available": True, "delta": delta, "recent": float(recent), "previous": float(prev), "m": m,
                 "direction": "toward High-Risk" if delta > 0.5 else "toward Disciplined" if delta < -0.5 else "stable"}
    cur_row = wins.iloc[-1]
    return {"ready": True, "windows_df": wins, "model": km, "scaler": scaler, "k": K,
            "current_cluster": cur_cluster, "current_profile": name_of[cur_cluster], "current_rank": rank_of[cur_cluster],
            "current_features": cur_row[FEATURES].astype(float).to_dict(), "current_scaled": dict(zip(FEATURES, cur[0])),
            "distances": {name_of[i]: float(d) for i, d in enumerate(dists)}, "centroids": centroids,
            "centroids_scaled": pd.DataFrame(km.cluster_centers_, columns=FEATURES).assign(profile=[name_of[i] for i in range(K)]),
            "inertia": float(km.inertia_), "silhouette": sil, "distribution": wins["profile"].value_counts().to_dict(),
            "coords": coords, "centroid_xy": cent_xy, "current_xy": cur_xy, "name_of": name_of, "cluster_of_name": {v: k for k, v in name_of.items()},
            "pca_var": pca.explained_variance_ratio_.tolist(), "drift": drift, "n_windows": n, "seed": seed,
            "window_desc": f"trailing {WINDOW_DAYS}-day windows, step {STEP_DAYS} days"}
