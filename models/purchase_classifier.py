"""Purchase Advisor: Logistic Regression (primary) + KNN (comparison) on SYNTHETIC / DEMO data.

NOT validated on real financial outcomes - educational use only."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils.constants import CATEGORY_WEIGHT

LABELS = ["Approved", "Caution", "Rejected"]
FEATURES = ["cost_balance_ratio", "days_left", "goal_urgency", "category_weight"]
FEATURE_LABELS = {"cost_balance_ratio": "Item cost / balance ratio", "days_left": "Days left in month",
                  "goal_urgency": "Goal urgency (1-5)", "category_weight": "Category impact weight"}
SYNTHETIC_NOTICE = ("SYNTHETIC / DEMO TRAINING DATA - generated from a hand-designed risk function plus noise. "
                    "Metrics describe how well the models learn this synthetic pattern, NOT real-world financial accuracy.")


def generate_synthetic_dataset(n: int = 800, seed: int = 42) -> pd.DataFrame:
    """Synthetic labelled purchases. Label = thresholds on a noisy latent risk score."""
    rng = np.random.default_rng(seed)
    cats = rng.choice(list(CATEGORY_WEIGHT), size=n)
    w = np.array([CATEGORY_WEIGHT[c] for c in cats])
    ratio = np.exp(rng.uniform(np.log(0.01), np.log(2.0), n))
    days = rng.integers(1, 31, n)
    urg = rng.integers(1, 6, n)
    score = 2.4 * np.clip(ratio, 0, 2) + 0.5 * days / 30 + 0.5 * (urg - 1) / 4 + 0.4 * w + rng.normal(0, 0.15, n)
    y = np.where(score < 1.0, "Approved", np.where(score < 1.8, "Caution", "Rejected"))
    return pd.DataFrame({"cost_balance_ratio": ratio, "days_left": days, "goal_urgency": urg,
                         "category_weight": w, "category": cats, "label": y, "is_synthetic": True})


def _metrics(y_true, y_pred) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "precision": float(pw), "recall": float(rw), "f1": float(fw),
            "precision_macro": float(pm), "recall_macro": float(rm), "f1_macro": float(fm),
            "per_class": pd.DataFrame({"class": LABELS, "precision": p, "recall": r, "f1": f, "support": s}),
            "confusion": confusion_matrix(y_true, y_pred, labels=LABELS), "n_test": int(len(y_true))}


def train_models(n: int = 800, seed: int = 42, k: int = 5, test_size: float = 0.25) -> dict:
    """Train both classifiers. Raises ValueError with a readable message when data is too small."""
    if n < 30:
        raise ValueError("Insufficient data for reliable training: at least 30 samples are required.")
    df = generate_synthetic_dataset(n, seed)
    counts = df["label"].value_counts()
    notes = []
    if counts.nunique() and len(counts) < 2:
        raise ValueError("Synthetic data contains fewer than 2 classes; cannot train a classifier.")
    stratify = df["label"] if (counts.min() >= 2 and int(n * test_size) >= len(counts)) else None
    if stratify is None:
        notes.append("Stratified split was not possible (very small class); using a plain random split.")
    Xtr, Xte, ytr, yte = train_test_split(df[FEATURES], df["label"], test_size=test_size, random_state=seed, stratify=stratify)
    k_eff = int(max(1, min(k, len(Xtr))))
    if k_eff != k:
        notes.append(f"k reduced from {k} to {k_eff} because the training set is small.")
    if len(yte) < 30:
        notes.append("Test set has fewer than 30 rows - metrics are unreliable.")
    lr = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, random_state=seed))]).fit(Xtr, ytr)
    knn = Pipeline([("scale", MinMaxScaler()), ("clf", KNeighborsClassifier(n_neighbors=k_eff, weights="distance"))]).fit(Xtr, ytr)
    return {"data": df, "X_train": Xtr, "y_train": ytr, "X_test": Xte, "y_test": yte, "notes": notes, "k": k_eff, "n": n, "seed": seed,
            "models": {"Logistic Regression": lr, "KNN": knn},
            "metrics": {"Logistic Regression": _metrics(yte, lr.predict(Xte)), "KNN": _metrics(yte, knn.predict(Xte))},
            "class_counts": counts.reindex(LABELS, fill_value=0).to_dict()}


def build_features(price: float, balance: float, days_left: int, urgency: int, category: str) -> dict:
    if balance <= 0:
        ratio = 5.0  # no liquid balance: treat as an extreme ratio (clipped in training range anyway)
    else:
        ratio = price / balance
    return {"cost_balance_ratio": float(ratio), "days_left": int(days_left), "goal_urgency": int(urgency),
            "category_weight": float(CATEGORY_WEIGHT.get(category, 0.5))}


def predict(bundle: dict, name: str, feats: dict) -> dict:
    model = bundle["models"][name]
    X = pd.DataFrame([feats])[FEATURES]
    proba = model.predict_proba(X)[0]
    full = {c: 0.0 for c in LABELS}
    for c, p in zip(model.classes_, proba):
        full[c] = float(p)
    pred = max(full, key=full.get)
    exp_risk = 0.5 * full["Caution"] + 1.0 * full["Rejected"]
    level = "Low" if exp_risk < 0.25 else "Medium" if exp_risk < 0.6 else "High"
    return {"model": name, "predicted": pred, "probabilities": full, "risk_level": level, "expected_risk": exp_risk}


def explain_lr(bundle: dict, feats: dict, predicted: str) -> pd.DataFrame:
    """Per-feature log-odds contribution (coef x standardised value) for the predicted class."""
    pipe = bundle["models"]["Logistic Regression"]
    clf, sc = pipe.named_steps["clf"], pipe.named_steps["scale"]
    z = sc.transform(pd.DataFrame([feats])[FEATURES])[0]
    ci = list(clf.classes_).index(predicted) if predicted in clf.classes_ else 0
    contrib = clf.coef_[ci] * z
    return pd.DataFrame({"feature": [FEATURE_LABELS[f] for f in FEATURES], "value": [feats[f] for f in FEATURES],
                         "coefficient": clf.coef_[ci], "contribution": contrib})


def neighbours(bundle: dict, feats: dict) -> pd.DataFrame:
    pipe = bundle["models"]["KNN"]
    Xq = pd.DataFrame([feats])[FEATURES]
    sc, clf = pipe.named_steps["scale"], pipe.named_steps["clf"]
    dist, idx = clf.kneighbors(sc.transform(Xq))
    nb = bundle["X_train"].iloc[idx[0]].copy()
    nb["label"] = bundle["y_train"].iloc[idx[0]].values
    nb["distance"] = dist[0]
    return nb.reset_index(drop=True)


def what_if(bundle: dict, feats: dict, price: float, balance: float, days_left, urgency, category, name="Logistic Regression") -> dict:
    """Model-driven what-if: largest price cut (or smallest balance) at which the model predicts Approved."""
    out = {"price_for_approved": None, "balance_for_approved": None}
    if balance > 0:
        for m in np.linspace(1.0, 0.02, 99):
            if predict(bundle, name, build_features(price * m, balance, days_left, urgency, category))["predicted"] == "Approved":
                out["price_for_approved"] = float(price * m)
                break
    for m in np.linspace(1.0, 20.0, 191):
        b = max(balance, 1.0) * m
        if predict(bundle, name, build_features(price, b, days_left, urgency, category))["predicted"] == "Approved":
            out["balance_for_approved"] = float(b)
            break
    return out
