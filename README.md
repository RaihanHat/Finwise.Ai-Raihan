# FinWise AI – Intelligent Personal Financial Optimizer & Purchase Advisor
CSE 3812 AI Lab · Group 10 · Team ShoshtaEngineers

A working Streamlit application where every number is computed from one central transaction dataset
(`data/transactions.csv`). Educational prototype – **not financial advice**.

## How to run
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Optional end-to-end smoke test (backs up and restores `data/`): `python tests/e2e_test.py`

## Required packages
Python 3.10+, streamlit>=1.50, pandas, numpy, scikit-learn, plotly (see `requirements.txt`).

## Project structure
```
finwise_ai/
├── app.py                      navigation (st.navigation), theme, sidebar
├── requirements.txt · README.md
├── .streamlit/config.toml      dark theme (Ink/Steel/Amber/Jade/Brick tokens)
├── data/  transactions.csv (live store) · sample_data.csv (DEMO data) · settings.json (created on save)
├── models/                     NO UI code
│   ├── preprocessing.py        missing values, encoding, MinMax, feature engineering, derived metrics
│   ├── spending_clustering.py  K-Means (k=3) spending profiles + drift
│   ├── purchase_classifier.py  Logistic Regression (primary) + KNN (comparison), synthetic data
│   ├── budget_csp.py           CSP: backtracking + forward checking + MRV (+ branch & bound)
│   ├── goal_astar.py           A* (heapq) goal planner
│   ├── rule_engine.py          IF-THEN rules R01–R06
│   └── guidance.py             integrated, traceable report builder
├── pages/                      one render() per page (Streamlit UI only)
├── utils/  constants.py · data_loader.py (CRUD, CSV import, settings) · calculations.py · helpers.py (CSS, cards, charts)
├── tests/e2e_test.py
└── assets/logo.svg
```

## Main algorithms
| Module | Algorithm | Notes |
|---|---|---|
| Data processing | Imputation, one-hot + target encoding, MinMaxScaler | Savings ratio, 7-day burn rate ×30, discretionary ratio |
| Spending profile | K-Means (k=3) on trailing 30-day windows | profile of latest window via `KMeans.predict`; names from centroid risk rank |
| Purchase advisor | Logistic Regression (primary), KNN (k=5, MinMax) | 4 features; trained on **synthetic/demo** data |
| Budget optimizer | CSP: recursive backtracking, forward checking, MRV, LP-bound branch & bound | 9 variables, 3 hard + 3 soft constraints |
| Rule advisor | Modular IF-THEN rule engine | each result has rule id, condition, current, threshold, priority |
| Goal planner | A* with `heapq`; f=g+h; h=⌈remaining/max saving⌉ | g = months + λ·(disruption)² ; admissible & consistent |
| Guidance | Aggregator | every item tagged ML / rule / CSP / A* / metric |

## Known limitations
* Purchase-risk classifiers are trained on **synthetic data** produced by a hand-made risk function; metrics show how well they learn
  that pattern, **not** real-world accuracy. Do not use for real financial decisions.
* K-Means clusters one user's overlapping 30-day windows (needs ≈ 30+ days of data); clusters describe your own regimes, not a population.
  Cluster names come from ranking centroid risk scores.
* Loan proceeds count toward balance but are excluded from "net income" in ratios. Entertainment (and dining-out food) is auto-tagged discretionary (editable per transaction).
* CSP amounts are discretised (domain step chosen automatically), so a small unallocated remainder can appear; the search has a node limit and reports if it is hit.
* A* uses a small discrete set of monthly saving levels and a quadratic disruption penalty (weight λ is a modelling choice); no interest or income variation.
* Emergency reserve, monthly debt payment and opening balance are manual inputs in Settings (default 0).
* Imputation of missing amounts (category mean) invents values by design – flagged in the pipeline (`amount_imputed`).
* Session results (purchase, CSP, A*) used by the Guidance page live in the Streamlit session and reset on browser refresh.
