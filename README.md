# Smart Concrete Calculator

An applied project built around the day-to-day work of a ready-mix
concrete company (inspired by **Renoa**, part of Al Bassami Group). It
combines a smart quantity/price calculator with a real machine learning
model that predicts concrete compressive strength from mix design.

## Project overview

The project has three layers, each in its own folder:

```
renoa-concrete-project/
├── notebook/    ← trains and evaluates the strength-prediction model
├── api/         ← FastAPI service that serves the trained model
└── frontend/    ← calculator page (demo version + API-connected version)
```

**Data flow:**

```
Concrete dataset (Kaggle)
        │
        ▼
notebook/  ← cleaning + EDA + Gradient Boosting training + evaluation (R², RMSE, MAE, CV)
        │
        ▼  (saves concrete_model.pkl + concrete_scaler.pkl)
api/       ← FastAPI loads both files and exposes POST /predict
        │
        ▼  (fetch)
frontend/  ← web page: quantity/price calculator + strength prediction form
```

## 1. `notebook/` — the model

Predicts 28-day concrete compressive strength (MPa) from 8 mix
variables: cement, blast furnace slag, fly ash, water, superplasticizer,
coarse aggregate, fine aggregate, and age.

**Final results:**

| Metric | Value |
|---|---|
| R² (single train/test split) | 0.9434 |
| RMSE | 4.11 MPa |
| MAE | 2.60 MPa |
| **R² across 5-fold cross-validation** | **0.9372 (± 0.0081)** |

The notebook documents a real issue found while building it: the first
cross-validation attempt came back unstable (mean R² of 0.45, with one
fold scoring -0.80), because `cross_val_score(cv=5)` does not shuffle
data by default. The fix was to use an explicit `KFold(shuffle=True)`,
after which the scores stabilized. Full details are in the notebook
itself.

**Stack:** pandas, scikit-learn (GradientBoostingRegressor), joblib.

## 2. `api/` — the service

A small FastAPI app that loads `concrete_model.pkl` and
`concrete_scaler.pkl` (produced by the notebook) and exposes:

- `GET /` — health check.
- `POST /predict` — takes mix quantities, returns the predicted
  strength.
- `GET /docs` — interactive Swagger UI to try the API directly.

Full setup instructions are in [`api/README.md`](./api/README.md), or
in short:

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## 3. `frontend/` — the interface

Two standalone HTML pages (no framework, just HTML/CSS/JS):

- **`renoa-concrete-demo.html`** — the quick-demo version, works in any
  browser instantly with no setup. The strength-prediction section uses
  a simplified approximation formula (not the real model), and the
  calculator section can parse a free-text order description with AI.
- **`renoa-concrete-demo-local.html`** — the same page, but the
  prediction section is actually connected to the real API (`fetch` to
  `http://localhost:8000`), with an automatic fallback to the
  approximation formula if the connection fails.

Both pages include a quantity/price calculator (multiple items,
editable price per m³, estimated truck count) and a field to analyze an
Arabic order description automatically.

## Running it all locally

1. Run `notebook/Concrete_Compressive_Strength_fixed.ipynb` end to end
   (Colab or Jupyter) — it downloads the dataset, trains the model, and
   saves `concrete_model.pkl` and `concrete_scaler.pkl`.
2. Place both files inside the `api/` folder.
3. `cd api && pip install -r requirements.txt && uvicorn main:app --reload --port 8000`
4. Open `frontend/renoa-concrete-demo-local.html` directly in a
   browser.

## Why this project

Built as a portfolio project to apply as an intern at a ready-mix
concrete company, to demonstrate the ability to:

- Understand a real operational problem (estimating and pricing
  concrete pours).
- Build and evaluate a machine learning model with sound methodology
  (correct train/test split, avoiding data leakage, cross-validation —
  not just a single accuracy number).
- Connect the model to a real service (API) instead of leaving it
  stuck in a notebook.
- Build an actual interface anyone can try.

## Tech stack

Python · pandas · scikit-learn · FastAPI · joblib · HTML/CSS/JavaScript
