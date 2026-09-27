# Customer Behavior Prediction Platform

A machine learning & deep learning system for churn prediction, customer
lifetime value estimation, and customer segmentation, built for Vantara
Retail Solutions against the real UCI "Online Retail II" dataset.

## What this is

An end-to-end pipeline: raw transaction data → cleaning/validation →
feature engineering → classical ML + deep learning models → customer
segmentation → SHAP/LIME explainability → a FastAPI scoring service → a
Streamlit dashboard. Runs either directly (Python venv) or fully
containerized (Docker Compose with Postgres).

## Quickstart — direct run

```bash
python -m venv .venv
source .venv/bin/activate          # fish: source .venv/bin/activate.fish
pip install -r requirements.txt

# Get the real dataset (see "Dataset" below), place it at
# data/raw/online_retail_II.xlsx — or skip this and run_pipeline.py
# will auto-generate a synthetic dataset with realistic per-customer
# behavior so the pipeline is runnable out of the box.

python run_pipeline.py             # full pipeline: clean → features → train → segment
pytest tests/ -v                   # run the test suite (15 tests)

uvicorn api.main:app --reload      # API on :8000 (see /docs for Swagger UI)
streamlit run frontend/dashboard.py  # dashboard on :8501
```

## Quickstart — Docker

```bash
cp .env.example .env
docker compose up --build
docker compose exec api python run_pipeline.py   # first run only, populates model_artifacts/ + data/processed/
```

- API: http://localhost:8000/docs
- Dashboard: http://localhost:8501
- Postgres: localhost:5432 (used instead of SQLite when running via Docker)

## Dataset

Real dataset: UCI "Online Retail II" (https://archive.ics.uci.edu/dataset/502/online+retail+ii).
Download the zip, extract `online_retail_II.xlsx`, place it at
`data/raw/online_retail_II.xlsx`. It must have two sheets named
`"Year 2009-2010"` and `"Year 2010-2011"` — that's the format the real
file ships in, and what `config/config.yaml` and `src/data/load.py`
expect.

If no real file is present, `run_pipeline.py` auto-generates a synthetic
dataset (`src/data/synthetic.py`) with persistent per-customer behavioral
archetypes (loyal high-value, occasional low-value, discount hunter,
seasonal gift shopper, at-risk/departing, new customer) and a stochastic
dormancy hazard — this gives genuinely learnable churn signal for
development/testing, unlike a naively-random synthetic dataset.

## Repository structure

```
├── api/                    # FastAPI app: routers, Pydantic schemas, SQLAlchemy models
├── config/config.yaml      # all paths/thresholds/hyperparameters
├── data/{raw,interim,processed}/
├── frontend/dashboard.py   # Streamlit dashboard (5 views)
├── model_artifacts/        # trained model .joblib files + metadata (gitignored)
├── notebooks/              # dev/experimentation notebooks
├── src/
│   ├── data/                # load, clean, validate, synthetic generator
│   ├── features/            # RFM/CLV/engagement builders, preprocessing
│   ├── models/               # 6 classical ML models + PyTorch ANN/LSTM/Autoencoder
│   ├── segmentation/         # K-Means + DBSCAN clustering
│   ├── explainability/       # SHAP/LIME wrappers
│   └── utils/                 # config loader, logging
├── tests/                  # pytest: feature leakage tests + API integration tests
├── run_pipeline.py         # single-command orchestrator
├── Dockerfile / Dockerfile.dashboard / docker-compose.yml
└── requirements.txt
```

## Key design decisions

- **Point-in-time cutoff discipline**: every feature-engineering function
  takes an explicit `cutoff_date` and only uses transactions strictly
  before it; the churn label is computed independently from data strictly
  after it. Enforced by `tests/test_features.py`'s leakage tests.
- **Product-affinity features are excluded from the classifier's feature
  set** (still computed and used for segmentation). They're variable-width
  (top-N product codes seen in training data), a poor fit for a fixed API
  request schema.
- **Model selection is dynamic**: `run_pipeline.py` picks the best
  classical model by test-set ROC-AUC and records it in
  `model_artifacts/best_model_name.json`; the API reads this at startup
  rather than hardcoding a model name.
- **DATABASE_URL defaults to local SQLite** for direct-run development;
  `docker-compose.yml` overrides this to point at the bundled Postgres
  service.
- **Docker uses CPU-only PyTorch** (`--index-url https://download.pytorch.org/whl/cpu`),
  installed as its own cached layer — the default PyPI torch wheel bundles
  full CUDA support (~800MB) which is unnecessary for CPU-only inference
  and much more prone to network timeouts during image builds.

## Known limitations (found and documented during real-data testing)

- **LSTM purchase-timing model does not outperform a naive baseline.**
  Implemented and evaluated per spec (predicts days-until-next-purchase
  from a sequence of [amount, gap_days, category_code]), but on the real
  dataset its RMSE/MAE are statistically indistinguishable from just
  predicting the median gap every time. Root cause: with a median of ~4
  orders per customer, there isn't enough sequence length for an LSTM to
  learn meaningful temporal patterns, regardless of tuning. Retained in
  the codebase for completeness and technique demonstration, but **not**
  used in production scoring — the API's `/predict` endpoint uses the
  best classical model (selected dynamically by ROC-AUC), not the LSTM.
- **Churn rate on the real dataset is ~56%**, notably high for a 90-day
  window. Investigated during EDA: ~25% of customers are one-time-only
  buyers (mechanically near-100% "churned" under any window definition),
  and even repeat customers show ~47% churn — plausible for this
  business (many wholesale/gift-driven customers with irregular purchase
  cycles per the original PRD), but worth knowing if you're comparing
  against a different retail dataset's typical churn rates.
- **Single cutoff-date snapshot**: the pipeline currently builds one
  feature table from one `cutoff_date` (`max(invoice_date) - 90 days`)
  rather than multiple rolling-window snapshots across the full 2-year
  history. This caps the training set at ~5,300 customers. A rolling-
  window approach would generate more training examples from the same
  raw data, at the cost of pipeline complexity — worth considering if
  model performance ever seems data-starved.

## Testing

```bash
pytest tests/ -v
```

15 tests: 8 feature-engineering tests (including explicit leakage tests
and a regression test for a division-by-zero bug found in
`seasonal_concentration` during real-data testing — customers whose
purchases and returns net to exactly zero spend), and 7 API integration
tests (schema validation, health check, prediction, batch upload).

## What's implemented vs. not

**Implemented**: full data pipeline, 12 engineered features, 6 classical
models with cross-validated hyperparameter search, ANN + LSTM +
Autoencoder (PyTorch), K-Means + DBSCAN segmentation with business-
readable labels, SHAP + LIME + plain-language explanations, FastAPI (4
endpoints: predict, batch-predict, model metadata, health), Streamlit
dashboard (5 views), Docker Compose with Postgres.

**Not implemented / deferred**: CLV regression model is coded
(`train_clv_regressor` in `src/models/train_classical.py`) but not wired
into `run_pipeline.py` or the API yet (the API's `predicted_clv` field
is currently always `null`). Rolling-window cutoff snapshots (see
"Known limitations" above). CI/CD, auth/RBAC, MLflow-style experiment
tracking (a CSV log is used instead).