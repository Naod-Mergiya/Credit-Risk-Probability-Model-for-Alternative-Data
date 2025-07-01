# Credit Risk Probability Model

This repository contains the code base for building, training, and deploying a credit-risk scoring service for **Bati Bank**.

## Project Structure

```text
credit-risk-model/
├── .github/workflows/ci.yml        # CI pipeline running tests & linting
├── data/                           # <- Git-ignored raw & processed data
│   ├── raw/                        # Raw data files
│   └── processed/                  # Cleaned / feature-engineered data
├── notebooks/
│   └── 1.0-eda.ipynb              # Exploratory data analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py         # Feature engineering utilities
│   ├── train.py                   # Model training script (CLI)
│   ├── predict.py                 # Batch inference script (CLI)
│   └── api/
│       ├── main.py                # FastAPI app exposing prediction endpoint
│       └── pydantic_models.py     # Request/response schemas
├── tests/
│   └── test_data_processing.py    # Unit tests
├── Dockerfile                     # Container image definition
├── docker-compose.yml             # Local orchestration (API + model)
├── requirements.txt               # Python dependencies
├── .gitignore                     # Files/directories excluded from VCS
└── README.md                      # Project docs
```

## Quickstart

1.  Build Docker image:

    ```bash
    docker compose build
    ```
2.  Start the API locally (hot-reloaded):

    ```bash
    docker compose up
    ```
3.  Run unit tests:

    ```bash
    pytest -q
    ```

## Model Training & Tracking (Task-5)

1. Install updated dependencies (adds `mlflow` & `pytest`):

   ```bash
   pip install -r requirements.txt
   ```

2. Launch the MLflow tracking UI (optional):

   ```bash
   mlflow ui  # open http://127.0.0.1:5000
   ```

3. Train & tune models (LogReg, Random Forest, GBM) and log runs to MLflow:

   ```bash
   python -m src.train \
       --raw-path data/data.csv \
       --model-out artifacts/best_model.pkl
   ```

   *Grid-search selects the best hyper-parameters (AUC on a hold-out set) and
   registers the top model under the MLflow Model Registry name
   `credit-risk-best`.*

4. Re-run the full test suite at any time:

   ```bash
   pytest -q
   ```

## Development Guidelines

* Use **feature branches** + PRs targeting `main`.
* All new code must include unit-tests and pass `pytest` & `ruff` linters.
* Keep notebooks lightweight; move reusable code to `src/`.
* Sensitive artefacts (models, data) must not be committed. Use DVC or S3.

## Credit Scoring Business Understanding

**1. Basel II & Model Interpretability**  
Basel II’s Internal Ratings-Based (IRB) approach lets a bank use its own credit-risk models *only* if they are transparent, well-documented, and regularly validated.  Supervisors must be able to trace every input to the Probability-of-Default (PD) estimate and see how it drives capital requirements.  Therefore our pipeline must prioritise interpretability (clear feature definitions, monotonic relationships, explainable math) and rigorous documentation so auditors can reproduce results and challenge assumptions.

**2. Why we build a proxy target & its risks**  
The dataset lacks an explicit "default" outcome, so we derive a *proxy* label (e.g. flagging high-risk RFM behaviour or historical fraud) to train a supervised model.  This is necessary to learn any predictive pattern, but it introduces *label risk*: if the proxy is only loosely correlated with real default, predictions may misstate credit risk, misprice loans, and expose the bank to unexpected losses or regulatory findings.  Continuous back-testing against true defaults, once available, is essential.

**3. Simple vs complex models in a regulated setting**  
• *Logistic Regression + Weight-of-Evidence*: high transparency, easy to justify, monotonic scorecards, straightforward stress-testing; usually lower Gini/ROC-AUC.  
• *Gradient Boosting / other ensemble*: higher predictive power and nonlinear capture, but opaque, harder to validate, and needs post-hoc explainability (SHAP, PDPs).  
Regulation values explainability and governance, so the marginal uplift in accuracy from complex models must outweigh the added validation cost and capital-use uncertainty.  A common compromise is to deploy a simple champion model for regulatory capital and use a complex challenger model for portfolio monitoring.

## License

MIT