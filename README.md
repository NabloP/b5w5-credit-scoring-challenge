
# B5W5: Credit Scoring & Risk Modeling — Week 5 Challenge | 10 Academy

## 🗂 Challenge Context
This repository documents the submission for 10 Academy’s **B5W5: Credit Scoring & Risk Modeling** challenge.

Bati Bank is partnering with an eCommerce platform to launch a Buy-Now-Pay-Later (BNPL) product. Our role is to support credit decisioning by building a predictive pipeline that:
- Estimates credit risk using behavioral signals
- Assigns creditworthiness scores to customers
- Determines optimal loan amount and duration

This project includes:
- 🧹 Clean ingestion and processing of transactional customer data
- 📊 EDA of user purchase and repayment behavior
- 🧠 Proxy-based target engineering using RFM features and clustering
- 🪛 Modular pipeline for feature engineering and model training
- 🧪 ML model evaluation and tracking via MLflow
- 🔁 CI/CD deployment via FastAPI, Docker, and GitHub Actions

---

## 🔧 Project Setup

1. Clone the repository:

```bash
git clone https://github.com/NabloP/b5w5-credit-scoring-challenge.git
cd b5w5-credit-scoring-challenge
```

2. Create and activate a virtual environment:

**On Windows:**
```bash
python -m venv credit-scoring-challenge-challenge
.\credit-scoring-challenge\Scripts\Activate.ps1
```

**On macOS/Linux:**
```bash
python3 -m venv credit-scoring-challenge
source credit-scoring-challenge/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ⚙️ CI/CD (GitHub Actions)

This project uses GitHub Actions for Continuous Integration. On every `push` or `pull_request` event, the following workflow is triggered:

- Checkout repo

- Set up Python 3.10

- Install dependencies from `requirements.txt`

CI workflow is defined at:

    `.github/workflows/unittests.yml`

--


## 🔐 Credit Scoring Business Understanding

### 1. Basel II Accord and the Need for Interpretability

The Basel II Capital Accord emphasizes **quantitative risk measurement and capital adequacy**. This mandates models that are not only predictive but also explainable. Regulators and internal risk teams must understand **why a credit decision was made**, so **transparency, traceability, and documentation** are critical. Complex models like XGBoost must be accompanied by tools like SHAP or surrogate models to remain compliant.

### 2. Why Use a Proxy Variable? What Are the Risks?

Our dataset lacks a labeled default column — the true "creditworthiness" of a customer is unobserved. Therefore, we create a **proxy variable** for "high-risk" behavior using **RFM (Recency, Frequency, Monetary) clustering**. This proxy is necessary to train a supervised model. However, poor proxy design may introduce **label noise**, bias, and incorrect risk assignment — leading to unfair loan rejections or approvals. The assumptions behind the proxy must be clearly defined, tested, and monitored.

### 3. Trade-Offs: Interpretable vs. Complex Models

- **Simple Models (e.g. Logistic Regression with WoE):**
  - ✅ High interpretability and regulatory alignment
  - ✅ Easy to monitor and deploy
  - ❌ May underfit complex patterns

- **Complex Models (e.g. Gradient Boosting):**
  - ✅ High predictive power, nonlinear relationships
  - ❌ Difficult to interpret without added tools (e.g., SHAP)
  - ❌ May face regulatory scrutiny

A hybrid approach is ideal: train complex models for backend performance, and **translate decisions into interpretable summaries** for stakeholders.

---

## 📁 Project Structure

<!-- TREE START -->
📁 Project Structure

solar-challenge-week1/
├── Dockerfile
├── LICENSE
├── README.md
├── docker-compose.yml
├── requirements.txt
├── .github/
│   ├── README.md
│   └── workflows/
│       ├── ci.yml
├── data/
│   ├── README.md
│   ├── processed/
│   └── raw/
│       ├── Xente_Variable_Definitions.csv
│       ├── data.csv
├── notebooks/
│   ├── README.md
│   ├── task-2-eda.ipynb
│   ├── task-3-proxy-labeling.ipynb
│   ├── task-4-model-training.ipynb
│   ├── task-5-api-deployement.ipynb
│   ├── task-6-experiment-tracking.ipynb
├── scripts/
│   ├── README.md
│   ├── eda_runner.py
│   ├── generate_tree.py
├── src/
│   ├── README.md
│   ├── __init__.py
│   ├── data_loader.py
│   ├── data_processing.py
│   ├── predict.py
│   ├── train.py
│   ├── api/
│   │   ├── main.py
│   │   ├── pydantic_models.py
│   ├── clustering/
│   ├── deployment/
│   ├── eda/
│   │   ├── customer_behavior_analyzer.py
│   │   ├── distribution_analyzer.py
│   │   ├── fraud_segment_analyzer.py
│   │   ├── schema_auditor.py
│   │   ├── temporal_behavior_analyzer.py
│   ├── features/
│   └── modeling/
└── tests/
    ├── README.md
    ├── test_data_processing.py
<!-- TREE END -->

## ✅ Interim Status (as of July 1)

- ✅ **Task 1 complete:** Credit scoring business understanding (this README)
- ✅ **Task 2 complete:** Robust CLI + notebook-based EDA across schema, nulls, categorical and behavioral distributions
- 🟡 Task 3 (Proxy Label Engineering): Underway with behavioral profiling and fraud heuristics
- 🔜 Tasks 4–6: Full pipeline assembly, model training, explainability, and final deployment polish

---

---

## 📋 Task Progress Tracker (Granular Updates for Tasks 2-6)

| Task # | Task Name                           | Status        | Detailed Description |
|--------|-------------------------------------|---------------|----------------------|
| 2      | Exploratory Data Analysis (EDA)     | ✅ Completed   | Executed full schema audit, null analysis, distribution visualizations (categorical, numerical, temporal behavior). Modules: `customer_behavior_analyzer.py`, `distribution_analyzer.py`, `temporal_behavior_analyzer.py`, `schema_auditor.py`. |
| 3      | Proxy Target Engineering            | ✅ Completed   | Developed proxy labels using behavioral heuristics and RFM clustering. Modules: `kmeans_labeler.py`, `cluster_diagnostics.py`. |
| 4      | Feature Engineering & Preparation   | ✅ Completed   | Engineered aggregate and temporal behavioral features; handled missing values, encoding, scaling via sklearn pipeline. Modules: `data_processing.py`, `feature_pipeline_builder.py`. |
| 5      | Model Training & Evaluation         | ✅ Completed   | Trained Logistic Regression and Random Forest models; performed hyperparameter tuning; evaluated using Accuracy, Precision, Recall, F1, ROC-AUC; SHAP analysis for interpretability. Modules: `train.py`, `model_evaluator.py`, `experiment_tracker.py`. |
| 6      | Deployment & CI/CD                  | ✅ Completed   | Built a FastAPI API with prediction endpoint (`/predict`); containerized using Docker and docker-compose; configured GitHub Actions for linting and testing. Modules: `main.py`, `pydantic_models.py`, `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`. |

---

## 🚀 API Usage Instructions

### 1. Build and Run the API Locally

```bash
docker-compose up --build
```

### 2. Predict Endpoint Usage

Send a POST request to the `/predict` endpoint with the following JSON structure:

```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{"features": [0.0, 1.0, 2.5, 100.0, 0.0, 0.1, 1, 0, 10, 5, 200, 1, 5, 1000, 200, 50, 10, 1, 2, 3, 1, 0, 1]}'
```

### 3. Example Response

```json
{
  "risk_probability": 0.72
}
```

---

## ⚙️ CI/CD Pipeline

- CI/CD configured through GitHub Actions at `.github/workflows/ci.yml`
- Runs automated flake8 linting and pytest unit tests on every push

---


## 📌 Task-to-Component Roadmap

| Task       | Description                                                   | Key Files/Modules                                  |
|------------|---------------------------------------------------------------|----------------------------------------------------|
| **Task 1** | Business Understanding, Basel II framing, and proxy rationale | `README.md`, `task-1-business-understanding.ipynb` |
| **Task 2** | Exploratory Data Analysis and RFM profiling                   | `task-2-eda.ipynb`, `src/eda/customer_behavior_analyzer.py`, `fraud_segment_analyzer.py`, `distribution_analyzer.py`, `temporal_behavior_analyzer.py`, `defensive_schema_auditor.py`         |
| **Task 3** | Feature Engineering with sklearn pipeline                     | `task-3-feature-engineering.ipynb`, `src/features/feature_pipeline_builder.py` (planned)                                                                                            |
| **Task 4** | Proxy Target Creation with RFM + KMeans clustering            | `src/clustering/kmeans_labeler.py`, `src/clustering/cluster_diagnostics.py` (planned)                                                                                                 |
| **Task 5** | Model Training, Validation, MLflow tracking                   | `src/modeling/model_trainer.py`, `model_evaluator.py`, `task-5-modeling.ipynb` (planned)                                                                                                  |
| **Task 6** | Deployment, CI/CD, Docker, API                                | `src/api/fastapi_app.py` (planned), `Dockerfile`, `.github/workflows/ci.yml`, `tests/unit/`                                                                                                  |


## 📌 References

Key sources for this challenge:

- [Basel II Primer (Statistica)](https://www3.stat.sinica.edu.tw/statistica/oldpdf/A28n535.pdf)
- [HKMA Alternative Credit Scoring](https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/alternative_credit_scoring.pdf)
- [World Bank Credit Scoring Guidelines](https://thedocs.worldbank.org/en/doc/935891585869698451-0130022020/original/CREDITSCORINGAPPROACHESGUIDELINESFINALWEB.pdf)
- [Intro to Scorecards (TDS)](https://towardsdatascience.com/how-to-develop-a-credit-risk-model-and-scorecard-91335fc01f03)
- [CFI on Credit Risk](https://corporatefinanceinstitute.com/resources/commercial-lending/credit-risk/)
- [Risk Officer on Credit Risk](https://www.risk-officer.com/Credit_Risk.htm)

---

## 👤 Author

**Nabil Mohamed**  
AIM Bootcamp Participant  
GitHub: [@NabloP](https://github.com/NabloP)
