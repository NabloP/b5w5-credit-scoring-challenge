
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
│   └── workflows/
│       ├── ci.yml
├── data/
│   ├── processed/
│   └── raw/
│       ├── Xente_Variable_Definitions.csv
│       ├── data.csv
├── notebooks/
│   ├── task-2-eda.ipynb
├── scripts/
│   ├── eda_runner.py
│   ├── generate_tree.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── data_processing.py
│   ├── predict.py
│   ├── train.py
│   ├── api/
│   │   ├── main.py
│   │   ├── pydantic_models.py
│   └── eda/
│       ├── customer_behavior_analyzer.py
│       ├── distribution_analyzer.py
│       ├── fraud_segment_analyzer.py
│       ├── schema_auditor.py
│       ├── temporal_behavior_analyzer.py
└── tests/
    ├── test_data_processing.py
<!-- TREE END -->

## ✅ Interim Status (as of July 1)

- ✅ **Task 1 complete:** Credit scoring business understanding (this README)
- ✅ **Task 2 complete:** Robust CLI + notebook-based EDA across schema, nulls, categorical and behavioral distributions
- 🟡 Task 3 (Proxy Label Engineering): Underway with behavioral profiling and fraud heuristics
- 🔜 Tasks 4–6: Full pipeline assembly, model training, explainability, and final deployment polish

---

## 📋 Task Progress Tracker

| Task # | Task Name                          | Status         | Description |
|--------|------------------------------------|----------------|-------------|
| 1      | Project Setup & Repo Structuring   | ✅ Complete     | Folder structure, env, and loader established |
| 2      | Exploratory Data Analysis (EDA)    | ✅ Complete     | EDA CLI modules and behavioral diagnostics done |
| 3      | Proxy Target Engineering           | 🟡 In Progress  | Designing proxy labels using RFM & fraud segments |
| 4      | Feature Engineering & Preparation  | ⏳ Not Started  | Construct final modeling dataset |
| 5      | Model Training & Evaluation        | ⏳ Not Started  | Train and interpret models (LogReg, XGB) |
| 6      | Deployment & Reporting             | ⏳ Not Started  | FastAPI deployment and final rubric polish |

---

## 📦 Key Capabilities (Planned)

- 🧠 Proxy label engineering using RFM + KMeans clustering
- 🧪 Modeling with both Logistic Regression and XGBoost
- 📉 Risk scoring and loan sizing predictions
- 📦 FastAPI model deployment with CI/CD and Docker
- 🔁 MLflow experiment tracking and registry

---

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
