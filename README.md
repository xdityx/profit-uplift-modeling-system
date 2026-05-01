# Profit-Optimized Customer Targeting using Uplift Modeling

[![CI](https://github.com/xdityx/profit-uplift-modeling-system/actions/workflows/ci.yml/badge.svg)](https://github.com/xdityx/profit-uplift-modeling-system/actions/workflows/ci.yml) | [Live Demo](https://profit-uplift-modeling-system-hrhwqrccm472emfe6lgnp8.streamlit.app/)

## Problem

Traditional predictive models optimize for classification accuracy:

```
P(Y=1 | X)
```

But marketing decisions require estimating the **incremental effect of an intervention**—which customers will change behavior *because* of your campaign?

Standard approaches waste budget on:
- **Sure buyers**: customers who convert anyway
- **Lost causes**: customers who never convert
- **Random targeting**: no differentiation by responsiveness

## Solution

This project implements a **causal uplift modeling system** that estimates heterogeneous treatment effects:

```
τ(x) = E[Y(1) − Y(0) | X]
```

By ranking customers by their individual responsiveness to treatment, we can target interventions to maximize profit under cost constraints—not just pick high-probability converters.

---

## Quick Start

### Prerequisites
- Python 3.12
- pip

### Installation

```bash
git clone https://github.com/xdityx/profit-uplift-modeling-system.git
cd profit-uplift-modeling-system
pip install -r requirements.txt
```

### Run Tests

```bash
python -m pytest tests/ -v
```

### Run MLflow Tracking

```bash
python -m src.train_and_log
```

### Docker

```bash
docker build -t uplift-tests . && docker run --rm uplift-tests
```

---

## Results

Four targeting strategies compared on the Hillstrom dataset (64,000 customers):

| Strategy | Profit |
|----------|--------|
| Random Targeting | $32,040 |
| Predictive Model | $150,000 |
| T-Learner Uplift | $117,780 |
| X-Learner Uplift | $77,640 |

**Key Finding**: Predictive targeting performed best because baseline conversion probability dominated treatment effect heterogeneity. Uplift modeling provides greatest advantage when treatment effect variance is large relative to baseline outcome probability.

---

## Methodology

1. **Propensity Score Modeling**: Estimate P(T=1 | X) to diagnose selection bias
2. **Predictive Baseline**: Random Forest on P(Y=1 | X) as benchmark
3. **T-Learner**: Separate outcome models for treated vs. control groups
4. **Uplift Evaluation**: Qini curves and AUUC metrics to assess ranking quality
5. **X-Learner**: Meta-learner for improved effect estimation under treatment imbalance
6. **Policy Comparison**: Profit-driven ranking and targeting optimization

---

## Data

- **Hillstrom MineThatData Dataset**: Real e-mail campaign with ~64,000 customers
- **Features**: Age, income, tenure, usage, purchase history
- **Treatment**: Email campaign segment (No E-Mail, Mens E-Mail, Womens E-Mail)
- **Outcome**: Binary conversion indicator

---

## Tech Stack

- **Python** — Core implementation
- **Scikit-learn** — ML models (Random Forest, propensity modeling)
- **Pandas / NumPy** — Data manipulation and numerical computing
- **MLflow** — Experiment tracking and model logging
- **Streamlit** — Interactive web interface
- **Docker** — Containerization for reproducibility
- **GitHub Actions** — CI/CD pipeline

---

## Key Insights

**Correlation ≠ Causation**

Predictive modeling optimizes P(Y=1 | X), while causal decision-making optimizes E[Y(1) − Y(0) | X]. This distinction is critical for profitable policy optimization.

**When Uplift Modeling Wins**

Uplift models are most valuable when:
- Treatment effect heterogeneity is large
- You have sufficient samples in treatment and control
- The cost of targeting justifies precision in targeting

**When Predictive Targeting Wins**

Predictive models dominate when:
- Baseline outcome probability varies much more than treatment effect
- Treatment effects are homogeneous across customers
- Simplicity and interpretability are prioritized

---

## Repository Structure

```
profit-uplift-modeling-system
├── data/                          # Hillstrom dataset (git-ignored)
├── notebooks/                     # Exploratory analysis & case studies
│   ├── 01_eda_selection_bias.ipynb
│   ├── 02_propensity_analysis.ipynb
│   ├── 03_naive_baseline_profit.ipynb
│   ├── 04_t_learner_uplift.ipynb
│   ├── 05_uplift_evaluation.ipynb
│   ├── 06_x_learner.ipynb
│   ├── 07_model_comparison.ipynb
│   └── 08_result_analysis.ipynb
├── src/                           # Core modules
│   ├── simulation.py
│   ├── propensity.py
│   ├── uplift_t_learner.py
│   ├── uplift_x_learner.py
│   ├── uplift_evaluation.py
│   ├── data_loader.py
│   └── train_and_log.py
├── tests/                         # Unit tests
├── reports/
│   └── results_summary.md
├── Dockerfile
├── .github/workflows/ci.yml       # CI/CD pipeline
├── requirements.txt
└── README.md
```

---

## Learning Outcomes

This project demonstrates:
- Causal inference fundamentals and meta-learner algorithms
- Treatment effect heterogeneity estimation (T-Learner, X-Learner)
- Propensity score diagnostics and selection bias
- Ranking-based evaluation metrics (Qini, AUUC)
- Profit-driven policy optimization
- The distinction between correlation-based prediction and causal decision-making

---

## License

MIT
