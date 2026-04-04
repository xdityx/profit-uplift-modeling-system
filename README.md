# Profit-Optimized Customer Targeting using Uplift Modeling

## Overview

Traditional predictive models estimate:

P(Y=1 | X)

However, marketing interventions require estimating incremental impact:

E[Y(1) − Y(0) | X]

This project implements a causal uplift modeling system to identify customers whose behavior changes due to treatment and to optimize marketing targeting policies under cost constraints.

The objective is **profit-maximizing decision policy**, not classification accuracy.

---

# Business Problem

Marketing campaigns often waste budget due to inefficient targeting:

- Some customers convert regardless of incentives (sure buyers)
- Some customers never convert (lost causes)
- Only a subset are persuadable

Predictive models prioritize customers with high purchase probability but do not estimate the **incremental effect of intervention**.

Uplift modeling estimates **heterogeneous treatment effects** to isolate customers whose behavior is influenced by treatment.

---

# Methodology Pipeline

Data Simulation
↓
Selection Bias Diagnosis
↓
Propensity Modeling
↓
Predictive Baseline
↓
T-Learner
↓
Uplift Evaluation (Qini / AUUC)
↓
X-Learner
↓
Policy Comparison
↓
Result Analysis

---

# Project Workflow

## 1. Synthetic Confounded Data Generation

Constructed a dataset with:

- Customer features (age, income, tenure, usage)
- Hidden confounder
- Non-random treatment assignment
- Heterogeneous treatment effects
- Logistic outcome generation

This introduces realistic **selection bias** and counterfactual structure.

---

## 2. Selection Bias Diagnosis

Exploratory analysis demonstrates:

- Treated and control groups differ in feature distributions
- Naive outcome comparisons are biased
- Treatment assignment is partially predictable from covariates

---

## 3. Propensity Score Modeling

Estimated:

e(x) = P(T=1 | X)

Diagnostics include:

- ROC-AUC of treatment prediction
- Propensity overlap visualization
- Positivity assumption verification

This prepares the dataset for causal meta-learners.

---

## 4. Predictive Baseline

Trained Random Forest predicting:

P(Y=1 | X)

Campaign parameters:

Cost per targeted customer: 10  
Revenue per successful conversion: 60  
Targeting ratio: 30%

Compared:

- Random targeting
- Predictive targeting

Predictive ranking improves profit over random allocation but does not isolate treatment effect.

---

## 5. T-Learner Uplift Modeling

Implemented the T-Learner meta-algorithm.

Two outcome models trained separately on:

- Treated customers
- Control customers

Estimated uplift:

τ̂(x) = Ŷ_treated(x) − Ŷ_control(x)

Customers ranked by estimated treatment effect.

---

## 6. Uplift Evaluation

Implemented standard uplift evaluation metrics:

- Uplift curve
- Qini curve
- AUUC (Area Under Uplift Curve)

These metrics evaluate the **ranking quality of treatment effect predictions**.

---

## 7. X-Learner Implementation

Implemented the X-Learner meta-algorithm.

Steps:

1. Train outcome models
2. Compute pseudo treatment effects
3. Train regression models on pseudo effects
4. Combine predictions using propensity weighting

X-Learner improves treatment effect estimation under treatment imbalance.

---

## 8. Policy Comparison

Four targeting strategies were compared under identical campaign constraints.

| Strategy | Profit |
|--------|--------|
| Random Targeting | 32,040 |
| Predictive Model | 150,000 |
| T-Learner | 117,780 |
| X-Learner | 77,640 |

### Interpretation

Predictive targeting performs best in this simulation because baseline conversion probability dominates treatment effect magnitude.

Customers with high predicted probability are also highly likely to convert when targeted.

This highlights an important property of uplift modeling:

Uplift models provide the greatest advantage when treatment effect heterogeneity is large relative to baseline outcome probability.

---

# Key Insight

Predictive modeling optimizes:

P(Y=1 | X)

Causal decision-making optimizes:

E[Y(1) − Y(0) | X]

This difference reflects the distinction between **correlation-based prediction** and **causal decision optimization**.

---

# Repository Structure
profit-uplift-modeling-system
│
├── data/
│
├── notebooks/
│ ├── 01_eda_selection_bias.ipynb
│ ├── 02_propensity_analysis.ipynb
│ ├── 03_naive_baseline_profit.ipynb
│ ├── 04_t_learner_uplift.ipynb
│ ├── 05_uplift_evaluation.ipynb
│ ├── 06_x_learner.ipynb
│ ├── 07_model_comparison.ipynb
│ └── 08_result_analysis.ipynb
│
├── src/
│ ├── simulation.py
│ ├── propensity.py
│ ├── uplift_t_learner.py
│ ├── uplift_x_learner.py
│ └── uplift_evaluation.py
│
├── reports/
│ └── results_summary.md
│
├── requirements.txt
└── README.md


---

# Technical Stack

Python  
NumPy  
Pandas  
Scikit-learn  
Matplotlib  
Seaborn  
SciPy

---

# Project Outcome

This project demonstrates:

- Causal inference workflow
- Treatment effect estimation
- Uplift meta-learning
- Ranking-based causal evaluation
- Profit-driven policy optimization
- Distinction between correlation and causation
