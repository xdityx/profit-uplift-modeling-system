# Profit-Optimized Customer Targeting using Uplift Modeling

## Overview

Predictive models estimate outcome probability:

P(Y=1 | X)

Marketing interventions require estimating incremental impact:

E[Y(1) − Y(0) | X]

This project builds a causal uplift modeling system to rank customers by expected treatment effect and optimize targeting under campaign cost constraints.

The objective is profit-maximizing decision policy, not classification accuracy.

---

# Business Problem

Marketing campaigns suffer from targeting inefficiency:

- Some customers convert regardless of intervention (sure buyers)
- Some customers never convert (lost causes)
- Only a subset are persuadable

Predictive models prioritize high-probability customers, not high incremental impact.

Uplift modeling estimates heterogeneous treatment effects to isolate persuadable customers.

---

# Current Progress (Day 1 – Day 8)

## 1. Synthetic Confounded Data Generation

Constructed a synthetic dataset with:

- Observed covariates (age, income, tenure, usage)
- Hidden confounder
- Non-random treatment assignment
- Heterogeneous treatment effects
- Logistic outcome generation

This creates realistic selection bias and counterfactual structure.

---

## 2. Selection Bias Diagnosis

Exploratory analysis demonstrates:

- Treated and control groups differ in feature distributions
- Naive outcome comparison is biased
- Treatment assignment is partially predictable from covariates

---

## 3. Propensity Score Modeling

Estimated:

e(x) = P(T=1 | X)

Diagnostics include:

- ROC-AUC of treatment prediction
- Propensity score overlap visualization
- Positivity assumption verification

This step prepares data for causal meta-learners.

---

## 4. Predictive Baseline

Trained Random Forest predicting:

P(Y=1 | X)

Campaign simulation parameters:

- Cost per targeted customer: 10
- Margin per conversion: 60
- Targeting ratio: 30%

Compared profit under:

- Random targeting
- Predictive targeting

Observation: predictive probability does not measure incremental effect.

---

## 5. T-Learner Uplift Modeling

Implemented T-Learner meta-algorithm.

Two separate outcome models:

- Model trained on treated customers
- Model trained on control customers

Estimated uplift:

τ̂(x) = Ŷ_treated(x) − Ŷ_control(x)

Customers ranked by estimated treatment effect.

---

## 6. Uplift Evaluation (Day 7)

Implemented ranking-based causal evaluation:

- Uplift curve
- Qini curve
- Cumulative incremental response
- AUUC (Area Under Uplift Curve)

These metrics evaluate ranking quality of treatment effect predictions.

---

## 7. X-Learner Implementation (Day 8)

Implemented X-Learner meta-algorithm.

Steps:

1. Train outcome models for treated and control groups
2. Compute pseudo treatment effects
3. Train regression models on pseudo effects
4. Combine predictions using propensity weighting

X-Learner improves treatment effect estimation when treatment groups are imbalanced.

Profit simulation repeated using X-Learner uplift ranking.

---

# Methodology Pipeline

Data Simulation  
→ Bias Diagnosis  
→ Propensity Modeling  
→ Predictive Baseline  
→ T-Learner  
→ Uplift Evaluation (Qini / AUUC)  
→ X-Learner  
→ (Next) Model Comparison & Policy Optimization

---

# Technical Stack

Python  
NumPy  
Pandas  
Scikit-learn  
Matplotlib / Seaborn  
SciPy

Planned extensions:

- Policy evaluation framework
- Treatment heterogeneity analysis
- Lightweight API for scoring

---

# Positioning

This project demonstrates:

- Causal inference workflow
- Treatment effect modeling
- Ranking-based uplift evaluation
- Decision optimization under economic constraints
- Distinction between correlation and causation