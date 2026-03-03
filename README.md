# Profit-Optimized Customer Targeting using Uplift Modeling

## Overview

Traditional predictive models estimate:

P(Y=1 | X)

Business intervention requires estimating:

E[Y(1) − Y(0) | X]

This project builds a causal uplift modeling system to identify customers whose behavior changes because of treatment and to optimize targeting policies under cost constraints.

The focus is on incremental impact, not outcome probability.

---

# Business Problem

Marketing campaigns face structural inefficiency:

- Some customers will convert without treatment.
- Some customers will not convert even if treated.
- Only a subset are persuadable.

Targeting based on predicted probability wastes budget.

Optimal targeting requires estimating heterogeneous treatment effects.

---

# Current Progress (Day 1 – Day 7)

## 1. Synthetic Confounded Data Generation

Created dataset with:

- Observed covariates
- Hidden confounder
- Non-random treatment assignment
- Heterogeneous treatment effects
- Logistic outcome generation

This introduces realistic selection bias.

---

## 2. Selection Bias Diagnosis

Exploratory analysis demonstrates:

- Treated and control groups differ in feature distributions.
- Naive outcome comparisons are biased.
- Treatment assignment is partially predictable from covariates.

---

## 3. Propensity Modeling

Estimated:

e(x) = P(T=1 | X)

Diagnostics include:

- ROC-AUC of treatment prediction
- Propensity overlap visualization
- Positivity assessment

This approximates covariate balancing under observed confounding.

---

## 4. Predictive Baseline

Trained a Random Forest predicting:

P(Y=1 | X)

Simulated campaign economics:

- Cost per target: 10
- Margin per conversion: 60
- Targeting ratio: 30%

Compared:

- Random targeting
- Predictive targeting

Finding: Predictive probability does not equal incremental effect.

---

## 5. T-Learner Uplift Modeling

Implemented T-Learner:

Two separate outcome models trained on treated and control subsets.

Estimated uplift:

τ̂(x) = Ŷ_treated(x) − Ŷ_control(x)

Customers ranked by estimated treatment effect rather than outcome probability.

Profit comparison showed improvement over naive targeting.

---

## 6. Formal Uplift Evaluation (Day 7)

Implemented ranking-based uplift evaluation:

- Uplift curve
- Qini curve
- Cumulative incremental response
- AUUC (Area Under Uplift Curve)

The Qini curve measures cumulative incremental gain as customers are targeted in descending uplift order.

This allows formal evaluation of causal ranking quality beyond raw profit simulations.

---

# Methodology Pipeline

Data Simulation  
→ Bias Diagnosis  
→ Propensity Modeling  
→ Predictive Baseline  
→ T-Learner  
→ Uplift Evaluation (Qini, AUUC)  
→ (Next) X-Learner  

---

# Technical Stack

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib / Seaborn
- SciPy

Planned extensions:

- X-Learner
- Policy optimization refinement
- Modular evaluation utilities

---

# Positioning

This project demonstrates:

- Causal inference reasoning
- Treatment effect modeling
- Ranking-based uplift evaluation
- Policy optimization under economic constraints
- Distinction between correlation and causation