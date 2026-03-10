# Profit-Optimized Customer Targeting using Uplift Modeling

## Overview

Traditional predictive models estimate:

P(Y=1 | X)

However, marketing interventions require estimating incremental impact:

E[Y(1) − Y(0) | X]

This project implements a causal uplift modeling pipeline to identify customers whose behavior changes due to treatment and to optimize campaign targeting under cost constraints.

The objective is **profit-maximizing policy design**, not classification accuracy.

---

# Business Problem

Marketing campaigns often waste budget due to incorrect targeting:

- Some customers convert regardless of incentives (sure buyers)
- Some customers never convert (lost causes)
- Only a subset are persuadable

Predictive models prioritize high-probability customers, but do not estimate incremental effect.

Uplift modeling estimates heterogeneous treatment effects to isolate customers whose behavior is influenced by intervention.

---

# Current Progress (Day 1 – Day 9)

## 1. Synthetic Confounded Data Generation

Constructed a dataset with:

- Observed customer features
- Hidden confounder
- Non-random treatment assignment
- Heterogeneous treatment effects
- Logistic outcome generation

This introduces realistic selection bias and counterfactual structure.

---

## 2. Selection Bias Diagnosis

Exploratory analysis demonstrates:

- Feature distributions differ between treated and control groups
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

This step prepares data for causal meta-learners.

---

## 4. Predictive Baseline

Trained Random Forest predicting:

P(Y=1 | X)

Campaign parameters:

Cost per targeted customer: 10  
Revenue per successful conversion: 60  
Targeting ratio: 30%

Profit comparison showed predictive targeting improves performance over random allocation.

However, this approach ignores treatment effect.

---

## 5. T-Learner Uplift Modeling

Implemented T-Learner meta-algorithm.

Two separate outcome models trained on:

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

These metrics evaluate the ranking quality of estimated treatment effects.

---

## 7. X-Learner Implementation

Implemented X-Learner meta-algorithm.

Steps:

1. Train outcome models
2. Compute pseudo treatment effects
3. Train regression models on pseudo effects
4. Combine predictions using propensity weighting

X-Learner reduces bias when treatment groups are imbalanced.

---

## 8. Policy Comparison (Day 9)

Compared four targeting strategies under identical campaign constraints:

| Strategy | Profit |
|--------|--------|
| Random Targeting | 32,040 |
| Predictive Model | 150,000 |
| T-Learner | 117,780 |
| X-Learner | 77,640 |

### Interpretation

Predictive targeting performs best in this simulation because baseline conversion probability dominates treatment effect magnitude.

Customers with high predicted probability are also likely to convert when targeted.

This highlights an important property of uplift modeling:

Uplift models provide the greatest advantage when treatment effect heterogeneity is large relative to baseline outcome probability.

---

# Methodology Pipeline

Data Simulation  
→ Bias Diagnosis  
→ Propensity Modeling  
→ Predictive Baseline  
→ T-Learner  
→ Uplift Evaluation (Qini / AUUC)  
→ X-Learner  
→ Policy Comparison

---

# Technical Stack

Python  
NumPy  
Pandas  
Scikit-learn  
Matplotlib / Seaborn  
SciPy

---

# Project Outcome

This project demonstrates:

- Causal inference workflow
- Treatment effect estimation
- Uplift meta-learning
- Ranking-based causal evaluation
- Profit-based policy optimization

The system shows how causal reasoning changes targeting decisions compared to standard predictive modeling.