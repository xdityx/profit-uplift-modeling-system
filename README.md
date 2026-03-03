# Profit-Optimized Customer Targeting using Uplift Modeling

## Overview

Traditional predictive models estimate:

P(Y=1 | X)

Business intervention requires estimating:

E[Y(1) − Y(0) | X]

This project builds a causal uplift modeling system to identify customers whose behavior changes because of treatment and to optimize campaign profit under cost constraints.

---

# Business Problem

Marketing campaigns face a structural inefficiency:

- Some customers will convert without incentives.
- Some customers will not convert even if targeted.
- Only a subset are persuadable.

Targeting based purely on predicted probability wastes budget.

Correct targeting requires estimating heterogeneous treatment effects.

---

# Current Progress (Day 1 – Day 6)

## 1. Synthetic Confounded Data Generation

Implemented dataset with:

- Observed covariates
- Hidden confounder
- Non-random treatment assignment
- Heterogeneous treatment effects
- Logistic outcome generation

This creates realistic selection bias.

---

## 2. Selection Bias Diagnosis

Exploratory analysis shows:

- Treated and control groups differ in feature distributions.
- Naive outcome comparison is biased.
- Treatment assignment is predictable from features.

---

## 3. Propensity Modeling

Estimated:

e(x) = P(T=1 | X)

Diagnostics include:

- ROC-AUC of treatment prediction
- Propensity overlap visualization
- Positivity assessment

---

## 4. Naive Predictive Baseline

Trained Random Forest predicting:

P(Y=1 | X)

Simulated campaign economics:

- Cost per target: 10
- Margin per conversion: 60
- Target ratio: 30%

Compared profit under:

- Random targeting
- Predictive targeting

Result: Predictive probability does not equal incremental value.

---

## 5. T-Learner Uplift Modeling (Day 6)

Implemented T-Learner meta-algorithm:

Two separate outcome models:

- Model 1: trained on treated group
- Model 2: trained on control group

Uplift estimated as:

τ(x) = Ŷ₁(x) − Ŷ₀(x)

Customers ranked by estimated treatment effect rather than outcome probability.

Profit simulation repeated using uplift-based targeting.

This marks transition from correlation-based decision-making to causal decision modeling.

---

# Methodology Pipeline

Data Simulation  
→ Bias Diagnosis  
→ Propensity Modeling  
→ Predictive Baseline  
→ T-Learner Uplift Modeling  
→ (Next) Uplift Evaluation Metrics  

---

# Next Steps

- Implement uplift curves
- Compute Qini and AUUC
- Compare policy performance formally
- Add X-Learner