# Profit-Optimized Customer Targeting using Uplift Modeling

## Overview

Traditional predictive models estimate outcome probability:

P(Y=1 | X)

However, marketing optimization requires estimating incremental impact:

E[Y(1) − Y(0) | X]

This project builds a causal uplift modeling system to identify customers whose behavior changes because of intervention.

The objective is profit-maximizing policy design under treatment cost constraints.

---

# Business Problem

Sending promotions to all customers increases marketing cost without maximizing profit.

Key challenge:

- Some customers will convert without treatment (sure buyers).
- Some customers will not convert even with treatment (lost causes).
- Only a subset are persuadable.

Correct targeting requires estimating heterogeneous treatment effects.

---

# Current Progress (Day 1 – Day 5)

## 1. Synthetic Confounded Data Generation

Implemented a synthetic dataset with:

- Observed covariates (age, income, tenure, usage)
- Hidden confounder (motivation)
- Non-random treatment assignment
- Heterogeneous treatment effects
- Binary outcome simulation via logistic process

Treatment assignment depends on observed + hidden variables, introducing realistic selection bias.

---

## 2. Selection Bias Demonstration

EDA shows:

- Treated and untreated groups differ in covariate distributions.
- Naive outcome comparison is biased.
- Treatment assignment is predictable from features.

This establishes the need for propensity modeling.

---

## 3. Propensity Score Modeling

Implemented logistic regression to estimate:

e(x) = P(T=1 | X)

Diagnostics include:

- Treatment prediction AUC
- Propensity score overlap visualization
- Discussion of positivity assumption

This step prepares data for causal meta-learners.

---

## 4. Naive Outcome Prediction Baseline

Trained a Random Forest to predict:

P(Y=1 | X)

Evaluated performance using ROC-AUC.

Customers were ranked by predicted probability.

---

## 5. Profit Simulation Framework

Defined campaign economics:

- Cost per targeted customer: 10
- Margin per conversion: 60
- Targeting budget: top 30%

Compared:

- Random targeting profit
- Naive predictive targeting profit

Finding:

High predictive probability does not guarantee high incremental profit.

This establishes the baseline against which uplift models will be evaluated.

---

# Methodology Pipeline

Data Simulation  
→ Selection Bias Diagnosis  
→ Propensity Modeling  
→ Predictive Baseline  
→ Profit Simulation  
→ (Next) Uplift Meta-Learners  

---

# Next Steps

- Implement T-Learner for treatment effect estimation  
- Implement X-Learner for improved bias correction  
- Construct uplift curves and Qini metric  
- Optimize targeting policy using estimated uplift  

---

# Technical Stack

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib / Seaborn
- SciPy

Deployment layer (planned):

- FastAPI

---

# Positioning

This project demonstrates:

- Causal inference reasoning
- Treatment effect modeling
- Policy optimization under cost constraints
- Distinction between correlation and causation

It is designed to simulate realistic business decision-making rather than pure classification performance.