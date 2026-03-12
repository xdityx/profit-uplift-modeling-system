# Experiment Results Summary

## Campaign Parameters

Cost per targeted customer: 10  
Revenue per conversion: 60  
Targeting ratio: 30%

## Strategy Comparison

| Strategy | Profit |
|--------|--------|
| Random | 32,040 |
| Predictive | 150,000 |
| T-Learner | 117,780 |
| X-Learner | 77,640 |

## Key Observation

Predictive targeting achieves the highest profit in this simulation because baseline conversion probability dominates treatment effect magnitude.

When baseline probability strongly correlates with outcome, predictive ranking captures profitable customers even without modeling causal impact.

## Implication

Uplift modeling provides the greatest advantage when:

- Treatment effect heterogeneity is large  
- Baseline conversion probability is weakly correlated with treatment effect  
- Customer response to incentives varies significantly

This experiment demonstrates the distinction between correlation-based prediction and causal decision modeling.