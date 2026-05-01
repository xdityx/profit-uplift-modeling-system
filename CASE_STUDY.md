# Profit Uplift Modeling — Case Study

## The Problem

Traditional machine learning is very good at predicting average outcomes. In a marketing setting, that usually means estimating something like `Y ~ X`: given a customer's features, how likely are they to convert, click, or purchase? That is useful, but it leaves a critical business question unanswered: what happens because of the intervention?

Treatment effects are rarely uniform. Some customers benefit from an email campaign, some would have converted anyway, and some react negatively because the message feels irrelevant or intrusive. In other words, the real decision variable is not "who is likely to buy?" but "who is more likely to buy if we intervene?"

That distinction matters in any campaign with finite budget. An email program may increase revenue for high-engagement users while slightly hurting inactive users through fatigue or unsubscribes. If both groups are treated the same way, the average lift may still look positive while a meaningful slice of the audience is being over-targeted. Without understanding which customers benefit, teams waste spend on non-responders and miss real incremental value.

## Why This Matters

This is not an academic edge case. Marketing teams spend millions on campaigns where average lift is often only 1-3%. If a team can identify the 40% of users who respond with a 10% lift, rather than broadcasting to everyone, campaign ROI can move dramatically even without increasing budget.

The same logic extends beyond email into pricing, recommendations, churn prevention, and product personalization. The operational question is always the same: where does the intervention create incremental value, and where does it not?

## What I Built

I built a production-style uplift modeling system focused on heterogeneous treatment effect estimation using the real Kevin Hillstrom MineThatData email marketing dataset: roughly 64,000 customers, three treatment arms (`No E-Mail`, `Mens E-Mail`, `Womens E-Mail`), and a binary conversion outcome. Features include recency, purchase history, channel behavior, and customer history segments.

The data pipeline converts the original dataset into an uplift-ready format. Treatment is collapsed into a binary indicator for whether a customer received an email, conversion is used as the outcome, and categorical variables such as `zip_code`, `channel`, and `history_segment` are one-hot encoded. That produces a clean feature matrix `X`, treatment vector `T`, and outcome vector `Y`.

On top of that data layer, I implemented two heterogeneous treatment effect algorithms: a T-Learner and an X-Learner. The T-Learner trains separate outcome models on treated and control customers, then estimates uplift as the difference between predicted outcomes. The X-Learner refines that approach by constructing pseudo-outcomes and blending arm-specific treatment effect models using propensity scores to reduce variance.

I also included a logistic-regression propensity model and uplift-specific evaluation metrics: Qini-style gains and AUUC (Area Under the Uplift Curve). The workflow is built around the right validation mindset for causal ML, where sample splitting and cross-validation matter.

## Key Technical Insights

One clear lesson from this project is that T-Learners and X-Learners solve related problems but behave differently in practice. The T-Learner is simple and intuitive, but it can be noisy when one arm has limited signal. The X-Learner adds more structure and often lowers variance, but that benefit depends on reasonable propensity estimation.

Another key takeaway is that propensity modeling is not the same thing as causal inference. Estimating treatment probability helps diagnose assignment behavior and supports downstream learners, but by itself it does not solve the counterfactual problem.

The project also reinforced how important feature interactions are in HTE modeling. A feature that looks weak in a standard predictive model can become highly informative once it interacts with treatment. That is why tools such as SHAP matter here: they help reveal which features drive treatment heterogeneity rather than baseline conversion alone. Uplift models are also easy to overfit, so proper validation discipline is essential.

## Architecture Highlights

The system is modular, with separate components for data loading, propensity modeling, T-Learner estimation, X-Learner estimation, and uplift evaluation. It also includes MLflow integration for experiment tracking, Docker support for reproducible execution, and a test suite with 9 passing unit tests. GitHub Actions CI keeps the test path green on each push.

## Real-World Example

Imagine an e-commerce team trying to optimize an email campaign. A baseline approach might send email to 100,000 customers because the average order value goes up 2% after send, producing a healthy-looking aggregate ROI.

With uplift modeling, the team can rank customers by predicted incremental effect. The top 30% may show an expected 8% lift and should definitely receive the campaign. The middle 50% may show only 1% lift and remain worthwhile depending on cost. The bottom 20% may have negative uplift because the email causes fatigue, unsubscribes, or cannibalization, which means the right decision is not to send. In that scenario, targeting becomes a causal allocation problem, not a blanket communication strategy.

## Why I Built This

I built this project because heterogeneous treatment effects sit at the frontier of practical causal ML. The skill transfers directly to pricing, recommendations, churn prevention, campaign optimization, and broader decision systems where the question is not just prediction but intervention policy.

It also reflects a deeper goal: demonstrating causal inference knowledge beyond the usual "correlation does not imply causation" talking point. Building T-Learners, X-Learners, propensity workflows, and uplift evaluation on real data shows the ability to reason about counterfactual structure.

## Results & Proof

- Real Hillstrom dataset with roughly 64,000 customers and three campaign treatment arms
- 9 passing unit tests covering data loading, propensity modeling, uplift estimation, and evaluation
- GitHub Actions CI/CD for automated validation on commit
- MLflow tracking for comparing T-Learner and X-Learner experiments
- Uplift-specific evaluation through Qini-style gains and AUUC
- A production-style modular codebase that demonstrates HTE expertise rather than average-effect reporting

## What's Next

The next extensions are clear: causal forests for nonparametric HTE estimation, double machine learning for more robust nuisance handling, sensitivity analysis for unobserved confounding risk, and a deployment layer for batch scoring new customer segments.
