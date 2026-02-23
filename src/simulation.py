import numpy as np
import pandas as pd

def generate_dataset(n_samples=10000, seed=42):
    np.random.seed(seed)

    # Core features
    age = np.random.normal(40, 12, n_samples)
    income = np.random.normal(60000, 15000, n_samples)
    tenure = np.random.exponential(3, n_samples)
    usage = np.random.normal(50, 10, n_samples)

    # Hidden confounder (not observed directly)
    motivation = np.random.normal(0, 1, n_samples)

    # Treatment assignment (confounded)
    logits = (
        0.03 * age
        + 0.00002 * income
        + 0.5 * motivation
        - 0.2 * tenure
    )
    treatment_prob = 1 / (1 + np.exp(-logits))
    treatment = np.random.binomial(1, treatment_prob)

    # Baseline outcome probability
    baseline_logit = (
        -2
        + 0.04 * usage
        + 0.00001 * income
        + 0.3 * motivation
    )

    baseline_prob = 1 / (1 + np.exp(-baseline_logit))

    # Heterogeneous treatment effect
    treatment_effect = (
        0.05 * (usage - 50)
        - 0.00001 * income
        + 0.3 * motivation
    )

    treated_prob = baseline_prob + treatment * treatment_effect
    treated_prob = np.clip(treated_prob, 0, 1)

    outcome = np.random.binomial(1, treated_prob)

    df = pd.DataFrame({
        "age": age,
        "income": income,
        "tenure": tenure,
        "usage": usage,
        "treatment": treatment,
        "outcome": outcome
    })

    return df




if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/simulated_campaign_data.csv", index=False)