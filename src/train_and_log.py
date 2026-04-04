import argparse
from pathlib import Path

import mlflow

from src.propensity import compute_propensity_scores, train_propensity_model
from src.simulation import generate_dataset
from src.uplift_evaluation import auuc
from src.uplift_t_learner import TLearner
from src.uplift_x_learner import XLearner


FEATURE_COLUMNS = ["age", "income", "tenure", "usage"]


def train_and_log(sample_size=1000, seed=42):
    project_root = Path(__file__).resolve().parent.parent
    tracking_dir = project_root / "mlruns"
    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())

    df = generate_dataset(n_samples=sample_size, seed=seed)
    X = df[FEATURE_COLUMNS]
    treatment = df["treatment"]
    outcome = df["outcome"]

    propensity_model, propensity_auc = train_propensity_model(df)
    scored_df = compute_propensity_scores(propensity_model, df.copy())

    t_learner = TLearner()
    t_learner.fit(X, treatment, outcome)
    t_uplift = t_learner.predict_uplift(X)
    t_learner_auuc = auuc(
        outcome.to_numpy(),
        treatment.to_numpy(),
        t_uplift,
    )

    x_learner = XLearner()
    x_learner.fit(X, treatment, outcome)
    x_uplift = x_learner.predict_uplift(X, scored_df["propensity_score"])
    x_learner_auuc = auuc(
        outcome.to_numpy(),
        treatment.to_numpy(),
        x_uplift,
    )

    params = {
        "sample_size": sample_size,
        "treatment_ratio": float(treatment.mean()),
        "random_seed": seed,
    }
    metrics = {
        "propensity_auc": float(propensity_auc),
        "t_learner_auuc": float(t_learner_auuc),
        "x_learner_auuc": float(x_learner_auuc),
    }

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

    return params, metrics


def main():
    parser = argparse.ArgumentParser(description="Train uplift models and log metrics to MLflow.")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of synthetic samples to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic data generation.")
    args = parser.parse_args()

    params, metrics = train_and_log(sample_size=args.sample_size, seed=args.seed)

    print("Logged MLflow run with params:")
    print(params)
    print("Metrics:")
    print(metrics)


if __name__ == "__main__":
    main()
