import numpy as np

from src.propensity import compute_propensity_scores, train_propensity_model
from src.simulation import generate_dataset
from src.uplift_x_learner import XLearner


def test_x_learner_predict_uplift_returns_valid_float_array():
    df = generate_dataset(n_samples=100, seed=42)
    X = df[["age", "income", "tenure", "usage"]]
    treatment = df["treatment"]
    outcome = df["outcome"]

    propensity_model, _ = train_propensity_model(df)
    scored_df = compute_propensity_scores(propensity_model, df.copy())

    model = XLearner()
    model.fit(X, treatment, outcome)
    uplift = model.predict_uplift(X, scored_df["propensity_score"])

    assert uplift.shape == (100,)
    assert np.issubdtype(uplift.dtype, np.floating)
    assert not np.isnan(uplift).any()
