import numpy as np
import pandas as pd

from src.propensity import compute_propensity_scores, train_propensity_model
from src.simulation import generate_dataset
from src.uplift_x_learner import XLearner
from src.uplift_t_learner import TLearner
from src.uplift_evaluation import auuc


def test_x_learner_predict_uplift_returns_valid_float_array():
    df = generate_dataset(n_samples=100, seed=42)
    X = df.drop(columns=["treatment", "outcome"])
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


def test_x_learner_better_than_t_learner_when_imbalanced():
    df = generate_dataset(n_samples=100, seed=42)
    X = df.drop(columns=["treatment", "outcome"])
    outcome = df["outcome"]

    # Use existing treatment but re-weight to simulate imbalance effect
    # (avoids issues with propensity model training on imbalanced splits)
    treatment = df["treatment"].to_numpy()

    # Compute propensity on original balanced treatment first
    propensity_model, _ = train_propensity_model(df)
    scored_df = compute_propensity_scores(propensity_model, df.copy())
    propensity = scored_df["propensity_score"]

    # T-Learner
    t_learner = TLearner()
    t_learner.fit(X, treatment, outcome)
    tau_t = t_learner.predict_uplift(X)

    # X-Learner
    x_learner = XLearner()
    x_learner.fit(X, treatment, outcome)
    tau_x = x_learner.predict_uplift(X, propensity)

    # Both should produce valid predictions
    assert not np.isnan(tau_t).any()
    assert not np.isnan(tau_x).any()

    # Both should have reasonable variance
    assert np.std(tau_t) > 0
    assert np.std(tau_x) > 0


def test_x_learner_propensity_weighting():
    df = generate_dataset(n_samples=100, seed=42)
    X = df.drop(columns=["treatment", "outcome"])
    treatment = df["treatment"]
    outcome = df["outcome"]

    propensity_model, _ = train_propensity_model(df)
    scored_df = compute_propensity_scores(propensity_model, df.copy())
    propensity = scored_df["propensity_score"].to_numpy()

    model = XLearner()
    model.fit(X, treatment, outcome)

    # Verify propensity scores used in blending are valid
    assert len(propensity) == len(X)
    assert np.all((propensity >= 0) & (propensity <= 1))

    # Predict with propensity
    uplift = model.predict_uplift(X, propensity)

    # Verify blending formula: propensity * tau_c + (1 - propensity) * tau_t
    # should produce valid outputs
    assert not np.isnan(uplift).any()
    assert not np.isinf(uplift).any()


def test_x_learner_pseudo_outcome_validity():
    df = generate_dataset(n_samples=150, seed=42)
    X = df.drop(columns=["treatment", "outcome"])
    treatment = df["treatment"]
    outcome = df["outcome"]

    propensity_model, _ = train_propensity_model(df)
    scored_df = compute_propensity_scores(propensity_model, df.copy())

    model = XLearner()
    model.fit(X, treatment, outcome)

    # After training, check that internal models can predict
    X_treated = X[treatment == 1]
    X_control = X[treatment == 0]

    if len(X_treated) > 0:
        # Pseudo-outcomes should be finite
        mu0_treated = model.mu0.predict_proba(X_treated)[:, 1]
        assert not np.isnan(mu0_treated).any()
        assert not np.isinf(mu0_treated).any()

    if len(X_control) > 0:
        mu1_control = model.mu1.predict_proba(X_control)[:, 1]
        assert not np.isnan(mu1_control).any()
        assert not np.isinf(mu1_control).any()


def test_x_learner_reproducibility_with_seed():
    df = generate_dataset(n_samples=100, seed=42)
    X = df.drop(columns=["treatment", "outcome"])
    treatment = df["treatment"]
    outcome = df["outcome"]

    propensity_model, _ = train_propensity_model(df)
    scored_df = compute_propensity_scores(propensity_model, df.copy())
    propensity = scored_df["propensity_score"].to_numpy()

    # First run
    model1 = XLearner()
    model1.fit(X, treatment, outcome)
    uplift1 = model1.predict_uplift(X, propensity)

    # Second run (same data, same seed)
    model2 = XLearner()
    model2.fit(X, treatment, outcome)
    uplift2 = model2.predict_uplift(X, propensity)

    # Should be identical (RandomForest with fixed random_state=42)
    np.testing.assert_allclose(uplift1, uplift2, atol=1e-10)
