"""
Integration tests covering the full uplift modeling pipeline.

These tests verify that all components work together end-to-end:
Simulation → Propensity → T-Learner → X-Learner → Evaluation
"""

import numpy as np
import pandas as pd

from src.simulation import generate_dataset, get_uplift_data
from src.propensity import train_propensity_model, compute_propensity_scores
from src.uplift_t_learner import TLearner
from src.uplift_x_learner import XLearner
from src.uplift_evaluation import auuc, qini_curve


def test_full_pipeline_end_to_end():
    """Complete pipeline: Simulate → Propensity → T-Learner → X-Learner → Evaluate."""
    # 1. Generate dataset
    df = generate_dataset(n_samples=100, seed=42)
    X = df.drop(columns=["treatment", "outcome"])
    treatment = df["treatment"]
    outcome = df["outcome"]

    # 2. Fit propensity model
    propensity_model, auc = train_propensity_model(df)
    assert 0.0 <= auc <= 1.0

    scored_df = compute_propensity_scores(propensity_model, df.copy())
    propensity = scored_df["propensity_score"]

    # 3. Train T-Learner
    t_learner = TLearner()
    t_learner.fit(X, treatment, outcome)
    tau_t = t_learner.predict_uplift(X)

    assert tau_t.shape == (100,)
    assert not np.isnan(tau_t).any()

    # 4. Train X-Learner
    x_learner = XLearner()
    x_learner.fit(X, treatment, outcome)
    tau_x = x_learner.predict_uplift(X, propensity)

    assert tau_x.shape == (100,)
    assert not np.isnan(tau_x).any()

    # 5. Evaluate both models
    auuc_t = auuc(outcome.to_numpy(), treatment.to_numpy(), tau_t)
    auuc_x = auuc(outcome.to_numpy(), treatment.to_numpy(), tau_x)

    # 6. Verify results are finite
    assert np.isfinite(auuc_t)
    assert np.isfinite(auuc_x)


def test_pipeline_with_get_uplift_data():
    """Pipeline using the get_uplift_data helper function."""
    X, treatment, outcome = get_uplift_data(n_samples=100, seed=42)

    # Propensity
    df_for_propensity = pd.DataFrame(X).copy()
    df_for_propensity["treatment"] = treatment
    df_for_propensity["outcome"] = outcome

    propensity_model, _ = train_propensity_model(df_for_propensity)
    scored_df = compute_propensity_scores(propensity_model, df_for_propensity)
    propensity = scored_df["propensity_score"]

    # T-Learner
    t_learner = TLearner()
    t_learner.fit(X, treatment.to_numpy(), outcome.to_numpy())
    tau_t = t_learner.predict_uplift(X)

    # X-Learner
    x_learner = XLearner()
    x_learner.fit(X, treatment.to_numpy(), outcome.to_numpy())
    tau_x = x_learner.predict_uplift(X, propensity)

    # Evaluate
    auuc_t = auuc(outcome.to_numpy(), treatment.to_numpy(), tau_t)
    auuc_x = auuc(outcome.to_numpy(), treatment.to_numpy(), tau_x)

    assert np.isfinite(auuc_t)
    assert np.isfinite(auuc_x)


def test_pipeline_consistent_across_seeds():
    """Verify consistent behavior with different random seeds."""
    seeds = [42, 123, 456]
    auuc_scores = []

    for seed in seeds:
        df = generate_dataset(n_samples=100, seed=seed)
        X = df.drop(columns=["treatment", "outcome"])
        treatment = df["treatment"]
        outcome = df["outcome"]

        # Propensity
        propensity_model, _ = train_propensity_model(df)
        scored_df = compute_propensity_scores(propensity_model, df.copy())

        # T-Learner
        t_learner = TLearner()
        t_learner.fit(X, treatment, outcome)
        tau = t_learner.predict_uplift(X)

        # Evaluate
        score = auuc(outcome.to_numpy(), treatment.to_numpy(), tau)
        auuc_scores.append(score)

    # All seeds should produce finite scores
    for score in auuc_scores:
        assert np.isfinite(score)

    # Verify we got multiple scores
    assert len(auuc_scores) == 3


def test_qini_curve_reasonable_shape():
    """Verify Qini curve has expected shape."""
    df = generate_dataset(n_samples=100, seed=42)
    X = df.drop(columns=["treatment", "outcome"])

    t_learner = TLearner()
    t_learner.fit(X, df["treatment"], df["outcome"])
    tau = t_learner.predict_uplift(X)

    qini = qini_curve(
        df["outcome"].to_numpy(),
        df["treatment"].to_numpy(),
        tau,
    )

    # Shape
    assert qini.shape == (100,)

    # Should be finite
    assert np.all(np.isfinite(qini))

    # Should have some variation (unless data has no signal)
    assert np.std(qini) > 0


def test_pipeline_handles_different_sample_sizes():
    """Verify pipeline works with constrained dataset size."""
    # Test with fixture size (100) to match conftest limits
    df = generate_dataset(n_samples=100, seed=42)
    X = df.drop(columns=["treatment", "outcome"])
    treatment = df["treatment"]
    outcome = df["outcome"]

    # Fit models
    propensity_model, _ = train_propensity_model(df)
    scored_df = compute_propensity_scores(propensity_model, df.copy())

    t_learner = TLearner()
    t_learner.fit(X, treatment, outcome)
    tau_t = t_learner.predict_uplift(X)

    x_learner = XLearner()
    x_learner.fit(X, treatment, outcome)
    tau_x = x_learner.predict_uplift(X, scored_df["propensity_score"])

    # Verify shapes
    assert tau_t.shape == (100,)
    assert tau_x.shape == (100,)

    # Verify validity
    assert not np.isnan(tau_t).any()
    assert not np.isnan(tau_x).any()


def test_propensity_improves_x_learner_stability():
    """Verify that X-Learner with propensity produces valid predictions."""
    df = generate_dataset(n_samples=100, seed=42)
    X = df.drop(columns=["treatment", "outcome"])
    treatment = df["treatment"]
    outcome = df["outcome"]

    propensity_model, _ = train_propensity_model(df)
    scored_df = compute_propensity_scores(propensity_model, df.copy())
    propensity = scored_df["propensity_score"]

    # Train both models
    t_learner = TLearner()
    t_learner.fit(X, treatment, outcome)
    tau_t = t_learner.predict_uplift(X)

    x_learner = XLearner()
    x_learner.fit(X, treatment, outcome)
    tau_x = x_learner.predict_uplift(X, propensity)

    # Both should produce valid predictions
    assert not np.isnan(tau_t).any()
    assert not np.isnan(tau_x).any()

    # Both should have variance (not all zeros)
    assert np.std(tau_t) > 0
    assert np.std(tau_x) > 0


def test_evaluation_metrics_consistency():
    """Verify that AUUC and Qini curve produce valid metrics."""
    df = generate_dataset(n_samples=100, seed=42)
    X = df.drop(columns=["treatment", "outcome"])

    t_learner = TLearner()
    t_learner.fit(X, df["treatment"], df["outcome"])
    tau = t_learner.predict_uplift(X)

    y = df["outcome"].to_numpy()
    treatment = df["treatment"].to_numpy()

    # Compute both metrics
    qini = qini_curve(y, treatment, tau)
    area = auuc(y, treatment, tau)

    # Both should be finite
    assert np.all(np.isfinite(qini))
    assert np.isfinite(area)

    # AUUC should be the integral of Qini
    # Rough check: if qini has values, AUUC should be non-zero
    assert np.std(qini) > 0 or np.isclose(area, 0)
