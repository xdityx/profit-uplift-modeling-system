import numpy as np
import pandas as pd
import pytest

from src.simulation import generate_dataset
from src.uplift_t_learner import TLearner
from src.uplift_evaluation import auuc


def test_t_learner_predict_uplift_returns_valid_float_array():
    df = generate_dataset(n_samples=100, seed=42)
    X = df.drop(columns=["treatment", "outcome"])
    treatment = df["treatment"]
    outcome = df["outcome"]

    model = TLearner()
    model.fit(X, treatment, outcome)
    uplift = model.predict_uplift(X)

    assert uplift.shape == (100,)
    assert np.issubdtype(uplift.dtype, np.floating)
    assert not np.isnan(uplift).any()


def test_t_learner_with_imbalanced_treatment():
    df = generate_dataset(n_samples=100, seed=42)

    # Create imbalanced dataset: 80/20 treatment split
    indices = np.arange(len(df))
    np.random.seed(42)
    treated_indices = np.random.choice(indices, size=int(0.2 * len(df)), replace=False)
    treatment_imbalanced = np.zeros(len(df))
    treatment_imbalanced[treated_indices] = 1

    X = df.drop(columns=["treatment", "outcome"])
    outcome = df["outcome"]

    model = TLearner()
    model.fit(X, treatment_imbalanced, outcome)
    uplift = model.predict_uplift(X)

    # Calculate AUUC on the imbalanced split
    score = auuc(outcome.to_numpy(), treatment_imbalanced, uplift)

    assert uplift.shape == (100,)
    assert not np.isnan(uplift).any()
    assert np.isfinite(score)  # Should produce valid score


def test_t_learner_with_small_sample_size():
    df = generate_dataset(n_samples=100, seed=42)

    # Use only first 100 samples
    df_small = df.head(100)
    X = df_small.drop(columns=["treatment", "outcome"])
    treatment = df_small["treatment"]
    outcome = df_small["outcome"]

    model = TLearner()
    model.fit(X, treatment, outcome)
    uplift = model.predict_uplift(X)

    assert uplift.shape == (100,)
    assert not np.isnan(uplift).any()
    assert not np.isinf(uplift).any()


def test_t_learner_heterogeneous_effects_detection():
    df = generate_dataset(n_samples=100, seed=42)
    X = df.drop(columns=["treatment", "outcome"])
    treatment = df["treatment"]
    outcome = df["outcome"]

    model = TLearner()
    model.fit(X, treatment, outcome)
    uplift = model.predict_uplift(X)

    # Check uplift variance (heterogeneity detected)
    uplift_variance = np.var(uplift)
    assert uplift_variance > 0, "Uplift should have variance (heterogeneous effects)"

    # Top by uplift should have different average than bottom
    if len(uplift) > 10:
        top_indices = np.argsort(-uplift)[:5]
        bottom_indices = np.argsort(-uplift)[-5:]
        top_mean = np.mean(uplift[top_indices])
        bottom_mean = np.mean(uplift[bottom_indices])
        # Top and bottom should have different means (if heterogeneous)
        assert top_mean != bottom_mean, "Top and bottom should have different uplift estimates"


def test_t_learner_edge_case_all_zeros_treatment():
    df = generate_dataset(n_samples=100, seed=42)
    X = df.drop(columns=["treatment", "outcome"])
    outcome = df["outcome"]

    # Edge case: all treatment=0
    treatment_all_zeros = np.zeros(len(df))

    model = TLearner()

    # Should either train gracefully or raise clear error
    with pytest.raises((ValueError, RuntimeError)):
        model.fit(X, treatment_all_zeros, outcome)
