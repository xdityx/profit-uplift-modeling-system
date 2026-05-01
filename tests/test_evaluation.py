import numpy as np
import pytest

from src.simulation import generate_dataset
from src.uplift_evaluation import auuc, qini_curve
from src.uplift_t_learner import TLearner


def test_qini_curve_returns_expected_shape():
    df = generate_dataset(n_samples=100, seed=42)
    X = df.drop(columns=["treatment", "outcome"])

    model = TLearner()
    model.fit(X, df["treatment"], df["outcome"])
    uplift = model.predict_uplift(X)

    qini = qini_curve(
        df["outcome"].to_numpy(),
        df["treatment"].to_numpy(),
        uplift,
    )

    assert qini.shape == (100,)


def test_auuc_returns_positive_scalar():
    df = generate_dataset(n_samples=100, seed=42)
    X = df.drop(columns=["treatment", "outcome"])

    model = TLearner()
    model.fit(X, df["treatment"], df["outcome"])
    uplift = model.predict_uplift(X)

    area = auuc(
        df["outcome"].to_numpy(),
        df["treatment"].to_numpy(),
        uplift,
    )

    assert np.isscalar(area)
    assert area > 0


def test_qini_curve_monotonic():
    df = generate_dataset(n_samples=100, seed=42)
    X = df.drop(columns=["treatment", "outcome"])

    model = TLearner()
    model.fit(X, df["treatment"], df["outcome"])
    uplift = model.predict_uplift(X)

    qini = qini_curve(
        df["outcome"].to_numpy(),
        df["treatment"].to_numpy(),
        uplift,
    )

    # Qini curve may not be strictly monotonic due to finite samples
    # But it should not dip too much (negative diffs indicate ranking errors)
    diffs = np.diff(qini)
    monotonic_violations = np.sum(diffs < -0.1)
    assert monotonic_violations < len(diffs) * 0.1, \
        "Qini curve should be mostly monotonic (< 10% dips)"


def test_auuc_edge_cases():
    y = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    treatment = np.array([1, 0, 1, 0, 1, 0, 1, 1, 0, 0])

    # Case 1: All zero uplift (no signal)
    uplift_zero = np.zeros(len(y))
    auuc_zero = auuc(y, treatment, uplift_zero)
    assert np.isfinite(auuc_zero)

    # Case 2: Perfect ranking (top 20% all positive effect, rest negative)
    uplift_perfect = np.array([2, 2, 1, 1, 0.5, 0.5, -1, -1, -2, -2])
    auuc_perfect = auuc(y, treatment, uplift_perfect)
    assert np.isfinite(auuc_perfect)
    assert auuc_perfect >= auuc_zero  # Signal should be >= zero signal


def test_qini_vs_random_baseline():
    df = generate_dataset(n_samples=200, seed=42)
    X = df.drop(columns=["treatment", "outcome"])
    y = df["outcome"].to_numpy()
    treatment = df["treatment"].to_numpy()

    model = TLearner()
    model.fit(X, treatment, y)
    uplift_signal = model.predict_uplift(X)

    # Random uplift (no signal)
    np.random.seed(42)
    uplift_random = np.random.randn(len(y))

    # Evaluate both
    qini_signal = qini_curve(y, treatment, uplift_signal)
    qini_random = qini_curve(y, treatment, uplift_random)

    auuc_signal = auuc(y, treatment, uplift_signal)
    auuc_random = auuc(y, treatment, uplift_random)

    # Signal-based ranking should beat random
    assert auuc_signal > auuc_random, \
        f"AUUC with signal ({auuc_signal}) should beat random ({auuc_random})"


def test_evaluation_with_missing_treatment_label():
    y = np.array([0, 1, 1, 0, 1])
    treatment = np.array([1.0, np.nan, 1, 0, 1])  # NaN in treatment
    uplift = np.array([0.1, 0.2, 0.3, 0.1, 0.2])

    # With NaN, computation will produce NaN result or raise error
    result = auuc(y, treatment, uplift)
    # Should either raise or return NaN
    assert np.isnan(result) or isinstance(result, (ValueError, RuntimeError, TypeError))
