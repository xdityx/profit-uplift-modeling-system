import numpy as np
import pandas as pd
from src.simulation import generate_dataset, get_uplift_data


def test_generate_dataset_returns_encoded_hillstrom_data():
    df = generate_dataset(n_samples=100, seed=42)

    assert len(df) == 100
    assert {"recency", "history", "mens", "womens", "newbie", "treatment", "outcome"}.issubset(df.columns)
    assert any(column.startswith("zip_code_") for column in df.columns)
    assert any(column.startswith("channel_") for column in df.columns)
    assert any(column.startswith("history_segment_") for column in df.columns)


def test_get_uplift_data_returns_x_t_y_ready_for_modeling():
    X, treatment, outcome = get_uplift_data(n_samples=100, seed=42)

    assert len(X) == 100
    assert len(treatment) == 100
    assert len(outcome) == 100
    assert "treatment" not in X.columns
    assert "outcome" not in X.columns
    assert treatment.nunique() == 2
    assert set(outcome.unique()).issubset({0, 1})


def test_generate_dataset_has_reasonable_treatment_balance():
    df = generate_dataset(n_samples=100, seed=42)
    treatment_rate = df["treatment"].mean()

    assert df["treatment"].nunique() == 2
    assert 0.2 < treatment_rate < 0.9


def test_simulation_reproducibility():
    """Generate synthetic data twice with same seed and verify reproducibility."""
    df1 = generate_dataset(n_samples=150, seed=42)
    df2 = generate_dataset(n_samples=150, seed=42)

    # DataFrames should be identical when seeded
    pd.testing.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True))


def test_simulation_data_quality():
    """Verify generated dataset has reasonable properties."""
    df = generate_dataset(n_samples=500, seed=42)

    # All columns should have valid values (no NaN/Inf in critical columns)
    critical_cols = ["treatment", "outcome", "history", "recency"]
    for col in critical_cols:
        assert col in df.columns
        assert not df[col].isna().any(), f"Column {col} should not have NaN"
        assert not np.isinf(df[col]).any(), f"Column {col} should not have Inf"

    # Treatment should be binary
    assert df["treatment"].isin([0, 1]).all()

    # Outcome should be binary or numeric
    assert df["outcome"].isin([0, 1]).all()

    # Numeric columns should have reasonable ranges
    assert df["history"].min() >= 0
    assert df["recency"].min() >= 0
