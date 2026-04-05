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
