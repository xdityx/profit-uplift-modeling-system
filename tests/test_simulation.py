from src.simulation import generate_dataset


def test_generate_dataset_has_expected_shape_and_columns():
    df = generate_dataset(n_samples=100, seed=42)

    assert df.shape == (100, 6)
    assert list(df.columns) == [
        "age",
        "income",
        "tenure",
        "usage",
        "treatment",
        "outcome",
    ]


def test_generate_dataset_has_reasonable_treatment_balance():
    df = generate_dataset(n_samples=100, seed=42)
    treatment_rate = df["treatment"].mean()

    assert df["treatment"].nunique() == 2
    assert 0.5 < treatment_rate < 0.99
