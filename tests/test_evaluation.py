import numpy as np

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
