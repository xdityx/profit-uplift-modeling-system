import numpy as np

from src.simulation import generate_dataset
from src.uplift_t_learner import TLearner


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
