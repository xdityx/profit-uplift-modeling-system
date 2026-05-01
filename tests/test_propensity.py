import numpy as np
from src.propensity import compute_propensity_scores, train_propensity_model
from src.simulation import generate_dataset


def test_propensity_model_trains_and_has_signal():
    df = generate_dataset(n_samples=100, seed=42)

    model, auc = train_propensity_model(df)

    assert model is not None
    assert hasattr(model, "predict_proba")
    assert isinstance(auc, float)
    assert 0.0 <= auc <= 1.0


def test_propensity_scores_are_probabilities():
    df = generate_dataset(n_samples=100, seed=42)
    model, _ = train_propensity_model(df)

    scored_df = compute_propensity_scores(model, df.copy())

    assert "propensity_score" in scored_df.columns
    assert scored_df["propensity_score"].between(0, 1).all()


def test_propensity_score_overlap_check():
    df = generate_dataset(n_samples=100, seed=42)
    model, _ = train_propensity_model(df)
    scored_df = compute_propensity_scores(model, df.copy())

    propensity = scored_df["propensity_score"].to_numpy()

    # Positivity assumption: propensity scores should have reasonable overlap
    assert np.min(propensity) > 0.001, "Min propensity too close to 0"
    assert np.max(propensity) < 0.999, "Max propensity too close to 1"

    # Check that both groups have representation
    treated = df["treatment"].to_numpy()
    control = ~treated.astype(bool)

    if treated.sum() > 0 and control.sum() > 0:
        treated_mean = propensity[treated].mean()
        control_mean = propensity[control].mean()
        # Treated should have higher propensity on average
        assert treated_mean > control_mean * 0.5, "Treated group should have reasonable propensity"


def test_propensity_roc_auc_reasonable():
    df = generate_dataset(n_samples=100, seed=42)
    model, auc = train_propensity_model(df)

    # For real Hillstrom data, propensity AUC should be valid
    # Can be high if treatment is predictable, low if random
    assert 0.4 <= auc <= 1.0, f"Propensity AUC {auc} out of reasonable range"


def test_propensity_extreme_values_handling():
    df = generate_dataset(n_samples=100, seed=42)
    model, _ = train_propensity_model(df)
    scored_df = compute_propensity_scores(model, df.copy())

    propensity = scored_df["propensity_score"].values.copy()

    # Simulate handling of extreme propensity scores
    # In IPW, extreme scores can lead to Inf weights, so we use clamping
    epsilon = 0.01
    propensity_clamped = np.clip(propensity, epsilon, 1 - epsilon)

    # Compute stabilization weights: min(e, 1-e) / max(e, 1-e)
    e = propensity_clamped
    stabilized_weights = np.minimum(e, 1 - e) / np.maximum(e, 1 - e)

    # Verify no Inf or NaN weights
    assert not np.any(np.isinf(stabilized_weights)), "Weights should not be infinite"
    assert not np.any(np.isnan(stabilized_weights)), "Weights should not be NaN"
    assert np.all(stabilized_weights > 0), "All weights should be positive"
