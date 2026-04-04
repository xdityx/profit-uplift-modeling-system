from src.propensity import compute_propensity_scores, train_propensity_model
from src.simulation import generate_dataset


def test_propensity_model_trains_and_has_signal():
    df = generate_dataset(n_samples=100, seed=42)

    model, auc = train_propensity_model(df)

    assert model is not None
    assert hasattr(model, "predict_proba")
    assert auc > 0.5


def test_propensity_scores_are_probabilities():
    df = generate_dataset(n_samples=100, seed=42)
    model, _ = train_propensity_model(df)

    scored_df = compute_propensity_scores(model, df.copy())

    assert "propensity_score" in scored_df.columns
    assert scored_df["propensity_score"].between(0, 1).all()
