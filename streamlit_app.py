"""Streamlit interface for interactive uplift modeling analysis."""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.data_loader import load_hillstrom_dataframe
from src.uplift_evaluation import auuc, qini_curve
from src.uplift_t_learner import TLearner
from src.uplift_x_learner import XLearner

try:
    import shap
    SHAP_AVAILABLE = True
    SHAP_IMPORT_ERROR = ""
except Exception as exc:  # noqa: BLE001
    shap = None
    SHAP_AVAILABLE = False
    SHAP_IMPORT_ERROR = str(exc)


st.set_page_config(
    page_title="Uplift Modeling Framework",
    layout="wide",
)

st.title("Uplift Modeling — Heterogeneous Treatment Effects")
st.write("Understand which customers benefit most from treatment")


DEFAULT_RANDOM_SEED = 42
DEFAULT_SAMPLE_ROWS = 2000
MIN_REQUIRED_COLUMNS = {"treatment", "outcome"}
IGNORE_COLUMNS = {"user_id", "treatment", "outcome"}
GREEN_SCALE = ["#d9f99d", "#65a30d", "#166534"]
RED_SCALE = ["#fecaca", "#dc2626", "#7f1d1d"]


@dataclass
class PreparedDataset:
    """Container for validated and encoded input data."""

    name: str
    raw_df: pd.DataFrame
    modeling_df: pd.DataFrame
    feature_columns: list[str]
    row_count: int


@dataclass
class ModelArtifacts:
    """Container for one model's metrics and outputs."""

    model_name: str
    uplift_scores: np.ndarray
    qini_score: float
    auuc_score: float
    qini_curve_values: np.ndarray
    train_seconds: float
    feature_importance: pd.DataFrame


@dataclass
class ExperimentResults:
    """Container for comparison results across both uplift models."""

    prepared: PreparedDataset
    test_df: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.Series
    treatment_test: pd.Series
    propensity_auc: float
    t_learner: ModelArtifacts
    x_learner: ModelArtifacts
    winner: str
    recommendation: str


def initialize_state() -> None:
    """Set default session state values used across tabs."""
    defaults = {
        "prepared_data": None,
        "results": None,
        "data_error": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def build_template_dataframe(rows: int = 200) -> pd.DataFrame:
    """Create a small CSV template users can download and edit."""
    rng = np.random.default_rng(DEFAULT_RANDOM_SEED)
    df = pd.DataFrame(
        {
            "user_id": np.arange(1, rows + 1),
            "treatment": rng.binomial(1, 0.5, rows),
            "feature_spend": rng.normal(120.0, 30.0, rows).round(2),
            "feature_recency": rng.integers(1, 30, rows),
            "feature_sessions": rng.poisson(5, rows),
            "feature_discount_affinity": rng.normal(0.0, 1.0, rows).round(3),
        }
    )
    uplift_signal = (
        0.04 * (df["feature_sessions"] - 5)
        + 0.02 * df["feature_discount_affinity"]
        - 0.01 * (df["feature_recency"] - 15)
    )
    baseline = 0.18 + 0.001 * df["feature_spend"] + 0.02 * df["feature_sessions"]
    outcome_prob = 1 / (1 + np.exp(-(baseline / 10 + df["treatment"] * uplift_signal)))
    df["outcome"] = rng.binomial(1, np.clip(outcome_prob, 0.02, 0.95))
    return df


@st.cache_data(show_spinner=False)
def load_sample_hillstrom(rows: int = DEFAULT_SAMPLE_ROWS) -> pd.DataFrame:
    """Load a sample of encoded Hillstrom data for the demo experience."""
    df = load_hillstrom_dataframe(n_samples=rows, seed=DEFAULT_RANDOM_SEED).copy()
    df.insert(0, "user_id", np.arange(1, len(df) + 1))
    return df


def read_uploaded_csv(uploaded_file: Any) -> pd.DataFrame:
    """Read an uploaded CSV file into a dataframe."""
    return pd.read_csv(uploaded_file)


def coerce_binary_series(series: pd.Series, name: str) -> pd.Series:
    """Convert supported binary encodings to integer 0/1 values."""
    cleaned = series.dropna()
    unique_values = frozenset(cleaned.astype(str).str.strip().str.lower().unique())
    binary_maps = {
        frozenset({"0", "1"}): {"0": 0, "1": 1},
        frozenset({"false", "true"}): {"false": 0, "true": 1},
        frozenset({"no", "yes"}): {"no": 0, "yes": 1},
    }
    if unique_values in binary_maps:
        mapping = binary_maps[frozenset(unique_values)]
        return series.astype(str).str.strip().str.lower().map(mapping).astype(int)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"`{name}` must be binary and convertible to 0/1 values.")
    unique_numeric = set(numeric.dropna().astype(int).unique())
    if not unique_numeric.issubset({0, 1}):
        raise ValueError(f"`{name}` must contain only binary 0/1 values.")
    return numeric.astype(int)


def validate_input_columns(df: pd.DataFrame) -> None:
    """Validate minimum required columns for uplift analysis."""
    missing = MIN_REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"Missing required column(s): {joined}.")
    feature_candidates = [column for column in df.columns if column not in IGNORE_COLUMNS]
    if not feature_candidates:
        raise ValueError("Upload at least one feature column in addition to treatment and outcome.")


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Encode feature columns into a numeric design matrix."""
    feature_df = df[[column for column in df.columns if column not in IGNORE_COLUMNS]].copy()
    if feature_df.empty:
        raise ValueError("No feature columns found after removing user_id, treatment, and outcome.")
    feature_df = pd.get_dummies(feature_df, drop_first=False)
    numeric_feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
    if numeric_feature_df.isna().any().any():
        raise ValueError("Feature columns contain values that could not be converted to numeric form.")
    return numeric_feature_df, numeric_feature_df.columns.tolist()


def prepare_dataset(df: pd.DataFrame, source_name: str) -> PreparedDataset:
    """Validate, coerce, and encode data for modeling."""
    validate_input_columns(df)
    prepared = df.copy()
    prepared["treatment"] = coerce_binary_series(prepared["treatment"], "treatment")
    prepared["outcome"] = coerce_binary_series(prepared["outcome"], "outcome")
    encoded_features, feature_columns = encode_features(prepared)
    modeling_df = encoded_features.copy()
    modeling_df["treatment"] = prepared["treatment"].to_numpy()
    modeling_df["outcome"] = prepared["outcome"].to_numpy()
    return PreparedDataset(
        name=source_name,
        raw_df=prepared,
        modeling_df=modeling_df,
        feature_columns=feature_columns,
        row_count=len(prepared),
    )


def safe_prepare_dataset(df: pd.DataFrame, source_name: str) -> PreparedDataset | None:
    """Prepare data and surface a friendly error in the UI if it fails."""
    try:
        prepared = prepare_dataset(df, source_name)
    except Exception as exc:  # noqa: BLE001
        st.session_state["data_error"] = str(exc)
        st.session_state["prepared_data"] = None
        return None
    st.session_state["data_error"] = None
    st.session_state["prepared_data"] = prepared
    st.session_state["results"] = None
    return prepared


def get_treatment_split(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize treatment group counts and percentages."""
    summary = (
        df["treatment"]
        .value_counts(normalize=False)
        .rename_axis("treatment")
        .reset_index(name="count")
        .sort_values("treatment")
    )
    summary["share_pct"] = (summary["count"] / summary["count"].sum() * 100).round(2)
    summary["label"] = summary["treatment"].map({0: "Control", 1: "Treated"})
    return summary


def build_basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Build compact numeric summary stats for display."""
    numeric = df.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        return pd.DataFrame()
    summary = numeric.describe().transpose().reset_index()
    summary = summary.rename(columns={"index": "feature"})
    return summary[["feature", "mean", "std", "min", "25%", "50%", "75%", "max"]]


def render_dataset_kpis(prepared: PreparedDataset) -> None:
    """Render top-level metrics for the loaded dataset."""
    treatment_share = prepared.raw_df["treatment"].mean() * 100
    outcome_rate = prepared.raw_df["outcome"].mean() * 100
    feature_count = len(prepared.feature_columns)
    cols = st.columns(4)
    cols[0].metric("Rows", f"{prepared.row_count:,}")
    cols[1].metric("Encoded Features", f"{feature_count}")
    cols[2].metric("Treatment Share", f"{treatment_share:.1f}%")
    cols[3].metric("Outcome Rate", f"{outcome_rate:.1f}%")


def percentile_breakdown(uplift_scores: np.ndarray) -> pd.DataFrame:
    """Summarize uplift scores by quartile segments."""
    score_series = pd.Series(uplift_scores, name="uplift")
    labels = ["0-25%", "25-50%", "50-75%", "75-100%"]
    segments = pd.qcut(score_series.rank(method="first"), q=4, labels=labels)
    summary = (
        pd.DataFrame({"segment": segments, "uplift": score_series})
        .groupby("segment", observed=False)
        .agg(customers=("uplift", "size"), avg_effect=("uplift", "mean"))
        .reset_index()
    )
    summary["share_pct"] = (summary["customers"] / summary["customers"].sum() * 100).round(1)
    return summary


def make_histogram(scores: np.ndarray, title: str) -> go.Figure:
    """Build a histogram for uplift score distribution."""
    color = np.where(scores >= 0, "Positive or Neutral", "Negative")
    plot_df = pd.DataFrame({"uplift": scores, "effect_direction": color})
    fig = px.histogram(
        plot_df,
        x="uplift",
        color="effect_direction",
        nbins=40,
        title=title,
        color_discrete_map={
            "Positive or Neutral": "#15803d",
            "Negative": "#dc2626",
        },
    )
    fig.update_layout(bargap=0.05, legend_title_text="")
    fig.add_vline(x=0, line_dash="dash", line_color="#111827")
    return fig


def make_qini_bar_chart(results: ExperimentResults) -> go.Figure:
    """Build a bar chart comparing Qini scores across models."""
    plot_df = pd.DataFrame(
        {
            "model": ["T-Learner", "X-Learner"],
            "qini_score": [
                results.t_learner.qini_score,
                results.x_learner.qini_score,
            ],
        }
    )
    fig = px.bar(
        plot_df,
        x="model",
        y="qini_score",
        title="Qini Coefficient Comparison",
        color="model",
        color_discrete_map={
            "T-Learner": "#65a30d",
            "X-Learner": "#0f766e",
        },
        text_auto=".2f",
    )
    fig.update_layout(showlegend=False)
    return fig


def make_auuc_line_chart(results: ExperimentResults) -> go.Figure:
    """Build a line chart for cumulative Qini gains."""
    x_axis = np.arange(1, len(results.t_learner.qini_curve_values) + 1)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=results.t_learner.qini_curve_values,
            mode="lines",
            name="T-Learner",
            line={"color": "#65a30d", "width": 3},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=results.x_learner.qini_curve_values,
            mode="lines",
            name="X-Learner",
            line={"color": "#0f766e", "width": 3},
        )
    )
    fig.update_layout(
        title="Cumulative Uplift Gains (Qini Curve)",
        xaxis_title="Ranked Customers",
        yaxis_title="Cumulative Gain",
    )
    return fig


def make_feature_importance_chart(importance_df: pd.DataFrame, title: str) -> go.Figure:
    """Build a horizontal bar chart for top feature drivers."""
    top_features = importance_df.sort_values("importance", ascending=True).tail(10)
    fig = px.bar(
        top_features,
        x="importance",
        y="feature",
        orientation="h",
        title=title,
        color="importance",
        color_continuous_scale=["#dbeafe", "#2563eb", "#1e3a8a"],
    )
    fig.update_layout(coloraxis_showscale=False, yaxis_title="")
    return fig


def choose_winner(t_model: ModelArtifacts, x_model: ModelArtifacts) -> tuple[str, str]:
    """Choose the winning model using AUUC as the primary criterion."""
    if t_model.auuc_score >= x_model.auuc_score:
        return (
            "T-Learner",
            "T-Learner currently leads on AUUC, so it is the better default model for targeting this dataset.",
        )
    return (
        "X-Learner",
        "X-Learner currently leads on AUUC, suggesting its propensity-weighted effect estimates rank customers more effectively.",
    )


def fit_propensity_model(X_train: pd.DataFrame, treatment_train: pd.Series) -> tuple[LogisticRegression, float]:
    """Fit a propensity model and compute in-sample train AUC."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, treatment_train)
    probs = model.predict_proba(X_train)[:, 1]
    auc_score = roc_auc_score(treatment_train, probs)
    return model, float(auc_score)


def safe_positive_class_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return the positive class probability even if the classifier saw one class."""
    proba = model.predict_proba(X)
    if proba.shape[1] == 1:
        learned_class = int(model.classes_[0])
        if learned_class == 1:
            return np.ones(len(X))
        return np.zeros(len(X))
    class_index = list(model.classes_).index(1)
    return proba[:, class_index]


def predict_t_uplift(model: TLearner, X: pd.DataFrame) -> np.ndarray:
    """Predict uplift for a T-Learner with one-class fallbacks."""
    treated_pred = safe_positive_class_proba(model.model_treated, X)
    control_pred = safe_positive_class_proba(model.model_control, X)
    return treated_pred - control_pred


def predict_x_uplift(model: XLearner, X: pd.DataFrame, propensity: np.ndarray) -> np.ndarray:
    """Predict uplift for an X-Learner."""
    return model.predict_uplift(X, propensity)


def qini_score_from_curve(curve: np.ndarray) -> float:
    """Reduce a Qini curve to a single comparison score."""
    if len(curve) == 0:
        return 0.0
    return float(curve[-1])


def build_shap_importance(model: Any, X: pd.DataFrame, sample_size: int = 300) -> pd.DataFrame:
    """Compute simple SHAP feature importance for tree-based models."""
    if not SHAP_AVAILABLE:
        raise RuntimeError(SHAP_IMPORT_ERROR or "SHAP is not available in this environment.")
    sample = X.sample(min(sample_size, len(X)), random_state=DEFAULT_RANDOM_SEED)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    if isinstance(shap_values, list):
        positive_values = shap_values[-1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        positive_values = shap_values[:, :, -1]
    else:
        positive_values = shap_values
    importance = np.abs(positive_values).mean(axis=0)
    return pd.DataFrame({"feature": sample.columns, "importance": importance})


def build_model_importance_fallback(model: Any, X: pd.DataFrame) -> pd.DataFrame:
    """Build feature importance from tree model attributes when SHAP is unavailable."""
    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_)
    else:
        importance = np.zeros(X.shape[1], dtype=float)
    return pd.DataFrame({"feature": X.columns, "importance": importance})


def summarize_effect_direction(X: pd.DataFrame, uplift_scores: np.ndarray, features: list[str]) -> tuple[str, str]:
    """Generate short interpretation text for top directional drivers."""
    correlations = []
    score_series = pd.Series(uplift_scores)
    for feature in features:
        series = pd.to_numeric(X[feature], errors="coerce")
        if series.nunique() <= 1:
            continue
        corr = series.corr(score_series)
        if pd.notna(corr):
            correlations.append((feature, corr))
    if not correlations:
        return "No strong increasing feature detected.", "No strong decreasing feature detected."
    ordered = sorted(correlations, key=lambda item: item[1])
    decreases = ordered[0][0]
    increases = ordered[-1][0]
    return (
        f"Treatment effect tends to increase with `{increases}`.",
        f"Treatment effect tends to decrease with `{decreases}`.",
    )


def train_single_t_learner(
    X_train: pd.DataFrame,
    treatment_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    treatment_test: pd.Series,
) -> ModelArtifacts:
    """Train the T-Learner and compute evaluation outputs."""
    started = time.perf_counter()
    model = TLearner()
    model.fit(X_train, treatment_train, y_train)
    uplift_scores = predict_t_uplift(model, X_test)
    elapsed = time.perf_counter() - started
    curve = qini_curve(y_test.to_numpy(), treatment_test.to_numpy(), uplift_scores)
    try:
        importance = build_shap_importance(model.model_treated, X_train)
    except Exception:  # noqa: BLE001
        importance = build_model_importance_fallback(model.model_treated, X_train)
    return ModelArtifacts(
        model_name="T-Learner",
        uplift_scores=uplift_scores,
        qini_score=qini_score_from_curve(curve),
        auuc_score=float(auuc(y_test.to_numpy(), treatment_test.to_numpy(), uplift_scores)),
        qini_curve_values=curve,
        train_seconds=elapsed,
        feature_importance=importance,
    )


def train_single_x_learner(
    X_train: pd.DataFrame,
    treatment_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    treatment_test: pd.Series,
    propensity_train: np.ndarray,
    propensity_test: np.ndarray,
) -> ModelArtifacts:
    """Train the X-Learner and compute evaluation outputs."""
    del propensity_train
    started = time.perf_counter()
    model = XLearner()
    model.fit(X_train, treatment_train, y_train)
    uplift_scores = predict_x_uplift(model, X_test, propensity_test)
    elapsed = time.perf_counter() - started
    curve = qini_curve(y_test.to_numpy(), treatment_test.to_numpy(), uplift_scores)
    try:
        treated_importance = build_shap_importance(model.mu1, X_train)
        control_importance = build_shap_importance(model.mu0, X_train)
    except Exception:  # noqa: BLE001
        treated_importance = build_model_importance_fallback(model.mu1, X_train)
        control_importance = build_model_importance_fallback(model.mu0, X_train)
    merged = treated_importance.merge(
        control_importance,
        on="feature",
        suffixes=("_treated", "_control"),
    )
    merged["importance"] = (
        (merged["importance_treated"] + merged["importance_control"]) / 2
    )
    importance = merged[["feature", "importance"]]
    return ModelArtifacts(
        model_name="X-Learner",
        uplift_scores=uplift_scores,
        qini_score=qini_score_from_curve(curve),
        auuc_score=float(auuc(y_test.to_numpy(), treatment_test.to_numpy(), uplift_scores)),
        qini_curve_values=curve,
        train_seconds=elapsed,
        feature_importance=importance,
    )


def run_experiment(
    prepared: PreparedDataset,
    test_size: float,
    random_seed: int,
) -> ExperimentResults:
    """Train both uplift models and collect comparison results."""
    modeling_df = prepared.modeling_df.copy()
    X = modeling_df[prepared.feature_columns]
    treatment = modeling_df["treatment"]
    outcome = modeling_df["outcome"]
    X_train, X_test, t_train, t_test, y_train, y_test = train_test_split(
        X,
        treatment,
        outcome,
        test_size=test_size,
        random_state=random_seed,
        stratify=treatment,
    )
    propensity_model, propensity_auc = fit_propensity_model(X_train, t_train)
    propensity_train = propensity_model.predict_proba(X_train)[:, 1]
    propensity_test = propensity_model.predict_proba(X_test)[:, 1]
    t_results = train_single_t_learner(X_train, t_train, y_train, X_test, y_test, t_test)
    x_results = train_single_x_learner(
        X_train,
        t_train,
        y_train,
        X_test,
        y_test,
        t_test,
        propensity_train,
        propensity_test,
    )
    winner, recommendation = choose_winner(t_results, x_results)
    test_df = pd.DataFrame(
        {
            "outcome": y_test.to_numpy(),
            "treatment": t_test.to_numpy(),
            "t_uplift": t_results.uplift_scores,
            "x_uplift": x_results.uplift_scores,
        },
        index=X_test.index,
    )
    return ExperimentResults(
        prepared=prepared,
        test_df=test_df,
        X_test=X_test,
        y_test=y_test,
        treatment_test=t_test,
        propensity_auc=propensity_auc,
        t_learner=t_results,
        x_learner=x_results,
        winner=winner,
        recommendation=recommendation,
    )


def run_experiment_with_guardrails(
    prepared: PreparedDataset,
    test_size: float,
    random_seed: int,
) -> ExperimentResults | None:
    """Run model training and show a friendly error if it fails."""
    try:
        with st.spinner("Training T-Learner, X-Learner, and propensity model..."):
            results = run_experiment(prepared, test_size, random_seed)
    except Exception as exc:  # noqa: BLE001
        st.error(
            "Model training could not complete. "
            f"Please verify the uploaded data has enough treated and control examples. Details: {exc}"
        )
        return None
    st.session_state["results"] = results
    return results


def selected_model_results(results: ExperimentResults, selected_model: str) -> ModelArtifacts:
    """Return the active model artifacts by name."""
    if selected_model == "X-Learner":
        return results.x_learner
    return results.t_learner


def build_effect_summary(scores: np.ndarray) -> tuple[float, float, float]:
    """Compute positive, negative, and near-zero uplift shares."""
    positive_pct = float((scores > 0).mean() * 100)
    negative_pct = float((scores < 0).mean() * 100)
    neutral_pct = float(((scores == 0)).mean() * 100)
    return positive_pct, negative_pct, neutral_pct


def effect_filter_bounds(scores: np.ndarray) -> tuple[float, float]:
    """Return slider-friendly bounds for effect filtering."""
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    if min_score == max_score:
        max_score = min_score + 1e-6
    return min_score, max_score


def render_recommendation_box(results: ExperimentResults) -> None:
    """Render a short plain-language recommendation based on model comparison."""
    st.success(f"Recommended model: **{results.winner}**")
    st.write(results.recommendation)
    st.caption(
        "Interpretation tip: prioritize customers with consistently positive uplift, "
        "and avoid blanket treatment for segments with negative or near-zero effect."
    )


def build_filtered_effect_frame(
    X_test: pd.DataFrame,
    scores: np.ndarray,
    min_effect: float,
    max_effect: float,
) -> pd.DataFrame:
    """Create a filtered frame for downstream effect analysis."""
    effect_df = X_test.copy()
    effect_df["uplift"] = scores
    mask = effect_df["uplift"].between(min_effect, max_effect)
    return effect_df.loc[mask].copy()


def render_template_download() -> None:
    """Render a downloadable CSV template for uploads."""
    template = build_template_dataframe()
    buffer = io.BytesIO()
    template.to_csv(buffer, index=False)
    st.download_button(
        label="Download template CSV",
        data=buffer.getvalue(),
        file_name="uplift_template.csv",
        mime="text/csv",
    )


def tab1_setup() -> None:
    """Render the experiment setup tab."""
    st.subheader("Experiment Setup")
    st.write("Upload your own treatment dataset or start with a sample Hillstrom slice.")
    left, right = st.columns([1.3, 1.0])
    with left:
        source = st.radio(
            "Data source",
            options=["Use sample Hillstrom data", "Upload custom CSV"],
            horizontal=True,
        )
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if st.button("Load selected dataset", type="primary", use_container_width=True):
            if source == "Upload custom CSV":
                if uploaded_file is None:
                    st.session_state["data_error"] = "Please upload a CSV file before loading."
                else:
                    loaded_df = read_uploaded_csv(uploaded_file)
                    safe_prepare_dataset(loaded_df, source_name=uploaded_file.name)
            else:
                safe_prepare_dataset(load_sample_hillstrom(), source_name="Sample Hillstrom data")
        render_template_download()
    with right:
        st.info(
            "Expected columns: `user_id` (optional), `treatment`, `outcome`, "
            "and one or more feature columns. Binary `treatment` and `outcome` are required."
        )
        st.caption(
            "If your features are categorical, the app will one-hot encode them automatically."
        )
    if st.session_state["data_error"]:
        st.error(st.session_state["data_error"])
    prepared = st.session_state.get("prepared_data")
    if prepared is None:
        st.warning("Load a dataset to unlock model comparison, treatment-effect analysis, and SHAP views.")
        return
    render_dataset_kpis(prepared)
    chart_col, stats_col = st.columns([0.9, 1.1])
    with chart_col:
        split_df = get_treatment_split(prepared.raw_df)
        split_chart = px.pie(
            split_df,
            names="label",
            values="count",
            title="Treatment Split",
            color="label",
            color_discrete_map={"Control": "#dc2626", "Treated": "#16a34a"},
        )
        st.plotly_chart(split_chart, use_container_width=True)
    with stats_col:
        st.dataframe(build_basic_stats(prepared.raw_df), use_container_width=True, height=350)
    with st.expander("Preview prepared modeling data", expanded=False):
        st.dataframe(prepared.modeling_df.head(20), use_container_width=True)


def tab2_comparison() -> None:
    """Render the model comparison tab."""
    st.subheader("Model Comparison")
    prepared = st.session_state.get("prepared_data")
    if prepared is None:
        st.info("Load data in Tab 1 before comparing models.")
        return
    controls = st.columns(3)
    test_size = controls[0].slider("Test size", min_value=0.1, max_value=0.5, value=0.3, step=0.05)
    random_seed = controls[1].slider("Random seed", min_value=1, max_value=999, value=42, step=1)
    run_clicked = controls[2].button("Run model comparison", type="primary", use_container_width=True)
    if run_clicked:
        run_experiment_with_guardrails(prepared, test_size=test_size, random_seed=random_seed)
    results = st.session_state.get("results")
    if results is None:
        st.warning("Run the comparison to compute AUUC, Qini, timing, and recommendations.")
        return
    metric_cols = st.columns(5)
    metric_cols[0].metric("Propensity AUC", f"{results.propensity_auc:.3f}")
    metric_cols[1].metric("T-Learner AUUC", f"{results.t_learner.auuc_score:.2f}")
    metric_cols[2].metric("X-Learner AUUC", f"{results.x_learner.auuc_score:.2f}")
    metric_cols[3].metric("T-Learner Time", f"{results.t_learner.train_seconds:.2f}s")
    metric_cols[4].metric("X-Learner Time", f"{results.x_learner.train_seconds:.2f}s")
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.plotly_chart(make_qini_bar_chart(results), use_container_width=True)
    with chart_right:
        st.plotly_chart(make_auuc_line_chart(results), use_container_width=True)
    render_recommendation_box(results)


def tab3_analysis() -> None:
    """Render the treatment-effect analysis tab."""
    st.subheader("Treatment Effect Analysis")
    results = st.session_state.get("results")
    if results is None:
        st.info("Run model comparison in Tab 2 before exploring heterogeneous treatment effects.")
        return
    selected_model = st.radio(
        "Model for HTE analysis",
        options=["T-Learner", "X-Learner"],
        horizontal=True,
    )
    model_results = selected_model_results(results, selected_model)
    min_bound, max_bound = effect_filter_bounds(model_results.uplift_scores)
    selected_range = st.slider(
        "Filter by uplift magnitude",
        min_value=min_bound,
        max_value=max_bound,
        value=(min_bound, max_bound),
    )
    filtered_effect_df = build_filtered_effect_frame(
        results.X_test,
        model_results.uplift_scores,
        min_effect=selected_range[0],
        max_effect=selected_range[1],
    )
    if filtered_effect_df.empty:
        st.warning("No customers fall within the selected uplift range.")
        return
    positive_pct, negative_pct, neutral_pct = build_effect_summary(model_results.uplift_scores)
    stat_cols = st.columns(3)
    stat_cols[0].metric("Benefit from Treatment", f"{positive_pct:.1f}%")
    stat_cols[1].metric("Potentially Harmed", f"{negative_pct:.1f}%")
    stat_cols[2].metric("Near Zero Effect", f"{neutral_pct:.1f}%")
    st.write(
        f"Using **{selected_model}**, about **{positive_pct:.1f}%** of customers show positive uplift "
        f"while **{negative_pct:.1f}%** show negative uplift."
    )
    hist_col, segment_col = st.columns([1.2, 0.8])
    with hist_col:
        st.plotly_chart(
            make_histogram(filtered_effect_df["uplift"].to_numpy(), f"{selected_model} HTE Distribution"),
            use_container_width=True,
        )
    with segment_col:
        segment_df = percentile_breakdown(model_results.uplift_scores)
        st.dataframe(segment_df, use_container_width=True, height=320)
    st.caption(
        "Interpretation tip: focus treatment on high-uplift segments first, and investigate negative-uplift segments before sending offers."
    )


def tab4_feature_importance() -> None:
    """Render the SHAP feature importance tab."""
    st.subheader("Feature Importance (SHAP)")
    results = st.session_state.get("results")
    if results is None:
        st.info("Run model comparison in Tab 2 before viewing SHAP feature importance.")
        return
    selected_model = st.radio(
        "Model for SHAP view",
        options=["T-Learner", "X-Learner"],
        horizontal=True,
        key="shap_model_selector",
    )
    model_results = selected_model_results(results, selected_model)
    importance_df = model_results.feature_importance.copy()
    if not SHAP_AVAILABLE:
        st.warning(
            "SHAP is unavailable in this environment, so this tab is using tree-model feature importance instead. "
            f"Import issue: {SHAP_IMPORT_ERROR}"
        )
    st.plotly_chart(
        make_feature_importance_chart(importance_df, f"Top SHAP Drivers — {selected_model}"),
        use_container_width=True,
    )
    top_features = (
        importance_df.sort_values("importance", ascending=False)["feature"].head(10).tolist()
    )
    increase_text, decrease_text = summarize_effect_direction(
        results.X_test,
        model_results.uplift_scores,
        top_features,
    )
    tip_left, tip_right = st.columns(2)
    tip_left.success(increase_text)
    tip_right.error(decrease_text)
    with st.expander("Optional waterfall-style local explanation", expanded=False):
        st.caption(
            "This lightweight demo focuses on global SHAP ranking. "
            "For production review, pair this with per-customer waterfall explanations in a notebook."
        )
        top_row_index = int(np.argmax(model_results.uplift_scores))
        example_row = results.X_test.iloc[[top_row_index]]
        st.dataframe(example_row, use_container_width=True)


def main() -> None:
    """Render the Streamlit application layout."""
    initialize_state()
    tabs = st.tabs(
        [
            "1. Experiment Setup",
            "2. Model Comparison",
            "3. Treatment Effect Analysis",
            "4. Feature Importance (SHAP)",
        ]
    )
    with tabs[0]:
        tab1_setup()
    with tabs[1]:
        tab2_comparison()
    with tabs[2]:
        tab3_analysis()
    with tabs[3]:
        tab4_feature_importance()


if __name__ == "__main__":
    main()
