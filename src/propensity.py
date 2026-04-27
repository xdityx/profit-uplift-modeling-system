"""
Module: Train and apply propensity score models.

This module handles treatment assignment modeling using observed covariates and
provides helpers to attach propensity scores back to a dataframe. It supports
the broader causal pipeline by estimating treatment likelihood for meta-learners.
"""

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def _get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Select feature columns used by the propensity model.

    Args:
        df: Modeling dataframe containing features and target columns.

    Returns:
        A list of covariate columns excluding treatment, outcome, and scores.
    """
    excluded = {"treatment", "outcome", "propensity_score"}
    return [column for column in df.columns if column not in excluded]


def train_propensity_model(df: pd.DataFrame) -> tuple[LogisticRegression, float]:
    """
    Fit a logistic model that predicts treatment assignment.

    Args:
        df: Modeling dataframe with encoded features and a `treatment` column.

    Returns:
        The fitted logistic regression model and test-set ROC-AUC.
    """
    X = df[_get_feature_columns(df)]
    y = df["treatment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)

    return model, auc


def compute_propensity_scores(
    model: ClassifierMixin,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Append estimated propensity scores to a modeling dataframe.

    Args:
        model: Fitted classifier exposing `predict_proba`.
        df: Modeling dataframe to score.

    Returns:
        The input dataframe with a new `propensity_score` column.
    """
    X = df[_get_feature_columns(df)]
    df["propensity_score"] = model.predict_proba(X)[:, 1]
    return df
