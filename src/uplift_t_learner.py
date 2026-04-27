"""
Module: Implement a T-Learner for uplift estimation.

This module trains separate outcome models for treated and control customers and
uses their difference as the estimated treatment effect. It is used as a simple
baseline meta-learner within the uplift modeling pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class TLearner:
    """Estimate uplift by contrasting treated and control outcome models."""

    def __init__(self) -> None:
        """Initialize paired random forest classifiers for both treatment arms."""
        self.model_treated = RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
        self.model_control = RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        treatment: pd.Series | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> None:
        """
        Fit separate outcome models on treated and control observations.

        Args:
            X: Feature matrix used to predict conversion outcomes.
            treatment: Binary treatment indicator for each observation.
            y: Binary outcome labels aligned with `X`.

        Returns:
            None.
        """
        X_treated = X[treatment == 1]
        y_treated = y[treatment == 1]

        X_control = X[treatment == 0]
        y_control = y[treatment == 0]

        self.model_treated.fit(X_treated, y_treated)
        self.model_control.fit(X_control, y_control)

    def predict_uplift(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Estimate uplift as treated probability minus control probability.

        Args:
            X: Feature matrix to score with both fitted outcome models.

        Returns:
            A NumPy array of individual treatment effect estimates.
        """
        treated_pred = self.model_treated.predict_proba(X)[:, 1]
        control_pred = self.model_control.predict_proba(X)[:, 1]

        uplift = treated_pred - control_pred
        return uplift
