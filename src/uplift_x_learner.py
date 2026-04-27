"""
Module: Implement an X-Learner for uplift estimation.

This module fits outcome models, derives pseudo treatment effects, and combines
arm-specific effect models using propensity weighting. It supports the broader
pipeline with a meta-learner that can work well under treatment imbalance.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier


class XLearner:
    """Estimate uplift using pseudo-outcomes and propensity-weighted blending."""

    def __init__(self) -> None:
        """Initialize outcome and treatment-effect models for each arm."""
        self.mu1 = RandomForestClassifier(n_estimators=200, random_state=42)
        self.mu0 = RandomForestClassifier(n_estimators=200, random_state=42)

        self.tau_treated = RandomForestRegressor(n_estimators=200, random_state=42)
        self.tau_control = RandomForestRegressor(n_estimators=200, random_state=42)

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        treatment: pd.Series | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> None:
        """
        Fit the X-Learner outcome and pseudo-effect models.

        Args:
            X: Feature matrix used for outcome and effect estimation.
            treatment: Binary treatment indicator for each observation.
            y: Binary outcome labels aligned with `X`.

        Returns:
            None.
        """
        X_treated = X[treatment == 1]
        y_treated = y[treatment == 1]

        X_control = X[treatment == 0]
        y_control = y[treatment == 0]

        # outcome models
        self.mu1.fit(X_treated, y_treated)
        self.mu0.fit(X_control, y_control)

        # pseudo effects
        d_treated = y_treated - self.mu0.predict_proba(X_treated)[:,1]
        d_control = self.mu1.predict_proba(X_control)[:,1] - y_control

        # treatment effect models
        self.tau_treated.fit(X_treated, d_treated)
        self.tau_control.fit(X_control, d_control)

    def predict_uplift(
        self,
        X: pd.DataFrame | np.ndarray,
        propensity: pd.Series | np.ndarray,
    ) -> np.ndarray:
        """
        Blend arm-specific effect estimates using propensity weights.

        Args:
            X: Feature matrix to score with the fitted effect models.
            propensity: Estimated treatment probabilities for each row in `X`.

        Returns:
            A NumPy array of uplift estimates for each observation.
        """
        tau_t = self.tau_treated.predict(X)
        tau_c = self.tau_control.predict(X)

        uplift = propensity * tau_c + (1 - propensity) * tau_t
        return uplift
