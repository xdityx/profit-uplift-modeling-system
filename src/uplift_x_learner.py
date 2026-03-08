import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier


class XLearner:

    def __init__(self):

        self.mu1 = RandomForestClassifier(n_estimators=200, random_state=42)
        self.mu0 = RandomForestClassifier(n_estimators=200, random_state=42)

        self.tau_treated = RandomForestRegressor(n_estimators=200, random_state=42)
        self.tau_control = RandomForestRegressor(n_estimators=200, random_state=42)

    def fit(self, X, treatment, y):

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

    def predict_uplift(self, X, propensity):

        tau_t = self.tau_treated.predict(X)
        tau_c = self.tau_control.predict(X)

        uplift = propensity * tau_c + (1 - propensity) * tau_t
        return uplift