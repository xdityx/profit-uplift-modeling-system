import numpy as np
from sklearn.ensemble import RandomForestClassifier


class TLearner:

    def __init__(self):
        self.model_treated = RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
        self.model_control = RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )

    def fit(self, X, treatment, y):

        X_treated = X[treatment == 1]
        y_treated = y[treatment == 1]

        X_control = X[treatment == 0]
        y_control = y[treatment == 0]

        self.model_treated.fit(X_treated, y_treated)
        self.model_control.fit(X_control, y_control)

    def predict_uplift(self, X):

        treated_pred = self.model_treated.predict_proba(X)[:, 1]
        control_pred = self.model_control.predict_proba(X)[:, 1]

        uplift = treated_pred - control_pred
        return uplift