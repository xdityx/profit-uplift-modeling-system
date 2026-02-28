import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def train_propensity_model(df):

    X = df[["age", "income", "tenure", "usage"]]
    y = df["treatment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)

    return model, auc


def compute_propensity_scores(model, df):
    X = df[["age", "income", "tenure", "usage"]]
    df["propensity_score"] = model.predict_proba(X)[:, 1]
    return df