"""Model factory (FYP proposal §7).

Central registry of the six classifiers and three regressors used in the study.
Each factory returns an *unfitted* estimator; the training script wraps it in a
Pipeline together with the right preprocessing.

``NEEDS_SCALING`` records which models require StandardScaler'd inputs so the
training script can build the correct preprocessor per model.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

# Distance / gradient based models want scaled inputs; trees do not.
NEEDS_SCALING = {
    "logistic_regression": True,
    "svm": True,
    "mlp": True,
    "random_forest": False,
    "xgboost": False,
    "lightgbm": False,
}


def get_classifier(name: str, seed: int = 42, params: dict | None = None):
    """Return an unfitted classifier by name."""
    params = params or {}
    name = name.lower()
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000, random_state=seed, **params)
    if name == "svm":
        return SVC(probability=True, random_state=seed, **params)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.pop("n_estimators", 400),
            random_state=seed, n_jobs=-1, **params,
        )
    if name == "xgboost":
        defaults = dict(
            n_estimators=400, learning_rate=0.1, max_depth=6,
            subsample=0.9, colsample_bytree=0.9,
            objective="multi:softprob", num_class=4,
            eval_metric="mlogloss", tree_method="hist",
            random_state=seed, n_jobs=-1,
        )
        defaults.update(params)
        return XGBClassifier(**defaults)
    if name == "lightgbm":
        defaults = dict(
            n_estimators=400, learning_rate=0.1, max_depth=-1,
            num_leaves=31, subsample=0.9, colsample_bytree=0.9,
            objective="multiclass", num_class=4,
            random_state=seed, n_jobs=-1, verbose=-1,
        )
        defaults.update(params)
        return LGBMClassifier(**defaults)
    if name == "mlp":
        defaults = dict(
            hidden_layer_sizes=(128, 64), activation="relu",
            alpha=1e-3, max_iter=500, early_stopping=True, random_state=seed,
        )
        defaults.update(params)
        return MLPClassifier(**defaults)
    raise ValueError(f"Unknown classifier: {name}")


def get_regressor(name: str, seed: int = 42, params: dict | None = None):
    """Return an unfitted regressor by name."""
    params = params or {}
    name = name.lower()
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=params.pop("n_estimators", 400),
            random_state=seed, n_jobs=-1, **params,
        )
    if name == "xgboost":
        defaults = dict(
            n_estimators=500, learning_rate=0.08, max_depth=6,
            subsample=0.9, colsample_bytree=0.9,
            objective="reg:squarederror", tree_method="hist",
            random_state=seed, n_jobs=-1,
        )
        defaults.update(params)
        return XGBRegressor(**defaults)
    if name == "lightgbm":
        defaults = dict(
            n_estimators=500, learning_rate=0.08, max_depth=-1,
            num_leaves=31, subsample=0.9, colsample_bytree=0.9,
            objective="regression", random_state=seed, n_jobs=-1, verbose=-1,
        )
        defaults.update(params)
        return LGBMRegressor(**defaults)
    if name == "svm":
        return SVR(**params)
    if name == "mlp":
        defaults = dict(hidden_layer_sizes=(128, 64), max_iter=500,
                        early_stopping=True, random_state=seed)
        defaults.update(params)
        return MLPRegressor(**defaults)
    raise ValueError(f"Unknown regressor: {name}")


PRETTY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "svm": "SVM",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "mlp": "MLP (Neural Net)",
}
