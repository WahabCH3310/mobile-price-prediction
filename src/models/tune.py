"""Hyper-parameter optimisation with Optuna (FYP proposal §7.1).

Bayesian (TPE) search over the XGBoost / LightGBM classifier hyper-parameters
listed in the proposal, scored by 5-fold stratified-CV macro-F1.  Returns the
best parameter dict, which the training script feeds back into the model zoo.
"""
from __future__ import annotations

import warnings

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.models.model_zoo import get_classifier, NEEDS_SCALING
from src.models.preprocessing import build_preprocessor

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


def _suggest_params(trial: optuna.Trial, model_name: str) -> dict:
    """Search space matching proposal §7.1."""
    common = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.30, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    if model_name == "lightgbm":
        common["num_leaves"] = trial.suggest_int("num_leaves", 15, 255)
    return common


def tune_model(
    model_name: str,
    X_train,
    y_train,
    seed: int = 42,
    n_trials: int = 40,
    timeout: int = 600,
    cv_folds: int = 5,
) -> dict:
    """Run an Optuna study and return the best hyper-parameters."""
    scale = NEEDS_SCALING[model_name]
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, model_name)
        params["n_jobs"] = 1  # single-thread the model; parallelise across folds
        clf = get_classifier(model_name, seed=seed, params=params)
        pipe = Pipeline([
            ("pre", build_preprocessor(X_train, scale_numeric=scale)),
            ("clf", clf),
        ])
        # parallelise across the CV folds, not inside the model, to avoid
        # thread oversubscription (which made LightGBM studies very slow).
        scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                 scoring="f1_macro", n_jobs=cv_folds)
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    print(f"[tune] optimising {model_name}: {n_trials} trials (timeout {timeout}s)")
    study.optimize(objective, n_trials=n_trials, timeout=timeout,
                   show_progress_bar=False)
    print(f"[tune] {model_name} best CV F1-macro = {study.best_value:.4f}")
    return study.best_params
