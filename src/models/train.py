"""End-to-end training pipeline (FYP proposal §7).

Steps
-----
1. Build dataset (raw Kaggle + proxy price) and engineered features.
2. Stratified train/test split.
3. Train & 5-fold CV-score all six classifiers; Optuna-tune XGBoost/LightGBM.
4. Train the regression models on the continuous price target.
5. Pick the best classifier (macro-F1) and regressor (R²); persist artifacts.
6. Emit metrics.json, comparison/confusion/fit plots and SHAP importances.

Run:  python -m src.models.train           # full run (uses config.yaml)
      python -m src.models.train --fast     # quick run, fewer Optuna trials
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from src.config import get_config
from src.data.make_dataset import build_dataset
from src.features.build_features import add_engineered_features
from src.models import evaluate
from src.models.model_zoo import (
    NEEDS_SCALING, PRETTY_NAMES, get_classifier, get_regressor,
)
from src.models.preprocessing import build_preprocessor
from src.models.tune import tune_model

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────
#  Data preparation
# ──────────────────────────────────────────────────────────────────
def prepare_data(cfg):
    df = add_engineered_features(build_dataset(save=True))
    y_clf = df[cfg.data.target_classification]
    y_reg = df[cfg.data.target_regression]
    X = df.drop(columns=[cfg.data.target_classification, cfg.data.target_regression])

    X_tr, X_te, ycl_tr, ycl_te, yrg_tr, yrg_te = train_test_split(
        X, y_clf, y_reg,
        test_size=cfg.data.test_size,
        stratify=y_clf,
        random_state=cfg.project.random_seed,
    )
    # persist the test features for SHAP / the app
    proc = cfg.path("processed_dir")
    proc.mkdir(parents=True, exist_ok=True)
    X_te.to_csv(proc / "X_test.csv", index=False)
    ycl_te.to_csv(proc / "y_test_clf.csv", index=False)
    return X, X_tr, X_te, ycl_tr, ycl_te, yrg_tr, yrg_te


# ──────────────────────────────────────────────────────────────────
#  Classification
# ──────────────────────────────────────────────────────────────────
def train_classifiers(cfg, X_tr, X_te, y_tr, y_te, n_trials):
    seed = cfg.project.random_seed
    cv = StratifiedKFold(n_splits=cfg.training.cv_folds, shuffle=True, random_state=seed)
    tuned = set(cfg.tuning.tuned_models) if cfg.tuning.enabled else set()
    results, fitted = {}, {}

    for name in cfg.training.classifiers:
        t0 = time.time()
        params = {}
        if name in tuned:
            params = tune_model(name, X_tr, y_tr, seed=seed,
                                n_trials=n_trials,
                                timeout=cfg.tuning.timeout_seconds,
                                cv_folds=cfg.training.cv_folds)
        clf = get_classifier(name, seed=seed, params=params)
        pipe = Pipeline([
            ("pre", build_preprocessor(X_tr, scale_numeric=NEEDS_SCALING[name])),
            ("clf", clf),
        ])

        cv_res = cross_validate(pipe, X_tr, y_tr, cv=cv,
                                scoring=["accuracy", "f1_macro"], n_jobs=-1)
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        y_proba = pipe.predict_proba(X_te) if hasattr(pipe, "predict_proba") else None
        test_metrics = evaluate.classification_metrics(y_te, y_pred, y_proba)

        results[PRETTY_NAMES[name]] = {
            "cv": {
                "accuracy_mean": float(cv_res["test_accuracy"].mean()),
                "accuracy_std": float(cv_res["test_accuracy"].std()),
                "f1_macro_mean": float(cv_res["test_f1_macro"].mean()),
                "f1_macro_std": float(cv_res["test_f1_macro"].std()),
            },
            "test": test_metrics,
            "best_params": params,
            "train_seconds": round(time.time() - t0, 1),
        }
        fitted[PRETTY_NAMES[name]] = pipe
        print(f"[train] {PRETTY_NAMES[name]:<20} "
              f"CV-F1={results[PRETTY_NAMES[name]]['cv']['f1_macro_mean']:.4f} "
              f"test-acc={test_metrics['accuracy']:.4f} "
              f"({results[PRETTY_NAMES[name]]['train_seconds']}s)")

    best_name = max(results, key=lambda n: results[n]["test"]["f1_macro"])
    return results, fitted, best_name


# ──────────────────────────────────────────────────────────────────
#  Regression
# ──────────────────────────────────────────────────────────────────
def train_regressors(cfg, X_tr, X_te, y_tr, y_te):
    seed = cfg.project.random_seed
    results, fitted = {}, {}
    for name in cfg.training.regressors:
        reg = get_regressor(name, seed=seed)
        pipe = Pipeline([
            ("pre", build_preprocessor(X_tr, scale_numeric=NEEDS_SCALING[name])),
            ("reg", reg),
        ])
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        results[PRETTY_NAMES[name]] = evaluate.regression_metrics(y_te, y_pred)
        fitted[PRETTY_NAMES[name]] = pipe
        print(f"[train] {PRETTY_NAMES[name]:<20} "
              f"R2={results[PRETTY_NAMES[name]]['r2']:.4f} "
              f"MAPE={results[PRETTY_NAMES[name]]['mape']:.4f}")
    best_name = max(results, key=lambda n: results[n]["r2"])
    return results, fitted, best_name


# ──────────────────────────────────────────────────────────────────
#  Orchestration
# ──────────────────────────────────────────────────────────────────
def main(fast: bool = False):
    cfg = get_config()
    seed = cfg.project.random_seed
    np.random.seed(seed)
    n_trials = 8 if fast else cfg.tuning.n_trials

    print("=" * 66)
    print(" Mobile Price Prediction — training pipeline")
    print("=" * 66)

    X, X_tr, X_te, ycl_tr, ycl_te, yrg_tr, yrg_te = prepare_data(cfg)
    print(f"[train] train={len(X_tr)}  test={len(X_te)}  features={X.shape[1]}\n")

    print("── Classification ────────────────────────────────────────────")
    clf_results, clf_fitted, best_clf = train_classifiers(
        cfg, X_tr, X_te, ycl_tr, ycl_te, n_trials)
    print(f"[train] BEST classifier: {best_clf}\n")

    print("── Regression ────────────────────────────────────────────────")
    reg_results, reg_fitted, best_reg = train_regressors(
        cfg, X_tr, X_te, yrg_tr, yrg_te)
    print(f"[train] BEST regressor: {best_reg}\n")

    # ── Persist artifacts ────────────────────────────────────────
    models_dir = cfg.path("models_dir")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf_fitted[best_clf], models_dir / "best_classifier.joblib")
    joblib.dump(reg_fitted[best_reg], models_dir / "best_regressor.joblib")

    metadata = {
        "feature_columns": list(X.columns),
        "tier_names": {0: "Budget", 1: "Mid", 2: "Premium", 3: "Flagship"},
        "best_classifier": best_clf,
        "best_regressor": best_reg,
        "feature_ranges": {c: [float(X[c].min()), float(X[c].max())]
                           for c in X.columns},
        "feature_medians": {c: float(X[c].median()) for c in X.columns},
    }
    with open(models_dir / "feature_metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)

    # ── Plots ────────────────────────────────────────────────────
    evaluate.plot_model_comparison(clf_results)
    best_pipe = clf_fitted[best_clf]
    evaluate.plot_confusion_matrix(ycl_te, best_pipe.predict(X_te), best_clf)
    best_reg_pipe = reg_fitted[best_reg]
    evaluate.plot_regression_fit(yrg_te, best_reg_pipe.predict(X_te), best_reg)

    # ── SHAP importance ──────────────────────────────────────────
    # The proposal's Fig. 4 uses tree-SHAP, so explain the best tree-based
    # classifier (fast, exact) rather than a linear model.
    tree_models = ["XGBoost", "LightGBM", "Random Forest"]
    shap_candidates = {n: clf_results[n]["test"]["f1_macro"]
                       for n in tree_models if n in clf_fitted}
    shap_model = max(shap_candidates, key=shap_candidates.get) if shap_candidates else best_clf
    shap_importance = {}
    try:
        from src.models.explain import global_importance
        print(f"[train] computing SHAP on {shap_model}")
        shap_importance = global_importance(clf_fitted[shap_model], X_te)
    except Exception as exc:
        print(f"[train] SHAP skipped: {exc}")

    # ── metrics.json ─────────────────────────────────────────────
    report = {
        "dataset": {"rows": int(len(X)), "n_features": int(X.shape[1]),
                    "test_size": cfg.data.test_size},
        "targets": dict(cfg.targets),
        "classification": clf_results,
        "regression": reg_results,
        "best_classifier": best_clf,
        "best_regressor": best_reg,
        "shap_model": shap_model,
        "shap_top_features": list(shap_importance.items())[:10],
    }
    reports_dir = cfg.path("reports_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "metrics.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n[train] wrote metrics -> {reports_dir / 'metrics.json'}")

    # ── Target check ─────────────────────────────────────────────
    best = clf_results[best_clf]["test"]
    reg_best = reg_results[best_reg]
    print("\n── Acceptance targets ────────────────────────────────────────")
    print(f"  classification accuracy : {best['accuracy']:.4f} "
          f"(target ≥ {cfg.targets.classification_accuracy})  "
          f"{'PASS' if best['accuracy'] >= cfg.targets.classification_accuracy else 'below'}")
    print(f"  classification F1-macro : {best['f1_macro']:.4f} "
          f"(target ≥ {cfg.targets.classification_f1_macro})  "
          f"{'PASS' if best['f1_macro'] >= cfg.targets.classification_f1_macro else 'below'}")
    print(f"  regression R²           : {reg_best['r2']:.4f} "
          f"(target ≥ {cfg.targets.regression_r2})  "
          f"{'PASS' if reg_best['r2'] >= cfg.targets.regression_r2 else 'below'}")
    print("=" * 66)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all models.")
    parser.add_argument("--fast", action="store_true",
                        help="Fewer Optuna trials for a quick smoke run.")
    args = parser.parse_args()
    main(fast=args.fast)
