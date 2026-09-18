"""SHAP explainability (FYP proposal §7.2, §9).

Produces global feature-importance plots for the best tree-based classifier:
  • mean |SHAP| bar chart (proposal Fig. 4),
  • beeswarm summary plot,
and a helper that returns a per-prediction contribution breakdown used by the
Streamlit app's waterfall explanation.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from src.config import get_config


def _transform(pipeline, X):
    """Apply every pipeline step except the final estimator; return array +names."""
    pre = pipeline[:-1]
    X_trans = pre.transform(X)
    try:
        names = list(pre.get_feature_names_out())
    except Exception:
        names = [f"f{i}" for i in range(X_trans.shape[1])]
    # strip the ColumnTransformer's "num__"/"cat__" prefixes for readability
    names = [n.split("__", 1)[-1] for n in names]
    return np.asarray(X_trans), names


def _save(fig, name: str) -> str:
    cfg = get_config()
    out_dir = cfg.path("figures_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[shap] saved {path}")
    return str(path)


TREE_MODELS = ("RandomForest", "XGB", "LGBM", "GradientBoosting", "DecisionTree",
               "ExtraTrees")


def _build_explainer(estimator, X_background):
    """Pick the right SHAP explainer for the estimator type."""
    cls_name = type(estimator).__name__
    if any(t in cls_name for t in TREE_MODELS):
        return shap.TreeExplainer(estimator), "tree"
    if "Logistic" in cls_name or "Linear" in cls_name:
        return shap.LinearExplainer(estimator, X_background), "linear"
    # generic fallback (slow) — sample a small background
    bg = shap.sample(X_background, min(100, len(X_background)))
    return shap.KernelExplainer(estimator.predict_proba, bg), "kernel"


def global_importance(pipeline, X_sample, max_display: int = 15) -> dict:
    """Compute SHAP values and save global-importance figures.

    Returns a dict of {feature: mean_abs_shap} sorted descending.
    """
    estimator = pipeline[-1]
    X_trans, names = _transform(pipeline, X_sample)

    explainer, kind = _build_explainer(estimator, X_trans)
    shap_values = explainer.shap_values(X_trans)

    # Normalise shapes across shap versions / multiclass vs single output.
    if isinstance(shap_values, list):                      # list per class
        stacked = np.stack([np.abs(sv) for sv in shap_values], axis=0)
        mean_abs = stacked.mean(axis=0).mean(axis=0)       # -> (n_features,)
    else:
        arr = np.abs(np.asarray(shap_values))
        if arr.ndim == 3:                                  # (n, feat, class)
            mean_abs = arr.mean(axis=2).mean(axis=0)
        else:                                              # (n, feat)
            mean_abs = arr.mean(axis=0)

    importance = dict(sorted(zip(names, mean_abs.tolist()),
                             key=lambda kv: kv[1], reverse=True))

    # Bar chart (proposal Fig. 4)
    top = list(importance.items())[:max_display][::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([k for k, _ in top], [v for _, v in top], color="#2a9d8f")
    ax.set(title="SHAP feature importance (mean |SHAP| value)",
           xlabel="mean(|SHAP value|)")
    _save(fig, "09_shap_importance.png")

    # Beeswarm summary (use class 0 slice if multiclass for a clean plot)
    try:
        plt.figure()
        sv = shap_values[0] if isinstance(shap_values, list) else (
            shap_values[..., 0] if np.asarray(shap_values).ndim == 3 else shap_values
        )
        shap.summary_plot(sv, X_trans, feature_names=names, show=False,
                          max_display=max_display)
        _save(plt.gcf(), "10_shap_summary.png")
    except Exception as exc:
        print(f"[shap] beeswarm skipped: {exc}")

    return importance


if __name__ == "__main__":
    import joblib
    cfg = get_config()
    pipe = joblib.load(cfg.path("models_dir") / "best_classifier.joblib")
    import pandas as pd
    X = pd.read_csv(cfg.path("processed_dir") / "X_test.csv")
    imp = global_importance(pipe, X)
    print("Top features:", list(imp.items())[:8])
