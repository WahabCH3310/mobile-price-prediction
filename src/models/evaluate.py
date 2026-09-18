"""Evaluation metrics & diagnostic plots (FYP proposal §9).

Classification: accuracy, macro-F1, ROC-AUC (one-vs-rest), confusion matrix.
Regression:     MAE, RMSE, R², MAPE, predicted-vs-actual plot.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, ConfusionMatrixDisplay,
)

from src.config import get_config

TIER_NAMES = ["Budget", "Mid", "Premium", "Flagship"]


def classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Return the classification metrics from proposal §9.1."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc_ovr"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            )
        except Exception:
            metrics["roc_auc_ovr"] = None
    return metrics


def regression_metrics(y_true, y_pred) -> dict:
    """Return the regression metrics from proposal §9.2."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


def _save(fig, name: str) -> str:
    cfg = get_config()
    out_dir = cfg.path("figures_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] saved {path}")
    return str(path)


def plot_confusion_matrix(y_true, y_pred, model_name: str) -> str:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=TIER_NAMES)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}")
    return _save(fig, "06_confusion_matrix.png")


def plot_model_comparison(results: dict) -> str:
    """Bar chart of accuracy & F1 across all classifiers (proposal Fig. 3)."""
    names = list(results.keys())
    acc = [results[n]["test"]["accuracy"] for n in names]
    f1 = [results[n]["test"]["f1_macro"] for n in names]

    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, acc, width, label="Accuracy")
    ax.bar(x + width / 2, f1, width, label="F1 (macro)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model performance comparison (held-out test set)")
    for i, (a, f) in enumerate(zip(acc, f1)):
        ax.text(i - width / 2, a + 0.005, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(i + width / 2, f + 0.005, f"{f:.3f}", ha="center", fontsize=8)
    ax.legend()
    return _save(fig, "07_model_comparison.png")


def plot_regression_fit(y_true, y_pred, model_name: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, edgecolor="none")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1.5, label="perfect")
    ax.set(xlabel="Actual price (USD)", ylabel="Predicted price (USD)",
           title=f"Predicted vs actual — {model_name}")
    ax.legend()
    return _save(fig, "08_regression_fit.png")
