"""Generate the project's Jupyter notebooks from source.

Run once (``python notebooks/_build_notebooks.py``) to (re)create the EDA and
modeling notebooks.  Keeping the notebooks generated from a script guarantees
they stay in sync with the ``src`` package and remain diff-friendly.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def preamble():
    return code(
        "import sys, os\n"
        "sys.path.insert(0, os.path.abspath('..'))  # make `src` importable\n"
        "import pandas as pd, numpy as np\n"
        "import matplotlib.pyplot as plt, seaborn as sns\n"
        "sns.set_theme(style='whitegrid')\n"
        "pd.set_option('display.max_columns', 40)"
    )


# ── 01 — EDA ──────────────────────────────────────────────────────
def build_eda():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# 01 · Exploratory Data Analysis\n"
           "**Mobile Price Prediction — FYP**  \n"
           "University of Education, Lahore (Vehari Campus)\n\n"
           "This notebook explores the Kaggle *Mobile Price Classification* "
           "dataset and the engineered features (proposal §5.2)."),
        preamble(),
        md("## 1. Load the data\n"
           "`build_dataset()` loads the raw Kaggle CSV and attaches the "
           "demonstration proxy `price_usd` for the regression sub-task."),
        code("from src.data.make_dataset import build_dataset\n"
             "from src.features.build_features import add_engineered_features\n\n"
             "df = add_engineered_features(build_dataset(save=False))\n"
             "print(df.shape)\n"
             "df.head()"),
        md("## 2. Class balance\n"
           "The classification target `price_range` is perfectly balanced "
           "(500 phones per tier)."),
        code("df['price_range'].value_counts().sort_index()"),
        md("## 3. Price distribution & spec relationships"),
        code("fig, ax = plt.subplots(1, 2, figsize=(13,4))\n"
             "sns.histplot(df['price_usd'], bins=40, kde=True, ax=ax[0])\n"
             "ax[0].set_title('price_usd distribution (proxy)')\n"
             "sns.scatterplot(data=df, x='ram', y='price_usd', hue='price_range',\n"
             "                palette='viridis', alpha=0.6, ax=ax[1])\n"
             "ax[1].set_title('RAM vs price'); plt.tight_layout()"),
        md("## 4. Correlation heatmap"),
        code("plt.figure(figsize=(13,10))\n"
             "sns.heatmap(df.select_dtypes('number').corr(), cmap='coolwarm',\n"
             "            center=0, square=True, cbar_kws={'shrink':0.7})\n"
             "plt.title('Correlation heatmap'); plt.show()"),
        md("**Key takeaway:** `ram` has by far the strongest correlation with "
           "`price_range` — a linear relationship that lets even Logistic "
           "Regression classify tiers very accurately."),
        md("## 5. Regenerate all EDA figures\n"
           "All figures used in the report are produced by the reusable module:"),
        code("from src.visualization.eda import run_all\n"
             "run_all()  # writes PNGs to ../reports/figures/"),
    ]
    nbf.write(nb, HERE / "01_eda.ipynb")
    print("wrote 01_eda.ipynb")


# ── 02 — Modeling ─────────────────────────────────────────────────
def build_modeling():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# 02 · Modeling, Tuning & Explainability\n"
           "**Mobile Price Prediction — FYP**\n\n"
           "Trains and compares the six classifiers, tunes the gradient-boosting "
           "models with Optuna, evaluates on a held-out test set, and produces "
           "SHAP explanations (proposal §7–§9)."),
        preamble(),
        md("## 1. Run the full training pipeline\n"
           "This trains every model, runs Optuna on XGBoost/LightGBM, saves the "
           "best artifacts, and writes `reports/metrics.json`.  \n"
           "*(Use `main(fast=True)` for a quick run with fewer trials.)*"),
        code("from src.models.train import main\n"
             "report = main(fast=False)"),
        md("## 2. Model comparison table"),
        code("import json\n"
             "report = json.load(open('../reports/metrics.json'))\n"
             "rows = []\n"
             "for name, r in report['classification'].items():\n"
             "    rows.append({'model': name,\n"
             "                 'cv_f1_macro': round(r['cv']['f1_macro_mean'],4),\n"
             "                 'test_accuracy': round(r['test']['accuracy'],4),\n"
             "                 'test_f1_macro': round(r['test']['f1_macro'],4),\n"
             "                 'roc_auc_ovr': round(r['test'].get('roc_auc_ovr') or 0,4)})\n"
             "pd.DataFrame(rows).sort_values('test_f1_macro', ascending=False)"),
        md("## 3. Regression results"),
        code("pd.DataFrame(report['regression']).T.round(4)"),
        md("## 4. SHAP feature importance\n"
           "Computed on the best tree model (proposal Fig. 4)."),
        code("pd.DataFrame(report['shap_top_features'],\n"
             "             columns=['feature','mean_abs_shap']).head(10)"),
        code("from IPython.display import Image\n"
             "Image('../reports/figures/09_shap_importance.png')"),
        md("## 5. Try a single prediction"),
        code("from src.models.predict import get_service, EXAMPLE_PHONE\n"
             "get_service().predict(EXAMPLE_PHONE)"),
        md("## 6. Acceptance targets\n"
           "Classification accuracy ≥ 93 %, F1-macro ≥ 0.92, regression "
           "R² ≥ 0.90, MAPE < 12 % (proposal §12.1)."),
        code("best = report['classification'][report['best_classifier']]['test']\n"
             "reg  = report['regression'][report['best_regressor']]\n"
             "print('accuracy :', round(best['accuracy'],4))\n"
             "print('f1_macro :', round(best['f1_macro'],4))\n"
             "print('reg R2   :', round(reg['r2'],4))\n"
             "print('reg MAPE :', round(reg['mape'],4))"),
    ]
    nbf.write(nb, HERE / "02_modeling.ipynb")
    print("wrote 02_modeling.ipynb")


if __name__ == "__main__":
    build_eda()
    build_modeling()
