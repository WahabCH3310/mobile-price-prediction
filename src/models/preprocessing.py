"""Preprocessing pipeline builder (FYP proposal §5.3).

Builds a scikit-learn ``ColumnTransformer`` that:
  • median-imputes numeric columns (mode-imputes categoricals),
  • one-hot encodes any *string / categorical* columns (none in the raw Kaggle
    data, but ready for scraped `brand` / `os` / `processor` columns),
  • optionally StandardScales numeric columns — enabled for the distance /
    gradient based models (Logistic Regression, SVM, MLP) and disabled for the
    tree ensembles that don't need it.

Returning a fresh transformer per model keeps every estimator's preprocessing
self-contained inside its own Pipeline, so a saved model carries its own
preprocessing and can predict on raw feature rows.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols) by dtype."""
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    return numeric, categorical


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    """Construct a ColumnTransformer appropriate for ``X``.

    Parameters
    ----------
    X : the feature frame (used only to detect column types/names).
    scale_numeric : add a StandardScaler after imputation for numeric columns.
    """
    numeric, categorical = split_column_types(X)

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    transformers = [("num", numeric_pipe, numeric)]

    if categorical:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", categorical_pipe, categorical))

    return ColumnTransformer(transformers=transformers, remainder="drop")
