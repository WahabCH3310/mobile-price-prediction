"""Unit tests for the core pipeline (FYP proposal §11 — Testing).

Run with:  pytest -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.make_dataset import add_proxy_price, load_raw, EXPECTED_COLUMNS
from src.features.build_features import (
    ENGINEERED_COLUMNS, add_engineered_features,
)


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    return load_raw()


# ── Data ──────────────────────────────────────────────────────────
def test_raw_has_expected_schema(raw_df):
    assert list(raw_df.columns) == EXPECTED_COLUMNS
    assert len(raw_df) == 2000
    assert raw_df.isnull().sum().sum() == 0


def test_price_range_is_balanced(raw_df):
    counts = raw_df["price_range"].value_counts()
    assert set(counts.index) == {0, 1, 2, 3}
    assert counts.min() == counts.max() == 500


def test_proxy_price_is_reasonable(raw_df):
    df = add_proxy_price(raw_df, seed=42)
    assert "price_usd" in df.columns
    assert df["price_usd"].between(60, 1600).all()
    # higher tiers should be more expensive on average
    means = df.groupby("price_range")["price_usd"].mean()
    assert means.is_monotonic_increasing


def test_proxy_price_is_deterministic(raw_df):
    a = add_proxy_price(raw_df, seed=42)["price_usd"]
    b = add_proxy_price(raw_df, seed=42)["price_usd"]
    assert np.allclose(a, b)


# ── Feature engineering ───────────────────────────────────────────
def test_engineered_columns_created(raw_df):
    out = add_engineered_features(raw_df)
    for col in ENGINEERED_COLUMNS:
        assert col in out.columns
    assert not out[ENGINEERED_COLUMNS].isnull().any().any()


def test_is_flagship_is_binary(raw_df):
    out = add_engineered_features(raw_df)
    assert set(out["is_flagship"].unique()).issubset({0, 1})


def test_feature_engineering_is_pure(raw_df):
    before = raw_df.copy()
    _ = add_engineered_features(raw_df)
    pd.testing.assert_frame_equal(before, raw_df)  # input unchanged


# ── Inference service (requires trained models) ──────────────────
@pytest.mark.skipif(
    not (__import__("pathlib").Path("models/best_classifier.joblib").exists()),
    reason="models not trained yet (run `make train`)",
)
def test_prediction_service_contract():
    from src.models.predict import EXAMPLE_PHONE, get_service
    res = get_service().predict(EXAMPLE_PHONE)
    assert res["price_range"] in {0, 1, 2, 3}
    assert res["price_tier"] in {"Budget", "Mid", "Premium", "Flagship"}
    assert 60 <= res["estimated_price_usd"] <= 1600
    assert abs(sum(res["class_probabilities"].values()) - 1.0) < 1e-6
