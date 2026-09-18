"""Inference service shared by the Streamlit app and the FastAPI endpoint.

Loads the trained artifacts once, turns a dict of *raw* phone specs into the
full engineered feature row, and returns the price-tier classification, the
class probabilities, the continuous price estimate, and a per-prediction SHAP
contribution breakdown (used for the dashboard's explanation panel).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import get_config
from src.features.build_features import add_engineered_features

# The 20 raw inputs a user supplies (engineered features are derived from these).
RAW_INPUTS = [
    "battery_power", "blue", "clock_speed", "dual_sim", "fc", "four_g",
    "int_memory", "m_dep", "mobile_wt", "n_cores", "pc", "px_height",
    "px_width", "ram", "sc_h", "sc_w", "talk_time", "three_g",
    "touch_screen", "wifi",
]


class PredictionService:
    """Holds the loaded models and performs inference."""

    def __init__(self) -> None:
        cfg = get_config()
        mdir = cfg.path("models_dir")
        self.clf = joblib.load(mdir / "best_classifier.joblib")
        self.reg = joblib.load(mdir / "best_regressor.joblib")
        with open(mdir / "feature_metadata.json") as fh:
            self.meta = json.load(fh)
        self.feature_columns = self.meta["feature_columns"]
        self.tier_names = {int(k): v for k, v in self.meta["tier_names"].items()}
        self._reg_explainer = self._try_build_tree_explainer(self.reg)

    # ── helpers ──────────────────────────────────────────────────
    @staticmethod
    def _try_build_tree_explainer(pipeline):
        """Build a TreeExplainer for the regressor if it is tree-based."""
        try:
            import shap
            est = pipeline[-1]
            if any(t in type(est).__name__ for t in ("XGB", "LGBM", "RandomForest")):
                return shap.TreeExplainer(est)
        except Exception:
            pass
        return None

    def _prepare(self, raw: dict) -> pd.DataFrame:
        """Raw spec dict -> single-row DataFrame with all model features."""
        missing = set(RAW_INPUTS) - set(raw)
        if missing:
            raise ValueError(f"Missing required inputs: {sorted(missing)}")
        row = pd.DataFrame([{k: raw[k] for k in RAW_INPUTS}])
        row = add_engineered_features(row)
        # ensure column order matches training
        return row.reindex(columns=self.feature_columns)

    # ── public API ───────────────────────────────────────────────
    def predict(self, raw: dict, explain: bool = True) -> dict:
        X = self._prepare(raw)

        tier = int(self.clf.predict(X)[0])
        proba = self.clf.predict_proba(X)[0]
        price = float(self.reg.predict(X)[0])

        result = {
            "price_range": tier,
            "price_tier": self.tier_names.get(tier, str(tier)),
            "class_probabilities": {
                self.tier_names[i]: round(float(p), 4) for i, p in enumerate(proba)
            },
            "estimated_price_usd": round(price, 2),
        }
        if explain:
            result["top_contributions"] = self._explain(X)
        return result

    def _explain(self, X: pd.DataFrame, top_k: int = 6) -> list[dict]:
        """Per-prediction SHAP contributions to the estimated price."""
        if self._reg_explainer is None:
            return []
        try:
            X_trans = self.reg[:-1].transform(X)
            names = [n.split("__", 1)[-1]
                     for n in self.reg[:-1].get_feature_names_out()]
            sv = self._reg_explainer.shap_values(X_trans)
            sv = np.asarray(sv)[0]
            order = np.argsort(np.abs(sv))[::-1][:top_k]
            return [{"feature": names[i], "contribution_usd": round(float(sv[i]), 2)}
                    for i in order]
        except Exception:
            return []


@lru_cache(maxsize=1)
def get_service() -> PredictionService:
    """Cached singleton so models load only once per process."""
    return PredictionService()


# A realistic mid-range example for demos / API defaults.
EXAMPLE_PHONE = {
    "battery_power": 3000, "blue": 1, "clock_speed": 2.0, "dual_sim": 1,
    "fc": 8, "four_g": 1, "int_memory": 64, "m_dep": 0.6, "mobile_wt": 160,
    "n_cores": 8, "pc": 13, "px_height": 1280, "px_width": 720, "ram": 3000,
    "sc_h": 14, "sc_w": 7, "talk_time": 15, "three_g": 1, "touch_screen": 1,
    "wifi": 1,
}


if __name__ == "__main__":
    svc = get_service()
    from pprint import pprint
    pprint(svc.predict(EXAMPLE_PHONE))
