"""Feature engineering (FYP proposal §5.4).

Creates domain-motivated features that give the models information not directly
present in the raw specs.  The proposal lists five engineered features; three
map cleanly onto the Kaggle schema, and two are *adapted* because the public
Kaggle dataset lacks the columns they need:

    proposal feature      status on Kaggle data
    ────────────────────  ─────────────────────────────────────────────
    pixel_density         ✔ implemented (total px / physical screen area)
    camera_score          ✔ implemented (adapted: no rear-camera count, so
                            combine primary + front megapixels)
    battery_per_gram      ✔ implemented
    is_flagship           ✔ implemented (top-decile RAM *and* storage)
    brand_tier            ✖ needs a `brand` column → available only after the
                            GSMArena scrape; replaced here by `screen_area`
                            until that data is merged.

All functions are pure (return a new DataFrame) so they compose safely inside
scikit-learn pipelines and unit tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-6  # guards against divide-by-zero on rows with a 0 dimension

# Names of the engineered columns, in creation order.
ENGINEERED_COLUMNS = [
    "screen_area",
    "pixel_density",
    "camera_score",
    "battery_per_gram",
    "is_flagship",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with the engineered feature columns appended."""
    df = df.copy()

    # Physical screen area in cm² (sc_h, sc_w are given in cm).
    df["screen_area"] = df["sc_h"] * df["sc_w"]

    # Pixel density: total pixels normalised by physical screen area.
    # High values ⇒ sharp displays, a strong premium-tier signal.
    df["pixel_density"] = (df["px_height"] * df["px_width"]) / (df["screen_area"] + EPS)

    # Composite camera quality. The Kaggle data has no rear-camera *count*,
    # so we combine primary (pc) and front (fc) megapixels; the primary
    # camera dominates real pricing, hence the 0.5 weight on the selfie cam.
    df["camera_score"] = df["pc"] + 0.5 * df["fc"]

    # Energy density: battery capacity per gram of device weight.
    df["battery_per_gram"] = df["battery_power"] / (df["mobile_wt"] + EPS)

    # Flagship flag: in the top decile of BOTH RAM and internal storage.
    ram_p90 = df["ram"].quantile(0.90)
    mem_p90 = df["int_memory"].quantile(0.90)
    df["is_flagship"] = ((df["ram"] >= ram_p90) & (df["int_memory"] >= mem_p90)).astype(int)

    # Replace any inf produced by extreme ratios, then leave NaNs for the
    # imputer downstream (there should be none on the clean Kaggle data).
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def get_feature_columns(df: pd.DataFrame, targets: list[str]) -> list[str]:
    """All model input columns = everything except the target(s)."""
    return [c for c in df.columns if c not in targets]


if __name__ == "__main__":  # quick self-test
    from src.data.make_dataset import build_dataset

    data = build_dataset(save=False)
    enriched = add_engineered_features(data)
    print("New columns:", ENGINEERED_COLUMNS)
    print(enriched[ENGINEERED_COLUMNS].describe().round(3).to_string())
