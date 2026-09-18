"""Dataset assembly.

Reads the raw Kaggle *Mobile Price Classification* CSV, validates its schema,
attaches a continuous ``price_usd`` column for the regression sub-task, and
writes a clean dataset to ``data/processed/dataset.csv``.

────────────────────────────────────────────────────────────────────
IMPORTANT — about ``price_usd``
────────────────────────────────────────────────────────────────────
The public Kaggle dataset ships only the *ordinal* target ``price_range``
(0=budget … 3=flagship); it has no exact currency price.  The FYP proposal's
regression sub-task needs a continuous target, which in production comes from
the scraped GSMArena / PriceOye market prices (see ``scrape_gsmarena.py`` and
``scrape_priceoye.py``).

Until those scraped prices are merged in, this module synthesises a
**demonstration proxy price** so the regression code path is runnable
end-to-end.  The proxy is a specs-driven, noisy function of the real hardware
columns — it is NOT a real market price and any R²/MAPE reported against it
measures pipeline correctness, not real-world pricing accuracy.  Replace it
with ``merge_scraped_prices()`` output before quoting regression numbers in the
final report.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_config

# The 20 raw feature columns + target expected from the Kaggle dataset.
EXPECTED_COLUMNS = [
    "battery_power", "blue", "clock_speed", "dual_sim", "fc", "four_g",
    "int_memory", "m_dep", "mobile_wt", "n_cores", "pc", "px_height",
    "px_width", "ram", "sc_h", "sc_w", "talk_time", "three_g",
    "touch_screen", "wifi", "price_range",
]


def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    """Load and schema-check the raw Kaggle CSV."""
    cfg = get_config()
    path = Path(path) if path else cfg.path("raw_data")
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}.\n"
            "Download the Kaggle 'Mobile Price Classification' dataset "
            "(train.csv) and place it there, or run:\n"
            "    python -m src.data.make_dataset --download"
        )
    df = pd.read_csv(path)
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Raw data is missing expected columns: {sorted(missing)}")
    return df[EXPECTED_COLUMNS].copy()


def add_proxy_price(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Attach a *demonstration* continuous ``price_usd`` derived from specs.

    See the module docstring: this is a stand-in for real scraped prices.
    The price is a linear-in-specs signal plus multiplicative log-normal noise,
    so the regression models must learn from hardware features rather than
    trivially inverting ``price_range``.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    signal = (
        60.0                                   # base cost of any device
        + 0.115 * df["ram"]                    # RAM (MB) — dominant driver
        + 2.60 * df["int_memory"]              # internal storage (GB)
        + 0.030 * df["battery_power"]          # battery (mAh)
        + 4.20 * (df["pc"] + df["fc"])         # combined camera megapixels
        + 6.00 * df["n_cores"] * df["clock_speed"]  # compute (cores × GHz)
        + 0.010 * (df["px_height"] * df["px_width"]) / 1000.0  # resolution
        + 35.0 * df["four_g"]                  # 4G connectivity premium
        + 15.0 * df["wifi"]
        + 10.0 * df["dual_sim"]
    )
    noise = rng.lognormal(mean=0.0, sigma=0.06, size=len(df))
    price = signal * noise
    df["price_usd"] = np.clip(price, 60.0, 1600.0).round(2)
    return df


def merge_scraped_prices(
    df: pd.DataFrame,
    gsmarena_csv: Path | str | None = None,
    priceoye_csv: Path | str | None = None,
) -> pd.DataFrame:
    """Placeholder hook for merging real scraped market prices.

    When ``data/raw/gsmarena_*.csv`` / ``data/raw/priceoye_*.csv`` are present,
    join them here (e.g. on a normalised model key) to overwrite the proxy
    ``price_usd`` with real currency values.  Left as a documented stub because
    scraping runs outside this sandbox.
    """
    raise NotImplementedError(
        "Merge real scraped prices here once GSMArena/PriceOye CSVs exist. "
        "Until then add_proxy_price() supplies a demonstration target."
    )


def build_dataset(save: bool = True) -> pd.DataFrame:
    """Full assembly: load raw → add proxy price → (optionally) save."""
    cfg = get_config()
    df = load_raw()
    df = add_proxy_price(df, seed=cfg.project.random_seed)

    if save:
        out = cfg.path("processed_dir") / "dataset.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"[make_dataset] wrote {len(df):,} rows × {df.shape[1]} cols -> {out}")
    return df


def _download_kaggle_seed() -> None:
    """Convenience downloader for the public Kaggle seed CSV (best effort)."""
    import urllib.request

    cfg = get_config()
    dest = cfg.path("raw_data")
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://raw.githubusercontent.com/amankharwal/"
        "Website-data/master/mobile_prices.csv"
    )
    print(f"[make_dataset] downloading Kaggle seed -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print("[make_dataset] done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble the modelling dataset.")
    parser.add_argument(
        "--download", action="store_true",
        help="Fetch the public Kaggle seed CSV before building.",
    )
    args = parser.parse_args()
    if args.download:
        _download_kaggle_seed()
    build_dataset(save=True)
