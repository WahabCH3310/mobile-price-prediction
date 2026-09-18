"""Exploratory Data Analysis (FYP proposal §5.2).

Generates and saves the core EDA figures:
  1. price_range class balance
  2. price_usd distribution
  3. RAM / internal-storage vs price scatter
  4. correlation heatmap of numeric features
  5. box plots of key specs across price tiers

Run:  python -m src.visualization.eda
Figures are written to ``reports/figures/``.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless backend so it runs on any server
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import get_config
from src.data.make_dataset import build_dataset
from src.features.build_features import add_engineered_features

sns.set_theme(style="whitegrid", palette="viridis")
TIER_NAMES = {0: "Budget", 1: "Mid", 2: "Premium", 3: "Flagship"}


def _save(fig, name: str) -> None:
    cfg = get_config()
    out_dir = cfg.path("figures_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] saved {path}")


def plot_class_balance(df) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["price_range"].map(TIER_NAMES).value_counts()[list(TIER_NAMES.values())]
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax.set(title="Price-tier class balance", xlabel="Price tier", ylabel="Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 3, str(v), ha="center")
    _save(fig, "01_class_balance.png")


def plot_price_distribution(df) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["price_usd"], bins=40, kde=True, ax=ax)
    ax.set(title="Distribution of price_usd (demonstration proxy)",
           xlabel="Price (USD)", ylabel="Count")
    _save(fig, "02_price_distribution.png")


def plot_ram_storage_vs_price(df) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.scatterplot(data=df, x="ram", y="price_usd", hue="price_range",
                    palette="viridis", alpha=0.6, ax=axes[0], legend="full")
    axes[0].set(title="RAM vs price", xlabel="RAM (MB)", ylabel="Price (USD)")
    sns.scatterplot(data=df, x="int_memory", y="price_usd", hue="price_range",
                    palette="viridis", alpha=0.6, ax=axes[1], legend=False)
    axes[1].set(title="Internal storage vs price", xlabel="Storage (GB)", ylabel="")
    _save(fig, "03_ram_storage_vs_price.png")


def plot_correlation_heatmap(df) -> None:
    numeric = df.select_dtypes("number")
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, square=True,
                cbar_kws={"shrink": 0.7}, ax=ax)
    ax.set_title("Correlation heatmap (raw + engineered features)")
    _save(fig, "04_correlation_heatmap.png")


def plot_specs_by_tier(df) -> None:
    specs = ["ram", "battery_power", "int_memory", "camera_score"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, spec in zip(axes.ravel(), specs):
        sns.boxplot(data=df, x="price_range", y=spec, ax=ax, hue="price_range",
                    palette="viridis", legend=False)
        ax.set(title=f"{spec} by price tier", xlabel="Price tier", ylabel=spec)
    fig.suptitle("Key specifications across price tiers", y=1.02, fontsize=13)
    _save(fig, "05_specs_by_tier.png")


def run_all() -> None:
    df = add_engineered_features(build_dataset(save=False))
    print(f"[eda] dataset: {df.shape[0]} rows × {df.shape[1]} cols")
    plot_class_balance(df)
    plot_price_distribution(df)
    plot_ram_storage_vs_price(df)
    plot_correlation_heatmap(df)
    plot_specs_by_tier(df)
    print("[eda] all figures generated.")


if __name__ == "__main__":
    run_all()
