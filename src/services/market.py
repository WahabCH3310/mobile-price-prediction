"""Serves the cached Pakistani-market price data (scraped from PriceOye.pk).

Important: real-time scraping *on every API request* is unreliable, slow and
unkind to the source site, so we never scrape live in the request path.
Instead ``src/data/scrape_priceoye.py`` is run periodically (manually, or via
the scheduled GitHub Action in ``.github/workflows/refresh_market_data.yml``)
and writes ``data/processed/priceoye_market.json``. This module just reads
that cached file, which is normally a few hours old at most — effectively
"real-time" from the end user's point of view.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.config import get_config


def get_market_data(limit: int = 200) -> dict:
    """Return the cached PriceOye listings, plus metadata about freshness."""
    cfg = get_config()
    path = cfg.path("processed_dir") / "priceoye_market.json"

    if not path.exists():
        return {
            "items": [],
            "updated_at": None,
            "note": (
                "No market data yet. Run `python -m src.data.scrape_priceoye` "
                "once (from a machine with internet access to priceoye.pk) to "
                "populate this — or let the scheduled GitHub Action do it."
            ),
        }

    with open(path, "r", encoding="utf-8") as fh:
        items = json.load(fh)

    return {
        "items": items[:limit],
        "updated_at": path.stat().st_mtime,
        "note": None,
    }


def find_similar(ram_mb: int, storage_gb: int, limit: int = 5) -> list[dict]:
    """Very lightweight similarity match, used to show 'phones like this one'.

    The Kaggle-trained model's own price is a demonstration proxy (see
    README), so we never claim the scraped PKR prices *are* the model's
    output — this is a separate, informational comparison against real
    listings for a similarly-specced phone.
    """
    data = get_market_data()
    items = data["items"]
    if not items:
        return []

    def score(item: dict) -> float:
        # crude heuristic: prefer listings whose *name* mentions a RAM/storage
        # figure close to what the user picked, when that info is present.
        name = (item.get("model") or "").lower()
        s = 0.0
        for gb in (storage_gb,):
            if f"{gb}gb" in name.replace(" ", ""):
                s -= 5
        return s

    ranked = sorted(items, key=score)
    return ranked[:limit]
