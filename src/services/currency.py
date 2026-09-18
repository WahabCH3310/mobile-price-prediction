"""Live USD → PKR exchange-rate helper.

Uses the free, keyless open.er-api.com endpoint. The rate is cached in memory
for ``CACHE_SECONDS`` so we don't hit the API on every single prediction, and
falls back to the last known-good rate (or a hard-coded constant) if the
network call fails — so the app never breaks just because the currency API is
briefly down.
"""
from __future__ import annotations

import time

import requests

RATE_URL = "https://open.er-api.com/v6/latest/USD"
CACHE_SECONDS = 3600  # refresh once an hour
FALLBACK_RATE = 284.0  # approx. USD->PKR as of writing; used only if the API fails

_cache: dict = {"rate": FALLBACK_RATE, "fetched_at": 0.0, "live": False}


def get_usd_to_pkr_rate() -> dict:
    """Return {"rate": float, "live": bool, "fetched_at": epoch_seconds}.

    ``live`` is False when we had to fall back to the cached/default rate
    because the live API could not be reached (e.g. no internet access, rate
    limited, etc.) — the frontend uses this to show an "approx." label.
    """
    now = time.time()
    if now - _cache["fetched_at"] < CACHE_SECONDS and _cache["live"]:
        return dict(_cache)

    try:
        resp = requests.get(RATE_URL, timeout=6)
        resp.raise_for_status()
        data = resp.json()
        rate = float(data["rates"]["PKR"])
        _cache.update(rate=rate, fetched_at=now, live=True)
    except Exception:
        # Keep whatever we had before (could still be the fallback constant);
        # just refresh the timestamp so we don't hammer the API on failure.
        _cache["fetched_at"] = now
        _cache["live"] = False

    return dict(_cache)


if __name__ == "__main__":
    print(get_usd_to_pkr_rate())
