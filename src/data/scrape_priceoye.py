"""PriceOye.pk live price scraper (rate-limited, robots.txt-aware).

Collects live Pakistani retail prices (PKR) and availability from PriceOye to
provide the *real market price* target for the regression sub-task
(FYP proposal §5.1).  Same ethics safeguards as the GSMArena scraper.

Runs on the authors' own machine (PriceOye is not reachable from the training
sandbox).  Writes ``data/raw/priceoye_prices.csv``.

Usage:
    python -m src.data.scrape_priceoye --category mobiles --max-pages 10
"""
from __future__ import annotations

import argparse
import random
import re
import time
import urllib.robotparser as robotparser
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import get_config

BASE_URL = "https://priceoye.pk/"
HEADERS = {
    "User-Agent": (
        "MobilePricePredictionFYP/1.0 (University of Education, Vehari; "
        "academic research; contact: airesearcher428@gmail.com)"
    )
}
MIN_DELAY, MAX_DELAY = 1.0, 2.0


def _robots_allows(url: str) -> bool:
    rp = robotparser.RobotFileParser()
    rp.set_url(urljoin(BASE_URL, "/robots.txt"))
    try:
        rp.read()
    except Exception:
        return False
    return rp.can_fetch(HEADERS["User-Agent"], url)


def _polite_get(url: str, session: requests.Session) -> requests.Response | None:
    if not _robots_allows(url):
        print(f"[priceoye] robots.txt disallows {url} — skipping")
        return None
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
    resp = session.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return resp


def _parse_price(text: str) -> int | None:
    """Turn 'Rs. 89,999' into the integer 89999."""
    digits = re.sub(r"[^\d]", "", text or "")
    return int(digits) if digits else None


def parse_listing(html: str) -> list[dict]:
    """Extract (model, price_pkr, availability) tuples from a listing page."""
    soup = BeautifulSoup(html, "lxml")
    rows: list[dict] = []
    for card in soup.select("div.productBox, div.product-box"):
        name = card.select_one(".p-title, .product-title")
        price = card.select_one(".price-box, .price")
        avail = card.select_one(".stock, .availability")
        if not name:
            continue
        rows.append({
            "model": name.get_text(strip=True),
            "price_pkr": _parse_price(price.get_text(strip=True) if price else ""),
            "availability": avail.get_text(strip=True) if avail else "unknown",
        })
    return rows


def main(category: str, max_pages: int, usd_pkr: float) -> None:
    cfg = get_config()
    out = cfg.path("processed_dir").parent / "raw" / "priceoye_prices.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    with requests.Session() as session:
        for page in range(1, max_pages + 1):
            url = urljoin(BASE_URL, f"{category}?page={page}")
            print(f"[priceoye] page {page}: {url}")
            resp = _polite_get(url, session)
            if resp is None:
                break
            page_rows = parse_listing(resp.text)
            if not page_rows:
                print("[priceoye] no products found — stopping.")
                break
            all_rows.extend(page_rows)

    if all_rows:
        df = pd.DataFrame(all_rows)
        # convert to USD so it aligns with the regression target unit
        df["price_usd"] = (df["price_pkr"] / usd_pkr).round(2)
        df.to_csv(out, index=False)
        print(f"[priceoye] wrote {len(df)} rows -> {out}")

        # Also drop a JSON copy next to the processed data — this is what the
        # FastAPI backend (src/services/market.py) reads to power the
        # "Pakistani market" comparison panel in the web app.
        market_json = cfg.path("processed_dir") / "priceoye_market.json"
        market_json.parent.mkdir(parents=True, exist_ok=True)
        df.dropna(subset=["price_pkr"]).to_json(market_json, orient="records", indent=2)
        print(f"[priceoye] wrote {market_json}")
    else:
        print("[priceoye] no rows scraped (blocked or offline).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape PriceOye live prices.")
    parser.add_argument("--category", default="mobiles")
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--usd-pkr", type=float, default=278.0,
                        help="PKR per 1 USD, for USD conversion.")
    args = parser.parse_args()
    main(args.category, args.max_pages, args.usd_pkr)
