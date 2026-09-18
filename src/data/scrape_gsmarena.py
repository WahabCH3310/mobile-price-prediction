"""GSMArena specification scraper (rate-limited, robots.txt-aware).

Collects phone specification sheets + announced prices from GSMArena to enrich
the modelling dataset, as described in the FYP proposal §5.1.

Ethics (proposal §13) — enforced here:
  • Honours robots.txt via urllib.robotparser before every fetch.
  • Fixed 1–2 s randomised delay between requests (no hammering).
  • Descriptive User-Agent identifying the academic project.
  • Collects ONLY public product specs/prices — never personal data.

This module is intended to run on the authors' own machine (GSMArena is not
reachable from the training sandbox).  It writes to
``data/raw/gsmarena_specs.csv``.

Usage:
    python -m src.data.scrape_gsmarena --brands samsung xiaomi --max-per-brand 50
"""
from __future__ import annotations

import argparse
import random
import time
import urllib.robotparser as robotparser
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import get_config

BASE_URL = "https://www.gsmarena.com/"
HEADERS = {
    "User-Agent": (
        "MobilePricePredictionFYP/1.0 (University of Education, Vehari; "
        "academic research; contact: airesearcher428@gmail.com)"
    )
}
MIN_DELAY, MAX_DELAY = 1.0, 2.0  # seconds between requests


def _robots_allows(url: str) -> bool:
    rp = robotparser.RobotFileParser()
    rp.set_url(urljoin(BASE_URL, "/robots.txt"))
    try:
        rp.read()
    except Exception:
        # If robots.txt can't be read, be conservative and refuse.
        return False
    return rp.can_fetch(HEADERS["User-Agent"], url)


def _polite_get(url: str, session: requests.Session) -> requests.Response | None:
    """GET a URL only if robots.txt allows it, with a randomised delay."""
    if not _robots_allows(url):
        print(f"[gsmarena] robots.txt disallows {url} — skipping")
        return None
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
    resp = session.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return resp


def parse_spec_page(html: str) -> dict:
    """Extract the fields we care about from a GSMArena device page."""
    soup = BeautifulSoup(html, "lxml")
    specs: dict[str, str] = {}

    name_tag = soup.select_one("h1.specs-phone-name-title")
    specs["model"] = name_tag.get_text(strip=True) if name_tag else None

    # GSMArena renders specs as <table> rows: <td class="ttl">key</td><td class="nfo">val</td>
    for row in soup.select("table tr"):
        key = row.select_one("td.ttl")
        val = row.select_one("td.nfo")
        if key and val:
            specs[key.get_text(strip=True).lower()] = val.get_text(" ", strip=True)
    return specs


def scrape_brand(brand: str, max_devices: int, session: requests.Session) -> list[dict]:
    """Scrape up to ``max_devices`` phones for one brand (skeleton flow)."""
    rows: list[dict] = []
    listing_url = urljoin(BASE_URL, f"{brand}-phones-f-list.php")
    resp = _polite_get(listing_url, session)
    if resp is None:
        return rows

    soup = BeautifulSoup(resp.text, "lxml")
    links = [urljoin(BASE_URL, a["href"])
             for a in soup.select("div.makers a[href]")][:max_devices]

    for link in links:
        try:
            page = _polite_get(link, session)
            if page is None:
                continue
            specs = parse_spec_page(page.text)
            specs["brand"] = brand
            specs["source_url"] = link
            rows.append(specs)
            print(f"[gsmarena] {specs.get('model', link)}")
        except Exception as exc:  # never let one bad page kill the run
            print(f"[gsmarena] failed {link}: {exc}")
    return rows


def main(brands: list[str], max_per_brand: int) -> None:
    cfg = get_config()
    out = cfg.path("processed_dir").parent / "raw" / "gsmarena_specs.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    with requests.Session() as session:
        for brand in brands:
            print(f"\n=== Scraping brand: {brand} ===")
            all_rows.extend(scrape_brand(brand, max_per_brand, session))

    if all_rows:
        pd.DataFrame(all_rows).to_csv(out, index=False)
        print(f"\n[gsmarena] wrote {len(all_rows)} rows -> {out}")
    else:
        print("\n[gsmarena] no rows scraped (blocked or offline).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape GSMArena phone specs.")
    parser.add_argument("--brands", nargs="+",
                        default=["samsung", "xiaomi", "apple", "oppo", "realme"])
    parser.add_argument("--max-per-brand", type=int, default=50)
    args = parser.parse_args()
    main(args.brands, args.max_per_brand)
