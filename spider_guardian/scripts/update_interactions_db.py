"""Update engagement metrics in the interactions table by scraping X (Twitter) with Selenium.

This script reads recent interactions from the normalized SQL table and refreshes
like/reply/repost/impression counts directly in the database.

It uses SeleniumTwitterClient for reliability (login/cookie injection and ETP disable in Firefox).
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from spider_guardian.config import SpiderGuardianConfig
from spider_guardian.twitter_client import SeleniumTwitterClient
from spider_guardian.storage.sql import SQLDataStore


def _safe_status_url_from_row(row: Dict[str, object]) -> Optional[str]:
    reply_id = str(row.get("reply_id") or "").strip()
    tweet_id = str(row.get("tweet_id") or "").strip()
    if reply_id and reply_id != "posted_no_id_found":
        return f"https://x.com/i/status/{reply_id}"
    if tweet_id:
        return f"https://x.com/i/status/{tweet_id}"
    # Try to infer from url column
    url = str(row.get("url") or "").strip()
    if url and "/status/" in url:
        return url
    return None


def refresh_interactions_in_db(
    db_path: str = "data/spider_guardian.sqlite",
    max_age_days: int = 3,
    limit: Optional[int] = None,
    show_browser: bool = False,
    driver: str = "firefox",
    wait_after_load_seconds: int = 1,
    interactive: bool = False,
) -> int:
    """Scrape and update engagement metrics directly in the interactions table.

    - Selects interactions created within the last `max_age_days` days.
    - Navigates to each reply/tweet page and extracts counts.
    - Updates like_count, reply_count, impression_count, repost_count, and updated_at.
    Returns number of DB rows updated.
    """
    # Configure logging
    logging.info("[interactions-db] Starting refresh: db=%s, max_age_days=%d, limit=%s, driver=%s%s",
                 db_path, max_age_days, str(limit or "none"), driver,
                 " (visible)" if show_browser else " (headless)")

    # Prepare config/clients
    cfg = SpiderGuardianConfig()
    cfg.sql_database_path = db_path
    cfg.selenium_headless = not show_browser
    cfg.selenium_driver = driver

    store = SQLDataStore(db_path)
    client = SeleniumTwitterClient(cfg)

    cutoff = datetime.utcnow() - timedelta(days=max_age_days)

    # Collect candidate rows
    rows: List[Dict[str, object]] = []
    for row in store.iter_interactions():
        try:
            created_at = row.get("created_at")
            dt = datetime.fromisoformat(created_at) if isinstance(created_at, str) else None
        except Exception:
            dt = None
        if dt is None or dt < cutoff:
            continue
        url = _safe_status_url_from_row(row)
        if not url:
            continue
        rows.append({**row, "_status_url": url})
        if limit and len(rows) >= limit:
            break

    if not rows:
        logging.info("[interactions-db] No recent interactions to refresh")
        client.close()
        return 0

    logging.info("[interactions-db] Will refresh %d interaction(s)", len(rows))

    # Proactively purge any stale rows that have the sentinel reply_id (no real reply created)
    try:
        purged = store.delete_interactions_with_reply_id("posted_no_id_found")
        if purged:
            logging.info("[interactions-db] Purged %d interaction(s) with sentinel reply_id", purged)
    except Exception as exc:
        logging.debug("[interactions-db] Purge of sentinel reply_id failed (non-fatal): %s", exc)

    # Helper for naive rate-limit detection/backoff
    wait_time = 60
    updates: List[Dict[str, object]] = []

    for idx, row in enumerate(rows, start=1):
        url = str(row["_status_url"])  # type: ignore[index]
        logging.info("[%d/%d] Fetching %s", idx, len(rows), url)
        try:
            client.driver.get(url)
            # wait for page to load and cards to appear
            try:
                client._wait_primary_column()
            except Exception:
                pass
            # Small pause after load to allow dynamic content and redirects to stabilise
            time.sleep(wait_after_load_seconds)

            # If interactive inspection requested, pause here so user can view the browser
            if interactive:
                try:
                    input(f"[inspect] Opened {url} in browser. Press Enter to continue to next item...")
                except Exception:
                    # In non-interactive environments input() may fail; continue silently
                    pass

            cards = []
            try:
                cards = client._wait_for_cards(min_cards=1)
            except Exception:
                pass
            if not cards:
                src = (client.driver.page_source or "").lower()
                if "something went wrong" in src or "rate limit" in src:
                    logging.warning("Rate-limit page detected. Sleeping %ds", wait_time)
                    time.sleep(wait_time)
                    wait_time = min(wait_time * 2, 300)
                    continue
                logging.warning("No tweet card found for %s", url)
                continue

            reply_c, repost_c, like_c, impression_c = client._extract_metrics(cards[0])
            
            # Use the proven regex method to extract Views (same as trending uploads)
            views_count = client.extract_views_from_page()
            if views_count > 0:
                impression_c = views_count
                
            logging.info("    metrics: likes=%d replies=%d reposts=%d impressions=%d",
                         like_c, reply_c, repost_c, impression_c)

            # Build update descriptor; prefer reply_id -> tweet_id -> url
            upd: Dict[str, object] = {
                "like_count": like_c,
                "reply_count": reply_c,
                "repost_count": repost_c,
                "impression_count": impression_c,
            }
            rid = str(row.get("reply_id") or "").strip()
            tid = str(row.get("tweet_id") or "").strip()
            if rid and rid != "posted_no_id_found":
                upd["reply_id"] = rid
            elif tid:
                upd["tweet_id"] = tid
            else:
                upd["url"] = url

            updates.append(upd)
        except Exception as exc:
            logging.warning("Failed to refresh %s: %s", url, exc)
            continue

    client.close()

    if not updates:
        logging.info("[interactions-db] Nothing to update in DB")
        return 0

    written = store.bulk_update_interactions_metrics(updates)
    logging.info("[interactions-db] Updated %d row(s) in interactions table", written)
    return written


def main():
    parser = argparse.ArgumentParser(description="Refresh interactions metrics directly in SQL DB")
    parser.add_argument("--db", dest="db_path", default="data/spider_guardian.sqlite", help="Path to SQLite DB")
    parser.add_argument("--max-age-days", type=int, default=3, help="Only refresh rows newer than this many days")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to refresh")
    parser.add_argument("--show-browser", action="store_true", help="Run Selenium with visible browser (Firefox)")
    parser.add_argument("--driver", type=str, default="firefox", choices=["firefox", "chrome"], help="Selenium driver")
    parser.add_argument("--wait-after-load-seconds", type=int, default=1, help="Seconds to wait after page load to allow JS to stabilise")
    parser.add_argument("--inspect", action="store_true", help="Pause after each loaded page and wait for Enter (use with --show-browser)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    updated = refresh_interactions_in_db(
        db_path=args.db_path,
        max_age_days=args.max_age_days,
        limit=args.limit,
        show_browser=args.show_browser,
        driver=args.driver,
        wait_after_load_seconds=args.wait_after_load_seconds,
        interactive=args.inspect,
    )
    print(f"[interactions-db] Updated {updated} rows")


if __name__ == "__main__":
    main()
