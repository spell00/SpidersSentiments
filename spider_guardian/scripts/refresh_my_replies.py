"""Refresh engagement metrics for our posted replies."""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from spider_guardian.config import SpiderGuardianConfig
from spider_guardian.twitter_client import SeleniumTwitterClient
from spider_guardian.storage import SQLDataStore
from spider_guardian.langsmith.simple import push_reply_to_dataset
import json

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def refresh_reply_metrics(config: SpiderGuardianConfig, max_age_days: int = 3, limit: int = None) -> int:
    """Scrape updated metrics for our replies and sync to LangSmith."""
    
    sql_store = SQLDataStore(config.sql_database_path)
    client = SeleniumTwitterClient(config)
    
    # Get recent replies we've posted (type=interaction)
    cutoff = datetime.utcnow() - timedelta(days=max_age_days)
    interactions = []
    
    for article in sql_store.iter_scraped_articles():
        if article.metadata.get("type") != "interaction":
            continue
        if article.created_at < cutoff:
            continue
        try:
            record = json.loads(article.content)
            reply_id = record.get("reply_id")
            if reply_id and reply_id != "posted_no_id_found":
                interactions.append((article, record, reply_id))
        except Exception:
            continue
    
    if not interactions:
        logging.info("No recent replies found to refresh (max age: %d days)", max_age_days)
        return 0
    
    logging.info("Found %d replies to refresh metrics for", len(interactions))
    
    refreshed = 0
    for article, record, reply_id in interactions[:limit] if limit else interactions:
        try:
            # Navigate to the reply tweet
            client.driver.get(f"https://x.com/i/status/{reply_id}")
            client.driver.implicitly_wait(3)
            
            # Extract metrics from the page
            from selenium.webdriver.common.by import By
            import time
            time.sleep(2)  # Let metrics load
            
            # Find our reply card and extract metrics
            cards = client.driver.find_elements(By.CSS_SELECTOR, "article[data-testid='tweet']")
            if cards:
                likes, reposts, replies, impressions = client._extract_metrics(cards[0])
                
                logging.info(
                    "Reply %s: likes=%d, replies=%d, reposts=%d, impressions=%d",
                    reply_id, likes, replies, reposts, impressions
                )
                
                # Push updated metrics to LangSmith
                push_reply_to_dataset(
                    tweet_text=record.get("tweet_text", ""),
                    author=record.get("author", "unknown"),
                    url=f"https://x.com/i/status/{record.get('tweet_id', '')}",
                    generated_reply=record.get("reply_text", ""),
                    likes=likes,
                    replies=replies,
                    impressions=impressions,
                    metadata={
                        "source": "reply_refresh",
                        "reply_id": reply_id,
                        "tone": record.get("tone", ""),
                        "refreshed_at": datetime.utcnow().isoformat(),
                    },
                    dataset_name="spider-replies-dataset"
                )
                refreshed += 1
            else:
                logging.warning("Could not find metrics for reply %s", reply_id)
                
        except Exception as exc:
            logging.error("Failed to refresh reply %s: %s", reply_id, exc)
            continue
    
    client.quit()
    logging.info("Refreshed metrics for %d/%d replies", refreshed, len(interactions))
    return refreshed


def main():
    parser = argparse.ArgumentParser(description="Refresh engagement metrics for posted replies")
    parser.add_argument("--max-age-days", type=int, default=3, help="Only refresh replies from last N days")
    parser.add_argument("--limit", type=int, default=None, help="Max replies to refresh per run")
    
    args = parser.parse_args()
    
    config = SpiderGuardianConfig()
    count = refresh_reply_metrics(config, max_age_days=args.max_age_days, limit=args.limit)
    
    print(f"✅ Refreshed {count} replies")


if __name__ == "__main__":
    main()
