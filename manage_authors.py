#!/usr/bin/env python3
"""Manage author follower database and select optimal tweets for high-reach responses.

This script provides utilities for:
1. Viewing top authors by follower count
2. Updating follower counts for known authors
3. Finding best tweets to respond to based on author reach
"""

import argparse
import json
import logging
import sys
import time
import datetime
from typing import List, Optional
from tqdm import tqdm

from spider_guardian.storage import SQLDataStore
from spider_guardian.config import GuardianConfig
from spider_guardian.langsmith.config import langsmith_integration


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def list_top_authors(db_path: str, limit: int = 50, min_followers: Optional[int] = None) -> None:
    """List top authors by follower count."""
    db = SQLDataStore(db_path)
    
    authors = db.get_all_authors(min_followers=min_followers, limit=limit)
    
    if not authors:
        logging.info("No authors found in database.")
        return
    
    print(f"\n{'Handle':<25} {'Followers':>12} {'Tweets':>8} {'Last Updated':<20}")
    print("-" * 70)
    
    for author in authors:
        handle = author["handle"]
        followers = author["follower_count"]
        tweet_count = author["tweet_count"]
        last_updated = author["last_updated"] or "Never"
        
        print(f"{handle:<25} {followers:>12,} {tweet_count:>8} {last_updated:<20}")
    
    print(f"\nTotal: {len(authors)} authors")


def update_author_followers(db_path: str, handles: List[str]) -> None:
    """Update follower counts for specific authors."""
    db = SQLDataStore(db_path)
    
    fetch_start_time = time.time()
    from spider_guardian.config import SpiderGuardianConfig
    config = SpiderGuardianConfig()
    fetch_timeout_seconds = getattr(config, "author_wait_seconds", 60)
    fetched_count = 0
    
    for handle in tqdm(handles, desc="Fetching followers", unit="author"):
        # Check timeout
        elapsed_time = time.time() - fetch_start_time
        if elapsed_time > fetch_timeout_seconds:
            logging.warning(
                f"⏱️ Timeout reached after {elapsed_time/60:.1f} minutes. "
                f"Fetched {fetched_count}/{len(handles)} authors."
            )
            break
        
        logging.info(f"Fetching follower count for @{handle}... ({elapsed_time:.1f}s elapsed)")
        
        try:
            follower_count = langsmith_integration.fetch_author_follower_count(handle)
            
            if follower_count is not None:
                db.upsert_author(handle, follower_count=follower_count)
                logging.info(f"✅ Updated @{handle}: {follower_count:,} followers")
                fetched_count += 1
            else:
                logging.warning(f"❌ Could not fetch follower count for @{handle}")
        except Exception as e:
            logging.error(f"Error updating @{handle}: {e}")
    
    if fetched_count < len(handles):
        logging.info(f"⚠️ Fetched {fetched_count}/{len(handles)} authors before timeout")


def refresh_top_authors(db_path: str, limit: int = 100) -> None:
    """Refresh follower counts for top N authors in database."""
    db = SQLDataStore(db_path)
    
    authors = db.get_top_authors(limit=limit)
    
    if not authors:
        logging.info("No authors found to refresh.")
        return
    
    logging.info(f"Refreshing follower counts for top {len(authors)} authors...")
    
    handles = [a["handle"] for a in authors]
    update_author_followers(db_path, handles)


def find_best_tweets(
    config_path: Optional[str] = None,
    db_path: str = "data/spider_guardian.sqlite",
    min_followers: int = 10000,
    limit: int = 10,
) -> None:
    """Find best tweets to respond to based on author follower count.
    
    This simulates what the bot would do but without actually replying.
    Useful for testing the selection logic.
    """
    from spider_guardian.twitter_client import TwitterClient
    
    # Load config
    if config_path:
        config = GuardianConfig.from_yaml(config_path)
    else:
        config = GuardianConfig()
    
    # Initialize database and Twitter client
    db = SQLDataStore(db_path)
    twitter = TwitterClient(config)
    
    logging.info(f"Searching for tweets matching: {config.twitter_query}")
    logging.info(f"Minimum follower threshold: {min_followers:,}")
    
    # Search tweets
    tweets = twitter.search_posts(config.twitter_query)
    logging.info(f"Found {len(tweets)} candidate tweets")
    # Track fetching time
    fetch_start_time = time.time()
    from spider_guardian.config import SpiderGuardianConfig
    config_obj = SpiderGuardianConfig()
    fetch_timeout_seconds = getattr(config_obj, "author_wait_seconds", 60)
    fetched_count = 0
    
    
    # Enrich with follower counts
    enriched_tweets = []
    for tweet in tqdm(tweets, desc="Enriching tweets", unit="tweet"):
        # Check timeout
        elapsed_time = time.time() - fetch_start_time
        if elapsed_time > fetch_timeout_seconds:
            logging.warning(
                f"⏱️ Timeout reached after {elapsed_time/60:.1f} minutes. "
                f"Processed {fetched_count}/{len(tweets)} tweets."
            )
            break
        
        if not tweet.author_handle:
            continue
        
        # Get cached follower count with metadata
        info = db.get_author_followers_info(tweet.author_handle)
        cached_count = info.get("follower_count")
        checked_iso = info.get("followers_checked_at")
        needs_refresh = False
        if checked_iso:
            try:
                checked_dt = datetime.datetime.fromisoformat(checked_iso)
                needs_refresh = (datetime.datetime.utcnow() - checked_dt).days >= 30
            except Exception:
                needs_refresh = True
        else:
            needs_refresh = True if cached_count in (None, -1) else False

        if cached_count is not None and cached_count != -1 and not needs_refresh:
            follower_count = cached_count
        else:
            # Fetch if not cached or stale
            follower_count = langsmith_integration.fetch_author_follower_count(tweet.author_handle)
            if follower_count is not None:
                db.upsert_author(tweet.author_handle, follower_count=follower_count)
        
        fetched_count += 1
        
        if follower_count and follower_count >= min_followers:
            enriched_tweets.append({
                "tweet_id": tweet.id,
                "author_handle": tweet.author_handle,
                "follower_count": follower_count,
                "text": (tweet.text or "")[:100],
                "url": tweet.url,
            })
    
    # Sort by follower count
    enriched_tweets.sort(key=lambda t: t["follower_count"], reverse=True)
    
    print(f"\n{'Author':<20} {'Followers':>12} {'Tweet Preview':<50}")
    print("-" * 85)
    
    for tweet in enriched_tweets[:limit]:
        print(f"@{tweet['author_handle']:<19} {tweet['follower_count']:>12,} {tweet['text'][:50]}")
        print(f"  → {tweet['url']}")
        print()
    
    print(f"Found {len(enriched_tweets)} tweets from authors with {min_followers:,}+ followers")


def main():
    parser = argparse.ArgumentParser(description="Manage author follower database")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List top authors")
    list_parser.add_argument("--limit", type=int, default=50, help="Number of authors to show")
    list_parser.add_argument("--min-followers", type=int, help="Minimum follower count filter")
    list_parser.add_argument("--db-path", default="data/spider_guardian.sqlite", help="Database path")
    
    # Update command
    update_parser = subparsers.add_parser("update", help="Update follower counts for specific authors")
    update_parser.add_argument("handles", nargs="+", help="Author handles to update (without @)")
    update_parser.add_argument("--db-path", default="data/spider_guardian.sqlite", help="Database path")
    
    # Refresh command
    refresh_parser = subparsers.add_parser("refresh", help="Refresh follower counts for top authors")
    refresh_parser.add_argument("--limit", type=int, default=100, help="Number of top authors to refresh")
    refresh_parser.add_argument("--db-path", default="data/spider_guardian.sqlite", help="Database path")
    
    # Find command
    find_parser = subparsers.add_parser("find", help="Find best tweets to respond to")
    find_parser.add_argument("--config", help="Path to config YAML file")
    find_parser.add_argument("--db-path", default="data/spider_guardian.sqlite", help="Database path")
    find_parser.add_argument("--min-followers", type=int, default=10000, help="Minimum follower threshold")
    find_parser.add_argument("--limit", type=int, default=10, help="Number of tweets to show")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == "list":
            list_top_authors(args.db_path, args.limit, args.min_followers)
        elif args.command == "update":
            update_author_followers(args.db_path, args.handles)
        elif args.command == "refresh":
            refresh_top_authors(args.db_path, args.limit)
        elif args.command == "find":
            find_best_tweets(args.config, args.db_path, args.min_followers, args.limit)
    except KeyboardInterrupt:
        logging.info("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
