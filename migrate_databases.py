"""
Database Migration Script for Spider Guardian Project
Consolidates and renames databases to follow a consistent naming scheme.

New structure:
- data/spider_news.sqlite          # Scraped articles, news data
- data/spider_sentiments.sqlite    # Sentiment analysis results
- data/spider_guardian.sqlite      # Bot interactions, replies, feedback
- data/spider_trending.sqlite      # Trending posts, engagement metrics
"""

import os
import shutil
import sqlite3
from pathlib import Path


def backup_database(db_path: str) -> str:
    """Create a backup of the database before migration."""
    if not os.path.exists(db_path):
        return None
    backup_path = f"{db_path}.backup"
    shutil.copy2(db_path, backup_path)
    print(f"✓ Backed up {db_path} to {backup_path}")
    return backup_path


def migrate_trending_to_spider_trending():
    """Migrate trending.sqlite to spider_trending.sqlite"""
    old_path = "data/trending.sqlite"
    new_path = "data/spider_trending.sqlite"
    
    if os.path.exists(old_path):
        backup_database(old_path)
        shutil.copy2(old_path, new_path)
        print(f"✓ Migrated {old_path} to {new_path}")
    else:
        print(f"⚠ {old_path} not found, creating new {new_path}")
        # Create new database with trending_posts table
        conn = sqlite3.connect(new_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trending_posts (
                post_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                author TEXT,
                like_count INTEGER DEFAULT 0,
                reply_count INTEGER DEFAULT 0,
                impression_count INTEGER DEFAULT 0,
                repost_count INTEGER DEFAULT 0,
                collected_at TIMESTAMP,
                post_created_at TIMESTAMP,
                url TEXT,
                last_update TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
        print(f"✓ Created new {new_path} with trending_posts table")


def ensure_spider_guardian_db():
    """Ensure spider_guardian.sqlite exists with proper schema"""
    db_path = "data/spider_guardian.sqlite"
    
    if os.path.exists(db_path):
        backup_database(db_path)
        print(f"✓ {db_path} exists, backed up")
    else:
        print(f"Creating new {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create interactions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id TEXT UNIQUE,
            original_text TEXT,
            reply_text TEXT,
            author TEXT,
            like_count INTEGER DEFAULT 0,
            reply_count INTEGER DEFAULT 0,
            impression_count INTEGER DEFAULT 0,
            repost_count INTEGER DEFAULT 0,
            posted_at TIMESTAMP,
            url TEXT,
            last_update TIMESTAMP,
            metadata TEXT
        )
    """)
    
    # Create feedback table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            interaction_id INTEGER,
            feedback_score REAL,
            feedback_comment TEXT,
            created_at TIMESTAMP,
            FOREIGN KEY (interaction_id) REFERENCES interactions(id)
        )
    """)
    
    conn.commit()
    conn.close()
    print(f"✓ {db_path} schema ready")


def migrate_legacy_csv_to_spider_news():
    """Migrate CSV data to spider_news.sqlite"""
    db_path = "data/spider_news.sqlite"
    csv_path = "data/Data_spider_news_global.csv"
    
    if os.path.exists(db_path):
        backup_database(db_path)
    
    if not os.path.exists(csv_path):
        print(f"⚠ {csv_path} not found, creating empty {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create scraped_articles table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scraped_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            link TEXT UNIQUE,
            title TEXT,
            content TEXT,
            created_at TIMESTAMP,
            source TEXT,
            metadata TEXT
        )
    """)
    
    conn.commit()
    conn.close()
    print(f"✓ {db_path} schema ready")


def migrate_sentiments():
    """Create spider_sentiments.sqlite for sentiment analysis results"""
    db_path = "data/spider_sentiments.sqlite"
    
    if os.path.exists(db_path):
        backup_database(db_path)
    else:
        print(f"Creating new {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create sentiment_results table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id INTEGER,
            text TEXT,
            sentiment TEXT,
            score REAL,
            classifier TEXT,
            preprocess_level INTEGER,
            analyzed_at TIMESTAMP,
            FOREIGN KEY (article_id) REFERENCES scraped_articles(id)
        )
    """)
    
    conn.commit()
    conn.close()
    print(f"✓ {db_path} schema ready")


def main():
    """Run all migrations"""
    print("=" * 60)
    print("Spider Guardian Database Migration")
    print("=" * 60)
    
    # Ensure data directory exists
    Path("data").mkdir(exist_ok=True)
    
    print("\n1. Migrating trending data...")
    migrate_trending_to_spider_trending()
    
    print("\n2. Setting up Spider Guardian database...")
    ensure_spider_guardian_db()
    
    print("\n3. Setting up Spider News database...")
    migrate_legacy_csv_to_spider_news()
    
    print("\n4. Setting up Spider Sentiments database...")
    migrate_sentiments()
    
    print("\n" + "=" * 60)
    print("Migration complete!")
    print("=" * 60)
    print("\nNew database structure:")
    print("  data/spider_news.sqlite       - Scraped articles, news data")
    print("  data/spider_sentiments.sqlite - Sentiment analysis results")
    print("  data/spider_guardian.sqlite   - Bot interactions, replies")
    print("  data/spider_trending.sqlite   - Trending posts, metrics")
    print("\nBackup files created with .backup extension")


if __name__ == "__main__":
    main()
