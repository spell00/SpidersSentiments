"""
Database migration script to consolidate Spider Guardian tables.

Migration steps:
1. Create new tables: spider_interactions, spider_content
2. Migrate data from scraped_articles to new tables
3. Drop old unused tables: dataset_entries, sentiment_results, interactions
4. Backup old scraped_articles table before dropping
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_PATH = "data/spider_guardian.sqlite"
BACKUP_PATH = "data/spider_guardian.sqlite.backup"


def backup_database():
    """Create a backup of the database before migration."""
    print(f"Creating backup: {BACKUP_PATH}")
    import shutil
    shutil.copy2(DB_PATH, BACKUP_PATH)
    print("✅ Backup created")


def create_new_tables(conn):
    """Create the new consolidated table structures."""
    cursor = conn.cursor()
    
    print("\nCreating new tables...")
    
    # Table 1: spider_interactions - All spider-related interactions and threads
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS spider_interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id TEXT UNIQUE,
            parent_post_id TEXT,
            thread_root_id TEXT,
            post_type TEXT NOT NULL CHECK(post_type IN (
                'original_bot_post', 
                'bot_reply', 
                'spider_post', 
                'reply_to_bot',
                'flagged_reply'
            )),
            author TEXT,
            author_is_bot INTEGER DEFAULT 0,
            input_tweet_text TEXT,
            output_tweet_text TEXT NOT NULL,
            like_count INTEGER DEFAULT 0,
            reply_count INTEGER DEFAULT 0,
            impression_count INTEGER DEFAULT 0,
            repost_count INTEGER DEFAULT 0,
            posted_at TEXT,
            last_updated_at TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            url TEXT,
            metadata TEXT,
            FOREIGN KEY (parent_post_id) REFERENCES spider_interactions(post_id),
            FOREIGN KEY (thread_root_id) REFERENCES spider_interactions(post_id)
        )
    """)
    
    # Table 2: spider_content - Spider posts we observed but didn't interact with
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS spider_content (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id TEXT UNIQUE,
            text TEXT NOT NULL,
            author TEXT,
            like_count INTEGER DEFAULT 0,
            reply_count INTEGER DEFAULT 0,
            impression_count INTEGER DEFAULT 0,
            repost_count INTEGER DEFAULT 0,
            discovered_at TEXT DEFAULT CURRENT_TIMESTAMP,
            post_created_at TEXT,
            url TEXT,
            lang TEXT,
            metadata TEXT
        )
    """)
    
    # Create indexes for better performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_parent ON spider_interactions(parent_post_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_thread ON spider_interactions(thread_root_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_type ON spider_interactions(post_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_author ON spider_interactions(author)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_author ON spider_content(author)")
    
    conn.commit()
    print("✅ New tables created with indexes")


def migrate_scraped_articles(conn):
    """Migrate data from scraped_articles to new tables."""
    cursor = conn.cursor()
    
    print("\nMigrating scraped_articles...")
    
    # Get all articles
    cursor.execute("SELECT id, title, link, content, metadata, created_at FROM scraped_articles")
    articles = cursor.fetchall()
    
    interactions_migrated = 0
    content_migrated = 0
    skipped = 0
    
    for article in articles:
        article_id, title, link, content_str, metadata_str, created_at = article
        
        try:
            # Parse JSON fields
            content = json.loads(content_str) if isinstance(content_str, str) else (content_str or {})
            metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else (metadata_str or {})
            article_type = metadata.get('type', 'unknown')
            
            if article_type == 'interaction':
                # Bot's reply to someone's post
                tweet_text = content.get('tweet_text', '')
                reply_text = content.get('reply_text', '')
                tweet_id = str(content.get('tweet_id', ''))
                reply_id = str(content.get('reply_id', ''))
                url = content.get('url') or link
                metrics = content.get('metrics', {})
                
                cursor.execute("""
                    INSERT OR IGNORE INTO spider_interactions (
                        post_id, parent_post_id, thread_root_id, post_type, 
                        author, author_is_bot, input_tweet_text, output_tweet_text,
                        like_count, reply_count, impression_count, repost_count,
                        posted_at, created_at, url, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    reply_id or f"reply_{article_id}",  # post_id
                    tweet_id,  # parent_post_id
                    tweet_id,  # thread_root_id (assume direct reply for now)
                    'bot_reply',  # post_type
                    'SpiderGuardianBot',  # author
                    1,  # author_is_bot
                    tweet_text,  # input_tweet_text
                    reply_text,  # output_tweet_text
                    int(metrics.get('likes', 0)),
                    int(metrics.get('replies', 0)),
                    int(metrics.get('impressions', 0)),
                    0,  # repost_count
                    created_at,  # posted_at
                    created_at,  # created_at
                    url,
                    json.dumps({
                        'tone': content.get('tone'),
                        'model': content.get('model'),
                        'migrated_from': 'scraped_articles'
                    })
                ))
                interactions_migrated += 1
                
            elif article_type == 'streamed_post':
                # Spider post we observed but didn't reply to
                text = content.get('text', title or '')
                post_id = str(content.get('id', ''))
                url = content.get('url') or link
                
                cursor.execute("""
                    INSERT OR IGNORE INTO spider_content (
                        post_id, text, author, like_count, reply_count, 
                        impression_count, repost_count, discovered_at, url, lang, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    post_id or f"stream_{article_id}",
                    text,
                    content.get('author_handle', content.get('author')),
                    int(content.get('like_count', 0)),
                    int(content.get('reply_count', 0)),
                    int(content.get('impression_count', 0)),
                    int(content.get('repost_count', 0)),
                    created_at,
                    url,
                    content.get('lang'),
                    json.dumps({'migrated_from': 'scraped_articles'})
                ))
                content_migrated += 1
                
            elif article_type == 'flagged_reply':
                # Flagged problematic reply
                reply_text = content.get('reply_text', title or '')
                url = content.get('url') or link
                
                cursor.execute("""
                    INSERT OR IGNORE INTO spider_interactions (
                        post_id, post_type, author, author_is_bot, output_tweet_text,
                        created_at, url, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    f"flagged_{article_id}",
                    'flagged_reply',
                    'unknown',
                    0,
                    reply_text,
                    created_at,
                    url,
                    json.dumps({
                        'reason': content.get('reason'),
                        'flagged': True,
                        'migrated_from': 'scraped_articles'
                    })
                ))
                interactions_migrated += 1
                
            else:
                skipped += 1
                
        except Exception as e:
            print(f"⚠️  Error migrating article {article_id}: {e}")
            skipped += 1
    
    conn.commit()
    print(f"✅ Migrated {interactions_migrated} interactions, {content_migrated} content items")
    if skipped > 0:
        print(f"⚠️  Skipped {skipped} articles")


def drop_old_tables(conn):
    """Drop old unused tables after successful migration."""
    cursor = conn.cursor()
    
    print("\nDropping old tables...")
    
    # Rename scraped_articles to keep as backup
    cursor.execute("ALTER TABLE scraped_articles RENAME TO scraped_articles_old")
    
    # Drop truly unused tables
    cursor.execute("DROP TABLE IF EXISTS dataset_entries")
    cursor.execute("DROP TABLE IF EXISTS sentiment_results")
    cursor.execute("DROP TABLE IF EXISTS interactions")  # Empty table
    
    conn.commit()
    print("✅ Old tables dropped/renamed")


def verify_migration(conn):
    """Verify the migration was successful."""
    cursor = conn.cursor()
    
    print("\n=== Migration Verification ===")
    
    cursor.execute("SELECT COUNT(*) FROM spider_interactions")
    interactions_count = cursor.fetchone()[0]
    print(f"spider_interactions: {interactions_count} rows")
    
    cursor.execute("SELECT COUNT(*) FROM spider_content")
    content_count = cursor.fetchone()[0]
    print(f"spider_content: {content_count} rows")
    
    cursor.execute("SELECT COUNT(*) FROM scraped_articles_old")
    old_count = cursor.fetchone()[0]
    print(f"scraped_articles_old (backup): {old_count} rows")
    
    # Sample some data
    print("\nSample spider_interactions:")
    cursor.execute("SELECT post_type, author, output_tweet_text FROM spider_interactions LIMIT 3")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]} - {row[2][:50]}...")
    
    print("\nSample spider_content:")
    cursor.execute("SELECT author, text FROM spider_content LIMIT 3")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1][:50]}...")


def main():
    print("=" * 60)
    print("Spider Guardian Database Migration")
    print("=" * 60)
    
    # Backup first
    backup_database()
    
    # Connect and migrate
    conn = sqlite3.connect(DB_PATH)
    
    try:
        create_new_tables(conn)
        migrate_scraped_articles(conn)
        drop_old_tables(conn)
        verify_migration(conn)
        
        print("\n" + "=" * 60)
        print("✅ Migration completed successfully!")
        print("=" * 60)
        print(f"\nBackup saved at: {BACKUP_PATH}")
        print("If everything looks good, you can delete scraped_articles_old table later.")
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        print("Database has been rolled back. Backup is safe.")
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
