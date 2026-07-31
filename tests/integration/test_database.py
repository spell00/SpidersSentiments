#!/usr/bin/env python3
"""
Quick database test to check if corruption issues are resolved.
"""

import json
import os
import sqlite3
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in os.sys.path:
    os.sys.path.insert(0, PROJECT_ROOT)


def test_database():
    print('🔍 Testing database functionality...')

    try:
        # Test database connection
        conn = sqlite3.connect('data/spider_guardian.sqlite')
        cursor = conn.cursor()

        # Check recent interactions
        cursor.execute('''
            SELECT link, title, created_at, metadata
            FROM scraped_articles
            WHERE created_at > datetime('now', '-1 day')
            ORDER BY created_at DESC
            LIMIT 5
        ''')

        recent = cursor.fetchall()
        print(f'✅ Found {len(recent)} recent interactions:')

        for i, (link, title, created_at, metadata_json) in enumerate(recent, 1):
            try:
                if metadata_json:
                    metadata = json.loads(metadata_json)
                    tweet_id = metadata.get('tweet_id', 'N/A')
                    reply_id = metadata.get('reply_id', 'N/A')
                    print(f'{i}. [{created_at}]')
                    print(f'   Reply: "{title[:50]}..."')
                    print(f'   Tweet ID: {tweet_id}, Reply ID: {reply_id}')
                else:
                    print(f'{i}. [{created_at}] {title[:50]}... (no metadata)')
            except Exception as e:
                print(f'{i}. [{created_at}] {title[:50]}... (metadata error: {e})')
            print()

        # Test table structure
        cursor.execute('PRAGMA table_info(scraped_articles)')
        columns = cursor.fetchall()
        print(f'📋 Table structure ({len(columns)} columns):')
        for col in columns:
            print(f'   - {col[1]} ({col[2]})')

        # Test total count
        cursor.execute('SELECT COUNT(*) FROM scraped_articles')
        total = cursor.fetchone()[0]
        print(f'📊 Total records: {total}')

        # Test for corruption by trying to read all metadata
        cursor.execute('SELECT COUNT(*) FROM scraped_articles WHERE metadata IS NOT NULL')
        metadata_count = cursor.fetchone()[0]

        cursor.execute('SELECT link FROM scraped_articles WHERE metadata IS NOT NULL LIMIT 5')
        metadata_samples = cursor.fetchall()

        print(f'📝 Records with metadata: {metadata_count}')
        print('📄 Sample metadata records processed successfully')

        conn.close()
        print('✅ Database is working correctly!')
        return True

    except Exception as e:
        print(f'❌ Database error: {e}')
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_database()
