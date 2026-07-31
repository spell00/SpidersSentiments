import sqlite3

# Verify all databases
databases = [
    "data/spider_trending.sqlite",
    "data/spider_guardian.sqlite",
    "data/spider_news.sqlite",
    "data/spider_sentiments.sqlite"
]

print("=" * 60)
print("Database Verification")
print("=" * 60)

for db_path in databases:
    print(f"\n{db_path}")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"  Tables: {', '.join(tables)}")
        
        # Get row counts
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"    - {table}: {count} rows")
        
        conn.close()
        print("  ✓ OK")
    except Exception as e:
        print(f"  ✗ Error: {e}")

print("\n" + "=" * 60)
print("Verification complete!")
print("=" * 60)
