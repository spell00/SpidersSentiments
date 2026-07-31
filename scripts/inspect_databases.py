import sqlite3

# Check spider_guardian.sqlite
print("=== spider_guardian.sqlite tables ===")
conn = sqlite3.connect('data/spider_guardian.sqlite')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
for row in cursor.fetchall():
    table_name = row[0]
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    print(f"\nTable: {table_name}")
    print("  Columns:", ", ".join([col[1] for col in columns]))
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    print(f"  Row count: {count}")
conn.close()

# Check spider_trending.sqlite
print("\n\n=== spider_trending.sqlite tables ===")
conn = sqlite3.connect('data/spider_trending.sqlite')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
for row in cursor.fetchall():
    table_name = row[0]
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    print(f"\nTable: {table_name}")
    print("  Columns:", ", ".join([col[1] for col in columns]))
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    print(f"  Row count: {count}")
conn.close()
