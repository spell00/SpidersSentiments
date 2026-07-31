# High-Reach Account Targeting Strategy

This document describes the system for maximizing reply reach by targeting tweets from high-follower accounts.

## Overview

The bot now tracks author follower counts in a local database and uses this information to:
1. **Prioritize replies** to accounts with the most followers
2. **Cache follower counts** to avoid repeated API calls
3. **Filter tweets** based on author reach thresholds
4. **Monitor author activity** (tweet counts, last seen)

## Database Schema

### Authors Table
```sql
CREATE TABLE authors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    handle TEXT NOT NULL UNIQUE,
    follower_count INTEGER,
    last_updated TEXT,
    first_seen TEXT NOT NULL,
    tweet_count INTEGER DEFAULT 0,
    metadata TEXT
)
```

**Key fields:**
- `handle`: Twitter handle (without @)
- `follower_count`: Cached follower count
- `last_updated`: When follower count was last refreshed
- `first_seen`: When this author was first encountered
- `tweet_count`: Number of times we've seen tweets from this author

## Usage

### 1. Bot with Follower Filtering

Run the bot with minimum follower threshold:

```bash
python -m spider_guardian --respond 1 --min-followers 10000 --selenium-driver firefox
```

This will:
- Only reply to tweets from accounts with 10,000+ followers
- Use cached follower counts from database (fast)
- Fetch and cache follower counts for new authors
- Sort tweets by follower count (highest first)
- Update author `tweet_count` as it encounters tweets

**Additional options:**
```bash
--max-followers 500000    # Avoid mega-celebrities (optional)
--min-impressions 1000    # Only reply to tweets with 1K+ views
```

### 2. Managing the Author Database

Use `manage_authors.py` to work with the author database:

#### List Top Authors
```bash
python manage_authors.py list --limit 50
python manage_authors.py list --min-followers 50000 --limit 20
```

Output:
```
Handle                    Followers   Tweets Last Updated        
----------------------------------------------------------------------
NatGeo                   245,123,456      12 2025-11-12T10:30:00
BBCEarth                 156,789,012       8 2025-11-12T09:15:00
...
```

#### Update Specific Authors
```bash
python manage_authors.py update NatGeo BBCEarth NewScientist
```

This fetches current follower counts from Twitter and updates the database.

#### Refresh Top Authors
```bash
python manage_authors.py refresh --limit 100
```

Refreshes follower counts for your top 100 authors (by current follower count).

#### Find Best Tweets
```bash
python manage_authors.py find --min-followers 50000 --limit 10
```

Searches for tweets matching your query and shows which ones are from high-follower accounts (without actually replying).

### 3. Integration with LangSmith

The bot automatically updates LangSmith dataset metadata when filtering is enabled:

```python
# Metadata includes:
{
    "author_followers": 125000,
    "author_followers_fetched_at": "2025-11-12T10:30:00",
    "input_metrics": {...},
    "reply_metrics": {...}
}
```

## How It Works

### First Run (No Database Cache)
1. Bot searches for tweets matching query
2. For each author encountered, checks database for cached follower count
3. If not cached, fetches from Twitter (slow, ~2-3 seconds per author)
4. Caches follower count in database
5. Filters and sorts tweets by follower count
6. Replies to highest-reach tweets first

### Subsequent Runs (With Database Cache)
1. Bot searches for tweets matching query
2. For each author, gets cached follower count from database (instant)
3. Filters and sorts tweets by follower count
4. Replies to highest-reach tweets first

### Handling No Qualifying Tweets

If the bot finds **no tweets meeting your filters** in a search batch, it will:

1. **Log a warning** with the current attempt count
2. **Wait progressively longer** before retrying:
   - 1st empty search: Wait 60 seconds
   - 2nd empty search: Wait 120 seconds  
   - 3rd empty search: Wait 180 seconds
   - etc. (60 × attempt number)
3. **Exit after N consecutive empty searches** (default: 5, configurable via `--max-empty-searches`)

This prevents:
- ❌ Infinite loops hammering the Twitter API
- ❌ Rate limiting issues
- ❌ Wasted API calls fetching the same tweets repeatedly

**Adjusting the retry limit:**
```bash
# More patient (10 attempts, up to 10 minutes wait between attempts)
python -m spider_guardian --respond 10 --min-followers 50000 --max-empty-searches 10

# Less patient (3 attempts, up to 3 minutes wait)
python -m spider_guardian --respond 10 --min-followers 50000 --max-empty-searches 3

# Very patient for rare high-reach opportunities (20 attempts)
python -m spider_guardian --respond 10 --min-followers 100000 --max-empty-searches 20
```

**Example scenario with default settings (5 attempts):**
```bash
# You run with very strict filters
python -m spider_guardian --respond 10 --min-followers 100000

# Output if no 100K+ follower accounts are tweeting about spiders:
[INFO] 🎯 Filtering enabled - min_followers=100000
[INFO] 🔄 Will retry up to 5 times if no qualifying tweets found
[INFO] Fetched 25 candidate tweet(s)
[INFO] Sorted 25 tweets by follower count (highest first)
[WARNING] No qualifying tweets found in this batch (attempt 1/5)
[INFO] Waiting 60 seconds before next search...
[INFO] Fetched 28 candidate tweet(s)
[WARNING] No qualifying tweets found in this batch (attempt 2/5)
[INFO] Waiting 120 seconds before next search...
[INFO] Fetched 23 candidate tweet(s)
[WARNING] No qualifying tweets found in this batch (attempt 3/5)
[INFO] Waiting 180 seconds before next search...
[INFO] Fetched 27 candidate tweet(s)
[WARNING] No qualifying tweets found in this batch (attempt 4/5)
[INFO] Waiting 240 seconds before next search...
[INFO] Fetched 19 candidate tweet(s)
[WARNING] No qualifying tweets found in this batch (attempt 5/5)
[ERROR] ❌ No qualifying tweets found after 5 attempts. Your filters might be too strict:
        min_followers=100000, max_followers=None, min_impressions=None
        Consider lowering thresholds or checking if high-reach accounts are tweeting about spiders.
```

**Recommended approach:**
- Start with moderate filters (e.g., `--min-followers 5000`) to build your database
- Gradually increase thresholds as you identify high-reach accounts
- Use `manage_authors.py find --min-followers X` to preview available tweets before running the bot

### Database Maintenance
- Follower counts can become stale over time
- Use `manage_authors.py refresh` periodically to update top authors
- The bot increments `tweet_count` each time it sees an author (useful for identifying frequent tweeters)

## Strategy for High-Reach Targeting

### Phase 1: Build Author Database (Current)
- Run bot with relaxed filters to populate database
- Cache follower counts for all encountered authors
- Build understanding of which accounts tweet about spiders

### Phase 2: Targeted Responding
- Run bot with `--min-followers 10000` or higher
- Focus replies on high-reach accounts
- Maximize impressions per reply

### Phase 3: Top Performer Tracking (Future)
Later we'll implement logic to:
- Identify "top performers" (high-reach accounts that tweet frequently)
- Track their recent tweets (even without "spider" keyword)
- Respond strategically to their content
- Relax keyword filters for proven high-reach accounts

## Examples

### Conservative Approach (Build Database)
```bash
python -m spider_guardian --respond 5 --selenium-driver firefox
```
Replies to 5 tweets, caching follower counts for all encountered authors.

### Aggressive High-Reach Strategy
```bash
python -m spider_guardian --respond 10 --min-followers 50000 --selenium-driver firefox
```
Only replies to tweets from accounts with 50K+ followers. Maximizes reach.

### Balanced Approach
```bash
python -m spider_guardian --respond 5 --min-followers 10000 --max-followers 500000
```
Targets mid-to-high reach accounts (10K-500K followers). Avoids mega-celebrities who might not see replies.

## Performance Notes

- **Database lookups**: Instant (SQLite indexed queries)
- **Twitter follower fetch**: ~2-3 seconds per author (Selenium + parsing)
- **Cache hit rate**: After initial population, ~90%+ (most authors seen multiple times)
- **Sorting overhead**: Negligible (in-memory sort of 10-50 tweets)

## Database Location

Default: `data/spider_guardian.sqlite`

The authors table shares the same database as interactions, content, etc.

## Monitoring

Check database stats:
```bash
python -c "from spider_guardian.storage import SQLDataStore; db = SQLDataStore(); print(f'Authors: {len(db.get_all_authors())}')"
```

View top authors:
```bash
python manage_authors.py list --limit 20
```
