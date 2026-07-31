# Database and Dataset Configuration

## Database Structure

The Spider Guardian project uses four separate SQLite databases for different purposes:

### 1. `data/spider_news.sqlite`
**Purpose:** Stores scraped news articles about spiders from various sources.

**Tables:**
- `scraped_articles`: News articles with metadata
  - `id`: Primary key
  - `link`: Unique article URL
  - `title`: Article title
  - `content`: Full article text
  - `created_at`: Timestamp of scraping
  - `source`: Source website (e.g., "google_news", "newsapi")
  - `metadata`: JSON field for additional data

**Used by:** 
- `scraper.py`, `scraper2.py`, `scraper_newsapi.py`
- `prepare_dataset.py`

### 2. `data/spider_sentiments.sqlite`
**Purpose:** Stores sentiment analysis results from news articles.

**Tables:**
- `sentiment_results`: Sentiment scores and classifications
  - `id`: Primary key
  - `article_id`: Foreign key to scraped_articles
  - `text`: Text that was analyzed
  - `sentiment`: Sentiment label (positive/negative/neutral)
  - `score`: Confidence score
  - `classifier`: Model used (flair/vader/huggingface_roberta)
  - `preprocess_level`: Preprocessing level (0 or 1)
  - `analyzed_at`: Timestamp

**Used by:**
- `main.py`, `analysis.py`, `train_from_sentiments.py`

### 3. `data/spider_guardian.sqlite`
**Purpose:** Stores bot interactions, replies, and feedback.

**Tables:**
- `interactions`: Bot replies and original posts
  - `id`: Primary key
  - `post_id`: Twitter/X post ID
  - `original_text`: Original post text
  - `reply_text`: Bot's reply
  - `author`: Original post author
  - `like_count`, `reply_count`, `impression_count`, `repost_count`: Engagement metrics
  - `posted_at`: When bot replied
  - `url`: Link to the interaction
  - `last_update`: Last time metrics were updated
  - `metadata`: JSON field for additional data

- `feedback`: User feedback on bot replies
  - `id`: Primary key
  - `interaction_id`: Foreign key to interactions
  - `feedback_score`: Numeric feedback score
  - `feedback_comment`: Text feedback
  - `created_at`: Timestamp

**Used by:**
- `spider_guardian/bot.py`
- `spider_guardian/storage/`
- Bot reply generation and tracking

### 4. `data/spider_trending.sqlite`
**Purpose:** Stores trending posts about spiders from Twitter/X for monitoring.

**Tables:**
- `trending_posts`: Trending posts and their metrics
  - `post_id`: Primary key (Twitter/X post ID)
  - `text`: Post content
  - `author`: Post author
  - `like_count`, `reply_count`, `impression_count`, `repost_count`: Engagement metrics
  - `collected_at`: When post was first scraped
  - `post_created_at`: When post was originally created
  - `url`: Link to post
  - `last_update`: Last time metrics were refreshed

**Used by:**
- `spider_guardian/storage/trending.py`
- `spider_guardian/langsmith/config.py` (for uploading to LangSmith)

## LangSmith Dataset Naming

LangSmith datasets are used for tracking and analyzing bot performance:

### Dataset Names:
1. **`spider-news-dataset`**: Scraped news articles (from `spider_news.sqlite`)
2. **`spider-sentiments-dataset`**: Sentiment analysis results (from `spider_sentiments.sqlite`)
3. **`spider-replies-dataset`**: Bot replies and interactions (from `spider_guardian.sqlite`)
4. **`spider-trending-dataset`**: Trending posts (from `spider_trending.sqlite`)

### Environment Variables:
```bash
# Set the active dataset name (defaults to spider-trending-dataset)
export LANGSMITH_DATASET="spider-trending-dataset"

# Set the project name
export LANGSMITH_PROJECT="spider-guardian-bot"

# Set your API key
export LANGSMITH_API_KEY="your-api-key-here"
```

## Migration

To migrate from the old naming scheme to the new one:

```bash
python migrate_databases.py
```

This will:
- Create backups of existing databases (`.backup` extension)
- Rename/copy databases to the new structure
- Create any missing databases with proper schemas
- Preserve all existing data

## Usage Examples

### Update trending posts metrics in LangSmith:
```bash
python update_datasets.py \
  --db data/spider_trending.sqlite \
  --dataset spider-trending-dataset \
  --update \
  --max-examples 10
```

### Upload new trending posts to LangSmith:
```bash
python update_datasets.py \
  --db data/spider_trending.sqlite \
  --dataset spider-trending-dataset \
  --upload \
  --max-examples 50
```

### Run both update and upload:
```bash
python update_datasets.py \
  --db data/spider_trending.sqlite \
  --dataset spider-trending-dataset \
  --update \
  --upload \
  --max-examples 20
```

## File References

Files that reference these databases have been updated:
- `spider_guardian/langsmith/config.py`
- `update_datasets.py`
- `.vscode/launch.json`
- `spider_guardian/storage/sql.py`
- `spider_guardian/storage/trending.py`

## Legacy Files

The following files are maintained for backward compatibility:
- `data/interactions.json` (legacy interaction log, now in `spider_guardian.sqlite`)
- `data/bot_interactions.csv` (legacy CSV export)
- `resultats/` directory (sentiment analysis CSVs, now in `spider_sentiments.sqlite`)
