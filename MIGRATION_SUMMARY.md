# Database Reorganization Summary

## ✅ Changes Completed

### 1. Database Migration
- ✅ Created `data/spider_trending.sqlite` (migrated from `trending.sqlite`)
- ✅ Verified `data/spider_guardian.sqlite` with proper schema
- ✅ Created `data/spider_news.sqlite` for scraped articles
- ✅ Created `data/spider_sentiments.sqlite` for sentiment analysis
- ✅ All original databases backed up with `.backup` extension

### 2. Code Updates
**Updated files:**
- ✅ `spider_guardian/langsmith/config.py`
  - Changed default db path: `data/trending.sqlite` → `data/spider_trending.sqlite`
  - Changed default dataset: `trending-dataset` → `spider-trending-dataset`
  
- ✅ `update_datasets.py`
  - Updated default database path for uploads
  
- ✅ `.vscode/launch.json`
  - Updated "Debug Spider Guardian (Periodic Updates)" configuration
  - New args: `--db data/spider_trending.sqlite --dataset spider-trending-dataset`

### 3. Documentation
- ✅ Created `DATABASE_CONFIG.md` - Complete database structure guide
- ✅ Created `migrate_databases.py` - Reusable migration script

## 📁 New Database Structure

```
data/
├── spider_news.sqlite          # Scraped articles (scraper.py, prepare_dataset.py)
├── spider_sentiments.sqlite    # Sentiment analysis (main.py, analysis.py)
├── spider_guardian.sqlite      # Bot interactions & feedback (bot.py)
└── spider_trending.sqlite      # Trending posts & metrics (trending.py)
```

## 🏷️ LangSmith Dataset Names

- `spider-news-dataset` → News articles
- `spider-sentiments-dataset` → Sentiment analysis
- `spider-replies-dataset` → Bot replies
- `spider-trending-dataset` → Trending posts ✨ (updated)

## 🔄 Migration Status

| Old Name | New Name | Status |
|----------|----------|--------|
| `trending.sqlite` | `spider_trending.sqlite` | ✅ Migrated & backed up |
| N/A | `spider_news.sqlite` | ✅ Created |
| N/A | `spider_sentiments.sqlite` | ✅ Created |
| `spider_guardian.sqlite` | (same) | ✅ Verified & backed up |

## ⚙️ Next Steps

1. **Test the migration:**
   ```bash
   # Test upload with new database
   python update_datasets.py \
     --db data/spider_trending.sqlite \
     --dataset spider-trending-dataset \
     --upload \
     --max-examples 2
   
   # Test update with new database
   python update_datasets.py \
     --db data/spider_trending.sqlite \
     --dataset spider-trending-dataset \
     --update \
     --max-examples 2
   ```

2. **Update environment variables (if set):**
   ```bash
   # PowerShell
   $env:LANGSMITH_DATASET = "spider-trending-dataset"
   
   # Or add to your .env file
   LANGSMITH_DATASET=spider-trending-dataset
   ```

3. **Migrate remaining code references:**
   - Update any hardcoded references to old database names in:
     - Custom scripts
     - Jupyter notebooks
     - Documentation

4. **Optional cleanup (after verifying everything works):**
   ```bash
   # Remove backup files once confirmed working
   Remove-Item data/*.backup
   
   # Archive old databases
   Move-Item data/trending.sqlite data/_archive/
   ```

## 🐛 Troubleshooting

### If you need to rollback:
```bash
# Restore from backups
Copy-Item data/trending.sqlite.backup data/trending.sqlite
Copy-Item data/spider_guardian.sqlite.backup data/spider_guardian.sqlite
```

### If datasets don't match:
```bash
# Delete and recreate LangSmith dataset
python update_datasets.py \
  --db data/spider_trending.sqlite \
  --dataset spider-trending-dataset \
  --upload
```

## 📝 Notes

- All original data is preserved in `.backup` files
- The migration script can be run multiple times safely
- Legacy CSV/JSON files remain untouched for backward compatibility
- Schema updates ensure all new databases have proper foreign keys and indexes

## ✨ Benefits

1. **Clear naming convention:** Each database's purpose is obvious from its name
2. **Separation of concerns:** News, sentiments, bot data, and trending posts are isolated
3. **Easier maintenance:** No confusion about which database stores what
4. **Better scalability:** Each database can be optimized independently
5. **Consistent LangSmith integration:** Dataset names match database purposes
