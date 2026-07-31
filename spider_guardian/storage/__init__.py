"""Storage backends for Spider Guardian data.

The storage package centralises all persistence concerns and exposes
lightweight helpers for both relational (SQL) and document (NoSQL)
databases. The goal is to replace the previous CSV-based approach with
robust, queryable data stores while keeping the rest of the application
API-friendly.

Modules
-------
sql
    SQLite-backed relational storage for structured datasets and analysis
    results.
nosql
    TinyDB-backed document storage for scraped article payloads and other
    semi-structured records.
migrator
    Utilities to import existing CSV exports into the new databases.
"""

from .sql import SQLDataStore, DatasetRecord, SentimentResult, ScrapedArticle
from .nosql import ArticleStore, ArticleDocument
from .migrator import migrate_csv_to_sql, migrate_article_csv_to_nosql
from .persistence import persist_scraped_articles
from .trending import TrendingStore, TrendingPost

__all__ = [
    "SQLDataStore",
    "DatasetRecord",
    "SentimentResult",
    "ScrapedArticle",
    "ArticleStore",
    "ArticleDocument",
    "migrate_csv_to_sql",
    "migrate_article_csv_to_nosql",
    "persist_scraped_articles",
    "TrendingStore",
    "TrendingPost",
]
