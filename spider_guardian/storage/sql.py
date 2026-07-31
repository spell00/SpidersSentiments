"""SQLite-backed storage for Spider Guardian datasets and analysis results."""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import logging

import pandas as pd


@dataclass
class DatasetRecord:
    """Representation of a dataset entry loaded from legacy CSV exports."""

    id: str
    url: str
    language: Optional[str]
    title: Optional[str]
    sensationalism: Optional[int]
    payload: Dict[str, Any]

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "DatasetRecord":
        record_id = str(payload.get("ID") or payload.get("id"))
        if not record_id:
            raise ValueError("Dataset row missing required 'ID' column")
        title = payload.get("Title") or payload.get("title")
        language = payload.get("Language") or payload.get("language")
        sensationalism_raw = payload.get("Sensationalism") or payload.get("sensationalism")
        try:
            sensationalism = int(sensationalism_raw) if sensationalism_raw not in (None, "") else None
        except (TypeError, ValueError):
            sensationalism = None
        url = payload.get("URL") or payload.get("url") or ""
        return cls(
            id=str(record_id),
            url=str(url),
            language=str(language) if language is not None else None,
            title=str(title) if title is not None else None,
            sensationalism=sensationalism,
            payload=payload,
        )


@dataclass
class SentimentResult:
    """Persisted sentiment analysis output for a dataset row."""

    dataset_id: str
    classifier: str
    preprocess: int
    pos: float
    neg: float
    neu: float
    created_at: datetime
    payload: Dict[str, Any]


@dataclass
class ScrapedArticle:
    """Metadata for ad-hoc scraped articles stored in SQL for auditing."""

    title: str
    link: str
    content: Optional[str]
    metadata: Dict[str, Any]
    created_at: datetime


class SQLDataStore:
    """High-level helper around SQLite for Spider Guardian data."""

    def __init__(self, path: str = "data/spider_guardian.sqlite") -> None:
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._ensure_initialised()

    # ------------------------------------------------------------------
    # connection helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        # Gracefully handle legacy records with non-UTF8 encodings by replacing undecodable bytes
        def _decode_text(value: Any) -> str:
            if isinstance(value, (bytes, bytearray)):
                return value.decode("utf-8", errors="replace")
            return value

        conn.text_factory = _decode_text
        return conn

    def _ensure_initialised(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dataset_entries (
                    id TEXT PRIMARY KEY,
                    url TEXT,
                    language TEXT,
                    title TEXT,
                    sensationalism INTEGER,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sentiment_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id TEXT NOT NULL,
                    classifier TEXT NOT NULL,
                    preprocess INTEGER NOT NULL,
                    pos REAL NOT NULL,
                    neg REAL NOT NULL,
                    neu REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT,
                    FOREIGN KEY(dataset_id) REFERENCES dataset_entries(id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sentiment_lookup ON sentiment_results(dataset_id, classifier, preprocess)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scraped_articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    link TEXT NOT NULL UNIQUE,
                    content TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            # New normalized tables
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tweet_id TEXT,
                    reply_id TEXT,
                    tweet_text TEXT,
                    reply_text TEXT,
                    author_handle TEXT,
                    lang TEXT,
                    url TEXT,
                    type TEXT,
                    parent_post_id TEXT,
                    thread_root_id TEXT,
                    like_count INTEGER,
                    reply_count INTEGER,
                    impression_count INTEGER,
                    repost_count INTEGER,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT
                )
                """
            )
            # Ensure interactions schema matches expected shape (handles legacy tables)
            self._ensure_interactions_schema()
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_interactions_keys ON interactions(tweet_id, reply_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_interactions_url ON interactions(url)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_interactions_created ON interactions(created_at)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS content (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    post_id TEXT,
                    text TEXT,
                    author_handle TEXT,
                    lang TEXT,
                    url TEXT,
                    like_count INTEGER,
                    reply_count INTEGER,
                    impression_count INTEGER,
                    repost_count INTEGER,
                    collected_at TEXT,
                    created_at TEXT NOT NULL,
                    metadata TEXT
                )
                """
            )
            # Ensure content schema matches expected shape (handles legacy tables)
            self._ensure_content_schema()
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_content_post ON content(post_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_content_url ON content(url)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_content_collected ON content(collected_at)"
            )
            # Author follower tracking table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS authors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    handle TEXT NOT NULL UNIQUE,
                    follower_count INTEGER,
                    last_updated TEXT,
                    first_seen TEXT NOT NULL,
                    tweet_count INTEGER DEFAULT 0,
                    metadata TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_authors_handle ON authors(handle)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_authors_followers ON authors(follower_count)"
            )
            conn.commit()

    # ------------------------------
    # schema helpers / migration
    # ------------------------------

    def _table_exists(self, name: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
            return cur.fetchone() is not None

    def _ensure_interactions_schema(self) -> None:
        """Ensure the interactions table has the expected columns; rebuild if necessary."""
        expected_cols = {
            "id","tweet_id","reply_id","tweet_text","reply_text","author_handle","lang","url","type",
            "parent_post_id","thread_root_id","like_count","reply_count","impression_count","repost_count",
            "metadata","created_at","updated_at"
        }
        with self._connect() as conn:
            cur = conn.execute("PRAGMA table_info(interactions)")
            cols = {row[1] for row in cur.fetchall()}
            if not cols:
                return
            if not expected_cols.issubset(cols):
                # Rebuild table with full schema
                conn.execute("BEGIN")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS interactions_new (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        tweet_id TEXT,
                        reply_id TEXT,
                        tweet_text TEXT,
                        reply_text TEXT,
                        author_handle TEXT,
                        lang TEXT,
                        url TEXT,
                        type TEXT,
                        parent_post_id TEXT,
                        thread_root_id TEXT,
                        like_count INTEGER,
                        reply_count INTEGER,
                        impression_count INTEGER,
                        repost_count INTEGER,
                        metadata TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT
                    )
                    """
                )
                # Copy over intersecting columns
                common = sorted(list(expected_cols.intersection(cols)))
                col_list = ",".join(common)
                conn.execute(f"INSERT INTO interactions_new ({col_list}) SELECT {col_list} FROM interactions")
                conn.execute("DROP TABLE interactions")
                conn.execute("ALTER TABLE interactions_new RENAME TO interactions")
                conn.commit()

    def _ensure_content_schema(self) -> None:
        """Ensure the content table has the expected columns; rebuild if necessary."""
        expected_cols = {
            "id","post_id","text","author_handle","lang","url","like_count","reply_count","impression_count","repost_count","collected_at","created_at","metadata"
        }
        with self._connect() as conn:
            cur = conn.execute("PRAGMA table_info(content)")
            cols = {row[1] for row in cur.fetchall()}
            if not cols:
                return
            if not expected_cols.issubset(cols):
                conn.execute("BEGIN")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS content_new (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        post_id TEXT,
                        text TEXT,
                        author_handle TEXT,
                        lang TEXT,
                        url TEXT,
                        like_count INTEGER,
                        reply_count INTEGER,
                        impression_count INTEGER,
                        repost_count INTEGER,
                        collected_at TEXT,
                        created_at TEXT NOT NULL,
                        metadata TEXT
                    )
                    """
                )
                common = sorted(list(expected_cols.intersection(cols)))
                col_list = ",".join(common)
                conn.execute(f"INSERT INTO content_new ({col_list}) SELECT {col_list} FROM content")
                conn.execute("DROP TABLE content")
                conn.execute("ALTER TABLE content_new RENAME TO content")
                conn.commit()

    def migrate_scraped_articles(self, limit: Optional[int] = None) -> Dict[str, int]:
        """Migrate legacy scraped_articles rows into normalized tables.

        - interaction, flagged_reply -> interactions
        - streamed_post -> content

        Returns counts of migrated rows per target table.
        """
        migrated = {"interactions": 0, "content": 0}
        if not self._table_exists("scraped_articles"):
            return migrated
        count = 0
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT title, link, content, metadata, created_at FROM scraped_articles ORDER BY id"
            )
            rows = cur.fetchall()
        for row in rows:
            if limit is not None and count >= limit:
                break
            title = row[0] or ""
            link = row[1] or ""
            content_raw = row[2]
            metadata_raw = row[3]
            created_at_raw = row[4]
            try:
                meta = {} if not metadata_raw else (meta if isinstance(metadata_raw, dict) else json.loads(metadata_raw))
            except Exception:
                try:
                    meta = json.loads(str(metadata_raw)) if metadata_raw else {}
                except Exception:
                    meta = {}
            # content field may be JSON string or dict
            content: Dict[str, Any]
            if isinstance(content_raw, dict):
                content = content_raw  # type: ignore[assignment]
            else:
                try:
                    content = json.loads(content_raw) if content_raw else {}
                except Exception:
                    content = {}
            try:
                created_at = (
                    created_at_raw if isinstance(created_at_raw, str) else (created_at_raw.isoformat() if created_at_raw else datetime.utcnow().isoformat())
                )
            except Exception:
                created_at = datetime.utcnow().isoformat()

            meta_type = (meta or {}).get("type") or content.get("type")
            if meta_type in ("interaction", "flagged_reply"):
                # Build interaction record
                metrics = content.get("metrics") or {}
                record = (
                    str(content.get("tweet_id") or ""),
                    str(content.get("reply_id") or ""),
                    str(content.get("tweet_text") or ""),
                    str(content.get("reply_text") or (title or "")),
                    str(content.get("author_handle") or meta.get("author_handle") or ""),
                    str(content.get("lang") or meta.get("lang") or ""),
                    str(content.get("url") or link),
                    str(meta_type),
                    str(content.get("parent_post_id") or ""),
                    str(content.get("thread_root_id") or ""),
                    int(metrics.get("like_count", 0) or 0),
                    int(metrics.get("reply_count", 0) or 0),
                    int(metrics.get("impression_count", 0) or 0),
                    int(metrics.get("repost_count", 0) or 0),
                    json.dumps({k: v for k, v in {**meta, **content}.items() if k not in {
                        "tweet_id","reply_id","tweet_text","reply_text","author_handle","lang","url","parent_post_id","thread_root_id","metrics","like_count","reply_count","impression_count","repost_count"
                    }}, ensure_ascii=False),
                    created_at,
                    datetime.utcnow().isoformat(),
                )
                with self._connect() as conn:
                    conn.execute(
                        """
                        INSERT INTO interactions (
                            tweet_id, reply_id, tweet_text, reply_text, author_handle, lang, url, type,
                            parent_post_id, thread_root_id, like_count, reply_count, impression_count, repost_count,
                            metadata, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        record,
                    )
                    conn.commit()
                migrated["interactions"] += 1
                count += 1
            elif meta_type == "streamed_post":
                record = (
                    str(content.get("id") or ""),
                    str(content.get("text") or title or ""),
                    str(content.get("author_handle") or meta.get("author_handle") or ""),
                    str(content.get("lang") or meta.get("lang") or ""),
                    str(content.get("url") or link),
                    int(content.get("like_count", 0) or 0),
                    int(content.get("reply_count", 0) or 0),
                    int(content.get("impression_count", 0) or 0),
                    int(content.get("repost_count", 0) or 0),
                    str(content.get("collected_at") or created_at),
                    created_at,
                    json.dumps({k: v for k, v in {**meta, **content}.items() if k not in {
                        "id","text","author_handle","lang","url","collected_at","like_count","reply_count","impression_count","repost_count"
                    }}, ensure_ascii=False),
                )
                with self._connect() as conn:
                    conn.execute(
                        """
                        INSERT INTO content (
                            post_id, text, author_handle, lang, url,
                            like_count, reply_count, impression_count, repost_count,
                            collected_at, created_at, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        record,
                    )
                    conn.commit()
                migrated["content"] += 1
                count += 1
            else:
                # Unknown type; skip
                continue

        return migrated

    # ------------------------------------------------------------------
    # dataset operations
    # ------------------------------------------------------------------

    def upsert_dataset(self, records: Iterable[DatasetRecord]) -> int:
        """Insert or update dataset records.

        Returns the number of rows written."""

        rows = [
            (
                rec.id,
                rec.url,
                rec.language,
                rec.title,
                rec.sensationalism,
                json.dumps(rec.payload, ensure_ascii=False),
            )
            for rec in records
        ]
        if not rows:
            return 0
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO dataset_entries (id, url, language, title, sensationalism, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    url = excluded.url,
                    language = excluded.language,
                    title = excluded.title,
                    sensationalism = excluded.sensationalism,
                    payload = excluded.payload
                """,
                rows,
            )
            conn.commit()
        return len(rows)

    def import_csv(self, csv_path: str, delimiter: str = "\t") -> int:
        """Load a tabular export into the dataset table."""

        df = pd.read_csv(csv_path, delimiter=delimiter)
        logging.info("Importing %d dataset rows from %s", len(df), csv_path)
        records = [DatasetRecord.from_payload(row.dropna().to_dict()) for _, row in df.iterrows()]
        return self.upsert_dataset(records)

    def iter_dataset(self) -> Iterator[DatasetRecord]:
        with self._connect() as conn:
            cursor = conn.execute("SELECT * FROM dataset_entries ORDER BY id")
            for row in cursor:
                payload = json.loads(row["payload"])
                yield DatasetRecord(
                    id=row["id"],
                    url=row["url"],
                    language=row["language"],
                    title=row["title"],
                    sensationalism=row["sensationalism"],
                    payload=payload,
                )

    def dataset_dataframe(self, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """Return the dataset as a DataFrame for compatibility with existing workflows."""

        records: List[Dict[str, Any]] = []
        for record in self.iter_dataset():
            if columns is None:
                records.append(record.payload)
            else:
                row = {col: record.payload.get(col) for col in columns}
                records.append(row)
        if not records:
            return pd.DataFrame(columns=columns or [])
        return pd.DataFrame(records)

    def get_dataset_record(self, dataset_id: str) -> Optional[DatasetRecord]:
        with self._connect() as conn:
            cursor = conn.execute("SELECT * FROM dataset_entries WHERE id = ?", (dataset_id,))
            row = cursor.fetchone()
            if row is None:
                return None
            payload = json.loads(row["payload"])
            return DatasetRecord(
                id=row["id"],
                url=row["url"],
                language=row["language"],
                title=row["title"],
                sensationalism=row["sensationalism"],
                payload=payload,
            )

    # ------------------------------------------------------------------
    # scraped articles
    # ------------------------------------------------------------------

    def upsert_scraped_articles(self, articles: Sequence[ScrapedArticle]) -> int:
        if not articles:
            return 0
        rows = [
            (
                article.title,
                article.link,
                article.content,
                json.dumps(article.metadata, ensure_ascii=False) if article.metadata else None,
                article.created_at.isoformat(),
            )
            for article in articles
        ]
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO scraped_articles (title, link, content, metadata, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(link) DO UPDATE SET
                    title = excluded.title,
                    content = excluded.content,
                    metadata = excluded.metadata,
                    created_at = excluded.created_at
                """,
                rows,
            )
            conn.commit()
        return len(rows)

    # New upsert helpers for normalized tables
    def upsert_interactions(self, rows: Sequence[Dict[str, Any]]) -> int:
        if not rows:
            return 0
        prepared = [
            (
                str(r.get("tweet_id") or ""),
                str(r.get("reply_id") or ""),
                r.get("tweet_text"),
                r.get("reply_text"),
                r.get("author_handle"),
                r.get("lang"),
                r.get("url"),
                r.get("type"),
                r.get("parent_post_id"),
                r.get("thread_root_id"),
                int(r.get("like_count", 0) or 0),
                int(r.get("reply_count", 0) or 0),
                int(r.get("impression_count", 0) or 0),
                int(r.get("repost_count", 0) or 0),
                json.dumps(r.get("metadata") or {}, ensure_ascii=False),
                (r.get("created_at") or datetime.utcnow()).isoformat() if not isinstance(r.get("created_at"), str) else r.get("created_at"),
                datetime.utcnow().isoformat(),
            )
            for r in rows
        ]
        with self._connect() as conn:
            # No natural unique constraint across (tweet_id, reply_id, url), so do best-effort upsert by (reply_id or url)
            # Create a temporary table to decide updates vs inserts
            conn.executemany(
                """
                INSERT INTO interactions (
                    tweet_id, reply_id, tweet_text, reply_text, author_handle, lang, url, type,
                    parent_post_id, thread_root_id, like_count, reply_count, impression_count, repost_count,
                    metadata, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                prepared,
            )
            conn.commit()
        return len(prepared)

    def upsert_content(self, rows: Sequence[Dict[str, Any]]) -> int:
        if not rows:
            return 0
        prepared = [
            (
                str(r.get("post_id") or ""),
                r.get("text"),
                r.get("author_handle"),
                r.get("lang"),
                r.get("url"),
                int(r.get("like_count", 0) or 0),
                int(r.get("reply_count", 0) or 0),
                int(r.get("impression_count", 0) or 0),
                int(r.get("repost_count", 0) or 0),
                (r.get("collected_at") or r.get("created_at") or datetime.utcnow()).isoformat() if not isinstance(r.get("collected_at"), str) else r.get("collected_at"),
                (r.get("created_at") or datetime.utcnow()).isoformat() if not isinstance(r.get("created_at"), str) else r.get("created_at"),
                json.dumps(r.get("metadata") or {}, ensure_ascii=False),
            )
            for r in rows
        ]
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO content (
                    post_id, text, author_handle, lang, url,
                    like_count, reply_count, impression_count, repost_count,
                    collected_at, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                prepared,
            )
            conn.commit()
        return len(prepared)

    def bulk_update_interactions_metrics(self, updates: Sequence[Dict[str, Any]]) -> int:
        """Update engagement metrics for existing interactions rows.

        Each update dict may include one of the keys identifying the row:
        - id (the interactions table primary key)
        - reply_id
        - tweet_id
        - url

        And should include integer counts for like_count, reply_count,
        impression_count, repost_count. Missing counts default to 0.

        Returns the number of rows that were updated (sum across all updates).
        """
        if not updates:
            return 0

        total_updated = 0
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            for u in updates:
                like_c = int(u.get("like_count", 0) or 0)
                reply_c = int(u.get("reply_count", 0) or 0)
                imp_c = int(u.get("impression_count", 0) or 0)
                repost_c = int(u.get("repost_count", 0) or 0)

                # Determine best key to match existing row
                params: List[Any]
                where_clause: Optional[str] = None

                if u.get("id") is not None:
                    where_clause = "id = ?"
                    params = [like_c, reply_c, imp_c, repost_c, now, u["id"]]
                elif u.get("reply_id"):
                    where_clause = "reply_id = ?"
                    params = [like_c, reply_c, imp_c, repost_c, now, str(u["reply_id"])]
                elif u.get("tweet_id"):
                    where_clause = "tweet_id = ?"
                    params = [like_c, reply_c, imp_c, repost_c, now, str(u["tweet_id"])]
                elif u.get("url"):
                    where_clause = "url = ?"
                    params = [like_c, reply_c, imp_c, repost_c, now, str(u["url"])]
                else:
                    # No usable key; skip
                    continue

                try:
                    cur = conn.execute(
                        f"""
                        UPDATE interactions
                        SET like_count = ?, reply_count = ?, impression_count = ?, repost_count = ?, updated_at = ?
                        WHERE {where_clause}
                        """,
                        params,
                    )
                    total_updated += cur.rowcount if cur.rowcount is not None else 0
                except Exception:
                    # Skip individual failures to keep bulk update robust
                    continue
            conn.commit()

        return total_updated

    def iter_scraped_articles(self) -> Iterator[ScrapedArticle]:
        with self._connect() as conn:
            cursor = conn.execute("SELECT title, link, content, metadata, created_at FROM scraped_articles ORDER BY created_at DESC")
            for title, link, content, metadata, created_at in cursor:
                try:
                    # Handle metadata safely - might be corrupted JSON or dict
                    if isinstance(metadata, dict):
                        metadata_dict = metadata
                    elif isinstance(metadata, str) and metadata:
                        try:
                            metadata_dict = json.loads(metadata)
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            metadata_dict = {}
                    else:
                        metadata_dict = {}
                    
                    # Handle created_at safely
                    try:
                        if isinstance(created_at, str):
                            created_dt = datetime.fromisoformat(created_at)
                        else:
                            created_dt = created_at or datetime.now()
                    except (ValueError, TypeError):
                        created_dt = datetime.now()
                    
                    yield ScrapedArticle(
                        title=title or "",
                        link=link or "",
                        content=content or "",
                        metadata=metadata_dict,
                        created_at=created_dt,
                    )
                except Exception as e:
                    # Skip corrupted records
                    import logging
                    logging.debug("Skipping corrupted scraped article record: %s", e)
                    continue

    # New iterators for normalized tables
    def iter_interactions(self) -> Iterator[Dict[str, Any]]:
        if not self._table_exists("interactions"):
            return iter(())
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT tweet_id, reply_id, tweet_text, reply_text, author_handle, lang, url, type,
                       parent_post_id, thread_root_id, like_count, reply_count, impression_count, repost_count,
                       metadata, created_at, updated_at
                FROM interactions
                ORDER BY datetime(created_at) DESC
                """
            )
            for r in cur:
                md = {}
                try:
                    md = json.loads(r[14]) if r[14] else {}
                except Exception:
                    md = {}
                yield {
                    "tweet_id": r[0] or "",
                    "reply_id": r[1] or "",
                    "tweet_text": r[2] or "",
                    "reply_text": r[3] or "",
                    "author_handle": r[4] or "",
                    "lang": r[5] or "",
                    "url": r[6] or "",
                    "type": r[7] or "interaction",
                    "parent_post_id": r[8] or "",
                    "thread_root_id": r[9] or "",
                    "like_count": r[10] or 0,
                    "reply_count": r[11] or 0,
                    "impression_count": r[12] or 0,
                    "repost_count": r[13] or 0,
                    "metadata": md,
                    "created_at": r[15],
                    "updated_at": r[16],
                }

    def iter_content(self) -> Iterator[Dict[str, Any]]:
        if not self._table_exists("content"):
            return iter(())
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT post_id, text, author_handle, lang, url,
                       like_count, reply_count, impression_count, repost_count,
                       collected_at, created_at, metadata
                FROM content
                ORDER BY datetime(collected_at) DESC
                """
            )
            for r in cur:
                md = {}
                try:
                    md = json.loads(r[11]) if r[11] else {}
                except Exception:
                    md = {}
                yield {
                    "post_id": r[0] or "",
                    "text": r[1] or "",
                    "author_handle": r[2] or "",
                    "lang": r[3] or "",
                    "url": r[4] or "",
                    "like_count": r[5] or 0,
                    "reply_count": r[6] or 0,
                    "impression_count": r[7] or 0,
                    "repost_count": r[8] or 0,
                    "collected_at": r[9],
                    "created_at": r[10],
                    "metadata": md,
                }

    # ------------------------------------------------------------------
    # sentiment results
    # ------------------------------------------------------------------

    def save_sentiment_results(self, results: Sequence[SentimentResult]) -> int:
        if not results:
            return 0
        rows = [
            (
                res.dataset_id,
                res.classifier,
                res.preprocess,
                res.pos,
                res.neg,
                res.neu,
                res.created_at.isoformat(),
                json.dumps(res.payload, ensure_ascii=False) if res.payload else None,
            )
            for res in results
        ]
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO sentiment_results (
                    dataset_id, classifier, preprocess, pos, neg, neu, created_at, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
        return len(rows)

    def clear_sentiment_results(self, classifier: Optional[str] = None, preprocess: Optional[int] = None) -> None:
        query = "DELETE FROM sentiment_results"
        params: List[Any] = []
        clauses: List[str] = []
        if classifier is not None:
            clauses.append("classifier = ?")
            params.append(classifier)
        if preprocess is not None:
            clauses.append("preprocess = ?")
            params.append(preprocess)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        with self._connect() as conn:
            conn.execute(query, params)
            conn.commit()

    def iter_sentiment_results(
        self,
        classifier: Optional[str] = None,
        preprocess: Optional[int] = None,
    ) -> Iterator[SentimentResult]:
        query = "SELECT dataset_id, classifier, preprocess, pos, neg, neu, created_at, payload FROM sentiment_results"
        params: List[Any] = []
        clauses: List[str] = []
        if classifier is not None:
            clauses.append("classifier = ?")
            params.append(classifier)
        if preprocess is not None:
            clauses.append("preprocess = ?")
            params.append(preprocess)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC"
        with self._connect() as conn:
            cursor = conn.execute(query, params)
            for row in cursor:
                payload = json.loads(row["payload"]) if row["payload"] else {}
                created_at = datetime.fromisoformat(row["created_at"])
                yield SentimentResult(
                    dataset_id=row["dataset_id"],
                    classifier=row["classifier"],
                    preprocess=row["preprocess"],
                    pos=row["pos"],
                    neg=row["neg"],
                    neu=row["neu"],
                    created_at=created_at,
                    payload=payload,
                )

    def sentiment_dataframe(self, classifier: Optional[str] = None, preprocess: Optional[int] = None) -> pd.DataFrame:
        results = [
            {
                "dataset_id": res.dataset_id,
                "classifier": res.classifier,
                "preprocess": res.preprocess,
                "pos": res.pos,
                "neg": res.neg,
                "neu": res.neu,
                "created_at": res.created_at,
                **res.payload,
            }
            for res in self.iter_sentiment_results(classifier=classifier, preprocess=preprocess)
        ]
        if not results:
            return pd.DataFrame(columns=["dataset_id", "classifier", "preprocess", "pos", "neg", "neu", "created_at"])
        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # utilities
    # ------------------------------------------------------------------

    def vacuum(self) -> None:
        with self._connect() as conn:
            conn.execute("VACUUM")
            conn.commit()

    # ------------------------------------------------------------------
    # author follower tracking
    # ------------------------------------------------------------------

    def upsert_author(
        self,
        handle: str,
        follower_count: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert or update an author record with follower count.
        
        If author exists, updates follower_count, last_updated, and increments tweet_count.
        If new, creates record with first_seen timestamp.
        """
        if not handle:
            return
        
        now = datetime.utcnow().isoformat()
        
        with self._connect() as conn:
            # Check if author exists
            cur = conn.execute("SELECT id, tweet_count, metadata FROM authors WHERE handle = ?", (handle,))
            existing = cur.fetchone()
            
            if existing:
                # Update existing author
                tweet_count = (existing[1] or 0) + 1
                # Merge metadata and stamp followers_checked_at only when follower_count is refreshed
                try:
                    existing_meta = json.loads(existing[2] or "{}")
                except Exception:
                    existing_meta = {}
                merged_meta = dict(existing_meta)
                if follower_count is not None:
                    merged_meta["followers_checked_at"] = now
                    conn.execute(
                        """
                        UPDATE authors 
                        SET follower_count = ?, last_updated = ?, tweet_count = ?, metadata = ?
                        WHERE handle = ?
                        """,
                        (follower_count, now, tweet_count, json.dumps(merged_meta, ensure_ascii=False), handle),
                    )
                else:
                    # Just increment tweet count if no follower_count provided
                    conn.execute(
                        """
                        UPDATE authors 
                        SET tweet_count = ?, last_updated = ?, metadata = ?
                        WHERE handle = ?
                        """,
                        (tweet_count, now, json.dumps(merged_meta, ensure_ascii=False), handle),
                    )
            else:
                # Insert new author
                # Use -1 as sentinel for unknown follower count
                initial_followers = follower_count if follower_count is not None else -1
                initial_meta = dict(metadata or {})
                if follower_count is not None:
                    initial_meta["followers_checked_at"] = now
                conn.execute(
                    """
                    INSERT INTO authors (handle, follower_count, first_seen, last_updated, tweet_count, metadata)
                    VALUES (?, ?, ?, ?, 1, ?)
                    """,
                    (handle, initial_followers, now, now, json.dumps(initial_meta, ensure_ascii=False)),
                )
            conn.commit()

    def get_author_follower_count(self, handle: str) -> Optional[int]:
        """Retrieve the cached follower count for an author.
        
        Returns None if author not found or follower_count not set.
        """
        if not handle:
            return None
        
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT follower_count FROM authors WHERE handle = ?",
                (handle,)
            )
            row = cur.fetchone()
            return row[0] if row and row[0] is not None else None

    def get_author_followers_info(self, handle: str) -> Dict[str, Any]:
        """Return follower_count and when it was last checked for an author.
        
        Returns a dict: {"follower_count": Optional[int], "followers_checked_at": Optional[str]}.
        followers_checked_at is an ISO timestamp from metadata or None.
        """
        result: Dict[str, Any] = {"follower_count": None, "followers_checked_at": None}
        if not handle:
            return result
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT follower_count, metadata FROM authors WHERE handle = ?",
                (handle,)
            )
            row = cur.fetchone()
            if not row:
                return result
            result["follower_count"] = row[0]
            try:
                meta = json.loads(row[1] or "{}")
                result["followers_checked_at"] = meta.get("followers_checked_at")
            except Exception:
                result["followers_checked_at"] = None
            return result

    def get_all_authors(
        self,
        min_followers: Optional[int] = None,
        max_followers: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve all authors matching follower criteria.
        
        Returns list of dicts with keys: handle, follower_count, last_updated, 
        first_seen, tweet_count.
        """
        query = "SELECT handle, follower_count, last_updated, first_seen, tweet_count FROM authors WHERE 1=1"
        params: List[Any] = []
        
        if min_followers is not None:
            query += " AND follower_count >= ?"
            params.append(min_followers)
        
        if max_followers is not None:
            query += " AND follower_count <= ?"
            params.append(max_followers)
        
        query += " ORDER BY follower_count DESC"
        
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        
        with self._connect() as conn:
            cur = conn.execute(query, params)
            rows = cur.fetchall()
        
        return [
            {
                "handle": row[0],
                "follower_count": row[1],
                "last_updated": row[2],
                "first_seen": row[3],
                "tweet_count": row[4],
            }
            for row in rows
        ]

    def get_top_authors(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve top N authors by follower count.
        
        Convenience method for getting high-reach accounts.
        """
        return self.get_all_authors(limit=limit)

    # ------------------------------------------------------------------
    # utilities
    # ------------------------------------------------------------------

    def vacuum(self) -> None:
        with self._connect() as conn:
            conn.execute("VACUUM")
            conn.commit()

    # ------------------------------------------------------------------
    # maintenance helpers
    # ------------------------------------------------------------------

    def delete_interactions_with_reply_id(self, reply_id: str = "posted_no_id_found") -> int:
        """Delete interactions rows matching a given reply_id sentinel.

        Returns number of rows deleted.
        """
        if not reply_id:
            return 0
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM interactions WHERE reply_id = ?", (reply_id,))
            conn.commit()
            return cur.rowcount or 0

    def close(self) -> None:
        # Connections are per-call; nothing to close but kept for API symmetry.
        pass


@contextlib.contextmanager
def sql_store(path: str = "data/spider_guardian.sqlite") -> Iterator[SQLDataStore]:
    store = SQLDataStore(path)
    try:
        yield store
    finally:
        store.close()
