"""SQLite store for trending social posts used as tone/style exemplars."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterator, List, Optional, Tuple, Any
import logging


@dataclass
class TrendingPost:
    post_id: str
    text: str
    author: Optional[str]
    lang: Optional[str]
    like_count: int
    repost_count: int
    reply_count: int
    impression_count: int  # Added field for impressions
    url: Optional[str]
    collected_at: datetime
    post_created_at: Optional[datetime] = None


class TrendingStore:
    def __init__(self, path: str = "data/trending.sqlite") -> None:
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._ensure_initialised()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_initialised(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trending_posts (
                    post_id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    author TEXT,
                    lang TEXT,
                    like_count INTEGER NOT NULL,
                    repost_count INTEGER NOT NULL,
                    reply_count INTEGER NOT NULL,
                    impression_count INTEGER NOT NULL,
                    url TEXT,
                    collected_at TEXT NOT NULL,
                    post_created_at TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trending_collected ON trending_posts(collected_at)")

            # Check if the impression_count column exists, and add it if missing
            try:
                conn.execute("SELECT impression_count FROM trending_posts LIMIT 1")
            except sqlite3.OperationalError as e:
                if "no such column: impression_count" in str(e).lower():
                    logging.info("Adding missing column: impression_count to trending_posts table.")
                    conn.execute("ALTER TABLE trending_posts ADD COLUMN impression_count INTEGER NOT NULL DEFAULT 0")

            conn.commit()

    def upsert(self, posts: List[TrendingPost]) -> int:
        if not posts:
            return 0
        rows = [
            (
                p.post_id,
                p.text,
                p.author,
                p.lang,
                int(p.like_count or 0),
                int(p.repost_count or 0),
                int(p.reply_count or 0),
                int(p.impression_count or 0),
                p.url,
                p.collected_at.isoformat(),
                p.post_created_at.isoformat() if p.post_created_at else None,
            )
            for p in posts
        ]
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                cur.executemany(
                    """
                    INSERT INTO trending_posts (
                        post_id, text, author, lang, like_count, repost_count, reply_count, impression_count, url, collected_at, post_created_at
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    ON CONFLICT(post_id) DO UPDATE SET
                        text = excluded.text,
                        author = excluded.author,
                        lang = excluded.lang,
                        like_count = excluded.like_count,
                        repost_count = excluded.repost_count,
                        reply_count = excluded.reply_count,
                        impression_count = excluded.impression_count,
                        url = excluded.url,
                        collected_at = excluded.collected_at,
                        post_created_at = excluded.post_created_at
                    """,
                    rows,
                )
                conn.commit()
                return cur.rowcount
            except sqlite3.Error as e:
                logging.error(f"Error upserting posts: {e}")
                return 0

    def purge_older_than_days(self, days: int) -> int:
        logging.info(f"Purging trending posts older than {days} days.")
        cutoff = datetime.utcnow() - timedelta(days=max(0, days))
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM trending_posts WHERE collected_at < ?", (cutoff.isoformat(),))
            conn.commit()
            return cur.rowcount or 0

    def top(self, limit: int = 20, since_hours: Optional[int] = None) -> Iterator[TrendingPost]:
        clauses: List[str] = []
        params: List[Any] = []
        if since_hours is not None:
            cutoff = datetime.utcnow() - timedelta(hours=max(0, since_hours))
            clauses.append("collected_at >= ?")
            params.append(cutoff.isoformat())
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        # Simple engagement score ordering: likes + 2*reposts + 3*replies
        query = (
            "SELECT post_id, text, author, lang, like_count, repost_count, reply_count, impression_count, url, collected_at, post_created_at "
            f"FROM trending_posts{where} "
            "ORDER BY (like_count + 2*repost_count + 3*reply_count) DESC, collected_at DESC "
            "LIMIT ?"
        )
        params.append(int(max(1, limit)))
        with self._connect() as conn:
            for row in conn.execute(query, params):
                yield TrendingPost(
                    post_id=row["post_id"],
                    text=row["text"],
                    author=row["author"],
                    lang=row["lang"],
                    like_count=row["like_count"],
                    repost_count=row["repost_count"],
                    reply_count=row["reply_count"],
                    impression_count=row["impression_count"],
                    url=row["url"],
                    collected_at=datetime.fromisoformat(row["collected_at"]),
                    post_created_at=datetime.fromisoformat(row["post_created_at"]) if row["post_created_at"] else None,
                )


__all__ = ["TrendingPost", "TrendingStore"]
