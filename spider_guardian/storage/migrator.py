"""Utilities to help migrate legacy CSV exports into the new storage backends."""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from typing import Optional

from .nosql import ArticleDocument, ArticleStore
from .sql import SQLDataStore


def migrate_csv_to_sql(csv_path: str, store: Optional[SQLDataStore] = None, delimiter: str = "\t") -> int:
    """Import a tabular CSV/TSV file into the SQL datastore."""

    created_store = store is None
    store = store or SQLDataStore()
    try:
        count = store.import_csv(csv_path, delimiter=delimiter)
        logging.info("Imported %d rows from %s into %s", count, csv_path, store.path)
        return count
    finally:
        if created_store:
            store.close()


def migrate_article_csv_to_nosql(
    csv_path: str,
    store: Optional[ArticleStore] = None,
    fetch_content: bool = False,
    encoding: str = "utf-8",
) -> int:
    """Import article metadata from a CSV into the document store.

    Parameters
    ----------
    csv_path:
        Legacy CSV file containing at least ``title`` and ``link`` columns.
    store:
        Optional pre-initialised :class:`ArticleStore`.
    fetch_content:
        When True, performs HTTP GET requests to populate ``content`` for each
        article. This can be slow; by default we only migrate metadata.
    encoding:
        Encoding of the source CSV file.
    """

    import requests  # Local import to keep optional dependency lazy

    created_store = store is None
    store = store or ArticleStore()
    inserted = 0
    try:
        with open(csv_path, newline="", encoding=encoding) as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                title = row.get("title") or row.get("Title")
                link = row.get("link") or row.get("Link")
                if not title or not link:
                    continue
                content = None
                if fetch_content:
                    try:
                        response = requests.get(link, timeout=15)
                        if response.ok:
                            content = response.text
                    except Exception:
                        content = None
                metadata = {k: v for k, v in row.items() if k not in {"title", "link", "Title", "Link"}}
                store.upsert(
                    ArticleDocument(
                        title=title,
                        link=link,
                        content=content,
                        metadata=metadata,
                        created_at=datetime.utcnow(),
                    )
                )
                inserted += 1
        logging.info("Migrated %d articles from %s into %s", inserted, csv_path, store.path)
        return inserted
    finally:
        if created_store:
            store.close()
