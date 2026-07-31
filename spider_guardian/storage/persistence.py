"""Shared helpers for persisting scraped content into the new storage backends."""

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Callable, Dict, Iterable, Optional

from .nosql import ArticleDocument, ArticleStore
from .sql import SQLDataStore, ScrapedArticle


def persist_scraped_articles(
    articles: Iterable[Dict[str, str]],
    *,
    query: str,
    output_dir: str,
    article_store_path: str,
    sql_db_path: Optional[str],
    fetch_content: Optional[Callable[[str], Optional[str]]] = None,
    save_html: bool = False,
    store_content: bool = False,
    legacy_csv_path: Optional[str] = None,
    source: str = "generic",
) -> int:
    """Persist scraped article metadata into document and relational stores.

    Parameters
    ----------
    articles:
        Iterable of article dictionaries containing at least ``title`` and ``link``.
    query:
        Search query used to generate the articles. Stored as metadata.
    output_dir:
        Directory where optional HTML snapshots will be written.
    article_store_path:
        Path to the TinyDB JSON file.
    sql_db_path:
        Path to the SQLite database. When ``None`` the SQL stage is skipped.
    fetch_content:
        Callable returning the HTML body for a given URL. When ``None`` no
        network requests are performed.
    save_html:
        When ``True`` saves HTML snapshots to ``output_dir``.
    store_content:
        When ``True`` embeds HTML content directly into the databases.
    legacy_csv_path:
        Optional CSV path to keep backwards compatibility exports.
    source:
        Identifier of the scraper/source.

    Returns
    -------
    int
        Number of articles successfully persisted.
    """

    articles = list(articles)
    if not articles:
        return 0

    os.makedirs(output_dir, exist_ok=True)
    store = ArticleStore(article_store_path)
    sql_store = SQLDataStore(sql_db_path) if sql_db_path else None

    article_docs = []
    sql_docs = []
    csv_rows = []

    for index, article in enumerate(articles):
        title = article.get("title") or article.get("name") or "Untitled article"
        link = article.get("link") or article.get("url")
        if not link:
            continue

        html_content = None
        if fetch_content is not None and (save_html or store_content):
            html_content = fetch_content(link)
            if html_content is not None and save_html:
                suffix = "html" if "<" in html_content[:100] else "txt"
                snapshot_path = os.path.join(output_dir, f"article_{index}.{suffix}")
                with open(snapshot_path, "w", encoding="utf-8") as handle:
                    handle.write(html_content)

        metadata = {
            "source": source,
            "query": query,
            "rank": index,
        }

        article_docs.append(
            ArticleDocument(
                title=title,
                link=link,
                content=html_content if store_content else None,
                metadata=metadata,
            )
        )

        if sql_store is not None:
            sql_docs.append(
                ScrapedArticle(
                    title=title,
                    link=link,
                    content=html_content if store_content else None,
                    metadata=metadata,
                    created_at=datetime.utcnow(),
                )
            )

        csv_rows.append({"title": title, "link": link})

    if article_docs:
        store.insert_many(article_docs)
    if sql_store is not None and sql_docs:
        sql_store.upsert_scraped_articles(sql_docs)
    if legacy_csv_path:
        with open(legacy_csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["title", "link"])
            writer.writeheader()
            writer.writerows(csv_rows)

    store.close()
    if sql_store is not None:
        sql_store.close()

    return len(article_docs)
