"""Document store powered by TinyDB for scraped articles and semi-structured data."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, Iterator, List, Optional

from tinydb import Query, TinyDB


@dataclass
class ArticleDocument:
    """Minimal representation of a scraped article."""

    title: str
    link: str
    content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def as_record(self) -> Dict[str, Any]:
        record = {
            "title": self.title,
            "link": self.link,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }
        return {k: v for k, v in record.items() if v is not None}


class ArticleStore:
    """Wrapper around TinyDB for storing and querying scraped articles."""

    def __init__(self, path: str = "data/articles.json") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.db = TinyDB(path)
        self.table = self.db.table("articles")

    def insert(self, document: ArticleDocument) -> int:
        return self.table.insert(document.as_record())

    def insert_many(self, documents: Iterable[ArticleDocument]) -> List[int]:
        records = [doc.as_record() for doc in documents]
        if not records:
            return []
        return self.table.insert_multiple(records)

    def upsert(self, document: ArticleDocument) -> int:
        query = Query()
        return self.table.upsert(document.as_record(), (query.link == document.link))

    def find_by_link(self, link: str) -> Optional[Dict[str, Any]]:
        query = Query()
        result = self.table.get(query.link == link)
        return result

    def search(self, text: str, limit: int = 10) -> List[Dict[str, Any]]:
        query = Query()
        lower = text.lower()
        results = [
            row
            for row in self.table
            if lower in (row.get("title", "").lower() + row.get("content", "").lower())
        ]
        return results[:limit]

    def iter_all(self) -> Iterator[Dict[str, Any]]:
        yield from self.table

    def purge(self) -> None:
        self.table.truncate()

    def close(self) -> None:
        self.db.close()


__all__ = ["ArticleStore", "ArticleDocument"]
