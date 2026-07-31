"""Human tweet loading and similarity helpers."""

from __future__ import annotations

import json
import os
import logging
import random
from typing import List, Optional, Sequence, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors


class HumanPostIndex:
    """Stores saved human posts and exposes similarity search."""

    def __init__(self, posts: Sequence[str], embedder: SentenceTransformer) -> None:
        self.posts = list(posts)
        self.embedder = embedder
        self.index: Optional[NearestNeighbors] = None
        self.embeddings: Optional[np.ndarray] = None
        if self.posts:
            self._build_index()

    def _build_index(self) -> None:
        try:
            embeddings = self.embedder.encode(
                self.posts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except Exception as exc:
            logging.warning("Failed to encode human posts for index: %s", exc)
            self.index = None
            self.embeddings = None
            return
        self.embeddings = embeddings
        self.index = NearestNeighbors(metric="cosine")
        self.index.fit(embeddings)

    def search(self, text: str, limit: int) -> List[str]:
        if limit <= 0 or not self.posts:
            return []
        if self.index is None or self.embeddings is None:
            return random.sample(self.posts, min(limit, len(self.posts)))
        try:
            query_vec = self.embedder.encode([text], convert_to_numpy=True, normalize_embeddings=True)
            k = min(limit, len(self.posts))
            distances, indices = self.index.kneighbors(query_vec, n_neighbors=k)
            seen: set[str] = set()
            results: List[str] = []
            for idx in indices[0]:
                post = self.posts[int(idx)]
                if post and post not in seen:
                    results.append(post)
                    seen.add(post)
                    if len(results) >= limit:
                        break
            return results
        except Exception as exc:
            logging.warning("Human post similarity search failed: %s", exc)
            return random.sample(self.posts, min(limit, len(self.posts)))


def load_human_posts(path: str) -> List[str]:
    """Load human posts from SQL database instead of JSON file."""
    posts: List[str] = []
    if not path:
        return posts
    
    # Try loading from SQL first
    try:
        from .storage import SQLDataStore
        # Use same database path but load human posts from metadata
        db_path = path.replace('streamed_posts.json', 'spider_guardian.sqlite').replace('data/', 'data/')
        if 'spider_guardian.sqlite' not in db_path:
            db_path = 'data/spider_guardian.sqlite'
        
        sql_store = SQLDataStore(db_path)
        for article in sql_store.iter_scraped_articles():
            try:
                # Handle metadata safely - it might be a dict or JSON string
                if isinstance(article.metadata, dict):
                    metadata = article.metadata
                elif isinstance(article.metadata, str):
                    metadata = json.loads(article.metadata)
                else:
                    continue
                    
                if metadata.get("type") == "human_post":
                    # Handle content safely - it might be corrupted
                    if isinstance(article.content, str):
                        try:
                            content = json.loads(article.content)
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            # Content might be plain text
                            content = {"text": article.content}
                    else:
                        continue
                        
                    text = content.get("text", "").strip()
                    if text and len(text) > 10:  # Only include meaningful posts
                        posts.append(text)
            except (json.JSONDecodeError, KeyError, UnicodeDecodeError, TypeError) as e:
                logging.debug("Skipping corrupted human post record: %s", e)
                continue
        
        if posts:
            logging.info("Loaded %d human posts from SQL database", len(posts))
            return posts
            
    except Exception as exc:
        logging.debug("Failed to load human posts from SQL: %s", exc)
    
    if posts:
        return posts

    # Fallback to legacy JSONL exports if SQL doesn't have data
    fallback_candidates: List[str] = []
    normalised_path = path or ""
    base_dir = os.path.dirname(normalised_path) or "data"

    if normalised_path.lower().endswith(".sqlite"):
        stem, _ = os.path.splitext(normalised_path)
        fallback_candidates.extend([
            f"{stem}.jsonl",
            f"{stem}.json",
            os.path.join(base_dir, "streamed_posts.jsonl"),
            os.path.join(base_dir, "streamed_posts.json"),
        ])
    elif normalised_path:
        fallback_candidates.append(normalised_path)

    # Always consider the default legacy path as a last resort
    fallback_candidates.extend([
        os.path.join("data", "streamed_posts.jsonl"),
        os.path.join("data", "streamed_posts.json"),
    ])

    deduped_candidates: List[str] = []
    seen_paths: set[str] = set()
    for candidate in fallback_candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        normalised = os.path.normpath(candidate)
        if normalised in seen_paths:
            continue
        seen_paths.add(normalised)
        deduped_candidates.append(candidate)

    loaded_from: Optional[str] = None
    for candidate in deduped_candidates:
        try:
            current_posts: List[str] = []
            with open(candidate, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = (data.get("text") or "").strip()
                    if text:
                        current_posts.append(text)
            if current_posts:
                posts = current_posts
                loaded_from = candidate
                logging.info("Loaded %d human posts from %s", len(posts), candidate)
                break
        except FileNotFoundError:
            logging.debug("Human posts fallback file not found: %s", candidate)
        except UnicodeDecodeError as exc:
            logging.warning("Failed to decode human posts from %s: %s", candidate, exc)
        except Exception as exc:
            logging.warning("Failed to load human posts from %s: %s", candidate, exc)

    if not posts:
        return posts

    # Migrate legacy posts to SQL for future runs
    try:
        from datetime import datetime

        from .storage import SQLDataStore
        from .storage.sql import ScrapedArticle

        db_path = normalised_path.replace('streamed_posts.json', 'spider_guardian.sqlite').replace('data/', 'data/')
        if 'spider_guardian.sqlite' not in db_path:
            db_path = 'data/spider_guardian.sqlite'

        sql_store = SQLDataStore(db_path)
        articles = []
        for i, text in enumerate(posts):
            articles.append(ScrapedArticle(
                link=f"human-post://{i}",
                title=text[:100],
                content=json.dumps({"text": text}),
                metadata={"type": "human_post"},
                created_at=datetime.utcnow(),
            ))

        count = sql_store.upsert_scraped_articles(articles)
        logging.info("Migrated %d human posts from JSON to SQL", count)

        if loaded_from and os.path.exists(loaded_from):
            try:
                os.rename(loaded_from, loaded_from + ".migrated")
            except OSError as exc:
                logging.warning("Failed to rename human posts file %s: %s", loaded_from, exc)

    except Exception as migrate_exc:
        logging.warning("Failed to migrate human posts to SQL: %s", migrate_exc)

    return posts


def build_human_post_index(posts: Sequence[str], embedder: SentenceTransformer) -> HumanPostIndex:
    return HumanPostIndex(posts, embedder)
