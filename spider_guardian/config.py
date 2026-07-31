"""Configuration dataclasses for Spider Guardian."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass
class ChatProviderConfig:
    """Configuration for a single chat provider."""

    name: str
    model: str
    temperature: float = 0.4
    timeout: float = 30.0


@dataclass
class SpiderGuardianConfig:
    """Top-level configuration for the Spider Guardian bot."""

    dataset_path: str = "data/dataset_Sensationalism.hf"
    embedder_name: str = "sentence-transformers/all-mpnet-base-v2"
    vector_top_k: int = 3
    context_max_chars: int = 160
    context_chunk_overlap: int = 40
    context_max_snippets: int = 6
    human_posts_path: str = "data/spider_guardian.sqlite"  # Now loads from SQL
    human_posts_top_k: int = 2
    human_style_examples: int = 2
    reply_min_words: int = 12
    reply_max_words: int = 50
    num_candidates: int = 1
    max_feedback_iterations: int = 1

    sql_database_path: str = "data/spider_guardian.sqlite"
    article_store_path: str = "data/articles.json"

    twitter_transport: str = "selenium"
    selenium_headless: bool = True
    selenium_driver: str = "chrome"
    selenium_wait_seconds: int = 20

    author_wait_seconds: int = 60  # Default 1 minutes between author profile requests

    providers: Sequence[ChatProviderConfig] = field(
        default_factory=lambda: (ChatProviderConfig(name="local", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"),)
    )

    twitter_query: str = "(spider OR spiders OR arachnid) lang:en -is:retweet"
    max_daily_replies: int = 15
    min_seconds_between_replies: int = 1200
    seed: Optional[int] = 7
    # Scoring weights for reply selection
    score_weight_followers: float = 10.0  # Primary: high-reach author
    score_weight_impressions: float = 0.0  # Secondary: already visible (set to 0 to ignore)
    score_weight_engagement: float = 0.0   # Tertiary: engaging audience (set to 0 to ignore)
