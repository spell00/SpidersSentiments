from __future__ import annotations

from click import prompt

from .langsmith.simple import push_reply_to_dataset
# from .langsmith.simple import push_reply_to_dataset as push_top_post_to_dataset
# from .langsmith.simple import push_reply_to_dataset as push_my_tweet_to_dataset
"""Core Spider Guardian bot orchestration without legacy script dependencies."""


import csv
import datetime
import json
import logging
import os
import random
import re
import threading
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple
import sqlite3

import numpy as np
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

from .config import SpiderGuardianConfig
from .feedback import FeedbackModel
from .human_posts import HumanPostIndex, build_human_post_index, load_human_posts
from .providers import ChatProvider, build_chat_providers
from .twitter_client import SeleniumTwitterClient, SocialPost
from .storage import TrendingStore, TrendingPost, SQLDataStore
from .vector_index import VectorIndex
from .telemetry import telemetry
from .langsmith.simple import push_reply_to_dataset
from spider_guardian.langsmith import langsmith_integration

# Import LangSmith integration
try:
    from .langsmith import log_reply_generation, log_engagement_metrics, log_sentiment_analysis
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    log_reply_generation = log_engagement_metrics = log_sentiment_analysis = None
    logging.warning("LangSmith integration not available")

# Import image analysis
try:
    from .image_analysis import image_analyzer, create_image_aware_prompt, ImageAnalysisResult
    IMAGE_ANALYSIS_AVAILABLE = True
except ImportError:
    IMAGE_ANALYSIS_AVAILABLE = False
    logging.warning("Image analysis not available")


class SpiderGuardianBot:
    """Encapsulates context retrieval, prompt building, and Selenium reply loop."""

    def __init__(self, config: SpiderGuardianConfig) -> None:
        self.config = config
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)

        self.vector_index: Optional[VectorIndex] = None
        if os.path.exists(config.dataset_path):
            self.vector_index = VectorIndex(config.embedder_name)

        self.feedback_model = FeedbackModel()
        self.sentiment = SentimentIntensityAnalyzer()
        self.sql_store = SQLDataStore(config.sql_database_path)
        # Initialize on first use - SQLDataStore constructor creates tables automatically
        logging.info("SQL database initialized: %s", config.sql_database_path)
        
        self.providers: List[ChatProvider] = build_chat_providers(config.providers)
        if not self.providers:
            raise RuntimeError("No providers configured or available.")
        if not any(getattr(provider, "generator", None) is not None for provider in self.providers):
            raise RuntimeError("No providers with text generation capability available.")

        self.twitter_client: Optional[SeleniumTwitterClient] = None
        self.human_posts = load_human_posts(config.human_posts_path)
        self._human_post_index: Optional[HumanPostIndex] = None
        self._embedder: Optional[SentenceTransformer] = None
        self.trending_store = TrendingStore()
        
        # Initialize NoSQL store for articles
        from .storage import ArticleStore
        self.article_store = ArticleStore(config.article_store_path)
        logging.info("Article store initialized: %s", config.article_store_path)

        # Migrate from legacy interactions.json if it exists and SQL is empty
        self._migrate_legacy_interactions()

        logging.info(
            "SpiderGuardian initialised with %d provider(s): %s",
            len(self.providers),
            [provider.config.name for provider in self.providers],
        )

        # Add a configuration option for the number of candidates to generate before choosing the best one
        self.config.num_candidates = getattr(config, "num_candidates", 1)
        # Ensure reply_max_words is also set in the configuration
        self.config.reply_max_words = getattr(config, "reply_max_words", 24)

    def _migrate_legacy_interactions(self) -> None:
        """One-time migration from interactions.json to SQLite."""
        legacy_path = "data/interactions.json"  # Fixed path since config no longer has this
        if not os.path.exists(legacy_path):
            return
        try:
            # Check if we already have data in SQL - just check if any records exist
            try:
                with self.sql_store._connect() as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM scraped_articles LIMIT 1")
                    count = cursor.fetchone()[0]
                    if count > 0:
                        return  # Already migrated
            except Exception:
                print('Table might not exist yet, continue with migration')
            
            # Load from JSON and insert into SQL
            records: List[Dict] = []
            with open(legacy_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            
            if records:
                from .storage.sql import ScrapedArticle
                articles = []
                for rec in records:
                    articles.append(ScrapedArticle(
                        link=f"https://x.com/i/status/{rec.get('tweet_id', 'unknown')}",
                        title=rec.get("reply_text", "")[:200],
                        content=json.dumps(rec),
                        created_at=datetime.datetime.fromtimestamp(rec.get("timestamp", time.time())),
                        metadata={k: v for k, v in rec.items() if k not in ("tweet_id", "reply_text", "timestamp")}
                    ))
                count = self.sql_store.upsert_scraped_articles(articles)
                logging.info("Migrated %d interaction records from JSON to SQLite", count)
                # Rename legacy file so we don't re-migrate
                os.rename(legacy_path, legacy_path + ".migrated")
        except Exception as exc:
            logging.warning("Failed to migrate legacy interactions: %s", exc)

    # ------------------------------------------------------------------
    # Bootstrap helpers

    def build_vector_index(self) -> None:
        if self.vector_index is None:
            logging.info("Dataset not found; skipping vector index build.")
            return
        self.vector_index.build(self.config.dataset_path)
        if self.vector_index.labels:
            logging.info("Bootstrapping feedback model with %d entries", len(self.vector_index.labels))
            self.feedback_model.update(self.vector_index.documents, self.vector_index.labels)
        if self.human_posts:
            self._ensure_human_post_index()

    def ensure_twitter_client(self) -> None:
        if self.twitter_client is None:
            self.twitter_client = SeleniumTwitterClient(self.config)

    # ------------------------------------------------------------------
    # Retrieval helpers

    def _get_embedder(self) -> Optional[SentenceTransformer]:
        if self.vector_index and getattr(self.vector_index, "embedder", None) is not None:
            return self.vector_index.embedder
        if self._embedder is None:
            try:
                self._embedder = SentenceTransformer(self.config.embedder_name)
            except Exception as exc:
                logging.warning("Unable to load embedder '%s': %s", self.config.embedder_name, exc)
                self._embedder = None
        return self._embedder

    def _ensure_human_post_index(self) -> None:
        if not self.human_posts:
            self._human_post_index = None
            return
        if self._human_post_index is None:
            embedder = self._get_embedder()
            if embedder is None:
                logging.info("Skipping human post index: embedder unavailable")
                return
            self._human_post_index = build_human_post_index(self.human_posts, embedder)

    def _retrieve_article_context(self, text: str, limit: int) -> List[str]:
        if limit <= 0:
            return []
        if self.vector_index is not None:
            try:
                results = self.vector_index.search(text, top_k=limit)
                return [doc for doc, _ in results]
            except Exception as exc:
                logging.warning("Vector index search failed: %s", exc)
        csv_path = "data/Data_spider_news_global.csv"
        try:
            lines: List[str] = []
            with open(csv_path, encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    for col in ("text", "article", "content", "body"):
                        val = (row.get(col) or "").strip()
                        if val:
                            lines.append(val)
                            break
                if not lines:
                    handle.seek(0)
                    raw_reader = csv.reader(handle)
                    for row in raw_reader:
                        for val in row:
                            val = (val or "").strip()
                            if val:
                                lines.append(val)
            sample_size = min(limit, len(lines))
            if sample_size <= 0:
                return []
            return random.sample(lines, sample_size)
        except FileNotFoundError:
            logging.info("Fallback CSV not found: %s", csv_path)
        except Exception as exc:
            logging.warning("Could not retrieve context from CSV: %s", exc)
        return []

    def _log_image_analysis_event(self, post: SocialPost, image_url: str, analysis: ImageAnalysisResult) -> None:
        """Persist image analysis metadata for later auditing."""

        log_path = Path("data/image_analysis_events.jsonl")
        taxonomy_dict = asdict(analysis.taxonomy) if analysis.taxonomy else None
        if taxonomy_dict:
            has_signal = any(
                value not in (None, "", 0.0)
                for key, value in taxonomy_dict.items()
                if key not in {"source", "confidence"}
            )
            if not has_signal:
                taxonomy_dict = None
        taxonomy_source = taxonomy_dict.get("source") if taxonomy_dict else None

        record = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "tweet_id": post.id,
            "conversation_id": post.conversation_id,
            "image_url": image_url,
            "description": analysis.description,
            "spider_detected": analysis.spider_detected,
            "detection_confidence": round(analysis.confidence, 4),
            "species_suggestion": analysis.species_suggestion,
            "danger_level": analysis.danger_level,
            "taxonomy": taxonomy_dict,
            "taxonomy_source": taxonomy_source,
            "animal_verification": analysis.animal_verification,
            "objects_detected": analysis.objects_detected,
            "analysis_source": (
                "iNaturalist" if taxonomy_source == "inaturalist" else ("heuristic" if taxonomy_dict else "none")
            ),
            "human_verdict": "NA",
        }

        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                json.dump(record, handle, ensure_ascii=True)
                handle.write("\n")
        except Exception as exc:  # pragma: no cover - best effort logging
            logging.debug("Image analysis logging skipped: %s", exc)

    def _retrieve_human_posts(self, text: str, limit: int) -> List[str]:
        if limit <= 0 or not self.human_posts:
            return []
        self._ensure_human_post_index()
        if self._human_post_index is not None:
            try:
                return self._human_post_index.search(text, limit)
            except Exception as exc:
                logging.warning("Human post similarity search failed: %s", exc)
        sample_size = min(limit, len(self.human_posts))
        if sample_size <= 0:
            return []
        return random.sample(self.human_posts, sample_size)

    def retrieve_context(self, text: str) -> List[Tuple[str, str]]:
        contexts: List[Tuple[str, str]] = []
        article_limit = max(0, getattr(self.config, "vector_top_k", 0))
        human_limit = max(0, getattr(self.config, "human_posts_top_k", 0))
        if article_limit > 0:
            contexts.extend(("article", doc) for doc in self._retrieve_article_context(text, article_limit))
        if human_limit > 0:
            contexts.extend(("human", doc) for doc in self._retrieve_human_posts(text, human_limit))
        # Include a couple of trending snippets to steer tone, when available
        try:
            trending_samples = [p.text for p in self.trending_store.top(limit=2, since_hours=24)]
            for t in trending_samples:
                contexts.append(("human", t))
        except Exception:
            pass
        return contexts

    @staticmethod
    def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
        cleaned = " ".join((text or "").split())
        if not cleaned:
            return []
        chunk_size = max(1, chunk_size)
        overlap = max(0, overlap)
        words = cleaned.split()
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0
        for word in words:
            if len(word) > chunk_size:
                if current:
                    chunks.append(" ".join(current))
                    current = []
                    current_len = 0
                for i in range(0, len(word), chunk_size):
                    chunks.append(word[i : i + chunk_size])
                continue
            addition = len(word) if not current else len(word) + 1
            if current and current_len + addition > chunk_size:
                chunks.append(" ".join(current))
                if overlap > 0 and current:
                    overlap_words: List[str] = []
                    overlap_len = 0
                    for existing in reversed(current):
                        add = len(existing) if not overlap_words else len(existing) + 1
                        if overlap_len + add > overlap:
                            break
                        overlap_words.insert(0, existing)
                        overlap_len += add
                    current = overlap_words + [word]
                    current_len = sum(len(w) for w in current) + max(0, len(current) - 1)
                else:
                    current = [word]
                    current_len = len(word)
            else:
                current.append(word)
                current_len += addition
        if current:
            chunks.append(" ".join(current))
        return [chunk for chunk in chunks if chunk]

    # ------------------------------------------------------------------
    # Prompt building + generation

    def classify_tone(self, text: str) -> str:
        scores = self.sentiment.polarity_scores(text)
        hostility = self.feedback_model.predict_hostility(text)
        
        # Log sentiment analysis to LangSmith
        if LANGSMITH_AVAILABLE:
            try:
                log_sentiment_analysis(
                    text=text,
                    sentiment_scores={
                        "positive": scores["pos"],
                        "negative": scores["neg"],
                        "neutral": scores["neu"],
                        "compound": scores["compound"],
                        "hostility": hostility
                    }
                )
            except Exception as e:
                logging.warning(f"Failed to log sentiment to LangSmith: {e}")
        
        if scores["neg"] > 0.45 or hostility > 0.6:
            return "deescalate"
        if scores["pos"] > 0.4 and hostility < 0.4:
            return "celebrate"
        return "neutral"

    def build_prompt(self, tweet_text: str, context_documents: Sequence[Tuple[str, str]], tone: str, post: Optional[SocialPost] = None) -> str:
        tones = {
            "deescalate": "Sound like a friendly commenter calming nerves with a quick personal take.",
            "celebrate": "Sound like an excited fan reacting with delight and playful energy.",
            "neutral": "Sound like a curious human sharing a neat observation in a conversational way.",
        }
        chunk_size = max(80, getattr(self.config, "context_max_chars", 160))
        overlap = max(0, getattr(self.config, "context_chunk_overlap", 0))
        max_snippets = max(1, getattr(self.config, "context_max_snippets", 6))

        ctx_chunks: List[Tuple[str, str]] = []
        style_samples: List[str] = []
        for source, doc in context_documents:
            for chunk in self._chunk_text(doc, chunk_size, overlap):
                label = "Article" if source == "article" else "Human"
                ctx_chunks.append((label, chunk))
                if label == "Human" and len(style_samples) < max(1, getattr(self.config, "human_style_examples", 2)):
                    style_samples.append(chunk)
                if len(ctx_chunks) >= max_snippets:
                    break
            if len(ctx_chunks) >= max_snippets:
                break

        if not style_samples:
            fallback = self._retrieve_human_posts(
                tweet_text, max(1, getattr(self.config, "human_style_examples", 2))
            )
            style_samples.extend(fallback)
        style_samples = style_samples[: max(1, getattr(self.config, "human_style_examples", 2))]

        ctx_block = (
            "\n".join(f"- [{label}] {chunk}" for label, chunk in ctx_chunks)
            if ctx_chunks
            else "- (no additional context snippets)"
        )
        style_block = (
            "\n".join(f"• {sample}" for sample in style_samples)
            if style_samples
            else "• (no human tone samples available)"
        )
        fillers = ["honestly", "wow", "hey", "ngl", "for real", "lol"]
        filler_hint = random.choice(fillers)
        min_words = max(6, getattr(self.config, "reply_min_words", 12))
        max_words = max(min_words + 2, getattr(self.config, "reply_max_words", 24))

        # Add image analysis context if available
        image_context = ""
        if post and post.image_urls and IMAGE_ANALYSIS_AVAILABLE:
            try:
                image_analyses = []
                for img_url in post.image_urls[:2]:  # Limit to 2 images max
                    analysis = image_analyzer.analyze_image_from_url(img_url)
                    if analysis:
                        if not analysis.image_url:
                            analysis.image_url = img_url
                        image_analyses.append(analysis)
                        self._log_image_analysis_event(post, analysis.image_url or img_url, analysis)
                
                if image_analyses:
                    image_context = create_image_aware_prompt(tweet_text, image_analyses, context_documents)
            except Exception as e:
                logging.warning(f"Failed to analyze images: {e}")

        return (
            "You are Spider cool advocate who replies to posts in a fire way.\n"
            "Your replies are short, sharp, and playful. You need to be cool and confident. Avoid being overly formal or verbose. Not too much hey there! And stuff like that\n\n"
            f"Tone: {tones.get(tone, tones['neutral'])}\n\n"
            "Use this context if helpful:\n"
            f"{ctx_block}\n\n"
            + (f"{image_context}\n\n" if image_context else "") +
            "Human tone cues from recent X posts (do NOT copy them):\n"
            f"{style_block}\n\n"
            "Rules:\n"
            "- CRITICAL: Do NOT copy, quote, or repeat ANY part of the original post below. BE COOL.\n"
            "- Create completely original content - no phrases, sentences, or words from the original.\n"
            "- Do not announce events, meetups, or anything requiring a real person to attend.\n"
            "- Do not invite people to DM/call or imply you can meet in real life.\n"
            "- Keep it conversational and human, without mentioning being an AI/bot.\n"
            "- Don't start with a greeting. Just respond naturally.\n"
            "- Don't respond to things that you could not understand the context. If it's about things that you can't know, skip it.\n"
            "- Be factual; avoid claims you cannot know or verify.\n\n"
            "Blend factual snippets with that vibe so the reply feels like a real person talking.\n"
            f"Reply in {min_words} to {max_words} words. Use contractions, vary sentence length, and drop one casual filler like \"{filler_hint}\" only if it fits naturally.\n"
            "No cut mid sentence. Avoid meta or instructions.\n\n"
            f"Post: {tweet_text}"
        )

    @staticmethod
    def _strip_label_prefix(text: str) -> str:
        candidate = (text or "").strip()
        # Remove colon-style prefixes like "Reply: ...", "Assistant: ..."
        while True:
            match = re.match(r"^\s*([A-Za-z][\w\s]{0,40}?):\s*(.+)$", candidate)
            if not match:
                break
            label, remainder = match.groups()
            words = label.strip().split()
            if 0 < len(words) <= 3:
                candidate = remainder.strip()
            else:
                break
        # Remove bracketed tags like [Reply], [Assistant], [Response]
        while True:
            match_b = re.match(r"^\s*\[([A-Za-z][^\]]{0,20})\]\s*(.+)$", candidate)
            if not match_b:
                break
            tag, remainder = match_b.groups()
            # Only strip common meta tags
            if tag.lower().split()[0] in {"reply", "assistant", "response", "system", "bot", "user"}:
                candidate = remainder.strip()
            else:
                break
        return candidate.strip()

    def _is_reply_suitable(self, reply: str, prompt: str, original_tweet: str = "") -> bool:
        logging.info("Checking reply suitability: %r", reply)
        if not reply:
            logging.info("Reply rejected: empty")
            return False
        reply = reply.lstrip()
        while reply and reply[0] in "!? .:;,-":
            reply = reply[1:]
        reply = reply.lstrip()
        words = reply.split()
        min_words = max(3, getattr(self.config, "reply_min_words", 12))
        max_words = max(min_words + 2, getattr(self.config, "reply_max_words", 24))
        logging.info("Word count check: %d words (min: %d, max: %d)", len(words), min_words, max_words + 3)
        if len(words) < min_words or len(words) > max_words + 3:
            logging.info("Reply rejected: word count out of range")
            return False
        lower = reply.lower()
        
        # Check for copying from original tweet - prevent any substantial overlap
        if original_tweet:
            logging.info("Checking for copying from original: %r", original_tweet)
            original_lower = original_tweet.lower()
            original_words = original_lower.split()
            reply_words = lower.split()
            
            # Check for consecutive word matches (phrases of 4+ words)
            for i in range(len(reply_words) - 3):
                reply_phrase = " ".join(reply_words[i:i+4])
                if reply_phrase in original_lower:
                    logging.warning("Reply contains copied phrase from original tweet: '%s'", reply_phrase)
                    return False
            
            # Check for high word overlap (>40% of reply words appear in original)
            overlap_count = sum(1 for word in reply_words if word in original_words and len(word) > 3)
            overlap_ratio = overlap_count / max(len(reply_words), 1)
            logging.info("Word overlap check: %d/%d = %.2f", overlap_count, len(reply_words), overlap_ratio)
            if overlap_ratio > 0.4:
                logging.warning("Reply has too much word overlap with original tweet: %.2f", overlap_ratio)
                return False
        
        bad_phrases = (
            "advertise",
            "todosendowhat",
            "best friends and family",
            "i'm sorry",
            "assistant:",
            "reply:",
            "answer:",
            "as an ai",
            "i cannot",
            "todo",
            "send",
            "do what",
            # human-identity revealing or deception-prone invitations
            "join me",
            "come to",
            "see you at",
            "meet at",
            "dm me",
            "my show",
            "live tonight",
            "tickets",
            "rsvp",
            "on stage",
            "book signing",
            "meet and greet",
            "but remember:",
        )
        if any(bad in lower for bad in bad_phrases):
            return False
        if lower.startswith("post:") or reply in prompt.lower():
            return False
        if reply in [".", "!", "?", "..."]:
            return False
        if "context:" in lower:
            logging.info("Reply rejected: contains 'context:'")
            return False
        logging.info("Reply passed all suitability checks ✓")
        return True

    def generate_reply_to_replies(self, conversation: dict[str, str], original_tweet: str = "") -> Optional[str]:
        """
        Generate a reply-to-replies based on the conversation history.
        """
        logging.info("=== Starting reply-to-replies generation ===")
        logging.info("Original tweet: %r", original_tweet)
        logging.info("Conversation len: %d", len(conversation))
        candidates: List[str] = []
        lock = threading.Lock()
        started_at = time.time()
        
        # Build prompt from conversation history
        conversation_text = "\n".join(f"{role}: {text}" for role, text in conversation.items())
        prompt = f"Continue this conversation naturally:\n{conversation_text}\nYou:"

        for provider in self.providers:
            logging.info("Trying provider: %s", provider.config.name)
            for attempt in range(10):
                try:
                    with lock:
                        try:
                            response = provider.generator(prompt)
                        except Exception as e:
                            logging.error(f"Error generating response: {e}")
                            response = None
                    if response:
                        normalized = " ".join(response.split())
                        if self._is_reply_suitable(normalized, prompt, original_tweet):
                            candidates.append(normalized)
                            break

                except Exception as e:
                    logging.warning("Provider %s failed on attempt %d: %s", provider.config.name, attempt + 1, e)

        if not candidates:
            logging.error("=== NO CANDIDATES GENERATED - Reply generation failed ===")
            return None

        logging.info("=== Got %d candidates ===", len(candidates))

        # Pick the first candidate for reply-to-replies
        best = candidates[0]
        logging.info("Selected reply-to-replies: %r", best)

        # Telemetry best-effort
        try:
            latency_ms = int((time.time() - started_at) * 1000)
            logging.info("Reply-to-replies generation latency: %d ms", latency_ms)
        except Exception:
            pass

        return best

    def generate_reply(self, prompt: str, original_tweet: str = "") -> Optional[str]:
        logging.info("=== Starting reply generation ===")
        logging.info("Original tweet: %r", original_tweet)
        logging.info("Prompt length: %d", len(prompt))
        # conversation: List[Dict[str, str]] = []
        candidates: List[str] = []
        lock = threading.Lock()
        started_at = time.time()

        def normalize(gen_text: str) -> str:
            reply = " ".join(gen_text.split())
            if reply and reply[-1] not in ".!?":
                last = max(reply.rfind("."), reply.rfind("!"), reply.rfind("?"))
                if last != -1:
                    reply = reply[:last + 1]
            reply = "".join(ch for ch in reply if ord(ch) <= 0xFFFF)
            reply = self._strip_label_prefix(reply)
            # Remove stray leading brackets such as "[She]" that occasionally leak from model output
            reply = re.sub(r"^\[([^\]\s]{1,30})\]", r"\1", reply)
            # Remove leading or trailing quotes
            if reply.startswith(("\"", "'")):
                reply = reply[1:]
            if reply.endswith(("\"", "'")):
                reply = reply[:-1]
            return reply

        for provider in self.providers:
            logging.info("Trying provider: %s", provider.config.name)
            for attempt in range(10):
                try:
                    with lock:
                        # Directly use the prompt for the first-time reply
                        try:
                            response = provider.generator(prompt)
                        except Exception as e:
                            logging.error(f"Error generating response: {e}")
                            response = None
                    if response:
                        try:
                            normalized = normalize(response)
                        except Exception as e:
                            normalized = normalize(response[0]['generated_text'])
                        
                        # Try normalized reply as-is first, then try stripping common prefixes
                        prefix_patterns = ['Reply: ', 'reply: ', 'Response: ', 'response: ']
                        candidates_to_try = [normalized] + [normalize(normalized.split(prefix)[-1]) for prefix in prefix_patterns]
                        
                        try:
                            normalized = normalize(response)
                        except Exception as e:
                            normalized = normalize(response[0]['generated_text'])
                        for candidate in candidates_to_try:
                            if self._is_reply_suitable(candidate, prompt, original_tweet):
                                candidates.append(candidate)
                                break
                        
                        # If we found a suitable candidate, break out of the attempt loop
                        if candidates:
                            break

                except Exception as e:
                    logging.warning("Provider %s failed on attempt %d: %s", provider.config.name, attempt + 1, e)

        if not candidates:
            logging.error("=== NO CANDIDATES GENERATED - Reply generation failed ===")
            return None

        logging.info("=== Got %d candidates ===", len(candidates))

        # Pick the candidate closest to target length
        min_words = max(6, getattr(self.config, "reply_min_words", 12))
        max_words = max(min_words + 2, getattr(self.config, "reply_max_words", 50))
        target = 0.5 * (min_words + max_words)

        def score(text: str) -> float:
            wc = len(text.split())
            return -abs(wc - target)

        candidates.sort(key=score, reverse=True)
        best = candidates[0]
        logging.info("Selected reply: %r", best)

        # Tone matching adjustment
        tone = self.classify_tone(original_tweet)
        if tone == "deescalate" and "!" in best:
            best = best.replace("!", ".")
        elif tone == "celebrate" and not any(punct in best for punct in "!?"):
            best += "!"

        # Fact-check enhancement: Add spider facts if space allows
        try:
            from .fact_check import SpiderFactChecker
            fact_checker = SpiderFactChecker()
            enhanced = fact_checker.enhance_reply_with_facts(best, original_tweet)
            if enhanced != best and len(enhanced) <= 280:
                logging.info("Enhanced reply with spider facts: %r", enhanced)
                best = enhanced
        except Exception as e:
            logging.warning("Failed to enhance reply with facts: %s", e)

        # Paraphrasing step
        # best = self._paraphrase_reply(best)

        # Telemetry best-effort
        try:
            latency_ms = int((time.time() - started_at) * 1000)
            logging.info("Reply generation latency: %d ms", latency_ms)
        except Exception:
            pass

        return best

    def _paraphrase_reply(self, reply: str) -> str:
        """Paraphrase the reply to ensure originality and natural tone."""
        # Placeholder for paraphrasing logic - could integrate with a paraphrasing model or library
        return reply.replace("you", "one").replace("I", "we")

    # ------------------------------------------------------------------
    # Feedback + logging helpers

    def label_feedback(self, text: str) -> int:
        scores = self.sentiment.polarity_scores(text)
        return int(scores["neg"] >= 0.5)

    def process_feedback(self, replies: Sequence[SocialPost]) -> None:
        texts = [reply.text for reply in replies if (reply.lang or "en").startswith("en")]
        if not texts:
            return
        labels = [self.label_feedback(text) for text in texts]
        self.feedback_model.update(texts, labels)

    def log_flagged_reply(self, reply_text: str, reason: str, metadata: Optional[Dict] = None) -> None:
        """Log replies that received bad feedback or were flagged as bot-like."""
        from .storage.sql import ScrapedArticle
        try:
            record = ScrapedArticle(
                link="",  # No specific link for flagged replies
                title=reply_text[:200],
                content=json.dumps({
                    "reply_text": reply_text,
                    "reason": reason,
                    "metadata": metadata or {}
                }),
                created_at=datetime.datetime.utcnow(),
                metadata={"type": "flagged_reply", "reason": reason}
            )
            self.sql_store.upsert_scraped_articles([record])
            logging.info("Flagged reply logged: %s", reply_text)
        except Exception as exc:
            logging.error("Failed to log flagged reply: %s", exc)

    # ------------------------------------------------------------------
    # Public actions

    def stream_spider_posts(self, interval: int = 60, log_path: Optional[str] = None) -> None:
        self.ensure_twitter_client()
        assert self.twitter_client is not None
        seen: set[str] = set()
        max_searches = 1  # Limit to prevent rate limiting
        searches_done = 0
        max_posts_per_search = 10  # Don't store too many posts per search
        
        # We'll store streamed posts in SQL instead of JSON file
        try:
            logging.info("Streaming recent posts about spiders (max %d searches). Press Ctrl+C to stop.", max_searches)
            while searches_done < max_searches:
                try:
                    posts = self.twitter_client.search_posts(self.config.twitter_query)
                    searches_done += 1
                    logging.info("Search %d/%d: Found %d posts", searches_done, max_searches, len(posts))
                    
                    new_posts = [post for post in posts if post.id not in seen]
                    # Limit posts per search to avoid overwhelming storage
                    new_posts = new_posts[:max_posts_per_search]
                    
                    stored_count = 0
                    for post in new_posts:
                        logging.info("[%s] %s", post.id, post.text)
                        seen.add(post.id)
                        
                        # Save to trending store with short retention
                        try:
                            self.trending_store.upsert(
                                [
                                    TrendingPost(
                                        post_id=str(post.id),
                                        text=post.text or "",
                                        author=post.author_handle,
                                        lang=getattr(post, "lang", "en"),
                                        like_count=int(getattr(post, "like_count", 0) or 0),
                                        repost_count=int(getattr(post, "repost_count", 0) or 0),
                                        reply_count=int(getattr(post, "reply_count", 0) or 0),
                                        url=f"https://x.com/i/status/{post.id}",
                                        collected_at=datetime.datetime.utcnow(),
                                        post_created_at=None,
                                        impression_count=int(getattr(post, "impression_count", 0) or 0)
                                    )
                                ]
                            )
                            # purge older than configured short retention (defaults to 3 days here)
                            self.trending_store.purge_older_than_days(getattr(self.config, "trending_retention_days", 3))
                        except Exception as exc:
                            logging.debug("Trending store upsert failed: %s", exc)
                        
                        # Store streamed post in SQL database (replaces log_file JSON writing)
                        try:
                            from .storage.sql import ScrapedArticle
                            streamed_record = {
                                "id": post.id,
                                "text": post.text,
                                "conversation_id": post.conversation_id,
                                "lang": post.lang,
                                "author_handle": post.author_handle,
                                "timestamp": time.time(),
                                "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            }
                            article = ScrapedArticle(
                                link=f"https://x.com/i/status/{post.id}",
                                title=(post.text or "")[:200],
                                content=json.dumps(streamed_record),
                                created_at=datetime.datetime.fromtimestamp(streamed_record["timestamp"]),
                                metadata={"type": "streamed_post", "lang": post.lang or "en"}
                            )
                            self.sql_store.upsert_scraped_articles([article])
                            stored_count += 1
                        except Exception as exc:
                            logging.debug("Failed to store streamed post in SQL: %s", exc)
                    
                    if stored_count > 0:
                        logging.info("Stored %d new posts in databases", stored_count)
                    
                    if not new_posts:
                        logging.info("No new posts found in search %d/%d", searches_done, max_searches)
                    
                    # Random wait between searches to avoid rate limiting
                    if searches_done < max_searches:
                        # Random interval between 45-90 seconds, with base interval as minimum
                        random_wait = random.randint(45, 90)
                        wait_time = max(interval, random_wait)
                        logging.info("Waiting %ds before next search (%d/%d)", wait_time, searches_done, max_searches)
                        time.sleep(wait_time)
                        
                except Exception as exc:
                    logging.error("Search %d failed: %s", searches_done, exc)
                    searches_done += 1
                    # Count failed searches to avoid infinite loops
                    if searches_done < max_searches:
                        # Longer wait after errors
                        error_wait = random.randint(120, 180)
                        logging.info("Error encountered, waiting %ds before retry", error_wait)
                        time.sleep(error_wait)
                        
            logging.info("Streaming completed after %d searches (limit reached)", max_searches)
        finally:
            logging.info("Streaming stopped")

    def respond_to_tweets(
        self,
        limit: int = 0,
        test_one_word_reply: bool = False,
        reply_to_replies: bool = False,
        first_tweet_only: bool = False,
    ) -> None:
        self.ensure_twitter_client()
        assert self.twitter_client is not None

        sent = 0
        last_reply_time: Optional[float] = None
        
        # Get follower/impression filters from config (if set via CLI)
        min_followers = getattr(self.config, "min_followers", None)
        max_followers = getattr(self.config, "max_followers", None)
        min_impressions = getattr(self.config, "min_impressions", None)
        max_empty_searches = getattr(self.config, "max_empty_searches", 5)  # Default to 5 if not set
        
        if min_followers is not None or max_followers is not None or min_impressions is not None:
            logging.info("🎯 Filtering enabled - min_followers=%s max_followers=%s min_impressions=%s",
                        min_followers, max_followers, min_impressions)
            logging.info("🔄 Will retry up to %d times if no qualifying tweets found", max_empty_searches)
        
        # Initialize database connection for author tracking
        from .storage import SQLDataStore
        db = SQLDataStore(self.config.sql_database_path)
        
        # Track consecutive failed searches (no qualifying tweets found)
        consecutive_empty_searches = 0
        
        # Track tweet IDs already replied to by the bot (from database)
        replied_tweet_ids = set()
        try:
            from .storage.sql import ScrapedArticle
            for article in self.sql_store.iter_scraped_articles():
                try:
                    record = json.loads(article.content)
                except Exception:
                    continue
                reply_id = record.get("reply_id")
                tweet_id = record.get("tweet_id")
                if reply_id and tweet_id:
                    replied_tweet_ids.add(tweet_id)
        except Exception as exc:
            logging.warning("Could not load replied tweet IDs: %s", exc)

        while True:
            tweets = self.twitter_client.search_posts(self.config.twitter_query)
            logging.info("Fetched %d candidate tweet(s)", len(tweets))

            if not tweets:
                logging.warning("No tweets found matching query. Waiting 60 seconds before retry...")
                time.sleep(60)
                continue

            # --- FIRST TWEET ONLY MODE ---
            if first_tweet_only:
                tweet = tweets[0]
                logging.info("[first_tweet_only] Replying to first tweet: %s (@%s)", tweet.id, tweet.author_handle)
                try:
                    if test_one_word_reply:
                        reply_text = "Guardian"
                        tone = "test"
                    else:
                        tone = self.classify_tone(tweet.text)
                        context_docs = self.retrieve_context(tweet.text)
                        prompt = self.build_prompt(tweet_text=tweet.text, context_documents=context_docs, tone=tone, post=tweet)
                        reply_text = self.generate_reply(prompt, tweet.text)
                        if not reply_text:
                            logging.info("No reply generated for tweet %s", tweet.id)
                            continue

                    logging.info("Replying to %s with tone=%s text=%r", tweet.id, tone, reply_text)
                    reply_id = None
                    try:
                        reply_id = self.twitter_client.reply(reply_text, reply_to_tweet_id=str(tweet.id))
                    except Exception as exc:
                        logging.error("Reply sent but ID resolution failed: %s", exc)

                    record = {
                        "tweet_id": str(tweet.id),
                        "conversation_id": str(tweet.conversation_id),
                        "reply_id": reply_id,
                        "tweet_text": tweet.text,
                        "reply_text": reply_text,
                        "tone": tone,
                        "timestamp": time.time(),
                        "date": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "is_reply": getattr(tweet, "is_reply", False),
                        "model": self.config.providers[0].model if self.config.providers else None,
                    }
                    from .storage.sql import ScrapedArticle
                    article = ScrapedArticle(
                        link=f"https://x.com/i/status/{tweet.id}",
                        title=reply_text[:200],
                        content=json.dumps(record),
                        created_at=datetime.datetime.fromtimestamp(record["timestamp"]),
                        metadata={"type": "interaction", "tone": tone, "model": record.get("model", "")}
                    )
                    self.sql_store.upsert_scraped_articles([article])
                    sent += 1
                    # Log posted reply to LangSmith dataset
                    try:
                        push_reply_to_dataset(
                            tweet_text=tweet.text,
                            author=tweet.author_handle or "unknown",
                            url=f"https://x.com/i/status/{tweet.id}",
                            generated_reply=reply_text,
                            likes=int(getattr(tweet, "like_count", 0) or 0),
                            replies=int(getattr(tweet, "reply_count", 0) or 0),
                            impressions=int(getattr(tweet, "impression_count", 0) or 0),
                            metadata={"stage": "posted_reply"}
                        )
                    except Exception:
                        pass
                    return  # Only reply to the first tweet and exit
                except Exception as exc:
                    logging.error("[first_tweet_only] Failed to reply: %s", exc)
                    return
            
            # Enrich tweets with cached follower counts from database
            # and fetch missing ones if needed
            if min_followers is not None or max_followers is not None:
                from .langsmith.config import langsmith_integration
                
                # First pass: identify which authors need fetching
                authors_to_fetch = []
                for tweet in tweets:
                    # Ensure we have an author handle; resolve from tweet page if missing
                    if not tweet.author_handle:
                        try:
                            resolved = self.twitter_client.resolve_author_handle(tweet.id)
                            if resolved:
                                tweet.author_handle = resolved
                                logging.debug("Resolved author_handle for %s -> @%s", tweet.id, resolved)
                            else:
                                logging.debug("Could not resolve author_handle for %s", tweet.id)
                        except Exception as e:
                            logging.debug("Resolve author_handle failed for %s: %s", tweet.id, e)
                            
                    if not tweet.author_handle:
                        continue
                    
                    # Try to get cached follower count from database
                    info = db.get_author_followers_info(tweet.author_handle)
                    cached_count = info.get("follower_count")
                    checked_iso = info.get("followers_checked_at")
                    needs_refresh = False
                    if checked_iso:
                        try:
                            checked_dt = datetime.datetime.fromisoformat(checked_iso)
                            needs_refresh = (datetime.datetime.utcnow() - checked_dt).days >= 30
                        except Exception:
                            needs_refresh = True
                    else:
                        # Never checked
                        needs_refresh = True if cached_count in (None, -1) else False

                    if cached_count is not None and cached_count != -1 and not needs_refresh:
                        tweet.author_followers = cached_count
                        logging.debug(
                            "Using cached follower count for @%s: %d (checked_at=%s)",
                            tweet.author_handle, cached_count, checked_iso or "never"
                        )
                    else:
                        # Mark for fetching (deduplicate by author handle)
                        if not any(a[0] == tweet.author_handle for a in authors_to_fetch):
                            authors_to_fetch.append((tweet.author_handle, tweet))
                
                # Second pass: batch fetch all uncached authors with progress bar
                if authors_to_fetch:
                    logging.info("📊 Fetching follower counts for %d uncached authors...", len(authors_to_fetch))
                    fetch_start_time = time.time()
                    fetch_timeout_seconds = getattr(self.config, "author_wait_seconds", 600)
                    fetched_count = 0
                    
                    for author_handle, _ in tqdm(authors_to_fetch, desc="Fetching followers", unit="author"):
                        # Check if we've exceeded the timeout
                        elapsed_time = time.time() - fetch_start_time
                        if elapsed_time > fetch_timeout_seconds:
                            logging.warning(
                                "⏱️ Follower fetching timeout reached (%.1f minutes). "
                                "Fetched %d/%d authors. Continuing with cached data only.",
                                elapsed_time / 60, fetched_count, len(authors_to_fetch)
                            )
                            break
                        
                        followers = langsmith_integration.fetch_author_follower_count(author_handle)
                        
                        # Update all tweets from this author
                        for t in tweets:
                            if t.author_handle == author_handle:
                                t.author_followers = followers
                        
                        # Store in database for future use
                        if followers is not None:
                            db.upsert_author(author_handle, follower_count=followers)
                            logging.debug("Cached follower count for @%s: %d (%.1fs elapsed)", 
                                        author_handle, followers, elapsed_time)
                        
                        fetched_count += 1
                    
                    if fetched_count < len(authors_to_fetch):
                        logging.info(
                            "⚠️ Only fetched %d/%d authors due to timeout. Remaining authors will be left as -1 (unknown).",
                            fetched_count, len(authors_to_fetch)
                        )
                
                # Sort by followers (descending) - highest reach first
                tweets = sorted(
                    tweets,
                    key=lambda t: getattr(t, 'author_followers', 0) or 0,
                    reverse=True
                )
                logging.info("Sorted %d tweets by follower count (highest first)", len(tweets))
            
            # Update tweet_count for all authors we see
            for tweet in tweets:
                if tweet.author_handle:
                    db.upsert_author(tweet.author_handle)
            
            # Track if we found any qualifying tweets in this batch
            found_qualifying_tweet = False
            
            # Collect all qualifying tweets instead of replying to the first one
            qualifying_tweets = []
            
            for tweet in tqdm(tweets, desc="Processing tweets", unit="tweet"):
                if limit >= 0 and sent >= limit:
                    logging.info("Requested reply limit reached (%d)", limit)
                    return
                skip_keywords = ("spider man", "spiderman", "spidey")
                author = (tweet.author_handle or "").lower()
                body = (tweet.text or "").lower()
                if any(kw in author for kw in skip_keywords) or any(kw in body for kw in skip_keywords):
                    continue
                if tweet.author_handle and "spider" in tweet.author_handle.lower():
                    continue
                # Prevent replying twice to the same post unless replying to own reply
                if not reply_to_replies:
                    if str(tweet.id) in replied_tweet_ids:
                        logging.debug("Skipping tweet %s - already replied to by bot", tweet.id)
                        continue
                # If replying to replies, allow only if tweet is a reply to a previous bot reply
                if reply_to_replies:
                    # Only reply if this tweet is a reply to a tweet the bot replied to
                    in_reply_to = getattr(tweet, "in_reply_to_status_id", None)
                    if not in_reply_to or str(in_reply_to) not in replied_tweet_ids:
                        continue
                if not reply_to_replies and getattr(tweet, "is_reply", False):
                    continue
                if "spider" not in body:
                    continue
                
                # Apply follower filters
                if min_followers is not None or max_followers is not None:
                    author_followers = getattr(tweet, 'author_followers', None)
                    if author_followers is None:
                        logging.debug("Skipping tweet %s - no follower count available", tweet.id)
                        continue
                    if min_followers is not None and author_followers < min_followers:
                        logging.debug("Skipping @%s (%d followers < %d min)", 
                                    tweet.author_handle, author_followers, min_followers)
                        continue
                    if max_followers is not None and max_followers > 0 and author_followers > max_followers:
                        logging.debug("Skipping @%s (%d followers > %d max)", 
                                    tweet.author_handle, author_followers, max_followers)
                        continue
                    logging.info("✅ @%s passed follower filter (%d followers)", 
                               tweet.author_handle, author_followers)
                
                # Apply impression filter
                if min_impressions is not None:
                    impressions = getattr(tweet, 'impression_count', 0) or 0
                    if impressions < min_impressions:
                        logging.debug("Skipping tweet %s - %d impressions < %d min", 
                                    tweet.id, impressions, min_impressions)
                        continue
                    logging.info("✅ Tweet %s passed impression filter (%d views)", 
                               tweet.id, impressions)

                # This tweet qualifies! Add it to candidates
                qualifying_tweets.append(tweet)
            
            # If we have qualifying tweets, score and select the best one
            if qualifying_tweets:
                found_qualifying_tweet = True
                
                # Score each tweet based on multiple factors
                def score_tweet(tweet) -> float:
                    """Calculate a composite score for tweet reach potential.
                    
                    Factors:
                    - Follower count (primary weight)
                    - Impressions (views the tweet already has)
                    - Engagement rate (likes + replies + reposts relative to impressions)
                    """
                    followers = getattr(tweet, 'author_followers', 0) or 0
                    impressions = getattr(tweet, 'impression_count', 0) or 0
                    likes = getattr(tweet, 'like_count', 0) or 0
                    replies = getattr(tweet, 'reply_count', 0) or 0
                    reposts = getattr(tweet, 'repost_count', 0) or 0

                    engagement_total = likes + replies + reposts
                    engagement_rate = engagement_total / max(impressions, 1)  # Avoid division by zero

                    # Use config weights for scoring
                    score = (
                        followers * self.config.score_weight_followers +
                        impressions * self.config.score_weight_impressions +
                        engagement_total * self.config.score_weight_engagement
                        # (engagement_rate * 10000)  # Bonus: high engagement rate
                    )
                    return score
                
                # Score and sort all qualifying tweets
                scored_tweets = [(score_tweet(t), t) for t in qualifying_tweets]
                scored_tweets.sort(key=lambda x: x[0], reverse=True)
                
                # Log the ranking
                logging.info("📊 Ranked %d qualifying tweets:", len(scored_tweets))
                for i, (score, tweet) in enumerate(scored_tweets[:5]):  # Show top 5
                    followers = getattr(tweet, 'author_followers', 0) or 0
                    impressions = getattr(tweet, 'impression_count', 0) or 0
                    likes = getattr(tweet, 'like_count', 0) or 0
                    logging.info(
                        "  %d. @%s (score: %.0f) - %d followers, %d views, %d likes",
                        i + 1, tweet.author_handle, score, followers, impressions, likes
                    )
                
                # Log all considered authors and their follower counts
                logging.info("Authors considered for reply (handle, followers):")
                for t in qualifying_tweets:
                    handle = getattr(t, 'author_handle', None)
                    followers = getattr(t, 'author_followers', None)
                    logging.info("  @%s: %s followers", handle, followers)
                
                # Select the best tweet
                best_score, best_tweet = scored_tweets[0]
                selected_followers = getattr(best_tweet, 'author_followers', None)
                followers_str = (
                    f"{selected_followers:,}"
                    if isinstance(selected_followers, int) and selected_followers >= 0
                    else "unknown"
                )
                logging.info(
                    "🎯 Selected best tweet: @%s (score: %.0f) — followers: %s",
                    best_tweet.author_handle,
                    best_score,
                    followers_str,
                )
                
                tweet = best_tweet  # Use the best tweet for reply

                try:
                    if test_one_word_reply:
                        reply_text = "Guardian"
                        tone = "test"
                    else:
                        tone = self.classify_tone(tweet.text)
                        context_docs = self.retrieve_context(tweet.text)
                        prompt = self.build_prompt(tweet_text=tweet.text, context_documents=context_docs, tone=tone, post=tweet)
                        reply_text = self.generate_reply(prompt, tweet.text)
                        if not reply_text:
                            logging.info("No reply generated for tweet %s", tweet.id)
                            continue

                    logging.info("Replying to %s with tone=%s text=%r", tweet.id, tone, reply_text)
                    reply_id = None
                    try:
                        reply_id = self.twitter_client.reply(reply_text, reply_to_tweet_id=str(tweet.id))
                    except Exception as exc:  # pragma: no cover - network interactions
                        logging.error("Reply sent but ID resolution failed: %s", exc)

                    record = {
                        "tweet_id": str(tweet.id),
                        "conversation_id": str(tweet.conversation_id),
                        "reply_id": reply_id,
                        "tweet_text": tweet.text,
                        "reply_text": reply_text,
                        "tone": tone,
                        "timestamp": time.time(),
                        "date": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "is_reply": getattr(tweet, "is_reply", False),
                        "model": self.config.providers[0].model if self.config.providers else None,
                    }
                    # Store interaction in SQL database
                    from .storage.sql import ScrapedArticle
                    article = ScrapedArticle(
                        link=f"https://x.com/i/status/{tweet.id}",
                        title=reply_text[:200],
                        content=json.dumps(record),
                        created_at=datetime.datetime.fromtimestamp(record["timestamp"]),
                        metadata={"type": "interaction", "tone": tone, "model": record.get("model", "")}
                    )
                    self.sql_store.upsert_scraped_articles([article])
                    sent += 1

                    # Log posted reply to LangSmith dataset
                    try:
                        push_reply_to_dataset(
                            tweet_text=tweet.text,
                            author=tweet.author_handle or "unknown",
                            url=f"https://x.com/i/status/{tweet.id}",
                            generated_reply=reply_text,
                            likes=int(getattr(tweet, "like_count", 0) or 0),
                            replies=int(getattr(tweet, "reply_count", 0) or 0),
                            impressions=int(getattr(tweet, "impression_count", 0) or 0),
                            metadata={
                                "stage": "posted_reply",
                                "tone": tone,
                                "tweet_id": str(tweet.id),
                                "reply_id": reply_id,
                                "is_reply": getattr(tweet, "is_reply", False),
                                "model": record.get("model", ""),
                            }
                        )
                    except Exception as exc:
                        logging.debug("LangSmith dataset log failed: %s", exc)
                    if LANGSMITH_AVAILABLE:
                        try:
                            # log_engagement_metrics(
                            #     reply_text=reply_text,
                            #     likes=int(getattr(tweet, "like_count", 0) or 0),
                            #     replies=int(getattr(tweet, "reply_count", 0) or 0),
                            #     impressions=int(getattr(tweet, "impression_count", 0) or 0),
                            #     tweet_id=str(tweet.id),
                            #     metadata={
                            #         "stage": "initial_reply",
                            #         "tone": tone,
                            #         "is_reply": getattr(tweet, "is_reply", False),
                            #     },
                            # )
                            # Properly pass a datetime for posted_at (was int earlier)
                            langsmith_integration.log_engagement_metrics(
                                reply_id=str(reply_id or ""),
                                likes=int(getattr(tweet, "like_count", 0) or 0),
                                replies=int(getattr(tweet, "reply_count", 0) or 0),
                                impressions=int(getattr(tweet, "impression_count", 0) or 0),
                                posted_at=datetime.datetime.utcnow(),
                            )
                            # langsmith_integration.log_engagement_metrics(
                            #     reply_id=run_id,
                            #     likes=15,
                            #     replies=3,
                            #     impressions=250,
                            #     posted_at=datetime.now(),
                            # )

                        except Exception as exc:  # pragma: no cover - telemetry best effort
                            logging.debug("LangSmith engagement log failed: %s", exc)
                        # Also log the posted reply to LangSmith for full traceability
                        try:
                            model_name = self.config.providers[0].model if self.config.providers else None
                            log_reply_generation(
                                original_tweet=tweet.text,
                                generated_reply=reply_text,
                                prompt=prompt if not test_one_word_reply else None,
                                model_name=model_name,
                                generation_time_ms=None,
                                metadata={
                                    "stage": "posted_reply",
                                    "tone": tone,
                                    "tweet_id": str(tweet.id),
                                    "reply_id": reply_id,
                                    "is_reply": getattr(tweet, "is_reply", False),
                                },
                            )
                        except Exception as exc:
                            logging.debug("LangSmith reply log failed: %s", exc)

                    now = time.monotonic()
                    if last_reply_time is not None:
                        elapsed = now - last_reply_time
                        wait_time = max(0, self.config.min_seconds_between_replies - elapsed)
                        if wait_time > 0:
                            logging.info("Waiting %.1fs before next reply", wait_time)
                            time.sleep(wait_time)
                    last_reply_time = time.monotonic()
                except Exception as exc:  # pragma: no cover - network interactions
                    logging.error("Failed to reply to %s: %s", tweet.id, exc)
                    continue
            
            # After processing all tweets in this batch, check if we found any qualifying tweets
            if not found_qualifying_tweet:
                consecutive_empty_searches += 1
                logging.warning("No qualifying tweets found in this batch (attempt %d/%d)", 
                              consecutive_empty_searches, max_empty_searches)
                
                if consecutive_empty_searches >= max_empty_searches:
                    if min_followers or max_followers or min_impressions:
                        logging.error(
                            "❌ No qualifying tweets found after %d attempts. Your filters might be too strict:\n"
                            "   min_followers=%s, max_followers=%s, min_impressions=%s\n"
                            "   Consider lowering thresholds or checking if high-reach accounts are tweeting about spiders.",
                            max_empty_searches, min_followers, max_followers, min_impressions
                        )
                    else:
                        logging.error("No qualifying tweets found after %d attempts. Exiting.", max_empty_searches)
                    return
                
                # Wait before retrying to avoid hammering the API
                wait_time = 60 * consecutive_empty_searches  # Progressive backoff: 60s, 120s, 180s
                logging.info("Waiting %d seconds before next search...", wait_time)
                time.sleep(wait_time)
            else:
                # Reset counter if we found qualifying tweets
                consecutive_empty_searches = 0

            if limit < 0:
                logging.debug("Limit < 0; continuing to fetch tweets")
                continue
            if limit == 0:
                logging.info("respond_to_tweets called with limit=0; exiting after one pass")
                return

    def collect_and_learn(self) -> None:
        self.ensure_twitter_client()
        assert self.twitter_client is not None
        # Load interactions from SQL instead of JSON
        i = 0
        for article in tqdm(self.sql_store.iter_scraped_articles(), desc="Processing articles"):
            try:
                record = json.loads(article.content)
            except json.JSONDecodeError:
                continue
            conversation_id = record.get("conversation_id")
            since_id = record.get("reply_id")
            if not conversation_id:
                continue
            try:
                replies = self.twitter_client.fetch_replies(conversation_id, since_id=since_id)
                logging.info(f"Fetched {len(replies)} replies for conversation {conversation_id} successfully")
            except Exception as exc:  # pragma: no cover - Selenium interaction
                logging.warning("Failed to fetch replies for conversation %s: %s", conversation_id, exc)
                continue
            self.process_feedback(replies)
            if i >= self.config.max_feedback_iterations:
                logging.info("Max feedback iterations reached")
                break
            i += 1

    # ------------------------------------------------------------------
    # Trending collection

    def collect_trending(self, hours: int = 24, retention_days: int = 3, mode: str = "top") -> int:
        """Collect trending posts for tone/style and store them with short retention.

        Returns number of posts inserted/updated.
        """
        self.ensure_twitter_client()
        assert self.twitter_client is not None
        posts = self.twitter_client.search_posts(self.config.twitter_query, mode=mode)
        to_upsert: List[TrendingPost] = []
        now = datetime.datetime.utcnow()
        for post in posts:
            sentiment_scores = self.sentiment.polarity_scores(post.text or "")
            hostility = self.feedback_model.predict_hostility(post.text or "")
            to_upsert.append(
                TrendingPost(
                    post_id=str(post.id),
                    text=post.text or "",
                    author=post.author_handle,
                    lang=getattr(post, "lang", "en"),
                    like_count=int(getattr(post, "like_count", 0) or 0),
                    repost_count=int(getattr(post, "repost_count", 0) or 0),
                    reply_count=int(getattr(post, "reply_count", 0) or 0),
                    url=f"https://x.com/i/status/{post.id}",
                    collected_at=now,
                    post_created_at=None,
                )
            )
            try:
                push_reply_to_dataset(
                    tweet_text=post.text or "",
                    author=post.author_handle or "unknown",
                    url=f"https://x.com/i/status/{post.id}",
                    generated_reply="",
                    likes=int(getattr(post, "like_count", 0) or 0),
                    replies=int(getattr(post, "reply_count", 0) or 0),
                    impressions=int(getattr(post, "impression_count", 0) or 0),
                    metadata={
                        "source": "top_post",
                        "lang": getattr(post, "lang", "en"),
                        "mode": mode,
                        "post_id": str(post.id),
                        "collected_at": now.isoformat(),
                        "sentiment": sentiment_scores,
                        "hostility": hostility,
                    },
                    dataset_name="top-tweets-dataset"
                )
            except Exception as exc:
                logging.debug("LangSmith top-tweets-dataset log failed: %s", exc)
        count = self.trending_store.upsert(to_upsert)
        try:
            self.trending_store.purge_older_than_days(retention_days)
        except Exception:
            pass
        return count


    def log_my_tweet(self, text: str, tweet_id: str, url: str, **kwargs) -> None:
        """Log user's own tweet to my-tweets-dataset with metrics."""
        sentiment_scores = self.sentiment.polarity_scores(text)
        hostility = self.feedback_model.predict_hostility(text)
        try:
            push_reply_to_dataset(
                tweet_text=text,
                author="me",
                url=url,
                generated_reply="",
                likes=kwargs.get("likes", 0),
                replies=kwargs.get("replies", 0),
                impressions=kwargs.get("impressions", 0),
                metadata={
                    "source": "my_tweet",
                    "tweet_id": tweet_id,
                    "sentiment": sentiment_scores,
                    "hostility": hostility,
                    **kwargs
                },
                dataset_name="my-tweets-dataset"
            )
        except Exception as exc:
            logging.debug("LangSmith my-tweets-dataset log failed: %s", exc)

    def generate_auto_post(self) -> str:
        """Generate an educational spider-related post automatically."""
        
        # General prompts for AI generation - let the model be creative
        general_prompts = [
            "Write an engaging educational tweet about spiders. Be informative but approachable, dispel common myths, and include relevant emojis. Keep under 280 characters.",
            "Create a fascinating tweet about spider biology or behavior that would interest someone who might be afraid of spiders. Make it positive and educational with emojis.",
            "Generate an interesting tweet about the ecological importance of spiders. Make it engaging and include emojis. Under 280 characters.",
            "Write a tweet that showcases amazing spider abilities or adaptations in an engaging way. Include emojis and keep it conversational.",
            "Create an educational tweet that helps people appreciate spiders better. Focus on their benefits or interesting characteristics. Include emojis."
        ]
        
        # Fallback facts only if AI generation completely fails
        fallback_facts = [
            "Spiders are found on every continent except Antarctica and play crucial roles in controlling pest populations 🕷️🌍",
            "Most spiders are completely harmless to humans and actually help keep our homes free of flying insects 🕷️✨",
            "Spider silk is one of nature's strongest materials - stronger than steel of the same thickness! 🕷️💪"
        ]
        
        # Try AI generation first
        if self.providers:
            try:
                prompt = random.choice(general_prompts)
                generated = self.generate_reply(prompt)
                
                if generated and len(generated.strip()) > 20 and len(generated.strip()) <= 280:
                    return generated.strip()
            except Exception as e:
                logging.debug("Auto-generation failed, falling back to simple facts: %s", e)
        
        # Only use fallback if AI generation fails
        return random.choice(fallback_facts)

    def retrieve_thread(self, conversation_id: str) -> List[Dict[str, str]]:
        """Retrieve the entire thread for a given conversation ID."""
        logging.info("Retrieving thread for conversation ID: %s", conversation_id)
        try:
            thread = self.twitter_client.get_thread(conversation_id)
            formatted_thread = []
            for message in thread:
                formatted_thread.append({
                    "author": message.author,
                    "content": message.text,
                    "timestamp": message.timestamp,
                })
            return formatted_thread
        except Exception as e:
            logging.error("Failed to retrieve thread: %s", e)
            return []

    def format_thread_for_model(self, thread: List[Dict[str, str]]) -> str:
        """Format the thread into a string suitable for the model."""
        formatted = []
        for message in thread:
            formatted.append(f"{message['author']}: {message['content']}")
        return "\n".join(formatted)

    def generate_reply_to_thread(self, conversation_id: str, prompt: str) -> Optional[str]:
        """Generate a reply based on the entire thread."""
        thread = self.retrieve_thread(conversation_id)
        if not thread:
            logging.warning("No thread found for conversation ID: %s", conversation_id)
            return None

        formatted_thread = self.format_thread_for_model(thread)
        full_prompt = f"Thread:\n{formatted_thread}\n\nPrompt:\n{prompt}"
        return self.generate_reply(full_prompt)

    def fetch_conversations(self) -> List[Dict[str, Any]]:
        """
        Discover conversations that may need replies by scanning stored interactions
        in scraped_articles (content JSON) and verifying thread state live.

        We consider a conversation needs a reply if:
        - The bot has participated (we've stored an interaction for its root tweet)
        - Someone else has replied after our last reply (we didn't send the last message)

        Returns a list like:
        [{"original_tweet": str, "conversation_id": str, "last_reply_author": str, "reply_count": int, "tweet_id": str}]
        """
        conversations: List[Dict[str, Any]] = []

        if not self.twitter_client or not self.twitter_client.username:
            logging.warning("Cannot fetch conversations: twitter client not initialized")
            return conversations

        my_handle = (self.twitter_client.username or "").lower()

        # Collect candidate conversation_ids from scraped_articles JSON payloads
        seen: set[str] = set()
        try:
            for article in self.sql_store.iter_scraped_articles():
                # Only consider records that look like our interaction logs
                if not article.content:
                    continue
                try:
                    rec = json.loads(article.content)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                # Must be an interaction we logged from respond_to_tweets
                conv_id = str(rec.get("conversation_id") or "").strip()
                tw_text = rec.get("tweet_text") or ""
                tw_id = str(rec.get("tweet_id") or "").strip()
                if not conv_id or not tw_id:
                    continue
                if conv_id in seen:
                    continue
                seen.add(conv_id)

                # Verify current state from Twitter to check who replied last
                try:
                    replies = self.twitter_client.fetch_replies(conv_id, since_id=None)
                    if not replies:
                        continue
                    replies_sorted = sorted(
                        replies,
                        key=lambda x: getattr(x, "timestamp", 0) or 0,
                        reverse=True,
                    )
                    last_author = (replies_sorted[0].author_handle or "").lower()
                    if last_author == my_handle:
                        logging.info("Skipping conversation %s - bot sent last message", conv_id)
                        continue
                    conversations.append(
                        {
                            "original_tweet": tw_text,
                            "conversation_id": conv_id,
                            "last_reply_author": last_author,
                            "reply_count": len(replies),
                            "tweet_id": tw_id,
                        }
                    )
                except Exception as e:
                    logging.debug("Could not fetch replies for conversation %s: %s", conv_id, e)
                    continue
        except Exception as e:
            logging.error("Error scanning scraped_articles for conversations: %s", e)
            return conversations

        logging.info("Found %d conversations needing replies", len(conversations))
        return conversations
