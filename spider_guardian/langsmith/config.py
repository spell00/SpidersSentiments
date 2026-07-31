"""LangSmith configuration and integration helpers for Spider Guardian."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Iterable, Set

import pandas as pd
try:
    from langsmith import Client
except ImportError:  # pragma: no cover - optional integration dependency
    Client = None  # type: ignore[assignment]
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import logging
import requests
import re

def fetch_final_url_with_selenium(url: str, *, headless: bool = True, timeout_seconds: int = 12) -> Optional[str]:
    """Resolve a potentially redirected URL by loading it in Firefox.

    - headless=False will open a visible browser window (as requested for debugging)
    - timeout_seconds controls how long we poll for a stable, final URL
    - Tracking protection is disabled to avoid X/Twitter blocks
    
    Skip resolution if URL already has the canonical format (username in path).
    Only resolve URLs starting with https://x.com/i/status/ (generic shortlink).
    """
    # Skip resolution if URL already has username (canonical format)
    # e.g., https://x.com/spider_marsh/status/1234 is already canonical
    if url and not url.startswith("https://x.com/i/status/"):
        logging.debug("[resolve] skipping resolution for canonical URL: %s", url)
        return url
    
    options = Options()
    if headless:
        options.add_argument("--headless")
    # Disable Enhanced Tracking Protection to prevent blocking Twitter/X.com
    options.set_preference("privacy.trackingprotection.enabled", False)
    options.set_preference("privacy.trackingprotection.pbmode.enabled", False)
    options.set_preference("privacy.trackingprotection.socialtracking.enabled", False)

    driver = webdriver.Firefox(options=options)

    try:
        logging.info("[resolve] navigating to: %s (headless=%s, timeout=%ss)", url, headless, timeout_seconds)
        driver.get(url)

        # First, wait briefly for any immediate redirect
        try:
            WebDriverWait(driver, min(5, timeout_seconds)).until(EC.url_changes(url))
        except Exception:
            pass

        # Then actively poll for a stable URL for up to timeout_seconds
        start = time.time()
        last_url = driver.current_url
        while time.time() - start < timeout_seconds:
            # Wait a bit to allow JS redirects/meta-refresh
            time.sleep(1)
            current = driver.current_url
            if current != last_url:
                logging.info("[resolve] url changed -> %s", current)
                last_url = current
                # Reset timer a bit to allow further redirects
                start = time.time()
            # Check if page finished loading
            try:
                ready_state = driver.execute_script("return document.readyState")
                if ready_state == "complete":
                    # Give a small extra margin for any late JS to run
                    time.sleep(0.5)
            except Exception:
                pass

        final_url = driver.current_url
        logging.info("[resolve] final url: %s", final_url)
        return final_url
    except Exception as e:
        logging.error("Failed to fetch final URL: %s", e)
        return None
    finally:
        try:
            driver.quit()
        except Exception:
            pass

class LangSmithIntegration:
    """Handle LangSmith operations for the Spider Guardian bot."""

    def __init__(self) -> None:
        self.client: Optional[Client] = None
        self.project_name = os.getenv("LANGSMITH_PROJECT", "spider-guardian-bot")
        # Default consolidated dataset for non-interaction posts: use streamed dataset
        self.dataset_name = os.getenv("LANGSMITH_DATASET", "spider-streamed-dataset")
        self._setup_client()

    def _setup_client(self) -> None:
        """Initialise the LangSmith client with API credentials."""
        try:
            api_key = os.getenv("LANGSMITH_API_KEY")
            if not api_key:
                logging.warning("LANGSMITH_API_KEY not found. Set it to enable LangSmith tracking.")
                return

            if Client is None:
                logging.warning("langsmith package is not installed. Install it to enable LangSmith tracking.")
                return

            self.client = Client(
                api_key=api_key,
                api_url=os.getenv("LANGSMITH_API_URL", "https://api.smith.langchain.com"),
            )
            logging.info("LangSmith client initialised successfully")

            os.environ.setdefault("LANGSMITH_PROJECT", self.project_name)
            self._ensure_project_exists()

        except Exception as exc:
            logging.error("Failed to initialise LangSmith client: %s", exc)
            self.client = None

    def _ensure_project_exists(self) -> None:
        """Ensure the configured project exists in LangSmith."""
        if not self.client:
            return

        try:
            projects = list(self.client.list_projects())
            project_names = [project.name for project in projects]

            if self.project_name not in project_names:
                self.client.create_project(
                    project_name=self.project_name,
                    description="Spider Guardian Bot reply generation and performance tracking",
                )
                logging.info("Created LangSmith project: %s", self.project_name)
        except Exception as exc:
            logging.error("Failed to ensure project exists: %s", exc)

    # ---- Metadata helpers -------------------------------------------------
    def _normalize_example_metadata(self, md: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Return a copy of metadata with guaranteed keys and safe defaults.

        Guarantees presence and caps where needed:
        - updates_done: int, capped to 0..3 (official cadence count)
        - completed: bool, set True when updates_done reaches 3
        - updates_count: int, default 0 (total updates performed, including forced/content changes)
        - created_at/modified_at: left as-is if present
        - level/type/key/reply_url/reply_id/tweet_id: left as-is if present
        """
        base: Dict[str, Any] = dict(md or {})
        # updates_done
        try:
            upd = int(base.get("updates_done", 0) or 0)
        except Exception:
            upd = 0
        if upd > 3:
            upd = 3
        base["updates_done"] = upd
        # completed
        if upd >= 3:
            base["completed"] = True
        elif base.get("completed") is None:
            base["completed"] = False
        # updates_count (ensure at least updates_done)
        try:
            ucnt = int(base.get("updates_count", 0) or 0)
        except Exception:
            ucnt = 0
        if ucnt < upd:
            ucnt = upd
        base["updates_count"] = ucnt
        # Back-compat: if legacy fields exist, fold them in but do not persist new values here
        try:
            legacy_fc = int(base.get("force_count", 0) or 0)
            if legacy_fc > 0 and base["updates_count"] < upd + legacy_fc:
                base["updates_count"] = upd + legacy_fc
        except Exception:
            pass
        return base

    def log_reply_generation(
        self,
        run_id: str,
        original_tweet: str,
        generated_reply: str,
        prompt: str,
        model_name: str,
        generation_time_ms: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Log a reply generation event to LangSmith."""
        if not self.client:
            return None

        try:
            if not run_id:
                run_id = str(uuid.uuid4())

            self.client.create_run(
                id=run_id,
                name="reply_generation",
                project_name=self.project_name,
                inputs={
                    "original_tweet": original_tweet,
                    "prompt": prompt,
                    "model": model_name,
                },
                outputs={
                    "generated_reply": generated_reply,
                },
                run_type="llm",
                extra={
                    "generation_time_ms": generation_time_ms,
                    "metadata": metadata or {},
                },
            )

            logging.info("Logged reply generation to LangSmith: %s", run_id)
            return run_id
        except Exception as exc:
            logging.error("Failed to log reply generation: %s", exc)
            return None

    def log_engagement_metrics(
        self,
        reply_id: str,
        likes: int,
        replies: int,
        impressions: int,
        posted_at: datetime,
    ) -> None:
        """Log engagement metrics for a posted reply."""
        if not self.client:
            return

        try:
            self.client.create_run(
                name="engagement_tracking",
                project_name=self.project_name,
                inputs={"reply_id": reply_id},
                outputs={
                    "likes": likes,
                    "replies": replies,
                    "impressions": impressions,
                    "posted_at": posted_at.isoformat(),
                },
                run_type="tool",
            )
            logging.info("Logged engagement metrics for reply %s", reply_id)
        except Exception as exc:
            logging.error("Failed to log engagement metrics: %s", exc)

    def create_feedback_run(
        self,
        run_id: str,
        feedback_score: float,
        feedback_comment: Optional[str] = None,
    ) -> None:
        """Attach feedback to a reply generation run."""
        if not self.client:
            return

        try:
            self.client.create_feedback(
                run_id=run_id,
                key="reply_quality",
                score=feedback_score,
                comment=feedback_comment,
            )
            logging.info("Added feedback to run %s: %s", run_id, feedback_score)
        except Exception as exc:
            logging.error("Failed to create feedback: %s", exc)

    def upload_dataset_from_db(self, db_path: str = "data/spider_trending.sqlite", max_examples: int = None, *, show_browser: bool = False, url_wait_seconds: int = 12) -> None:
        """Upload trending posts data to LangSmith as a dataset.

        Non-destructive: Always upserts by post_id/url and preserves other examples.
        Created timestamps are preserved; only a modified_at field is updated on changes.
        """
        if not self.client:
            return

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(trending_posts)")
            columns = [row[1] for row in cursor.fetchall()]
            has_impressions = "impression_count" in columns

            if has_impressions:
                query = (
                    "SELECT post_id, text, author, like_count, reply_count, impression_count, "
                    "collected_at, post_created_at, url FROM trending_posts "
                    "WHERE collected_at >= datetime('now', '-7 days') ORDER BY collected_at DESC"
                )
            else:
                query = (
                    "SELECT post_id, text, author, like_count, reply_count, 0 as impression_count, "
                    "collected_at, post_created_at, url FROM trending_posts "
                    "WHERE collected_at >= datetime('now', '-7 days') ORDER BY collected_at DESC"
                )

            df = pd.read_sql_query(query, conn)
            conn.close()

            examples: List[Dict[str, Any]] = []
            show_progress = bool(os.getenv("UPD_PROGRESS"))
            for idx, row in enumerate(df.iterrows()):
                if max_examples is not None and idx >= max_examples:
                    logging.info(f"Test mode: Stopping after {idx} examples.")
                    break
                _, row = row

                url = row["url"]
                if show_progress:
                    print(f"[{idx+1}] resolve: {url}", flush=True)
                url = fetch_final_url_with_selenium(url, headless=not show_browser, timeout_seconds=url_wait_seconds) or url
                if show_progress:
                    print(f"[{idx+1}] resolved -> {url}", flush=True)

                # Calculate how long ago the tweet was made
                post_created_at = row.get("collected_at")
                if post_created_at:
                    post_created_at = datetime.fromisoformat(post_created_at)
                    time_since_post = datetime.now() - post_created_at
                    logging.info("Tweet was created %s ago", time_since_post)

                if show_progress:
                    print(f"[{idx+1}] fetch: {url}", flush=True)
                response = requests.get(url, allow_redirects=True)
                response.raise_for_status()

                # Check if the response is JSON
                if response.headers.get("Content-Type", "").startswith("application/json"):
                    new_data = response.json()
                elif "text/html" in response.headers.get("Content-Type", ""):
                    # Only attempt HTML parsing for X/Twitter domains; otherwise skip to avoid false positives
                    from urllib.parse import urlparse
                    host = urlparse(url).netloc.lower()
                    if host.endswith("x.com") or host.endswith("twitter.com") or host.endswith("www.x.com") or host.endswith("www.twitter.com"):
                        options = Options()
                        if not show_browser:
                            options.add_argument("--headless")
                        # Disable Enhanced Tracking Protection to prevent blocking Twitter/X.com
                        options.set_preference("privacy.trackingprotection.enabled", False)
                        options.set_preference("privacy.trackingprotection.pbmode.enabled", False)
                        options.set_preference("privacy.trackingprotection.socialtracking.enabled", False)
                        driver = webdriver.Firefox(options=options)
                        driver.get(url)
                        time.sleep(5)
                        page_source = driver.page_source
                        soup = BeautifulSoup(page_source, "html.parser")
                        logging.info("Parsing HTML content for X/Twitter page.")

                        # Extract metrics using tolerant regex (captures numbers with separators)
                        # Extend corpus with aria-label and title attributes (many counts only appear there)
                        attr_texts: List[str] = []
                        for tag in soup.find_all(True):
                            for attr in ("aria-label", "title"):
                                try:
                                    val = tag.get(attr)
                                    if isinstance(val, str) and any(word in val.lower() for word in ("view", "like", "repl", "repost")):
                                        attr_texts.append(val)
                                except Exception:
                                    pass
                        view_text = (soup.text + "\n" + "\n".join(attr_texts))

                        # Helper to normalize compact counts like 2.3K / 1.1M / 4B
                        def _parse_count(raw_num: str) -> int:
                            try:
                                raw_num = raw_num.strip()
                                mult = 1
                                if raw_num.endswith(('K','k')):
                                    mult = 1_000; raw_num = raw_num[:-1]
                                elif raw_num.endswith(('M','m')):
                                    mult = 1_000_000; raw_num = raw_num[:-1]
                                elif raw_num.endswith(('B','b')):
                                    mult = 1_000_000_000; raw_num = raw_num[:-1]
                                raw_num = re.sub(r"[.,]", "", raw_num)
                                if raw_num.isdigit():
                                    return int(raw_num) * mult
                                # Fallback: float then int
                                return int(float(raw_num) * mult)
                            except Exception:
                                return 0
                        
                        # Views/Impressions
                        match = re.search(r"([\d.,KMBkmb]+)\s+Views", view_text, flags=re.IGNORECASE)
                        if match:
                            views = _parse_count(match.group(1))
                        else:
                            views = 0

                        # Likes
                        match_likes = re.search(r"([\d.,KMBkmb]+)\s+Likes?", view_text, flags=re.IGNORECASE)
                        if match_likes:
                            likes = _parse_count(match_likes.group(1))
                        else:
                            likes = 0

                        # Replies
                        match_replies = re.search(r"([\d.,KMBkmb]+)\s+Repl(?:y|ies)", view_text, flags=re.IGNORECASE)
                        if match_replies:
                            replies = _parse_count(match_replies.group(1))
                        else:
                            replies = 0

                        # Reposts
                        match_reposts = re.search(r"([\d.,KMBkmb]+)\s+Reposts?", view_text, flags=re.IGNORECASE)
                        if match_reposts:
                            reposts = _parse_count(match_reposts.group(1))
                        else:
                            reposts = 0

                        # Sanity cap: discard implausible values
                        if views < 0 or views > 1_000_000_000:
                            logging.warning("Discarding implausible view count %s for %s", views, url)
                            views = 0
                        if likes < 0 or likes > 10_000_000:
                            logging.warning("Discarding implausible like count %s for %s", likes, url)
                            likes = 0
                        if replies < 0 or replies > 1_000_000:
                            logging.warning("Discarding implausible reply count %s for %s", replies, url)
                            replies = 0
                        if reposts < 0 or reposts > 10_000_000:
                            logging.warning("Discarding implausible repost count %s for %s", reposts, url)
                            reposts = 0

                        new_data = [{
                            "impression_count": views,
                            "like_count": likes,
                            "reply_count": replies,
                            "repost_count": reposts
                        }]
                        logging.info("Extracted metrics from HTML: views=%d likes=%d replies=%d reposts=%d", views, likes, replies, reposts)
                    else:
                        logging.info("Skipping HTML views extraction for non-X domain: %s", host)
                        new_data = []  # preserve existing DB values
                else:
                    logging.error("Unsupported Content-Type: %s", response.headers.get("Content-Type"))
                    new_data = []  # Set to an empty list or handle appropriately

                # Log a snippet of the response for debugging
                logging.debug("Response content: %s", response.text[:500])  # Log the first 500 characters

                # Insert new data into the database
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                for item in new_data:
                    # TODO need to insert here time of update and time gap since last update and since original post
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO trending_posts (post_id, text, author, like_count, reply_count, impression_count, repost_count, collected_at, url)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            row["post_id"],
                            row["text"],
                            row["author"],
                            item.get("like_count", row.get("like_count", 0)),
                            item.get("reply_count", row.get("reply_count", 0)),
                            item.get("impression_count", row.get("impression_count", 0)),
                            item.get("repost_count", row.get("repost_count", 0)),
                            row["collected_at"],
                            row["url"],
                        ),
                    )

                conn.commit()
                conn.close()
                logging.info("New data fetched and stored in the database.")
                if show_progress:
                    print(f"[{idx+1}] stored + staged for upload", flush=True)

                # Use freshly scraped metrics if available, otherwise fall back to DB row
                final_likes = new_data[0]["like_count"] if new_data else int(row.get("like_count", 0) or 0)
                final_replies = new_data[0]["reply_count"] if new_data else int(row.get("reply_count", 0) or 0)
                final_impressions = new_data[0]["impression_count"] if new_data else int(row.get("impression_count", 0) or 0)
                final_reposts = new_data[0]["repost_count"] if new_data else int(row.get("repost_count", 0) or 0)

                examples.append(
                    {
                        "inputs": {
                            "tweet_text": row["text"],
                            "author": row["author"],
                            "url": row["url"],
                            # Use freshly scraped metrics for input visibility
                            "engagement_metrics": {
                                "likes": final_likes,
                                "replies": final_replies,
                                "impressions": final_impressions,
                                "reposts": final_reposts,
                            },
                        },
                        "outputs": {
                            "engagement_metrics": {
                                "likes": final_likes,
                                "replies": final_replies,
                                "impressions": final_impressions,
                                "reposts": final_reposts,
                            },
                            # Top-level convenience fields for table view
                            "impressions": final_impressions,
                            "likes": final_likes,
                            "replies": final_replies,
                            "reposts": final_reposts,
                        },
                        "metadata": {
                            "post_id": row["post_id"],
                            "collected_at": row["collected_at"],
                            "updates_done": 0,  # initialize refresh cadence counter
                            "completed": False,
                            "updates_count": 0,
                            "forced_updates": 0,
                            # Store metrics directly on metadata for easier querying
                            "input_metrics": {
                                "likes": final_likes,
                                "replies": final_replies,
                                "impressions": final_impressions,
                                "reposts": final_reposts,
                            },
                            "reply_metrics": None,  # trending posts have no reply-level metrics
                        },
                    }
                )

            if not examples:
                logging.info("No data found to upload to LangSmith")
                return

            # Ensure dataset exists, create if not
            try:
                self.client.create_dataset(
                    dataset_name=self.dataset_name,
                    description="Spider-related tweets and their engagement metrics",
                )
                logging.info("Created dataset: %s", self.dataset_name)
            except Exception:
                pass

            dataset = self.client.read_dataset(dataset_name=self.dataset_name)
            # Always upsert by key
            try:
                existing = list(self.client.list_examples(dataset_id=dataset.id))
            except Exception as exc:
                logging.warning("Could not list existing examples for upsert: %s", exc)
                existing = []

            def _key_from_example(ex) -> str:
                meta = ex.metadata or {}
                inputs = ex.inputs or {}
                return str(meta.get("post_id") or inputs.get("post_id") or inputs.get("url") or "")

            existing_by_key = { _key_from_example(ex): ex for ex in existing if _key_from_example(ex) }

            to_create_inputs: List[Dict[str, Any]] = []
            to_create_outputs: List[Dict[str, Any]] = []
            to_create_metadata: List[Dict[str, Any]] = []
            updated_count = 0
            created_count = 0

            for ex in examples:
                key = str(ex["metadata"].get("post_id") or ex["inputs"].get("post_id") or ex["inputs"].get("url") or "")
                # Preserve created_at and stamp modified_at
                ex_meta = dict(ex["metadata"] or {})
                ex_meta["modified_at"] = datetime.now().isoformat()
                if key and key in existing_by_key:
                    try:
                        old_meta = existing_by_key[key].metadata or {}
                        if old_meta.get("created_at"):
                            ex_meta["created_at"] = old_meta["created_at"]
                        # Ensure updates_done never exceeds 3
                        upd = int(old_meta.get("updates_done", 0) or 0)
                        # Preserve but do not escalate; cap to 3
                        ex_meta["updates_done"] = min(upd, 3)
                        if old_meta.get("completed") is True:
                            ex_meta["completed"] = True
                        # Preserve any other prior metadata fields, then normalize
                        merged_meta = {**old_meta, **ex_meta}
                        merged_meta = self._normalize_example_metadata(merged_meta)
                        self.client.update_example(
                            example_id=existing_by_key[key].id,
                            inputs=ex["inputs"],
                            outputs=ex["outputs"],
                            metadata=merged_meta,
                        )
                        updated_count += 1
                    except Exception as exc:
                        logging.warning("Failed to update example %s: %s", key, exc)
                else:
                    to_create_inputs.append(ex["inputs"])
                    to_create_outputs.append(ex["outputs"])
                    to_create_metadata.append(self._normalize_example_metadata(ex_meta))

            if to_create_inputs:
                try:
                    self.client.create_examples(
                        inputs=to_create_inputs,
                        outputs=to_create_outputs,
                        metadata=to_create_metadata,
                        dataset_name=self.dataset_name,
                    )
                    created_count = len(to_create_inputs)
                except Exception as exc:
                    logging.error("Failed to create %d new examples: %s", len(to_create_inputs), exc)

            logging.info(
                "Upsert complete: %d updated, %d created, 0 deleted (kept others intact)",
                updated_count,
                created_count,
            )

        except Exception as exc:
            logging.error("Failed to upload dataset: %s", exc)

    def update_dataset_from_db(self, db_path: str, dataset_name: str, match_key: str = "post_id", max_examples: int = None) -> None:
        """
        Update all examples in the LangSmith dataset with latest metrics from the database, matching by post_id.
        Adds a cadence: only refresh once per day for up to 3 days after the post.
        Cadence is tracked in example metadata (modified_at, updates_done).
        """
        if not self.client:
            logging.error("LangSmith client not initialized.")
            return

        # Load all examples from LangSmith dataset
        dataset = self.client.read_dataset(dataset_name=dataset_name)
        examples = list(self.client.list_examples(dataset_id=dataset.id))
        logging.info(f"Loaded {len(examples)} examples from LangSmith dataset '{dataset_name}'")
        # Prioritize never-updated trending posts first, then lowest updates_done, then oldest created
        def _priority(ex) -> tuple:
            m = ex.metadata or {}
            try:
                u = int(m.get("updates_done", 0) or 0)
            except Exception:
                u = 0
            compl = 1 if (m.get("completed") is True) else 0
            cstr = m.get("created_at") or m.get("collected_at") or m.get("post_created_at")
            try:
                cts = datetime.fromisoformat(str(cstr)) if cstr else datetime.max
            except Exception:
                cts = datetime.max
            return (compl, u, cts)
        examples = sorted(examples, key=_priority)

        # Load all rows from SQLite
        conn = sqlite3.connect(db_path)
        try:
            df = pd.read_sql_query("SELECT * FROM trending_posts", conn)
        except Exception:
            df = None
        finally:
            conn.close()
        if df is None:
            logging.error("No interactions/replies table found in database.")
            return
        db_rows = {str(row[match_key]): row for _, row in df.iterrows() if match_key in row}
        logging.info(f"Loaded {len(db_rows)} rows from database '{db_path}'")

        updated = 0
        now = datetime.utcnow()
        one_day = timedelta(days=1)
        max_age = timedelta(days=3)

        for ex in examples:
            ex_id = ex.id
            ex_meta = ex.metadata or {}
            ex_key = str(ex_meta.get(match_key) or ex.inputs.get(match_key) or ex.inputs.get("url"))
            if not ex_key or ex_key not in db_rows:
                continue

            # Cadence gating
            post_ts = None
            post_ts_str = ex_meta.get("post_created_at") or ex_meta.get("collected_at")
            if post_ts_str:
                try:
                    post_ts = datetime.fromisoformat(post_ts_str)
                except Exception:
                    post_ts = None
            if post_ts is not None:
                if (now - post_ts) > max_age:
                    continue
            modified_at_ts = None
            modified_at_str = ex_meta.get("modified_at")
            if modified_at_str:
                try:
                    modified_at_ts = datetime.fromisoformat(str(modified_at_str))  # type: ignore[arg-type]
                except Exception:
                    modified_at_ts = None
            if modified_at_ts is not None and (now - modified_at_ts) < one_day:
                continue
            try:
                updates_done = int(ex_meta.get("updates_done", 0) or 0)
            except Exception:
                updates_done = 0
            if updates_done >= 3:
                continue

            db_row = db_rows[ex_key]
            # Extract metrics from DB row
            new_metrics = {
                "likes": int(db_row.get("like_count", 0)),
                "replies": int(db_row.get("reply_count", 0)),
                "impressions": int(db_row.get("impression_count", 0)),
                "reposts": int(db_row.get("repost_count", 0)),
            }
            # Get current metrics from example
            old_metrics = (ex.outputs.get("engagement_metrics") if ex.outputs else {}) or {}
            # Only update if metrics differ
            if any(new_metrics[k] != old_metrics.get(k, 0) for k in new_metrics):
                new_meta = {**(ex_meta or {}), "modified_at": now.isoformat()}
                # Scheduled trending update: increment both official and total counters
                try:
                    new_meta["updates_done"] = min(int(ex_meta.get("updates_done", 0) or 0) + 1, 3)
                except Exception:
                    new_meta["updates_done"] = 1
                try:
                    new_meta["updates_count"] = int(ex_meta.get("updates_count", 0) or 0) + 1
                except Exception:
                    new_meta["updates_count"] = 1
                if int(new_meta.get("updates_done", 0) or 0) >= 3:
                    new_meta["completed"] = True
                # Attach metrics to metadata (input=post metrics, reply none)
                new_meta["input_metrics"] = new_metrics
                # Preserve any existing reply_metrics (should remain None for trending)
                if "reply_metrics" not in new_meta:
                    new_meta["reply_metrics"] = None
                # Normalize and preserve any prior keys
                new_meta = self._normalize_example_metadata(new_meta)
                self.client.update_example(
                    example_id=ex_id,
                    outputs={"engagement_metrics": new_metrics},
                    metadata=new_meta,
                )
                updated += 1
                if max_examples is not None and updated >= max_examples:
                    logging.info(f"Test mode: Stopping after {updated} updates.")
                    break
        logging.info(f"Updated {updated} examples in LangSmith dataset '{dataset_name}'")

    def plan_refresh_interactions_dataset_from_sql(
        self,
        db_path: str = "data/spider_guardian.sqlite",
        dataset_name: str = "spider-interactions-dataset",
        max_examples: Optional[int] = None,
        report: bool = True,
        report_limit: int = 50,
    ) -> Dict[str, Any]:
        """Preview which examples would be refreshed vs skipped, without performing updates.

        Fast preflight that avoids any external fetches. It:
        - Loads dataset examples and local SQL rows
        - Applies cadence gating using metadata (created_at/modified_at/updates_done)
        - For eligible items, compares DB metrics to current outputs
        - Returns a summary with reasons for NOT updating (e.g., not_yet_due, completed_schedule, past_total_window, no_db_row, no_change)

        Returns a dict summary with counts and small key samples per reason.
        """
        if not self.client:
            logging.error("LangSmith client not initialized.")
            return {"updated_would": 0, "skipped": {}, "eligible": 0, "scanned": 0}

        # Load examples from dataset
        dataset = self.client.read_dataset(dataset_name=dataset_name)
        examples = list(self.client.list_examples(dataset_id=dataset.id))
        logging.info("[plan] Loaded %d examples from dataset '%s'", len(examples), dataset_name)

        # Prioritize never-updated first, then lowest updates_done, then oldest created_at.
        def _priority(ex) -> tuple:
            m = ex.metadata or {}
            try:
                u = int(m.get("updates_done", 0) or 0)
            except Exception:
                u = 0
            compl = 1 if (m.get("completed") is True) else 0  # completed last
            cstr = m.get("created_at") or m.get("collected_at")
            try:
                cts = datetime.fromisoformat(str(cstr)) if cstr else datetime.max
            except Exception:
                cts = datetime.max
            return (compl, u, cts)

        examples = sorted(examples, key=_priority)

        # Load SQL rows from interactions and content
        try:
            conn = sqlite3.connect(db_path)
        except Exception as exc:
            logging.error("[plan] Failed to open SQL DB %s: %s", db_path, exc)
            return {"updated_would": 0, "skipped": {"db_error": 0}, "eligible": 0, "scanned": 0}
        try:
            try:
                df_inter = pd.read_sql_query(
                    "SELECT tweet_id, reply_id, url, like_count, reply_count, impression_count, repost_count, created_at FROM interactions",
                    conn,
                )
            except Exception:
                df_inter = pd.DataFrame()
            try:
                df_cont = pd.read_sql_query(
                    "SELECT post_id, url, like_count, reply_count, impression_count, repost_count, created_at FROM content",
                    conn,
                )
            except Exception:
                df_cont = pd.DataFrame()
        finally:
            conn.close()

        # Build lookup maps by multiple keys
        inter_by_key: Dict[str, Any] = {}
        for _, r in df_inter.iterrows():
            url = str(r.get("url") or "")
            rep = str(r.get("reply_id") or "")
            tw = str(r.get("tweet_id") or "")
            if url:
                inter_by_key[url] = r
            if rep:
                inter_by_key[rep] = r
            if tw:
                inter_by_key[tw] = r
        cont_by_key: Dict[str, Any] = {}
        for _, r in df_cont.iterrows():
            url = str(r.get("url") or "")
            pid = str(r.get("post_id") or "")
            if url:
                cont_by_key[url] = r
            if pid:
                cont_by_key[pid] = r

        now = datetime.utcnow()
        cadence_schedule = [1, 2, 3]  # days
        total_window_days = sum(cadence_schedule)

        scanned = 0
        eligible = 0  # total examples that pass cadence + have DB row
        would_update = 0  # limited count respecting max_examples cap (if provided)
        would_update_sample: List[str] = []  # sample of URLs/keys that would update
        # We no longer early-break when hitting the cap; we continue scanning to collect full skip stats.
        skipped_reasons: Dict[str, List[str]] = {
            "past_total_window": [],
            "completed_schedule": [],
            "not_yet_due_first": [],
            "not_yet_due_subsequent": [],
            "no_db_row": [],
            "no_key": [],
        }

        for ex in examples:
            meta = ex.metadata or {}
            inputs = ex.inputs or {}
            ex_type = meta.get("type") or "interaction"
            # Prefer a human-friendly display URL where possible
            reply_id = str(meta.get("reply_id") or "").strip()
            reply_url = meta.get("reply_url") or (f"https://x.com/i/status/{reply_id}" if reply_id else "")
            url_val = inputs.get("url") or meta.get("url") or reply_url
            key = str(
                meta.get("key")
                or url_val
                or reply_id
                or meta.get("tweet_id")
                or meta.get("post_id")
                or ""
            )
            if not key:
                skipped_reasons["no_key"].append("<no-key>")
                scanned += 1
                continue
            scanned += 1

            # Cadence gating (variable schedule)
            created_str = meta.get("created_at") or meta.get("collected_at")
            try:
                created_ts = datetime.fromisoformat(created_str) if created_str else None
            except Exception:
                created_ts = None
            try:
                updates_done = int(meta.get("updates_done", 0) or 0)
            except Exception:
                updates_done = 0
            modified_at_ts = None
            modified_at_str = meta.get("modified_at")
            if modified_at_str:
                try:
                    modified_at_ts = datetime.fromisoformat(str(modified_at_str))
                except Exception:
                    modified_at_ts = None

            if created_ts is not None and (now - created_ts).days > total_window_days:
                if len(skipped_reasons["past_total_window"]) < report_limit:
                    skipped_reasons["past_total_window"].append(key)
                continue
            # Completed schedule: either explicit flag or updates_done consumed all slots
            if meta.get("completed") is True or updates_done >= len(cadence_schedule):
                if len(skipped_reasons["completed_schedule"]) < report_limit:
                    skipped_reasons["completed_schedule"].append(key)
                continue
            required_wait_days = cadence_schedule[updates_done]
            if updates_done == 0:
                if created_ts is None or (now - created_ts) < timedelta(days=required_wait_days):
                    if len(skipped_reasons["not_yet_due_first"]) < report_limit:
                        skipped_reasons["not_yet_due_first"].append(key)
                    continue
            else:
                if modified_at_ts is None or (now - modified_at_ts) < timedelta(days=required_wait_days):
                    if len(skipped_reasons["not_yet_due_subsequent"]) < report_limit:
                        skipped_reasons["not_yet_due_subsequent"].append(key)
                    continue

            # If it passes cadence, check DB row presence
            db_row = None
            if ex_type in ("interaction", "flagged_reply"):
                db_row = inter_by_key.get(key)
            elif ex_type == "streamed_post":
                db_row = cont_by_key.get(key)
            if db_row is None:
                # looser lookup via url if present
                url_key = inputs.get("url") or meta.get("url")
                if url_key:
                    if ex_type in ("interaction", "flagged_reply"):
                        db_row = inter_by_key.get(str(url_key))
                    else:
                        db_row = cont_by_key.get(str(url_key))
            if db_row is None:
                if len(skipped_reasons["no_db_row"]) < report_limit:
                    skipped_reasons["no_db_row"].append(key)
                continue

            # Would be updated (cadence allows and DB row exists)
            eligible += 1
            if max_examples is None or would_update < max_examples:
                would_update += 1  # count toward limited update set
                if len(would_update_sample) < 25:
                    # Record a nice display of what will be updated
                    display = str(url_val or key)
                    would_update_sample.append(display)
            # else: suppress counting additional updates but keep scanning for statistics

        # Convert lists to counts while preserving small samples
        skipped_summary: Dict[str, Any] = {}
        for reason, keys in skipped_reasons.items():
            skipped_summary[reason] = {
                "count": len(keys),
                "sample": keys[: min(len(keys), 10)],
            }

        # Determine partial if we suppressed additional potential updates beyond cap
        partial = max_examples is not None and eligible > would_update
        summary = {
            "scanned": scanned,
            "eligible": eligible,
            "updated_would": would_update,
            "would_update_sample": would_update_sample[:10],
            "skipped": skipped_summary,
            "partial": partial,
        }

        if report:
            logging.info(
                "[plan] scanned=%d eligible=%d would_update=%d | skipped: %s",
                scanned,
                eligible,
                would_update,
                {k: v["count"] for k, v in skipped_summary.items()},
            )
            # Print a small sample for quick inspection
            def _print_sample(title: str, key: str) -> None:
                data = skipped_summary.get(key, {"count": 0, "sample": []})
                if data["count"]:
                    print(f"[plan] {title}: count={data['count']} sample={data['sample']}")

            _print_sample("past_total_window", "past_total_window")
            _print_sample("completed_schedule", "completed_schedule")
            _print_sample("not_yet_due_first", "not_yet_due_first")
            _print_sample("not_yet_due_subsequent", "not_yet_due_subsequent")
            _print_sample("no_db_row", "no_db_row")

        return summary

    def refresh_interactions_dataset_from_sql(
        self,
        db_path: str = "data/spider_guardian.sqlite",
        dataset_name: str = "spider-interactions-dataset",
        max_examples: Optional[int] = None,
        cadence_days: int = 3,
        scrape_live: bool = False,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Update examples in the consolidated interactions dataset from spider_guardian.sqlite.

        - Matches examples by key (metadata.key or inputs.url) and refreshes engagement metrics.
        - Supports rows coming from both interactions (replies) and content (streamed posts).
        - Applies a cadence: at most once per day for up to `cadence_days` after created_at/collected_at.
        - If scrape_live=True, fetches live metrics from reply_url (slower but accurate).
        Returns a summary dict including the count and the list of updated keys.
        """
        if not self.client:
            logging.error("LangSmith client not initialized.")
            return {"updated": 0, "keys": [], "scanned": 0, "considered": 0}

        # Load the dataset examples
        dataset = self.client.read_dataset(dataset_name=dataset_name)
        examples = list(self.client.list_examples(dataset_id=dataset.id))
        logging.info("Loaded %d examples from dataset '%s'", len(examples), dataset_name)

        # Prioritize: incomplete first, then lowest updates_done (never updated first), then oldest created
        def _priority(ex) -> tuple:
            m = ex.metadata or {}
            try:
                u = int(m.get("updates_done", 0) or 0)
            except Exception:
                u = 0
            compl = 1 if (m.get("completed") is True) else 0
            cstr = m.get("created_at") or m.get("collected_at")
            try:
                cts = datetime.fromisoformat(str(cstr)) if cstr else datetime.max
            except Exception:
                cts = datetime.max
            return (compl, u, cts)

        examples = sorted(examples, key=_priority)

        # Load SQL rows from interactions and content
        try:
            conn = sqlite3.connect(db_path)
        except Exception as exc:
            logging.error("Failed to open SQL DB %s: %s", db_path, exc)
            return {"updated": 0, "keys": [], "scanned": 0, "considered": 0}
        try:
            try:
                df_inter = pd.read_sql_query(
                    "SELECT tweet_id, reply_id, url, like_count, reply_count, impression_count, repost_count, created_at FROM interactions",
                    conn,
                )
            except Exception:
                df_inter = pd.DataFrame()
            try:
                df_cont = pd.read_sql_query(
                    "SELECT post_id, url, like_count, reply_count, impression_count, repost_count, created_at FROM content",
                    conn,
                )
            except Exception:
                df_cont = pd.DataFrame()
            try:
                df_trend = pd.read_sql_query(
                    "SELECT post_id, url, like_count, reply_count, impression_count, repost_count, collected_at as created_at FROM trending_posts",
                    conn,
                )
            except Exception:
                df_trend = pd.DataFrame()
        finally:
            conn.close()

        # Build lookup maps by multiple keys
        inter_by_key: Dict[str, Any] = {}
        for _, r in df_inter.iterrows():
            url = str(r.get("url") or "")
            rep = str(r.get("reply_id") or "")
            tw = str(r.get("tweet_id") or "")
            if url:
                inter_by_key[url] = r
            if rep:
                inter_by_key[rep] = r
            if tw:
                inter_by_key[tw] = r
        cont_by_key: Dict[str, Any] = {}
        for _, r in df_cont.iterrows():
            url = str(r.get("url") or "")
            pid = str(r.get("post_id") or "")
            if url:
                cont_by_key[url] = r
            if pid:
                cont_by_key[pid] = r
        trend_by_key: Dict[str, Any] = {}
        for _, r in df_trend.iterrows():
            url = str(r.get("url") or "")
            pid = str(r.get("post_id") or "")
            if url:
                trend_by_key[url] = r
            if pid:
                trend_by_key[pid] = r

        updated = 0
        updated_keys: List[str] = []
        now = datetime.utcnow()
        # Cadence schedule (days between updates): 1 day after creation, then 2 days after first update, then 3 days after second update.
        cadence_schedule = [1, 2, 3]
        total_window_days = sum(cadence_schedule)  # hard cap after full schedule

        show_progress = bool(os.getenv("UPD_PROGRESS"))
        scanned = 0
        considered = 0  # number of actually attempted/updated examples (eligibility passed)
        for ex in examples:
            meta = ex.metadata or {}
            inputs = ex.inputs or {}
            ex_type = meta.get("type") or "interaction"
            key = str(meta.get("key") or inputs.get("url") or meta.get("url") or meta.get("reply_id") or meta.get("tweet_id") or meta.get("post_id") or "")
            if not key:
                continue
            scanned += 1

            # Cadence gating (variable schedule)
            created_str = meta.get("created_at") or meta.get("collected_at")
            try:
                created_ts = datetime.fromisoformat(created_str) if created_str else None
            except Exception:
                created_ts = None
            try:
                updates_done = int(meta.get("updates_done", 0) or 0)
            except Exception:
                updates_done = 0
            modified_at_ts = None
            if meta.get("modified_at"):
                try:
                    modified_at_ts = datetime.fromisoformat(meta["modified_at"])  # type: ignore[arg-type]
                except Exception:
                    modified_at_ts = None
            if not force_refresh:
                # Hard stop if past total window (sum of scheduled waits) since creation
                if created_ts is not None and (now - created_ts).days > total_window_days:
                    continue
                # Stop after completing all scheduled refreshes
                if (meta.get("completed") is True) or (updates_done >= len(cadence_schedule)):
                    continue
                required_wait_days = cadence_schedule[updates_done]
                if updates_done == 0:
                    # First refresh waits required_wait_days from creation
                    if created_ts is None or (now - created_ts) < timedelta(days=required_wait_days):
                        continue
                else:
                    # Subsequent refresh waits required interval since modified_at
                    if modified_at_ts is None or (now - modified_at_ts) < timedelta(days=required_wait_days):
                        continue

            # Resolve matching row
            db_row = None
            if ex_type in ("interaction", "flagged_reply"):
                db_row = inter_by_key.get(key)
            elif ex_type == "streamed_post":
                db_row = cont_by_key.get(key)
            if db_row is None:
                # Try a looser lookup using url if available
                url_key = inputs.get("url") or meta.get("url")
                if url_key:
                    if ex_type in ("interaction", "flagged_reply"):
                        db_row = inter_by_key.get(str(url_key))
                    else:
                        db_row = cont_by_key.get(str(url_key))
            if db_row is None:
                continue

            # Compute new metrics from DB (reply metrics for interactions; post metrics for streamed)
            new_metrics = {
                "likes": int(db_row.get("like_count", 0) or 0),
                "replies": int(db_row.get("reply_count", 0) or 0),
                "impressions": int(db_row.get("impression_count", 0) or 0),
                "reposts": int(db_row.get("repost_count", 0) or 0),
            }

            # If we have a reply_url for interactions, attempt live scrape to override DB metrics
            reply_url_live = None
            if ex_type in ("interaction", "flagged_reply"):
                reply_url_live = meta.get("reply_url") or (
                    f"https://x.com/i/status/{str(meta.get('reply_id')).strip()}" if meta.get("reply_id") else None
                )
                # Resolve potential redirects (mirrors trending fetch logic) before scraping so we hit canonical URL
                try:
                    if reply_url_live:
                        resolved = fetch_final_url_with_selenium(reply_url_live, headless=True, timeout_seconds=8)
                        if resolved and isinstance(resolved, str):
                            reply_url_live = resolved
                            meta["reply_url"] = reply_url_live  # keep metadata coherent for subsequent refreshes
                except Exception as exc:
                    logging.debug("Reply URL resolution failed for %s: %s", reply_url_live, exc)
                live_metrics = None
                if reply_url_live and scrape_live:
                    live_metrics = self._scrape_x_metrics(reply_url_live, show_browser=False, wait_seconds=6)
                if live_metrics:
                    # Override metrics if live scrape produced non-zero values (fallback to DB when zero)
                    for k in ("likes", "replies", "impressions", "reposts"):
                        try:
                            v_live = int(live_metrics.get(k, 0) or 0)
                        except Exception:
                            v_live = 0
                        if v_live > 0:
                            new_metrics[k] = v_live
                    # Write back to DB for persistence (best-effort)
                    try:
                        conn2 = sqlite3.connect(db_path)
                        cur2 = conn2.cursor()
                        # Update interactions table where reply_id matches
                        rid = str(meta.get("reply_id") or "").strip()
                        if rid:
                            cur2.execute(
                                "UPDATE interactions SET like_count=?, reply_count=?, impression_count=?, repost_count=? WHERE reply_id=?",
                                (
                                    new_metrics["likes"],
                                    new_metrics["replies"],
                                    new_metrics["impressions"],
                                    new_metrics["reposts"],
                                    rid,
                                ),
                            )
                            conn2.commit()
                        conn2.close()
                    except Exception as exc:
                        logging.debug("Live metric DB sync failed for reply_id=%s: %s", meta.get("reply_id"), exc)

            # Derive input (original tweet) metrics when available
            input_metrics: Dict[str, int] = {}
            if ex_type in ("interaction", "flagged_reply"):
                # Try to map original tweet metrics using multiple keys and sources
                orig_row = None
                try:
                    src_tweet_id = str(db_row.get("tweet_id") or meta.get("tweet_id") or "").strip()
                except Exception:
                    src_tweet_id = ""
                if src_tweet_id:
                    orig_row = cont_by_key.get(src_tweet_id) or trend_by_key.get(src_tweet_id)
                if orig_row is None:
                    orig_url = str(inputs.get("url") or meta.get("url") or "").strip()
                    if orig_url:
                        orig_row = cont_by_key.get(orig_url) or trend_by_key.get(orig_url)
                if orig_row is not None:
                    input_metrics = {
                        "likes": int(orig_row.get("like_count", 0) or 0),
                        "replies": int(orig_row.get("reply_count", 0) or 0),
                        "impressions": int(orig_row.get("impression_count", 0) or 0),
                        "reposts": int(orig_row.get("repost_count", 0) or 0),
                    }
                # Fallback: if not found or impressions still zero, try live scraping the original tweet URL
                if (not input_metrics or int(input_metrics.get("impressions", 0) or 0) == 0):
                    try:
                        orig_url_scrape = str(inputs.get("url") or meta.get("url") or "").strip()
                        if orig_url_scrape and scrape_live:
                            resolved_orig = fetch_final_url_with_selenium(orig_url_scrape, headless=True, timeout_seconds=8) or orig_url_scrape
                            live_in = self._scrape_x_metrics(resolved_orig, show_browser=False, wait_seconds=6)
                            if live_in:
                                input_metrics = {
                                    "likes": int(live_in.get("likes", 0) or 0),
                                    "replies": int(live_in.get("replies", 0) or 0),
                                    "impressions": int(live_in.get("impressions", 0) or 0),
                                    "reposts": int(live_in.get("reposts", 0) or 0),
                                }
                    except Exception as exc:
                        logging.debug("Live scrape for original tweet metrics failed: %s", exc)
            else:
                # For streamed posts, inputs are the original post; mirror metrics
                input_metrics = dict(new_metrics)
            
            # Update regardless of whether metrics changed - cadence already gated this
            considered += 1
            new_meta = dict(meta)
            new_meta["modified_at"] = now.isoformat()
            # Update counters (total vs forced)
            try:
                total_prev = int(new_meta.get("updates_count", 0) or 0)
            except Exception:
                total_prev = 0
            try:
                forced_prev = int(new_meta.get("forced_updates", 0) or 0)
            except Exception:
                forced_prev = 0
            if force_refresh:
                forced_prev += 1
            total_prev += 1
            new_meta["updates_count"] = total_prev
            new_meta["forced_updates"] = forced_prev
            # Increment cadence without showing a '3'; mark completed when last slot applied
            try:
                upd = int(meta.get("updates_done", 0) or 0)
            except Exception:
                upd = 0
            cadence_len = len(cadence_schedule)
            # Only advance official cadence count for non-forced refreshes
            if not force_refresh:
                new_val = upd + 1
                if new_val >= 3:
                    new_val = 3
                    new_meta["completed"] = True
                new_meta["updates_done"] = new_val
            # Always advance total updates counter
            try:
                new_meta["updates_count"] = int(new_meta.get("updates_count", 0) or 0) + 1
            except Exception:
                new_meta["updates_count"] = 1
            # Ensure thread depth level metadata
            if ex_type in ("interaction", "flagged_reply"):
                new_meta.setdefault("level", 1)
            elif ex_type == "streamed_post":
                new_meta.setdefault("level", 0)
            
            # Build/backfill reply_url for interactions
            reply_url_value = None
            if ex_type in ("interaction", "flagged_reply"):
                reply_url_value = meta.get("reply_url")
                reply_id = str(meta.get("reply_id") or "").strip()
                if not reply_url_value and reply_id:
                    reply_url_value = f"https://x.com/i/status/{reply_id}"
                    new_meta["reply_url"] = reply_url_value
                
                # If still no reply_url, delete the example
                if not reply_url_value:
                    try:
                        self.client.delete_example(example_id=ex.id)
                        logging.info("Deleted example without reply_url: key=%s type=%s", key, ex_type)
                        continue
                    except Exception as exc:
                        logging.warning("Failed to delete example %s: %s", ex.id, exc)
                        continue
            
            # Preserve all existing outputs, update engagement_metrics and ensure reply_url in outputs
            old_outputs = dict(ex.outputs or {})
            # Use distinct names to avoid confusion: reply_engagement_metrics (reply level)
            old_outputs["reply_engagement_metrics"] = new_metrics
            # Backwards compatibility: keep legacy key engagement_metrics pointing to REPLY metrics
            old_outputs["engagement_metrics"] = new_metrics
            # Set input (original tweet) metrics under inputs.engagement_metrics only; do not put reply metrics into inputs
            try:
                ex_inputs = dict(inputs)
                # If we were able to resolve original tweet metrics, expose them as the canonical input engagement_metrics
                if input_metrics:
                    ex_inputs["engagement_metrics"] = input_metrics
                inputs = ex_inputs  # for final update
            except Exception:
                pass
            # Convenience: expose top-level metrics for easier table viewing in LangSmith UI
            try:
                old_outputs["reply_impressions"] = int(new_metrics.get("impressions", 0) or 0)
                old_outputs["reply_likes"] = int(new_metrics.get("likes", 0) or 0)
                old_outputs["reply_replies"] = int(new_metrics.get("replies", 0) or 0)
                old_outputs["reply_reposts"] = int(new_metrics.get("reposts", 0) or 0)
            except Exception:
                old_outputs["reply_impressions"] = new_metrics.get("impressions", 0)
                old_outputs["reply_likes"] = new_metrics.get("likes", 0)
                old_outputs["reply_replies"] = new_metrics.get("replies", 0)
                old_outputs["reply_reposts"] = new_metrics.get("reposts", 0)
            if reply_url_value:
                old_outputs["reply_url"] = reply_url_value
            
            self.client.update_example(
                example_id=ex.id,
                inputs=inputs,
                outputs=old_outputs,
                metadata=self._normalize_example_metadata({
                    **new_meta,
                    # Store both input and reply metrics at metadata level for unified querying
                    "input_metrics": input_metrics,
                    "reply_metrics": new_metrics,
                }),
            )
            updated += 1
            # Track this updated example's key for downstream scoping (verification, etc.)
            try:
                updated_keys.append(str(key))
            except Exception:
                pass
            if show_progress:
                # Include reply URL and impressions count when available
                imp = new_metrics.get("impressions", 0)
                in_imp = 0
                try:
                    in_imp = int((old_outputs.get("input_engagement_metrics") or {}).get("impressions", 0) or 0)
                except Exception:
                    in_imp = (old_outputs.get("input_engagement_metrics") or {}).get("impressions", 0)
                if ex_type in ("interaction", "flagged_reply"):
                    print(f"[refresh {updated}/{considered}] updated key={key} type={ex_type} url={reply_url_value or inputs.get('url') or 'NA'} impr={imp} in_impr={in_imp}", flush=True)
                else:
                    print(f"[refresh {updated}/{considered}] updated key={key} type={ex_type} impr={imp} in_impr={in_imp}", flush=True)
            # Only count successfully updated examples toward max_examples
            if max_examples is not None and updated >= max_examples:
                logging.info("Test mode: Stopping after %d updates.", updated)
                break

        logging.info("Refreshed %d examples in '%s' from %s (scanned=%d considered=%d)", updated, dataset_name, db_path, scanned, considered)
        
        # Auto-prune invalid examples (missing generated_reply for interactions)
        try:
            pruned = self.prune_invalid_examples(dataset_name=dataset_name)
            if pruned > 0:
                logging.info("Auto-pruned %d invalid examples during refresh", pruned)
        except Exception as exc:
            logging.warning("Auto-prune failed (non-fatal): %s", exc)
        
        return {"updated": updated, "keys": updated_keys, "scanned": scanned, "considered": considered}

    def prune_invalid_examples(
        self,
        dataset_name: str = "spider-interactions-dataset",
        require_generated_reply: bool = True,
    ) -> int:
        """Remove examples from dataset that don't meet quality criteria.
        
        By default, removes interaction/reply examples that lack a generated_reply output.
        Streamed posts without replies are kept (they're observation-only).
        
        Returns count of deleted examples.
        """
        if not self.client:
            logging.error("LangSmith client not initialized.")
            return 0
        
        try:
            dataset = self.client.read_dataset(dataset_name=dataset_name)
            examples = list(self.client.list_examples(dataset_id=dataset.id))
            logging.info("Pruning scan: loaded %d examples from '%s'", len(examples), dataset_name)
        except Exception as exc:
            logging.error("Failed to load dataset '%s': %s", dataset_name, exc)
            return 0
        
        deleted = 0
        skipped_streamed = 0
        for ex in examples:
            meta = ex.metadata or {}
            outputs = ex.outputs or {}
            ex_type = str(meta.get("type") or "").strip().lower()
            
            # Streamed posts don't need replies, skip them
            if ex_type == "streamed_post":
                skipped_streamed += 1
                continue
            
            # Delete sentinel reply rows that never captured an actual reply_id
            if ex_type in ("interaction", "flagged_reply") and str(meta.get("reply_id") or "").strip() == "posted_no_id_found":
                try:
                    self.client.delete_example(example_id=ex.id)
                    deleted += 1
                    key = meta.get("key") or meta.get("reply_id") or meta.get("tweet_id") or "unknown"
                    logging.info("Pruned invalid example (sentinel reply_id): type='%s', key=%s", ex_type or "(empty)", key)
                    continue
                except Exception as exc:
                    logging.warning("Failed to delete sentinel example %s: %s", ex.id, exc)
            # For interactions/flagged_reply or anything else (default to requiring reply text)
            if require_generated_reply:
                reply_text = str(outputs.get("generated_reply") or "").strip()
                if not reply_text:
                    try:
                        self.client.delete_example(example_id=ex.id)
                        deleted += 1
                        key = meta.get("key") or meta.get("reply_id") or meta.get("tweet_id") or "unknown"
                        logging.info("Pruned invalid example (missing generated_reply): type='%s', key=%s", ex_type or "(empty)", key)
                    except Exception as exc:
                        logging.warning("Failed to delete example %s: %s", ex.id, exc)

        return deleted

    def verify_reply_visibility(
        self,
        dataset_name: str = "spider-interactions-dataset",
        delete_missing: bool = False,
        max_examples: Optional[int] = None,
        limit_keys: Optional[Iterable[str]] = None,
        timeout: int = 10,
    ) -> Dict[str, Any]:
        """Check if replies are still visible on X/Twitter.

        - Builds reply_url from metadata.reply_url or reply_id
        - Performs a lightweight HTTP GET and checks status and page markers
        - If delete_missing=True, deletes examples deemed missing
        - If limit_keys is provided, only examples whose key/url/reply_id match the provided set are checked
        Returns a summary with counts and a small sample of affected keys.
        """
        if not self.client:
            logging.error("LangSmith client not initialized.")
            return {"checked": 0, "visible": 0, "missing": 0, "unknown": 0, "deleted": 0, "samples": {}}

        try:
            dataset = self.client.read_dataset(dataset_name=dataset_name)
            examples = list(self.client.list_examples(dataset_id=dataset.id))
        except Exception as exc:
            logging.error("Failed to load dataset '%s': %s", dataset_name, exc)
            return {"checked": 0, "visible": 0, "missing": 0, "unknown": 0, "deleted": 0, "samples": {}}

        # Prepare limiting set when provided
        limit_set: Optional[Set[str]] = None
        if limit_keys:
            try:
                limit_set = {str(k) for k in limit_keys if k is not None}
            except Exception:
                limit_set = set()

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        }
        checked = visible = missing = unknown = deleted = 0
        missing_keys: List[str] = []
        unknown_keys: List[str] = []
        visible_keys: List[str] = []

        for i, ex in enumerate(examples):
            meta = ex.metadata or {}
            ex_type = str(meta.get("type") or "").strip().lower()
            if ex_type not in ("interaction", "flagged_reply"):
                continue
            # If limited, filter examples to the provided keys (accept match on key/url/reply_id/tweet_id/post_id)
            if limit_set is not None:
                inp = ex.inputs or {}
                candidates = [
                    meta.get("key"),
                    meta.get("reply_id"),
                    inp.get("url"),
                    meta.get("url"),
                    meta.get("tweet_id"),
                    meta.get("post_id"),
                ]
                cand_norm = {str(c) for c in candidates if c}
                if not (cand_norm & limit_set):
                    continue
            # Build reply URL
            reply_url = meta.get("reply_url")
            reply_id = str(meta.get("reply_id") or "").strip()
            if not reply_url and reply_id:
                reply_url = f"https://x.com/i/status/{reply_id}"
            if not reply_url:
                # Can't check
                unknown += 1
                if len(unknown_keys) < 10:
                    unknown_keys.append(str(meta.get("key") or reply_id or ""))
                continue

            # Fetch
            try:
                resp = requests.get(reply_url, headers=headers, timeout=timeout, allow_redirects=True)
                status = resp.status_code
                text = resp.text.lower() if isinstance(resp.text, str) else ""
                # Heuristics: 404/410 -> missing; 200 with tombstone markers -> missing; 302/301 to login -> unknown
                if status in (404, 410):
                    missing += 1
                    if len(missing_keys) < 10:
                        missing_keys.append(str(meta.get("key") or reply_id))
                    if delete_missing:
                        try:
                            self.client.delete_example(example_id=ex.id)
                            deleted += 1
                        except Exception as exc:
                            logging.warning("Failed to delete missing reply example %s: %s", ex.id, exc)
                    continue
                if status in (301, 302, 303, 307, 308):
                    # Likely login wall or redirect; mark unknown
                    unknown += 1
                    if len(unknown_keys) < 10:
                        unknown_keys.append(str(meta.get("key") or reply_id))
                    continue
                # For 200-range: look for common tombstone messages
                tombstones = (
                    "this post was deleted",
                    "this post is unavailable",
                    "account suspended",
                    "something went wrong",
                    "you are rate limited",
                )
                if any(tok in text for tok in tombstones):
                    missing += 1
                    if len(missing_keys) < 10:
                        missing_keys.append(str(meta.get("key") or reply_id))
                    if delete_missing:
                        try:
                            self.client.delete_example(example_id=ex.id)
                            deleted += 1
                        except Exception as exc:
                            logging.warning("Failed to delete missing reply example %s: %s", ex.id, exc)
                    continue

                # Otherwise consider visible
                visible += 1
                if len(visible_keys) < 10:
                    # Prefer reply_url for visibility samples
                    visible_keys.append(str(reply_url or meta.get("key") or reply_id))
            except Exception as exc:
                logging.debug("Reply check failed for %s: %s", reply_url, exc)
                unknown += 1
                if len(unknown_keys) < 10:
                    unknown_keys.append(str(meta.get("key") or reply_id))

            checked += 1
            if max_examples is not None and checked >= max_examples:
                break

        summary = {
            "checked": checked,
            "visible": visible,
            "missing": missing,
            "unknown": unknown,
            "deleted": deleted,
            "samples": {
                "visible_urls": visible_keys,
                "missing_keys": missing_keys,
                "unknown_keys": unknown_keys,
            },
        }
        logging.info("[verify] replies: checked=%d visible=%d missing=%d unknown=%d deleted=%d", checked, visible, missing, unknown, deleted)
        return summary

    def _scrape_x_metrics(self, url: str, *, show_browser: bool = False, wait_seconds: int = 8) -> Optional[Dict[str, int]]:
        """Fetch engagement metrics directly from a Twitter/X URL using headless Firefox.

        More robust: combines visible text + aria-label/title attributes and inspects action buttons
        to find counts adjacent to reply, repost, like, and views icons. Supports compact K/M/B suffixes.

        Returns a dict with keys: likes, replies, impressions, reposts; or None on failure.
        """
        try:
            from urllib.parse import urlparse
            import re
            import time
            import requests
            from bs4 import BeautifulSoup
            from selenium import webdriver
            from selenium.webdriver.firefox.options import Options

            host = urlparse(url).netloc.lower()
            if not (host.endswith("x.com") or host.endswith("twitter.com") or host.endswith("www.x.com") or host.endswith("www.twitter.com")):
                return None

            # Lightweight availability probe (non-fatal)
            try:
                head = requests.get(url, allow_redirects=True, timeout=10)
                if head.status_code >= 400:
                    logging.debug("_scrape_x_metrics: preliminary GET %s -> %s", url, head.status_code)
            except Exception:
                pass

            options = Options()
            if not show_browser:
                options.add_argument("--headless")
            options.set_preference("privacy.trackingprotection.enabled", False)
            options.set_preference("privacy.trackingprotection.pbmode.enabled", False)
            options.set_preference("privacy.trackingprotection.socialtracking.enabled", False)

            driver = webdriver.Firefox(options=options)
            try:
                driver.get(url)
                time.sleep(max(3, int(wait_seconds)))
                page_source = driver.page_source
            finally:
                driver.quit()

            soup = BeautifulSoup(page_source, "html.parser")

            # Collect attribute texts (aria-label/title) referencing metrics
            attr_chunks: List[str] = []
            for tag in soup.find_all(True):
                for attr in ("aria-label", "title"):
                    val = tag.get(attr)
                    if isinstance(val, str) and any(w in val.lower() for w in ("view", "like", "repl", "repost")):
                        attr_chunks.append(val)

            corpus = soup.get_text("\n") + "\n" + "\n".join(attr_chunks)

            # Helper for K/M/B suffixes
            def _parse(raw: str) -> int:
                try:
                    raw = raw.strip()
                    mult = 1
                    if raw.endswith(('K','k')):
                        mult = 1_000; raw = raw[:-1]
                    elif raw.endswith(('M','m')):
                        mult = 1_000_000; raw = raw[:-1]
                    elif raw.endswith(('B','b')):
                        mult = 1_000_000_000; raw = raw[:-1]
                    raw_clean = re.sub(r"[.,]", "", raw)
                    if raw_clean.isdigit():
                        return int(raw_clean) * mult
                    return int(float(raw_clean) * mult)
                except Exception:
                    return 0

            def _search(pattern: str) -> int:
                m = re.search(pattern, corpus, flags=re.IGNORECASE)
                return _parse(m.group(1)) if m else 0

            impressions = _search(r"([\d.,KMBkmb]+)\s+Views")
            likes = _search(r"([\d.,KMBkmb]+)\s+Likes?")
            replies = _search(r"([\d.,KMBkmb]+)\s+Repl(?:y|ies)")
            reposts = _search(r"([\d.,KMBkmb]+)\s+Reposts?")

            # If likes still zero, attempt structural extraction near buttons (data-testid="like")
            if likes == 0:
                try:
                    like_btn = soup.find(attrs={"data-testid": "like"}) or soup.find(attrs={"data-testid": "unlike"})
                    if like_btn:
                        # Traverse parent chain to find a sibling span/div with numeric text
                        parent = like_btn.parent
                        attempts = 0
                        while parent and attempts < 4 and likes == 0:
                            for sib in parent.find_all(True, recursive=False):
                                if sib is like_btn:
                                    continue
                                txt = sib.get_text(strip=True)
                                if re.match(r"^[\d.,KMBkmb]+$", txt):
                                    likes = _parse(txt)
                                    break
                            parent = parent.parent
                            attempts += 1
                except Exception:
                    pass

            # Same structural fallback for replies (data-testid="reply")
            if replies == 0:
                try:
                    reply_btn = soup.find(attrs={"data-testid": "reply"})
                    if reply_btn:
                        parent = reply_btn.parent
                        attempts = 0
                        while parent and attempts < 4 and replies == 0:
                            for sib in parent.find_all(True, recursive=False):
                                if sib is reply_btn:
                                    continue
                                txt = sib.get_text(strip=True)
                                if re.match(r"^[\d.,KMBkmb]+$", txt):
                                    replies = _parse(txt)
                                    break
                            parent = parent.parent
                            attempts += 1
                except Exception:
                    pass

            # Structural fallback for reposts (data-testid retweet)
            if reposts == 0:
                try:
                    repost_btn = soup.find(attrs={"data-testid": "retweet"}) or soup.find(attrs={"data-testid": "unretweet"})
                    if repost_btn:
                        parent = repost_btn.parent
                        attempts = 0
                        while parent and attempts < 4 and reposts == 0:
                            for sib in parent.find_all(True, recursive=False):
                                if sib is repost_btn:
                                    continue
                                txt = sib.get_text(strip=True)
                                if re.match(r"^[\d.,KMBkmb]+$", txt):
                                    reposts = _parse(txt)
                                    break
                            parent = parent.parent
                            attempts += 1
                except Exception:
                    pass

            metrics = {
                "impressions": impressions,
                "likes": likes,
                "replies": replies,
                "reposts": reposts,
            }

            # Sanity caps
            if metrics["impressions"] < 0 or metrics["impressions"] > 1_000_000_000:
                metrics["impressions"] = 0
            if metrics["likes"] < 0 or metrics["likes"] > 10_000_000:
                metrics["likes"] = 0
            if metrics["replies"] < 0 or metrics["replies"] > 1_000_000:
                metrics["replies"] = 0
            if metrics["reposts"] < 0 or metrics["reposts"] > 10_000_000:
                metrics["reposts"] = 0

            return metrics
        except Exception as exc:
            logging.debug("_scrape_x_metrics failed for %s: %s", url, exc)
            return None

    # ------------------------------
    # New: Upload scraped SQL articles (interactions, streamed, flagged)
    # ------------------------------

    def _ensure_dataset(self, dataset_name: str, description: str) -> None:
        if not self.client:
            return
        try:
            self.client.create_dataset(dataset_name=dataset_name, description=description)
        except Exception:
            pass

    def _list_examples_by_key(self, dataset_name: str, key_name: str = "key") -> Dict[str, Any]:
        """Return a mapping of key->example for an existing dataset based on metadata[key_name] or inputs[url].
        
        Also deduplicates by inputs+outputs (excluding metrics) to prevent multiple examples with identical content.
        For interactions: same tweet_text + url + generated_reply = duplicate
        For streamed posts: same tweet_text + url = duplicate (no reply to compare)
        If duplicates are found, keeps the one with the metadata key and deletes others.
        """
        existing_map: Dict[str, Any] = {}
        by_content: Dict[str, List[Any]] = {}  # Track all examples by content signature
        
        if not self.client:
            return existing_map
        try:
            dataset = self.client.read_dataset(dataset_name=dataset_name)
            all_examples = list(self.client.list_examples(dataset_id=dataset.id))
            
            # First pass: collect all examples and group by content signature (inputs + outputs minus metrics)
            for ex in all_examples:
                meta = ex.metadata or {}
                inputs = ex.inputs or {}
                outputs = ex.outputs or {}
                
                # Build a signature from inputs + non-metric outputs to detect duplicates
                url = str(inputs.get("url") or "").strip().lower()
                tweet_text = str(inputs.get("tweet_text") or "").strip()[:200]
                reply_text = str(outputs.get("generated_reply") or "").strip()[:200]
                
                content_sig = (url, tweet_text, reply_text)
                
                if content_sig not in by_content:
                    by_content[content_sig] = []
                by_content[content_sig].append(ex)
                
                # Still map by key for updates
                k = str(meta.get(key_name) or inputs.get("url") or meta.get("url") or "")
                if k:
                    existing_map[k] = ex
            
            # Second pass: find and delete duplicate examples (keep one per content signature)
            duplicates_deleted = 0
            for sig, examples in by_content.items():
                if len(examples) <= 1:
                    continue
                    
                # Multiple examples with same inputs+outputs - keep the one with metadata.key, or the newest
                examples_sorted = sorted(
                    examples, 
                    key=lambda e: (
                        1 if (e.metadata or {}).get(key_name) else 0,  # Prefer examples with key
                        (e.metadata or {}).get("created_at") or "",     # Then by created_at
                    ),
                    reverse=True
                )
                
                keeper = examples_sorted[0]
                for dup in examples_sorted[1:]:
                    try:
                        self.client.delete_example(example_id=dup.id)
                        duplicates_deleted += 1
                        logging.info("Deleted duplicate example id=%s (kept id=%s for content sig: url=%s, reply=%s)", 
                                   dup.id, keeper.id, sig[0][:50], sig[2][:30])
                    except Exception as exc:
                        logging.debug("Failed to delete duplicate example %s: %s", dup.id, exc)
            
            if duplicates_deleted > 0:
                logging.info("Deduplication complete: deleted %d duplicate examples", duplicates_deleted)
                
        except Exception as exc:
            logging.warning("Error during example listing/deduplication: %s", exc)
            
        return existing_map

    def upload_scraped_articles_dataset(
        self,
        db_path: str,
        dataset_name: str,
        filter_type: str,
        max_examples: Optional[int] = None,
    ) -> Dict[str, int]:
        """Upload records from SQL scraped_articles to a LangSmith dataset.

        - filter_type: one of {"interaction", "streamed_post", "flagged_reply"}
        - Upserts by unique key (tweet/reply URL) and preserves other examples
        """
        from spider_guardian.storage.sql import SQLDataStore

        if not self.client:
            logging.error("LangSmith client not initialized.")
            return {"created": 0, "updated": 0}

        desc = f"Spider Guardian {filter_type.replace('_', ' ')} records from scraped_articles"
        self._ensure_dataset(dataset_name, desc)

        existing_by_key = self._list_examples_by_key(dataset_name, key_name="key")

        store = SQLDataStore(db_path)
        created = 0
        updated = 0
        to_create_inputs: List[Dict[str, Any]] = []
        to_create_outputs: List[Dict[str, Any]] = []
        to_create_metadata: List[Dict[str, Any]] = []
        
        # Track content signatures (inputs + outputs minus metrics) to prevent creating duplicates within this batch
        seen_content_sigs: set[tuple[str, str, str]] = set()

        count = 0
        show_progress = bool(os.getenv("UPD_PROGRESS"))
        use_new = False
        try:
            use_new = store._table_exists("interactions") or store._table_exists("content")
        except Exception:
            use_new = False

        if use_new:
            # Prefer normalized tables
            if filter_type in ("interaction", "flagged_reply"):
                # Build a lookup of content/trending posts by multiple keys for input metrics resolution
                cont_by_key: Dict[str, Any] = {}
                try:
                    for r in store.iter_content():
                        pid = str(r.get("post_id") or "").strip()
                        url_r = r.get("url") or (f"https://x.com/i/status/{pid}" if pid else "")
                        for k in (pid, url_r):
                            if k:
                                cont_by_key[str(k)] = r
                except Exception:
                    pass
                
                iterator = store.iter_interactions()
                for row in iterator:
                    if row.get("type") != filter_type:
                        continue
                    # Skip rows where a reply was supposedly posted but we never captured its ID
                    try:
                        if str(row.get("reply_id") or "").strip() == "posted_no_id_found":
                            continue
                    except Exception:
                        pass
                    try:
                        # Original tweet URL (what we replied to)
                        orig_tweet_id = str(row.get("tweet_id") or "").strip()
                        url = row.get("url") or (f"https://x.com/i/status/{orig_tweet_id}" if orig_tweet_id else "")
                        
                        # Reply URL (our bot's reply) - this is the unique key
                        reply_id = str(row.get("reply_id") or "").strip()
                        reply_url_val = f"https://x.com/i/status/{reply_id}" if reply_id else None
                        # Resolve canonical reply URL like trending path does, to improve scraping reliability
                        try:
                            if reply_url_val:
                                resolved = fetch_final_url_with_selenium(reply_url_val, headless=True, timeout_seconds=8)
                                if resolved and isinstance(resolved, str):
                                    reply_url_val = resolved
                        except Exception as exc:
                            logging.debug("reply_url resolution failed for %s: %s", reply_url_val, exc)
                        key = reply_url_val or reply_id or url  # Prefer reply URL as unique key
                        
                        # Resolve original tweet metrics from content table (and optionally live scrape as fallback)
                        input_metrics: Dict[str, int] = {}
                        orig_row = None
                        if orig_tweet_id:
                            orig_row = cont_by_key.get(orig_tweet_id)
                        if orig_row is None and url:
                            orig_row = cont_by_key.get(url)
                        if orig_row is not None:
                            input_metrics = {
                                "likes": int(orig_row.get("like_count", 0) or 0),
                                "replies": int(orig_row.get("reply_count", 0) or 0),
                                "impressions": int(orig_row.get("impression_count", 0) or 0),
                                "reposts": int(orig_row.get("repost_count", 0) or 0),
                            }
                        # If still missing/zero and we have the original tweet URL, scrape it to populate true input metrics
                        try:
                            live_on_upload_inputs = os.getenv("LIVE_SCRAPE_ORIG_ON_UPLOAD", "1") not in ("0", "false", "False")
                        except Exception:
                            live_on_upload_inputs = True
                        if (not input_metrics or int(input_metrics.get("impressions", 0) or 0) == 0) and url and live_on_upload_inputs:
                            try:
                                resolved_orig = fetch_final_url_with_selenium(url, headless=True, timeout_seconds=8) or url
                                live_in = self._scrape_x_metrics(resolved_orig, show_browser=False, wait_seconds=6)
                                if live_in:
                                    input_metrics = {
                                        "likes": int(live_in.get("likes", 0) or 0),
                                        "replies": int(live_in.get("replies", 0) or 0),
                                        "impressions": int(live_in.get("impressions", 0) or 0),
                                        "reposts": int(live_in.get("reposts", 0) or 0),
                                    }
                            except Exception as exc:
                                logging.debug("Live scrape for original tweet failed for %s: %s", url, exc)
                        
                        inputs = {
                            "tweet_text": row.get("tweet_text", ""),
                            "url": url,
                        }
                        # Add input (original tweet) engagement metrics to inputs if available
                        if input_metrics:
                            inputs["engagement_metrics"] = input_metrics
                        
                        # Check for duplicate content signature (inputs + outputs minus metrics) in this batch
                        reply_text = str(row.get("reply_text", "")).strip()[:200]
                        content_sig = (
                            str(url or "").strip().lower(),
                            str(row.get("tweet_text", "")).strip()[:200],
                            reply_text
                        )
                        if content_sig in seen_content_sigs and content_sig[0] and content_sig[1]:
                            logging.debug("Skipping duplicate content in batch: url=%s, reply=%s", url, reply_text[:30])
                            continue
                        
                        # Reply metrics from interactions table; optionally live-scrape Views/metrics from reply_url
                        impressions_val = int(row.get("impression_count", 0) or 0)
                        reply_metrics = {
                            "likes": int(row.get("like_count", 0) or 0),
                            "replies": int(row.get("reply_count", 0) or 0),
                            "impressions": impressions_val,
                        }
                        # If DB has zero/low metrics and we have a reply_url, scrape it now to populate Views like trending path
                        try:
                            live_on_upload = os.getenv("LIVE_SCRAPE_ON_UPLOAD", "1") not in ("0", "false", "False")
                        except Exception:
                            live_on_upload = True
                        if reply_url_val and live_on_upload:
                            try:
                                live = self._scrape_x_metrics(reply_url_val, show_browser=False, wait_seconds=6)
                                if live:
                                    # Override when live has non-zero values
                                    for k in ("likes", "replies", "impressions", "reposts"):
                                        try:
                                            v = int(live.get(k, 0) or 0)
                                        except Exception:
                                            v = 0
                                        if k == "impressions" and v > 0:
                                            impressions_val = v
                                        if v > 0:
                                            if k == "reposts":
                                                # maintain compatibility; we don't store reposts in reply_metrics elsewhere yet
                                                pass
                                            else:
                                                reply_metrics[k] = v
                            except Exception as exc:
                                logging.debug("Live scrape on upload failed for %s: %s", reply_url_val, exc)
                        outputs = {
                            "generated_reply": row.get("reply_text", ""),
                            "reply_url": reply_url_val,
                            # Reply-only metrics
                            "reply_engagement_metrics": reply_metrics,
                            # Backwards compatibility: engagement_metrics continues to mean reply metrics for interactions
                            "engagement_metrics": reply_metrics,
                            # Top-level convenience fields for LangSmith table view
                            "reply_impressions": impressions_val,
                            "reply_likes": int(reply_metrics.get("likes", 0) or 0),
                            "reply_replies": int(reply_metrics.get("replies", 0) or 0),
                            "reply_reposts": int(reply_metrics.get("reposts", 0) or 0),
                            "impressions": impressions_val,  # legacy reply impressions
                        }
                        # Original tweet metrics live ONLY in inputs.engagement_metrics (not outputs) to avoid confusion
                        md = {
                            "key": key,
                            "tweet_id": str(row.get("tweet_id") or ""),
                            "reply_id": str(row.get("reply_id") or ""),
                            "reply_url": reply_url_val,
                            "type": row.get("type"),
                            "level": 1,  # reply depth relative to original tweet
                            "created_at": str(row.get("created_at") or datetime.utcnow().isoformat()),
                            # 'completed' indicates the cadence is finished
                            "completed": bool((row.get("metadata") or {}).get("completed", False)),
                            **(row.get("metadata") or {}),
                            # New: store metrics at metadata level
                            "input_metrics": input_metrics,
                            "reply_metrics": reply_metrics,
                        }
                        action = None
                        if key in existing_by_key:
                            try:
                                old_ex = existing_by_key[key]
                                old_meta = old_ex.metadata or {}
                                old_inputs = old_ex.inputs or {}
                                old_outputs = old_ex.outputs or {}

                                # Preserve counters and created_at
                                if old_meta.get("created_at"):
                                    md["created_at"] = old_meta["created_at"]
                                for f in ("updates_done", "reply_url", "completed"):
                                    if old_meta.get(f) is not None:
                                        md[f] = old_meta.get(f)
                                # Backfill reply_url if missing but reply_id exists
                                if not md.get("reply_url") and (md.get("reply_id") or ""):
                                    rid = str(md.get("reply_id") or "").strip()
                                    if rid:
                                        md["reply_url"] = f"https://x.com/i/status/{rid}"
                                # Ensure canonical URL saved when we resolved it
                                if reply_url_val:
                                    md["reply_url"] = reply_url_val

                                # Check cadence: don't update unless enough time has passed and not exhausted
                                cadence_schedule = [1, 2, 3]  # days between updates
                                now = datetime.now()
                                try:
                                    updates_done_int = int(md.get("updates_done", 0) or 0)
                                except Exception:
                                    updates_done_int = 0
                                
                                # If schedule exhausted, skip update
                                if md.get("completed") is True or updates_done_int >= len(cadence_schedule):
                                    continue
                                
                                required_wait_days = cadence_schedule[updates_done_int]
                                
                                # Determine reference time
                                if updates_done_int == 0:
                                    # First update: wait from created_at
                                    ref_str = md.get("created_at")
                                else:
                                    # Subsequent: wait from modified_at
                                    ref_str = md.get("modified_at")
                                
                                if ref_str:
                                    try:
                                        ref_ts = datetime.fromisoformat(str(ref_str))
                                        elapsed_days = (now - ref_ts).total_seconds() / 86400
                                        if elapsed_days < required_wait_days:
                                            # Too soon
                                            continue
                                    except Exception:
                                        pass  # if parse fails, allow update

                                # Check if content changed (inputs or generated_reply)
                                # Detect content changes excluding metrics (compare tweet_text + generated_reply)
                                content_changed = (
                                    str(inputs.get("tweet_text") or "") != str(old_inputs.get("tweet_text") or "") or
                                    str(outputs.get("generated_reply") or "") != str(old_outputs.get("generated_reply") or "")
                                )
                                
                                if content_changed:
                                    md["modified_at"] = now.isoformat()
                                    md["last_update_forced"] = False
                                    # Don't increment updates_done for content changes; preserve existing value
                                    # Only refresh function should increment for metric updates
                                    # Cadence completion marking
                                    if updates_done_int + 1 >= len(cadence_schedule):
                                        md["completed"] = True
                                    # Preserve reply_url in outputs if it was there before
                                    if not outputs.get("reply_url") and old_outputs.get("reply_url"):
                                        outputs["reply_url"] = old_outputs["reply_url"]
                                    self.client.update_example(
                                        example_id=old_ex.id,
                                        inputs=inputs,
                                        outputs=outputs,
                                        metadata=self._normalize_example_metadata({
                                            **md,
                                            "input_metrics": input_metrics or md.get("input_metrics") or {},
                                            "reply_metrics": reply_metrics or md.get("reply_metrics") or {},
                                        }),
                                    )
                                    updated += 1
                                    action = "updated"
                            except Exception as exc:
                                logging.debug("Failed to update example for key=%s: %s", key, exc)
                        else:
                            md["created_at"] = md.get("created_at")
                            md["modified_at"] = datetime.now().isoformat()
                            # Initialize cadence counters on creation if missing
                            md.setdefault("updates_done", 0)
                            try:
                                md["updates_done"] = min(int(md.get("updates_done") or 0), 3)
                            except Exception:
                                md["updates_done"] = 0
                            if md.get("updates_done", 0) >= len(cadence_schedule) - 1:
                                md["completed"] = True
                            to_create_inputs.append(inputs)
                            to_create_outputs.append(outputs)
                            to_create_metadata.append(self._normalize_example_metadata(md))
                            seen_content_sigs.add(content_sig)
                            created += 1
                            action = "created"

                        count += 1
                        if show_progress and action:
                            print(f"[{count}] {filter_type}: {action} key={key}", flush=True)
                        if max_examples is not None and count >= max_examples:
                            break
                    except Exception as exc:
                        logging.debug("Skipping interaction row due to error: %s", exc)
                        continue
            else:  # streamed_post
                iterator = store.iter_content()
                for row in iterator:
                    try:
                        url = row.get("url") or (f"https://x.com/i/status/{row.get('post_id')}" if row.get("post_id") else "")
                        key = url or str(row.get("post_id") or "")
                        inputs = {
                            "tweet_text": row.get("text", ""),
                            "url": url,
                        }
                        impressions_val = int(row.get("impression_count", 0) or 0)
                        likes_val = int(row.get("like_count", 0) or 0)
                        replies_val = int(row.get("reply_count", 0) or 0)
                        reposts_val = int(row.get("repost_count", 0) or 0)
                        outputs = {
                            "engagement_metrics": {
                                "likes": likes_val,
                                "replies": replies_val,
                                "impressions": impressions_val,
                                "reposts": reposts_val,
                            },
                            # Top-level convenience fields for LangSmith table view
                            "impressions": impressions_val,
                            "likes": likes_val,
                            "replies": replies_val,
                            "reposts": reposts_val,
                        }
                        md = {
                            "key": key,
                            "tweet_id": str(row.get("post_id") or ""),
                            "author": row.get("author_handle"),
                            "lang": row.get("lang"),
                            "type": "streamed_post",
                            "level": 0,  # original tweet depth
                            "created_at": str(row.get("created_at") or datetime.utcnow().isoformat()),
                            "num_updates": int((row.get("metadata") or {}).get("num_updates", 0) or 0),
                            **(row.get("metadata") or {}),
                            # Metadata metrics (no reply metrics for streamed posts)
                            "input_metrics": {
                                "likes": likes_val,
                                "replies": replies_val,
                                "impressions": impressions_val,
                                "reposts": reposts_val,
                            },
                            "reply_metrics": None,
                        }
                        action = None
                        if key in existing_by_key:
                            try:
                                old_ex = existing_by_key[key]
                                old_meta = old_ex.metadata or {}
                                old_inputs = old_ex.inputs or {}
                                old_outputs = old_ex.outputs or {}

                                if old_meta.get("created_at"):
                                    md["created_at"] = old_meta["created_at"]
                                for f in ("updates_done", "completed"):
                                    if old_meta.get(f) is not None:
                                        md[f] = old_meta.get(f)

                                # Check cadence for streamed posts too
                                cadence_schedule = [1, 2, 3]
                                now = datetime.now()
                                try:
                                    updates_done_int = int(md.get("updates_done", 0) or 0)
                                except Exception:
                                    updates_done_int = 0
                                
                                if md.get("completed") is True or updates_done_int >= len(cadence_schedule):
                                    continue
                                
                                required_wait_days = cadence_schedule[updates_done_int]
                                if updates_done_int == 0:
                                    ref_str = md.get("created_at")
                                else:
                                    ref_str = md.get("modified_at")
                                
                                if ref_str:
                                    try:
                                        ref_ts = datetime.fromisoformat(str(ref_str))
                                        elapsed_days = (now - ref_ts).total_seconds() / 86400
                                        if elapsed_days < required_wait_days:
                                            continue
                                    except Exception:
                                        pass

                                # For streamed posts, update only if metrics or inputs changed
                                metrics_changed = (outputs.get("engagement_metrics") or {}) != (old_outputs.get("engagement_metrics") or {})
                                inputs_changed = (inputs or {}) != (old_inputs or {})
                                if metrics_changed or inputs_changed:
                                    md["modified_at"] = now.isoformat()
                                    # Don't increment updates_done for content/metric changes from upload
                                    # Only refresh function increments this for cadence tracking
                                    if updates_done_int + 1 >= len(cadence_schedule):
                                        md["completed"] = True
                                    self.client.update_example(
                                        example_id=old_ex.id,
                                        inputs=inputs,
                                        outputs=outputs,
                                        metadata=self._normalize_example_metadata({
                                            **md,
                                            "input_metrics": md.get("input_metrics") or {
                                                "likes": likes_val,
                                                "replies": replies_val,
                                                "impressions": impressions_val,
                                                "reposts": reposts_val,
                                            },
                                            "reply_metrics": None,
                                        }),
                                    )
                                    updated += 1
                                    action = "updated"
                            except Exception as exc:
                                logging.debug("Failed to update example for key=%s: %s", key, exc)
                        else:
                            md["created_at"] = md.get("created_at")
                            md["modified_at"] = datetime.now().isoformat()
                            md.setdefault("updates_done", 0)
                            try:
                                md["updates_done"] = min(int(md.get("updates_done") or 0), 3)
                            except Exception:
                                md["updates_done"] = 0
                            if md.get("updates_done", 0) >= len(cadence_schedule) - 1:
                                md["completed"] = True
                            to_create_inputs.append(inputs)
                            to_create_outputs.append(outputs)
                            to_create_metadata.append(self._normalize_example_metadata(md))
                            created += 1
                            action = "created"

                        count += 1
                        if show_progress and action:
                            print(f"[{count}] {filter_type}: {action} key={key}", flush=True)
                        if max_examples is not None and count >= max_examples:
                            break
                    except Exception as exc:
                        logging.debug("Skipping content row due to error: %s", exc)
                        continue
        else:
            # Legacy fallback: scraped_articles
            for article in store.iter_scraped_articles():
                meta_type = (article.metadata or {}).get("type")
                if meta_type != filter_type:
                    continue
                try:
                    content = article.content
                    if isinstance(content, str):
                        try:
                            import json as _json
                            content = _json.loads(content)
                        except Exception:
                            content = {}
                    content = content or {}
                    tweet_id = str(content.get("tweet_id") or "")
                    url = content.get("url") or article.link or (f"https://x.com/i/status/{tweet_id}" if tweet_id else article.link)
                    key = url or tweet_id or article.link
                    if filter_type == "interaction":
                        reply_url_legacy = f"https://x.com/i/status/{content.get('reply_id')}" if content.get("reply_id") else None
                        inputs = {"tweet_text": content.get("tweet_text", ""), "url": url}
                        outputs = {
                            "generated_reply": content.get("reply_text", ""),
                            "reply_url": reply_url_legacy,
                            "engagement_metrics": content.get("metrics", {}) or {"likes": 0, "replies": 0, "impressions": 0},
                        }
                        md = {
                            "key": key,
                            "tweet_id": tweet_id,
                            "reply_id": str(content.get("reply_id") or ""),
                            "reply_url": reply_url_legacy,
                            "tone": content.get("tone"),
                            "model": content.get("model"),
                            "type": meta_type,
                            "created_at": article.created_at.isoformat(),
                            # Legacy path: provide metrics under metadata
                            "input_metrics": content.get("input_metrics") or {},
                            "reply_metrics": content.get("metrics") or {},
                        }
                    elif filter_type == "streamed_post":
                        inputs = {"tweet_text": content.get("text", ""), "url": url}
                        outputs = {
                            "engagement_metrics": {
                                "likes": int(content.get("like_count", 0) or 0),
                                "replies": int(content.get("reply_count", 0) or 0),
                                "impressions": int(content.get("impression_count", 0) or 0),
                            }
                        }
                        md = {
                            "key": key,
                            "tweet_id": str(content.get("id") or ""),
                            "author": content.get("author_handle"),
                            "lang": content.get("lang"),
                            "type": meta_type,
                            "created_at": article.created_at.isoformat(),
                            "input_metrics": {
                                "likes": int(content.get("like_count", 0) or 0),
                                "replies": int(content.get("reply_count", 0) or 0),
                                "impressions": int(content.get("impression_count", 0) or 0),
                                "reposts": 0,  # legacy scraped_articles may not have reposts
                            },
                            "reply_metrics": None,
                        }
                    else:  # flagged_reply
                        inputs = {"reply_text": content.get("reply_text", article.title or ""), "url": url}
                        outputs = {"label": "flagged"}
                        md = {"key": key, "reason": content.get("reason"), "type": meta_type, "created_at": article.created_at.isoformat()}
                    action = None
                    if key in existing_by_key:
                        try:
                            old_meta = existing_by_key[key].metadata or {}
                            if old_meta.get("created_at"):
                                md["created_at"] = old_meta["created_at"]
                            if old_meta.get("reply_url") and not md.get("reply_url"):
                                md["reply_url"] = old_meta.get("reply_url")
                            if not md.get("reply_url") and (md.get("reply_id") or ""):
                                rid = str(md.get("reply_id") or "").strip()
                                if rid:
                                    md["reply_url"] = f"https://x.com/i/status/{rid}"
                            md["modified_at"] = datetime.now().isoformat()
                            # Preserve cadence counters if present and cap updates_done
                            if old_meta.get("updates_done") is not None:
                                try:
                                    md["updates_done"] = min(int(old_meta.get("updates_done") or 0), 3)
                                except Exception:
                                    md["updates_done"] = 0
                            else:
                                md.setdefault("updates_done", 0)
                            if old_meta.get("num_updates") is not None:
                                md["num_updates"] = old_meta.get("num_updates")
                            md.setdefault("num_updates", int(md.get("num_updates") or 0))
                            self.client.update_example(
                                example_id=existing_by_key[key].id,
                                inputs=inputs,
                                outputs=outputs,
                                metadata=md,
                            )
                            updated += 1
                            action = "updated"
                        except Exception as exc:
                            logging.debug("Failed to update example for key=%s: %s", key, exc)
                    else:
                        md["created_at"] = md.get("created_at") or article.created_at.isoformat()
                        md["modified_at"] = datetime.now().isoformat()
                        md.setdefault("updates_done", 0)
                        try:
                            md["updates_done"] = min(int(md.get("updates_done") or 0), 3)
                        except Exception:
                            md["updates_done"] = 0
                        md.setdefault("num_updates", int(md.get("num_updates") or 0))
                        to_create_inputs.append(inputs)
                        to_create_outputs.append(outputs)
                        to_create_metadata.append(md)
                        created += 1
                        action = "created"
                    count += 1
                    if show_progress and action:
                        print(f"[{count}] {filter_type}: {action} key={key}", flush=True)
                    if max_examples is not None and count >= max_examples:
                        break
                except Exception as exc:
                    logging.debug("Skipping article due to error: %s", exc)
                    continue

        if to_create_inputs:
            try:
                self.client.create_examples(
                    inputs=to_create_inputs,
                    outputs=to_create_outputs,
                    metadata=to_create_metadata,
                    dataset_name=dataset_name,
                )
            except Exception as exc:
                logging.error("Failed to create %d examples: %s", len(to_create_inputs), exc)

        logging.info(
            "Uploaded to %s (%s): %d created, %d updated",
            dataset_name,
            filter_type,
            created,
            updated,
        )
        
        # Auto-prune invalid examples after upload
        try:
            pruned = self.prune_invalid_examples(dataset_name=dataset_name)
            if pruned > 0:
                logging.info("Auto-pruned %d invalid examples after upload", pruned)
        except Exception as exc:
            logging.warning("Auto-prune failed (non-fatal): %s", exc)
        
        return {"created": created, "updated": updated}

    def upload_replies_from_sql(self, db_path: str, dataset_name: str = "spider-interactions-dataset", max_examples: Optional[int] = None) -> Dict[str, int]:
        return self.upload_scraped_articles_dataset(db_path, dataset_name, filter_type="interaction", max_examples=max_examples)

    def upload_streamed_from_sql(self, db_path: str, dataset_name: str = "spider-streamed-dataset", max_examples: Optional[int] = None) -> Dict[str, int]:
        return self.upload_scraped_articles_dataset(db_path, dataset_name, filter_type="streamed_post", max_examples=max_examples)

    def upload_flagged_from_sql(self, db_path: str, dataset_name: str = "spider-interactions-dataset", max_examples: Optional[int] = None) -> Dict[str, int]:
        return self.upload_scraped_articles_dataset(db_path, dataset_name, filter_type="flagged_reply", max_examples=max_examples)

    def generate_performance_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate a performance report from LangSmith data."""
        if not self.client:
            return {}

        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)

            runs = list(
                self.client.list_runs(
                    project_name=self.project_name,
                    start_time=start_time,
                    end_time=end_time,
                )
            )

            reply_runs = [run for run in runs if run.name == "reply_generation"]
            engagement_runs = [run for run in runs if run.name == "engagement_tracking"]

            report = {
                "period": f"{days} days",
                "total_replies_generated": len(reply_runs),
                "total_engagement_tracked": len(engagement_runs),
                "avg_generation_time_ms": 0,
                "total_engagement": {
                    "likes": 0,
                    "replies": 0,
                    "impressions": 0,
                },
            }

            if reply_runs:
                times = [run.extra.get("generation_time_ms", 0) for run in reply_runs]
                report["avg_generation_time_ms"] = sum(times) / len(times)

            for run in engagement_runs:
                if run.outputs:
                    report["total_engagement"]["likes"] += run.outputs.get("likes", 0)
                    report["total_engagement"]["replies"] += run.outputs.get("replies", 0)
                    report["total_engagement"]["impressions"] += run.outputs.get("impressions", 0)

            logging.info("Generated performance report: %s", report)
            return report

        except Exception as exc:
            logging.error("Failed to generate performance report: %s", exc)
            return {}

    def get_langsmith_url(self) -> str:
        """Return the LangSmith project URL."""
        if not self.client:
            return "LangSmith not configured"
        return f"https://smith.langchain.com/projects/{self.project_name}"

    # ---- Experiment helpers for impressions optimization ------------------
    def fetch_author_follower_count(self, author_handle: str) -> Optional[int]:
        """Fetch follower count for a Twitter/X author using selenium scraping.
        
        Args:
            author_handle: Twitter handle (without @)
            
        Returns:
            Follower count or None if unavailable
        """
        if not author_handle or not isinstance(author_handle, str):
            return None
            
        # Clean handle (remove @ if present)
        handle = author_handle.lstrip("@").strip()
        if not handle:
            return None
            
        try:
            from urllib.parse import urlparse
            from selenium import webdriver
            from selenium.webdriver.firefox.options import Options
            import time
            from bs4 import BeautifulSoup
            import re
            
            profile_url = f"https://x.com/{handle}"
            
            options = Options()
            # options.add_argument("--headless")
            options.set_preference("privacy.trackingprotection.enabled", False)
            options.set_preference("privacy.trackingprotection.pbmode.enabled", False)
            options.set_preference("privacy.trackingprotection.socialtracking.enabled", False)
            
            driver = webdriver.Firefox(options=options)
            try:
                driver.get(profile_url)
                time.sleep(4)  # Reduced from 4s - profile loads quickly
                page_source = driver.page_source
            finally:
                driver.quit()
            
            soup = BeautifulSoup(page_source, "html.parser")
            
            # Look for follower count patterns in text and aria-labels
            texts = [soup.get_text()]
            for tag in soup.find_all(True):
                for attr in ("aria-label", "title"):
                    val = tag.get(attr)
                    if isinstance(val, str) and ("follower" in val.lower() or "abonné" in val.lower()):
                        texts.append(val)

            corpus = "\n".join(texts)

            # Helper to parse K/M/B suffixes
            def _parse_count(raw: str) -> int:
                try:
                    raw = raw.strip()
                    mult = 1
                    if raw.endswith(('K','k')):
                        mult = 1_000; raw = raw[:-1]
                    elif raw.endswith(('M','m')):
                        mult = 1_000_000; raw = raw[:-1]
                    elif raw.endswith(('B','b')):
                        mult = 1_000_000_000; raw = raw[:-1]
                    raw_clean = re.sub(r"[.,]", "", raw)
                    if raw_clean.isdigit():
                        return int(raw_clean) * mult
                    return int(float(raw_clean) * mult)
                except Exception:
                    return 0

            # Try English pattern first
            match_en = re.search(r"([\d.,KMBkmb]+)\s+Followers?", corpus, flags=re.IGNORECASE)
            if match_en:
                count = _parse_count(match_en.group(1))
                if 0 < count < 1_000_000_000:
                    logging.info("Fetched follower count for @%s: %d (en)", handle, count)
                    return count

            # Try French pattern: match only 'abonnés' (followers), not 'abonnements' (following)
            # Ensure the number is directly followed by 'abonnés' (no space, no 'abonnements')
            match_fr = re.search(r"abonnements([\d.,KMBkmb]+)\s*", corpus, flags=re.IGNORECASE)
            if match_fr:
                count = _parse_count(match_fr.group(1))
                if 0 < count < 1_000_000_000:
                    logging.info("Fetched follower count for @%s: %d (fr)", handle, count)
                    return count

            return None
            
        except Exception as exc:
            logging.debug("Failed to fetch follower count for @%s: %s", handle, exc)
            return None
    
    def enrich_dataset_with_follower_counts(
        self,
        dataset_name: str,
        max_examples: Optional[int] = None,
        skip_existing: bool = True,
    ) -> Dict[str, Any]:
        """Add author follower counts to dataset examples metadata.
        
        Fetches follower counts from Twitter profiles and stores in metadata.author_followers.
        Rate-limited to avoid overwhelming Twitter.
        
        Args:
            dataset_name: Dataset to enrich
            max_examples: Maximum examples to process
            skip_existing: Skip examples that already have follower count
            
        Returns:
            Summary with counts of enriched examples
        """
        if not self.client:
            logging.error("LangSmith client not initialized.")
            return {"enriched": 0, "skipped": 0, "failed": 0}
        
        try:
            dataset = self.client.read_dataset(dataset_name=dataset_name)
            examples = list(self.client.list_examples(dataset_id=dataset.id))
            logging.info("Enriching %d examples with follower counts", len(examples))
        except Exception as exc:
            logging.error("Failed to load dataset: %s", exc)
            return {"error": str(exc)}
        
        enriched = 0
        skipped = 0
        failed = 0
        
        for i, ex in enumerate(examples):
            if max_examples is not None and i >= max_examples:
                break
                
            meta = ex.metadata or {}
            inputs = ex.inputs or {}
            
            # Skip if already has follower count
            if skip_existing and meta.get("author_followers") is not None:
                skipped += 1
                continue
            
            # Get author handle
            author = meta.get("author") or inputs.get("author") or meta.get("author_handle")
            if not author:
                skipped += 1
                continue
            
            # Fetch follower count
            follower_count = self.fetch_author_follower_count(author)
            
            if follower_count is not None:
                new_meta = dict(meta)
                new_meta["author_followers"] = follower_count
                new_meta["author_followers_fetched_at"] = datetime.utcnow().isoformat()
                
                try:
                    self.client.update_example(
                        example_id=ex.id,
                        metadata=self._normalize_example_metadata(new_meta)
                    )
                    enriched += 1
                    logging.info("[%d/%d] Enriched @%s with %d followers", 
                               i+1, len(examples), author, follower_count)
                except Exception as exc:
                    logging.warning("Failed to update example: %s", exc)
                    failed += 1
                
                # Rate limit: wait between requests
                time.sleep(4)
            else:
                failed += 1
        
        summary = {
            "enriched": enriched,
            "skipped": skipped,
            "failed": failed,
            "total_processed": enriched + skipped + failed,
        }
        logging.info("Enrichment complete: %s", summary)
        return summary
    
    def create_experiment_dataset(
        self,
        source_dataset: str,
        experiment_name: str,
        max_examples: Optional[int] = None,
        min_impressions: Optional[int] = None,
        min_followers: Optional[int] = None,
        filter_completed: bool = True,
    ) -> str:
        """Clone a subset of examples into a new experiment dataset.
        
        Args:
            source_dataset: Name of the source dataset to clone from
            experiment_name: Name for the new experiment dataset
            max_examples: Maximum number of examples to include
            min_impressions: Only include examples with input impressions >= this value
            min_followers: Only include authors with followers >= this value
            filter_completed: If True, exclude examples with completed=True
            
        Returns:
            Name of the created experiment dataset
        """
        if not self.client:
            logging.error("LangSmith client not initialized.")
            return ""
            
        exp_dataset_name = f"{source_dataset}-exp-{experiment_name}"
        
        try:
            # Load source examples
            source_ds = self.client.read_dataset(dataset_name=source_dataset)
            examples = list(self.client.list_examples(dataset_id=source_ds.id))
            logging.info("Loaded %d examples from source dataset '%s'", len(examples), source_dataset)
            
            # Filter and prioritize
            filtered = []
            for ex in examples:
                meta = ex.metadata or {}
                # Skip completed if requested
                if filter_completed and meta.get("completed") is True:
                    continue
                    
                # Check minimum impressions
                if min_impressions is not None:
                    input_met = (meta.get("input_metrics") or {})
                    impr = int(input_met.get("impressions", 0) or 0)
                    if impr < min_impressions:
                        continue
                
                # Check minimum follower count
                if min_followers is not None:
                    try:
                        followers = int(meta.get("author_followers", 0) or 0)
                    except Exception:
                        followers = 0
                    if followers < min_followers:
                        continue
                        
                filtered.append(ex)
            
            # Prioritize: highest followers first, then lowest updates_done, then highest impressions
            def _priority(ex) -> tuple:
                m = ex.metadata or {}
                try:
                    u = int(m.get("updates_done", 0) or 0)
                except Exception:
                    u = 0
                try:
                    followers = int(m.get("author_followers", 0) or 0)
                except Exception:
                    followers = 0
                try:
                    input_met = m.get("input_metrics") or {}
                    impr = int(input_met.get("impressions", 0) or 0)
                except Exception:
                    impr = 0
                cstr = m.get("created_at") or m.get("collected_at")
                try:
                    cts = datetime.fromisoformat(str(cstr)) if cstr else datetime.max
                except Exception:
                    cts = datetime.max
                # Sort by: followers desc, updates_done asc, impressions desc, oldest first
                return (-followers, u, -impr, cts)
            
            filtered = sorted(filtered, key=_priority)
            
            if max_examples is not None:
                filtered = filtered[:max_examples]
                
            logging.info("Selected %d examples for experiment dataset", len(filtered))
            
            # Create new dataset
            try:
                self.client.create_dataset(
                    dataset_name=exp_dataset_name,
                    description=f"Experiment: {experiment_name} | Source: {source_dataset}",
                )
                logging.info("Created experiment dataset: %s", exp_dataset_name)
            except Exception as exc:
                logging.warning("Dataset may already exist: %s", exc)
            
            # Copy examples with experiment metadata
            to_create_inputs = []
            to_create_outputs = []
            to_create_metadata = []
            
            for ex in filtered:
                meta = dict(ex.metadata or {})
                meta["experiment"] = experiment_name
                meta["experiment_created_at"] = datetime.utcnow().isoformat()
                meta["source_dataset"] = source_dataset
                
                to_create_inputs.append(ex.inputs or {})
                to_create_outputs.append(ex.outputs or {})
                to_create_metadata.append(meta)
            
            if to_create_inputs:
                self.client.create_examples(
                    inputs=to_create_inputs,
                    outputs=to_create_outputs,
                    metadata=to_create_metadata,
                    dataset_name=exp_dataset_name,
                )
                logging.info("Created %d examples in experiment dataset", len(to_create_inputs))
            
            return exp_dataset_name
            
        except Exception as exc:
            logging.error("Failed to create experiment dataset: %s", exc)
            return ""
    
    def evaluate_impressions_experiment(
        self,
        experiment_dataset: str,
        window_days: int = 3,
    ) -> Dict[str, Any]:
        """Evaluate experiment by measuring impression performance.
        
        Compares reply impressions to input (original tweet) impressions
        to calculate relative engagement lift.
        
        Args:
            experiment_dataset: Name of the experiment dataset
            window_days: Number of days since reply creation to consider
            
        Returns:
            Dict with metrics: total_replies, avg_reply_impressions, avg_input_impressions,
            impression_lift_ratio, top_performers (list of high-performing examples)
        """
        if not self.client:
            logging.error("LangSmith client not initialized.")
            return {}
            
        try:
            dataset = self.client.read_dataset(dataset_name=experiment_dataset)
            examples = list(self.client.list_examples(dataset_id=dataset.id))
            
            now = datetime.utcnow()
            cutoff = now - timedelta(days=window_days)
            
            metrics_list = []
            for ex in examples:
                meta = ex.metadata or {}
                
                # Filter by age
                created_str = meta.get("created_at") or meta.get("collected_at")
                try:
                    created_ts = datetime.fromisoformat(str(created_str)) if created_str else None
                except Exception:
                    created_ts = None
                    
                if created_ts and created_ts < cutoff:
                    continue
                
                # Extract metrics
                reply_met = meta.get("reply_metrics") or {}
                input_met = meta.get("input_metrics") or {}
                
                reply_impr = int(reply_met.get("impressions", 0) or 0)
                input_impr = int(input_met.get("impressions", 0) or 0)
                
                if input_impr > 0:  # Only count examples with trackable input
                    lift = reply_impr / input_impr if input_impr > 0 else 0
                    metrics_list.append({
                        "key": meta.get("key", ""),
                        "reply_url": meta.get("reply_url", ""),
                        "reply_impressions": reply_impr,
                        "input_impressions": input_impr,
                        "lift_ratio": lift,
                        "reply_text": (ex.outputs or {}).get("generated_reply", "")[:100],
                    })
            
            if not metrics_list:
                return {
                    "total_replies": 0,
                    "message": "No examples with tracked impressions in window"
                }
            
            # Calculate aggregate metrics
            total = len(metrics_list)
            avg_reply = sum(m["reply_impressions"] for m in metrics_list) / total
            avg_input = sum(m["input_impressions"] for m in metrics_list) / total
            avg_lift = sum(m["lift_ratio"] for m in metrics_list) / total
            
            # Top performers
            top_performers = sorted(
                metrics_list,
                key=lambda m: m["reply_impressions"],
                reverse=True
            )[:10]
            
            report = {
                "experiment_dataset": experiment_dataset,
                "window_days": window_days,
                "total_replies": total,
                "avg_reply_impressions": round(avg_reply, 2),
                "avg_input_impressions": round(avg_input, 2),
                "impression_lift_ratio": round(avg_lift, 3),
                "top_performers": top_performers,
            }
            
            logging.info("Experiment evaluation complete: %s", {k: v for k, v in report.items() if k != "top_performers"})
            return report
            
        except Exception as exc:
            logging.error("Failed to evaluate experiment: %s", exc)
            return {"error": str(exc)}
    
    def run_impressions_experiment(
        self,
        source_dataset: str,
        experiment_name: str,
        model_variants: List[str],
        max_examples: int = 50,
        min_input_impressions: int = 1000,
        min_followers: int = 1000,
    ) -> Dict[str, Any]:
        """End-to-end experiment runner for testing different models/prompts on impressions.
        
        This is a placeholder for the full workflow:
        1. Create experiment dataset
        2. For each model variant, generate replies
        3. Post replies (if enabled)
        4. Wait for metrics to accumulate
        5. Evaluate and compare
        
        Args:
            source_dataset: Source dataset name
            experiment_name: Unique experiment identifier
            model_variants: List of model names or prompt templates to test
            max_examples: Number of tweets to reply to
            min_input_impressions: Minimum impressions threshold for source tweets
            min_followers: Minimum follower count threshold for authors
            
        Returns:
            Summary with dataset name and next steps
        """
        # Step 1: Create experiment dataset
        exp_ds = self.create_experiment_dataset(
            source_dataset=source_dataset,
            experiment_name=experiment_name,
            max_examples=max_examples,
            min_impressions=min_input_impressions,
            min_followers=min_followers,
            filter_completed=True,
        )
        
        if not exp_ds:
            return {"error": "Failed to create experiment dataset"}
        
        # Step 2-4: Generate and post replies would go here
        # This requires integration with your bot's reply generation and posting logic
        # For now, return setup info
        
        return {
            "status": "experiment_dataset_created",
            "experiment_dataset": exp_ds,
            "experiment_name": experiment_name,
            "model_variants": model_variants,
            "max_examples": max_examples,
            "min_followers": min_followers,
            "next_steps": [
                f"Dataset prioritizes authors with {min_followers}+ followers",
                "Generate replies using each model variant",
                "Post replies and capture reply_ids",
                "Wait 1-3 days for metrics to accumulate",
                f"Run evaluate_impressions_experiment('{exp_ds}')",
            ],
        }

    def update_all_datasets_periodically(self, interval_hours: int = 24) -> None:
        """Periodically update all datasets and databases."""
        while True:
            logging.info("Starting periodic dataset update...")
            try:
                self.upload_dataset_from_db()
                logging.info("All datasets updated successfully.")
            except Exception as exc:
                logging.error("Failed to update datasets: %s", exc)

            logging.info("Next update in %d hours.", interval_hours)
            time.sleep(interval_hours * 3600)


def setup_langsmith_env() -> None:
    """Print instructions for configuring LangSmith environment variables."""
    print("To set up LangSmith integration, you need to:")
    print("1. Sign up at https://smith.langchain.com/")
    print("2. Get your API key from the settings")
    print("3. Set the environment variable:")
    print("   $env:LANGSMITH_API_KEY='your-api-key-here'  (PowerShell)")
    print("   set LANGSMITH_API_KEY=your-api-key-here     (CMD)")
    print("4. Restart your application")


langsmith_integration = LangSmithIntegration()


__all__ = [
    "LangSmithIntegration",
    "langsmith_integration",
    "setup_langsmith_env",
]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, help="Path to SQLite database")
    parser.add_argument("--dataset", type=str, help="LangSmith dataset name")
    parser.add_argument("--match-key", type=str, default="post_id", help="Key to match examples (default: post_id)")
    parser.add_argument("--upload", action="store_true", help="Upload dataset from DB (original method)")
    parser.add_argument("--update", action="store_true", help="Update metrics in LangSmith dataset from DB")
    parser.add_argument("--max-examples", type=int, default=2, help="Maximum number of examples to update")
    args = parser.parse_args()

    if os.getenv("LANGSMITH_API_KEY"):
        print("✓ LangSmith API key found")
        print(f"Project URL: {langsmith_integration.get_langsmith_url()}")

        if args.upload:
            langsmith_integration.upload_dataset_from_db(args.db if args.db else "data/trending.sqlite")
        if args.update:
            if not args.db or not args.dataset:
                print("--db and --dataset are required for --update mode")
            else:
                langsmith_integration.update_dataset_from_db(
                    args.db, args.dataset, args.match_key, max_examples=args.max_examples
                )
        else:
            report = langsmith_integration.generate_performance_report()
            print(json.dumps(report, indent=2))
    else:
        setup_langsmith_env()
