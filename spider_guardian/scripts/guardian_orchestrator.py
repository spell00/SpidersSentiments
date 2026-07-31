"""Unified Spider Guardian orchestrator."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import random
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import List, Optional, TYPE_CHECKING

import numpy as np
from zoneinfo import ZoneInfo

from .auto_poster import DailyScheduler, GenerationContext, log_highlights
from .generate_best_post import INTENT_DIRECTIVES
from ..config import ChatProviderConfig, SpiderGuardianConfig
from ..storage.sql import SQLDataStore, ScrapedArticle

if TYPE_CHECKING:
    from ..bot import SpiderGuardianBot


def _build_bot(config: SpiderGuardianConfig) -> "SpiderGuardianBot":
    from ..bot import SpiderGuardianBot

    return SpiderGuardianBot(config)


def _log_engagement_metrics(**kwargs) -> None:
    try:
        from ..langsmith import log_engagement_metrics
    except Exception:
        return
    log_engagement_metrics(**kwargs)


@dataclass
class AutoPostOptions:
    topic: str = "funny spider facts"
    intent: str = "myth-busting"
    max_words: int = 60
    num_candidates: int = 4
    sample_limit: int = 50
    prompt_examples: int = 6
    min_engagement: float = 0.0
    daily_rate: float = 2.5
    start_hour: int = 8
    end_hour: int = 22
    min_gap_minutes: int = 180


class GuardianOrchestrator:
    def __init__(
        self,
        *,
        config: SpiderGuardianConfig,
        timezone: str,
        autopost: AutoPostOptions,
        cycle_mean_minutes: float,
        cycle_std_minutes: float,
        cycle_min_minutes: float,
        cycle_max_minutes: float,
        reply_limit_min: int,
        reply_limit_max: int,
        followup_conversations: int,
        followup_replies: int,
        trending_hours: int,
        trending_retention_days: int,
        seed: Optional[int] = None,
        dry_run: bool = False,
    ) -> None:
        self.config = config
        self.autopost = autopost
        self.tz = ZoneInfo(timezone)
        self.reply_limit_min = max(0, reply_limit_min)
        self.reply_limit_max = max(self.reply_limit_min, reply_limit_max)
        self.followup_conversations = max(1, followup_conversations)
        self.followup_replies = max(0, followup_replies)
        self.trending_hours = max(1, trending_hours)
        self.trending_retention_days = max(1, trending_retention_days)
        self.dry_run = dry_run

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.bot = None
        self.twitter_ready = False
        if self.dry_run:
            self.generation = GenerationContext(
                config=self.config,
                provider=None,
                store=SQLDataStore(self.config.sql_database_path),
            )
        else:
            self.bot = _build_bot(self.config)
            self.bot.build_vector_index()
            try:
                self.bot.ensure_twitter_client()
                self.twitter_ready = True
            except RuntimeError as exc:
                logging.warning(
                    "Twitter credentials missing (%s); falling back to dry-run mode",
                    exc,
                )
                self.dry_run = True
            provider = next((p for p in self.bot.providers if p.is_available()), None)
            self.generation = GenerationContext(config=self.config, provider=provider, store=self.bot.sql_store)

        self.scheduler = DailyScheduler(
            tz=self.tz,
            rate=max(0.0, autopost.daily_rate),
            start_hour=autopost.start_hour,
            end_hour=autopost.end_hour,
            min_gap_minutes=autopost.min_gap_minutes,
            seed=seed,
        )
        self.posts_today = 0
        self.schedule_day: Optional[date] = None

        self.cycle_mean = max(1.0, cycle_mean_minutes) * 60.0
        self.cycle_std = max(1.0, cycle_std_minutes) * 60.0
        self.cycle_min = max(60.0, cycle_min_minutes * 60.0)
        self.cycle_max = max(self.cycle_min + 60.0, cycle_max_minutes * 60.0)
        self.next_cycle = self._plan_next_cycle()

        self._initialised = False

    # ------------------------------------------------------------------
    # scheduling helpers

    def _now(self) -> datetime:
        return datetime.now(self.tz)

    def _plan_next_cycle(self) -> datetime:
        base = self._now()
        for _ in range(8):
            sample = random.gauss(self.cycle_mean, self.cycle_std)
            if sample >= self.cycle_min:
                return base + timedelta(seconds=min(sample, self.cycle_max))
        sample = max(self.cycle_min, min(self.cycle_max, sample))  # type: ignore[arg-type]
        return base + timedelta(seconds=sample)

    def _refresh_schedule(self, now: datetime) -> None:
        if self.schedule_day != now.date():
            self.schedule_day = now.date()
            self.posts_today = 0
        self.scheduler.refresh(now, posts_already_done=self.posts_today)

    # ------------------------------------------------------------------
    # public entrypoint

    def run(self, skip_initial_post: bool) -> None:
        logging.info("Guardian orchestrator started")
        if not skip_initial_post:
            self._run_initial_post()
        self._initialised = True

        while True:
            now = self._now()
            self._refresh_schedule(now)
            next_post = self.scheduler.peek_next(now)
            targets: List[tuple[str, datetime]] = [("cycle", self.next_cycle)]
            if next_post is not None:
                targets.append(("autopost", next_post))

            event_type, event_time = min(targets, key=lambda item: item[1])
            wait_seconds = max(5.0, (event_time - now).total_seconds())
            if wait_seconds > 30.0:
                sleep_chunk = min(wait_seconds, 300.0)
                time.sleep(sleep_chunk)
                continue

            if event_type == "autopost" and next_post is not None and now >= next_post:
                self._run_autopost(now)
                continue

            if now >= self.next_cycle:
                self._run_cycle()
                self.next_cycle = self._plan_next_cycle()
                continue

            time.sleep(10.0)

    # ------------------------------------------------------------------
    # cycles

    def _run_initial_post(self) -> None:
        logging.info("Planning initial post")
        post, highlights = self.generation.generate_post(
            topic=self.autopost.topic,
            intent=self.autopost.intent,
            max_words=self.autopost.max_words,
            num_candidates=self.autopost.num_candidates,
            sample_limit=self.autopost.sample_limit,
            prompt_examples=self.autopost.prompt_examples,
            min_engagement=self.autopost.min_engagement,
        )
        log_highlights(highlights)
        if not post:
            logging.warning("Initial generation returned empty text; skipping launch post")
            return
        if self._post_to_x(post):
            self.posts_today += 1
            self.scheduler.refresh(self._now(), posts_already_done=self.posts_today)

    def _run_autopost(self, now: datetime) -> None:
        slot = self.scheduler.pop_next(now) or now
        logging.info("[autopost] Slot triggered at %s", slot.astimezone(self.tz))

        post, highlights = self.generation.generate_post(
            topic=self.autopost.topic,
            intent=self.autopost.intent,
            max_words=self.autopost.max_words,
            num_candidates=self.autopost.num_candidates,
            sample_limit=self.autopost.sample_limit,
            prompt_examples=self.autopost.prompt_examples,
            min_engagement=self.autopost.min_engagement,
        )
        log_highlights(highlights)
        if not post:
            logging.warning("[autopost] Generator returned empty text; slot skipped")
            return
        if self._post_to_x(post):
            self.posts_today += 1
            logging.info("[autopost] Posts sent today: %d", self.posts_today)
        self.scheduler.refresh(self._now(), posts_already_done=self.posts_today)

    def _run_cycle(self) -> None:
        logging.info("[cycle] Starting hourly maintenance run")
        if self.dry_run or not self.twitter_ready:
            logging.info("[cycle] Dry-run mode enabled; skipping replies and posts")
        else:
            limit = random.randint(self.reply_limit_min, self.reply_limit_max)
            if limit > 0:
                try:
                    self.bot.respond_to_tweets(limit=limit, reply_to_replies=True)
                except Exception as exc:
                    logging.warning("[cycle] respond_to_tweets failed: %s", exc)
        try:
            self.bot.collect_and_learn()
        except Exception as exc:
            logging.warning("[cycle] collect_and_learn failed: %s", exc)
        try:
            mode = random.choice(["live", "top", "latest"])
            inserted = self.bot.collect_trending(
                hours=self.trending_hours,
                retention_days=self.trending_retention_days,
                mode=mode,
            )
            logging.info("[cycle] Trending store upserted %d posts (mode=%s)", inserted, mode)
        except Exception as exc:
            logging.warning("[cycle] collect_trending failed: %s", exc)

        if not self.dry_run and self.twitter_ready:
            try:
                followups = self._handle_followups()
                if followups:
                    logging.info("[cycle] Follow-up replies sent: %d", followups)
            except Exception as exc:
                logging.warning("[cycle] follow-up processing failed: %s", exc)
        logging.info("[cycle] Maintenance run complete")

    # ------------------------------------------------------------------
    # follow-up handling

    def _handle_followups(
        self,
        *,
        max_replies: Optional[int] = None,
        max_conversations: Optional[int] = None,
    ) -> int:
        client = self.bot.twitter_client
        if client is None:
            return 0

        responded = 0
        conversations_checked = 0
        now_utc = datetime.utcnow().isoformat()

        reply_cap = self.followup_replies if max_replies is None else max_replies
        convo_cap = self.followup_conversations if max_conversations is None else max_conversations

        target_replies = math.inf if reply_cap is None or reply_cap < 0 else max(0, int(reply_cap))
        target_conversations = math.inf if convo_cap is None or convo_cap < 0 else max(0, int(convo_cap))

        for article in self.bot.sql_store.iter_scraped_articles():
            if conversations_checked >= target_conversations or responded >= target_replies:
                break
            metadata = dict(article.metadata or {})
            if metadata.get("type") != "interaction":
                continue
            try:
                record = json.loads(article.content or "{}")
            except json.JSONDecodeError:
                continue
            conversation_id = record.get("conversation_id")
            if not conversation_id:
                continue

            since_id = metadata.get("last_seen_reply_id") or record.get("reply_id")
            try:
                replies = client.fetch_replies(conversation_id, since_id=since_id)
            except Exception as exc:
                logging.warning("[follow-up] Fetch failed for %s: %s", conversation_id, exc)
                continue
            if not replies:
                continue

            conversations_checked += 1
            replies.sort(key=lambda post: int(post.id))
            highest_seen = self._safe_int(since_id)

            for reply in replies:
                if responded >= target_replies:
                    break
                author = (reply.author_handle or "").lower()
                if client.username and author == (client.username or "").lower():
                    highest_seen = max(highest_seen, self._safe_int(reply.id))
                    continue
                if not self._should_follow_up(reply.text):
                    highest_seen = max(highest_seen, self._safe_int(reply.id))
                    continue

                follow_text = self._compose_follow_up(reply, record)
                if not follow_text:
                    highest_seen = max(highest_seen, self._safe_int(reply.id))
                    continue

                try:
                    follow_id = client.reply(follow_text, reply_to_tweet_id=reply.id)
                except Exception as exc:
                    logging.warning("[follow-up] Failed to reply to %s: %s", reply.id, exc)
                    highest_seen = max(highest_seen, self._safe_int(reply.id))
                    continue

                responded += 1
                highest_seen = max(highest_seen, self._safe_int(reply.id), self._safe_int(follow_id))
                entries = list(metadata.get("followups", []))
                entries.append(
                    {
                        "target_reply_id": reply.id,
                        "sent_reply_id": follow_id,
                        "text": follow_text,
                        "timestamp": now_utc,
                    }
                )
                metadata["followups"] = entries[-10:]
                logging.info(
                    "[follow-up] Responded to %s from @%s", reply.id, reply.author_handle or "unknown"
                )

                try:
                    _log_engagement_metrics(
                        reply_text=follow_text,
                        likes=int(getattr(reply, "like_count", 0) or 0),
                        replies=int(getattr(reply, "reply_count", 0) or 0),
                        impressions=int(getattr(reply, "impression_count", 0) or 0),
                        tweet_id=str(reply.id),
                        metadata={
                            "stage": "follow_up",
                            "source_conversation": conversation_id,
                        },
                    )
                except Exception as exc:
                    logging.debug("LangSmith follow-up engagement log failed: %s", exc)

            if highest_seen is not None:
                metadata["last_seen_reply_id"] = str(highest_seen)
            metadata["last_followup_check"] = now_utc
            self.bot.sql_store.upsert_scraped_articles(
                [
                    ScrapedArticle(
                        title=article.title,
                        link=article.link,
                        content=article.content,
                        metadata=metadata,
                        created_at=article.created_at,
                    )
                ]
            )

        return responded

    def _should_follow_up(self, text: Optional[str]) -> bool:
        if not text:
            return False
        snippet = text.lower()
        if any(bad in snippet for bad in ("http", "www.", "buy now", "free crypto")):
            return False
        return len(snippet.strip()) >= 5

    def _compose_follow_up(self, reply_post, record: dict) -> Optional[str]:
        context = list(self.bot.retrieve_context(reply_post.text))
        original_reply = record.get("reply_text")
        if original_reply:
            context.append(("human", original_reply))
        prompt = self.bot.build_prompt(
            tweet_text=reply_post.text,
            context_documents=context,
            tone=self.bot.classify_tone(reply_post.text),
            post=reply_post,
        )
        return self.bot.generate_reply(prompt, original_tweet=reply_post.text)

    @staticmethod
    def _safe_int(value: Optional[str]) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # posting helper

    def _post_to_x(self, text: str) -> bool:
        if self.dry_run or not self.twitter_ready:
            logging.info("[dry-run] Would post: %s", text)
            return False
        client = self.bot.twitter_client
        if client is None:
            logging.warning("[autopost] Twitter client unavailable")
            return False
        try:
            client.post_tweet(text)
            logging.info("[autopost] Posted new status (%d chars)", len(text))
            return True
        except Exception as exc:
            logging.warning("[autopost] Failed to post tweet: %s", exc)
            return False


# ----------------------------------------------------------------------
# CLI utilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all Spider Guardian loops together")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    parser.add_argument("--timezone", default="America/New_York", help="IANA timezone for scheduling")
    parser.add_argument("--poisson-rate", type=float, default=2.5, help="Average original posts per day")
    parser.add_argument("--poisson-start-hour", type=int, default=8, help="Posting window start hour")
    parser.add_argument("--poisson-end-hour", type=int, default=22, help="Posting window end hour")
    parser.add_argument("--poisson-min-gap", type=int, default=180, help="Minimum minutes between original posts")
    parser.add_argument("--cycle-mean-minutes", type=float, default=60.0, help="Average minutes between maintenance cycles")
    parser.add_argument("--cycle-std-minutes", type=float, default=18.0, help="Stddev for maintenance cycle spacing")
    parser.add_argument("--cycle-min-minutes", type=float, default=30.0, help="Shortest maintenance interval")
    parser.add_argument("--cycle-max-minutes", type=float, default=120.0, help="Longest maintenance interval")
    parser.add_argument("--reply-limit-min", type=int, default=1, help="Minimum replies per cycle")
    parser.add_argument("--reply-limit-max", type=int, default=3, help="Maximum replies per cycle")
    parser.add_argument("--followup-conversations", type=int, default=3, help="Conversations to scan per cycle")
    parser.add_argument("--followup-replies", type=int, default=4, help="Maximum follow-up replies per cycle")
    parser.add_argument("--trending-hours", type=int, default=24, help="Window for trending sampling")
    parser.add_argument("--trending-retention-days", type=int, default=3, help="Retention horizon for trending store")
    parser.add_argument("--topic", default="funny spider facts", help="Theme for original posts")
    parser.add_argument("--intent", choices=sorted(INTENT_DIRECTIVES.keys()), default="myth-busting", help="Communication intent for original posts")
    parser.add_argument("--max-words", type=int, default=60, help="Max words for generated original posts")
    parser.add_argument("--num-candidates", type=int, default=4, help="Candidate generations per slot")
    parser.add_argument("--sample-limit", type=int, default=50, help="Historic post sample size for prompts")
    parser.add_argument("--prompt-examples", type=int, default=6, help="Historic highlight snippets per prompt")
    parser.add_argument("--min-engagement", type=float, default=0.0, help="Minimum engagement score for highlights")
    parser.add_argument("--chat-model", type=str, help="Override chat model identifier")
    parser.add_argument("--provider", type=str, default="local", help="Provider backend name")
    parser.add_argument("--temperature", type=float, default=0.6, help="Generation temperature override")
    parser.add_argument("--show-browser", action="store_true", help="Run Selenium with visible browser window")
    parser.add_argument("--selenium-driver", choices=("chrome", "firefox"), help="Explicit Selenium driver choice")
    parser.add_argument("--skip-initial-post", action="store_true", help="Do not send the immediate kickoff post")
    parser.add_argument("--dry-run", action="store_true", help="Log activity without posting or replying")
    parser.add_argument("--seed", type=int, help="Seed RNGs for reproducible scheduling")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> SpiderGuardianConfig:
    config = SpiderGuardianConfig()
    if args.show_browser:
        config.selenium_headless = False
    if args.selenium_driver:
        config.selenium_driver = args.selenium_driver
    if args.chat_model:
        config.providers = (
            ChatProviderConfig(
                name=args.provider or "local",
                model=args.chat_model,
                temperature=args.temperature,
            ),
        )
    return config


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    config = build_config(args)
    autopost = AutoPostOptions(
        topic=args.topic,
        intent=args.intent,
        max_words=max(20, args.max_words),
        num_candidates=max(1, args.num_candidates),
        sample_limit=max(10, args.sample_limit),
        prompt_examples=max(1, args.prompt_examples),
        min_engagement=max(0.0, args.min_engagement),
        daily_rate=max(0.0, args.poisson_rate),
        start_hour=args.poisson_start_hour,
        end_hour=args.poisson_end_hour,
        min_gap_minutes=max(10, args.poisson_min_gap),
    )

    orchestrator = GuardianOrchestrator(
        config=config,
        timezone=args.timezone,
        autopost=autopost,
        cycle_mean_minutes=args.cycle_mean_minutes,
        cycle_std_minutes=args.cycle_std_minutes,
        cycle_min_minutes=args.cycle_min_minutes,
        cycle_max_minutes=args.cycle_max_minutes,
        reply_limit_min=args.reply_limit_min,
        reply_limit_max=args.reply_limit_max,
        followup_conversations=args.followup_conversations,
        followup_replies=args.followup_replies,
        trending_hours=args.trending_hours,
        trending_retention_days=args.trending_retention_days,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    try:
        orchestrator.run(skip_initial_post=args.skip_initial_post)
    except KeyboardInterrupt:
        logging.info("Guardian orchestrator stopped via Ctrl+C")


__all__ = [
    "AutoPostOptions",
    "GuardianOrchestrator",
    "build_config",
    "main",
]


if __name__ == "__main__":
    main()
