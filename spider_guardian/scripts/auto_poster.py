"""Automate funny spider posts throughout the day using a Poisson schedule."""

from __future__ import annotations

import argparse
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, time as dtime
from typing import Any, List, Optional, Sequence

import numpy as np
from zoneinfo import ZoneInfo

from .generate_best_post import INTENT_DIRECTIVES, generate_candidates
from ..config import ChatProviderConfig, SpiderGuardianConfig
from ..storage import SQLDataStore
from ..twitter_client import SeleniumTwitterClient


@dataclass
class GenerationContext:
    config: SpiderGuardianConfig
    provider: Optional[Any]
    store: SQLDataStore

    def generate_post(
        self,
        *,
        topic: str,
        intent: str,
        max_words: int,
        num_candidates: int,
        sample_limit: int,
        prompt_examples: int,
        min_engagement: float,
    ) -> tuple[str, Sequence[str]]:
        """Return a funny post plus the supporting highlight snippets."""

        generated, highlights, _raw_highlights, active_provider = generate_candidates(
            self.config,
            topic=topic,
            intent=intent,
            provider=self.provider,
            store=self.store,
            max_words=max_words,
            num_candidates=num_candidates,
            sample_limit=sample_limit,
            prompt_examples=prompt_examples,
            min_engagement=min_engagement,
        )
        self.provider = active_provider
        if not generated:
            return "", highlights
        return random.choice(generated), highlights


@dataclass
class DaySchedule:
    day: date
    slots: List[datetime] = field(default_factory=list)


class DailyScheduler:
    def __init__(
        self,
        *,
        tz: ZoneInfo,
        rate: float,
        start_hour: int,
        end_hour: int,
        min_gap_minutes: int,
        seed: Optional[int],
    ) -> None:
        self.tz = tz
        self.rate = max(0.0, rate)
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.min_gap = timedelta(minutes=max(1, min_gap_minutes))
        self.rng = np.random.default_rng(seed)
        self.random = random.Random(seed)
        self.plan: Optional[DaySchedule] = None

    def refresh(self, now: datetime, *, posts_already_done: int = 0) -> None:
        """Ensure a schedule exists for the current day."""

        current_date = now.date()
        if self.plan is None or self.plan.day != current_date:
            total = int(self.rng.poisson(self.rate)) if self.rate > 0 else 0
            remaining = max(0, total - posts_already_done)
            self.plan = DaySchedule(day=current_date, slots=self._sample_times(now, remaining))

    def peek_next(self, now: datetime) -> Optional[datetime]:
        if self.plan is None:
            return None
        for slot in self.plan.slots:
            if slot > now:
                return slot
        return None

    def pop_next(self, now: datetime) -> Optional[datetime]:
        if self.plan is None or not self.plan.slots:
            return None
        # Drop any stale slots that are far in the past (should be rare)
        tolerance = timedelta(minutes=10)
        while self.plan.slots and self.plan.slots[0] < now - tolerance:
            expired = self.plan.slots.pop(0)
            logging.warning("[schedule] Dropping expired slot at %s", expired.astimezone(self.tz))
        if self.plan.slots and self.plan.slots[0] <= now + tolerance:
            return self.plan.slots.pop(0)
        return None

    def seconds_until_next_window(self, now: datetime) -> float:
        start_time = dtime(hour=self.start_hour)
        today_start = datetime.combine(now.date(), start_time, tzinfo=self.tz)
        if now < today_start:
            return max((today_start - now).total_seconds(), 60.0)
        next_day = now.date() + timedelta(days=1)
        next_start = datetime.combine(next_day, start_time, tzinfo=self.tz)
        return max((next_start - now).total_seconds(), 60.0)

    def _sample_times(self, now: datetime, count: int) -> List[datetime]:
        if count <= 0:
            return []

        start_time = datetime.combine(now.date(), dtime(hour=self.start_hour), tzinfo=self.tz)
        end_time = datetime.combine(now.date(), dtime(hour=self.end_hour), tzinfo=self.tz)
        if end_time <= start_time:
            logging.warning("[schedule] Invalid window; end hour must be after start hour")
            return []

        earliest = max(start_time, now + self.min_gap)
        latest = end_time
        if latest <= earliest:
            return []

        window_seconds = (latest - earliest).total_seconds()
        picks: List[datetime] = []
        attempts = 0
        max_attempts = max(20, count * 12)

        while len(picks) < count and attempts < max_attempts:
            attempts += 1
            offset = self.random.uniform(0, window_seconds)
            candidate = earliest + timedelta(seconds=offset)
            if candidate > latest:
                continue
            if all(abs((candidate - other).total_seconds()) >= self.min_gap.total_seconds() for other in picks):
                picks.append(candidate)

        picks.sort()
        return picks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuously post funny spider content")
    parser.add_argument("--topic", default="funny spider facts", help="Theme for generated posts")
    parser.add_argument(
        "--intent",
        choices=sorted(INTENT_DIRECTIVES.keys()),
        default="myth-busting",
        help="High-level communication goal",
    )
    parser.add_argument("--model", default=None, help="Optional Hugging Face model identifier")
    parser.add_argument("--provider", default="local", help="Provider name to initialise")
    parser.add_argument("--temperature", type=float, default=0.6, help="Generation temperature")
    parser.add_argument("--num-candidates", type=int, default=4, help="Number of variants to sample")
    parser.add_argument("--max-words", type=int, default=60, help="Maximum words per post")
    parser.add_argument("--sample-limit", type=int, default=50, help="Historic posts to inspect")
    parser.add_argument("--prompt-examples", type=int, default=6, help="Historic snippets to feed the model")
    parser.add_argument("--min-engagement", type=float, default=0.0, help="Minimum score for highlight inclusion")
    parser.add_argument("--daily-rate", type=float, default=2.0, help="Lambda for Poisson daily schedule")
    parser.add_argument("--start-hour", type=int, default=8, help="Day window start hour (EST)")
    parser.add_argument("--end-hour", type=int, default=22, help="Day window end hour (EST)")
    parser.add_argument("--min-gap-minutes", type=int, default=180, help="Minimum minutes between posts")
    parser.add_argument("--timezone", default="America/New_York", help="IANA timezone for scheduling")
    parser.add_argument("--seed", type=int, help="Deterministic seed for RNG")
    parser.add_argument("--dry-run", action="store_true", help="Skip Selenium and only log output")
    parser.add_argument("--show-browser", action="store_true", help="Disable Selenium headless mode")
    parser.add_argument(
        "--selenium-driver",
        choices=("chrome", "firefox"),
        default=None,
        help="Override Selenium driver choice",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, ...)")
    return parser.parse_args()


def sleep_with_heartbeat(seconds: float) -> None:
    remaining = max(0.0, seconds)
    while remaining > 0:
        chunk = min(remaining, 300.0)
        time.sleep(chunk)
        remaining -= chunk


def configure_providers(config: SpiderGuardianConfig, args: argparse.Namespace) -> None:
    existing = tuple(config.providers)
    if args.model:
        config.providers = (
            ChatProviderConfig(name=args.provider, model=args.model, temperature=args.temperature),
        )
        return

    if not existing:
        raise RuntimeError("No default providers configured in SpiderGuardianConfig")

    provider_configs = tuple(
        ChatProviderConfig(name=prov.name, model=prov.model, temperature=args.temperature)
        for prov in existing
    )
    if args.provider and provider_configs:
        first = provider_configs[0]
        if args.provider != first.name:
            provider_configs = (
                ChatProviderConfig(name=args.provider, model=first.model, temperature=args.temperature),
            )
    config.providers = provider_configs


def log_highlights(highlights: Sequence[str]) -> None:
    if not highlights:
        logging.info("[inspiration] No high-engagement snippets available today")
        return
    logging.info("[inspiration] Today's comedic cues:")
    for snippet in highlights[:3]:
        logging.info("    - %s", snippet)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    if args.start_hour >= args.end_hour:
        raise SystemExit("start-hour must be earlier than end-hour")

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    tz = ZoneInfo(args.timezone)

    config = SpiderGuardianConfig()
    config.selenium_headless = not args.show_browser
    if args.selenium_driver:
        config.selenium_driver = args.selenium_driver

    configure_providers(config, args)
    store = SQLDataStore(config.sql_database_path)

    from ..providers import build_chat_providers

    providers = build_chat_providers(config.providers)
    provider = next((p for p in providers if p.is_available()), None)
    if provider is None:
        logging.warning("No generation provider resolved; fallback copy will be used")

    generation = GenerationContext(config=config, provider=provider, store=store)
    scheduler = DailyScheduler(
        tz=tz,
        rate=args.daily_rate,
        start_hour=args.start_hour,
        end_hour=args.end_hour,
        min_gap_minutes=args.min_gap_minutes,
        seed=args.seed,
    )

    twitter_client: Optional[SeleniumTwitterClient] = None
    if not args.dry_run:
        twitter_client = SeleniumTwitterClient(config)

    def post_to_x(text: str) -> Optional[str]:
        if args.dry_run:
            logging.info("[dry-run] %s", text)
            return None
        if twitter_client is None:
            raise RuntimeError("Twitter client not initialised")
        return twitter_client.post_tweet(text)

    try:
        now = datetime.now(tz)
        first_post, highlights = generation.generate_post(
            topic=args.topic,
            intent=args.intent,
            max_words=args.max_words,
            num_candidates=args.num_candidates,
            sample_limit=args.sample_limit,
            prompt_examples=args.prompt_examples,
            min_engagement=args.min_engagement,
        )
        log_highlights(highlights)

        if first_post:
            logging.info("[post] Launching with immediate post")
            try:
                post_to_x(first_post)
            except Exception as exc:
                logging.exception("[post] Initial post failed: %s", exc)
        else:
            logging.warning("[post] No content generated; skipping initial post")

        scheduler.refresh(now, posts_already_done=1 if first_post else 0)

        while True:
            now = datetime.now(tz)
            scheduler.refresh(now)
            next_slot = scheduler.peek_next(now)
            if next_slot is None:
                sleep_seconds = scheduler.seconds_until_next_window(now)
                wake_time = now + timedelta(seconds=sleep_seconds)
                logging.info(
                    "[schedule] No more posts today; next planning pass at %s", wake_time.astimezone(tz)
                )
                sleep_with_heartbeat(sleep_seconds)
                continue

            wait_seconds = max(0.0, (next_slot - now).total_seconds())
            logging.info(
                "[schedule] Next post at %s (%.1f minutes)", next_slot.astimezone(tz), wait_seconds / 60.0
            )
            sleep_with_heartbeat(wait_seconds)

            now = datetime.now(tz)
            slot = scheduler.pop_next(now)
            if slot is None:
                logging.debug("[schedule] Slot already consumed; recalculating")
                continue

            post_text, highlights = generation.generate_post(
                topic=args.topic,
                intent=args.intent,
                max_words=args.max_words,
                num_candidates=args.num_candidates,
                sample_limit=args.sample_limit,
                prompt_examples=args.prompt_examples,
                min_engagement=args.min_engagement,
            )
            if not post_text:
                logging.warning("[post] Skip scheduled slot: generator returned empty text")
                scheduler.refresh(now, posts_already_done=0)
                continue

            log_highlights(highlights)
            try:
                post_to_x(post_text)
            except Exception as exc:
                logging.exception("[post] Scheduled post failed: %s", exc)

    except KeyboardInterrupt:
        logging.info("Interrupted; shutting down autoposter")
    finally:
        if twitter_client is not None:
            twitter_client.close()


__all__ = [
    "GenerationContext",
    "DaySchedule",
    "DailyScheduler",
    "log_highlights",
    "main",
]


if __name__ == "__main__":
    main()
