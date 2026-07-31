# Runtime schematic: long-running processes and loops

This document gives a one-glance overview of everything that “runs forever,” the order of operations, and where to look when something stalls.

## Quick map

- Guardian Orchestrator (Python, runs forever)
  - File: `spider_guardian/scripts/guardian_orchestrator.py`
  - Purpose: Hourly maintenance cycles (replies, trending, follow-ups) + scheduled original posts

- Dataset update loop (Python, loops forever when `--loop` or via PowerShell)
  - File: `update_datasets.py` (with `--loop`) or launched by PowerShell scripts
  - Purpose: Upload/update datasets (streamed posts, interactions, flagged replies) to LangSmith

- Reply metrics refresh loop (PowerShell loop)
  - Launcher: `scripts/run_all_background*.ps1` → Worker: `spider_guardian/scripts/refresh_my_replies.py`
  - Purpose: Periodically re-scrape engagement metrics for our replies and sync to LangSmith

- ML training loop (PowerShell loop)
  - Launcher: `scripts/run_all_background*.ps1` → Worker: `spider_guardian/ml_bot.py train`
  - Purpose: Daily model training based on recent replies and trending posts

- Background supervisor (PowerShell, runs forever)
  - Scripts: `scripts/run_all_background.ps1` or `scripts/run_all_background_monitored.ps1`
  - Purpose: Start and keep all loops running, redirect logs, provide basic monitoring

---

## Guardian Orchestrator (runs forever)
File: `spider_guardian/scripts/guardian_orchestrator.py`

- Boot
  - Build config (providers, model, Selenium driver)
  - Create `SpiderGuardianBot` and vector index
  - Ensure Twitter client (unless `--dry-run`)
  - Build day scheduler for original posts (Poisson-like between `start_hour`/`end_hour`)

- Optional initial post (once)
  - Generate several candidates → pick one → post (or skip in dry-run)

- Main loop (forever)
  - Compute next deadlines:
    - Next maintenance cycle (randomized within bounds)
    - Next scheduled autopost slot
  - Sleep in chunks until a deadline is due
  - If autopost slot due:
    - Generate post → log highlights → post → update count → refresh schedule
  - If cycle due: run maintenance cycle
    - Replies: `bot.respond_to_tweets(limit=random[min,max], reply_to_replies=True)`
    - Learning: `bot.collect_and_learn()`
    - Trending: `bot.collect_trending(hours, retention_days, mode)`
    - Follow-ups: scan recent conversations and post follow-up replies
  - Plan next cycle and repeat

---

## Reply engine (called inside cycles)
File: `spider_guardian/bot.py` → `SpiderGuardianBot.respond_to_tweets()`

1) Search and author enrichment
- Search tweets by `config.twitter_query`
- Ensure `author_handle` (resolve from tweet page if missing)
- Look up cached followers via SQL: `get_author_followers_info(handle)` → returns:
  - `follower_count` (int)
  - `followers_checked_at` (ISO timestamp in metadata)
- Refresh policy:
  - If `follower_count` is -1/None OR last check ≥ 30 days → mark for fetch
  - Else use cached value
- Batch-fetch uncached/stale authors (deduped) with progress bar and a 10-minute overall timeout
  - After fetch: `upsert_author(handle, follower_count)` stamps `followers_checked_at`
- Sort tweets by author followers (desc)
- Increment `tweet_count` for seen authors

2) Candidate filtering
- Skip “spider man/spiderman/spidey” noise and brand handles
- Skip replies if `reply_to_replies` is False
- Require “spider” in body
- Apply follower filters (`--min-followers`, `--max-followers`) and `--min-impressions` if set

3) Score and select best
- Collect all qualifying tweets
- Score (higher is better):
  - `followers*1.0 + impressions*0.3 + engagement_total*0.5 + engagement_rate*10000`
- Log top-5 ranking
- Select best → log author and followers: “Selected best tweet: @handle … — followers: N”

4) Generate and post reply
- Classify tone → retrieve context (vector index + human posts + trending)
- Build prompt (optionally image-aware)
- Generate candidates from configured providers; choose best by length/quality
- Suitability checks (length bounds, no copying from original, safe phrasing)
- Post reply via Selenium; store interaction in SQL; send telemetry to LangSmith
- Respect pacing (min seconds between replies)

5) Loop and backoff
- If no qualifying tweets: progressive backoff (60, 120, 180… up to `max_empty_searches`), then exit
- If hit requested `limit`: exit (orchestrator will call again next cycle)

Follower cache sentinel and freshness
- New authors inserted with `follower_count = -1` (unknown)
- On successful fetch, we set `metadata.followers_checked_at`
- We refresh when count is -1/None OR when last check ≥ 30 days

Timeout safety
- Batch follower fetch stops after 10 minutes and proceeds with fetched subset

---

## Dataset update loop (forever when enabled)
File: `update_datasets.py`

- Single cycle (`run_once`)
  - Upload streamed posts from SQL → dataset: `spider-streamed-dataset`
  - Upload interactions and flagged replies from SQL → dataset: `spider-interactions-dataset`
  - Or print LangSmith performance report
- Loop mode (`--loop` or via PowerShell wrapper)
  - while True: run one cycle → sleep `interval` ± `jitter`

---

## Reply metrics refresh loop
Launcher: `scripts/run_all_background*.ps1` → Worker: `spider_guardian/scripts/refresh_my_replies.py`

- Worker (one-shot):
  - Read recent replies (type=interaction, ≤ N days)
  - Open each reply URL; extract likes/replies/reposts/impressions
  - Push updated metrics to LangSmith; exit
- PowerShell wrapper: re-runs worker every configured interval

---

## ML training loop
Launcher: `scripts/run_all_background*.ps1` → Worker: `spider_guardian/ml_bot.py train`

- Worker (one-shot):
  - Train based on enough recent replies/trending posts
  - Exit
- PowerShell wrapper: re-runs daily (default)

---

## Background supervisors (PowerShell)
Files: `scripts/run_all_background.ps1`, `scripts/run_all_background_monitored.ps1`

- Responsibilities
  - Set working dir and env (Twitter cookie, LANGSMITH key)
  - Create `logs/` and manage a `logs/background_pids.json`
  - Start:
    - Guardian Orchestrator (Python, long-running)
    - Dataset update loop (PowerShell while True)
    - Reply metrics refresh loop (PowerShell while True)
    - ML training loop (PowerShell while True)
  - Monitored variant: periodically prints/tails logs

- Logs
  - Orchestrator: `logs/guardian_orchestrator.*.log`
  - Dataset updates: `logs/datasets_update_loop.*.log`
  - Reply refresh: `logs/refresh_replies_loop.*.log`
  - ML training: `logs/ml_training_loop.*.log`

---

## Where to look when it feels “stuck”

- Followers taking long to fetch:
  - We added a 10-minute cap and batch-deduped fetches; progress bars show authors processed
  - Each successful fetch updates SQL with `followers_checked_at`, minimizing re-fetches

- No replies sent for a while:
  - Check filters (`--min-followers`, `--min-impressions`) and backoff logs
  - See ranked candidates log and “Selected best tweet” lines for decisions

- Datasets not updating:
  - Confirm `update_datasets.py` loop logs; verify LANGSMITH_API_KEY is set

- Reply metrics not refreshing:
  - Check refresh worker logs; verify replies are within `--max-age-days`

---

Tip: Use `scripts/run_all_background_monitored.ps1` for a watch-mode experience (colored headings and periodic log summaries).
