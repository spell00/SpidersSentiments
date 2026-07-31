"""
Script to periodically update all datasets and databases for Spider Guardian.
Adds optional live progress output so you can see updates as they're collected.

Now supports continuous looping with --loop and configurable interval.
"""
import os
import sys
import json
import logging
import time
import random
from spider_guardian.langsmith.config import langsmith_integration, setup_langsmith_env

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, help="Path to SQLite database")
    parser.add_argument("--dataset", type=str, help="LangSmith dataset name")
    parser.add_argument("--match-key", type=str, default="post_id", help="Key to match examples (default: post_id)")
    
    # Trending dataset operations (default-on with negatives)
    parser.add_argument("--upload", action="store_true", help="Upload trending dataset from DB (default: on)")
    parser.add_argument("--no-upload", action="store_true", help="Disable trending upload")
    parser.add_argument("--update", action="store_true", help="Update trending metrics in LangSmith (default: on)")
    parser.add_argument("--no-update", action="store_true", help="Disable trending update")
    
    parser.add_argument("--max-examples", type=int, default=None, help="Maximum number of examples to process")
    parser.add_argument("--show-browser", action="store_true", help="Open visible Firefox (not headless) for URL resolution")
    parser.add_argument("--url-wait-seconds", type=int, default=12, help="Seconds to wait for final URL resolution")
    
    # SQL dataset operations (default-on with negatives)
    parser.add_argument("--sql-db", type=str, default=None, help="Path to SQL scraped_articles database (default: data/spider_guardian.sqlite)")
    parser.add_argument("--upload-sql-interactions", action="store_true", help="Upload replies/interactions from SQL (default: on)")
    parser.add_argument("--no-upload-sql-interactions", action="store_true", help="Disable interactions upload")
    parser.add_argument("--upload-sql-streamed", action="store_true", help="Upload streamed posts from SQL (default: on)")
    parser.add_argument("--no-upload-sql-streamed", action="store_true", help="Disable streamed upload")
    parser.add_argument("--upload-sql-flagged", action="store_true", help="Upload flagged replies from SQL (default: on)")
    parser.add_argument("--no-upload-sql-flagged", action="store_true", help="Disable flagged upload")
    parser.add_argument("--upload-all-sql", action="store_true", help="Upload all scraped SQL datasets (interactions, streamed, flagged)")
    parser.add_argument("--no-upload-all-sql", action="store_true", help="Disable all SQL uploads")
    parser.add_argument("--refresh-sql", action="store_true", help="Refresh metrics for existing examples (default: on)")
    parser.add_argument("--no-refresh-sql", action="store_true", help="Disable refresh")
    parser.add_argument("--scrape-live-metrics", action="store_true", help="Scrape live metrics from reply URLs during refresh (default: on)")
    parser.add_argument("--no-scrape-live-metrics", action="store_true", help="Disable live scraping during refresh (use DB-only counts)")
    parser.add_argument("--force-refresh-sql", action="store_true", help="Bypass cadence windows and refresh all eligible examples (still respects --max-examples)")
    parser.add_argument("--update-sql-interactions-db", action="store_true", help="Refresh engagement metrics directly in the local SQL interactions table using Selenium")
    parser.add_argument("--max-age-days", type=int, default=3, help="Only refresh DB rows created within the last N days (for --update-sql-interactions-db)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of DB rows to refresh (for --update-sql-interactions-db)")
    parser.add_argument("--force-update-sql-interactions-db", action="store_true", help="Force DB interactions refresh ignoring max-age-days (overrides to include all rows unless --limit set)")
    parser.add_argument("--migrate-sql", action="store_true", help="Migrate legacy scraped_articles to normalized tables (interactions, content)")
    parser.add_argument("--progress", action="store_true", help="Print per-example progress while uploading (live updates) [default: enabled]")
    parser.add_argument("--hide-progress", action="store_true", help="Disable per-example progress output (overrides --progress)")
    # Looping controls
    parser.add_argument("--loop", action="store_true", help="Run forever, repeating updates at a fixed interval")
    parser.add_argument("--interval", type=int, default=3600, help="Seconds to wait between update cycles when --loop is set (default: 3600)")
    parser.add_argument("--jitter", type=float, default=0.1, help="Relative jitter to add to interval (0.1 = ±10%)")
    parser.add_argument("--full-update", action="store_true", help="Run all available update operations in a safe sequence")
    # Debugging / planning
    parser.add_argument("--debug-plan", action="store_true", help="Print the actions that would run (after expanding --full-update) and exit")
    parser.add_argument("--plan-refresh", action="store_true", help="Preview which examples would be refreshed (no writes)")
    # Verification of replies
    parser.add_argument("--verify-replies", action="store_true", help="Check if reply URLs are still visible; print summary")
    parser.add_argument("--delete-missing-replies", action="store_true", help="When used with --verify-replies, delete examples whose replies are missing")
    args = parser.parse_args()
    
    # Convert -1 to None for unlimited processing
    if args.max_examples == -1:
        args.max_examples = None
    
    # Configure logging and optional unbuffered output for live progress
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    # Decide progress behavior: default enabled unless explicitly hidden
    progress_enabled = True
    if getattr(args, "hide_progress", False):
        progress_enabled = False
    elif getattr(args, "progress", False):
        progress_enabled = True
    elif getattr(args, "full_update", False):
        progress_enabled = True

    # Apply environment flag consumed downstream by helpers
    if progress_enabled:
        os.environ["UPD_PROGRESS"] = "1"
    else:
        # Remove if present to avoid stale state in long-lived shells
        if "UPD_PROGRESS" in os.environ:
            del os.environ["UPD_PROGRESS"]

    def _auto_defaults_disabled() -> bool:
        """Check env override to disable automatic default actions."""
        return os.getenv("DISABLE_AUTO_DEFAULTS", "").strip().lower() in {"1", "true", "yes", "y"}

    def _expand_full_update(mut):
        """Return a shallow-copied args-like object with --full-update expansions applied (no side effects)."""
        from types import SimpleNamespace
        # Shallow copy
        a = SimpleNamespace(**vars(mut))
        if getattr(a, "full_update", False):
            # Auto-enable trending upload and update when db+dataset provided
            if getattr(a, "db", None) and getattr(a, "dataset", None):
                if not getattr(a, "upload", False):
                    a.upload = True
                if not getattr(a, "update", False):
                    a.update = True
            # Fallback: enable upload when default trending DB exists (no dataset needed)
            elif not getattr(a, "upload", False) and not getattr(a, "update", False):
                if os.path.exists("data/spider_trending.sqlite"):
                    a.upload = True
                    a.db = "data/spider_trending.sqlite"
            # Ensure SQL DB default path
            if not getattr(a, "sql_db", None):
                a.sql_db = "data/spider_guardian.sqlite"
            # Enable all SQL uploads and refresh
            a.upload_all_sql = True
            a.refresh_sql = True
            # Prefer accurate metrics during full update
            if not getattr(a, "no_scrape_live_metrics", False):
                a.scrape_live_metrics = True
            # Include reply visibility verification by default during full update
            a.verify_replies = True
            # Attempt DB refresh of recent rows unless explicitly disabled
            if not getattr(a, "update_sql_interactions_db", False):
                a.update_sql_interactions_db = True
                if not hasattr(a, "max_age_days") or a.max_age_days is None:
                    a.max_age_days = 3
        return a

    def _has_any_action_flag(a) -> bool:
        """Return True if any action-like flags are explicitly set by the user."""
        action_flags = (
            "upload", "update", "migrate_sql", "refresh_sql", "update_sql_interactions_db",
            "upload_sql_interactions", "upload_sql_streamed", "upload_sql_flagged", "upload_all_sql",
            "verify_replies",
            # Include negative flags to detect explicit user override
            "no_upload", "no_update", "no_refresh_sql",
            "no_upload_sql_interactions", "no_upload_sql_streamed", "no_upload_sql_flagged", "no_upload_all_sql",
        )
        return any(bool(getattr(a, f, False)) for f in action_flags)

    def _expand_auto_defaults(mut):
        """Apply automatic default actions when user provides no explicit action flags.

        Defaults: upload/update trending, upload all SQL datasets, refresh interactions, verify replies.
        Guarded by DISABLE_AUTO_DEFAULTS env var and negative flags.
        """
        from types import SimpleNamespace
        a = SimpleNamespace(**vars(mut))
        if _auto_defaults_disabled():
            return a
        if _has_any_action_flag(a) or getattr(a, "full_update", False):
            # User gave explicit flags; apply negative flag overrides only
            if getattr(a, "no_upload", False):
                a.upload = False
            if getattr(a, "no_update", False):
                a.update = False
            if getattr(a, "no_refresh_sql", False):
                a.refresh_sql = False
            if getattr(a, "no_upload_sql_interactions", False):
                a.upload_sql_interactions = False
            if getattr(a, "no_upload_sql_streamed", False):
                a.upload_sql_streamed = False
            if getattr(a, "no_upload_sql_flagged", False):
                a.upload_sql_flagged = False
            if getattr(a, "no_upload_all_sql", False):
                a.upload_all_sql = False
            # Respect negative toggles for refresh behaviors
            if getattr(a, "no_scrape_live_metrics", False):
                a.scrape_live_metrics = False
            return a
        
        # No explicit actions provided: enable defaults
        enabled = []
        
        # Ensure default SQL DB path
        if not getattr(a, "sql_db", None):
            a.sql_db = "data/spider_guardian.sqlite"
        
        # Default: upload trending if DB exists
        if not getattr(a, "no_upload", False):
            if os.path.exists("data/spider_trending.sqlite"):
                a.upload = True
                a.db = "data/spider_trending.sqlite"
                enabled.append("upload")
        
        # Default: update trending if DB + dataset params present
        if not getattr(a, "no_update", False):
            if getattr(a, "db", None) and getattr(a, "dataset", None):
                a.update = True
                enabled.append("update")
        
        # Default: upload all SQL datasets
        if not getattr(a, "no_upload_all_sql", False):
            a.upload_all_sql = True
            enabled.append("upload_all_sql")
        
        # Default: refresh interactions
        if not getattr(a, "no_refresh_sql", False):
            a.refresh_sql = True
            enabled.append("refresh_sql")
            # Default to live scraping for accuracy unless explicitly disabled
            if not getattr(a, "no_scrape_live_metrics", False):
                a.scrape_live_metrics = True
                enabled.append("scrape_live_metrics")
        
        # Default: verify replies
        a.verify_replies = True
        enabled.append("verify_replies")
        
        if enabled:
            logging.info(f"[auto-defaults] Enabled: {', '.join(enabled)} (set DISABLE_AUTO_DEFAULTS=1 or use --no-* to disable)")
        
        # Provide a sensible default progress mode
        os.environ.setdefault("UPD_PROGRESS", "1")
        return a

    def _build_plan(a):
        """Build a list of human-readable actions that would be executed based on args."""
        plan = []
        # Note: detect LangSmith availability without mutating env
        has_ls = bool(os.getenv("LANGSMITH_API_KEY"))
        plan.append(f"[env] LANGSMITH_API_KEY: {'set' if has_ls else 'missing'}")
        # Loop info (scheduling)
        if getattr(a, "loop", False):
            base = max(5, int(getattr(a, "interval", 3600)))
            jitter = max(0.0, float(getattr(a, "jitter", 0.1)))
            plan.append(f"[loop] Continuous: interval={base}s, jitter=±{int(jitter*100)}%")
        # Actions gated by LangSmith
        if has_ls:
            if getattr(a, "upload", False):
                db_path = a.db if getattr(a, "db", None) else "data/spider_trending.sqlite"
                plan.append(f"[upload] Trending dataset from DB: {db_path} (max_examples={a.max_examples}, show_browser={a.show_browser}, url_wait_seconds={a.url_wait_seconds})")
            if getattr(a, "update", False):
                if not getattr(a, "db", None) or not getattr(a, "dataset", None):
                    plan.append("[update] SKIP (requires --db and --dataset)")
                else:
                    plan.append(f"[update] Update LangSmith dataset from DB: db={a.db}, dataset={a.dataset}, match_key={a.match_key}, max_examples={a.max_examples}")
            if getattr(a, "migrate_sql", False):
                sql_path = a.sql_db if getattr(a, "sql_db", None) else "data/spider_guardian.sqlite"
                plan.append(f"[migrate] Migrate scraped_articles to normalized tables in {sql_path}")
            if getattr(a, "refresh_sql", False):
                sql_path = a.sql_db if getattr(a, "sql_db", None) else "data/spider_guardian.sqlite"
                plan.append(f"[refresh] Refresh interactions dataset from SQL: {sql_path} (max_examples={a.max_examples}, live={getattr(a,'scrape_live_metrics', False)}, force={getattr(a,'force_refresh_sql', False)})")
            if getattr(a, "update_sql_interactions_db", False):
                sql_path = a.sql_db if getattr(a, "sql_db", None) else "data/spider_guardian.sqlite"
                limit = a.limit if getattr(a, "limit", None) is not None else a.max_examples
                plan.append(f"[db-refresh] Refresh interactions table in DB via Selenium: {sql_path} (max_age_days={a.max_age_days}, limit={limit}, show_browser={a.show_browser}, driver=firefox)")
            if getattr(a, "upload_sql_interactions", False) or getattr(a, "upload_all_sql", False):
                sql_path = a.sql_db if getattr(a, "sql_db", None) else "data/spider_guardian.sqlite"
                plan.append(f"[upload] Upload replies from SQL: {sql_path} (dataset=spider-interactions-dataset, max_examples={a.max_examples})")
            if getattr(a, "upload_sql_streamed", False) or getattr(a, "upload_all_sql", False):
                sql_path = a.sql_db if getattr(a, "sql_db", None) else "data/spider_guardian.sqlite"
                plan.append(f"[upload] Upload streamed posts from SQL: {sql_path} (dataset=spider-streamed-dataset, max_examples={a.max_examples})")
            if getattr(a, "upload_sql_flagged", False) or getattr(a, "upload_all_sql", False):
                sql_path = a.sql_db if getattr(a, "sql_db", None) else "data/spider_guardian.sqlite"
                plan.append(f"[upload] Upload flagged replies from SQL: {sql_path} (dataset=spider-interactions-dataset, max_examples={a.max_examples})")
            # If nothing else will run, report generation
            actions_only = [p for p in plan if p.startswith("[") and not p.startswith("[env]") and not p.startswith("[loop]")]
            if not actions_only:
                plan.append("[report] Generate performance report (no uploads/updates selected)")
        else:
            plan.append("[langsmith] SKIP uploads/updates (no API key)")
        return plan

    def debug_plan(mut_args):
        # Apply full-update expansion first, then auto-defaults if still no actions
        expanded = _expand_full_update(mut_args)
        expanded = _expand_auto_defaults(expanded)
        plan = _build_plan(expanded)
        print("\n=== Debug Plan ===")
        for line in plan:
            print(line)
        print("=== End Plan ===\n")

    def run_once(args) -> bool:
        """Run a single update cycle based on provided flags. Returns True if any action executed."""
        if not os.getenv("LANGSMITH_API_KEY"):
            # Try to set up env; if still missing, continue with non-upload operations (if any in future)
            setup_langsmith_env()

        did_something = False
        last_refreshed_keys = None  # keys updated during this run (for scoping verification)
        if os.getenv("LANGSMITH_API_KEY"):
            print("[langsmith] API key found", flush=True)
            print(f"[langsmith] Project URL: {langsmith_integration.get_langsmith_url()}", flush=True)

            # Expand flags for full-update or apply auto-defaults when no actions were provided
            if args.full_update:
                expanded = _expand_full_update(args)
                # Reflect expansions back to args for execution
                for k, v in vars(expanded).items():
                    setattr(args, k, v)
                print("[full-update] Expanded flags: migrate_sql, refresh_sql, upload_all_sql, update_sql_interactions_db, verify_replies", flush=True)
            else:
                before = _has_any_action_flag(args)
                expanded = _expand_auto_defaults(args)
                # Reflect expansions back to args for execution (may be identical)
                for k, v in vars(expanded).items():
                    setattr(args, k, v)
                if not before and _has_any_action_flag(args):
                    print("[auto-defaults] Enabled: upload_all_sql, refresh_sql, verify_replies (set DISABLE_AUTO_DEFAULTS=1 to disable)", flush=True)

            if args.upload:
                langsmith_integration.upload_dataset_from_db(
                    args.db if args.db else "data/spider_trending.sqlite",
                    max_examples=args.max_examples,
                    show_browser=args.show_browser,
                    url_wait_seconds=args.url_wait_seconds,
                )
                did_something = True
            if args.update:
                if not args.db or not args.dataset:
                    print("--db and --dataset are required for --update mode")
                else:
                    langsmith_integration.update_dataset_from_db(
                        args.db, args.dataset, args.match_key, max_examples=args.max_examples
                    )
                    did_something = True
            if args.migrate_sql:
                from spider_guardian.storage.sql import SQLDataStore
                path = args.sql_db if args.sql_db else "data/spider_guardian.sqlite"
                store = SQLDataStore(path)
                res = store.migrate_scraped_articles()
                print(f"[migrate] migrated: {res}", flush=True)
                did_something = True
            if args.refresh_sql:
                path = args.sql_db if args.sql_db else "data/spider_guardian.sqlite"
                if args.plan_refresh:
                    print(f"[plan-refresh] starting preview: dataset=spider-interactions-dataset db={path} max_examples={args.max_examples}", flush=True)
                    summary = langsmith_integration.plan_refresh_interactions_dataset_from_sql(
                        db_path=path,
                        dataset_name="spider-interactions-dataset",
                        max_examples=args.max_examples,
                        report=True,
                    )
                    print(f"[plan-refresh] summary: would_update={summary.get('updated_would')} scanned={summary.get('scanned')} partial={summary.get('partial')}", flush=True)
                    # If nothing eligible, we skip real refresh
                    if summary.get("updated_would", 0) == 0:
                        print("[refresh] skipped (no eligible updates)", flush=True)
                        did_something = True
                    else:
                        print(f"[refresh] starting (will attempt up to {args.max_examples} actual updates)", flush=True)
                        res = langsmith_integration.refresh_interactions_dataset_from_sql(
                            db_path=path,
                            dataset_name="spider-interactions-dataset",
                            max_examples=args.max_examples,
                            scrape_live=args.scrape_live_metrics,
                            force_refresh=args.force_refresh_sql,
                        )
                        # Handle new summary return type (backward compatible if int)
                        if isinstance(res, dict):
                            updated = int(res.get("updated", 0) or 0)
                            last_refreshed_keys = list(res.get("keys", []) or [])
                        else:
                            updated = int(res or 0)
                        print(f"[refresh] spider-interactions-dataset: {updated} updated", flush=True)
                        did_something = True
                else:
                    print(f"[refresh-sql] starting: dataset=spider-interactions-dataset db={path} max_examples={args.max_examples} scrape_live={args.scrape_live_metrics}", flush=True)
                    res = langsmith_integration.refresh_interactions_dataset_from_sql(
                        db_path=path,
                        dataset_name="spider-interactions-dataset",
                        max_examples=args.max_examples,
                        scrape_live=args.scrape_live_metrics,
                        force_refresh=args.force_refresh_sql,
                    )
                    if isinstance(res, dict):
                        updated = int(res.get("updated", 0) or 0)
                        last_refreshed_keys = list(res.get("keys", []) or [])
                    else:
                        updated = int(res or 0)
                    print(f"[refresh-sql] spider-interactions-dataset: {updated} updated", flush=True)
                    did_something = True
            if args.verify_replies:
                # Always run on interactions dataset by default
                summary = langsmith_integration.verify_reply_visibility(
                    dataset_name="spider-interactions-dataset",
                    delete_missing=bool(args.delete_missing_replies),
                    max_examples=args.max_examples,
                    limit_keys=last_refreshed_keys,
                )
                print("[verify-replies]", json.dumps(summary, indent=2), flush=True)
                did_something = True
            if args.update_sql_interactions_db:
                path = args.sql_db if args.sql_db else "data/spider_guardian.sqlite"
                from spider_guardian.scripts.update_interactions_db import refresh_interactions_in_db
                print(f"[db-refresh] starting: interactions table in {path} (max_age_days={args.max_age_days}, limit={args.limit if args.limit is not None else args.max_examples})", flush=True)
                effective_max_age = args.max_age_days
                if args.force_update_sql_interactions_db:
                    effective_max_age = 9999
                updated = refresh_interactions_in_db(
                    db_path=path,
                    max_age_days=effective_max_age,
                    limit=args.limit if args.limit is not None else args.max_examples,
                    show_browser=args.show_browser,
                    driver="firefox",
                )
                print(f"[db-refresh] interactions table: {updated} rows updated", flush=True)
                did_something = True
            if args.upload_sql_interactions or args.upload_all_sql:
                print(f"[upload] starting: interactions -> spider-interactions-dataset (max_examples={args.max_examples})", flush=True)
                res = langsmith_integration.upload_replies_from_sql(
                    args.sql_db if args.sql_db else "data/spider_guardian.sqlite",
                    dataset_name="spider-interactions-dataset",
                    max_examples=args.max_examples,
                )
                print(f"[upload] replies: {res}", flush=True)
                did_something = True
            if args.upload_sql_streamed or args.upload_all_sql:
                print(f"[upload] starting: streamed -> spider-streamed-dataset (max_examples={args.max_examples})", flush=True)
                res = langsmith_integration.upload_streamed_from_sql(
                    args.sql_db if args.sql_db else "data/spider_guardian.sqlite",
                    max_examples=args.max_examples,
                )
                print(f"[upload] streamed posts: {res}", flush=True)
                did_something = True
            if args.upload_sql_flagged or args.upload_all_sql:
                print(f"[upload] starting: flagged -> spider-interactions-dataset (max_examples={args.max_examples})", flush=True)
                res = langsmith_integration.upload_flagged_from_sql(
                    args.sql_db if args.sql_db else "data/spider_guardian.sqlite",
                    dataset_name="spider-interactions-dataset",
                    max_examples=args.max_examples,
                )
                print(f"[upload] flagged replies: {res}", flush=True)
                did_something = True
            if not did_something:
                report = langsmith_integration.generate_performance_report()
                print(json.dumps(report, indent=2), flush=True)
                did_something = True
        else:
            print("[langsmith] API key not configured; skipping LangSmith uploads/updates.", flush=True)
        return did_something

    # Early exit for debug planning (no side effects)
    if args.debug_plan:
        debug_plan(args)
        sys.exit(0)

    if args.loop:
        base = max(5, int(args.interval))
        jitter = max(0.0, float(args.jitter))
        logging.info("[loop] Continuous mode enabled: interval=%ss, jitter=±%d%%", base, int(jitter * 100))
        try:
            cycle = 1
            while True:
                logging.info("[loop] === Update cycle %d starting ===", cycle)
                try:
                    run_once(args)
                except Exception as exc:
                    logging.exception("[loop] Update cycle failed: %s", exc)
                # Sleep with jitter to avoid thundering herd
                delta = 1.0 + random.uniform(-jitter, jitter) if jitter > 0 else 1.0
                wait = max(5, int(base * delta))
                logging.info("[loop] Sleeping %ss before next cycle", wait)
                time.sleep(wait)
                cycle += 1
        except KeyboardInterrupt:
            logging.info("[loop] Received interrupt. Exiting cleanly.")
    else:
        # Single-shot mode
        run_once(args)