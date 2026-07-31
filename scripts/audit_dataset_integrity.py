"""Audit LangSmith dataset & local DB for engagement anomalies and integrity issues.

Usage (PowerShell):

    python scripts/audit_dataset_integrity.py --dataset spider-interactions-dataset --db data/spider_guardian.sqlite --top-n 50

Flags:
    --dataset            LangSmith dataset name (default: spider-interactions-dataset)
    --db                 Path to local SQLite DB (for cross-check; default: data/spider_guardian.sqlite)
    --top-n              Examine top-N examples by impressions for deep checks (default: 100)
    --export-json        Path to write full anomaly report JSON (optional)
    --include-streamed   Also audit streamed dataset (spider-streamed-dataset)

Findings categories:
    unreachable_url          HTTP error or timeout (>6s) on HEAD/GET
    metrics_outlier          Impressions above P99.5 or likes above P99.5 of distribution
    improbable_ratio         likes > impressions OR replies > impressions OR likes/impressions > 0.5 for large impressions
    missing_reply            interaction type without generated_reply (should be pruned already)
    empty_metrics            all metrics zero but age > 1 day
    stale_after_schedule     updates_done == schedule length but metrics still changing in DB (schedule exhaustion)
    duplicate_key_conflict   same key mapped to multiple different URLs/texts

The script is read-only toward LangSmith; it does NOT modify examples.
"""
from __future__ import annotations
import os
import json
import math
import statistics
import sqlite3
import logging
import argparse
import time
from typing import Any, Dict, List, Tuple
from datetime import datetime, timedelta

try:
    from langsmith import Client  # type: ignore
except Exception:
    Client = None  # type: ignore
import requests

SCHEDULE = [1, 2, 3]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)


def _percentile(sorted_vals: List[int], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if pct <= 0:
        return float(sorted_vals[0])
    if pct >= 1:
        return float(sorted_vals[-1])
    idx = pct * (len(sorted_vals) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(sorted_vals[lo])
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def load_dataset_examples(dataset_name: str) -> List[Any]:
    if not Client or not os.getenv("LANGSMITH_API_KEY"):
        logging.warning("LangSmith unavailable; skipping remote dataset load")
        return []
    client = Client(
        api_key=os.getenv("LANGSMITH_API_KEY"),
        api_url=os.getenv("LANGSMITH_API_URL", "https://api.smith.langchain.com"),
    )
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        return list(client.list_examples(dataset_id=dataset.id))
    except Exception as exc:
        logging.error("Failed to load dataset '%s': %s", dataset_name, exc)
        return []


def load_db_rows(db_path: str) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(db_path):
        logging.warning("DB path %s does not exist", db_path)
        return rows
    try:
        conn = sqlite3.connect(db_path)
        # interactions
        try:
            inter = conn.execute("SELECT tweet_id, reply_id, url, like_count, reply_count, impression_count, repost_count, created_at FROM interactions").fetchall()
            for r in inter:
                tweet_id, reply_id, url, like_c, reply_c, impr_c, repost_c, created_at = r
                row = {
                    "tweet_id": tweet_id,
                    "reply_id": reply_id,
                    "url": url,
                    "like_count": like_c or 0,
                    "reply_count": reply_c or 0,
                    "impression_count": impr_c or 0,
                    "repost_count": repost_c or 0,
                    "created_at": created_at,
                    "table": "interactions",
                }
                for k in [str(tweet_id or ""), str(reply_id or ""), str(url or "")]:
                    if k:
                        rows.setdefault(k, row)
        except Exception:
            pass
        # content
        try:
            cont = conn.execute("SELECT post_id, url, like_count, reply_count, impression_count, repost_count, created_at FROM content").fetchall()
            for r in cont:
                post_id, url, like_c, reply_c, impr_c, repost_c, created_at = r
                row = {
                    "post_id": post_id,
                    "url": url,
                    "like_count": like_c or 0,
                    "reply_count": reply_c or 0,
                    "impression_count": impr_c or 0,
                    "repost_count": repost_c or 0,
                    "created_at": created_at,
                    "table": "content",
                }
                for k in [str(post_id or ""), str(url or "")]:
                    if k:
                        rows.setdefault(k, row)
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return rows


def quick_url_check(url: str, timeout: float = 6.0) -> Tuple[bool, int]:
    if not url:
        return False, 0
    try:
        # Prefer HEAD; fallback to GET if not allowed
        resp = requests.head(url, allow_redirects=True, timeout=timeout)
        if resp.status_code >= 400 or resp.status_code == 405:
            resp = requests.get(url, allow_redirects=True, timeout=timeout)
        return (200 <= resp.status_code < 400), resp.status_code
    except Exception:
        return False, 0


def analyze_examples(examples: List[Any], db_rows: Dict[str, Dict[str, Any]], top_n: int) -> Dict[str, Any]:
    anomalies: Dict[str, List[Dict[str, Any]]] = {
        "unreachable_url": [],
        "metrics_outlier": [],
        "improbable_ratio": [],
        "missing_reply": [],
        "empty_metrics": [],
        "stale_after_schedule": [],
        "duplicate_key_conflict": [],
    }
    metrics_impressions: List[int] = []
    metrics_likes: List[int] = []
    seen_keys: Dict[str, Dict[str, Any]] = {}

    # Collect metrics for distribution
    for ex in examples:
        meta = ex.metadata or {}
        outputs = ex.outputs or {}
        em = (outputs.get("engagement_metrics") or {})
        metrics_impressions.append(int(em.get("impressions", 0)))
        metrics_likes.append(int(em.get("likes", 0)))

    metrics_impressions_sorted = sorted(metrics_impressions)
    metrics_likes_sorted = sorted(metrics_likes)
    p995_impr = _percentile(metrics_impressions_sorted, 0.995)
    p995_likes = _percentile(metrics_likes_sorted, 0.995)

    # Examine examples (restrict deep URL checks to top N by impressions)
    ranked = sorted(examples, key=lambda e: ((e.outputs or {}).get("engagement_metrics", {}).get("impressions", 0)), reverse=True)
    deep_set = set(ranked[:top_n])

    for ex in examples:
        meta = ex.metadata or {}
        inputs = ex.inputs or {}
        outputs = ex.outputs or {}
        em = (outputs.get("engagement_metrics") or {})
        key = str(meta.get("key") or inputs.get("url") or meta.get("url") or meta.get("reply_id") or meta.get("tweet_id") or meta.get("post_id") or "")
        url = str(inputs.get("url") or meta.get("url") or "")
        ex_type = meta.get("type") or "interaction"
        impressions = int(em.get("impressions", 0) or 0)
        likes = int(em.get("likes", 0) or 0)
        replies = int(em.get("replies", 0) or 0)
        updates_done = int(meta.get("updates_done", 0) or 0)
        created_at_str = meta.get("created_at") or meta.get("collected_at")
        try:
            created_at = datetime.fromisoformat(created_at_str) if created_at_str else None
        except Exception:
            created_at = None

        # duplicate key conflicts
        if key:
            sig = {
                "url": url[:120],
                "tweet_text": (inputs.get("tweet_text") or "")[:120],
                "reply_text": (outputs.get("generated_reply") or "")[:120],
            }
            if key not in seen_keys:
                seen_keys[key] = sig
            else:
                if seen_keys[key] != sig:
                    anomalies["duplicate_key_conflict"].append({"key": key, "existing": seen_keys[key], "new": sig})

        # missing reply (should be rare due to pruning)
        if ex_type in ("interaction", "flagged_reply"):
            reply_text = str(outputs.get("generated_reply") or "").strip()
            if not reply_text:
                anomalies["missing_reply"].append({"key": key, "url": url})

        # empty metrics after >1 day
        if impressions == 0 and likes == 0 and replies == 0 and created_at and (datetime.utcnow() - created_at) > timedelta(days=1):
            anomalies["empty_metrics"].append({"key": key, "age_days": (datetime.utcnow() - created_at).days})

        # improbable ratio conditions
        if impressions > 0:
            like_ratio = likes / impressions if impressions else 0
            if likes > impressions or replies > impressions or like_ratio > 0.5 and impressions > 1000:
                anomalies["improbable_ratio"].append({"key": key, "url": url, "impressions": impressions, "likes": likes, "replies": replies})

        # metrics outlier (beyond P99.5)
        if impressions > p995_impr or likes > p995_likes:
            anomalies["metrics_outlier"].append({"key": key, "impressions": impressions, "likes": likes, "threshold_impr_p995": p995_impr, "threshold_likes_p995": p995_likes})

        # stale after schedule: schedule exhausted but DB row shows further changes
        if updates_done >= len(SCHEDULE) and key in db_rows:
            db_r = db_rows[key]
            db_impr = int(db_r.get("impression_count", 0) or 0)
            if db_impr != impressions:
                anomalies["stale_after_schedule"].append({"key": key, "dataset_impr": impressions, "db_impr": db_impr})

        # deep URL reachability check for top N examples
        if ex in deep_set and url:
            ok, status = quick_url_check(url)
            if not ok:
                anomalies["unreachable_url"].append({"key": key, "url": url, "http_status": status})

    return {
        "counts": {k: len(v) for k, v in anomalies.items()},
        "anomalies": anomalies,
        "distribution": {
            "impressions_p995": p995_impr,
            "likes_p995": p995_likes,
            "total_examples": len(examples),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="spider-interactions-dataset")
    ap.add_argument("--db", type=str, default="data/spider_guardian.sqlite")
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--export-json", type=str, default=None)
    ap.add_argument("--include-streamed", action="store_true")
    args = ap.parse_args()

    examples = load_dataset_examples(args.dataset)
    db_rows = load_db_rows(args.db)
    audit = analyze_examples(examples, db_rows, args.top_n)

    print("\n=== Integrity Audit:", args.dataset, "===")
    print(json.dumps(audit["counts"], indent=2))
    print("Distribution:", json.dumps(audit["distribution"], indent=2))

    for category, items in audit["anomalies"].items():
        if not items:
            continue
        print(f"\n-- {category} (showing up to 10) --")
        for it in items[:10]:
            print(json.dumps(it, ensure_ascii=False))

    if args.export_json:
        with open(args.export_json, "w", encoding="utf-8") as f:
            json.dump(audit, f, ensure_ascii=False, indent=2)
        print(f"[export] wrote {args.export_json}")

    if args.include_streamed:
        streamed = load_dataset_examples("spider-streamed-dataset")
        audit_streamed = analyze_examples(streamed, db_rows, args.top_n)
        print("\n=== Integrity Audit: spider-streamed-dataset ===")
        print(json.dumps(audit_streamed["counts"], indent=2))
        print("Distribution:", json.dumps(audit_streamed["distribution"], indent=2))

if __name__ == "__main__":
    main()
