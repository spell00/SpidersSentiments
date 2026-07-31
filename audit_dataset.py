"""
Audit and clean spider-interactions-dataset examples.

Scans all examples, reports which have missing generated_reply, and optionally deletes them.
"""
import os
import sys
import logging
from typing import List, Dict, Any
from spider_guardian.langsmith.config import langsmith_integration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def audit_dataset(
    dataset_name: str = "spider-interactions-dataset",
    delete_invalid: bool = False,
    export_csv: bool = False,
) -> Dict[str, Any]:
    """
    Audit all examples in the dataset and report on generated_reply status.
    
    Args:
        dataset_name: LangSmith dataset name
        delete_invalid: If True, delete examples missing generated_reply (interactions only)
        export_csv: If True, export findings to CSV
    
    Returns:
        Dict with summary stats and lists of problematic examples
    """
    client = langsmith_integration.client
    if not client:
        logging.error("LangSmith client not initialized. Set LANGSMITH_API_KEY.")
        return {}
    
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        examples = list(client.list_examples(dataset_id=dataset.id))
        logging.info("Loaded %d examples from '%s'", len(examples), dataset_name)
    except Exception as exc:
        logging.error("Failed to load dataset '%s': %s", dataset_name, exc)
        return {}
    
    # Categorize examples
    valid_interactions: List[Dict[str, Any]] = []
    invalid_interactions: List[Dict[str, Any]] = []
    streamed_posts: List[Dict[str, Any]] = []
    flagged_replies: List[Dict[str, Any]] = []
    unknown_type: List[Dict[str, Any]] = []
    
    for ex in examples:
        meta = ex.metadata or {}
        outputs = ex.outputs or {}
        ex_type = str(meta.get("type") or "").strip().lower()
        
        reply_text = str(outputs.get("generated_reply") or "").strip()
        key = meta.get("key") or meta.get("reply_id") or meta.get("tweet_id") or "unknown"
        
        record = {
            "id": ex.id,
            "type": ex_type or "(empty)",
            "key": key[:60],
            "has_reply": bool(reply_text),
            "reply_length": len(reply_text) if reply_text else 0,
            "created_at": meta.get("created_at", ""),
            "url": (ex.inputs or {}).get("url", "")[:60],
        }
        
        if ex_type == "streamed_post":
            streamed_posts.append(record)
        elif ex_type == "flagged_reply":
            flagged_replies.append(record)
        elif ex_type in ("interaction", ""):
            # Empty type defaults to interaction
            if reply_text:
                valid_interactions.append(record)
            else:
                invalid_interactions.append(record)
        else:
            unknown_type.append(record)
    
    # Summary
    summary = {
        "total_examples": len(examples),
        "valid_interactions": len(valid_interactions),
        "invalid_interactions": len(invalid_interactions),
        "streamed_posts": len(streamed_posts),
        "flagged_replies": len(flagged_replies),
        "unknown_type": len(unknown_type),
    }
    
    print("\n=== Dataset Audit Summary ===")
    print(f"Dataset: {dataset_name}")
    print(f"Total examples: {summary['total_examples']}")
    print(f"  ✓ Valid interactions (with generated_reply): {summary['valid_interactions']}")
    print(f"  ✗ Invalid interactions (missing generated_reply): {summary['invalid_interactions']}")
    print(f"  ℹ Streamed posts (no reply needed): {summary['streamed_posts']}")
    print(f"  ⚑ Flagged replies: {summary['flagged_replies']}")
    print(f"  ? Unknown type: {summary['unknown_type']}")
    
    # Show sample invalid entries
    if invalid_interactions:
        print(f"\n=== Sample Invalid Interactions (first 10 of {len(invalid_interactions)}) ===")
        for i, rec in enumerate(invalid_interactions[:10], 1):
            print(f"{i}. Type='{rec['type']}', Key={rec['key']}, URL={rec['url']}")
    
    # Export to CSV
    if export_csv:
        import csv
        csv_path = "dataset_audit.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "type", "key", "has_reply", "reply_length", "created_at", "url"])
            writer.writeheader()
            for rec in valid_interactions + invalid_interactions + streamed_posts + flagged_replies + unknown_type:
                writer.writerow(rec)
        print(f"\n✓ Exported detailed audit to {csv_path}")
    
    # Delete invalid if requested
    if delete_invalid and invalid_interactions:
        print(f"\n=== Deleting {len(invalid_interactions)} invalid interactions ===")
        deleted = 0
        for rec in invalid_interactions:
            try:
                client.delete_example(example_id=rec["id"])
                deleted += 1
                print(f"Deleted: {rec['key']}")
            except Exception as exc:
                logging.warning("Failed to delete example %s: %s", rec["id"], exc)
        print(f"✓ Deleted {deleted} of {len(invalid_interactions)} invalid examples")
        summary["deleted"] = deleted
    
    print("\n=== Audit Complete ===\n")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Audit and clean LangSmith dataset examples")
    parser.add_argument(
        "--dataset",
        type=str,
        default="spider-interactions-dataset",
        help="LangSmith dataset name (default: spider-interactions-dataset)"
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete invalid interactions (missing generated_reply)"
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export detailed audit to CSV"
    )
    args = parser.parse_args()
    
    if not os.getenv("LANGSMITH_API_KEY"):
        print("ERROR: LANGSMITH_API_KEY not set.")
        print("Set it in PowerShell: $env:LANGSMITH_API_KEY='your-key'")
        sys.exit(1)
    
    if args.delete:
        confirm = input(f"⚠ About to DELETE invalid examples from '{args.dataset}'. Continue? [y/N]: ")
        if confirm.lower() != "y":
            print("Aborted.")
            sys.exit(0)
    
    audit_dataset(
        dataset_name=args.dataset,
        delete_invalid=args.delete,
        export_csv=args.export_csv,
    )
