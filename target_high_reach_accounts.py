#!/usr/bin/env python3
"""Target high-reach accounts for maximum visibility.

This script helps you:
1. Enrich your dataset with follower counts
2. Filter to only reply to accounts with many followers
3. Create experiment datasets prioritized by author reach
"""

import argparse
import logging
from spider_guardian.langsmith.config import langsmith_integration

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Target high-reach accounts")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Enrich command
    enrich = subparsers.add_parser("enrich", help="Add follower counts to dataset")
    enrich.add_argument("--dataset", required=True, help="Dataset name to enrich")
    enrich.add_argument("--max", type=int, help="Max examples to enrich")
    enrich.add_argument("--force", action="store_true", help="Re-fetch even if already has count")
    
    # Create experiment command
    experiment = subparsers.add_parser("experiment", help="Create high-reach experiment dataset")
    experiment.add_argument("--source", required=True, help="Source dataset name")
    experiment.add_argument("--name", required=True, help="Experiment name (e.g., high-reach-nov-12)")
    experiment.add_argument("--min-followers", type=int, default=10000, help="Minimum follower count (default: 10000)")
    experiment.add_argument("--min-impressions", type=int, default=1000, help="Minimum impressions (default: 1000)")
    experiment.add_argument("--max-examples", type=int, default=50, help="Max tweets to include (default: 50)")
    
    # Evaluate command
    evaluate = subparsers.add_parser("evaluate", help="Evaluate experiment results")
    evaluate.add_argument("--dataset", required=True, help="Experiment dataset name")
    evaluate.add_argument("--window", type=int, default=3, help="Days since reply (default: 3)")
    
    args = parser.parse_args()
    
    if args.command == "enrich":
        print(f"\n🔍 Enriching {args.dataset} with follower counts...")
        print("⚠️  This will take time - fetches from Twitter profiles (2s delay per author)")
        result = langsmith_integration.enrich_dataset_with_follower_counts(
            dataset_name=args.dataset,
            max_examples=args.max,
            skip_existing=not args.force,
        )
        print(f"\n✅ Enrichment complete:")
        print(f"   • Enriched: {result['enriched']}")
        print(f"   • Skipped: {result['skipped']}")
        print(f"   • Failed: {result['failed']}")
        
    elif args.command == "experiment":
        print(f"\n🎯 Creating experiment targeting {args.min_followers}+ follower accounts...")
        result = langsmith_integration.run_impressions_experiment(
            source_dataset=args.source,
            experiment_name=args.name,
            model_variants=["default"],  # You can customize this
            max_examples=args.max_examples,
            min_input_impressions=args.min_impressions,
            min_followers=args.min_followers,
        )
        if "error" in result:
            print(f"\n❌ Error: {result['error']}")
            return
        
        print(f"\n✅ Experiment dataset created: {result['experiment_dataset']}")
        print(f"\n📊 Configuration:")
        print(f"   • Min followers: {args.min_followers:,}")
        print(f"   • Min impressions: {args.min_impressions:,}")
        print(f"   • Max examples: {args.max_examples}")
        print(f"\n📝 Next steps:")
        for step in result['next_steps']:
            print(f"   • {step}")
        
    elif args.command == "evaluate":
        print(f"\n📊 Evaluating {args.dataset}...")
        result = langsmith_integration.evaluate_impressions_experiment(
            experiment_dataset=args.dataset,
            window_days=args.window,
        )
        if "error" in result:
            print(f"\n❌ Error: {result['error']}")
            return
        
        print(f"\n✅ Evaluation Results ({args.window} day window):")
        print(f"   • Total replies: {result.get('total_replies', 0)}")
        print(f"   • Avg reply impressions: {result.get('avg_reply_impressions', 0):,.0f}")
        print(f"   • Avg input impressions: {result.get('avg_input_impressions', 0):,.0f}")
        print(f"   • Lift ratio: {result.get('impression_lift_ratio', 0):.3f}")
        
        top = result.get('top_performers', [])
        if top:
            print(f"\n🏆 Top {len(top)} performers:")
            for i, perf in enumerate(top[:5], 1):
                print(f"   {i}. {perf['reply_impressions']:,} impressions - {perf['reply_url']}")
                print(f"      └─ {perf['reply_text']}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
