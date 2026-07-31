"""Master orchestration script that coordinates all spider advocacy power-ups."""

import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict

from spider_guardian.config import SpiderGuardianConfig
from spider_guardian.bot import SpiderGuardianBot
from spider_guardian.storage.sql import SQLDataStore
from spider_guardian.fact_check import SpiderFactChecker
from spider_guardian.engagement_optimizer import EngagementOptimizer
from spider_guardian.langsmith.config import (
    upload_dataset_from_db,
    update_dataset_from_db,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class SpiderAdvocacyOrchestrator:
    """Coordinate all spider advocacy features for maximum impact."""
    
    def __init__(self, config: SpiderGuardianConfig):
        self.config = config
        self.sql_store = SQLDataStore(config.sql_database_path)
        self.fact_checker = SpiderFactChecker()
        self.engagement_optimizer = EngagementOptimizer()
    
    def run_full_cycle(
        self,
        respond: bool = False,
        refresh_replies: bool = True,
        update_datasets: bool = True,
        analyze_engagement: bool = True,
        analyze_trends: bool = False,
        train_ml: bool = False,
    ):
        """
        Run a full spider advocacy cycle.
        
        Args:
            respond: Generate and post new replies
            refresh_replies: Scrape updated metrics for posted replies
            update_datasets: Push updates to LangSmith datasets
            analyze_engagement: Analyze what's working and optimize
            analyze_trends: Generate sentiment trend reports
            train_ml: Train machine learning models on historical data
        """
        logger.info("🕷️ Starting Spider Advocacy Orchestrator")
        
        results = {}
        
        # 1. Respond to new tweets (if enabled)
        if respond:
            logger.info("📢 Phase 1: Responding to tweets...")
            try:
                from spider_guardian.bot import SpiderGuardianBot
                bot = SpiderGuardianBot(self.config)
                reply_count = bot.respond_to_tweets(
                    max_new_replies=self.config.max_new_replies
                )
                results["new_replies"] = reply_count
                logger.info(f"✅ Posted {reply_count} new replies")
            except Exception as e:
                logger.error(f"❌ Error responding to tweets: {e}")
                results["new_replies"] = 0
        
        # 2. Refresh metrics for posted replies
        if refresh_replies:
            logger.info("🔄 Phase 2: Refreshing reply metrics...")
            try:
                from spider_guardian.scripts.refresh_my_replies import refresh_reply_metrics
                updated_count = refresh_reply_metrics(
                    db_path=self.config.sql_database_path,
                    dataset_name="trending-dataset",
                    max_age_days=3,
                )
                results["refreshed_replies"] = updated_count
                logger.info(f"✅ Refreshed {updated_count} replies")
            except Exception as e:
                logger.error(f"❌ Error refreshing replies: {e}")
                results["refreshed_replies"] = 0
        
        # 3. Update LangSmith datasets
        if update_datasets:
            logger.info("📊 Phase 3: Updating datasets...")
            try:
                # Update trending dataset metrics
                _ = update_dataset_from_db(
                    dataset_name="trending-dataset",
                    db_path=self.config.sql_database_path,
                    max_examples=500,
                )

                # Ensure all SQL-sourced data are present in LangSmith datasets
                from spider_guardian.langsmith.config import langsmith_integration as _ls
                # Consolidated: use spider-interactions-dataset for replies + flagged; spider-streamed-dataset for streamed posts
                rep = _ls.upload_replies_from_sql(self.config.sql_database_path, dataset_name="spider-interactions-dataset", max_examples=500)
                flg = _ls.upload_flagged_from_sql(self.config.sql_database_path, dataset_name="spider-interactions-dataset", max_examples=200)
                stm = _ls.upload_streamed_from_sql(self.config.sql_database_path, dataset_name="spider-streamed-dataset", max_examples=500)
                results["dataset_updates"] = {
                    "trending_updated": True,
                    "interactions": rep,
                    "flagged": flg,
                    "streamed": stm,
                }
                logger.info("✅ Datasets ensured (consolidated): interactions=%s flagged=%s streamed=%s", rep, flg, stm)
            except Exception as e:
                logger.error(f"❌ Error updating datasets: {e}")
                results["dataset_updates"] = {"error": str(e)}
        
        # 4. Analyze engagement patterns
        if analyze_engagement:
            logger.info("📈 Phase 4: Analyzing engagement...")
            try:
                replies = self._fetch_reply_history()
                if len(replies) >= 10:
                    analysis = self.engagement_optimizer.analyze_top_performers(replies)
                    results["engagement_analysis"] = analysis
                    
                    # Log insights
                    patterns = analysis.get("patterns", {})
                    logger.info("🎯 Engagement Insights:")
                    
                    if "optimal_length" in patterns:
                        length_info = patterns["optimal_length"]
                        logger.info(f"  📏 Optimal length: {length_info['recommendation']}")
                    
                    if "emoji_effectiveness" in patterns:
                        emoji_info = patterns["emoji_effectiveness"]
                        logger.info(f"  😊 Emoji: {emoji_info['recommendation']}")
                    
                    if "tone_preference" in patterns:
                        tone_info = patterns["tone_preference"]
                        logger.info(f"  💬 Tone: {tone_info['recommendation']}")
                    
                    # Save analysis
                    self._save_analysis_report(analysis)
                else:
                    logger.info(f"⏳ Insufficient data for engagement analysis (need 10+, have {len(replies)})")
                    results["engagement_analysis"] = {"error": "Insufficient data"}
            except Exception as e:
                logger.error(f"❌ Error analyzing engagement: {e}")
                results["engagement_analysis"] = {"error": str(e)}
        
        # 5. Generate trend reports
        if analyze_trends:
            logger.info("📊 Phase 5: Analyzing sentiment trends...")
            try:
                from spider_guardian.scripts.analyze_trends import generate_trend_report
                report_path = generate_trend_report(
                    db_path=self.config.sql_database_path,
                    output_dir="figures/advocacy_trends",
                )
                results["trend_report"] = str(report_path)
                logger.info(f"✅ Saved trend report to {report_path}")
            except Exception as e:
                logger.error(f"❌ Error generating trend report: {e}")
                results["trend_report"] = None
        
        # 6. Train ML models
        if train_ml:
            logger.info("🤖 Phase 6: Training ML models...")
            try:
                from spider_guardian.ml_bot import MLTrainingPipeline
                pipeline = MLTrainingPipeline(self.config)
                ml_results = pipeline.train_all_models(min_replies=50, min_trending=20)
                results["ml_training"] = ml_results
                logger.info("✅ ML models trained successfully")
            except Exception as e:
                logger.error(f"❌ Error training ML models: {e}")
                results["ml_training"] = {"error": str(e)}
        
        logger.info("🎉 Spider Advocacy Orchestrator completed")
        return results
    
    def _fetch_reply_history(self) -> List[Dict]:
        """Fetch posted replies with metrics from database."""
        replies = []
        
        for article in self.sql_store.iter_scraped_articles():
            if article.metadata.get("type") == "interaction":
                content_data = article.content
                if isinstance(content_data, str):
                    import json
                    content_data = json.loads(content_data)
                
                reply_content = content_data.get("reply_text", "")
                metrics = content_data.get("metrics", {})
                
                replies.append({
                    "content": reply_content,
                    "metrics": metrics,
                    "created_at": article.created_at or article.metadata.get("created_at"),
                })
        
        return replies
    
    def _save_analysis_report(self, analysis: Dict):
        """Save engagement analysis to file."""
        report_dir = Path("figures/engagement_analysis")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"engagement_report_{timestamp}.json"
        
        import json
        with open(report_path, "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"💾 Saved engagement report to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Spider Advocacy Orchestrator - Coordinate all advocacy tools"
    )
    parser.add_argument(
        "--respond",
        action="store_true",
        help="Generate and post new replies to tweets",
    )
    parser.add_argument(
        "--refresh-replies",
        action="store_true",
        default=True,
        help="Refresh metrics for posted replies (default: True)",
    )
    parser.add_argument(
        "--update-datasets",
        action="store_true",
        default=True,
        help="Update LangSmith datasets (default: True)",
    )
    parser.add_argument(
        "--analyze-engagement",
        action="store_true",
        default=True,
        help="Analyze engagement patterns (default: True)",
    )
    parser.add_argument(
        "--analyze-trends",
        action="store_true",
        help="Generate sentiment trend reports",
    )
    parser.add_argument(
        "--train-ml",
        action="store_true",
        help="Train machine learning models",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full cycle (all phases enabled)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config JSON file",
    )
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = SpiderGuardianConfig.from_json(args.config)
    else:
        config = SpiderGuardianConfig()
    
    # Initialize orchestrator
    orchestrator = SpiderAdvocacyOrchestrator(config)
    
    # Run cycle
    if args.full:
        results = orchestrator.run_full_cycle(
            respond=True,
            refresh_replies=True,
            update_datasets=True,
            analyze_engagement=True,
            analyze_trends=True,
            train_ml=True,
        )
    else:
        results = orchestrator.run_full_cycle(
            respond=args.respond,
            refresh_replies=args.refresh_replies,
            update_datasets=args.update_datasets,
            analyze_engagement=args.analyze_engagement,
            analyze_trends=args.analyze_trends,
            train_ml=args.train_ml,
        )
    
    # Print summary
    print("\n" + "="*60)
    print("🕷️  SPIDER ADVOCACY SUMMARY")
    print("="*60)
    
    if "new_replies" in results:
        print(f"📢 New replies posted: {results['new_replies']}")
    
    if "refreshed_replies" in results:
        print(f"🔄 Replies refreshed: {results['refreshed_replies']}")
    
    if "dataset_updates" in results:
        print(f"📊 Dataset examples updated: {results['dataset_updates']}")
    
    if "engagement_analysis" in results:
        analysis = results["engagement_analysis"]
        if isinstance(analysis, dict) and "total_replies" in analysis:
            print(f"📈 Engagement analysis: {analysis['total_replies']} replies analyzed")
            avg_score = analysis.get("avg_engagement_score", 0)
            print(f"   Average engagement score: {avg_score:.4f}")
    
    if "trend_report" in results:
        print(f"📊 Trend report: {results['trend_report']}")
    
    if "ml_training" in results:
        ml_results = results["ml_training"]
        if isinstance(ml_results, dict) and not ml_results.get("error"):
            print(f"🤖 ML models trained successfully")
            if ml_results.get("quality_predictor"):
                qp = ml_results["quality_predictor"]
                if qp.get("trained"):
                    print(f"   Quality predictor: ✅ ({qp['samples']} samples)")
            if ml_results.get("rl_stats"):
                rl = ml_results["rl_stats"]
                if rl.get("recommendations"):
                    print(f"   RL recommendations: {len(rl['recommendations'])} insights")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
