"""ML-enhanced bot integration and training pipeline."""

import logging
from pathlib import Path
from typing import Any, Optional, List, Dict, Tuple
from datetime import datetime

from spider_guardian.config import SpiderGuardianConfig
from spider_guardian.storage.sql import SQLDataStore
from spider_guardian.ml_trainer import ReplyQualityPredictor, PopularityAnalyzer
from spider_guardian.rl_learner import ReplyReinforcementLearner, AdaptivePromptGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLEnhancedBot:
    """Spider Guardian Bot with machine learning enhancements."""
    
    def __init__(self, config: SpiderGuardianConfig, enable_ml: bool = True):
        from spider_guardian.bot import SpiderGuardianBot

        self._bot = SpiderGuardianBot(config)
        self.config = self._bot.config
        
        self.enable_ml = enable_ml
        
        if enable_ml:
            # Initialize ML components
            self.quality_predictor = ReplyQualityPredictor()
            self.rl_learner = ReplyReinforcementLearner()
            self.prompt_generator = AdaptivePromptGenerator(self.rl_learner)
            self.popularity_analyzer = PopularityAnalyzer()
            
            logger.info("🤖 ML enhancements enabled")
        else:
            logger.info("Standard bot mode (ML disabled)")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._bot, name)
    
    def generate_reply(self, prompt: str, original_tweet: str = "") -> Optional[str]:
        """Generate reply with ML optimization."""
        
        if not self.enable_ml:
            return self._bot.generate_reply(prompt, original_tweet)
        
        # Use RL to adapt prompt for best strategy
        adapted_prompt, strategy = self.prompt_generator.generate_adapted_prompt(
            prompt,
            context={"tweet": original_tweet}
        )
        
        logger.info(f"🎯 Using strategy: {strategy}")
        
        # Generate multiple candidates
        candidates = []
        for attempt in range(3):  # Generate 3 candidates
            candidate = self._bot.generate_reply(adapted_prompt, original_tweet)
            if candidate:
                candidates.append(candidate)
        
        if not candidates:
            logger.warning("No candidates generated")
            return None
        
        # Use ML predictor to rank candidates
        ranked = self.quality_predictor.rank_candidates(candidates, original_tweet)
        
        if ranked:
            best_reply, predicted_score = ranked[0]
            logger.info(
                f"✅ Selected reply with predicted score: {predicted_score:.4f}"
            )
            
            # Store strategy used for later feedback (unconditionally)
            self._current_strategy = strategy
            
            return best_reply
        
        return candidates[0]
    
    def record_reply_outcome(
        self,
        reply_text: str,
        metrics: Dict,
        original_tweet: str = ""
    ):
        """Record outcome for RL learning."""
        if not self.enable_ml:
            return
        
        # Get strategy used (stored during generation)
        strategy = getattr(self, '_current_strategy', 'unknown')
        
        # Record for RL
        self.rl_learner.record_outcome(
            strategy=strategy,
            metrics=metrics,
            original_impressions=metrics.get("impression_count", 0)
        )
        
        logger.info(
            f"📊 Recorded outcome for strategy '{strategy}': "
            f"{metrics.get('like_count', 0)} likes, "
            f"{metrics.get('reply_count', 0)} replies"
        )


class MLTrainingPipeline:
    """Pipeline for training and updating ML models."""
    
    def __init__(self, config: SpiderGuardianConfig):
        self.config = config
        self.sql_store = SQLDataStore(config.sql_database_path)
        self.quality_predictor = ReplyQualityPredictor()
        self.popularity_analyzer = PopularityAnalyzer()
        self.rl_learner = ReplyReinforcementLearner()
    
    def train_all_models(
        self,
        min_replies: int = 50,
        min_trending: int = 20
    ) -> Dict:
        """
        Train all ML models on historical data.
        
        Returns:
            Training results and metrics
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "quality_predictor": None,
            "popularity_analysis": None,
            "rl_stats": None,
        }
        
        logger.info("🚀 Starting ML training pipeline...")
        
        # 1. Train quality predictor on reply history
        logger.info("\n📈 Phase 1: Training quality predictor...")
        replies = self._fetch_reply_history()
        
        if len(replies) >= min_replies:
            success = self.quality_predictor.train(replies, min_samples=min_replies)
            results["quality_predictor"] = {
                "trained": success,
                "samples": len(replies),
            }
            logger.info(f"✅ Quality predictor trained on {len(replies)} replies")
        else:
            logger.warning(
                f"⚠️ Insufficient replies for training: {len(replies)} < {min_replies}"
            )
            results["quality_predictor"] = {
                "trained": False,
                "reason": "insufficient_data",
                "samples": len(replies),
            }
        
        # 2. Analyze trending content for popularity patterns
        logger.info("\n🔥 Phase 2: Analyzing trending content...")
        trending = self._fetch_trending_posts()
        
        if len(trending) >= min_trending:
            analysis = self.popularity_analyzer.analyze_trending_content(trending)
            recommendations = self.popularity_analyzer.generate_recommendations(analysis)
            
            results["popularity_analysis"] = {
                "analyzed": True,
                "samples": len(trending),
                "top_topics": analysis["top_topics"][:3],
                "top_angles": analysis["top_angles"][:3],
                "recommendations": recommendations,
            }
            
            logger.info(f"✅ Analyzed {len(trending)} trending posts")
            logger.info("📊 Top recommendations:")
            for rec in recommendations:
                logger.info(f"  • {rec}")
        else:
            logger.warning(
                f"⚠️ Insufficient trending data: {len(trending)} < {min_trending}"
            )
            results["popularity_analysis"] = {
                "analyzed": False,
                "reason": "insufficient_data",
                "samples": len(trending),
            }
        
        # 3. Update RL learner with recent outcomes
        logger.info("\n🎯 Phase 3: Updating RL learner...")
        self._update_rl_learner(replies)
        
        stats = self.rl_learner.get_strategy_stats()
        recommendations = self.rl_learner.get_recommendations()
        
        results["rl_stats"] = {
            "strategies": {
                name: {
                    "win_rate": data["win_rate"],
                    "trials": data["trials"],
                    "recommended": data["recommended"],
                }
                for name, data in stats.items()
            },
            "recommendations": recommendations,
        }
        
        logger.info("✅ RL learner updated")
        logger.info("💡 Strategy recommendations:")
        for rec in recommendations:
            logger.info(f"  • {rec}")
        
        # Save results
        self._save_training_results(results)
        
        logger.info("\n🎉 ML training pipeline complete!")
        return results
    
    def _fetch_reply_history(self) -> List[Dict]:
        """Fetch posted replies with metrics."""
        replies = []
        
        for article in self.sql_store.iter_scraped_articles():
            if article.metadata.get("type") == "interaction":
                content_data = article.content
                if isinstance(content_data, str):
                    import json
                    content_data = json.loads(content_data)
                
                replies.append({
                    "content": content_data.get("reply_text", ""),
                    "metrics": content_data.get("metrics", {}),
                    "original_tweet": content_data.get("original_text", ""),
                    "created_at": article.created_at,
                })
        
        return replies
    
    def _fetch_trending_posts(self) -> List[Dict]:
        """Fetch trending spider posts."""
        from spider_guardian.storage.trending import TrendingStore
        
        trending_store = TrendingStore()
        posts = []
        
        try:
            for post in trending_store.top(limit=100, since_hours=24*7):  # Last week
                posts.append({
                    "text": post.text,
                    "metrics": {
                        "like_count": post.like_count,
                        "reply_count": post.reply_count,
                        "repost_count": post.repost_count,
                        "impression_count": post.impression_count,
                    },
                    "created_at": post.post_created_at,
                })
        except Exception as e:
            logger.error(f"Error fetching trending posts: {e}")
        
        return posts
    
    def _update_rl_learner(self, replies: List[Dict]):
        """Update RL learner with historical outcomes."""
        # This would ideally extract strategy from metadata
        # For now, we infer strategy from reply characteristics
        
        for reply in replies[-50:]:  # Use most recent 50
            strategy = self._infer_strategy(reply["content"])
            self.rl_learner.record_outcome(
                strategy=strategy,
                metrics=reply["metrics"],
                original_impressions=reply["metrics"].get("impression_count", 0)
            )
    
    def _infer_strategy(self, reply_text: str) -> str:
        """Infer which strategy was likely used for a reply."""
        text_lower = reply_text.lower()
        
        # Simple heuristics to infer strategy
        if any(word in text_lower for word in ["actually", "fact", "research", "study"]):
            return "educational_fact"
        elif any(word in text_lower for word in ["myth", "false", "not true", "actually no"]):
            return "myth_busting"
        elif "?" in reply_text and reply_text.index("?") < len(reply_text) // 2:
            return "question_hook"
        elif reply_text and reply_text[0] in "🕷🕸💡🔬📚":
            return "emoji_first"
        elif any(word in text_lower for word in ["love", "appreciate", "thank", "awesome"]):
            return "friendly_warm"
        else:
            return "personal_story"
    
    def _save_training_results(self, results: Dict):
        """Save training results to file."""
        output_dir = Path("models/training_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"training_results_{timestamp}.json"
        
        import json
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"💾 Training results saved to {output_file}")


def main():
    """CLI entrypoint for ML-enhanced bot: train models or run the bot."""
    import argparse

    parser = argparse.ArgumentParser(
        description="ML-enhanced Spider Guardian bot: train models or run replies"
    )
    subparsers = parser.add_subparsers(dest="command", required=False)

    # Train subcommand (default)
    train_parser = subparsers.add_parser("train", help="Train ML models (default)")
    train_parser.add_argument(
        "--min-replies",
        type=int,
        default=50,
        help="Minimum replies needed for training (default: 50)"
    )
    train_parser.add_argument(
        "--min-trending",
        type=int,
        default=20,
        help="Minimum trending posts for analysis (default: 20)"
    )
    train_parser.add_argument(
        "--config",
        type=str,
        help="Path to config file"
    )

    # Run subcommand
    run_parser = subparsers.add_parser("run", help="Run the bot to post ML-enhanced replies")
    run_parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of replies to send before exiting (default: 1). Use 0 for one pass, -1 to loop."
    )
    run_parser.add_argument(
        "--reply-to-replies",
        action="store_true",
        help="Also reply to replies (not just original tweets)"
    )
    run_parser.add_argument(
        "--test-one-word",
        action="store_true",
        help="Send a one-word test reply instead of generating"
    )
    run_parser.add_argument(
        "--ml-disable",
        action="store_true",
        help="Disable ML ranking/strategies and use the base bot generation"
    )
    run_parser.add_argument(
        "--config",
        type=str,
        help="Path to config file"
    )

    args = parser.parse_args()

    # Default to train if no subcommand is provided (backward compatible)
    command = args.command or "train"

    # Load config
    cfg_path = getattr(args, "config", None)
    if cfg_path:
        config = SpiderGuardianConfig.from_json(cfg_path)
    else:
        config = SpiderGuardianConfig()

    if command == "run":
        # Instantiate ML-enhanced bot and respond
        bot = MLEnhancedBot(config, enable_ml=not getattr(args, "ml_disable", False))
        logger.info("Starting ML-enhanced bot run (limit=%s)…", getattr(args, "limit", 1))
        bot.respond_to_tweets(
            limit=getattr(args, "limit", 1),
            test_one_word_reply=getattr(args, "test_one_word", False),
            reply_to_replies=getattr(args, "reply_to_replies", False),
        )
        logger.info("Bot run complete.")
        return

    # Train flow
    pipeline = MLTrainingPipeline(config)
    results = pipeline.train_all_models(
        min_replies=getattr(args, "min_replies", 50),
        min_trending=getattr(args, "min_trending", 20)
    )

    # Print summary
    print("\n" + "=" * 60)
    print("🤖 ML TRAINING SUMMARY")
    print("=" * 60)

    if results["quality_predictor"]:
        qp = results["quality_predictor"]
        print(f"\n📈 Quality Predictor:")
        print(f"  Trained: {'✅ Yes' if qp['trained'] else '❌ No'}")
        print(f"  Samples: {qp['samples']}")

    if results["popularity_analysis"]:
        pa = results["popularity_analysis"]
        print(f"\n🔥 Popularity Analysis:")
        print(f"  Analyzed: {'✅ Yes' if pa['analyzed'] else '❌ No'}")
        print(f"  Samples: {pa['samples']}")

        if pa.get("recommendations"):
            print(f"  Top recommendations:")
            for rec in pa["recommendations"][:3]:
                print(f"    • {rec}")

    if results["rl_stats"]:
        rl = results["rl_stats"]
        print(f"\n🎯 Reinforcement Learning:")

        # Show top strategies
        sorted_strategies = sorted(
            rl["strategies"].items(),
            key=lambda x: x[1]["win_rate"],
            reverse=True
        )

        print(f"  Top strategies:")
        for name, data in sorted_strategies[:3]:
            if data["trials"] > 0:
                print(f"    • {name}: {data['win_rate']:.1%} win rate ({data['trials']} trials)")

        if rl.get("recommendations"):
            print(f"  Recommendations:")
            for rec in rl["recommendations"]:
                print(f"    • {rec}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
