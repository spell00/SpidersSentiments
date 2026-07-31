"""Reinforcement learning system to continuously improve reply generation."""

import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from collections import deque
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReplyReinforcementLearner:
    """
    Reinforcement learning agent that learns optimal reply strategies.
    Uses bandit-style learning to explore/exploit different reply styles.
    """
    
    def __init__(self, state_file: str = "models/rl_state.json"):
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Strategy arms (different reply approaches)
        self.strategies = {
            "educational_fact": {
                "description": "Lead with scientific fact",
                "wins": 0,
                "trials": 0,
                "recent_rewards": deque(maxlen=20),
            },
            "friendly_warm": {
                "description": "Warm, friendly tone",
                "wins": 0,
                "trials": 0,
                "recent_rewards": deque(maxlen=20),
            },
            "myth_busting": {
                "description": "Debunk misinformation",
                "wins": 0,
                "trials": 0,
                "recent_rewards": deque(maxlen=20),
            },
            "question_hook": {
                "description": "Ask engaging question",
                "wins": 0,
                "trials": 0,
                "recent_rewards": deque(maxlen=20),
            },
            "personal_story": {
                "description": "Share anecdote/experience",
                "wins": 0,
                "trials": 0,
                "recent_rewards": deque(maxlen=20),
            },
            "emoji_first": {
                "description": "Start with attention emoji",
                "wins": 0,
                "trials": 0,
                "recent_rewards": deque(maxlen=20),
            },
        }
        
        # Hyperparameters
        self.epsilon = 0.2  # Exploration rate (20% random, 80% best)
        self.min_trials = 5  # Min trials before trusting a strategy
        
        self._load_state()
    
    def choose_strategy(self, context: Optional[Dict] = None) -> str:
        """
        Choose reply strategy using epsilon-greedy algorithm.
        
        Args:
            context: Optional context (e.g., tweet sentiment, topic)
        
        Returns:
            Strategy name to use
        """
        # Exploration: Try random strategy
        if random.random() < self.epsilon:
            strategy = random.choice(list(self.strategies.keys()))
            logger.info(f"🎲 Exploring strategy: {strategy}")
            return strategy
        
        # Exploitation: Choose best performing strategy
        best_strategy = self._get_best_strategy()
        logger.info(f"🎯 Exploiting best strategy: {best_strategy}")
        return best_strategy
    
    def record_outcome(
        self,
        strategy: str,
        metrics: Dict,
        original_impressions: int = 0
    ):
        """
        Record outcome of using a strategy and update learning.
        
        Args:
            strategy: Strategy name used
            metrics: Engagement metrics received
            original_impressions: Impressions to normalize against
        """
        if strategy not in self.strategies:
            logger.warning(f"Unknown strategy: {strategy}")
            return
        
        # Calculate reward (normalized engagement rate)
        reward = self._calculate_reward(metrics, original_impressions)
        
        # Update strategy stats
        self.strategies[strategy]["trials"] += 1
        self.strategies[strategy]["recent_rewards"].append(reward)
        
        # Count as "win" if reward above threshold
        if reward > 0.01:  # 1% engagement rate threshold
            self.strategies[strategy]["wins"] += 1
        
        logger.info(
            f"📊 Strategy '{strategy}' reward: {reward:.4f} "
            f"(Win rate: {self._get_win_rate(strategy):.2%})"
        )
        
        # Save updated state
        self._save_state()
    
    def get_strategy_stats(self) -> Dict:
        """Get current statistics for all strategies."""
        stats = {}
        
        for name, data in self.strategies.items():
            win_rate = self._get_win_rate(name)
            avg_reward = self._get_avg_reward(name)
            confidence = min(1.0, data["trials"] / 20.0)  # Confidence based on trials
            
            stats[name] = {
                "description": data["description"],
                "trials": data["trials"],
                "wins": data["wins"],
                "win_rate": win_rate,
                "avg_reward": avg_reward,
                "confidence": confidence,
                "recommended": (
                    win_rate > 0.3 and data["trials"] >= self.min_trials
                ),
            }
        
        return stats
    
    def get_recommendations(self) -> List[str]:
        """Get actionable recommendations based on learning."""
        recommendations = []
        stats = self.get_strategy_stats()
        
        # Find best performing strategies
        reliable_strategies = [
            (name, data)
            for name, data in stats.items()
            if data["trials"] >= self.min_trials
        ]
        
        if reliable_strategies:
            # Sort by win rate
            reliable_strategies.sort(key=lambda x: x[1]["win_rate"], reverse=True)
            
            best = reliable_strategies[0]
            recommendations.append(
                f"✅ '{best[0]}' strategy performing best "
                f"({best[1]['win_rate']:.1%} win rate)"
            )
            
            if len(reliable_strategies) > 1:
                worst = reliable_strategies[-1]
                if worst[1]["win_rate"] < 0.2:
                    recommendations.append(
                        f"⚠️ Consider avoiding '{worst[0]}' strategy "
                        f"({worst[1]['win_rate']:.1%} win rate)"
                    )
        
        # Check for under-explored strategies
        under_explored = [
            name for name, data in self.strategies.items()
            if data["trials"] < self.min_trials
        ]
        
        if under_explored:
            recommendations.append(
                f"🔍 Explore these strategies more: {', '.join(under_explored)}"
            )
        
        return recommendations
    
    def _get_best_strategy(self) -> str:
        """Get the best performing strategy using UCB1 algorithm."""
        total_trials = sum(s["trials"] for s in self.strategies.values())
        
        if total_trials == 0:
            return random.choice(list(self.strategies.keys()))
        
        best_score = -float('inf')
        best_strategy = None
        
        for name, data in self.strategies.items():
            if data["trials"] == 0:
                # Prioritize unexplored strategies
                return name
            
            # UCB1 score: average reward + exploration bonus
            avg_reward = self._get_avg_reward(name)
            exploration_bonus = np.sqrt(2 * np.log(total_trials) / data["trials"])
            ucb_score = avg_reward + exploration_bonus
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_strategy = name
        
        return best_strategy or random.choice(list(self.strategies.keys()))
    
    def _calculate_reward(self, metrics: Dict, impressions: int = 0) -> float:
        """Calculate reward from engagement metrics."""
        likes = metrics.get("like_count", 0)
        replies = metrics.get("reply_count", 0)
        reposts = metrics.get("repost_count", 0)
        actual_impressions = max(metrics.get("impression_count", impressions), 1)
        
        # Weighted engagement (replies worth more than likes)
        weighted_engagement = (replies * 10) + (reposts * 5) + (likes * 1)
        
        # Normalize by impressions
        engagement_rate = weighted_engagement / actual_impressions
        
        return engagement_rate
    
    def _get_win_rate(self, strategy: str) -> float:
        """Get win rate for a strategy."""
        data = self.strategies[strategy]
        if data["trials"] == 0:
            return 0.0
        return data["wins"] / data["trials"]
    
    def _get_avg_reward(self, strategy: str) -> float:
        """Get average reward for a strategy."""
        rewards = self.strategies[strategy]["recent_rewards"]
        if not rewards:
            return 0.0
        return np.mean(rewards)
    
    def _save_state(self):
        """Save learning state to disk."""
        try:
            state = {
                "strategies": {
                    name: {
                        "description": data["description"],
                        "wins": data["wins"],
                        "trials": data["trials"],
                        "recent_rewards": list(data["recent_rewards"]),
                    }
                    for name, data in self.strategies.items()
                },
                "epsilon": self.epsilon,
                "last_updated": datetime.now().isoformat(),
            }
            
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)
            
            logger.debug(f"State saved to {self.state_file}")
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def _load_state(self):
        """Load learning state from disk."""
        try:
            if not self.state_file.exists():
                logger.info("No existing state found, starting fresh")
                return
            
            with open(self.state_file, "r") as f:
                state = json.load(f)
            
            # Restore strategy stats
            for name, data in state.get("strategies", {}).items():
                if name in self.strategies:
                    self.strategies[name]["wins"] = data.get("wins", 0)
                    self.strategies[name]["trials"] = data.get("trials", 0)
                    self.strategies[name]["recent_rewards"] = deque(
                        data.get("recent_rewards", []),
                        maxlen=20
                    )
            
            self.epsilon = state.get("epsilon", self.epsilon)
            
            logger.info(f"State loaded from {self.state_file}")
        except Exception as e:
            logger.error(f"Error loading state: {e}")


class AdaptivePromptGenerator:
    """Generate prompts adapted to learned patterns."""
    
    def __init__(self, rl_learner: ReplyReinforcementLearner):
        self.rl_learner = rl_learner
        
        # Strategy-specific prompt templates
        self.strategy_templates = {
            "educational_fact": """Lead with a fascinating spider fact from research.
                Use phrases like "Actually," "Did you know," or "Fun fact:"
                Include source or credibility marker (📚, 🔬).""",
            
            "friendly_warm": """Use warm, approachable tone with friendly language.
                Include words like "appreciate," "love," "amazing," "help"
                Add friendly emoji (😊, 💚, ❤️).""",
            
            "myth_busting": """Identify and debunk the misconception clearly.
                Use "Actually, that's not true..." or "Common myth, but..."
                Provide the correct information with authority.""",
            
            "question_hook": """Start or end with an engaging question.
                Make it thought-provoking: "Ever wonder why...?"
                Encourage the person to think differently.""",
            
            "personal_story": """Share a brief, relatable anecdote.
                Use conversational tone: "I used to think..." or "Once I learned..."
                Make it personal and authentic.""",
            
            "emoji_first": """Start with an attention-grabbing emoji (🕷️, 💡, 🔬).
                Use emoji strategically throughout for visual breaks.
                Don't overdo it - 2-3 emoji max.""",
        }
    
    def generate_adapted_prompt(
        self,
        base_prompt: str,
        context: Optional[Dict] = None
    ) -> Tuple[str, str]:
        """
        Generate prompt adapted to best-performing strategy.
        
        Returns:
            (adapted_prompt, strategy_used)
        """
        # Choose strategy using RL
        strategy = self.rl_learner.choose_strategy(context)
        
        # Get strategy-specific guidance
        strategy_guidance = self.strategy_templates.get(
            strategy,
            "Use your best judgment based on the context."
        )
        
        # Inject strategy guidance into prompt
        adapted_prompt = base_prompt + (
            f"\n\nSTRATEGY FOCUS (proven to work well):\n{strategy_guidance}\n"
        )
        
        return adapted_prompt, strategy
    
    def generate_ab_test_prompts(self, base_prompt: str) -> List[Tuple[str, str]]:
        """
        Generate multiple prompt variations for A/B testing.
        
        Returns:
            List of (prompt, strategy_name) tuples
        """
        variants = []
        
        # Generate one variant per strategy
        for strategy, guidance in self.strategy_templates.items():
            adapted = base_prompt + (
                f"\n\nSTRATEGY FOCUS:\n{guidance}\n"
            )
            variants.append((adapted, strategy))
        
        return variants


def run_ab_test(
    bot,
    tweet,
    rl_learner: ReplyReinforcementLearner,
    num_variants: int = 3
) -> Dict:
    """
    Run A/B test with multiple reply variants.
    
    This would be used in a testing phase to rapidly learn which strategies work.
    WARNING: Only use this in controlled testing, not production!
    """
    logger.info("🧪 Running A/B test...")
    
    prompt_generator = AdaptivePromptGenerator(rl_learner)
    
    # Generate variant prompts
    base_prompt = bot.build_prompt(tweet.text, [], "neutral")
    variants = prompt_generator.generate_ab_test_prompts(base_prompt)
    
    # Select random subset
    selected_variants = random.sample(variants, min(num_variants, len(variants)))
    
    results = []
    for adapted_prompt, strategy in selected_variants:
        # Generate reply with this strategy
        reply = bot.generate_reply(adapted_prompt, tweet.text)
        
        if reply:
            results.append({
                "strategy": strategy,
                "reply": reply,
                "prompt": adapted_prompt,
            })
    
    return {
        "tweet": tweet.text,
        "variants": results,
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    print("🤖 Reinforcement Learning Demo\n")
    
    # Initialize RL learner
    rl = ReplyReinforcementLearner(state_file="models/rl_demo.json")
    
    # Simulate learning over time
    print("📚 Simulating learning over 100 replies...\n")
    
    for i in range(100):
        # Choose strategy
        strategy = rl.choose_strategy()
        
        # Simulate outcome (biased toward certain strategies for demo)
        if strategy == "educational_fact":
            # This strategy performs well
            impressions = random.randint(500, 2000)
            likes = random.randint(20, 60)
            replies = random.randint(3, 10)
            reposts = random.randint(5, 15)
        elif strategy == "myth_busting":
            # This also works well
            impressions = random.randint(500, 2000)
            likes = random.randint(15, 50)
            replies = random.randint(2, 8)
            reposts = random.randint(3, 12)
        else:
            # Other strategies perform okay
            impressions = random.randint(500, 2000)
            likes = random.randint(5, 25)
            replies = random.randint(0, 5)
            reposts = random.randint(0, 5)
        
        metrics = {
            "like_count": likes,
            "reply_count": replies,
            "repost_count": reposts,
            "impression_count": impressions,
        }
        
        rl.record_outcome(strategy, metrics, impressions)
    
    # Show results
    print("\n" + "="*60)
    print("📊 LEARNING RESULTS AFTER 100 REPLIES")
    print("="*60 + "\n")
    
    stats = rl.get_strategy_stats()
    
    # Sort by win rate
    sorted_strategies = sorted(
        stats.items(),
        key=lambda x: x[1]["win_rate"],
        reverse=True
    )
    
    for name, data in sorted_strategies:
        confidence_bar = "█" * int(data["confidence"] * 10)
        print(f"{name}:")
        print(f"  Win Rate: {data['win_rate']:.1%}")
        print(f"  Avg Reward: {data['avg_reward']:.4f}")
        print(f"  Trials: {data['trials']}")
        print(f"  Confidence: {confidence_bar} ({data['confidence']:.0%})")
        print(f"  Recommended: {'✅ Yes' if data['recommended'] else '❌ No'}")
        print()
    
    # Show recommendations
    print("💡 RECOMMENDATIONS:")
    recommendations = rl.get_recommendations()
    for rec in recommendations:
        print(f"  {rec}")
