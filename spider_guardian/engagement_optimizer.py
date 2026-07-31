"""Learn from top-performing replies to optimize future engagement."""

from typing import List, Dict, Optional, Tuple
import re
from collections import Counter
from datetime import datetime, timedelta
import statistics


class EngagementOptimizer:
    """Analyze what makes replies successful and optimize future ones."""
    
    def __init__(self):
        self.min_sample_size = 10  # Need at least 10 replies to draw patterns
    
    def calculate_engagement_score(self, metrics: Dict) -> float:
        """
        Calculate overall engagement score.
        Weighted to value meaningful interaction over passive likes.
        """
        likes = metrics.get("like_count", 0)
        replies = metrics.get("reply_count", 0)
        reposts = metrics.get("repost_count", 0)
        impressions = metrics.get("impression_count", 1)  # Avoid division by zero
        
        # Weighted engagement: replies > reposts > likes
        weighted_engagement = (replies * 10) + (reposts * 5) + (likes * 1)
        
        # Normalize by impressions (engagement rate)
        engagement_rate = weighted_engagement / impressions if impressions > 0 else 0
        
        return engagement_rate
    
    def analyze_top_performers(self, replies: List[Dict], top_n: int = 10) -> Dict:
        """
        Analyze characteristics of top-performing replies.
        
        Args:
            replies: List of reply dicts with 'content', 'metrics', 'created_at'
            top_n: Number of top replies to analyze
        
        Returns:
            Analysis dict with patterns from successful replies
        """
        if len(replies) < self.min_sample_size:
            return {"error": "Insufficient data", "sample_size": len(replies)}
        
        # Score all replies
        scored_replies = []
        for reply in replies:
            metrics = reply.get("metrics", {})
            score = self.calculate_engagement_score(metrics)
            scored_replies.append({
                "content": reply.get("content", ""),
                "score": score,
                "metrics": metrics,
                "created_at": reply.get("created_at"),
            })
        
        # Sort by score
        scored_replies.sort(key=lambda x: x["score"], reverse=True)
        
        # Analyze top performers
        top_replies = scored_replies[:top_n]
        bottom_replies = scored_replies[-top_n:]
        
        analysis = {
            "total_replies": len(replies),
            "top_performers": top_replies[:3],  # Top 3 for reference
            "avg_engagement_score": statistics.mean(r["score"] for r in scored_replies),
            "median_engagement_score": statistics.median(r["score"] for r in scored_replies),
            "patterns": self._extract_patterns(top_replies, bottom_replies),
        }
        
        return analysis
    
    def _extract_patterns(self, top_replies: List[Dict], bottom_replies: List[Dict]) -> Dict:
        """Extract patterns from top vs bottom performers."""
        
        def extract_features(replies: List[Dict]) -> Dict:
            """Extract features from a set of replies."""
            lengths = [len(r["content"]) for r in replies]
            has_emoji = sum(1 for r in replies if self._contains_emoji(r["content"]))
            has_question = sum(1 for r in replies if "?" in r["content"])
            has_link = sum(1 for r in replies if "http" in r["content"].lower())
            has_facts = sum(1 for r in replies if self._contains_fact_indicators(r["content"]))
            starts_with_emoji = sum(1 for r in replies if self._starts_with_emoji(r["content"]))
            
            # Tone analysis
            friendly_tone = sum(1 for r in replies if self._is_friendly(r["content"]))
            educational_tone = sum(1 for r in replies if self._is_educational(r["content"]))
            
            return {
                "avg_length": statistics.mean(lengths) if lengths else 0,
                "emoji_rate": has_emoji / len(replies) if replies else 0,
                "question_rate": has_question / len(replies) if replies else 0,
                "link_rate": has_link / len(replies) if replies else 0,
                "fact_rate": has_facts / len(replies) if replies else 0,
                "starts_emoji_rate": starts_with_emoji / len(replies) if replies else 0,
                "friendly_rate": friendly_tone / len(replies) if replies else 0,
                "educational_rate": educational_tone / len(replies) if replies else 0,
            }
        
        top_features = extract_features(top_replies)
        bottom_features = extract_features(bottom_replies)
        
        # Compare features
        patterns = {
            "optimal_length": {
                "top_avg": round(top_features["avg_length"]),
                "bottom_avg": round(bottom_features["avg_length"]),
                "recommendation": self._recommend_length(top_features["avg_length"]),
            },
            "emoji_effectiveness": {
                "top_rate": round(top_features["emoji_rate"] * 100, 1),
                "bottom_rate": round(bottom_features["emoji_rate"] * 100, 1),
                "recommendation": "Use emojis" if top_features["emoji_rate"] > bottom_features["emoji_rate"] else "Emojis neutral",
            },
            "starting_emoji": {
                "top_rate": round(top_features["starts_emoji_rate"] * 100, 1),
                "bottom_rate": round(bottom_features["starts_emoji_rate"] * 100, 1),
                "recommendation": "Start with emoji" if top_features["starts_emoji_rate"] > bottom_features["starts_emoji_rate"] else "Starting emoji neutral",
            },
            "questions": {
                "top_rate": round(top_features["question_rate"] * 100, 1),
                "bottom_rate": round(bottom_features["question_rate"] * 100, 1),
                "recommendation": "Ask questions" if top_features["question_rate"] > bottom_features["question_rate"] else "Questions neutral",
            },
            "facts": {
                "top_rate": round(top_features["fact_rate"] * 100, 1),
                "bottom_rate": round(bottom_features["fact_rate"] * 100, 1),
                "recommendation": "Include facts" if top_features["fact_rate"] > bottom_features["fact_rate"] else "Facts neutral",
            },
            "tone_preference": {
                "friendly_top": round(top_features["friendly_rate"] * 100, 1),
                "educational_top": round(top_features["educational_rate"] * 100, 1),
                "recommendation": self._recommend_tone(top_features),
            },
        }
        
        return patterns
    
    def _contains_emoji(self, text: str) -> bool:
        """Check if text contains emoji."""
        # Common spider/nature emojis
        emoji_pattern = r'[🕷🕸🦗🐛🐜🦋🌿🌱🔬📚💚🌍✅❤️💡😊]'
        return bool(re.search(emoji_pattern, text))
    
    def _starts_with_emoji(self, text: str) -> bool:
        """Check if text starts with an emoji."""
        if not text:
            return False
        emoji_pattern = r'^[🕷🕸🦗🐛🐜🦋🌿🌱🔬📚💚🌍✅❤️💡😊]'
        return bool(re.search(emoji_pattern, text))
    
    def _contains_fact_indicators(self, text: str) -> bool:
        """Check if text contains fact indicators."""
        indicators = ["actually", "fact", "research", "study", "science", "according to", "source:", "🔬", "📚"]
        return any(indicator in text.lower() for indicator in indicators)
    
    def _is_friendly(self, text: str) -> bool:
        """Check if tone is friendly."""
        friendly_words = ["thank", "appreciate", "love", "amazing", "wonderful", "great", "help", "😊", "❤️", "💚"]
        return any(word in text.lower() for word in friendly_words)
    
    def _is_educational(self, text: str) -> bool:
        """Check if tone is educational."""
        edu_words = ["actually", "did you know", "fun fact", "fact", "research shows", "studies", "🔬", "📚"]
        return any(word in text.lower() for word in edu_words)
    
    def _recommend_length(self, avg_length: float) -> str:
        """Recommend optimal length."""
        if avg_length < 100:
            return "Keep it concise (under 100 chars)"
        elif avg_length < 200:
            return "Medium length (100-200 chars) works well"
        else:
            return "Detailed responses (200+ chars) engage readers"
    
    def _recommend_tone(self, features: Dict) -> str:
        """Recommend tone based on success rates."""
        if features["friendly_rate"] > features["educational_rate"]:
            return "Friendly, warm tone performs best"
        elif features["educational_rate"] > features["friendly_rate"]:
            return "Educational, informative tone performs best"
        else:
            return "Mix friendly and educational tones"
    
    def generate_improvement_suggestions(self, draft_reply: str, patterns: Dict) -> List[str]:
        """
        Generate specific suggestions to improve a draft reply.
        
        Args:
            draft_reply: The draft reply text
            patterns: Patterns from analyze_top_performers
        
        Returns:
            List of actionable suggestions
        """
        suggestions = []
        
        # Check length
        draft_len = len(draft_reply)
        optimal = patterns.get("optimal_length", {})
        if optimal.get("top_avg"):
            top_avg = optimal["top_avg"]
            if abs(draft_len - top_avg) > 50:
                suggestions.append(f"Consider adjusting length closer to {top_avg} chars (currently {draft_len})")
        
        # Check emoji
        emoji_rec = patterns.get("emoji_effectiveness", {}).get("recommendation", "")
        if "use emojis" in emoji_rec.lower() and not self._contains_emoji(draft_reply):
            suggestions.append("Add relevant emoji (🕷️, 📚, 💡) to increase engagement")
        
        # Check starting emoji
        starts_emoji_rec = patterns.get("starting_emoji", {}).get("recommendation", "")
        if "start with emoji" in starts_emoji_rec.lower() and not self._starts_with_emoji(draft_reply):
            suggestions.append("Consider starting with an attention-grabbing emoji")
        
        # Check facts
        fact_rec = patterns.get("facts", {}).get("recommendation", "")
        if "include facts" in fact_rec.lower() and not self._contains_fact_indicators(draft_reply):
            suggestions.append("Add a fact or source to boost credibility")
        
        # Check questions
        question_rec = patterns.get("questions", {}).get("recommendation", "")
        if "ask questions" in question_rec.lower() and "?" not in draft_reply:
            suggestions.append("Consider adding a question to encourage replies")
        
        # Check tone
        tone_rec = patterns.get("tone_preference", {}).get("recommendation", "")
        if "friendly" in tone_rec.lower() and not self._is_friendly(draft_reply):
            suggestions.append("Add friendly language to warm up the tone")
        elif "educational" in tone_rec.lower() and not self._is_educational(draft_reply):
            suggestions.append("Emphasize educational content for better engagement")
        
        return suggestions if suggestions else ["Draft looks good based on current patterns!"]
    
    def optimize_reply(self, draft_reply: str, patterns: Dict) -> str:
        """
        Auto-optimize a draft reply based on learned patterns.
        Returns improved version.
        """
        optimized = draft_reply
        
        # Add starting emoji if pattern suggests it
        starts_emoji_rec = patterns.get("starting_emoji", {}).get("recommendation", "")
        if "start with emoji" in starts_emoji_rec.lower() and not self._starts_with_emoji(optimized):
            optimized = "🕷️ " + optimized
        
        # Ensure emoji presence if effective
        emoji_rec = patterns.get("emoji_effectiveness", {}).get("recommendation", "")
        if "use emojis" in emoji_rec.lower() and not self._contains_emoji(optimized):
            # Add contextual emoji at end
            if "fact" in optimized.lower() or "research" in optimized.lower():
                optimized += " 📚"
            elif "help" in optimized.lower() or "protect" in optimized.lower():
                optimized += " 💚"
        
        return optimized


if __name__ == "__main__":
    # Demo with sample data
    optimizer = EngagementOptimizer()
    
    # Simulate replies with metrics
    sample_replies = [
        {
            "content": "🕷️ Actually, spiders eat 400-800M tons of insects yearly! They're nature's pest control 📚",
            "metrics": {"like_count": 45, "reply_count": 8, "repost_count": 12, "impression_count": 2000},
            "created_at": "2024-01-15",
        },
        {
            "content": "Don't kill house spiders! They eat mosquitoes and flies. Let them be your roommates 😊",
            "metrics": {"like_count": 38, "reply_count": 5, "repost_count": 7, "impression_count": 1500},
            "created_at": "2024-01-14",
        },
        {
            "content": "Spiders are beneficial and should not be killed.",
            "metrics": {"like_count": 2, "reply_count": 0, "repost_count": 0, "impression_count": 500},
            "created_at": "2024-01-13",
        },
        # Add more samples...
    ]
    
    # Need 10+ for real analysis
    sample_replies.extend([sample_replies[2]] * 8)  # Pad for demo
    
    analysis = optimizer.analyze_top_performers(sample_replies)
    
    print("📊 Engagement Analysis\n")
    print(f"Total replies analyzed: {analysis['total_replies']}")
    print(f"Average engagement score: {analysis['avg_engagement_score']:.4f}\n")
    
    print("📈 Top Performing Reply:")
    if analysis.get("top_performers"):
        top = analysis["top_performers"][0]
        print(f"  Content: {top['content']}")
        print(f"  Score: {top['score']:.4f}")
        print(f"  Metrics: {top['metrics']}\n")
    
    print("🎯 Patterns Found:")
    patterns = analysis.get("patterns", {})
    for key, value in patterns.items():
        print(f"\n{key}:")
        for k, v in value.items():
            print(f"  {k}: {v}")
    
    # Test optimization
    print("\n\n🔧 Reply Optimization Demo")
    draft = "Spiders are helpful creatures that control pests."
    suggestions = optimizer.generate_improvement_suggestions(draft, patterns)
    
    print(f"\nOriginal: {draft}")
    print("\nSuggestions:")
    for s in suggestions:
        print(f"  • {s}")
    
    optimized = optimizer.optimize_reply(draft, patterns)
    print(f"\nOptimized: {optimized}")
