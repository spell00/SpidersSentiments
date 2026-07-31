"""Machine learning trainer for optimizing spider advocacy replies."""

import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReplyQualityPredictor:
    """Predict engagement score for a reply before posting it."""
    
    def __init__(self, model_dir: str = "models/reply_quality"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.engagement_model = None
        self.tfidf_vectorizer = None
        self.scaler = None
        self.feature_extractors = ReplyFeatureExtractor()
        
        self._load_models()
    
    def _load_models(self):
        """Load trained models if they exist."""
        model_path = self.model_dir / "engagement_predictor.pkl"
        tfidf_path = self.model_dir / "tfidf_vectorizer.pkl"
        scaler_path = self.model_dir / "scaler.pkl"
        
        try:
            if model_path.exists():
                self.engagement_model = joblib.load(model_path)
                logger.info("Loaded engagement predictor model")
            
            if tfidf_path.exists():
                self.tfidf_vectorizer = joblib.load(tfidf_path)
                logger.info("Loaded TF-IDF vectorizer")
            
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("Loaded feature scaler")
        except Exception as e:
            logger.warning(f"Error loading models: {e}")
    
    def train(self, replies: List[Dict], min_samples: int = 50):
        """
        Train engagement predictor on historical reply data.
        
        Args:
            replies: List of dicts with 'content', 'metrics', 'original_tweet'
            min_samples: Minimum samples needed for training
        """
        if len(replies) < min_samples:
            logger.warning(f"Insufficient data for training: {len(replies)} < {min_samples}")
            return False
        
        logger.info(f"Training engagement predictor on {len(replies)} replies...")
        
        # Extract features and labels
        X = []
        y = []
        
        for reply_data in replies:
            try:
                features = self._extract_features(
                    reply_data.get("content", ""),
                    reply_data.get("original_tweet", "")
                )
                
                # Calculate engagement score as target
                metrics = reply_data.get("metrics", {})
                engagement = self._calculate_engagement_score(metrics)
                
                # Only train on replies with some engagement
                if engagement > 0 or metrics.get("impression_count", 0) > 0:
                    X.append(features)
                    y.append(engagement)
            except Exception as e:
                logger.warning(f"Error extracting features: {e}")
                continue
        
        if len(X) < min_samples:
            logger.warning(f"Insufficient valid samples: {len(X)} < {min_samples}")
            return False
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model (Gradient Boosting for engagement prediction)
        self.engagement_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
        )
        self.engagement_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.engagement_model.score(X_train_scaled, y_train)
        test_score = self.engagement_model.score(X_test_scaled, y_test)
        
        logger.info(f"Model training complete!")
        logger.info(f"  Train R²: {train_score:.3f}")
        logger.info(f"  Test R²: {test_score:.3f}")
        
        # Train TF-IDF on reply texts for semantic features
        reply_texts = [r.get("content", "") for r in replies if r.get("content")]
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),
            stop_words='english',
        )
        self.tfidf_vectorizer.fit(reply_texts)
        
        # Save models
        self._save_models()
        
        return True
    
    def predict_engagement(self, reply_text: str, original_tweet: str = "") -> float:
        """
        Predict engagement score for a draft reply.
        
        Returns:
            Predicted engagement score (higher = better)
        """
        if self.engagement_model is None:
            logger.warning("No trained model available")
            return 0.0
        
        try:
            features = self._extract_features(reply_text, original_tweet)
            features_scaled = self.scaler.transform([features])
            prediction = self.engagement_model.predict(features_scaled)[0]
            return max(0.0, prediction)  # Ensure non-negative
        except Exception as e:
            logger.error(f"Error predicting engagement: {e}")
            return 0.0
    
    def rank_candidates(
        self,
        candidates: List[str],
        original_tweet: str = ""
    ) -> List[Tuple[str, float]]:
        """
        Rank multiple reply candidates by predicted engagement.
        
        Returns:
            List of (reply_text, predicted_score) sorted by score descending
        """
        ranked = []
        for candidate in candidates:
            score = self.predict_engagement(candidate, original_tweet)
            ranked.append((candidate, score))
        
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
    
    def _extract_features(self, reply_text: str, original_tweet: str = "") -> np.ndarray:
        """Extract feature vector from reply and context."""
        features = []
        
        # Basic text features
        basic_features = self.feature_extractors.extract_basic_features(reply_text)
        features.extend(basic_features)
        
        # Sentiment features
        sentiment_features = self.feature_extractors.extract_sentiment_features(reply_text)
        features.extend(sentiment_features)
        
        # Style features
        style_features = self.feature_extractors.extract_style_features(reply_text)
        features.extend(style_features)
        
        # Context features (if original tweet provided)
        if original_tweet:
            context_features = self.feature_extractors.extract_context_features(
                reply_text, original_tweet
            )
            features.extend(context_features)
        else:
            features.extend([0.0] * 5)  # Padding
        
        return np.array(features)
    
    def _calculate_engagement_score(self, metrics: Dict) -> float:
        """Calculate engagement score from metrics."""
        likes = metrics.get("like_count", 0)
        replies = metrics.get("reply_count", 0)
        reposts = metrics.get("repost_count", 0)
        impressions = max(metrics.get("impression_count", 1), 1)
        
        # Weighted engagement rate
        weighted = (replies * 10) + (reposts * 5) + (likes * 1)
        engagement_rate = weighted / impressions
        
        return engagement_rate
    
    def _save_models(self):
        """Save trained models to disk."""
        try:
            joblib.dump(
                self.engagement_model,
                self.model_dir / "engagement_predictor.pkl"
            )
            joblib.dump(
                self.tfidf_vectorizer,
                self.model_dir / "tfidf_vectorizer.pkl"
            )
            joblib.dump(
                self.scaler,
                self.model_dir / "scaler.pkl"
            )
            logger.info(f"Models saved to {self.model_dir}")
        except Exception as e:
            logger.error(f"Error saving models: {e}")


class ReplyFeatureExtractor:
    """Extract features from reply text for ML models."""
    
    def extract_basic_features(self, text: str) -> List[float]:
        """Extract basic text statistics."""
        words = text.split()
        chars = len(text)
        
        return [
            len(words),                          # Word count
            chars,                               # Character count
            chars / max(len(words), 1),          # Avg word length
            sum(1 for c in text if c.isupper()) / max(chars, 1),  # Uppercase ratio
            text.count('!'),                     # Exclamation marks
            text.count('?'),                     # Questions
            text.count('.'),                     # Periods
        ]
    
    def extract_sentiment_features(self, text: str) -> List[float]:
        """Extract sentiment-related features."""
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        
        return [
            scores['pos'],
            scores['neg'],
            scores['neu'],
            scores['compound'],
        ]
    
    def extract_style_features(self, text: str) -> List[float]:
        """Extract stylistic features."""
        import re
        
        # Emoji detection
        emoji_pattern = r'[🕷🕸🦗🐛🐜🦋🌿🌱🔬📚💚🌍✅❤️💡😊👍🎉]'
        emoji_count = len(re.findall(emoji_pattern, text))
        
        # Check for facts/sources
        has_fact = any(word in text.lower() for word in [
            'actually', 'fact', 'research', 'study', 'according'
        ])
        
        # Check for engagement hooks
        has_question = '?' in text
        has_call_to_action = any(word in text.lower() for word in [
            'check', 'look', 'learn', 'imagine', 'think'
        ])
        
        # Friendly words
        friendly_words = ['thank', 'love', 'amazing', 'great', 'awesome', 'help']
        friendly_count = sum(1 for word in friendly_words if word in text.lower())
        
        # Spider-related terms
        spider_terms = ['spider', 'web', 'silk', 'arachnid', 'pest control']
        spider_count = sum(1 for term in spider_terms if term in text.lower())
        
        return [
            emoji_count,
            float(has_fact),
            float(has_question),
            float(has_call_to_action),
            friendly_count,
            spider_count,
        ]
    
    def extract_context_features(self, reply: str, original_tweet: str) -> List[float]:
        """Extract features about reply-to-tweet relationship."""
        reply_lower = reply.lower()
        tweet_lower = original_tweet.lower()
        
        # Word overlap
        reply_words = set(reply_lower.split())
        tweet_words = set(tweet_lower.split())
        overlap = len(reply_words & tweet_words) / max(len(reply_words), 1)
        
        # Length ratio
        length_ratio = len(reply) / max(len(original_tweet), 1)
        
        # Tone match (both positive/negative)
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        
        reply_sentiment = analyzer.polarity_scores(reply)['compound']
        tweet_sentiment = analyzer.polarity_scores(original_tweet)['compound']
        sentiment_diff = abs(reply_sentiment - tweet_sentiment)
        
        # Counter-response (tweet negative, reply positive)
        is_counter = (tweet_sentiment < -0.3 and reply_sentiment > 0.3)
        
        # Addressing the user
        has_direct_address = any(word in reply_lower for word in ['you', 'your'])
        
        return [
            overlap,
            length_ratio,
            sentiment_diff,
            float(is_counter),
            float(has_direct_address),
        ]


class PopularityAnalyzer:
    """Analyze what's trending and popular in spider discussions."""
    
    def __init__(self):
        self.popular_topics = defaultdict(int)
        self.popular_angles = defaultdict(int)
        self.popular_emojis = defaultdict(int)
        self.popular_facts = defaultdict(int)
    
    def analyze_trending_content(self, posts: List[Dict]) -> Dict:
        """
        Analyze trending posts to understand what's popular.
        
        Args:
            posts: List of trending spider posts with engagement metrics
        
        Returns:
            Analysis dict with popular topics, angles, etc.
        """
        # Reset counters
        self.popular_topics.clear()
        self.popular_angles.clear()
        self.popular_emojis.clear()
        self.popular_facts.clear()
        
        # Analyze each post
        for post in posts:
            text = post.get("text", "").lower()
            metrics = post.get("metrics", {})
            engagement = (
                metrics.get("like_count", 0) +
                metrics.get("reply_count", 0) * 3 +
                metrics.get("repost_count", 0) * 2
            )
            
            # Extract topics
            topics = self._extract_topics(text)
            for topic in topics:
                self.popular_topics[topic] += engagement
            
            # Extract angles (fear, education, humor, etc.)
            angle = self._classify_angle(text)
            self.popular_angles[angle] += engagement
            
            # Extract emojis
            emojis = self._extract_emojis(text)
            for emoji in emojis:
                self.popular_emojis[emoji] += engagement
            
            # Check if contains facts
            if self._contains_facts(text):
                self.popular_facts["uses_facts"] += engagement
        
        # Return sorted results
        return {
            "top_topics": sorted(
                self.popular_topics.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            "top_angles": sorted(
                self.popular_angles.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "top_emojis": sorted(
                self.popular_emojis.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            "fact_effectiveness": self.popular_facts.get("uses_facts", 0),
        }
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract spider-related topics from text."""
        topics = []
        
        topic_keywords = {
            "pest_control": ["pest", "insect", "mosquito", "fly", "eat"],
            "fear": ["scary", "afraid", "phobia", "scared", "terrified"],
            "venom": ["bite", "poison", "venomous", "dangerous"],
            "web": ["web", "silk", "thread"],
            "home": ["house", "home", "indoor", "bedroom"],
            "garden": ["garden", "outdoor", "yard", "plant"],
            "identification": ["species", "type", "kind", "identify"],
            "conservation": ["protect", "save", "ecosystem", "biodiversity"],
        }
        
        for topic, keywords in topic_keywords.items():
            if any(kw in text for kw in keywords):
                topics.append(topic)
        
        return topics
    
    def _classify_angle(self, text: str) -> str:
        """Classify the communication angle."""
        if any(word in text for word in ["actually", "fact", "research", "study"]):
            return "educational"
        elif any(word in text for word in ["scary", "afraid", "kill", "hate"]):
            return "fear_based"
        elif any(word in text for word in ["lol", "lmao", "funny", "😂"]):
            return "humorous"
        elif any(word in text for word in ["help", "useful", "beneficial"]):
            return "practical"
        elif any(word in text for word in ["cute", "amazing", "beautiful", "love"]):
            return "appreciation"
        else:
            return "neutral"
    
    def _extract_emojis(self, text: str) -> List[str]:
        """Extract emojis from text."""
        import re
        emoji_pattern = r'[🕷🕸🦗🐛🐜🦋🌿🌱🔬📚💚🌍✅❤️💡😊👍🎉😂🤔]'
        return re.findall(emoji_pattern, text)
    
    def _contains_facts(self, text: str) -> bool:
        """Check if text contains factual content."""
        fact_indicators = [
            'actually', 'fact', 'research', 'study', 'according',
            'scientists', 'data', 'evidence', '📚', '🔬'
        ]
        return any(indicator in text.lower() for indicator in fact_indicators)
    
    def generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate actionable recommendations based on popularity analysis."""
        recommendations = []
        
        # Topic recommendations
        top_topics = analysis.get("top_topics", [])
        if top_topics:
            top_topic = top_topics[0][0]
            recommendations.append(
                f"Focus on {top_topic.replace('_', ' ')} content - it's trending"
            )
        
        # Angle recommendations
        top_angles = analysis.get("top_angles", [])
        if top_angles:
            top_angle = top_angles[0][0]
            recommendations.append(
                f"Use {top_angle.replace('_', ' ')} angle - it drives engagement"
            )
        
        # Emoji recommendations
        top_emojis = analysis.get("top_emojis", [])
        if top_emojis and len(top_emojis) > 0:
            top_emoji = top_emojis[0][0]
            recommendations.append(
                f"Include {top_emoji} emoji - it's popular right now"
            )
        
        # Fact recommendations
        if analysis.get("fact_effectiveness", 0) > 1000:
            recommendations.append(
                "Include scientific facts - they're driving high engagement"
            )
        
        return recommendations


if __name__ == "__main__":
    # Demo: Train predictor on sample data
    print("🤖 Machine Learning Reply Optimizer Demo\n")
    
    # Simulate historical reply data
    sample_replies = [
        {
            "content": "🕷️ Actually, spiders eat 400-800M tons of insects yearly! Nature's pest control 📚",
            "metrics": {"like_count": 45, "reply_count": 8, "repost_count": 12, "impression_count": 2000},
            "original_tweet": "I hate spiders they're so useless",
        },
        {
            "content": "Spiders are beneficial creatures.",
            "metrics": {"like_count": 2, "reply_count": 0, "repost_count": 0, "impression_count": 500},
            "original_tweet": "Kill all spiders",
        },
        # Add more samples...
    ]
    
    # Extend for training (need 50+)
    sample_replies.extend([sample_replies[0]] * 25)
    sample_replies.extend([sample_replies[1]] * 25)
    
    # Train predictor
    predictor = ReplyQualityPredictor()
    success = predictor.train(sample_replies, min_samples=30)
    
    if success:
        print("✅ Model trained successfully!\n")
        
        # Test predictions
        test_candidates = [
            "Spiders are good.",
            "🕷️ Fun fact: Spider silk is 5x stronger than steel! These little engineers are amazing 💚",
            "Why would you kill a spider? They eat pests and help your home!",
        ]
        
        print("📊 Predicting engagement for candidates:\n")
        ranked = predictor.rank_candidates(
            test_candidates,
            original_tweet="Should I kill this spider in my house?"
        )
        
        for i, (reply, score) in enumerate(ranked, 1):
            print(f"{i}. Score: {score:.4f}")
            print(f"   Reply: {reply}\n")
    
    # Demo: Popularity analyzer
    print("\n🔥 Analyzing trending content...\n")
    
    trending_posts = [
        {
            "text": "🕷️ Did you know spiders eat 400-800M tons of insects per year? 🔬",
            "metrics": {"like_count": 150, "reply_count": 30, "repost_count": 45},
        },
        {
            "text": "Spider in my room help I'm scared 😱",
            "metrics": {"like_count": 20, "reply_count": 50, "repost_count": 5},
        },
    ]
    
    analyzer = PopularityAnalyzer()
    analysis = analyzer.analyze_trending_content(trending_posts)
    
    print("Top Topics:")
    for topic, score in analysis["top_topics"][:3]:
        print(f"  • {topic}: {score} engagement points")
    
    print("\nTop Angles:")
    for angle, score in analysis["top_angles"][:3]:
        print(f"  • {angle}: {score} engagement points")
    
    recommendations = analyzer.generate_recommendations(analysis)
    print("\n💡 Recommendations:")
    for rec in recommendations:
        print(f"  • {rec}")
