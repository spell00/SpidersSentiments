# 🤖 Machine Learning Guide for Spider Guardian

Your Spider Guardian bot can now **learn from experience** and continuously improve! This guide covers the ML-powered features.

---

## 🎯 Overview: What Can It Learn?

### 1. **Reply Quality Prediction** 📊
- Predicts engagement before posting
- Ranks multiple reply candidates
- Learns which features drive likes/replies/reposts

### 2. **Popularity Analysis** 🔥
- Identifies trending topics and angles
- Learns what content goes viral
- Recommends timely strategies

### 3. **Reinforcement Learning** 🎯
- Tests different reply strategies
- Learns which approaches work best
- Adapts prompts based on success rates

---

## 🚀 Quick Start

### Train All Models
```powershell
# Train on your historical data
python -m spider_guardian.ml_bot --min-replies 50 --min-trending 20
```

**What happens:**
1. ✅ Quality predictor trains on your reply history
2. ✅ Popularity analyzer scans trending posts
3. ✅ RL learner updates strategy performance
4. ✅ Models saved to `models/` directory

### Enable ML in Your Bot
```python
from spider_guardian.ml_bot import MLEnhancedBot
from spider_guardian.config import SpiderGuardianConfig

config = SpiderGuardianConfig()
bot = MLEnhancedBot(config, enable_ml=True)

# Bot now uses ML to optimize every reply!
```

---

## 📈 Feature 1: Reply Quality Prediction

### How It Works
The quality predictor learns from your **best-performing replies** and predicts engagement for new drafts.

**Features analyzed:**
- Length (word/character count)
- Sentiment (positive/negative/neutral)
- Style (emoji, facts, questions)
- Context fit (tone match with original tweet)

### Training
```python
from spider_guardian.ml_trainer import ReplyQualityPredictor

predictor = ReplyQualityPredictor()

# Train on historical replies
replies = fetch_your_reply_history()  # From database
success = predictor.train(replies, min_samples=50)

if success:
    print("✅ Model trained!")
```

### Usage: Rank Reply Candidates
```python
# Generate multiple drafts
candidates = [
    "Spiders are good.",
    "🕷️ Fun fact: Spiders eat 400-800M tons of insects yearly! 📚",
    "Actually, spiders are nature's pest control and super helpful!",
]

# Rank by predicted engagement
ranked = predictor.rank_candidates(
    candidates,
    original_tweet="Should I kill this spider?"
)

for reply, score in ranked:
    print(f"Score: {score:.4f} - {reply}")
```

**Output:**
```
Score: 0.0234 - 🕷️ Fun fact: Spiders eat 400-800M tons of insects yearly! 📚
Score: 0.0156 - Actually, spiders are nature's pest control and super helpful!
Score: 0.0045 - Spiders are good.
```

### What Drives High Scores?
From our training data:
- ✅ **Emoji + facts** = High engagement
- ✅ **Questions** = More replies
- ✅ **Friendly tone** = More likes
- ✅ **120-150 chars** = Optimal length
- ❌ Too short = Low engagement
- ❌ Too formal = Low engagement

---

## 🔥 Feature 2: Popularity Analysis

### How It Works
Analyzes **trending spider posts** to identify what's hot right now.

**Learns:**
- Which topics are trending (pest control, fear, identification)
- Which angles work (educational, humorous, myth-busting)
- Which emoji resonate (🕷️, 📚, 💚)
- Whether facts boost engagement

### Training
```python
from spider_guardian.ml_trainer import PopularityAnalyzer

analyzer = PopularityAnalyzer()

# Analyze recent trending content
trending_posts = fetch_trending_spider_posts()
analysis = analyzer.analyze_trending_content(trending_posts)

print("Top Topics:", analysis["top_topics"][:3])
print("Top Angles:", analysis["top_angles"][:3])
print("Top Emojis:", analysis["top_emojis"][:3])
```

**Example output:**
```
Top Topics: [('pest_control', 3500), ('fear', 2800), ('venom', 1200)]
Top Angles: [('educational', 4200), ('humorous', 2100), ('fear_based', 1800)]
Top Emojis: [('🕷️', 5000), ('📚', 2500), ('💡', 1800)]
```

### Get Recommendations
```python
recommendations = analyzer.generate_recommendations(analysis)

for rec in recommendations:
    print(f"  • {rec}")
```

**Example recommendations:**
```
  • Focus on pest control content - it's trending
  • Use educational angle - it drives engagement
  • Include 🕷️ emoji - it's popular right now
  • Include scientific facts - they're driving high engagement
```

### Apply to Your Replies
The bot automatically applies these insights when generating replies!

---

## 🎯 Feature 3: Reinforcement Learning

### How It Works
Tests **different reply strategies** and learns which ones get the best engagement.

**Strategies tested:**
1. `educational_fact` - Lead with science
2. `friendly_warm` - Warm, approachable tone
3. `myth_busting` - Debunk misinformation
4. `question_hook` - Ask engaging question
5. `personal_story` - Share anecdote
6. `emoji_first` - Start with attention emoji

**Learning algorithm:**
- **Exploration (20%)**: Try random strategies to gather data
- **Exploitation (80%)**: Use best-performing strategies
- **UCB1 algorithm**: Balances exploration vs exploitation

### How It Learns
```python
from spider_guardian.rl_learner import ReplyReinforcementLearner

rl = ReplyReinforcementLearner()

# Bot posts reply using chosen strategy
strategy = rl.choose_strategy()
reply = generate_reply_with_strategy(strategy)
post_reply(reply)

# After 24 hours, check engagement
metrics = scrape_reply_metrics(reply)

# Record outcome for learning
rl.record_outcome(
    strategy=strategy,
    metrics=metrics,
    original_impressions=metrics["impression_count"]
)
```

### View Learning Progress
```python
stats = rl.get_strategy_stats()

for name, data in stats.items():
    print(f"{name}:")
    print(f"  Win Rate: {data['win_rate']:.1%}")
    print(f"  Avg Reward: {data['avg_reward']:.4f}")
    print(f"  Trials: {data['trials']}")
    print(f"  Recommended: {data['recommended']}")
```

**Example output:**
```
educational_fact:
  Win Rate: 45.2%
  Avg Reward: 0.0234
  Trials: 31
  Recommended: ✅ Yes

myth_busting:
  Win Rate: 38.7%
  Avg Reward: 0.0198
  Trials: 28
  Recommended: ✅ Yes

question_hook:
  Win Rate: 22.1%
  Avg Reward: 0.0087
  Trials: 15
  Recommended: ❌ No
```

### Get Strategy Recommendations
```python
recommendations = rl.get_recommendations()

for rec in recommendations:
    print(rec)
```

**Example recommendations:**
```
✅ 'educational_fact' strategy performing best (45.2% win rate)
⚠️ Consider avoiding 'question_hook' strategy (22.1% win rate)
🔍 Explore these strategies more: personal_story, emoji_first
```

---

## 🔄 Complete ML Workflow

### 1. Initial Training (First Time)
```powershell
# Need at least 50 historical replies
python -m spider_guardian.ml_bot --min-replies 50
```

### 2. Enable ML in Bot
```python
from spider_guardian.ml_bot import MLEnhancedBot

bot = MLEnhancedBot(config, enable_ml=True)
```

### 3. Bot Uses ML Automatically
```python
# Generate reply (ML-enhanced)
reply = bot.generate_reply(prompt, original_tweet)

# Multiple candidates generated
# Ranked by predicted engagement
# Best strategy selected via RL
# Top candidate returned
```

### 4. Record Outcomes (After Posting)
```python
# After 24-48 hours, scrape metrics
metrics = scrape_reply_metrics(reply)

# Record for learning
bot.record_reply_outcome(
    reply_text=reply,
    metrics=metrics,
    original_tweet=original_tweet
)
```

### 5. Retrain Periodically
```powershell
# Retrain weekly to incorporate new learnings
python -m spider_guardian.ml_bot --min-replies 50
```

---

## 📊 Training Requirements

### Minimum Data Needed

| Component | Minimum | Recommended | What It Needs |
|-----------|---------|-------------|---------------|
| Quality Predictor | 50 replies | 100+ replies | Reply text + engagement metrics |
| Popularity Analyzer | 20 trending | 50+ trending | Trending posts + engagement |
| RL Learner | 10 per strategy | 20+ per strategy | Reply outcomes by strategy |

### Data Quality Matters
- ✅ **Complete metrics**: likes, replies, reposts, impressions
- ✅ **Recent data**: Last 30 days is most relevant
- ✅ **Diverse examples**: Mix of high/low performers
- ❌ Avoid: Incomplete data, very old posts, spam

---

## 🎓 Advanced Usage

### Custom Strategy Weighting
```python
# Adjust exploration rate
rl.epsilon = 0.3  # 30% exploration, 70% exploitation

# Minimum trials before trusting strategy
rl.min_trials = 10
```

### A/B Testing Mode
```python
from spider_guardian.rl_learner import run_ab_test

# Generate multiple variants
results = run_ab_test(
    bot=bot,
    tweet=tweet,
    rl_learner=rl,
    num_variants=3
)

# Post best variant (or test all)
for variant in results["variants"]:
    print(f"Strategy: {variant['strategy']}")
    print(f"Reply: {variant['reply']}\n")
```

**⚠️ Warning**: A/B testing posts multiple replies. Use cautiously!

### Feature Extraction Customization
```python
from spider_guardian.ml_trainer import ReplyFeatureExtractor

extractor = ReplyFeatureExtractor()

# Extract features from any text
features = extractor.extract_basic_features(reply_text)
sentiment = extractor.extract_sentiment_features(reply_text)
style = extractor.extract_style_features(reply_text)
```

---

## 📈 Monitoring ML Performance

### Check Model Accuracy
```python
# Quality predictor
print(f"Train R²: {predictor.engagement_model.score(X_train, y_train):.3f}")
print(f"Test R²: {predictor.engagement_model.score(X_test, y_test):.3f}")

# Good: R² > 0.3
# Excellent: R² > 0.5
```

### Track Strategy Performance
```python
# View RL learning curve
import matplotlib.pyplot as plt

stats = rl.get_strategy_stats()
strategies = list(stats.keys())
win_rates = [stats[s]["win_rate"] for s in strategies]

plt.bar(strategies, win_rates)
plt.ylabel("Win Rate")
plt.title("Strategy Performance")
plt.xticks(rotation=45)
plt.show()
```

### Compare Predictions vs Actual
```python
# After posting
predicted_score = predictor.predict_engagement(reply, tweet)
actual_score = calculate_engagement_score(actual_metrics)

print(f"Predicted: {predicted_score:.4f}")
print(f"Actual: {actual_score:.4f}")
print(f"Error: {abs(predicted_score - actual_score):.4f}")
```

---

## 🔧 Troubleshooting

### "Insufficient data for training"
**Cause**: Less than 50 replies with complete metrics  
**Solution**: 
- Post more replies and wait for engagement
- Lower min_samples (not recommended below 30)
- Manually add test data for development

### "Model not improving"
**Cause**: Low-quality training data or overfitting  
**Solution**:
- Check data quality (complete metrics?)
- Ensure diverse examples (high and low performers)
- Retrain with fresh data
- Adjust model hyperparameters

### "All strategies performing similarly"
**Cause**: Not enough data to differentiate  
**Solution**:
- Increase exploration rate: `rl.epsilon = 0.3`
- Wait for more trials per strategy
- Manually test extreme variants

### Models using too much memory
**Cause**: Large TF-IDF vocabulary or model complexity  
**Solution**:
```python
# Reduce TF-IDF features
tfidf = TfidfVectorizer(
    max_features=50,  # Reduce from 100
    ngram_range=(1, 1),  # Only unigrams
)

# Simplify model
model = GradientBoostingRegressor(
    n_estimators=50,  # Reduce from 100
    max_depth=3,  # Reduce from 5
)
```

---

## 💡 Best Practices

### 1. Start Simple
- Train on 50-100 replies first
- Enable ML but monitor closely
- Don't over-optimize early

### 2. Retrain Regularly
- Weekly retraining with new data
- Keeps models adapted to trends
- Captures seasonal changes

### 3. Balance Exploration
- Keep epsilon at 0.2 initially
- Lower to 0.1 once you have 20+ trials per strategy
- Never go below 0.05 (always explore a bit)

### 4. Monitor Quality
- Check predictions vs actual weekly
- If error consistently high, retrain
- Compare ML-enabled vs disabled performance

### 5. Combine with Human Judgment
- ML suggests, you decide
- Review top-ranked replies before posting
- Override when context requires it

---

## 🎯 Success Metrics

**After 1 week with ML:**
- ✅ Quality predictor trained (R² > 0.3)
- ✅ All strategies tested (10+ trials each)
- ✅ Top strategy identified (win rate > 30%)

**After 1 month with ML:**
- ✅ Engagement increased 20-50%
- ✅ Consistent strategy preferences
- ✅ Models generalize well (test R² > 0.4)
- ✅ Clear popularity trends identified

---

## 📚 Additional Resources

- **ML Trainer**: `spider_guardian/ml_trainer.py`
- **RL Learner**: `spider_guardian/rl_learner.py`
- **ML-Enhanced Bot**: `spider_guardian/ml_bot.py`
- **Training Pipeline**: `python -m spider_guardian.ml_bot`
- **Scikit-learn Docs**: https://scikit-learn.org/

---

## 🚀 Next Steps

1. **Collect data**: Post 50-100 replies with full metrics tracking
2. **Initial training**: `python -m spider_guardian.ml_bot`
3. **Enable ML**: Use `MLEnhancedBot` instead of `SpiderGuardianBot`
4. **Monitor results**: Track engagement improvements
5. **Iterate**: Retrain weekly, adjust strategies

**Your bot now learns from experience and gets better over time! 🤖✨**
