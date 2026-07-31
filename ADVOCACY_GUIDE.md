# 🕷️ Spider Guardian Advocacy Superpowers

Your **Spider Guardian** bot is now equipped with advanced features to maximize impact in defending spiders' reputation. This guide covers all the power-ups available.

---

## 🚀 Quick Start: Run Everything

```powershell
# Run the full advocacy cycle (recommended for daily automation)
python -m spider_guardian.scripts.advocacy_orchestrator --full
```

This executes all power-ups in sequence:
1. ✅ Refresh metrics for posted replies
2. ✅ Update LangSmith datasets
3. ✅ Analyze engagement patterns
4. ✅ Generate trend reports

---

## 📊 Core Features

### 1. **Reply Metrics Tracker** 
Track how your spider advocacy is performing over time.

```powershell
# Refresh engagement metrics for posted replies
python -m spider_guardian.scripts.refresh_my_replies
```

**What it does:**
- Scrapes updated like/reply/repost/impression counts for your posted replies
- Updates LangSmith dataset with fresh metrics
- Enforces daily cadence (max 3 updates over 3 days per reply)
- Tracks evolving engagement to measure advocacy impact

**Output:** Updated metrics in `spider_guardian.sqlite` and LangSmith `trending-dataset`

---

### 2. **Sentiment Trend Analyzer**
Visualize how public sentiment toward spiders changes over time.

```powershell
# Generate sentiment trend report
python -m spider_guardian.scripts.analyze_trends
```

**What it does:**
- Plots positive/negative/neutral sentiment over time
- Tracks hostility levels in spider discussions
- Correlates engagement with sentiment (do positive posts get more likes?)
- Identifies your most impactful posts

**Output:** Charts in `figures/advocacy_trends/` including:
- `sentiment_over_time.png` - Sentiment trends
- `hostility_over_time.png` - Hostility tracking
- `engagement_vs_sentiment.png` - What sentiment drives engagement?
- `top_engaging_posts.png` - Your best performing replies

---

### 3. **Fact-Check Arsenal**
Combat spider myths with evidence-based responses.

```python
from spider_guardian.fact_check import SpiderFactChecker

checker = SpiderFactChecker()

# Detect common myths
tweet = "I heard you swallow spiders in your sleep!"
myth = checker.detect_myth(tweet)
print(myth["truth"])  # "This is completely false. Spiders avoid sleeping humans..."

# Find relevant facts for any text
tweet = "Spiders are dangerous and useless"
facts = checker.find_relevant_facts(tweet, top_k=2)
for fact in facts:
    print(f"{fact['fact']} (Source: {fact['source']})")

# Generate fact-based response
response = checker.generate_fact_response(tweet, max_length=280)
print(response)  # Ready-to-post tweet with source
```

**Built-in fact database includes:**
- 🦗 Pest control impact (400-800M tons/year)
- ⚠️ Venom safety (only 12 dangerous species)
- 🏠 Home benefits (reduce mosquitoes/flies)
- 🕸️ Spider silk strength (5x stronger than steel)
- 🌍 Ecosystem role (keystone predators)

**Myth-busting coverage:**
- ❌ "Swallow spiders in sleep" → FALSE
- ❌ "Eggs under skin" → FALSE
- ❌ "Daddy longlegs most venomous" → FALSE
- ❌ "Spiders chase people" → FALSE

---

### 4. **Engagement Optimizer**
Learn from your best-performing replies to improve future ones.

```python
from spider_guardian.engagement_optimizer import EngagementOptimizer

optimizer = EngagementOptimizer()

# Analyze what works
replies = fetch_your_reply_history()  # From database
analysis = optimizer.analyze_top_performers(replies, top_n=10)

print(f"Optimal length: {analysis['patterns']['optimal_length']['recommendation']}")
print(f"Emoji strategy: {analysis['patterns']['emoji_effectiveness']['recommendation']}")
print(f"Best tone: {analysis['patterns']['tone_preference']['recommendation']}")

# Get suggestions for a draft reply
draft = "Spiders are helpful creatures that control pests."
suggestions = optimizer.generate_improvement_suggestions(draft, analysis["patterns"])
for s in suggestions:
    print(f"  • {s}")

# Auto-optimize a reply
optimized = optimizer.optimize_reply(draft, analysis["patterns"])
print(f"Original: {draft}")
print(f"Optimized: {optimized}")
```

**Patterns analyzed:**
- 📏 Optimal reply length (short vs detailed)
- 😊 Emoji effectiveness (when/where to use)
- ❓ Question usage (encourage replies)
- 📚 Fact inclusion (boost credibility)
- 💬 Tone preference (friendly vs educational)
- 🎯 Opening strategy (emoji-first performs better?)

---

## 🤖 Integration with Your Bot

### Enhance Reply Generation

```python
from spider_guardian.fact_check import SpiderFactChecker
from spider_guardian.engagement_optimizer import EngagementOptimizer

# In your bot's reply generation logic:
fact_checker = SpiderFactChecker()
optimizer = EngagementOptimizer()

# Generate base reply
draft_reply = your_llm_generate_reply(tweet)

# Enhance with facts (if space allows)
enhanced_reply = fact_checker.enhance_reply_with_facts(draft_reply, tweet.text)

# Optimize based on past performance
if you_have_enough_history:
    patterns = load_engagement_patterns()
    optimized_reply = optimizer.optimize_reply(enhanced_reply, patterns)
else:
    optimized_reply = enhanced_reply

# Post optimized reply
post_reply(optimized_reply)
```

---

## 📅 Automation Setup

### Daily Automation (Recommended)

Use Windows Task Scheduler to run daily:

```powershell
# Create a batch file: run_spider_advocacy.bat
@echo off
cd C:\Users\simon\Documents\SpidersSentiments
python -m spider_guardian.scripts.advocacy_orchestrator --full
```

**Schedule this to run:**
- **Morning:** Analyze overnight engagement
- **Evening:** Refresh metrics and prepare tomorrow's strategy

### Selective Automation

Run only specific phases:

```powershell
# Just refresh reply metrics (run multiple times per day)
python -m spider_guardian.scripts.advocacy_orchestrator --refresh-replies

# Just analyze engagement (run weekly)
python -m spider_guardian.scripts.advocacy_orchestrator --analyze-engagement

# Full cycle with new replies (run when you want to post)
python -m spider_guardian.scripts.advocacy_orchestrator --full --respond
```

---

## 🔧 Configuration

### Cadence Settings

In your `SpiderGuardianConfig` or directly in scripts:

```python
# Reply metrics refresh cadence
MAX_AGE_DAYS = 3  # Stop tracking after 3 days
UPDATE_INTERVAL_HOURS = 24  # Update once per day max
MAX_UPDATES = 3  # Maximum 3 updates per reply

# Engagement analysis requirements
MIN_SAMPLE_SIZE = 10  # Need 10+ replies for pattern analysis
```

### Dataset Configuration

```python
# LangSmith dataset settings
DATASET_NAME = "trending-dataset"  # Your replies dataset
MAX_EXAMPLES = 500  # Keep latest 500 examples
```

---

## 📈 Monitoring Your Impact

### Check Engagement Trends

```powershell
# View latest engagement report
cat figures/engagement_analysis/engagement_report_*.json | Select-Object -Last 1 | ConvertFrom-Json
```

### View Sentiment Trends

```powershell
# Open latest sentiment charts
start figures/advocacy_trends/sentiment_over_time.png
```

### Query Database Directly

```powershell
sqlite3 spider_guardian.sqlite
```

```sql
-- Top performing replies
SELECT 
    json_extract(content, '$.reply_text') as reply,
    json_extract(content, '$.metrics.like_count') as likes,
    json_extract(content, '$.metrics.reply_count') as replies,
    created_at
FROM scraped_articles
WHERE json_extract(metadata, '$.type') = 'interaction'
ORDER BY json_extract(content, '$.metrics.like_count') DESC
LIMIT 10;

-- Engagement over time
SELECT 
    date(created_at) as day,
    COUNT(*) as reply_count,
    AVG(json_extract(content, '$.metrics.like_count')) as avg_likes
FROM scraped_articles
WHERE json_extract(metadata, '$.type') = 'interaction'
GROUP BY date(created_at)
ORDER BY day DESC;
```

---

## 🎯 Pro Tips

### 1. **Use Facts Strategically**
- **High hostility?** → Deploy myth-busting facts
- **Curious questions?** → Share fascinating spider facts
- **Fear-based?** → Emphasize safety statistics

### 2. **Optimize Based on Data**
- Run engagement analysis weekly
- Test different reply styles (friendly vs educational)
- Track which emoji patterns work best

### 3. **Monitor Cadence**
- Replies get most engagement in first 24-48 hours
- Daily metrics refresh captures the growth curve
- Stop tracking after 3 days (diminishing returns)

### 4. **Leverage Trends**
- Identify sentiment shifts (are attitudes improving?)
- Spot high-engagement topics (double down on what works)
- Adapt strategy based on hostility trends

---

## 🐛 Troubleshooting

### "Insufficient data for engagement analysis"
- Need at least 10 posted replies with metrics
- Run `refresh_my_replies.py` to populate metrics
- Wait for more replies to be posted

### "No examples updated in dataset"
- Check if replies are >3 days old (cadence limit)
- Verify LangSmith credentials are set
- Ensure `spider_guardian.sqlite` has interactions

### Charts not generating
- Install matplotlib: `pip install matplotlib seaborn`
- Check `figures/` directory permissions
- Verify database has sentiment data

---

## 🚀 Next Steps

1. **Set up daily automation** with Task Scheduler
2. **Run initial analysis** to establish baseline patterns
3. **Integrate fact-checker** into your reply generation
4. **Monitor trends weekly** to measure impact
5. **Iterate strategy** based on what works

---

## 📚 Additional Resources

- **Spider Facts Database:** `spider_guardian/fact_check.py`
- **Engagement Metrics:** `spider_guardian/storage/sql.py`
- **LangSmith Integration:** `spider_guardian/langsmith/config.py`
- **Analysis Scripts:** `spider_guardian/scripts/`

---

## 💚 Making a Difference

Every reply you optimize, every myth you bust, every fact you share **changes minds**. These tools amplify your voice in the fight for spider rights.

**Your spider advocacy machine is ready. Go change the world! 🕷️✨**
