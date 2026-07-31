"""Track sentiment trends over time for spider-related content."""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)


def analyze_sentiment_trends(
    db_path: str = "data/spider_trending.sqlite",
    days_back: int = 30,
    output_dir: str = "figures/trends"
) -> pd.DataFrame:
    """
    Analyze how spider sentiment is changing over time.
    
    Returns DataFrame with daily aggregates of positive/negative sentiment.
    """
    
    conn = sqlite3.connect(db_path)
    
    # Get trending posts with engagement metrics
    cutoff = (datetime.utcnow() - timedelta(days=days_back)).isoformat()
    query = """
        SELECT 
            DATE(collected_at) as date,
            COUNT(*) as post_count,
            AVG(like_count) as avg_likes,
            AVG(reply_count) as avg_replies,
            AVG(impression_count) as avg_impressions,
            text
        FROM trending_posts
        WHERE collected_at >= ?
        GROUP BY DATE(collected_at)
        ORDER BY date
    """
    
    df = pd.read_sql_query(query, conn, params=(cutoff,))
    conn.close()
    
    if df.empty:
        logging.warning("No data found for sentiment trend analysis")
        return df
    
    # Simple sentiment scoring based on keywords
    positive_keywords = ["cute", "cool", "amazing", "beautiful", "fascinating", "friend", "helpful", "love"]
    negative_keywords = ["scary", "kill", "hate", "disgusting", "gross", "afraid", "fear", "dangerous"]
    
    def score_sentiment(text):
        if not isinstance(text, str):
            return 0
        text_lower = text.lower()
        pos_score = sum(1 for kw in positive_keywords if kw in text_lower)
        neg_score = sum(1 for kw in negative_keywords if kw in text_lower)
        return pos_score - neg_score
    
    # Apply sentiment scoring (this is simplified; your existing VADER could be used too)
    # For now, we'll just track engagement as a proxy for sentiment shift
    
    # Create visualization
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Spider Content Trends (Last {days_back} Days)', fontsize=16)
    
    # Post volume over time
    axes[0, 0].plot(df['date'], df['post_count'], marker='o')
    axes[0, 0].set_title('Posts per Day')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Average engagement
    axes[0, 1].plot(df['date'], df['avg_likes'], marker='o', label='Likes')
    axes[0, 1].plot(df['date'], df['avg_replies'], marker='s', label='Replies')
    axes[0, 1].set_title('Average Engagement per Post')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Impressions trend
    axes[1, 0].plot(df['date'], df['avg_impressions'], marker='o', color='purple')
    axes[1, 0].set_title('Average Impressions per Post')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Impressions')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Engagement rate (likes + replies) / impressions
    df['engagement_rate'] = ((df['avg_likes'] + df['avg_replies']) / df['avg_impressions'].replace(0, 1)) * 100
    axes[1, 1].plot(df['date'], df['engagement_rate'], marker='o', color='green')
    axes[1, 1].set_title('Engagement Rate (%)')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Rate')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path / f'spider_sentiment_trends_{days_back}d.png', dpi=150, bbox_inches='tight')
    logging.info(f"Saved trend chart to {output_path / f'spider_sentiment_trends_{days_back}d.png'}")
    
    # Calculate trend direction
    if len(df) >= 7:
        recent_engagement = df.tail(7)['engagement_rate'].mean()
        older_engagement = df.head(7)['engagement_rate'].mean()
        
        if recent_engagement > older_engagement * 1.1:
            trend = "📈 POSITIVE - Spider content engagement is UP!"
        elif recent_engagement < older_engagement * 0.9:
            trend = "📉 NEGATIVE - Spider content engagement is DOWN"
        else:
            trend = "➡️ STABLE - Spider content engagement is steady"
        
        logging.info(f"Trend: {trend}")
        logging.info(f"  Recent avg: {recent_engagement:.2f}%")
        logging.info(f"  Earlier avg: {older_engagement:.2f}%")
    
    return df


def compare_our_performance(
    interactions_db: str = "data/spider_guardian.sqlite",
    trending_db: str = "data/spider_trending.sqlite",
    days_back: int = 7
) -> dict:
    """
    Compare how well OUR replies perform vs general spider content.
    """
    
    # Get our replies' performance
    conn = sqlite3.connect(interactions_db)
    cutoff = (datetime.utcnow() - timedelta(days=days_back)).isoformat()
    
    our_query = """
        SELECT 
            COUNT(*) as reply_count,
            json_extract(content, '$.reply_text') as reply_text
        FROM scraped_articles
        WHERE metadata LIKE '%interaction%'
          AND created_at >= ?
    """
    
    our_df = pd.read_sql_query(our_query, conn, params=(cutoff,))
    conn.close()
    
    # Get trending posts performance
    conn = sqlite3.connect(trending_db)
    trending_query = """
        SELECT 
            AVG(like_count) as avg_likes,
            AVG(reply_count) as avg_replies,
            AVG(impression_count) as avg_impressions
        FROM trending_posts
        WHERE collected_at >= ?
    """
    
    trending_df = pd.read_sql_query(trending_query, conn, params=(cutoff,))
    conn.close()
    
    stats = {
        "our_reply_count": int(our_df['reply_count'].sum()),
        "trending_avg_likes": float(trending_df['avg_likes'].iloc[0]) if not trending_df.empty else 0,
        "trending_avg_replies": float(trending_df['avg_replies'].iloc[0]) if not trending_df.empty else 0,
        "trending_avg_impressions": float(trending_df['avg_impressions'].iloc[0]) if not trending_df.empty else 0,
    }
    
    logging.info("Performance Comparison (Last %d days):", days_back)
    logging.info("  🤖 Our replies posted: %d", stats["our_reply_count"])
    logging.info("  📊 Trending posts avg likes: %.1f", stats["trending_avg_likes"])
    logging.info("  📊 Trending posts avg replies: %.1f", stats["trending_avg_replies"])
    logging.info("  📊 Trending posts avg impressions: %.1f", stats["trending_avg_impressions"])
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze spider sentiment trends")
    parser.add_argument("--days", type=int, default=30, help="Days to look back")
    parser.add_argument("--compare", action="store_true", help="Compare our performance vs trending")
    
    args = parser.parse_args()
    
    trends = analyze_sentiment_trends(days_back=args.days)
    
    if args.compare:
        stats = compare_our_performance(days_back=min(args.days, 7))
        print("\n📊 Performance Stats:", stats)
