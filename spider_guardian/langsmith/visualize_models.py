import langsmith
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import json
from langsmith.client import Client as LangSmithClient
import os


# Import our LangSmith integration
try:
    from spider_guardian.langsmith.config import langsmith_integration
except ImportError:
    langsmith_integration = None
    print("LangSmith integration not available")

# Load your dataset
file_path = "resultats/resultats_sentiments.csv"
data = pd.read_csv(file_path)

# Initialize LangSmith client using environment variables
client = LangSmithClient(
    api_key=os.environ.get("LANGSMITH_API_KEY"),
    # org_id=os.environ.get("LANGSMITH_ORG_ID"),
    workspace_id=os.environ.get("LANGSMITH_WORKSPACE_ID")
) if os.environ.get("LANGSMITH_API_KEY") else None

# Example: Visualize sentiment distribution
def plot_sentiment_distribution(data):
    plt.figure(figsize=(8, 6))
    data['sentiment'].value_counts().plot(kind='bar', color='skyblue')
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Frequency')
    plt.savefig('resultats/sentiment_distribution.png')
    plt.close()

# Example: Generate LangSmith visualization
def generate_langsmith_visualization(data):
    """Generate visualizations using LangSmith data and plot metrics.

    Note: LangSmith's Python client doesn't support an arbitrary `send_metrics` call.
    We rely on existing runs/examples and generate a performance report instead.
    """
    if not langsmith_integration or not langsmith_integration.client:
        print("LangSmith integration not available")
        return

    try:
        # Generate performance report from tracked runs
        report = langsmith_integration.generate_performance_report(days=7)

        if report:
            print(f"LangSmith Performance Report (7 days):")
            print(f"- Total replies generated: {report['total_replies_generated']}")
            print(f"- Average generation time: {report['avg_generation_time_ms']:.1f}ms")
            print(f"- Total engagement: {report['total_engagement']}")
            print(f"- Project URL: {langsmith_integration.get_langsmith_url()}")

            # Create visualization of engagement metrics
            if report['total_engagement']:
                engagement_data = report['total_engagement']
                plt.figure(figsize=(10, 6))

                # Plot engagement metrics
                metrics = list(engagement_data.keys())
                values = list(engagement_data.values())

                plt.subplot(1, 2, 1)
                plt.bar(metrics, values, color=['skyblue', 'orange', 'green'])
                plt.title('Total Engagement (7 days)')
                plt.ylabel('Count')

                # Plot generation stats
                plt.subplot(1, 2, 2)
                stats = ['Replies Generated', 'Avg Gen Time (ms)']
                stat_values = [report['total_replies_generated'], report['avg_generation_time_ms']]
                plt.bar(stats, stat_values, color=['purple', 'red'])
                plt.title('Generation Stats (7 days)')

                plt.tight_layout()
                plt.savefig('resultats/langsmith_engagement_metrics.png')
                plt.close()
        else:
            print("No LangSmith data available")

    except Exception as e:
        print(f"Error generating LangSmith visualization: {e}")
# Example: Visualize engagement metrics
def plot_engagement_metrics(data, title='default', save_path='resultats'):
    metrics = [
        'like_count',
        'reply_count',
        # 'impression_count'
    ]
    data[metrics].sum().plot(kind='bar', color=[
        'skyblue',
        'orange',
        # 'green'
        ])
    plt.title('Engagement Metrics')
    plt.xlabel('Metrics')
    plt.ylabel('Total Count')
    plt.savefig(f'{save_path}/engagement_metrics_{title}.png')

# Enhance data gathering
# Example: Fetch replies and their metrics
def fetch_replies_with_metrics():
    # Connect to the database containing replies
    db_path = "data/spider_guardian.sqlite"  # Replace with the correct database path
    connection = sqlite3.connect(db_path)

    try:
        # Query the normalized interactions table to fetch replies and their metrics
        query = """
        SELECT reply_id AS id,
               reply_text AS text,
               like_count AS likes,
               reply_count AS responses,
               impression_count AS impressions,
               created_at
        FROM interactions
        WHERE type = 'interaction'
        ORDER BY datetime(created_at) DESC
        """
        replies = pd.read_sql_query(query, connection)
        return replies
    except Exception as e:
        print(f"Error fetching replies with metrics: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error
    finally:
        connection.close()

# Connect to the database
def fetch_data_from_database(db_path):
    connection = sqlite3.connect(db_path)
    query = "SELECT * FROM collected_data;"  # Replace with your actual table name
    data = pd.read_sql_query(query, connection)
    connection.close()
    return data

# Fetch data from Spider Guardian database
def fetch_data_from_spider_guardian():
    db_path = "data/trending.sqlite"  # Correct database path for trending_posts
    connection = sqlite3.connect(db_path)

    # Query the trending_posts table
    query = "SELECT * FROM trending_posts;"
    data = pd.read_sql_query(query, connection)
    connection.close()
    return data

# Load and visualize JSON data
def load_json_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

# Define schema for interactions database
def create_interactions_table():
    db_path = "data/spider_guardian.sqlite"
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    # Create interactions table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            post_id TEXT,
            interaction_type TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    connection.commit()
    connection.close()

# Migrate interactions.json to database without duplicates
def migrate_interactions_to_db():
    interactions_path = "data/interactions.json"
    db_path = "data/spider_guardian.sqlite"

    # Load interactions from JSON
    with open(interactions_path, "r", encoding="utf-8") as f:
        interactions = json.load(f)

    # Insert interactions into database
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    for interaction in interactions:
        cursor.execute(
            """
            INSERT OR IGNORE INTO interactions (user_id, post_id, interaction_type)
            VALUES (?, ?, ?)
            """,
            (interaction.get("user_id"), interaction.get("post_id"), interaction.get("interaction_type"))
        )

    connection.commit()
    connection.close()

if __name__ == "__main__":
    # Create interactions table
    create_interactions_table()

    # Migrate interactions to database
    migrate_interactions_to_db()

    # Load and preprocess data
    file_path = "resultats/resultats_sentiments.csv"
    data = pd.read_csv(file_path)

    # Fetch data from database
    db_path = "data/your_database.db"  # Replace with your actual database path
    database_data = fetch_data_from_database(db_path)

    # Call visualization functions
    plot_sentiment_distribution(data)
    plot_engagement_metrics(database_data)
    generate_langsmith_visualization(database_data)

    # Fetch data from Spider Guardian database
    spider_guardian_data = fetch_data_from_spider_guardian()

    # Visualize the data
    plot_engagement_metrics(spider_guardian_data)
    generate_langsmith_visualization(spider_guardian_data)

    # Load and visualize JSON data
    interactions_path = "data/interactions.json"
    streamed_posts_path = "data/streamed_posts.json"

    interactions_data = load_json_data(interactions_path)
    streamed_posts_data = load_json_data(streamed_posts_path)

    # Visualize interactions
    plot_engagement_metrics(interactions_data)
    generate_langsmith_visualization(interactions_data)

    # Visualize streamed posts
    plot_engagement_metrics(streamed_posts_data)
    generate_langsmith_visualization(streamed_posts_data)
