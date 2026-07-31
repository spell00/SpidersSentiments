from typing import Optional

import os
import requests

from spider_guardian.storage import persist_scraped_articles

def fetch_articles_from_newsapi(api_key, query, exclude, num_articles=1000, page_size=100):
    base_url = "https://thenewsapi.org/v2/everything"
    all_articles = []

    total_pages = num_articles // page_size
    if num_articles % page_size != 0:
        total_pages += 1

    for page in range(1, total_pages + 1):
        # Construct the query parameters
        params = {
            'q': query,
            'apiKey': api_key,
            'pageSize': page_size,
            'page': page,
            'language': 'en',
        }

        # Make the API request
        response = requests.get(base_url, params=params)

        # Check if the request was successful
        if response.status_code != 200:
            print(f"Failed to retrieve articles: {response.status_code}, {response.json()}")
            return []

        articles = response.json().get('articles', [])
        if not articles:
            print("No more articles available.")
            break

        all_articles.extend(articles)

        # If we've already collected the number of articles we wanted, stop
        if len(all_articles) >= num_articles:
            break

    return all_articles[:num_articles]

def filter_articles(articles, exclude_terms):
    filtered_articles = []

    for article in articles:
        title = article.get('title', '').lower()
        # Check if any exclude terms are in the title
        if not any(term in title for term in exclude_terms):
            filtered_articles.append({
                'title': article['title'],
                'link': article['url']
            })

    return filtered_articles

def fetch_content(url: str) -> Optional[str]:
    try:
        response = requests.get(url, timeout=15)
        if response.ok:
            return response.text
    except Exception as exc:
        print(f"Failed to fetch {url}: {exc}")
    return None

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default=os.getenv("NEWSAPI_KEY"), help="NewsAPI key, or set NEWSAPI_KEY")
    parser.add_argument("--query", type=str, default="spiders", help="Search query")
    parser.add_argument("--num_articles", type=int, default=1000, help="Total number of articles to retrieve")
    parser.add_argument("--chemin_articles", type=str, default="articles", help="Directory for optional HTML snapshots")
    parser.add_argument("--article-store", type=str, default="data/articles.json", help="Path to TinyDB JSON store")
    parser.add_argument("--sql-db", type=str, default="data/spider_guardian.sqlite", help="Path to SQLite database")
    parser.add_argument("--legacy-csv", type=str, default=None, help="Optional CSV export for backwards compatibility")
    parser.add_argument("--save-html", action="store_true", default=False, help="Persist HTML snapshots alongside DB storage")
    parser.add_argument("--store-content", action="store_true", help="Persist HTML in the databases")
    args = parser.parse_args()

    exclude_terms = ["spider-man", "mx-5", "mx-124"]

    if not args.api_key:
        parser.error("NewsAPI key required: pass --api_key or set NEWSAPI_KEY")

    # Fetch articles from NewsAPI with pagination
    articles = fetch_articles_from_newsapi(args.api_key, args.query, exclude_terms, args.num_articles)

    # Filter out excluded articles
    filtered_articles = filter_articles(articles, exclude_terms)

    persist_scraped_articles(
        filtered_articles,
        query=args.query,
        output_dir=args.chemin_articles,
        article_store_path=args.article_store,
        sql_db_path=args.sql_db,
        fetch_content=fetch_content if (args.save_html or args.store_content) else None,
        save_html=args.save_html,
        store_content=args.store_content,
        legacy_csv_path=args.legacy_csv,
        source="newsapi",
    )

    print(f"Retrieved {len(filtered_articles)} articles related to '{args.query}'.")
