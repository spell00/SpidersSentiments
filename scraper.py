import time
import urllib.parse

import requests
from bs4 import BeautifulSoup

from spider_guardian.storage import persist_scraped_articles

def search_google_news(query, num_articles=100):
    base_url = "https://news.google.com/search?q="
    all_results = []

    # Loop to paginate through results
    for start in range(0, num_articles, 10):  # Google News typically shows 10 results per page
        search_url = f"{base_url}{urllib.parse.quote(query)}&start={start}"
        print(f"Fetching results from: {search_url}")

        # Send a request to Google News
        response = requests.get(search_url)

        # Check if the request was successful
        if response.status_code != 200:
            print(f"Failed to retrieve the articles, status code: {response.status_code}")
            break

        # Parse the HTML content
        soup = BeautifulSoup(response.text, "html.parser")

        # Find all articles
        articles = soup.find_all('article')
        if not articles:
            print("No more articles found.")
            break  # Exit if no articles are found

        for article in articles:
            # Attempt to extract the title from different tags
            title_tags = article.find_all('a')  # Look for an anchor tag which usually wraps the title
            for title_tag in title_tags:
                if title_tag and 'aria-label' in title_tag.attrs:  # Check for aria-label attribute
                    title = title_tag['aria-label']  # The title might be stored here
                elif title_tag:
                    title = title_tag.get_text(strip=True)  # Fallback to get text from anchor
                else:
                    title = "Title not found"  # Fallback if title is not found

            link = title_tag['href'] if title_tag else None
            # The link might be a relative URL, so we need to format it properly
            if link and link.startswith('.'):
                link = urllib.parse.urljoin("https://news.google.com", link)

            all_results.append({'title': title, 'link': link})

        # Sleep to avoid overwhelming the server (optional)
        time.sleep(1)  # Sleep for 1 second

    return all_results

def make_request(url):
    """_summary_

    Args:
        url (_type_): _description_

    Returns:
        _type_: _description_
    """
    retries = 5
    for i in range(retries):
        response = requests.get(url)
        if response.status_code == 200:
            return response
        elif response.status_code == 429:
            wait_time = 2 ** i  # Exponential backoff
            print(f"Rate limited. Waiting for {wait_time} seconds...")
            time.sleep(wait_time)
        else:
            print(f"Error: {response.status_code}")
            break
    return None

def persist_articles(articles, args):
    if not articles:
        print("No articles found. Nothing to persist.")
        return

    def fetch_content(link: str):
        response = make_request(link)
        if response and response.status_code == 200:
            return response.text
        if response is not None:
            print(f"Failed to fetch article: {link} (status {response.status_code})")
        return None

    persist_scraped_articles(
        articles,
        query=args.query,
        output_dir=args.chemin_articles,
        article_store_path=args.article_store,
        sql_db_path=args.sql_db,
        fetch_content=fetch_content,
        save_html=args.save_html,
        store_content=args.store_content,
        legacy_csv_path=args.legacy_csv,
        source="google_news",
    )


# Main script to search for spider-related articles
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--chemin_articles", type=str, default="articles")
    parser.add_argument("--article-store", type=str, default="data/articles.json", help="Path to TinyDB JSON store")
    parser.add_argument("--sql-db", type=str, default="data/spider_guardian.sqlite", help="Path to SQLite database")
    parser.add_argument("--legacy-csv", type=str, default=None, help="Optional path to also export a CSV copy")
    parser.add_argument("--save-html", action="store_true", default=False, help="Persist raw HTML snapshots alongside database storage")
    parser.add_argument("--store-content", action="store_true", help="Persist fetched HTML in the databases")
    parser.add_argument("--query", type=str, default='spiders')  # Query to search for spiders
    parser.add_argument("--num_articles", type=int, default=10)  # Number of articles to retrieve
    args = parser.parse_args()

    # Search for articles
    articles = search_google_news(args.query, num_articles=args.num_articles)

    # Persist scraped metadata into the configured storage backends
    persist_articles(articles, args)

    print(f"Retrieved {len(articles)} articles.")
