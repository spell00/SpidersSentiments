import urllib.parse
from typing import Optional

import requests
from bs4 import BeautifulSoup

from spider_guardian.storage import persist_scraped_articles

# BBC Scraper
def scrape_bbc(query, num_articles=10):
    base_url = "https://www.bbc.co.uk/search?q="
    search_url = base_url + urllib.parse.quote(query)
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    articles = []
    for article in soup.find_all()[:num_articles]:
        title_tag = article.find('h1')
        link = article.find('a')['href']
        if title_tag and link:
            articles.append({
                'title': title_tag.get_text(),
                'link': urllib.parse.urljoin("https://www.bbc.co.uk", link)
            })
    return articles

# CNN Scraper
def scrape_cnn(query, num_articles=10):
    base_url = "https://edition.cnn.com/search?q="
    search_url = base_url + urllib.parse.quote(query)
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    articles = []
    for article in soup.find_all('article')[:num_articles]:
        title_tag = article.find('h3', class_='cnn-search__result-headline')
        link = article.find('a')['href']
        if title_tag and link:
            articles.append({
                'title': title_tag.get_text(),
                'link': urllib.parse.urljoin("https://edition.cnn.com", link)
            })
    return articles

# National Geographic Scraper
def scrape_natgeo(query, num_articles=10):
    base_url = "https://www.nationalgeographic.com/search?q="
    search_url = base_url + urllib.parse.quote(query)
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    articles = []
    for article in soup.find_all('a', class_='SearchResults-searchResult')[:num_articles]:
        title_tag = article.find('h2')
        link = article['href']
        if title_tag and link:
            articles.append({
                'title': title_tag.get_text(),
                'link': urllib.parse.urljoin("https://www.nationalgeographic.com", link)
            })
    return articles

# Function to filter articles based on exclusion terms
def filter_articles(articles, exclude_terms):
    filtered_articles = []

    for article in articles:
        title = article.get('title', '').lower()
        # Check if any exclude terms are in the title
        if not any(term in title for term in exclude_terms):
            filtered_articles.append(article)

    return filtered_articles

def fetch_content(url: str) -> Optional[str]:
    try:
        response = requests.get(url, timeout=15)
        if response.ok:
            return response.text
    except Exception as exc:
        print(f"Failed to fetch {url}: {exc}")
    return None

# Main Function
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="spiders", help="Search query")
    parser.add_argument("--num_articles", type=int, default=-1, help="Number of articles to retrieve")
    parser.add_argument("--chemin_articles", type=str, default="articles", help="Directory for optional HTML snapshots")
    parser.add_argument("--article-store", type=str, default="data/articles.json", help="Path to TinyDB JSON store")
    parser.add_argument("--sql-db", type=str, default="data/spider_guardian.sqlite", help="Path to SQLite database")
    parser.add_argument("--legacy-csv", type=str, default=None, help="Optional CSV export for backwards compatibility")
    parser.add_argument("--save-html", action="store_true", default=False, help="Persist HTML snapshots alongside DB storage")
    parser.add_argument("--store-content", action="store_true", help="Persist HTML bodies inside the databases")
    args = parser.parse_args()

    exclude_terms = ["spider-man", "mx-5", "mx-124", "Cleveland Spiders"]

    # Collect articles from multiple sources
    all_articles = []
    all_articles.extend(scrape_bbc(args.query, args.num_articles))
    all_articles.extend(scrape_cnn(args.query, args.num_articles))
    all_articles.extend(scrape_natgeo(args.query, args.num_articles))

    # Filter out excluded articles
    filtered_articles = filter_articles(all_articles, exclude_terms)

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
        source="multi_news",
    )

    print(f"Retrieved {len(filtered_articles)} articles related to '{args.query}'.")
