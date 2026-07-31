"""
Test script for reply editor detection in SeleniumTwitterClient.
Usage: python test_reply_editor.py <tweet_url>
"""
import sys
import logging
from spider_guardian.config import SpiderGuardianConfig
from spider_guardian.twitter_client import SeleniumTwitterClient

logging.basicConfig(level=logging.INFO)

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_reply_editor.py <tweet_url>")
        sys.exit(1)
    tweet_url = sys.argv[1]
    config = SpiderGuardianConfig()
    client = SeleniumTwitterClient(config)
    print(f"Navigating to: {tweet_url}")
    client.driver.get(tweet_url)
    input("Press Enter after the tweet page is fully loaded...")
    editor = client._locate_reply_editor()
    if editor:
        print("✅ Reply editor found!")
    else:
        print("❌ Reply editor NOT found. Check logs/ for diagnostics.")
    client.close()

if __name__ == "__main__":
    main()
