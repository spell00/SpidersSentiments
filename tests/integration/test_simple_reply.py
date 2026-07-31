#!/usr/bin/env python3
"""
Simplified test to debug just the reply posting functionality.
"""

import logging
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from spider_guardian.config import SpiderGuardianConfig
from spider_guardian.twitter_client import SeleniumTwitterClient

logging.basicConfig(level=logging.DEBUG)


def test_simple_reply():
    """Test if we can post a simple reply."""

    # Create config with Firefox
    config = SpiderGuardianConfig()
    config.selenium_driver = "firefox"
    config.selenium_headless = False  # Keep visible for debugging

    # Create Twitter client
    client = SeleniumTwitterClient(config)

    try:
        # Test reply to a specific tweet
        test_tweet_id = "1980892987367518352"  # From our database

        print(f"🔍 Testing reply to tweet {test_tweet_id}")

        # Post a simple test reply
        reply_id = client.reply("Test reply - debugging button", reply_to_tweet_id=test_tweet_id)

        if reply_id:
            print(f"✅ Successfully posted reply! ID: {reply_id}")
        else:
            print("❌ Failed to post reply")

    except Exception as e:
        print(f"❌ Error during reply: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Keep browser open for inspection
        print("⏱️ Keeping browser open for 60 seconds...")
        time.sleep(60)
        client.driver.quit()


if __name__ == "__main__":
    test_simple_reply()
