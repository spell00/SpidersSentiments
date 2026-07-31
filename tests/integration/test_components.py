#!/usr/bin/env python3
"""Minimal test to identify where the hanging occurs."""

import logging
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def test_components():
    """Test each component individually to find the hanging point."""

    print("🔍 Testing individual components...")

    try:
        print("1️⃣ Testing basic imports...")
        from spider_guardian.config import SpiderGuardianConfig
        print("   ✅ Config import OK")

        from spider_guardian.bot import SpiderGuardianBot
        print("   ✅ Bot import OK")

        print("2️⃣ Testing config creation...")
        config = SpiderGuardianConfig()
        # Disable heavy models for testing
        config.providers = []
        print("   ✅ Config creation OK")

        print("3️⃣ Testing bot initialization (without models)...")
        # Add a dummy provider to bypass the check
        from spider_guardian.providers import ProviderConfig
        config.providers = [ProviderConfig(name="dummy", model="test")]

        # This should be fast without sentence transformers
        try:
            bot = SpiderGuardianBot(config)
            print("   ✅ Bot initialization OK")
        except RuntimeError as e:
            if "No providers" in str(e):
                print("   ⚠️ Provider issue, but bot structure OK")
                # Create bot without provider validation for testing
                import spider_guardian.bot

                old_init = spider_guardian.bot.SpiderGuardianBot.__init__

                def mock_init(self, config):
                    self.config = config
                    self.twitter_client = None

                spider_guardian.bot.SpiderGuardianBot.__init__ = mock_init
                bot = SpiderGuardianBot(config)
                spider_guardian.bot.SpiderGuardianBot.__init__ = old_init
                print("   ✅ Bot mock initialization OK")
            else:
                raise

        print("4️⃣ Testing Twitter client initialization...")
        bot.ensure_twitter_client()
        print("   ✅ Twitter client OK")

        print("5️⃣ Testing single search (no replies)...")
        tweets = bot.twitter_client.search_posts("spider", limit=1)
        print(f"   ✅ Found {len(tweets)} tweets")

        print("\n✅ All components working! The issue is likely in:")
        print("   - Model loading (sentence transformers)")
        print("   - Reply generation")
        print("   - Selenium interactions")

    except Exception as e:
        print(f"\n❌ Component test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_components()
