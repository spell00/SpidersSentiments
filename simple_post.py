#!/usr/bin/env python3
"""Simple tweet posting script without heavy ML dependencies."""

import argparse
import datetime
import logging
import os
import random
import sys
import time
from typing import Optional

# Add the spider_guardian package to path
sys.path.insert(0, os.path.dirname(__file__))

def generate_spider_content(debug_model: bool = False) -> str:
    """Generate educational spider content using lightweight methods."""

    def debug_print(message: str) -> None:
        if debug_model:
            print(message)
    
    # Inspirational spider facts (fallbacks kept uplifting)
    spider_facts = [
        "Every web a spider spins is a sunrise for the ecosystem, quietly protecting harvests and homes around the world. 🕷️�",
        "When a spider rebuilds its web after the storm, it reminds us that resilience can be delicate and still unstoppable. 🕷️✨",
        "Spider silk is lighter than feather and stronger than steel, proving that nature's greatest power can feel like grace. 🕷️💪",
        "Jumping spiders see in vibrant color, mapping a world of possibilities six inches ahead—proof that vision drives courage. 👀🕷️",
        "A single patient spider can erase thousands of pests, showing how quiet guardianship keeps the planet in balance. 🕷️🌍",
        "Ballooning spiders trust the wind with miles of open sky; let their courage remind you to launch bold ideas. 🕷️🎈",
        "Peacock spiders dance with brilliant confidence, a tiny celebration that joy and science can move hearts together. 💃🕷️🌈",
        "Wolf spiders carry their young across every terrain, a lesson in protection and love that fits in the palm of a hand. 🕷️👶",
        "Even an everyday house spider is a partner in wellbeing, keeping the air clearer for sleepy midnight dreamers. 🏠🕷️"
    ]
    
    try:
        # Try to use a simple AI provider if available
        print("🔧 Starting AI generation...")
        
        from spider_guardian.bot import SpiderGuardianBot
        from spider_guardian.config import ChatProviderConfig, SpiderGuardianConfig
        
        print("🔧 Creating provider config...")
        # Create minimal config for AI generation  
        provider_config = ChatProviderConfig(
            name="local",
            model="Open-Orca/Mistral-7B-OpenOrca",  # Use Open-Orca like CLI
            temperature=0.6,  # Less creative for more factual content
            timeout=600  # Longer timeout for larger model
        )
        
        print("🔧 Creating guardian config...")
        config = SpiderGuardianConfig()
        config.providers = (provider_config,)
        config.device = "cpu"
        
        print("🔧 Creating SpiderGuardianBot like CLI does...")
        try:
            bot = SpiderGuardianBot(config)
            providers = bot.providers  # Get providers from bot
            print(f"🔧 Bot created with {len(providers) if providers else 0} providers")
            if providers:
                provider = providers[0]
                generator = getattr(provider, "generator", None)
                if generator is not None:
                    model = getattr(generator, "model", None)
                    tokenizer = getattr(generator, "tokenizer", None)
                    debug_print(f"🔍 Generator type: {type(generator).__name__}")
                    if model is not None:
                        debug_print(f"🔍 Model class: {model.__class__.__name__}")
                        device = getattr(model, "device", None)
                        debug_print(f"🔍 Model device: {device}")
                        device_map = getattr(model, "hf_device_map", None)
                        if device_map:
                            debug_print(f"🔍 hf_device_map keys: {list(device_map.keys())[:10]}")
                            offload_count = sum(1 for v in device_map.values() if isinstance(v, str) and v == "cpu")
                            debug_print(f"🔍 hf_device_map cpu shards: {offload_count}")
                    if tokenizer is not None:
                        debug_print(f"🔍 Tokenizer class: {tokenizer.__class__.__name__}")
        except Exception as provider_error:
            print(f"❌ Bot initialization failed: {provider_error}")
            raise provider_error
        
        if providers:
            prompts = [
                "Craft an inspiring, factual (<=220 chars) celebrating why spiders uplift our planet:",
                "Share a verified spider fact that feels motivational and hopeful; keep it concise",
                "Write a short, awe-filled spider insight that encourages curiosity and respect; stay under 220 characters:",
                "Create an uplifting spider fact that sparks wonder and gratitude; deliver it as a complete tweet with emoji:"
            ]
            
            prompt = random.choice(prompts)
            print(f"🔧 Generating with prompt: {prompt[:50]}...")
            try:
                generated = providers[0].generate(prompt)
            except Exception as gen_exc:
                if debug_model:
                    import traceback
                    print(f"💥 Generation threw: {gen_exc}")
                    print("🔍 Generation traceback:")
                    print(traceback.format_exc())
                raise
            debug_print(f"🔧 Raw generated: '{generated}'")
            
            if generated and len(generated.strip()) > 20 and len(generated.strip()) <= 280:
                # Clean up the generated text
                cleaned = generated.strip()

                if len(cleaned) > 260:
                    debug_print(f"🔧 Rejected because length {len(cleaned)} risks truncation")
                    cleaned = None
                
                # Check for incomplete sentences (common AI issue)
                if cleaned.endswith(('thanks to', 'because of', 'due to', 'such as', 'like', 'including', 'which', 'that', 'and', 'or', 'but')):
                    debug_print(f"🔧 Rejected incomplete sentence ending with: '{cleaned[-20:]}'")
                    cleaned = None
                
                # Check for obvious factual errors
                if cleaned and any(word in cleaned.lower() for word in ['20-foot', 'giant', 'massive spider', 'enormous', 'huge spider']):
                    debug_print("🔧 Rejected likely inaccurate content about spider size")
                    cleaned = None
                
                if cleaned:
                    # Ensure it's a complete sentence
                    if not cleaned.endswith(('!', '.', '?')):
                        cleaned += "!"
                        
                    # Add spider emoji if not present
                    if "🕷️" not in cleaned:
                        cleaned += " 🕷️"
                        
                    print("✨ Generated AI content!")
                    return cleaned
                    
            debug_print(f"🔧 Generated content rejected: length={len(generated.strip()) if generated else 0}, incomplete or inaccurate")
                
    except Exception as e:
        import traceback
        print(f"⚠️ AI generation failed: {e}")
        print(f"🔍 Full traceback: {traceback.format_exc()}")
        print("📚 Falling back to curated content")
    
    # Fallback to curated facts
    selected = random.choice(spider_facts)
    print("📚 Using curated educational content")
    return selected

def post_tweet(text: str, headless: bool = True) -> bool:
    """Post a tweet using minimal dependencies."""
    try:
        from spider_guardian.twitter_client import SeleniumTwitterClient
        from spider_guardian.config import SpiderGuardianConfig
        
        # Create minimal config
        config = SpiderGuardianConfig(
            dataset_path="",
            embedder_name="",
            providers=[],
            selenium_headless=headless,
            selenium_driver="firefox",
            selenium_wait_seconds=10
        )
        
        # Initialize Twitter client
        client = SeleniumTwitterClient(config)
        
        # Post the tweet
        tweet_id = client.post_tweet(text)
        
        # Check if posting was successful
        # tweet_id might be None even if posting succeeded (Twitter redirects to home)
        # So we check for success confirmation or valid URL state
        current_url = client.driver.current_url if hasattr(client, 'driver') else ""
        
        if tweet_id:
            print(f"✅ Tweet posted successfully!")
            print(f"Tweet ID: {tweet_id}")
            return True
        elif current_url and "x.com/home" in current_url:
            print(f"✅ Tweet posted successfully!")
            print("(Successfully redirected to home page)")
            return True
        else:
            print("❌ Failed to post tweet")
            return False
            
    except Exception as e:
        import traceback
        print(f"❌ Error posting tweet: {e}")
        print(f"🔍 Traceback: {traceback.format_exc()}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Simple tweet poster")
    parser.add_argument("text", nargs="?", help="Tweet text to post (optional if using --auto)")
    parser.add_argument("--auto", action="store_true", help="Auto-generate spider educational content")
    parser.add_argument("--visible", action="store_true", help="Show browser window")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--log-level", default=None, help="Logging level (e.g. DEBUG, INFO, WARNING)")
    parser.add_argument("--debug-model", action="store_true", help="Print detailed model diagnostics during generation")
    parser.add_argument("--loop", action="store_true", help="Run continuously with Poisson-distributed intervals")
    parser.add_argument(
        "--mean-posts-per-day",
        type=float,
        default=1.5,
        help="Average number of posts per 24 hours when running in loop mode",
    )
    parser.add_argument(
        "--max-wait-hours",
        type=float,
        default=24.0,
        help="Maximum wait time between posts in hours when running in loop mode (set 0 to disable)",
    )
    
    args = parser.parse_args()
    
    try:
        log_level: Optional[str] = None
        if args.log_level:
            log_level = args.log_level.upper()
        if args.debug:
            log_level = "DEBUG"
        if log_level:
            logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
        
        def format_duration(seconds: float) -> str:
            seconds = int(max(0, seconds))
            parts: list[str] = []
            days, rem = divmod(seconds, 86400)
            hours, rem = divmod(rem, 3600)
            minutes, secs = divmod(rem, 60)
            if days:
                parts.append(f"{days}d")
            if hours:
                parts.append(f"{hours}h")
            if minutes:
                parts.append(f"{minutes}m")
            if secs or not parts:
                parts.append(f"{secs}s")
            return " ".join(parts)

        def run_single_post() -> bool:
            if args.auto:
                print("🤖 Generating educational spider content...")
                text_local = generate_spider_content(debug_model=args.debug_model or log_level == "DEBUG")
                print(f"📝 Generated: {text_local}")
            elif args.text:
                text_local = args.text
            else:
                print("❌ Please provide text to post or use --auto for generated content")
                return False

            if len(text_local.strip()) == 0:
                print("❌ Cannot post empty text")
                return False

            if len(text_local) > 280:
                print(f"❌ Tweet too long: {len(text_local)} characters (max 280)")
                return False

            print(f"🚀 Posting: {text_local}")
            return post_tweet(text_local, headless=not args.visible)

        if args.loop:
            if not args.auto:
                print("❌ Loop mode requires --auto to generate fresh content each cycle")
                return 1

            mean_per_day = max(args.mean_posts_per_day, 0.05)
            rate_per_second = mean_per_day / 86400.0
            max_wait_seconds: Optional[float]
            if args.max_wait_hours and args.max_wait_hours > 0:
                max_wait_seconds = max(args.max_wait_hours, 0.1) * 3600.0
            else:
                max_wait_seconds = None
            print("🔁 Loop mode enabled. Press Ctrl+C to stop.")
            while True:
                success = run_single_post()
                if success:
                    print("✅ Post completed")
                else:
                    print("⚠️ Post attempt failed; will retry after the next interval")

                wait_seconds = random.expovariate(rate_per_second) if rate_per_second > 0 else 86400.0
                wait_seconds = max(wait_seconds, 300.0)
                if max_wait_seconds is not None and wait_seconds > max_wait_seconds:
                    print(
                        f"🎯 Drawn wait of {format_duration(wait_seconds)} exceeds cap; capping to {format_duration(max_wait_seconds)}"
                    )
                    wait_seconds = max_wait_seconds
                    if wait_seconds < 300.0:
                        wait_seconds = 300.0
                next_run = datetime.datetime.now() + datetime.timedelta(seconds=wait_seconds)
                print(
                    f"⏱️ Next post in {format_duration(wait_seconds)} (≈ {next_run:%Y-%m-%d %H:%M:%S})"
                )
                try:
                    time.sleep(wait_seconds)
                except KeyboardInterrupt:
                    print("🛑 Loop interrupted by user")
                    return 0
        else:
            success = run_single_post()
            return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⚠️ Script interrupted by user")
        return 130
    except Exception as e:
        import traceback
        print(f"\n💥 CRASH DETECTED!")
        print(f"❌ Error: {e}")
        print(f"🔍 Location: {traceback.format_exc()}")
        print("\nPlease report this error if it persists.")
        return 1

if __name__ == "__main__":
    sys.exit(main())