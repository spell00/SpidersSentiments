import logging
from spider_guardian.bot import SpiderGuardianBot
from spider_guardian.config import SpiderGuardianConfig

def debug_send_tweet():
    # Load configuration
    config = SpiderGuardianConfig.load("config.yaml")
    bot = SpiderGuardianBot(config)

    # Ensure Twitter client is initialized
    bot.ensure_twitter_client()

    # Example tweet content
    tweet_text = "This is a debug tweet from SpiderGuardianBot."

    try:
        # Send the tweet
        tweet_id = bot.twitter_client.tweet(tweet_text)
        logging.info(f"Tweet sent successfully with ID: {tweet_id}")

        # Log the tweet to my-tweets-dataset
        bot.log_my_tweet(
            text=tweet_text,
            tweet_id=str(tweet_id),
            url=f"https://x.com/i/status/{tweet_id}",
            likes=0,
            replies=0,
            impressions=0
        )
        logging.info("Tweet logged to my-tweets-dataset.")
    except Exception as e:
        logging.error(f"Failed to send or log tweet: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    debug_send_tweet()