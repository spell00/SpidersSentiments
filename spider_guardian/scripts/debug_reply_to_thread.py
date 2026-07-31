import logging
from spider_guardian.bot import SpiderGuardianBot
from spider_guardian.config import SpiderGuardianConfig

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Load configuration
    config = SpiderGuardianConfig.load("config.json")
    bot = SpiderGuardianBot(config)

    # Example conversation ID for debugging
    conversation_id = "example_conversation_id"

    # Retrieve the thread
    thread = bot.retrieve_thread(conversation_id)
    if not thread:
        logging.warning("No thread found for conversation ID: %s", conversation_id)
    else:
        # Check if the bot is the last one to reply
        last_message = thread[-1]
        if last_message["author"] == config.bot_username:
            logging.info("Bot is the last one to reply. Skipping reply.")
        else:
            # Generate a reply
            prompt = "Respond to the latest message in the thread."
            reply = bot.generate_reply_to_thread(conversation_id, prompt)
            if reply:
                logging.info("Generated reply: %s", reply)
            else:
                logging.warning("Failed to generate a reply.")