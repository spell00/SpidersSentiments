"""
Test script to generate sample data in LangSmith
"""
import os
import sys
import time
from datetime import datetime
import uuid

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from spider_guardian.langsmith import langsmith_integration


def test_langsmith_logging():
    """Generate some test data for LangSmith"""
    if not langsmith_integration or not langsmith_integration.client:
        print("❌ LangSmith not configured")
        return

    print("🚀 Testing LangSmith logging...")

    # Test 1: Log a sample reply generation
    test_tweet = "Just saw a huge spider in my garden! Should I be worried? 🕷️"
    test_reply = "That's probably just a harmless garden spider! They're actually beneficial for controlling pests. Most garden spiders are completely safe."
    test_prompt = f"Generate a helpful reply about spiders for: {test_tweet}"
    run_id = "test-run-id"
    run_id = str(uuid.uuid4())
    langsmith_integration.log_reply_generation(
        run_id=run_id,
        original_tweet=test_tweet,
        generated_reply=test_reply,
        prompt=test_prompt,
        model_name="test-model",
        generation_time_ms=1500,
        metadata={
            "test": True,
            "provider": "test-provider",
        },
    )

    if run_id:
        print(f"✅ Logged test reply generation: {run_id}")

        # Test 2: Log engagement metrics
        langsmith_integration.log_engagement_metrics(
            reply_id=run_id,
            likes=15,
            replies=3,
            impressions=250,
            posted_at=datetime.now(),
        )
        print("✅ Logged test engagement metrics")

        # Test 3: Add feedback
        langsmith_integration.create_feedback_run(
            run_id=run_id,
            feedback_score=0.8,
            feedback_comment="Good informative reply about spider safety",
        )
        print("✅ Added test feedback")
    else:
        print("⚠️ Reply generation logging returned None - check LangSmith connection")

    # Test 4: Generate performance report
    print("\n📊 Generating performance report...")
    report = langsmith_integration.generate_performance_report(days=1)
    if report:
        print(f"Total replies: {report['total_replies_generated']}")
        print(f"Average generation time: {report['avg_generation_time_ms']:.1f}ms")
        print(f"Total engagement: {report['total_engagement']}")

    print(f"\n🌐 View your data at: {langsmith_integration.get_langsmith_url()}")


if __name__ == "__main__":
    test_langsmith_logging()
