"""Simplified LangSmith integration helpers for Spider Guardian."""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from typing import Dict, Optional

# Ensure UTF-8 encoding for stdout on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python < 3.7 doesn't have reconfigure
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

try:
    from langsmith import Client, traceable
except ImportError:  # pragma: no cover - optional integration dependency
    Client = None  # type: ignore[assignment]

    def traceable(*args, **kwargs):  # type: ignore[no-redef]
        def decorator(func):
            return func
        return decorator

PROJECT_NAME = os.getenv("LANGSMITH_PROJECT", "spider-guardian-bot")


def _project_url(project: str) -> Optional[str]:
    if Client is None:
        return None
    try:
        client = Client()
        proj = client.read_project(project_name=project)
    except Exception:
        return None

    org_id = getattr(proj, "organization_id", None)
    project_id = getattr(proj, "id", None)
    if org_id and project_id:
        return f"https://smith.langchain.com/o/{org_id}/projects/p/{project_id}"
    if project_id:
        return f"https://smith.langchain.com/projects/{project_id}"
    return None


if os.getenv("LANGSMITH_API_KEY"):
    os.environ.setdefault("LANGSMITH_PROJECT", PROJECT_NAME)
    print(f"✅ LangSmith configured for project: {PROJECT_NAME}")
    resolved = _project_url(PROJECT_NAME)
    if resolved:
        print(f"🌐 View at: {resolved}")
    else:
        print(f"🌐 View at: https://smith.langchain.com/projects/{PROJECT_NAME}")
else:
    print("⚠️ LANGSMITH_API_KEY not set - LangSmith tracking disabled")


@traceable(name="spider_reply_generation", project_name=PROJECT_NAME)
def log_reply_generation(
    original_tweet: str,
    generated_reply: str,
    prompt: str,
    model_name: str,
    generation_time_ms: int,
    metadata: Optional[Dict] = None,
) -> Dict:
    """Log reply generation with LangSmith tracing."""
    # Argument validation and debug logging
    errors = []
    if not original_tweet or not isinstance(original_tweet, str):
        errors.append("original_tweet is empty or not a string")
    if not generated_reply or not isinstance(generated_reply, str):
        errors.append("generated_reply is empty or not a string")
    if not prompt or not isinstance(prompt, str):
        errors.append("prompt is empty or not a string")
    if not model_name or not isinstance(model_name, str):
        errors.append("model_name is empty or not a string")
    if generation_time_ms is None or not isinstance(generation_time_ms, (int, float)):
        generation_time_ms = 0
    if errors:
        print(f"⚠️ log_reply_generation called with invalid arguments: {errors}")
    else:
        print(f"✅ log_reply_generation called with valid arguments: tweet='{original_tweet[:40]}', reply='{generated_reply[:40]}', model='{model_name}', gen_time={generation_time_ms}")
    return {
        "original_tweet": original_tweet,
        "generated_reply": generated_reply,
        "model": model_name,
        "generation_time_ms": generation_time_ms,
        "metadata": metadata or {},
        "timestamp": datetime.now().isoformat(),
        "validation_errors": errors,
    }


@traceable(name="spider_engagement_metrics", project_name=PROJECT_NAME)
def log_engagement_metrics(
    reply_text: str,
    likes: int,
    replies: int,
    impressions: int,
    tweet_id: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> Dict:
    """Log engagement metrics with LangSmith tracing."""
    errors = []
    if not reply_text or not isinstance(reply_text, str):
        errors.append("reply_text is empty or not a string")
    if not isinstance(likes, int):
        errors.append("likes is not an int")
    if not isinstance(replies, int):
        errors.append("replies is not an int")
    if not isinstance(impressions, int):
        errors.append("impressions is not an int")
    if tweet_id is not None and not isinstance(tweet_id, str):
        errors.append("tweet_id is not a string")
    if errors:
        print(f"⚠️ log_engagement_metrics called with invalid arguments: {errors}")
    else:
        print(f"✅ log_engagement_metrics called with valid arguments: reply='{reply_text[:40]}', likes={likes}, replies={replies}, impressions={impressions}, tweet_id={tweet_id}")
    return {
        "reply_text": reply_text,
        "likes": likes,
        "replies": replies,
        "impressions": impressions,
        "engagement_rate": (likes + replies) / max(impressions, 1),
        "timestamp": datetime.now().isoformat(),
        "tweet_id": tweet_id,
        "metadata": metadata or {},
        "validation_errors": errors,
    }


@traceable(name="spider_sentiment_analysis", project_name=PROJECT_NAME)
def log_sentiment_analysis(text: str, sentiment_scores: Dict) -> Dict:
    """Log sentiment analysis results."""
    errors = []
    if not text or not isinstance(text, str):
        errors.append("text is empty or not a string")
    if not isinstance(sentiment_scores, dict) or not sentiment_scores:
        errors.append("sentiment_scores is not a non-empty dict")
    if errors:
        print(f"⚠️ log_sentiment_analysis called with invalid arguments: {errors}")
    else:
        print(f"✅ log_sentiment_analysis called with valid arguments: text='{text[:40]}', scores={sentiment_scores}")
    return {
        "text": text,
        "sentiment_scores": sentiment_scores,
        "dominant_sentiment": max(sentiment_scores, key=sentiment_scores.get) if sentiment_scores else None,
        "timestamp": datetime.now().isoformat(),
        "validation_errors": errors,
    }


def test_tracing() -> None:
    """Test the tracing helpers."""
    print("🧪 Testing LangSmith tracing...")

    _ = log_reply_generation(
        original_tweet="Found a spider in my house! What should I do?",
        generated_reply="Most house spiders are harmless! Just gently capture it with a glass and paper and release it outside.",
        prompt="Generate helpful spider advice",
        model_name="test-model",
        generation_time_ms=1200,
        metadata={"test": True},
    )
    print("✅ Logged reply generation")

    _ = log_engagement_metrics(
        reply_text="Most house spiders are harmless!",
        likes=12,
        replies=2,
        impressions=150,
    )
    print("✅ Logged engagement metrics")

    _ = log_sentiment_analysis(
        text="That's a beautiful spider!",
        sentiment_scores={"positive": 0.8, "negative": 0.1, "neutral": 0.1},
    )
    print("✅ Logged sentiment analysis")


    resolved = _project_url(PROJECT_NAME)
    if resolved:
        print(f"🌐 View traces at: {resolved}")
    else:
        print(f"🌐 View traces at: https://smith.langchain.com/projects/{PROJECT_NAME}")



def push_reply_to_dataset(
    tweet_text: str,
    author: str,
    url: str,
    generated_reply: str,
    likes: int = 0,
    replies: int = 0,
    impressions: int = 0,
    metadata: Optional[Dict] = None,
    dataset_name: str = "spider-replies-dataset"
) -> None:
    """Push a real bot reply to the trending-dataset."""
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        print("⚠️ LANGSMITH_API_KEY not set - cannot push to dataset.")
        return
    try:
        client = Client(api_key=api_key)
        # Create dataset if not exists
        try:
            client.create_dataset(
                dataset_name=dataset_name,
                description="Spider-related replies from bot."
            )
        except Exception:
            pass
        example = {
            "inputs": {
                "tweet_text": tweet_text,
                "author": author,
                "url": url
            },
            "outputs": {
                "generated_reply": generated_reply,
                "engagement_metrics": {
                    "likes": likes,
                    "replies": replies,
                    "impressions": impressions
                }
            },
            "metadata": metadata or {"timestamp": datetime.now().isoformat()}
        }
        client.create_examples(
            inputs=[example["inputs"]],
            outputs=[example["outputs"]],
            metadata=[example["metadata"]],
            dataset_name=dataset_name
        )
        print(f"✅ Bot reply pushed to dataset '{dataset_name}'")
    except Exception as e:
        print(f"❌ Failed to push bot reply to dataset: {e}")


__all__ = [
    "log_reply_generation",
    "log_engagement_metrics",
    "log_sentiment_analysis",
    "test_tracing",
]


if __name__ == "__main__":
    test_tracing()
    # Demo: push a bot reply to the dataset
    push_reply_to_dataset(
        tweet_text="Why do spiders build webs?",
        author="demo_user",
        url="https://twitter.com/demo_user/status/456",
        generated_reply="Spiders build webs to catch prey and protect themselves.",
        likes=10,
        replies=2,
        impressions=200,
        metadata={"source": "demo", "test_case": "main_call"}
    )
