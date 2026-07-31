"""Open or print LangSmith dashboard details."""

from __future__ import annotations

import os
import webbrowser


PROJECT_URL = "https://smith.langchain.com/projects/spider-guardian-bot"


def open_langsmith_dashboard() -> None:
    """Open the LangSmith dashboard in the default browser."""
    if not os.getenv("LANGSMITH_API_KEY"):
        print("❌ LANGSMITH_API_KEY not set")
        print("Set your API key with: $env:LANGSMITH_API_KEY='your-key'")
        return

    print("🌐 Opening LangSmith dashboard...")
    print(f"📊 Project URL: {PROJECT_URL}")

    try:
        webbrowser.open(PROJECT_URL)
        print("✅ Dashboard opened in your default browser")
    except Exception as exc:
        print(f"❌ Failed to open browser: {exc}")
        print(f"Please manually visit: {PROJECT_URL}")


def print_langsmith_info() -> None:
    """Print LangSmith setup information."""
    print("📈 LangSmith Integration for Spider Guardian Bot")
    print("=" * 50)

    if os.getenv("LANGSMITH_API_KEY"):
        print("✅ LangSmith API Key: Configured")
        print("✅ Project: spider-guardian-bot")
        print(f"📊 Dashboard: {PROJECT_URL}")
        print("\n🔍 What you can track:")
        print("  • Reply generation performance")
        print("  • Sentiment analysis results")
        print("  • Engagement metrics (likes, replies, impressions)")
        print("  • Generation latency and success rates")
        print("\n🚀 To see data:")
        print("  1. Run your bot with: python -m spider_guardian --respond 1")
        print("  2. Wait for it to generate replies")
        print("  3. Check the dashboard for real-time traces")
    else:
        print("❌ LangSmith API Key: Not configured")
        print("\n🔧 Setup instructions:")
        print("  1. Sign up at https://smith.langchain.com/")
        print("  2. Get your API key from settings")
        print("  3. Set environment variable:")
        print("     PowerShell: $env:LANGSMITH_API_KEY='your-key'")
        print("     CMD: set LANGSMITH_API_KEY=your-key")
        print("  4. Restart your application")


__all__ = ["open_langsmith_dashboard", "print_langsmith_info"]


if __name__ == "__main__":
    print_langsmith_info()
    print("\n" + "=" * 50)

    if os.getenv("LANGSMITH_API_KEY"):
        choice = input("Open dashboard in browser? (y/N): ").strip().lower()
        if choice in {"y", "yes"}:
            open_langsmith_dashboard()
    else:
        print("\nConfigure your API key first, then run this script again.")
