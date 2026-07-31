"""Simple LangSmith diagnostic helpers."""

from __future__ import annotations

import os

from langsmith import Client


def test_basic_langsmith() -> None:
    """Test basic LangSmith connectivity."""
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        print("❌ No LANGSMITH_API_KEY found")
        return

    print(f"✅ API Key found: {api_key[:10]}...")

    try:
        client = Client(api_key=api_key)
        print("✅ Client created successfully")

        projects = list(client.list_projects())
        print(f"✅ Found {len(projects)} projects")
        for project in projects:
            print(f"   - {project.name}")

        run = client.create_run(
            name="test_run",
            project_name="spider-guardian-bot",
            inputs={"test": "input"},
            outputs={"test": "output"},
            run_type="llm",
        )
        print(f"✅ Created test run: {run}")
        if hasattr(run, "id"):
            print(f"   Run ID: {run.id}")

    except Exception as exc:  # pragma: no cover - diagnostic helper
        print(f"❌ Error: {exc}")
        import traceback

        traceback.print_exc()


__all__ = ["test_basic_langsmith"]


if __name__ == "__main__":
    test_basic_langsmith()
