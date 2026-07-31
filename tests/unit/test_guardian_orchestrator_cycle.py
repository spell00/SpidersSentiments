import types
import logging

from spider_guardian.scripts.guardian_orchestrator import GuardianOrchestrator, AutoPostOptions, build_config


class DummyBot:
    def __init__(self):
        self.calls = {"respond_to_tweets": 0, "collect_and_learn": 0, "collect_trending": 0}
        self.twitter_client = None

    def respond_to_tweets(self, *args, **kwargs):
        self.calls["respond_to_tweets"] += 1
        return 0

    def collect_and_learn(self):
        self.calls["collect_and_learn"] += 1

    def collect_trending(self, *args, **kwargs):
        self.calls["collect_trending"] += 1
        return 0


def test_run_cycle_dry_run(monkeypatch, caplog):
    caplog.set_level(logging.INFO)

    config = build_config(types.SimpleNamespace(show_browser=False, selenium_driver=None, chat_model=None, provider="local", temperature=0.5))
    autopost = AutoPostOptions()

    orch = GuardianOrchestrator(
        config=config,
        timezone="UTC",
        autopost=autopost,
        cycle_mean_minutes=60,
        cycle_std_minutes=10,
        cycle_min_minutes=30,
        cycle_max_minutes=120,
        reply_limit_min=0,
        reply_limit_max=0,
        followup_conversations=1,
        followup_replies=0,
        trending_hours=1,
        trending_retention_days=1,
        dry_run=True,
    )

    # Inject dummy bot and force not-ready twitter to skip any posting
    dummy = DummyBot()
    orch.bot = dummy
    orch.twitter_ready = False

    orch._run_cycle()

    # Should have performed learning + trending steps even in dry-run
    assert dummy.calls["collect_and_learn"] == 1
    assert dummy.calls["collect_trending"] == 1

    # Ensure log contains maintenance marker
    assert any("Maintenance run complete" in rec.message for rec in caplog.records)
