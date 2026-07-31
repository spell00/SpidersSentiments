import sys
import types
import logging
import pytest

from spider_guardian import ml_bot
from spider_guardian.config import SpiderGuardianConfig


def test_ml_bot_train_cli(monkeypatch, capsys):
    # Stub out the heavy training call
    class DummyPipeline:
        def __init__(self, cfg):
            self.cfg = cfg
        def train_all_models(self, min_replies=50, min_trending=20):
            return {
                "quality_predictor": {"trained": False, "samples": 0},
                "popularity_analysis": {"analyzed": False, "samples": 0},
                "rl_stats": {"strategies": {}, "recommendations": []},
            }

    monkeypatch.setattr(ml_bot, "MLTrainingPipeline", DummyPipeline)

    # Simulate CLI args for training
    argv = ["ml_bot.py", "train", "--min-replies", "1", "--min-trending", "1"]
    old_argv = sys.argv[:]
    sys.argv = argv
    try:
        ml_bot.main()
    finally:
        sys.argv = old_argv

    out = capsys.readouterr().out
    assert "ML TRAINING SUMMARY" in out
