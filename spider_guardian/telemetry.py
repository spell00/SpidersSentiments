"""Optional telemetry integrations (LangSmith and Neptune).

This module keeps integrations optional and no-op by default. When users enable
flags in the CLI, we check environment variables and either initialise the
integration or print concise instructions on how to set it up in PowerShell.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


def _pwsh_hint(k: str, v: str = "<value>") -> str:
    return f"$env:{k}='{v}'"


@dataclass
class _TelemetryState:
    langsmith_enabled: bool = False
    neptune_run: Any = None  # type: ignore


class Telemetry:
    def __init__(self) -> None:
        self.state = _TelemetryState()

    # --------------------- LangSmith ---------------------
    def enable_langsmith(self, enable: bool, project: Optional[str] = None) -> bool:
        if not enable:
            self.state.langsmith_enabled = False
            return False

        api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
        if api_key:
            # Bridge old/new env names so downstream tooling sees a value either way.
            os.environ.setdefault("LANGCHAIN_API_KEY", api_key)
            os.environ.setdefault("LANGSMITH_API_KEY", api_key)

        project_env = (
            os.getenv("LANGCHAIN_PROJECT")
            or os.getenv("LANGSMITH_PROJECT")
            or project
        )
        if project_env:
            os.environ["LANGCHAIN_PROJECT"] = project_env
            os.environ.setdefault("LANGSMITH_PROJECT", project_env)

        # Honor custom LangSmith endpoints while keeping legacy vars in sync.
        endpoint = (
            os.getenv("LANGSMITH_API_URL")
            or os.getenv("LANGCHAIN_ENDPOINT")
            or os.getenv("LANGCHAIN_BASE_URL")
        )
        if endpoint:
            os.environ.setdefault("LANGCHAIN_ENDPOINT", endpoint)
            os.environ.setdefault("LANGCHAIN_BASE_URL", endpoint)

        if os.getenv("LANGCHAIN_TRACING_V2") != "true":
            # Enable tracing for this process; user keeps their shell persistent choice.
            os.environ["LANGCHAIN_TRACING_V2"] = "true"

        missing = []
        if not api_key:
            missing.append("LANGSMITH_API_KEY (or LANGCHAIN_API_KEY)")
        if not project_env:
            missing.append("LANGSMITH_PROJECT (or LANGCHAIN_PROJECT)")

        if missing:
            logging.info(
                "LangSmith not initialised. Missing: %s. To enable, set: %s and %s (and optionally %s)",
                ", ".join(missing),
                _pwsh_hint("LANGSMITH_API_KEY", "sk-..."),
                _pwsh_hint("LANGSMITH_PROJECT", "my-project"),
                _pwsh_hint("LANGCHAIN_TRACING_V2", "true"),
            )
            self.state.langsmith_enabled = False
            return False

        logging.info("LangSmith tracing is enabled for project=%s", os.getenv("LANGCHAIN_PROJECT"))
        self.state.langsmith_enabled = True
        return True

    # --------------------- Neptune ----------------------
    def enable_neptune(self, enable: bool, project: Optional[str] = None, tags: Optional[list[str]] = None) -> bool:
        if not enable:
            self.state.neptune_run = None
            return False
        try:
            import neptune  # type: ignore
        except Exception:
            logging.info(
                "Neptune package not installed. To enable, add dependency 'neptune' and set %s and %s",
                "NEPTUNE_API_TOKEN",
                "NEPTUNE_PROJECT",
            )
            self.state.neptune_run = None
            return False
        api_token = os.getenv("NEPTUNE_API_TOKEN")
        project_env = os.getenv("NEPTUNE_PROJECT") or project
        if not api_token or not project_env:
            pairs = [("NEPTUNE_API_TOKEN", api_token), ("NEPTUNE_PROJECT", project_env)]
            missing = [k for (k, v) in pairs if not v]
            logging.info(
                "Neptune not initialised. Missing: %s. To enable, set: %s and %s",
                ", ".join(missing),
                _pwsh_hint("NEPTUNE_API_TOKEN"),
                _pwsh_hint("NEPTUNE_PROJECT", "workspace/project"),
            )
            self.state.neptune_run = None
            return False
        try:
            run = neptune.init_run(project=project_env)
            if tags:
                run["sys/tags"].add(tags)
            self.state.neptune_run = run
            logging.info("Neptune run initialised for project=%s", project_env)
            return True
        except Exception as exc:
            logging.info("Neptune initialisation failed: %s", exc)
            self.state.neptune_run = None
            return False

    # --------------------- Logging helpers ---------------
    def log_generation(self, meta: Dict[str, Any]) -> None:
        """Record a generation event (provider/model/prompt_len/output_len/latency_ms/success)."""
        run = self.state.neptune_run
        if run is None:
            return
        try:
            # Use series logging for quick charts
            for key in ("latency_ms", "prompt_len", "output_len"):
                if key in meta and isinstance(meta[key], (int, float)):
                    run[f"metrics/{key}"].append(meta[key])
            # Log metadata as text fields
            for key in ("provider", "model", "tone", "result"):
                if key in meta and meta[key] is not None:
                    run[f"context/{key}"] = str(meta[key])
        except Exception:
            pass


# Global singleton
telemetry = Telemetry()
