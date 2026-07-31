"""Spider Guardian package.

Early, safe configuration for headless environments.
"""

# Prevent Tkinter-related crashes in headless/background runs by forcing a non-GUI
# matplotlib backend before any potential pyplot import occurs.
try:
    import os
    if not os.getenv("DISPLAY") and not os.getenv("FORCE_GUI") and os.getenv("MPLBACKEND", "").lower() != "agg":
        import matplotlib
        # Only switch if a GUI backend might be selected
        try:
            current = matplotlib.get_backend().lower()
        except Exception:
            current = ""
        if current and current not in ("agg", "svg", "pdf", "ps", "cairo"):
            matplotlib.use("Agg")
        elif not current:
            # Backend not initialised yet; set to Agg
            matplotlib.use("Agg")
except Exception:
    # Never block package import due to plotting config
    pass

from .config import ChatProviderConfig, SpiderGuardianConfig


def __getattr__(name: str):
    if name == "SpiderGuardianBot":
        from .bot import SpiderGuardianBot

        return SpiderGuardianBot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "SpiderGuardianConfig",
    "ChatProviderConfig",
    "SpiderGuardianBot",
]
