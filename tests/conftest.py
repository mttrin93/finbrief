"""Test isolation: no `.env`, no cached settings, no leaked logger state.

Tests must not depend on the developer's local `.env` or exported shell variables — a
suite that passes only on a machine with a key is worse than no suite. Everything a test
needs comes from `monkeypatch.setenv` or an explicit mapping.
"""

import logging
import os

import pytest

from finbrief import config
from finbrief.observability.logging_setup import PACKAGE_LOGGER

#: `LANGCHAIN_`/`LANGSMITH_` are here for the "no network" half of the contract, not the
#: config half: with tracing exported, every LangChain invoke in the suite — the fake chat
#: models included — POSTs its run to LangSmith.
_MANAGED_PREFIXES = (
    "OPENROUTER_",
    "FINBRIEF_",
    "SEC_EDGAR_",
    "ALPHAVANTAGE_",
    "FRED_",
    "LANGCHAIN_",
    "LANGSMITH_",
)


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch):
    monkeypatch.setattr(config, "load_dotenv", lambda *args, **kwargs: None)
    for name in [n for n in os.environ if n.startswith(_MANAGED_PREFIXES) or n == "LOG_LEVEL"]:
        monkeypatch.delenv(name, raising=False)
    config.load_env.cache_clear()
    config.get_settings.cache_clear()
    yield
    config.load_env.cache_clear()
    config.get_settings.cache_clear()


@pytest.fixture(autouse=True)
def pristine_package_logger():
    """Restore the `finbrief` logger, so `configure_logging` in one test can't mute another."""
    logger = logging.getLogger(PACKAGE_LOGGER)
    handlers, level, propagate = logger.handlers[:], logger.level, logger.propagate
    yield
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate
