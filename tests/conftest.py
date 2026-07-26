"""Test isolation: no `.env`, no cached settings, no leaked logger state.

Tests must not depend on the developer's local `.env` or exported shell variables — a
suite that passes only on a machine with a key is worse than no suite. Everything a test
needs comes from `monkeypatch.setenv` or an explicit mapping.
"""

import gzip
import json
import logging
import os
from pathlib import Path

import pytest

from finbrief import config
from finbrief.ingestion.model import ExtractedFiling, FilingRef, Section
from finbrief.observability.logging_setup import PACKAGE_LOGGER

#: Real EDGAR extractions, recorded by `scripts/record_edgar_fixtures.py`. Recorded rather
#: than fetched because the suite is hermetic by contract (CLAUDE.md) — and recorded rather
#: than hand-written because the failures worth testing (a bank's 400k MD&A, six different
#: wordings of "incorporated by reference", an Item 7/8 boundary miss) are ones nobody
#: would think to invent.
FIXTURES = Path(__file__).parent / "fixtures" / "edgar"

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


@pytest.fixture(scope="session")
def recorded_filing() -> ExtractedFiling:
    """Apple's real FY2025 10-K, as extracted — all four Sections, verbatim."""
    raw = json.loads(gzip.decompress((FIXTURES / "aapl-fy2025.json.gz").read_bytes()))
    return ExtractedFiling(
        ref=FilingRef(**raw["ref"]),
        latest_annual_form=raw["latest_annual_form"],
        sections={Section(key): text for key, text in raw["sections"].items()},
    )


@pytest.fixture(scope="session")
def recorded_sections() -> dict[str, str]:
    """Individual real Section texts that broke something: pointers and a boundary miss."""
    return json.loads((FIXTURES / "section-samples.json").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def pristine_package_logger():
    """Restore the `finbrief` logger, so `configure_logging` in one test can't mute another."""
    logger = logging.getLogger(PACKAGE_LOGGER)
    handlers, level, propagate = logger.handlers[:], logger.level, logger.propagate
    yield
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate
