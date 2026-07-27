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
from fakes import KeywordEmbeddings

from finbrief import config
from finbrief.ingestion.model import ExtractedFiling, FilingRef, Section
from finbrief.observability.logging_setup import PACKAGE_LOGGER

#: Real EDGAR extractions, recorded by `scripts/record_edgar_fixtures.py`. Recorded rather
#: than fetched because the suite is hermetic by contract (CLAUDE.md) — and recorded rather
#: than hand-written because the failures worth testing (a bank's 400k MD&A, six different
#: wordings of "incorporated by reference", an Item 7/8 boundary miss) are ones nobody
#: would think to invent.
FIXTURES = Path(__file__).parent / "fixtures" / "edgar"

#: cl100k_base's BPE table, vendored.
#:
#: `tiktoken.get_encoding` does **not** resolve the table from its wheel — the wheel ships
#: no data. It downloads from openaipublic.blob.core.windows.net and caches the result
#: under `TIKTOKEN_CACHE_DIR`, so a machine with a warm cache passes and clean CI silently
#: egresses. That is exactly the hermetic breach CLAUDE.md forbids, and it is invisible
#: locally, which is what makes it worth 1.6 MB in the repo.
#:
#: The filename is `sha1(url)`, which is how tiktoken looks it up.
TIKTOKEN_CACHE = Path(__file__).parent / "fixtures" / "tiktoken"
TIKTOKEN_CL100K = TIKTOKEN_CACHE / "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"

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
def offline_tiktoken(monkeypatch):
    """Point tiktoken at the vendored table, and fail loudly if it is missing.

    The assertion matters as much as the redirect. Without it a missing fixture just
    reinstates the silent download — the test would still pass, and the suite would still
    be making a network call nobody can see.
    """
    assert TIKTOKEN_CL100K.exists(), (
        f"the vendored cl100k_base table is missing from {TIKTOKEN_CACHE}. Without it "
        f"tiktoken downloads it, which breaks the hermetic contract (CLAUDE.md). Restore "
        f"the file rather than letting the suite reach the network."
    )
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(TIKTOKEN_CACHE))


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


@pytest.fixture
def filings_store(tmp_path, recorded_filing):
    """A real on-disk `filings` collection holding Apple's real FY2025 Sections.

    Real Chroma and real chunks, a fake embedding: retrieval behaviour (ordering, `k`,
    metadata) is the store's, and nothing reaches the network. Deliberately *not* the
    ingested `data/chroma` — that index is only searchable by the paid model that wrote
    it, so pointing a test at it would either need a key or return noise.
    """
    from finbrief.ingestion.chunking import chunk_filing
    from finbrief.retrieval.vectorstore import build_filings_store, write_chunks

    store = build_filings_store(
        persist_directory=str(tmp_path / "chroma"), embeddings=KeywordEmbeddings()
    )
    write_chunks(store, chunk_filing(recorded_filing))
    return store


@pytest.fixture
def empty_filings_store(tmp_path):
    """A `filings` collection with nothing in it — the retrieval-level fallback's input."""
    from finbrief.retrieval.vectorstore import build_filings_store

    return build_filings_store(
        persist_directory=str(tmp_path / "empty-chroma"), embeddings=KeywordEmbeddings()
    )


@pytest.fixture
def agent_builds(monkeypatch):
    """Stop an `AppTest` from constructing a real agent, and count the attempts.

    Shared by both app-level test files because the app builds its agent under
    `@st.cache_resource`, and an unpatched build would open a SQLite checkpoint file *in the
    repo's `data/`* on any test that sends a message — a hermetic breach (CLAUDE.md) that no
    assertion in either file would notice. Seam 3 stubs the agent entirely anyway: what the
    app layer is tested for is which `thread_id` a turn lands on and what the page renders,
    never what an agent does with either.

    Clearing `st.cache_resource` on both sides is what keeps the stub honest: the cache is
    keyed by qualified name, not by function identity, so without it one test's cached stub is
    handed to the next — and a test asserting "the same agent survived" would pass on a stale
    object from a previous test.

    Yields the list of agents built, so a test can assert the cache built exactly one.
    """
    import streamlit as st

    from finbrief.agent import agent as agent_module

    st.cache_resource.clear()
    built: list[object] = []

    def fake_build_agent(**kwargs):
        built.append(object())
        return built[-1]

    monkeypatch.setattr(agent_module, "build_agent", fake_build_agent)
    yield built
    st.cache_resource.clear()


@pytest.fixture(autouse=True)
def pristine_package_logger():
    """Restore the `finbrief` logger, so `configure_logging` in one test can't mute another."""
    logger = logging.getLogger(PACKAGE_LOGGER)
    handlers, level, propagate = logger.handlers[:], logger.level, logger.propagate
    yield
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate
