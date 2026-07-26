"""Central configuration: the Universe, Peers, models, strategy switches, feature flags.

Every knob FinBrief has lives here, so the app and the evaluation harness read the same
switches (ADR-0003: one code path, config-selected strategy). Secrets are read from the
environment — `.env` locally, platform env vars / `st.secrets` when deployed — and are
never logged or reprinted.

`Settings.from_env()` takes an explicit mapping so configuration is a pure function of
its environment; `get_settings()` is the cached application entrypoint on top of it.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache
from types import MappingProxyType

from dotenv import load_dotenv


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or malformed."""


# --------------------------------------------------------------------------------------
# Universe and Peers (ADR-0009)
# --------------------------------------------------------------------------------------


class Sector(StrEnum):
    """A curated same-sector peer cluster within the Universe.

    Doubles as the peer pool: `calculate_ratios` compares a company against the other
    Universe members of its sector and never against an out-of-Universe ticker.
    """

    BIG_TECH = "big_tech"
    AUTOS = "autos"
    BANKS = "banks"
    EU_TECH = "eu_tech"


@dataclass(frozen=True, slots=True)
class Company:
    """One member of the Universe."""

    ticker: str
    name: str
    sector: Sector


#: The Universe: fixed at ingest time, curated as same-sector peer clusters so it can
#: serve triple duty — KB scope, demo cast, and peer pool (CONTEXT.md, ADR-0009).
UNIVERSE: tuple[Company, ...] = (
    Company("AAPL", "Apple Inc.", Sector.BIG_TECH),
    Company("MSFT", "Microsoft Corporation", Sector.BIG_TECH),
    Company("NVDA", "NVIDIA Corporation", Sector.BIG_TECH),
    Company("AMZN", "Amazon.com, Inc.", Sector.BIG_TECH),
    Company("GOOGL", "Alphabet Inc.", Sector.BIG_TECH),
    Company("META", "Meta Platforms, Inc.", Sector.BIG_TECH),
    Company("TSLA", "Tesla, Inc.", Sector.AUTOS),
    Company("F", "Ford Motor Company", Sector.AUTOS),
    Company("GM", "General Motors Company", Sector.AUTOS),
    Company("JPM", "JPMorgan Chase & Co.", Sector.BANKS),
    Company("BAC", "Bank of America Corporation", Sector.BANKS),
    Company("GS", "The Goldman Sachs Group, Inc.", Sector.BANKS),
    Company("SAP", "SAP SE", Sector.EU_TECH),
    Company("ASML", "ASML Holding N.V.", Sector.EU_TECH),
    Company("STM", "STMicroelectronics N.V.", Sector.EU_TECH),
)


def _index_by_ticker(universe: tuple[Company, ...]) -> Mapping[str, Company]:
    return MappingProxyType({company.ticker: company for company in universe})


def _build_peers(universe: tuple[Company, ...]) -> Mapping[str, tuple[str, ...]]:
    """Derive the static PEERS map from the Universe's sector clusters.

    ADR-0009 calls for a static map in `config.py`. Deriving it from the single Universe
    declaration keeps it static (computed once at import, no I/O) while making it
    impossible for the map to drift from the Universe it describes.
    """
    by_sector: dict[Sector, list[str]] = {}
    for company in universe:
        by_sector.setdefault(company.sector, []).append(company.ticker)
    return MappingProxyType(
        {
            company.ticker: tuple(t for t in by_sector[company.sector] if t != company.ticker)
            for company in universe
        }
    )


COMPANIES: Mapping[str, Company] = _index_by_ticker(UNIVERSE)
TICKERS: frozenset[str] = frozenset(COMPANIES)

#: ticker -> same-sector Universe peers, excluding the ticker itself.
#: Every cluster holds at least three members, so no company is ever compared against a
#: single peer (a "peer mean" of one). Phase 3 (`calculate_ratios`) still reports the peer
#: set and n inline, per ADR-0009.
PEERS: Mapping[str, tuple[str, ...]] = _build_peers(UNIVERSE)


# --------------------------------------------------------------------------------------
# Retrieval strategy switches (ADR-0004, ADR-0005)
# --------------------------------------------------------------------------------------


class RetrievalStrategy(StrEnum):
    """The retrieval strategies the A/B harness compares."""

    VECTOR = "vector"
    HYBRID = "hybrid"


#: Pre-registered shipping default (ADR-0005), fixed before any A/B data exists.
DEFAULT_STRATEGY = RetrievalStrategy.HYBRID
DEFAULT_TRANSLATION_ENABLED = True

#: Maximum characters per chunk — the PLAN.md baseline for section-aware chunking
#: (RecursiveCharacterTextSplitter, 1000 chars with a 200-char overlap; the overlap
#: belongs to the chunker itself and lands with it in Phase 1).
#:
#: **T2 owns tuning this value**, and it is the single source of truth for it: two things
#: depend on it. `retrieval/embeddings.py` sends raw strings with no length-safe
#: splitting, so a chunk over the embedding model's 8191-token window would fail the
#: request outright — `tests/test_chunk_token_limit.py` imports this constant and guards
#: that ceiling, so raising it here re-checks the guarantee instead of silently voiding
#: it. Changing it also invalidates an existing index, which must be re-ingested.
#: A deliberately plain constant, not a `Settings` field: chunk size is not a Tier-1 A/B
#: axis, and an env override would let a running app disagree with the index on disk.
CHUNK_SIZE_CHARS = 1000


# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------

_ENV_FILE_HINT = (
    "Copy .env.example to .env and fill it in (or set it in the deployment environment)."
)

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


@dataclass(frozen=True, slots=True)
class Settings:
    """Resolved runtime configuration.

    `openrouter_api_key` is `repr=False` so the secret cannot leak into a traceback,
    a log line, or a debugger dump of this object.
    """

    openrouter_api_key: str = field(repr=False)
    openrouter_base_url: str
    chat_model: str
    embedding_model: str
    retrieval_strategy: RetrievalStrategy
    query_translation_enabled: bool
    retrieval_k: int
    max_sub_queries: int
    eval_mode: bool
    alphavantage_enabled: bool
    sec_edgar_user_agent: str

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> Settings:
        """Build settings from an environment mapping, failing loudly on bad input."""
        return cls(
            openrouter_api_key=_required(env, "OPENROUTER_API_KEY"),
            openrouter_base_url=_string(
                env, "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
            chat_model=_string(env, "FINBRIEF_CHAT_MODEL", "openai/gpt-4o-mini"),
            # Served by OpenRouter's /v1/embeddings, so it needs no key or base URL of
            # its own. One model for ingest and query — see retrieval/embeddings.py.
            embedding_model=_string(
                env, "FINBRIEF_EMBEDDING_MODEL", "openai/text-embedding-3-small"
            ),
            retrieval_strategy=_strategy(env, "FINBRIEF_RETRIEVAL_STRATEGY", DEFAULT_STRATEGY),
            query_translation_enabled=_boolean(
                env, "FINBRIEF_QUERY_TRANSLATION", DEFAULT_TRANSLATION_ENABLED
            ),
            retrieval_k=_integer(env, "FINBRIEF_RETRIEVAL_K", 5, minimum=1),
            # ADR-0004 caps translation at 3 sub-queries for latency.
            max_sub_queries=_integer(env, "FINBRIEF_MAX_SUB_QUERIES", 3, minimum=0),
            eval_mode=_boolean(env, "FINBRIEF_EVAL_MODE", False),
            alphavantage_enabled=_boolean(env, "FINBRIEF_ALPHAVANTAGE_ENABLED", False),
            sec_edgar_user_agent=_string(env, "SEC_EDGAR_USER_AGENT", ""),
        )


def _required(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise ConfigError(f"{name} is not set. {_ENV_FILE_HINT}")
    return value


def _string(env: Mapping[str, str], name: str, default: str) -> str:
    value = env.get(name, "").strip()
    return value or default


def _boolean(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise ConfigError(
        f"{name}={env[name]!r} is not a boolean. Use one of: "
        f"{', '.join(sorted(_TRUE | _FALSE))}."
    )


def _integer(env: Mapping[str, str], name: str, default: int, *, minimum: int) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name}={raw!r} is not an integer.") from exc
    if value < minimum:
        raise ConfigError(f"{name}={value} is below the minimum of {minimum}.")
    return value


def _strategy(
    env: Mapping[str, str], name: str, default: RetrievalStrategy
) -> RetrievalStrategy:
    raw = env.get(name, "").strip().lower()
    if not raw:
        return default
    try:
        return RetrievalStrategy(raw)
    except ValueError as exc:
        valid = ", ".join(s.value for s in RetrievalStrategy)
        raise ConfigError(f"{name}={raw!r} is not a strategy. Use one of: {valid}.") from exc


@lru_cache(maxsize=1)
def load_env() -> None:
    """Load `.env` into the process environment, once.

    Existing environment variables win, so deployment env vars and `st.secrets`
    (which Streamlit also exposes as env vars) are never overwritten by a stray
    local `.env`.
    """
    load_dotenv(override=False)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The application's settings. Cached; call `get_settings.cache_clear()` in tests."""
    load_env()
    return Settings.from_env(os.environ)


def resolve_log_level(env: Mapping[str, str] | None = None) -> int:
    """Resolve `LOG_LEVEL` to a `logging` level.

    Deliberately independent of `Settings`: logging must be configurable before — and
    without — a valid API key, so a config failure can still be reported through it.
    """
    load_env()
    raw = (env if env is not None else os.environ).get("LOG_LEVEL", "").strip().upper()
    if not raw:
        return logging.INFO
    levels = logging.getLevelNamesMapping()
    if raw not in levels:
        valid = ", ".join(sorted(name for name in levels if name != "NOTSET"))
        raise ConfigError(f"LOG_LEVEL={raw!r} is not a log level. Use one of: {valid}.")
    return levels[raw]
