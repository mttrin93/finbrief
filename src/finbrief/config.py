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


class PeerCluster(StrEnum):
    """A curated same-sector peer cluster within the Universe.

    Doubles as the peer pool: `calculate_ratios` compares a company against the other
    Universe members of its cluster and never against an out-of-Universe ticker.

    Deliberately not named `Sector`: these are curated for ratio comparability, not drawn
    from a sector taxonomy. `big_tech` spans three GICS sectors, and AMZN sits apart from
    the autos despite sharing one with them — the cluster is the unit, GICS an input to
    curating it (ADR-0009).
    """

    BIG_TECH = "big_tech"
    AUTOS = "autos"
    BANKS = "banks"
    HEALTHCARE = "healthcare"

    @property
    def label(self) -> str:
        """Display name for the UI: `big_tech` -> "Big Tech"."""
        return self.value.replace("_", " ").title()


@dataclass(frozen=True, slots=True)
class Company:
    """One member of the Universe."""

    ticker: str
    name: str
    cluster: PeerCluster


#: The Universe: fixed at ingest time, curated as same-sector peer clusters so it can
#: serve triple duty — KB scope, demo cast, and peer pool (CONTEXT.md, ADR-0009).
#:
#: **Every member must file a 10-K.** ADR-0007 scopes the KB to Items 1/1A/7/7A of the
#: annual 10-K, and foreign private issuers file a 20-F, which has no such items — so a
#: 20-F filer here would silently produce zero Sections at ingest. `test_config.py` guards
#: this; ADR-0007 records it as a stated limitation.
UNIVERSE: tuple[Company, ...] = (
    Company("AAPL", "Apple Inc.", PeerCluster.BIG_TECH),
    Company("MSFT", "Microsoft Corporation", PeerCluster.BIG_TECH),
    Company("NVDA", "NVIDIA Corporation", PeerCluster.BIG_TECH),
    Company("AMZN", "Amazon.com, Inc.", PeerCluster.BIG_TECH),
    Company("GOOGL", "Alphabet Inc.", PeerCluster.BIG_TECH),
    Company("META", "Meta Platforms, Inc.", PeerCluster.BIG_TECH),
    Company("TSLA", "Tesla, Inc.", PeerCluster.AUTOS),
    Company("F", "Ford Motor Company", PeerCluster.AUTOS),
    Company("GM", "General Motors Company", PeerCluster.AUTOS),
    Company("JPM", "JPMorgan Chase & Co.", PeerCluster.BANKS),
    Company("BAC", "Bank of America Corporation", PeerCluster.BANKS),
    Company("GS", "The Goldman Sachs Group, Inc.", PeerCluster.BANKS),
    Company("JNJ", "Johnson & Johnson", PeerCluster.HEALTHCARE),
    Company("LLY", "Eli Lilly and Company", PeerCluster.HEALTHCARE),
    Company("PFE", "Pfizer Inc.", PeerCluster.HEALTHCARE),
)

#: Foreign private issuers file a 20-F, not a 10-K, so they cannot supply the Sections
#: ADR-0007 scopes the KB to.
#:
#: **No longer the source of truth.** Ticket T2 (#3) put the real check in
#: `ingestion/gate.py`, which asks EDGAR for each company's most recent annual filing and
#: fails loudly unless it is in the 10-K family — that catches *any* foreign private
#: issuer, not merely one someone thought to write down.
#:
#: Kept anyway, in its demoted role, because it is free and offline: it fails in
#: `test_config.py` in milliseconds with no network, where the gate needs a live EDGAR
#: round trip and a full ingest run to say the same thing. A tripwire that catches the
#: plausible re-addition (the `eu_tech` cluster this replaced — SAP/ASML/STM — was all
#: three) before anyone waits for the gate. The invariant itself remains the `UNIVERSE`
#: docstring's "every member files a 10-K"; neither list nor test proves it, and the gate
#: is what enforces it.
_TWENTY_F_FILERS: frozenset[str] = frozenset(
    {
        # Tech / big_tech candidates
        "SAP", "ASML", "STM", "TSM", "BABA", "INFY", "SONY", "SHOP",
        # Autos candidates
        "TM", "HMC", "RACE", "STLA",
        # Banks candidates
        "HSBC", "UBS", "DB", "BCS", "MUFG", "RY", "TD",
        # Healthcare candidates
        "NVO", "AZN", "SNY", "GSK", "NVS", "TAK",
        # Energy / other frequently-suggested large caps
        "SHEL", "BP", "TTE", "PBR", "BHP", "RIO",
    }
)  # fmt: skip


#: The Universe filers hand-verified to answer Item 7A with a pointer into Item 7 — every
#: bank and every healthcare name, 6 of 15, and nobody else (ADR-0007 amendment; the rows
#: marked "incorporated by reference" in docs/verification/section-starts.md).
#:
#: A tripwire, not documentation: `ingestion/gate.py` fails any filing where the
#: incorporation-by-reference excusal fires for a ticker outside this set
#: (`pointer_filer_is_recorded`). Either the filer newly hands the Item off — verify
#: against the filing on EDGAR, then record it here — or the pointer detection misfired
#: on a broken parse. Both deserve a human; neither may be a silent drop.
ITEM_7A_POINTER_FILERS: frozenset[str] = frozenset({"BAC", "GS", "JNJ", "JPM", "LLY", "PFE"})


def _build_clusters(
    universe: tuple[Company, ...],
) -> Mapping[PeerCluster, tuple[str, ...]]:
    """Group the Universe by peer cluster, preserving declaration order."""
    by_cluster: dict[PeerCluster, list[str]] = {}
    for company in universe:
        by_cluster.setdefault(company.cluster, []).append(company.ticker)
    return MappingProxyType({cluster: tuple(t) for cluster, t in by_cluster.items()})


def _build_peers(
    clusters: Mapping[PeerCluster, tuple[str, ...]], universe: tuple[Company, ...]
) -> Mapping[str, tuple[str, ...]]:
    """Derive the static PEERS map from the Universe's peer clusters.

    ADR-0009 calls for a static map in `config.py`. Deriving it from the single Universe
    declaration keeps it static (computed once at import, no I/O) while making it
    impossible for the map to drift from the Universe it describes.
    """
    return MappingProxyType(
        {
            company.ticker: tuple(t for t in clusters[company.cluster] if t != company.ticker)
            for company in universe
        }
    )


COMPANIES: Mapping[str, Company] = MappingProxyType({c.ticker: c for c in UNIVERSE})
TICKERS: frozenset[str] = frozenset(COMPANIES)

#: The Universe filers that have an `Item 7A` Section of their own — the complement of
#: `ITEM_7A_POINTER_FILERS`. Derived rather than typed: "9 of 15" appears in the app's
#: grounding-scope panel, in the retrieval smoke check's rationale and in CONTEXT.md, and a
#: hand-typed copy of it is wrong the moment a filer starts or stops handing the Item off.
ITEM_7A_SECTION_FILERS: frozenset[str] = TICKERS - ITEM_7A_POINTER_FILERS

#: peer cluster -> its tickers. The one place the grouping is computed: `_build_peers` and
#: the UI's Universe panel both read it rather than re-deriving it from `UNIVERSE`.
CLUSTERS: Mapping[PeerCluster, tuple[str, ...]] = _build_clusters(UNIVERSE)

#: ticker -> same-cluster Universe peers, excluding the ticker itself.
#: Every cluster holds at least three members, so no company is ever compared against a
#: single peer (a "peer mean" of one). Phase 3 (`calculate_ratios`) still reports the peer
#: set and n inline, per ADR-0009.
PEERS: Mapping[str, tuple[str, ...]] = _build_peers(CLUSTERS, UNIVERSE)


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

#: Reciprocal Rank Fusion's rank-smoothing constant: a candidate list's vote for the chunk it
#: ranks `r`th is `1 / (RRF_K + r)` (`retrieval/hybrid.py`).
#:
#: **60 is the published default, stated and not tuned** — Cormack, Clarke & Buettcher (2009),
#: who introduced RRF and report it as insensitive over a wide range. Deliberately *not* a
#: `Settings` field, on the same reasoning as `CHUNK_SIZE_CHARS` below and for one extra one:
#: ADR-0005 pre-registers the shipping default before any A/B data exists so that the winner
#: cannot be picked after seeing the numbers, and a fusion constant somebody could sweep is
#: exactly the back door into that. A `hybrid` result obtained at the published constant is a
#: prediction that survived a test; one obtained at the best of several `RRF_K` values is a
#: number about the sweep. If it is ever changed, it is changed here, in one commit, with the
#: A/B re-run — never per environment.
RRF_K = 60

#: Maximum characters per chunk — the PLAN.md baseline for section-aware chunking
#: (RecursiveCharacterTextSplitter, 1000 chars with a 200-char overlap; the overlap
#: belongs to the chunker itself and lands with it in Phase 1).
#:
#: **Ticket T2 (KB ingest, #3) owns tuning this value**, since it lands with the chunker in
#: Phase 1 — Tier-1 work, not post-gate Tier-2 — and it is the single source of truth for
#: it: two things depend on it. `retrieval/embeddings.py` sends raw strings with no length-safe
#: splitting, so a chunk over the embedding model's 8191-token window would fail the
#: request outright — `tests/test_chunk_token_limit.py` imports this constant and asserts
#: the worst-case token count it implies stays under that window, so raising it here
#: re-checks the guarantee instead of silently voiding it.
#: Changing it also invalidates an existing index, which must be re-ingested.
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
    chroma_dir: str
    checkpoint_db: str

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
            # ADR-0004 caps translation at 3 sub-queries for latency. The cap is enforced,
            # not merely defaulted: the latency budget ADR-0005 judges dominance within
            # (<=1.5s p50 added by translation) assumes it, so an env override must not be
            # able to quietly invalidate the A/B result.
            max_sub_queries=_integer(env, "FINBRIEF_MAX_SUB_QUERIES", 3, minimum=0, maximum=3),
            eval_mode=_boolean(env, "FINBRIEF_EVAL_MODE", False),
            alphavantage_enabled=_boolean(env, "FINBRIEF_ALPHAVANTAGE_ENABLED", False),
            sec_edgar_user_agent=resolve_sec_edgar_user_agent(env),
            # Where the persisted Chroma collections live. An override exists because a
            # deployment's writable path is not the repo's, and because the evaluation
            # harness builds throwaway indexes; the default keeps ingest and app pointed
            # at the same directory without configuration.
            chroma_dir=_string(env, "FINBRIEF_CHROMA_DIR", "data/chroma"),
            # The agent's memory of record (ADR-0008): a SQLite file, overridable for the
            # same reason `chroma_dir` is — a deployment's writable path is not the repo's.
            # Ephemeral by design on Streamlit Community Cloud; conversations are not durable
            # data, and losing them costs a conversation rather than the knowledge base.
            checkpoint_db=_string(env, "FINBRIEF_CHECKPOINT_DB", "data/checkpoints.sqlite"),
        )


def _raw(env: Mapping[str, str], name: str) -> str:
    """The trimmed value of `name`, or `""` when unset or blank.

    The single entry point the typed readers below share, so "unset" and "whitespace only"
    mean the same thing everywhere — see `test_blank_api_key_is_treated_as_missing`.
    """
    return env.get(name, "").strip()


def _required(env: Mapping[str, str], name: str) -> str:
    value = _raw(env, name)
    if not value:
        raise ConfigError(f"{name} is not set. {_ENV_FILE_HINT}")
    return value


def _string(env: Mapping[str, str], name: str, default: str) -> str:
    return _raw(env, name) or default


def _boolean(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = _raw(env, name).lower()
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


def _integer(
    env: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    raw = _raw(env, name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name}={raw!r} is not an integer.") from exc
    if value < minimum:
        raise ConfigError(f"{name}={value} is below the minimum of {minimum}.")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{name}={value} is above the maximum of {maximum}.")
    return value


def _strategy(
    env: Mapping[str, str], name: str, default: RetrievalStrategy
) -> RetrievalStrategy:
    raw = _raw(env, name).lower()
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


def resolve_sec_edgar_user_agent(env: Mapping[str, str]) -> str:
    """Resolve `SEC_EDGAR_USER_AGENT` alone, without the rest of `Settings`.

    Deliberately independent of `Settings` for the same reason `resolve_log_level` is:
    a `--dry-run` ingest needs an EDGAR identity and no OpenRouter key, and
    `Settings.from_env` hard-requires the key. `Settings.from_env` reads the field
    through this same function, so the two resolutions cannot drift.
    """
    return _raw(env, "SEC_EDGAR_USER_AGENT")


def resolve_log_level(env: Mapping[str, str]) -> int:
    """Resolve `LOG_LEVEL` to a `logging` level.

    Deliberately independent of `Settings`: logging must be configurable before — and
    without — a valid API key, so a config failure can still be reported through it.
    Takes an explicit mapping for the same reason `Settings.from_env` does — one
    resolution path, and no hidden `.env` read on the way; the caller loads the
    environment (`load_env()`) and passes it in.
    """
    raw = _raw(env, "LOG_LEVEL").upper()
    if not raw:
        return logging.INFO
    # `getLevelNamesMapping()` includes NOTSET (0), which is not a threshold but "inherit
    # from the parent". Accepting it would set the package logger to 0 and — with
    # `propagate=False` — let its effective level fall back to root's WARNING, silently
    # dropping every INFO event Phase 6/7 reads back. A blackout is worse than a failure.
    levels = {
        name: level
        for name, level in logging.getLevelNamesMapping().items()
        if name != "NOTSET"
    }
    if raw not in levels:
        valid = ", ".join(sorted(levels))
        raise ConfigError(f"LOG_LEVEL={raw!r} is not a log level. Use one of: {valid}.")
    return levels[raw]
