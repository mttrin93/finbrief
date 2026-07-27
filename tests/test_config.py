"""Config: Universe/Peers invariants (ADR-0009) and environment resolution.

`Settings.from_env` takes an explicit mapping, so these tests never touch the process
environment or a local `.env`.
"""

import logging

import pytest

from finbrief.config import (
    _TWENTY_F_FILERS,
    CLUSTERS,
    COMPANIES,
    ITEM_7A_POINTER_FILERS,
    PEERS,
    TICKERS,
    UNIVERSE,
    ConfigError,
    PeerCluster,
    RetrievalStrategy,
    Settings,
    get_settings,
    resolve_log_level,
)

MINIMAL_ENV = {"OPENROUTER_API_KEY": "sk-test"}


# --- Universe and Peers (ADR-0009) ----------------------------------------------------


def test_universe_is_ten_to_fifteen_unique_companies():
    assert 10 <= len(UNIVERSE) <= 15
    assert len(TICKERS) == len(UNIVERSE)


def test_every_company_has_at_least_two_peers():
    # A two-member cluster would make "the peer mean" a mean of one, which is a
    # comparison against a single company wearing a statistic's clothes.
    for ticker in TICKERS:
        assert len(PEERS[ticker]) >= 2, f"{ticker}'s cluster is too thin to average"


def test_peers_are_same_cluster_universe_members_excluding_self():
    assert set(PEERS) == TICKERS
    for ticker, peers in PEERS.items():
        assert ticker not in peers
        assert set(peers) <= TICKERS, "peers must be drawn exclusively from the Universe"
        assert {COMPANIES[p].cluster for p in peers} == {COMPANIES[ticker].cluster}


def test_peer_relation_is_symmetric():
    for ticker, peers in PEERS.items():
        for peer in peers:
            assert ticker in PEERS[peer]


def test_clusters_partition_the_universe():
    # CLUSTERS is the one computed grouping — PEERS and the UI's Universe panel both read
    # it instead of re-deriving it, so it has to be a true partition.
    assert set(CLUSTERS) == set(PeerCluster)
    grouped = [ticker for cluster in CLUSTERS.values() for ticker in cluster]
    assert sorted(grouped) == sorted(TICKERS), "every company belongs to exactly one cluster"


def test_a_cluster_label_is_display_ready():
    assert PeerCluster.BIG_TECH.label == "Big Tech"
    assert PeerCluster.HEALTHCARE.label == "Healthcare"


def test_every_peer_cluster_is_populated():
    populated = {company.cluster for company in UNIVERSE}
    assert populated == set(PeerCluster)


def test_the_universe_trips_no_known_20f_filer():
    # A tripwire, not the proof. `ingestion/gate.py` asks EDGAR whether each company's
    # most recent annual filing is a 10-K and is the real check (ADR-0007 amendment, #3);
    # this one is kept because it is offline and instant, so a plausible re-addition
    # (the `eu_tech` cluster this replaced — SAP/ASML/STM — was all three) fails here in
    # milliseconds rather than waiting on a live ingest run.
    offenders = TICKERS & _TWENTY_F_FILERS
    assert not offenders, (
        f"{sorted(offenders)} file a 20-F, not a 10-K — no Item 1A/7/7A to ingest (ADR-0007)"
    )


def test_every_recorded_pointer_filer_is_in_the_universe():
    # `ITEM_7A_POINTER_FILERS` is a tripwire, and a typo'd ticker in it is a tripwire that
    # never trips: it silently stops covering the real filer, whose next pointer then
    # becomes a `pointer_filer_is_recorded` finding on a live run. The neighbouring
    # `_TWENTY_F_FILERS` list has this check; this one is the same shape.
    strays = ITEM_7A_POINTER_FILERS - TICKERS
    assert not strays, f"{sorted(strays)} are not Universe tickers (ADR-0007 amendment)"


# --- Settings -------------------------------------------------------------------------


def test_missing_api_key_fails_with_an_actionable_message():
    with pytest.raises(ConfigError) as excinfo:
        Settings.from_env({})
    message = str(excinfo.value)
    assert "OPENROUTER_API_KEY" in message
    assert ".env.example" in message


def test_blank_api_key_is_treated_as_missing():
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        Settings.from_env({"OPENROUTER_API_KEY": "   "})


def test_defaults_match_the_pre_registered_shipping_strategy():
    # ADR-0005: hybrid + translation, fixed before any A/B data exists.
    settings = Settings.from_env(MINIMAL_ENV)
    assert settings.chroma_dir == "data/chroma"  # ingest and app agree without config
    assert settings.retrieval_strategy is RetrievalStrategy.HYBRID
    assert settings.query_translation_enabled is True
    assert settings.max_sub_queries == 3  # ADR-0004 latency cap
    assert settings.openrouter_base_url == "https://openrouter.ai/api/v1"
    # Served through the same OpenRouter base URL — no second provider (CONTEXT.md).
    assert settings.embedding_model == "openai/text-embedding-3-small"


def test_environment_overrides_every_switch():
    # Every field, so a misread name (the failure mode: an override that silently does
    # nothing) cannot hide — `FINBRIEF_EMBEDDING_MODEL` above all, since ingest and query
    # drifting apart degrades retrieval to noise with no error.
    settings = Settings.from_env(
        {
            "OPENROUTER_API_KEY": "sk-override",
            "OPENROUTER_BASE_URL": "https://proxy.example/v1",
            "FINBRIEF_CHAT_MODEL": "anthropic/claude-haiku-4.5",
            "FINBRIEF_EMBEDDING_MODEL": "openai/text-embedding-3-large",
            "FINBRIEF_RETRIEVAL_STRATEGY": "vector",
            "FINBRIEF_QUERY_TRANSLATION": "off",
            "FINBRIEF_RETRIEVAL_K": "8",
            "FINBRIEF_MAX_SUB_QUERIES": "1",
            "FINBRIEF_EVAL_MODE": "yes",
            "FINBRIEF_ALPHAVANTAGE_ENABLED": "true",
            "SEC_EDGAR_USER_AGENT": "finbrief someone@example.com",
            "FINBRIEF_CHROMA_DIR": "/tmp/finbrief-index",
        }
    )
    assert settings.openrouter_api_key == "sk-override"
    assert settings.openrouter_base_url == "https://proxy.example/v1"
    assert settings.chat_model == "anthropic/claude-haiku-4.5"
    assert settings.embedding_model == "openai/text-embedding-3-large"
    assert settings.retrieval_strategy is RetrievalStrategy.VECTOR
    assert settings.query_translation_enabled is False
    assert settings.retrieval_k == 8
    assert settings.max_sub_queries == 1
    assert settings.eval_mode is True
    assert settings.alphavantage_enabled is True
    assert settings.sec_edgar_user_agent == "finbrief someone@example.com"
    assert settings.chroma_dir == "/tmp/finbrief-index"


def test_get_settings_reads_the_process_environment_and_caches(monkeypatch):
    # The path the app actually takes. `Settings.from_env` being correct is worth nothing
    # if `get_settings` hands it the wrong mapping.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-from-environ")
    monkeypatch.setenv("FINBRIEF_RETRIEVAL_K", "9")

    settings = get_settings()
    assert settings.openrouter_api_key == "sk-from-environ"
    assert settings.retrieval_k == 9
    assert get_settings() is settings, "settings are cached for the process"


def test_get_settings_raises_rather_than_caching_a_failure(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        get_settings()

    # A cached failure would make the banner's "copy .env.example" advice unfollowable.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-late")
    assert get_settings().openrouter_api_key == "sk-late"


def test_the_api_key_never_appears_in_a_repr():
    settings = Settings.from_env(MINIMAL_ENV | {"OPENROUTER_API_KEY": "sk-super-secret"})
    assert "sk-super-secret" not in repr(settings)
    assert settings.openrouter_api_key == "sk-super-secret"


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"FINBRIEF_EVAL_MODE": "maybe"}, "boolean"),
        ({"FINBRIEF_RETRIEVAL_K": "many"}, "integer"),
        ({"FINBRIEF_RETRIEVAL_K": "0"}, "minimum"),
        # ADR-0004's 3-sub-query latency cap is a ceiling, not just a default.
        ({"FINBRIEF_MAX_SUB_QUERIES": "99"}, "maximum"),
        ({"FINBRIEF_RETRIEVAL_STRATEGY": "graph"}, "strategy"),
    ],
)
def test_malformed_values_fail_loudly(env, expected):
    with pytest.raises(ConfigError, match=expected):
        Settings.from_env(MINIMAL_ENV | env)


# --- Log level ------------------------------------------------------------------------


def test_log_level_defaults_to_info_and_accepts_a_name():
    assert resolve_log_level({}) == logging.INFO
    assert resolve_log_level({"LOG_LEVEL": "debug"}) == logging.DEBUG


def test_unknown_log_level_fails_loudly():
    with pytest.raises(ConfigError, match="LOG_LEVEL"):
        resolve_log_level({"LOG_LEVEL": "chatty"})


def test_notset_is_rejected_rather_than_silently_muting_the_logs():
    # NOTSET is in `logging.getLevelNamesMapping()` but means "inherit", not a threshold:
    # accepting it would set the package logger to 0, and with `propagate=False` its
    # effective level would fall back to root's WARNING — every INFO event Phase 6/7 reads
    # back would vanish with no error at all.
    with pytest.raises(ConfigError, match="LOG_LEVEL"):
        resolve_log_level({"LOG_LEVEL": "NOTSET"})


def test_the_log_level_error_lists_only_levels_it_accepts():
    with pytest.raises(ConfigError) as excinfo:
        resolve_log_level({"LOG_LEVEL": "chatty"})
    message = str(excinfo.value)
    assert "NOTSET" not in message, "the message must not offer a level that is rejected"
    for accepted in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        assert accepted in message


def test_a_blank_log_level_falls_back_to_info():
    assert resolve_log_level({"LOG_LEVEL": "   "}) == logging.INFO
