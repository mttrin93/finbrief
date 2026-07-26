"""Config: Universe/Peers invariants (ADR-0009) and environment resolution.

`Settings.from_env` takes an explicit mapping, so these tests never touch the process
environment or a local `.env`.
"""

import logging

import pytest

from finbrief.config import (
    _TWENTY_F_FILERS,
    COMPANIES,
    PEERS,
    TICKERS,
    UNIVERSE,
    ConfigError,
    PeerCluster,
    RetrievalStrategy,
    Settings,
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


def test_every_peer_cluster_is_populated():
    populated = {company.cluster for company in UNIVERSE}
    assert populated == set(PeerCluster)


def test_the_universe_holds_only_10k_filers():
    # ADR-0007 scopes the KB to Items 1/1A/7/7A of the annual 10-K. A foreign private
    # issuer files a 20-F, which has no such items, so it would contribute zero Sections
    # at ingest — and the failure would surface in Phase 1, long after the Universe was
    # settled. The `eu_tech` cluster this replaced (SAP/ASML/STM) was all three.
    offenders = TICKERS & _TWENTY_F_FILERS
    assert not offenders, (
        f"{sorted(offenders)} file a 20-F, not a 10-K — no Item 1A/7/7A to ingest (ADR-0007)"
    )


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
    assert settings.retrieval_strategy is RetrievalStrategy.HYBRID
    assert settings.query_translation_enabled is True
    assert settings.max_sub_queries == 3  # ADR-0004 latency cap
    assert settings.openrouter_base_url == "https://openrouter.ai/api/v1"
    # Served through the same OpenRouter base URL — no second provider (CONTEXT.md).
    assert settings.embedding_model == "openai/text-embedding-3-small"


def test_environment_overrides_every_switch():
    settings = Settings.from_env(
        MINIMAL_ENV
        | {
            "OPENROUTER_BASE_URL": "https://proxy.example/v1",
            "FINBRIEF_CHAT_MODEL": "anthropic/claude-haiku-4.5",
            "FINBRIEF_RETRIEVAL_STRATEGY": "vector",
            "FINBRIEF_QUERY_TRANSLATION": "off",
            "FINBRIEF_RETRIEVAL_K": "8",
            "FINBRIEF_EVAL_MODE": "yes",
        }
    )
    assert settings.openrouter_base_url == "https://proxy.example/v1"
    assert settings.chat_model == "anthropic/claude-haiku-4.5"
    assert settings.retrieval_strategy is RetrievalStrategy.VECTOR
    assert settings.query_translation_enabled is False
    assert settings.retrieval_k == 8
    assert settings.eval_mode is True


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
