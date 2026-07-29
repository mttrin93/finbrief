"""Config: Universe/Peers invariants (ADR-0009) and environment resolution.

`Settings.from_env` takes an explicit mapping, so these tests never touch the process
environment or a local `.env`.
"""

import logging
import re
import subprocess
from pathlib import Path

import pytest

from finbrief.config import (
    _TWENTY_F_FILERS,
    CLUSTERS,
    COMPANIES,
    ITEM_7A_POINTER_FILERS,
    ITEM_7A_SECTION_FILERS,
    MIN_LEXICAL_TICKER_CHARS,
    PEERS,
    TICKER_BY_COMPANY_NAME,
    TICKERS,
    UNIVERSE,
    ConfigError,
    PeerCluster,
    RetrievalStrategy,
    Settings,
    get_settings,
    resolve_log_file,
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


def test_the_filers_with_their_own_item_7a_are_derived_from_the_pointer_set():
    # "9 of 15" is read out by the app's grounding-scope panel and by the retrieval smoke
    # check's rationale, which lands verbatim in a committed artifact. Both take the count
    # from here, so neither can be the stale copy (issue #5 review).
    assert ITEM_7A_SECTION_FILERS | ITEM_7A_POINTER_FILERS == TICKERS
    assert not ITEM_7A_SECTION_FILERS & ITEM_7A_POINTER_FILERS
    assert len(ITEM_7A_SECTION_FILERS) == len(TICKERS) - len(ITEM_7A_POINTER_FILERS)


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
    # The gate's own model, priced against one YES/NO per turn rather than a brief
    # (ADR-0006 layer 3).
    assert settings.classifier_model == "openai/gpt-4o-mini"


def test_environment_overrides_every_switch():
    # Every field, so a misread name (the failure mode: an override that silently does
    # nothing) cannot hide — `FINBRIEF_EMBEDDING_MODEL` above all, since ingest and query
    # drifting apart degrades retrieval to noise with no error.
    settings = Settings.from_env(
        {
            "OPENROUTER_API_KEY": "sk-override",
            "OPENROUTER_BASE_URL": "https://proxy.example/v1",
            "FINBRIEF_CHAT_MODEL": "anthropic/claude-haiku-4.5",
            "FINBRIEF_CLASSIFIER_MODEL": "google/gemini-2.5-flash-lite",
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
    assert settings.classifier_model == "google/gemini-2.5-flash-lite"
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


# --- The observability sink (T8, #10) -------------------------------------------------


def test_the_log_file_is_off_unless_a_path_is_named():
    # Default *off*, which is what keeps the hermetic suite from writing files: conftest
    # strips `FINBRIEF_`, so every test resolves this path and must resolve it to nothing.
    assert resolve_log_file({}) is None
    assert resolve_log_file({"FINBRIEF_LOG_FILE": ""}) is None
    assert resolve_log_file({"FINBRIEF_LOG_FILE": "   "}) is None


def test_a_named_log_file_resolves_to_that_path():
    assert resolve_log_file({"FINBRIEF_LOG_FILE": "data/events.jsonl"}) == Path(
        "data/events.jsonl"
    )


# --- `.env.example`'s recommendation, bound to `.gitignore` ----------------------------
#
# One fact in two files with nothing between them: `.env.example` recommends a path and
# `.gitignore` has to already ignore it. The rules there cover `data/` wholesale *and*
# `events.jsonl` by name, which looks like belt and braces — but the braces only hold for a
# changed *directory*. A changed **filename** (`logs/run.jsonl`, measured) is matched by
# neither, and the run log it makes committable carries blocked questions' normalised text
# (issue #10 review). So the two files are bound here, through git's own matcher rather than
# through a description of it.

REPO_ROOT = Path(__file__).parents[1]
ENV_EXAMPLE = REPO_ROOT / ".env.example"
#: Matches the recommendation whether it is commented out (which it is, and must be — see
#: `resolve_log_file`) or live, so commenting it in or out cannot silently unbind this test.
_LOG_FILE_LINE = re.compile(r"^\s*#?\s*FINBRIEF_LOG_FILE\s*=\s*(\S+)\s*$", re.MULTILINE)


def recommended_log_path() -> str:
    (path,) = _LOG_FILE_LINE.findall(ENV_EXAMPLE.read_text(encoding="utf-8"))
    return path


def test_the_env_example_recommendation_is_commented_out():
    # The default-off half of the same fact: `cp .env.example .env` is the documented setup
    # step, so a live line here is a sink every reader enables without deciding to.
    body = ENV_EXAMPLE.read_text(encoding="utf-8")
    assert re.search(r"^\s*#\s*FINBRIEF_LOG_FILE\s*=", body, re.MULTILINE), (
        "shipping this uncommented makes 'off by default' false for anyone following the README"
    )
    assert not re.search(r"^FINBRIEF_LOG_FILE\s*=", body, re.MULTILINE)


def test_the_recommended_sink_path_is_already_gitignored():
    """Git's own matcher, not a substring search of `.gitignore`.

    A run log is an artifact and carries the one user-derived field in the whole log. Asserting
    that `.gitignore` *mentions* something would be a claim about prose; `git check-ignore` is
    the thing that actually decides, and it is what fails here if the recommendation moves to a
    filename the rules do not cover.
    """
    path = recommended_log_path()
    result = subprocess.run(  # noqa: S603 — fixed argv, no shell, repo-local
        ["git", "check-ignore", "--quiet", path],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode > 1:
        pytest.skip(f"git could not answer for {path!r}: {result.stderr.decode().strip()}")
    assert result.returncode == 0, (
        f".env.example recommends {path!r} and .gitignore does not ignore it, so a recorded "
        "run — including blocked questions' normalised text — is committable"
    )


def test_the_binding_notices_a_recommendation_git_would_not_ignore():
    # The check that the check works, because a `check-ignore` wrapper that always returned 0
    # would pass the test above forever. `logs/run.jsonl` is the concrete counter-example from
    # the review: it matches neither `data/` nor `events.jsonl`.
    result = subprocess.run(  # noqa: S603 — fixed argv, no shell, repo-local
        ["git", "check-ignore", "--quiet", "logs/run.jsonl"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode > 1:
        pytest.skip("git could not answer")
    assert result.returncode == 1, (
        "if this is ignored too, the test above cannot distinguish a covered path from any path"
    )


# --------------------------------------------------------------------------------------
# Query-side entity normalisation (ADR-0004 amendment, T6)
# --------------------------------------------------------------------------------------


def test_every_universe_company_declares_a_form_an_analyst_would_actually_type():
    # `retrieval/query_translation.py` substitutes a name for its ticker, and it can only do
    # that for forms declared here. A company whose only declared form is its legal name
    # ("The Goldman Sachs Group, Inc.") is one nobody's question will ever match.
    for company in UNIVERSE:
        assert company.aliases, f"{company.ticker} has no shortened name form"
        assert all(alias.strip() == alias and alias for alias in company.aliases)


def test_no_name_form_is_claimed_by_two_companies():
    # An ambiguous form would silently resolve a question about one filer into a query about
    # another — the worst failure this feature can have, and one no downstream test would catch,
    # since the substituted query still retrieves *something*.
    claimed: dict[str, str] = {}
    for company in UNIVERSE:
        for form in (company.name, *company.aliases):
            key = form.lower()
            assert key not in claimed or claimed[key] == company.ticker, (
                f"{form!r} is claimed by both {claimed.get(key)} and {company.ticker}"
            )
            claimed[key] = company.ticker


def test_a_single_character_ticker_is_left_out_of_the_normalisation_map():
    # Ford, and the reason is measured: `f` is a token in 534 of the ingested collection's 5,842
    # chunks across five filers (a single letter is also a footnote marker and a table label),
    # while `ford` is in 232 across one. Substituting the ticker would make the query *less*
    # discriminating. The name still works, because the original query is always retained.
    assert "F" in TICKERS
    assert "F" not in TICKER_BY_COMPANY_NAME.values()
    assert "ford" not in TICKER_BY_COMPANY_NAME
    assert min(len(t) for t in TICKER_BY_COMPANY_NAME.values()) >= MIN_LEXICAL_TICKER_CHARS


def test_the_normalisation_map_only_ever_yields_universe_tickers():
    # It is derived from `UNIVERSE`, so this is the guard that it stays derived: a hand-added
    # entry pointing at a ticker the KB does not hold would make BM25 search for a filer that
    # is not in the collection.
    assert set(TICKER_BY_COMPANY_NAME.values()) <= TICKERS
    assert all(form == form.lower() for form in TICKER_BY_COMPANY_NAME)


def test_every_normalisable_company_can_be_reached_by_its_legal_name_too():
    # The legal name is what a filing calls the filer, so it is the form a question quoting a
    # filing uses. It is included automatically rather than repeated in `aliases`.
    for company in UNIVERSE:
        if len(company.ticker) >= MIN_LEXICAL_TICKER_CHARS:
            assert TICKER_BY_COMPANY_NAME[company.name.lower()] == company.ticker
