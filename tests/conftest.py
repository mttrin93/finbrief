"""Test isolation: no `.env`, no cached settings, no network, no leaked logger state.

Tests must not depend on the developer's local `.env` or exported shell variables — a
suite that passes only on a machine with a key is worse than no suite. Everything a test
needs comes from `monkeypatch.setenv` or an explicit mapping.

The no-network half is **enforced from here** rather than asserted (`_install_egress_guard`
below), because it had been asserted twice and had been false twice. The recorded market and
news fixtures it makes unnecessary to fetch are loaded by `fakes.py`, which plain helper
functions can reach as well as fixtures can.
"""

import gzip
import json
import logging
import os
import socket
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fakes
import pytest
from fakes import KeywordEmbeddings

from finbrief import config
from finbrief.ingestion.model import ExtractedFiling, FilingRef, Section
from finbrief.observability.logging_setup import PACKAGE_LOGGER

# --------------------------------------------------------------------------------------
# The no-network half of the hermetic contract, enforced (CLAUDE.md)
# --------------------------------------------------------------------------------------

#: Hosts a test may reach. Loopback only, and it is here for the local processes a test
#: legitimately talks to (Streamlit's `AppTest` machinery, a Chroma client) — never for a
#: data source.
_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost", "", "0.0.0.0"})


class EgressBlocked(RuntimeError):
    """A test tried to reach the network. That is a defect in the test, not in the guard."""


#: Every target this guard has refused, in order, for the whole session.
#:
#: **Because raising is only loud if the caller lets the exception through**, and a library that
#: wraps its own side channel in `try/except` swallows it. That is not hypothetical: with
#: `security/advice.py`'s telemetry switch removed, `guardrails-ai` POSTs to its endpoint from
#: OpenTelemetry's `BatchSpanProcessor`, which catches the `EgressBlocked` on its export thread
#: and logs `Exception while exporting Span.` — so `Guard.validate` returns a completely normal
#: verdict, and a test asserting only on that verdict passes while a packet is being attempted
#: (issue #8 review). A recorded attempt cannot be swallowed: the append happens inside the
#: refusal, before any caller sees it.
#:
#: Compare a `len()` before and after the call under test rather than asserting the whole
#: list is empty: another test's legitimate assertion of a blocked call is not this one's
#: failure.
EGRESS_ATTEMPTS: list[str] = []


def _blocked(target: object, hint: str = "") -> EgressBlocked:
    """The one refusal message, shared by all three backends the guard covers.

    Shared because a per-backend wording is a per-backend thing to get out of step, and because
    `tests/test_hermetic_suite.py` asserts on the word "hermetic" appearing in whatever a
    wrapping library re-raises.

    Records the target in `EGRESS_ATTEMPTS` on the way past, for the reason that list gives.
    """
    EGRESS_ATTEMPTS.append(str(target))
    return EgressBlocked(
        f"the test suite tried to reach {target!r}. Tests are hermetic — no network "
        f"(CLAUDE.md). Record a fixture for this data instead of fetching it: see "
        f"scripts/record_market_fixtures.py and scripts/record_edgar_fixtures.py.{hint}"
    )


def _local_host(host: object) -> bool:
    return host is None or host in _LOOPBACK


def _local_address(address: object) -> bool:
    if not isinstance(address, tuple) or not address:
        return True  # an AF_UNIX path, which never leaves the machine
    return _local_host(address[0])


def _install_egress_guard() -> None:
    """Make an outbound connection raise, for the whole session, from collection onwards.

    **Installed at conftest import rather than in a fixture, and deliberately.** A fixture
    runs per test, and the breach worth catching is the one CLAUDE.md names explicitly — "never
    add a test dependency that fetches data at import time" — which happens while pytest is
    *collecting*, before any fixture exists. conftest is imported before the test modules it
    collects, so this is the earliest hook there is.

    **Why a guard at all, given the suite was already hermetic by contract.** Six times now a
    ticket has shipped a "hermetic" claim that was false. Twice it was `tiktoken.get_encoding`,
    which downloads its BPE table and caches it, so the author's warm cache passed and clean CI
    egressed silently. The third time was **this function's own first draft**, whose docstring
    claimed it covered `curl_cffi` and did not — see `_block_curl_cffi` below. The fourth was
    **this function's second draft**, which patched `getaddrinfo` and left the three
    `gethostby*` resolvers open — see the resolver note below. The fifth and sixth came with
    T7's one dependency (#8): `guardrails-ai` posts validation telemetry to its own endpoint,
    and it installs `uvloop` as the process-wide event loop policy, which resolves DNS in libuv
    — see `_block_uvloop_resolution` below. An assertion that the suite does not reach the
    network is worth exactly as much as the author's cache state; this is the mechanism that
    makes it worth more, and `tests/test_hermetic_suite.py` is what says the mechanism works,
    backend by backend.

    Three of the six were a *docstring or comment* claiming coverage the code lacked, which is
    the pattern to distrust: prose about a guard cannot fail, so every claim one of these makes
    now has a test that exercises the call rather than describing it.

    **Raising is not the whole mechanism, and `EGRESS_ATTEMPTS` is the other half.** A guard
    that only raises is loud only when the caller propagates the exception; a library that wraps
    its own side channel swallows it and the call looks clean from outside. See that list.

    **What it is: a denylist over the egress backends this repo can reach, not a proof.**
    Python has no in-process way to stop a C library from opening a socket, so a guard like
    this can only cover the paths it knows about, and a new HTTP dependency is a new path. That
    limit is stated rather than papered over — a guard advertised as total is how the third
    instance happened. What makes it worth having anyway is that the failure mode is now
    *loud*: an uncovered backend shows up as a live call in `test_hermetic_suite.py`'s per-
    backend tests, which is where the `curl_cffi` hole was found.

    This function covers the **Python socket layer** — `connect`/`connect_ex` (the low-level
    path), `create_connection` (what `urllib3`, and so `requests`, calls), and the **four
    resolver entry points**, because a DNS query is egress too and blocking it fails earlier and
    reads more clearly than a connect timeout. Anything with a non-tuple address is left alone:
    an `AF_UNIX` path.

    **Why four resolvers and not just `getaddrinfo`.** This function's *second* draft patched
    `getaddrinfo` alone and a comment in `test_hermetic_suite.py` asserted that
    "`socket.gethostbyname` routes through `getaddrinfo`". It does not: `gethostbyname`,
    `gethostbyname_ex` and `gethostbyaddr` are each their own CPython C entry point
    (`socket_gethostbyname` and friends in `socketmodule.c`), calling the platform resolver
    directly and touching the Python-level `getaddrinfo` this guard replaces not at all.
    Measured with only `getaddrinfo` patched, `socket.gethostbyname("example.com")` returned
    `104.20.23.154` — a real DNS query, from inside a suite advertised as hermetic. That is the
    **fourth** false hermetic claim on this repo, and the third to be a docstring or comment
    claiming coverage the code lacked, which is why the tests below now *call* each resolver
    instead of asserting about it in prose (issue #9 review).
    """
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_create_connection = socket.create_connection
    real_getaddrinfo = socket.getaddrinfo
    real_gethostbyname = socket.gethostbyname
    real_gethostbyname_ex = socket.gethostbyname_ex
    real_gethostbyaddr = socket.gethostbyaddr

    def guarded_connect(self, address):
        if not _local_address(address):
            raise _blocked(address)
        return real_connect(self, address)

    def guarded_connect_ex(self, address):
        if not _local_address(address):
            raise _blocked(address)
        return real_connect_ex(self, address)

    def guarded_create_connection(address, *args, **kwargs):
        if not _local_address(address):
            raise _blocked(address)
        return real_create_connection(address, *args, **kwargs)

    def guarded_getaddrinfo(host, *args, **kwargs):
        if not _local_host(host):
            raise _blocked(host)
        return real_getaddrinfo(host, *args, **kwargs)

    def guarded_gethostbyname(hostname):
        if not _local_host(hostname):
            raise _blocked(hostname)
        return real_gethostbyname(hostname)

    def guarded_gethostbyname_ex(hostname):
        if not _local_host(hostname):
            raise _blocked(hostname)
        return real_gethostbyname_ex(hostname)

    def guarded_gethostbyaddr(ip_address):
        # A reverse lookup queries a PTR record, so the address here is the *question* asked of
        # the resolver rather than a host being connected to — but it is still a packet, and
        # `_local_host` reads it the same way.
        if not _local_host(ip_address):
            raise _blocked(ip_address)
        return real_gethostbyaddr(ip_address)

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    socket.create_connection = guarded_create_connection
    socket.getaddrinfo = guarded_getaddrinfo
    socket.gethostbyname = guarded_gethostbyname
    socket.gethostbyname_ex = guarded_gethostbyname_ex
    socket.gethostbyaddr = guarded_gethostbyaddr


def _block_curl_cffi() -> None:
    """Block `curl_cffi`, which does its own DNS and connect inside libcurl.

    **The hole `_install_egress_guard`'s first draft had, and claimed it did not.** That
    docstring asserted `socket.create_connection` was "what `urllib3`, and so `requests` and
    `curl_cffi`, calls". Only the first half is true: `curl_cffi` binds libcurl, which resolves
    and connects in C without touching Python's `socket` module at all. Measured with the socket
    guard installed, `curl_cffi.requests.get("https://query1.finance.yahoo.com/")` returned
    **HTTP 429 — a live call**.

    That is not a hypothetical dependency. `yfinance._http.new_session()` returns
    `curl_cffi.Session(impersonate="chrome")` whenever `curl_cffi` is importable, and it is in
    `uv.lock`, so the one library this ticket added is the one the guard did not cover. The
    suite stayed hermetic only because `finance/quotes.py` defers its `import yfinance` and
    every test injects a recorded source — hermetic *by contract*, which is the exact thing the
    guard exists to stop relying on.

    **`Curl.perform` is the chokepoint**, the libcurl analogue of `socket.connect`: every
    synchronous request funnels through it whatever session or convenience helper called in.
    `AsyncSession.request` covers the async path, which uses multi-handles instead.

    **No loopback exemption, unlike the socket guard.** The target URL is set through `setopt`
    and is not an argument to `perform`, so this cannot tell a local address from a remote one —
    and nothing in this repo has any reason to use `curl_cffi` for loopback. Blocking it
    outright is the safe reading of an ambiguous case.

    Absent `curl_cffi`, this is a no-op: it is an optional `yfinance` backend, and a guard that
    hard-required it would fail the suite on an environment that is *more* hermetic, not less.
    """
    try:
        import curl_cffi
    except ImportError:  # pragma: no cover — the more-hermetic environment
        return

    def blocked_perform(*args: object, **kwargs: object) -> None:
        # Recorded like every other refusal, even though the target is unknowable here: the
        # target URL is set through `setopt` and is not an argument to `perform`. A backend that
        # did not append would be a backend `EGRESS_ATTEMPTS` cannot see, which is the same
        # denylist-with-a-hole shape the guard itself warns about.
        EGRESS_ATTEMPTS.append("curl_cffi (libcurl, target set via setopt)")
        raise EgressBlocked(
            "the test suite tried to reach the network through curl_cffi (libcurl), which "
            "bypasses Python's socket layer. Tests are hermetic — no network (CLAUDE.md). "
            "yfinance uses this backend; inject a recorded source instead, as "
            "tests/fakes.py's `a_quote_source` does."
        )

    curl_cffi.Curl.perform = blocked_perform
    curl_cffi.AsyncSession.request = blocked_perform


def _block_uvloop_resolution() -> None:
    """Block `uvloop`, which resolves and connects in libuv without touching Python's `socket`.

    **The `curl_cffi` story a second time, found in T7 (#8) and worth the same treatment.**
    uvloop replaces asyncio's event loop with a Cython one over libuv: `Loop.getaddrinfo` calls
    `uv_getaddrinfo` in C, so `socket.getaddrinfo` — patched above, in Python — is never
    consulted. Measured: with the socket guard installed and nothing else, `loop.getaddrinfo` on
    a uvloop event loop returned a real Yahoo address, while the same call on the stock
    `_UnixSelectorEventLoop` raised `EgressBlocked`.

    **It is not a dependency anyone chose.** uvloop arrives transitively through
    `uvicorn[standard]`, which both `streamlit` and `chromadb` depend on, so it has been
    importable since Phase 0. What made it *reachable* is T7:
    `guardrails.validator_service.get_loop` calls
    `asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())` on every `Guard.validate`, which
    swapped the policy for the whole process and put every subsequent async DNS lookup outside
    the guard. `security/advice.py` stops that happening in the app (`GUARDRAILS_RUN_SYNC`), and
    this exists anyway: the reason `_block_curl_cffi` is a separate function rather than a fixed
    comment is that a backend the guard cannot see is a hole whether or not today's code walks
    through it.

    Only the two resolvers and `create_connection` are covered, and the limit is stated rather
    than implied: `sock_connect` to a literal address through libuv is still C, and the guard
    remains a denylist over the backends this repo can reach rather than a proof (see
    `_install_egress_guard`). Blocking resolution is what covers a *hostname*, which is the only
    form a data source is named in here.
    """
    try:
        import uvloop
    except ImportError:  # pragma: no cover — the more-hermetic environment
        return

    real_getaddrinfo = uvloop.Loop.getaddrinfo
    real_getnameinfo = uvloop.Loop.getnameinfo
    real_create_connection = uvloop.Loop.create_connection

    async def guarded_getaddrinfo(self, host, *args, **kwargs):
        if not _local_host(host):
            raise _blocked(host, " uvloop resolves in libuv, not through Python's socket.")
        return await real_getaddrinfo(self, host, *args, **kwargs)

    async def guarded_getnameinfo(self, sockaddr, *args, **kwargs):
        if not _local_address(sockaddr):
            raise _blocked(sockaddr, " uvloop resolves in libuv, not through Python's socket.")
        return await real_getnameinfo(self, sockaddr, *args, **kwargs)

    async def guarded_create_connection(self, factory=None, host=None, *args, **kwargs):
        if not _local_host(host):
            raise _blocked(host, " uvloop connects in libuv, not through Python's socket.")
        return await real_create_connection(self, factory, host, *args, **kwargs)

    uvloop.Loop.getaddrinfo = guarded_getaddrinfo
    uvloop.Loop.getnameinfo = guarded_getnameinfo
    uvloop.Loop.create_connection = guarded_create_connection


def _silence_ragas_telemetry() -> None:
    """Turn off `ragas`' analytics POST for the whole session, before anything imports it.

    The **seventh** egress path (T10, #11), and the second one that a dependency added *for*
    measurement opens by default: `ragas._analytics.track` POSTs every metric completion to
    `t.explodinggradients.com`. Measured with this guard installed and the switch absent — a
    single `track()` call attempted `t.explodinggradients.com`, raised nothing, logged nothing,
    and appeared in `EGRESS_ATTEMPTS`; `track` is decorated `@silent`.

    Set here **as well as** in `finbrief.evaluation.judge`, which is what covers the script, for
    the reason `security/advice.py` gives about the guardrails switch: two mechanisms that fail
    independently, and neither is allowed to be the only one. Note the switch has to be in
    place before the first read, because `ragas._analytics.do_not_track` is `lru_cache`d — which
    is why this is a module-level call at conftest import and not a fixture.

    Assigned rather than `setdefault`: a developer with `RAGAS_DO_NOT_TRACK=false` exported
    would otherwise run the suite with tracking live.
    """
    os.environ["RAGAS_DO_NOT_TRACK"] = "true"


_install_egress_guard()
_block_curl_cffi()
_block_uvloop_resolution()
_silence_ragas_telemetry()

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


@pytest.fixture(scope="session")
def recorded_quotes():
    """Every Universe company's real quote, parsed from its recorded `.info` and closes.

    The loader lives in `fakes.py` because plain helper functions need it too — `test_agent`
    builds a real agent outside any fixture — and two loaders would be two places the fixture
    layout is known.
    """
    return fakes.recorded_quotes()


@pytest.fixture(scope="session")
def recorded_feeds():
    """ticker -> the raw RSS bytes Yahoo served, verbatim (see `fakes.recorded_feed_bytes`)."""
    return fakes.recorded_feed_bytes()


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
def planted_store(tmp_path):
    """The **dedicated injection-test collection** ADR-0006 requires — never the demo KB.

    Real Chroma, real chunking metadata, a fake embedding, and bodies that are attack payloads
    (`finbrief.security.corpus.PLANTED_PAYLOADS`). Separate from `filings_store` on purpose: the
    point of the exercise is that retrieval treats a poisoned chunk exactly as it treats an
    EDGAR one, and a payload written into the collection real questions are answered from would
    make every other test's fixture adversarial.

    A chunk from here is identifiable on sight by its provenance rather than its ticker: the
    accession is all zeroes and the fiscal year is 1970. The ticker is deliberately a real
    Universe member — `corpus.PLANTED_PAYLOADS` records why, and it is a finding rather than a
    convenience.
    """
    from finbrief.retrieval.vectorstore import build_filings_store, write_chunks
    from finbrief.security.corpus import planted_chunks

    store = build_filings_store(
        persist_directory=str(tmp_path / "planted-chroma"), embeddings=KeywordEmbeddings()
    )
    write_chunks(store, planted_chunks())
    return store


@pytest.fixture
def empty_filings_store(tmp_path):
    """A `filings` collection with nothing in it — the retrieval-level fallback's input."""
    from finbrief.retrieval.vectorstore import build_filings_store

    return build_filings_store(
        persist_directory=str(tmp_path / "empty-chroma"), embeddings=KeywordEmbeddings()
    )


@pytest.fixture(autouse=True)
def offline_injection_classifier(monkeypatch):
    """Stop the input gate's layer-3 model call, leaving layers 1 and 2 real (T7, #8).

    **Autouse, because the gate is now in front of every question the app is asked.** Any
    `AppTest` that sends a message reaches `screen()`, and without this it reaches a real
    `ChatOpenAI` — which the egress guard blocks, so `classify` fails *open* and the app carries
    on. That is the designed behaviour and exactly the wrong thing in a test: the layer under
    the app's own tests would be silently absent, and the log would fill with
    `gate_classifier_unavailable` on every page render.

    **Only layer 3 is stubbed**, and the choice matters: normalisation and the denylist stay
    real, so an app-level test of the refusal path uses a *denylisted* payload and is
    deterministic without scripting anything. Patched at `input_gate`'s own name because that is
    the binding `screen` calls; the module's `classify` is still what `tests/test_input_gate.py`
    drives.

    **The stub yields to an injected model**, which is what keeps it from hiding the gate's own
    suite: `tests/test_input_gate.py` drives `screen(question, model=a_model("YES"))`, and a
    blanket `lambda: SAFE` would have made every one of those tests assert about the stub. So
    this delegates to the real `classify` whenever a caller named a model — a test that scripted
    a reply means to exercise the layer — and answers `SAFE` only for the callers that named
    none, which is the app. Being autouse rather than opt-in is deliberate for the reason
    CLAUDE.md gives about checks that cannot fail: a new `AppTest` case that forgot to request a
    fixture would make a live call and pass anyway.
    """
    from finbrief.security import input_gate
    from finbrief.security.classifier import Verdict, classify

    def offline(question, *, model=None, settings=None):
        if model is not None:
            return classify(question, model=model, settings=settings)
        return Verdict.SAFE

    monkeypatch.setattr(input_gate, "classify", offline)


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

    Yields one `Build` per construction, so a test can assert the cache built exactly one —
    and, since T14 (#15), *which model* each construction was for. The keyword arguments are
    recorded because the model the app picked is only visible here: `build_agent` is handed a
    constructed chat model, so the slug is an argument to this call and appears nowhere in the
    object it returns.
    """
    import streamlit as st

    from finbrief.agent import agent as agent_module

    st.cache_resource.clear()
    built: list[Build] = []

    def fake_build_agent(**kwargs):
        built.append(Build(agent=object(), kwargs=kwargs))
        return built[-1].agent

    monkeypatch.setattr(agent_module, "build_agent", fake_build_agent)
    yield built
    st.cache_resource.clear()


@dataclass(frozen=True, slots=True)
class Build:
    """One `build_agent` call the app made: what it returned, and what it was asked for."""

    agent: object
    kwargs: Mapping[str, Any]

    @property
    def model(self) -> str | None:
        """The slug of the chat model this agent was built with, if it was built with one.

        `None` rather than a configured default when the app passed no model at all — the
        distinction the mutation-detecting test in `test_app_state.py` turns on, since dropping
        the picker's argument is precisely what makes every build modelless.
        """
        model = self.kwargs.get("model")
        return None if model is None else getattr(model, "model_name", None) or str(model)


@pytest.fixture(autouse=True)
def pristine_package_logger():
    """Restore the `finbrief` logger, so `configure_logging` in one test can't mute another."""
    logger = logging.getLogger(PACKAGE_LOGGER)
    handlers, level, propagate = logger.handlers[:], logger.level, logger.propagate
    yield
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate
