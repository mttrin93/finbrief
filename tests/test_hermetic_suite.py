"""The hermetic contract, tested **per backend** rather than asserted (CLAUDE.md).

The mechanisms in `conftest.py` that make the "no network" half of that contract true all fail
*open*: a redirect pointing at a missing file reinstates the download it was hiding, and a guard
that patched the wrong function blocks nothing while every test still passes. This file is what
notices — one test per egress path, because the guard is a denylist over the backends this repo
can reach and not a proof that none exists.

It exists because the failure has a history, now **six** instances long. Twice it was
`tiktoken.get_encoding`, which downloads its BPE table and caches it under
`TIKTOKEN_CACHE_DIR`, so the author's warm cache passed and clean CI egressed silently. The
third was the *guard's own first draft*, which claimed to cover `curl_cffi` and did not — found
by `test_curl_cffi_cannot_reach_out_even_though_it_bypasses_the_socket_layer` below, which
returned a live HTTP 429 before the guard grew a second half. The fourth was the guard's
*second* draft, which patched `getaddrinfo` while a comment in **this file** asserted that
`gethostbyname` routed through it; it does not, and the call returned a real address.

The fifth and sixth arrived together with T7's one new dependency (#8), which is the point about
a new dependency being a new path. `guardrails-ai` POSTs anonymous validation metrics to its own
endpoint unless configured otherwise — so a library added *for* a security control would have
sent a record of every validated answer to a third party. And it reaches `uvloop`, whose event
loop resolves DNS inside libuv: importing it is harmless, but `guardrails.validator_service`
installs it as the **process-wide** event loop policy on every `Guard.validate`, which moved
every subsequent async lookup outside a guard written in Python. Both have a test below, and
both were found by running this file rather than by reading the dependency tree.

Two lessons are built into the shape of this file. A per-backend test is what turns the next
instance into a red test instead of a quiet packet — so a networking dependency without a case
here is an uncovered path by default. And a *claim in a comment cannot fail*: three of the six
instances were prose asserting coverage the code lacked, so each test below exercises the call
it is about rather than reasoning about it.
"""

from __future__ import annotations

import socket
import urllib.request

import pytest
import tiktoken
from conftest import TIKTOKEN_CL100K, EgressBlocked


def test_a_tcp_connection_to_a_real_host_is_blocked():
    # The guard's own subject. Asserted at the lowest level a data source can use, so it holds
    # for anything layered on top of it.
    with pytest.raises(EgressBlocked):
        socket.create_connection(("query1.finance.yahoo.com", 443), timeout=1)


@pytest.mark.parametrize(
    ("name", "lookup"),
    [
        ("getaddrinfo", lambda: socket.getaddrinfo("feeds.finance.yahoo.com", 443)),
        ("gethostbyname", lambda: socket.gethostbyname("feeds.finance.yahoo.com")),
        ("gethostbyname_ex", lambda: socket.gethostbyname_ex("feeds.finance.yahoo.com")),
        ("gethostbyaddr", lambda: socket.gethostbyaddr("104.20.23.154")),
    ],
)
def test_a_dns_lookup_for_a_real_host_is_blocked(name, lookup):
    # Blocked as well as `connect`, because a resolution is already a packet leaving the
    # machine — and because failing here names the host, where a connect timeout names a
    # socket.
    #
    # **Each resolver is called, not reasoned about.** This test's previous shape exercised
    # `getaddrinfo` alone and carried a comment asserting "`socket.gethostbyname` routes through
    # `getaddrinfo`" — which is false. Each of these is its own CPython C entry point into the
    # platform resolver, and with only `getaddrinfo` patched `gethostbyname` returned a real
    # address (issue #9 review). A comment cannot fail; a parametrised call can, so the claim
    # now costs four lines and proves itself. Add a resolver to `conftest`, add a case here.
    with pytest.raises(EgressBlocked):
        lookup()


def test_urllib_cannot_reach_out():
    # `urllib` bottoms out in `create_connection`, so the socket guard covers it. This is the
    # backend `finance/news.py` uses for the RSS feed.
    with pytest.raises((EgressBlocked, OSError)) as caught:
        urllib.request.urlopen("https://query1.finance.yahoo.com/", timeout=1)  # noqa: S310
    assert "hermetic" in str(caught.value) or isinstance(
        caught.value.__cause__ or caught.value, EgressBlocked
    )


def test_httpx_cannot_reach_out():
    # The backend the **paid** path uses, and the one with the most to lose: the `openai` SDK
    # that `retrieval/embeddings.py` constructs speaks through `httpx`, and so does `chromadb`.
    # It bottoms out in `create_connection` via its own transport, so the socket guard covers
    # it — but "covered by inference" is what the `curl_cffi` hole was too, so it gets a test
    # (issue #9 review).
    httpx = pytest.importorskip("httpx")

    with pytest.raises((EgressBlocked, httpx.HTTPError)) as caught:
        httpx.get("https://query1.finance.yahoo.com/", timeout=1)
    assert "hermetic" in str(caught.value) or isinstance(
        caught.value.__cause__ or caught.value, EgressBlocked
    )


def test_aiohttp_cannot_reach_out():
    # `aiohttp` arrives transitively and resolves through its own `ThreadedResolver`, which
    # calls `getaddrinfo` in an executor rather than on the event loop — a different call path
    # to `httpx`'s, and the reason it is worth its own test rather than being folded into one
    # "async HTTP" case.
    aiohttp = pytest.importorskip("aiohttp")
    asyncio = pytest.importorskip("asyncio")

    async def fetch():
        async with aiohttp.ClientSession() as session:  # noqa: SIM117 — the nesting is the API
            async with session.get("https://query1.finance.yahoo.com/"):
                pass

    with pytest.raises((EgressBlocked, aiohttp.ClientError, OSError)) as caught:
        asyncio.run(fetch())
    assert "hermetic" in str(caught.value) or isinstance(
        caught.value.__cause__ or caught.value, EgressBlocked
    )


def test_curl_cffi_cannot_reach_out_even_though_it_bypasses_the_socket_layer():
    # **The test that found the guard's first draft wrong.** That draft's docstring claimed
    # `socket.create_connection` was "what `urllib3`, and so `requests` and `curl_cffi`, calls".
    # Only the first half is true: `curl_cffi` binds libcurl, which resolves and connects in C.
    # With the socket guard installed and nothing else, this exact call returned HTTP 429 — a
    # live request, from inside a suite advertised as hermetic.
    #
    # It is not a hypothetical backend either: `yfinance._http.new_session()` returns
    # `curl_cffi.Session(impersonate="chrome")` whenever `curl_cffi` imports, so the library
    # this ticket added is the one that was uncovered.
    curl_cffi = pytest.importorskip("curl_cffi")

    with pytest.raises(EgressBlocked):
        curl_cffi.requests.get("https://query1.finance.yahoo.com/", timeout=1)


def test_a_curl_cffi_session_cannot_reach_out_either():
    # The shape yfinance actually constructs, rather than the module-level convenience helper:
    # both funnel through `Curl.perform`, which is why that is where the guard sits.
    curl_cffi = pytest.importorskip("curl_cffi")

    with pytest.raises(EgressBlocked), curl_cffi.Session(impersonate="chrome") as session:
        session.get("https://query1.finance.yahoo.com/", timeout=1)


def test_the_yfinance_backend_is_one_the_guard_covers():
    # The general claim ("no egress") is not provable in-process, so this asserts the specific
    # one that matters: the session type the paid data path would build is a backend the guard
    # knows about. A yfinance release that switched to a third HTTP library would fail here
    # rather than quietly egressing — which is the failure mode that has now happened twice.
    yfinance_http = pytest.importorskip("yfinance._http")

    session = yfinance_http.new_session()

    assert type(session).__module__.split(".")[0] in {"curl_cffi", "requests"}


def test_loopback_is_still_reachable():
    # The guard must not be a blanket ban: `AppTest` and a Chroma client are local processes,
    # and a guard that broke them would be turned off rather than fixed. Nothing listens on
    # this port, so a *refusal* is the pass — what must not happen is `EgressBlocked`.
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection(("127.0.0.1", 9), timeout=1)


def test_a_unix_socket_path_is_not_mistaken_for_a_host():
    # An `AF_UNIX` address is a filesystem path, not a `(host, port)` pair. Treating it as one
    # would read the path's first character as a hostname and block it.
    with (
        pytest.raises((FileNotFoundError, ConnectionRefusedError, OSError)) as caught,
        socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as unix,
    ):
        unix.connect("/tmp/finbrief-nothing-listens-here.sock")
    assert not isinstance(caught.value, EgressBlocked)


def test_uvloop_cannot_resolve_even_though_it_bypasses_the_socket_layer():
    # **The `curl_cffi` hole a second time, found in T7 (#8).** uvloop's event loop is Cython
    # over libuv: `Loop.getaddrinfo` calls `uv_getaddrinfo` in C and never consults the patched
    # `socket.getaddrinfo`. Measured before `conftest._block_uvloop_resolution` existed, this
    # exact call returned a real Yahoo address while the same call on the stock
    # `_UnixSelectorEventLoop` raised — with the socket guard fully installed and
    # `socket.getaddrinfo` verifiably patched.
    #
    # It has been importable since Phase 0, arriving through `uvicorn[standard]` under both
    # `streamlit` and `chromadb`. What made it *reachable* was `guardrails.validator_service`,
    # which sets the process-wide event loop policy to uvloop on every `Guard.validate`;
    # `security/advice.py` stops that with `GUARDRAILS_RUN_SYNC` — which is why this test builds
    # a uvloop loop **itself** rather than relying on some other code path to install one. A
    # test that only passed because nothing currently uses the backend is a test of today's
    # imports.
    uvloop = pytest.importorskip("uvloop")

    loop = uvloop.new_event_loop()
    try:
        with pytest.raises(EgressBlocked):
            loop.run_until_complete(loop.getaddrinfo("feeds.finance.yahoo.com", 443))
    finally:
        loop.close()


def test_a_uvloop_loop_can_still_resolve_loopback():
    # The same non-blanket requirement `test_loopback_is_still_reachable` states for sockets: a
    # guard that broke local resolution would be switched off rather than fixed.
    uvloop = pytest.importorskip("uvloop")

    loop = uvloop.new_event_loop()
    try:
        assert loop.run_until_complete(loop.getaddrinfo("127.0.0.1", 9))
    finally:
        loop.close()


def test_guardrails_validation_makes_no_call_of_its_own():
    # **The fifth instance of the pattern, caught before it shipped** (T7, #8). `guardrails-ai`
    # posts anonymous validation metrics to its own endpoint unless `~/.guardrailsrc` says
    # otherwise, and `enable_metrics` defaults to `True` — so a library added *for* a security
    # control would, unconfigured, send a record of every validated answer to a third party.
    # Measured with this guard installed and `security/advice.py`'s telemetry switch removed:
    # `Guard.validate` resolved `hty0gc1ok3.execute-api.us-east-1.amazonaws.com`.
    #
    # This is the per-backend test for that path, and it exercises the *shipped* call — the real
    # `Guard`, over both a passing and a failing answer, since only one of the two produces a
    # `FailResult` for the tracer to report. A comment in `advice.py` asserting telemetry is off
    # is exactly the kind of claim three of the four earlier instances were.
    from finbrief.security.advice import validate_answer

    assert validate_answer("Tesla identifies supply chain risks [1].").refused is False
    assert validate_answer("You should buy Tesla now.").refused is True


def test_tiktoken_resolves_its_table_from_the_vendored_fixture():
    # The specific breach that shipped twice, now checkable in one line: with egress blocked,
    # `get_encoding` either finds the vendored table or raises. That it returns is the proof
    # the redirect works — and `conftest.offline_tiktoken` asserts the file is there, so a
    # deleted fixture fails loudly instead of downloading.
    assert TIKTOKEN_CL100K.exists()

    encoding = tiktoken.get_encoding("cl100k_base")

    assert encoding.encode("Item 1A. Risk Factors"), "the BPE table loaded, with no network"
