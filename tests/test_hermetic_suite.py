"""The hermetic contract, tested **per backend** rather than asserted (CLAUDE.md).

The mechanisms in `conftest.py` that make the "no network" half of that contract true all fail
*open*: a redirect pointing at a missing file reinstates the download it was hiding, and a guard
that patched the wrong function blocks nothing while every test still passes. This file is what
notices — one test per egress path, because the guard is a denylist over the backends this repo
can reach and not a proof that none exists.

It exists because the failure has a history, now three instances long. Twice it was
`tiktoken.get_encoding`, which downloads its BPE table and caches it under
`TIKTOKEN_CACHE_DIR`, so the author's warm cache passed and clean CI egressed silently. The
third was the *guard's own first draft*, which claimed to cover `curl_cffi` and did not — found
by `test_curl_cffi_cannot_reach_out_even_though_it_bypasses_the_socket_layer` below, which
returned a live HTTP 429 before the guard grew a second half. A per-backend test is what turns
the next instance into a red test instead of a quiet packet.
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


def test_a_dns_lookup_for_a_real_host_is_blocked():
    # Blocked as well as `connect`, because a resolution is already a packet leaving the
    # machine — and because failing here names the host, where a connect timeout names a
    # socket. `socket.gethostbyname` routes through `getaddrinfo`.
    with pytest.raises(EgressBlocked):
        socket.getaddrinfo("feeds.finance.yahoo.com", 443)


def test_urllib_cannot_reach_out():
    # `urllib` bottoms out in `create_connection`, so the socket guard covers it. This is the
    # backend `finance/news.py` uses for the RSS feed.
    with pytest.raises((EgressBlocked, OSError)) as caught:
        urllib.request.urlopen("https://query1.finance.yahoo.com/", timeout=1)  # noqa: S310
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


def test_tiktoken_resolves_its_table_from_the_vendored_fixture():
    # The specific breach that shipped twice, now checkable in one line: with egress blocked,
    # `get_encoding` either finds the vendored table or raises. That it returns is the proof
    # the redirect works — and `conftest.offline_tiktoken` asserts the file is there, so a
    # deleted fixture fails loudly instead of downloading.
    assert TIKTOKEN_CL100K.exists()

    encoding = tiktoken.get_encoding("cl100k_base")

    assert encoding.encode("Item 1A. Risk Factors"), "the BPE table loaded, with no network"
