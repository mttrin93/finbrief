"""The hermetic contract, tested rather than asserted (CLAUDE.md).

Two mechanisms in `conftest.py` make the "no network" half of that contract true, and both of
them are the kind that fail *open*: a redirect that points at a missing file reinstates the
download it was hiding, and an egress guard that patched the wrong function blocks nothing
while every test still passes. This file is what notices.

It exists because the failure has a history. Two tickets shipped a "hermetic" claim that was
false in the same way — `tiktoken.get_encoding` downloads its BPE table and caches it under
`TIKTOKEN_CACHE_DIR`, so the author's warm cache passed and clean CI egressed silently. Both
times the fix was to vendor the table; neither time was anything left behind that would catch
the third instance. These are the tests that would have caught the first two.
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


def test_an_http_client_cannot_reach_out_either():
    # The path a data source actually takes: `urllib`, `requests` and `curl_cffi` all bottom
    # out in `create_connection` or `getaddrinfo`, which is why the guard patches both rather
    # than trying to intercept one library's session object.
    with pytest.raises((EgressBlocked, OSError)) as caught:
        urllib.request.urlopen("https://query1.finance.yahoo.com/", timeout=1)  # noqa: S310
    assert "hermetic" in str(caught.value) or isinstance(
        caught.value.__cause__ or caught.value, EgressBlocked
    )


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
