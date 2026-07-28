"""`Headline` — recent news for one company, HTML-stripped, and the one place RSS is read.

The engine half of `get_recent_news`. Free and keyless (PLAN §4): Yahoo Finance's per-ticker
headline RSS, parsed with `feedparser`, wrapped in the same TTL + retry + stale-fallback policy
as the quotes.

**HTML stripping is a security control, not tidying.** These summaries are attacker-reachable
text: anybody who can get a post onto a syndicated feed can put a sentence in it addressed to
this assistant, and the answer would carry it into the model's context (user story 17, and
ADR-0006's indirect-injection threat). Two things follow, and only the first is here.

1. *Strip the markup, keep the words.* A summary arrives as HTML — the recorded TSLA fixture has
   `<p>` tags in it — and markup is where a payload hides from a reader while staying in the
   prompt: an `<a>` title, a comment, a `style` block. So the text is reduced to its visible
   words before anything sees it, entities and all, and `<script>`/`<style>` bodies are dropped
   entirely rather than flattened into the text.
2. *Frame it as data.* That is `prompts.py`'s job and the tool's, not this module's: the same
   quarantine wording as `sources_block`, written once (ADR-0006).

Stripping with a regex is a deliberate, bounded choice. A parser (`html.parser`, lxml) would be
the right instrument for *interpreting* the markup; what is wanted here is the opposite — throw
the markup away and keep nothing structural — and the bound that makes a regex safe is that its
output is never re-rendered as HTML. Streamlit renders these as text in a card.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from html import unescape

from finbrief.config import (
    FETCH_ATTEMPTS,
    FETCH_BACKOFF_SECONDS,
    NEWS_MAX_HEADLINES,
    NEWS_TTL_SECONDS,
)
from finbrief.finance.cache import Fetched, TimedCache

#: Yahoo Finance's per-ticker headline feed. Keyless and public; `{ticker}` is the only
#: substitution and it is always a validated Universe member, never a model's raw argument
#: (`tools/finance.py` resolves it against the whitelist first).
FEED_URL_TEMPLATE = (
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
)

#: Sent on the feed request. The endpoint serves an empty body to an unidentified client, the
#: same way the SEC does — hence a name and not a browser impersonation string.
NEWS_USER_AGENT = "finbrief/0.1 (equity-research assistant; +https://github.com/finbrief)"

#: `<script>` and `<style>` bodies are removed **with their contents**, before tags are dropped.
#: Stripping the tags first would leave a stylesheet's or a script's text in the summary — which
#: is exactly the place a payload would sit, since a reader never sees it rendered.
_SCRIPT_OR_STYLE = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL)

#: An HTML comment, likewise removed with its contents: invisible to a reader, plain text to a
#: model.
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

#: Any remaining tag, **attribute-aware**. The obvious pattern (`<[^>]*>`) gets this wrong on
#: legal markup: given `<a title="a > b" href="#">Ford</a>` it matches up to the `>` inside the
#: title and leaves `b" href="#">Ford` in the text — markup surviving the strip, which is the
#: one thing this must not do. So the pattern alternates unquoted runs with quoted attribute
#: values, and a `>` inside quotes cannot end the tag.
#:
#: It also requires a tag to *start* like one (a letter, `/`, `!` or `?`), so a bare `<` in
#: prose — "revenue < $1bn" — is left alone instead of eating the rest of the sentence. A
#: deliberate widening of what survives: a visible angle bracket is text, and the threat model
#: here is text a reader **cannot** see (see the module docstring).
_TAG = re.compile(r"""<[a-zA-Z!/?][^>"']*(?:(?:"[^"]*"|'[^']*')[^>"']*)*>""")

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class Headline:
    """One story: its title, where it came from, when, and the link to read it.

    `summary` is already stripped — there is no path into this class that leaves markup in it,
    because a class whose invariant is "the text is safe" must not have a raw constructor beside
    a stripping one.
    """

    title: str
    summary: str
    #: The publisher's domain (`fool.com`), derived from the link rather than from the feed's
    #: own `source` element, which Yahoo leaves empty on every entry in all three recorded
    #: fixtures. Shown on the card because "who says so" is half of what a headline is worth.
    source: str
    link: str
    #: ISO-8601 UTC, or `None` when the entry carried no parseable date. `None` rather than
    #: "now": a story of unknown age is not a story from this minute, and the day count the
    #: analyst asked for is filtered against this field.
    published: str | None


def strip_html(raw: str) -> str:
    """The visible words of `raw`, with markup and entities resolved. See the module docstring.

    Order is load-bearing: script/style bodies and comments go **with their contents** first,
    then tags, then entities — unescaping before the tags were dropped would turn a `&lt;b&gt;`
    in the source text into a tag this function has already stopped looking for, so an author
    could smuggle markup through by escaping it once.

    **Not idempotent, and that follows from the same order.** A publisher who wrote
    `&lt;script&gt;` meant those characters as *text*, so one pass correctly yields the visible
    string `<script>` — which a second pass would then delete as markup. Stripping twice is
    therefore lossy, and `parse_feed` is the only caller: once per field, at the boundary.
    """
    without_hidden = _COMMENT.sub(" ", _SCRIPT_OR_STYLE.sub(" ", raw))
    text = unescape(_TAG.sub(" ", without_hidden))
    return _WHITESPACE.sub(" ", text).strip()


def parse_feed(raw: str | bytes, *, ticker: str) -> tuple[Headline, ...]:
    """The headlines in an RSS document, newest first, stripped.

    Takes the document rather than a URL, which is what makes the whole parse path testable
    against a recorded fixture: `feedparser.parse` accepts bytes, so the suite runs the real
    parser over the real feed with no network (`tests/test_news_feed.py`).

    A malformed document is not an error. `feedparser` sets `bozo` and returns whatever entries
    it recovered — which for a truncated response is most of them — and salvaging four headlines
    beats reporting a parse failure for a card that is decoration on a grounded answer.
    """
    import feedparser

    parsed = feedparser.parse(raw)
    headlines = []
    for entry in parsed.entries:
        title = strip_html(str(entry.get("title") or "")).strip()
        if not title:
            continue  # an entry with no title is not a headline, whatever else it carries
        link = str(entry.get("link") or "")
        headlines.append(
            Headline(
                title=title,
                summary=strip_html(str(entry.get("summary") or "")),
                source=_domain(link) or ticker,
                link=link,
                published=_published(entry.get("published_parsed")),
            )
        )
    return tuple(headlines)


def within_days(
    headlines: tuple[Headline, ...], days: int, *, now: datetime | None = None
) -> tuple[Headline, ...]:
    """The headlines published within `days` of `now`, newest first, capped.

    **An undated headline is kept.** The alternative is dropping a story because its publisher's
    RSS omitted a timestamp, which loses information to enforce a window the analyst gave as a
    round number. It sorts last, so it can never displace a dated story from the cap.

    Filtering happens *here* rather than in the fetch, so the cache holds one list per ticker
    regardless of the window asked for: `days=3` and `days=30` are the same API call.
    """
    cutoff = (now or datetime.now(UTC)) - timedelta(days=days)
    kept = [
        headline
        for headline in headlines
        if headline.published is None or datetime.fromisoformat(headline.published) >= cutoff
    ]
    kept.sort(key=lambda headline: headline.published or "", reverse=True)
    return tuple(kept[:NEWS_MAX_HEADLINES])


def _domain(link: str) -> str:
    """`fool.com` from a story URL — the publisher, with any `www.` prefix dropped."""
    from urllib.parse import urlparse

    host = urlparse(link).netloc.lower()
    return host.removeprefix("www.")


def _published(parsed: object) -> str | None:
    """A feedparser `struct_time` as an ISO-8601 UTC string, or `None` if there wasn't one.

    Normalised to a string here rather than kept as a `datetime`, because a `Headline` crosses
    the agent's checkpoint as JSON (ADR-0008) and a shape that needs a custom encoder on the way
    out is a shape that comes back as something else on the way in.
    """
    if parsed is None:
        return None
    try:
        # feedparser normalises every entry date to UTC and reports it as a 9-tuple, so the
        # first six fields are the UTC calendar time.
        return datetime(*tuple(parsed)[:6], tzinfo=UTC).isoformat()  # type: ignore[misc]
    except (TypeError, ValueError):
        return None


def fetch_headlines(ticker: str) -> tuple[Headline, ...]:
    """Fetch and parse one ticker's feed. The one network call in this module.

    `urllib` rather than `requests`: this is a single keyless GET of a fixed template, and the
    project already depends on `urllib` through the standard library. The timeout is not
    optional — an unofficial free endpoint that hangs would otherwise hold the cache's lock for
    as long as the socket allowed.
    """
    import urllib.request

    request = urllib.request.Request(  # noqa: S310 — a fixed https template, validated ticker
        FEED_URL_TEMPLATE.format(ticker=ticker), headers={"User-Agent": NEWS_USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310
        raw = response.read()
    headlines = parse_feed(raw, ticker=ticker)
    if not headlines:
        # An empty feed is indistinguishable from a soft failure here — Yahoo serves an empty
        # `<channel>` to a client it is throttling — and caching "no news" for a TTL window is
        # the wrong answer to both. Raising hands it to the retry.
        raise LookupError(f"the headline feed returned no entries for {ticker}")
    return headlines


#: The process-level headline cache: one handle, one TTL window per ticker, shared by every
#: session — for the reason `quotes.QUOTE_CACHE` is a singleton.
NEWS_CACHE: TimedCache[tuple[Headline, ...]] = TimedCache(
    name="news",
    ttl_seconds=NEWS_TTL_SECONDS,
    attempts=FETCH_ATTEMPTS,
    backoff_seconds=FETCH_BACKOFF_SECONDS,
)


def headlines(
    ticker: str, *, cache: TimedCache[tuple[Headline, ...]] | None = None
) -> Fetched[tuple[Headline, ...]]:
    """One company's recent headlines, cached, retried, and stale-marked rather than failed.

    The **unwindowed** list: `within_days` narrows it per call, so two questions about the same
    company with different windows share one fetch.
    """
    return (cache or NEWS_CACHE).fetch(ticker, lambda: fetch_headlines(ticker))
