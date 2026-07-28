"""The three finance tools: the engine in `finance/` as something the agent can call (T5, #9).

The **wrappers**, never a second implementation — the same split `tools/search_filings.py` makes
for retrieval and for the same ADR-0003 reason: the fetch/cache policy and the ratio arithmetic
live under `finance/`, where they are measurable on their own (spec seam 5), and what is here is
the crossing into the agent. Anything in this module that started deciding a *number* would be a
difference between the measured engine and the shipped tool.

What it adds to the engine, and why each addition belongs at this boundary and not below it:

1. **Validation.** A model's argument is untrusted input. `resolve_ticker` bounds its length and
   resolves it against the Universe whitelist *before* anything is fetched or interpolated into
   a URL (user story 21). The engine functions may then assume a valid ticker, which is why the
   whitelist is not checked twice.
2. **Refusals as results, not exceptions.** An unknown ticker, an over-long argument and a dead
   endpoint all come back as a *sentence the model can act on* in the same turn. Raising would
   end the turn in a traceback the analyst sees, and the honest recovery — "I cover these
   fifteen companies" — is one only the model can write.
3. **Framing.** Headlines are quarantined in `prompts.news_block`, the same control
   `sources_block` applies to filing text, because a summary is written by a stranger (user
   story 17). Numbers need no quarantine; news does.
4. **Cards.** Each reply carries a JSON-safe artifact beside its text, so the UI renders the
   same figures the model was given rather than re-fetching them — a second fetch is a second
   chance to disagree with the answer, and with a TTL cache in the way it would also be a
   second *number*. Same reasoning as `search_filings`' artifact, and the same tolerance
   rules: a checkpoint outlives a deploy, so `from_payload` never requires a field (CLAUDE.md).

**Staleness travels; it is never dropped.** `finance/cache.py` returns `Fetched(value, age,
stale)` and both halves of that reach a surface: `prompts.stale_notice` tells the model, the
card's `Freshness` tells the reader. A tool that returned the value alone would be a tool whose
caller cannot honour user story 22.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from langchain_core.messages import AnyMessage, ToolMessage
from langchain_core.tools import BaseTool, tool

from finbrief.config import (
    NEWS_DEFAULT_DAYS,
    NEWS_MAX_DAYS,
    PEERS,
    TICKER_MAX_CHARS,
    TICKERS,
)
from finbrief.finance import news as news_engine
from finbrief.finance import quotes as quotes_engine
from finbrief.finance.cache import Fetched
from finbrief.finance.news import Headline, safe_link
from finbrief.finance.quotes import Close, Quote
from finbrief.finance.ratios import PeerComparison, Unit, compare
from finbrief.observability.logging_setup import log_event
from finbrief.prompts import (
    CALCULATE_RATIOS_DESCRIPTION,
    GET_RECENT_NEWS_DESCRIPTION,
    GET_STOCK_DATA_DESCRIPTION,
    news_block,
    over_long_ticker_message,
    stale_notice,
    unavailable_message,
    unknown_ticker_message,
)

logger = logging.getLogger(__name__)

#: The tool names, as the model sees them and as the UI looks for them in the message history.
#: One constant each because those two uses must agree — the same reason
#: `search_filings.TOOL_NAME` is a constant — and because **T10's tool-calling eval executes
#: against these exact strings**: the golden set's `tool_expectation.name` fields name them, and
#: `tests/test_golden_set.py` asserts the set below covers every one. A rename here is a change
#: to the evaluation artifact and cannot be made on its own.
STOCK_TOOL_NAME = "get_stock_data"
RATIOS_TOOL_NAME = "calculate_ratios"
NEWS_TOOL_NAME = "get_recent_news"

FINANCE_TOOL_NAMES = frozenset({STOCK_TOOL_NAME, RATIOS_TOOL_NAME, NEWS_TOOL_NAME})

type Artifact = dict[str, Any]


class TickerRefused(ValueError):
    """A tool argument that is not a Universe ticker. Carries the model-facing sentence."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def resolve_ticker(raw: object) -> str:
    """`" tsla "` → `"TSLA"`, or `TickerRefused` with the sentence to hand back.

    **Length first, whitelist second, and that order is the point.** The length cap is what
    makes it safe for `unknown_ticker_message` to echo the rejected argument: without it, a tool
    call carrying a page of prose would have that prose reflected straight back into the prompt
    as a tool result — an injection delivery path that costs the attacker one malformed argument
    (ADR-0006). Checking the whitelist first would reject the prose too, but on a code path that
    quotes it.

    Normalisation is deliberately minimal — strip, upper-case — and nothing clever: no suffix
    trimming (`NVDA.US` → `NVDA`), no fuzzy matching, no name lookup. A tool that guessed which
    company the model meant would be a second entity resolver with no evaluation of its own, and
    the description already lists the fifteen tickers verbatim so there is nothing to guess.
    """
    text = str(raw).strip()
    if len(text) > TICKER_MAX_CHARS:
        raise TickerRefused(over_long_ticker_message(len(text), TICKER_MAX_CHARS))
    ticker = text.upper()
    if ticker not in TICKERS:
        raise TickerRefused(unknown_ticker_message(text))
    return ticker


# --------------------------------------------------------------------------------------
# What a tool call renders as
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Freshness:
    """How old a card's figures are, and whether that is because a refresh failed.

    Both facts, because they are different: 12 minutes old inside a 15-minute TTL is a normal
    cached read, and 12 minutes old *after a failed refresh* is user story 22's banner. A card
    carrying only the age could not tell them apart, and one carrying only `stale` could not say
    how stale.
    """

    stale: bool
    age_seconds: float

    @property
    def age_minutes(self) -> int:
        """Rounded down, so a banner never claims a figure is fresher than it is."""
        return int(self.age_seconds // 60)

    @classmethod
    def of(cls, *fetched: Fetched[Any]) -> Freshness:
        """The freshness of a card built from several fetches — the **worst** of them.

        `calculate_ratios` reads the company and every peer through the cache, so a card can be
        part fresh and part stale. Reporting the best of them would put a banner-free card on
        screen whose peer mean is an hour old; the worst is the only summary that cannot
        overstate.
        """
        return cls(
            stale=any(one.stale for one in fetched),
            age_seconds=max((one.age_seconds for one in fetched), default=0.0),
        )


@dataclass(frozen=True, slots=True)
class QuoteCard:
    """`get_stock_data`'s result: one quote, with a price-history series for the chart."""

    KIND = "quote"

    quote: Quote
    freshness: Freshness

    @property
    def ticker(self) -> str:
        return self.quote.ticker

    def as_payload(self) -> Artifact:
        return _payload(self.KIND, self)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> QuoteCard:
        raw = payload.get("quote") or {}
        return cls(
            quote=Quote(
                ticker=str(raw.get("ticker", "")),
                name=str(raw.get("name", "")),
                currency=str(raw.get("currency", "USD")),
                closes=tuple(
                    Close(date=str(row.get("date", "")), close=float(row.get("close", 0.0)))
                    for row in raw.get("closes", ())
                ),
                # Every figure through `_optional_float`, which is what makes a payload written
                # before a field existed come back as **not reported** rather than as a
                # `KeyError` on the first rerun after a deploy (CLAUDE.md). `None` is the honest
                # absence here and it is also the value the whole engine already uses for one.
                **{field: _optional_float(raw.get(field)) for field in QUOTE_FIGURES},
            ),
            freshness=_freshness_from(payload),
        )


@dataclass(frozen=True, slots=True)
class RatiosCard:
    """`calculate_ratios`' result: one company's metrics against its cluster (ADR-0009)."""

    KIND = "ratios"

    comparison: PeerComparison
    freshness: Freshness

    @property
    def ticker(self) -> str:
        return self.comparison.ticker

    def as_payload(self) -> Artifact:
        return _payload(self.KIND, self)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> RatiosCard:
        # Rebuilt through `compare`'s own types rather than re-derived: a card read back from a
        # checkpoint must show the peer set the answer above it was written against, even if
        # `config.PEERS` has since changed — recomputing here would silently relabel a
        # historical comparison with today's cluster.
        from finbrief.config import PeerCluster
        from finbrief.finance.ratios import Metric, PeerValue

        raw = payload.get("comparison") or {}
        return cls(
            comparison=PeerComparison(
                ticker=str(raw.get("ticker", "")),
                name=str(raw.get("name", "")),
                cluster=PeerCluster(raw.get("cluster") or next(iter(PeerCluster)).value),
                peers=tuple(str(peer) for peer in raw.get("peers", ())),
                metrics=tuple(
                    Metric(
                        key=str(metric.get("key", "")),
                        label=str(metric.get("label", "")),
                        unit=Unit(metric.get("unit") or Unit.MULTIPLE.value),
                        value=metric.get("value"),
                        peer_values=tuple(
                            PeerValue(
                                str(peer.get("ticker", "")), float(peer.get("value", 0.0))
                            )
                            for peer in metric.get("peer_values", ())
                        ),
                    )
                    for metric in raw.get("metrics", ())
                ),
                unavailable=tuple(str(peer) for peer in raw.get("unavailable", ())),
            ),
            freshness=_freshness_from(payload),
        )


@dataclass(frozen=True, slots=True)
class NewsCard:
    """`get_recent_news`' result: the headlines inside the window that was asked for."""

    KIND = "news"

    ticker: str
    days: int
    headlines: tuple[Headline, ...]
    freshness: Freshness

    def as_payload(self) -> Artifact:
        return _payload(self.KIND, self)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> NewsCard:
        return cls(
            ticker=str(payload.get("ticker", "")),
            days=int(payload.get("days", NEWS_DEFAULT_DAYS)),
            headlines=tuple(
                Headline(
                    title=str(row.get("title", "")),
                    summary=str(row.get("summary", "")),
                    source=str(row.get("source", "")),
                    # Re-checked on the way back in, not trusted because it was checked on the
                    # way out: a checkpoint written before `safe_link` existed can hold a URL
                    # this app will not render, and a live conversation outlives a deploy.
                    link=safe_link(str(row.get("link", ""))),
                    published=row.get("published"),
                )
                for row in payload.get("headlines", ())
            ),
            freshness=_freshness_from(payload),
        )


@dataclass(frozen=True, slots=True)
class FailedCard:
    """A tool call that produced no data, and why — so the UI can say so.

    A distinct card rather than an empty one of the three above, because "no figures" and "the
    endpoint is down" are different things to put in front of a reader, and only the second one
    means waiting will help. The same distinction `render_sources` draws between "did not
    search" and "retrieved nothing".
    """

    KIND = "failed"

    kind: str
    ticker: str
    message: str

    def as_payload(self) -> Artifact:
        return {
            "kind": self.KIND,
            "failed_kind": self.kind,
            "ticker": self.ticker,
            "message": self.message,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> FailedCard:
        return cls(
            kind=str(payload.get("failed_kind", "")),
            ticker=str(payload.get("ticker", "")),
            message=str(payload.get("message", "")),
        )


type FinanceCard = QuoteCard | RatiosCard | NewsCard | FailedCard

_CARDS: Mapping[str, Any] = {
    QuoteCard.KIND: QuoteCard,
    RatiosCard.KIND: RatiosCard,
    NewsCard.KIND: NewsCard,
    FailedCard.KIND: FailedCard,
}

#: `Quote`'s optional numeric fields, in `Quote`'s own order — the ones `QuoteCard.from_payload`
#: reads back through `_optional_float`.
#:
#: Written out rather than reflected off the dataclass, because the list is a **decision**:
#: these are the fields whose honest absence value is `None`, and a `Quote` field added later
#: with a different absence value (a count, say, whose absence is not zero) must not be silently
#: swept in here. `tests/test_finance_tools.py` asserts this covers every `float | None` field,
#: so adding one is a test failure rather than a field that stops surviving a checkpoint.
QUOTE_FIGURES: tuple[str, ...] = (
    "price",
    "previous_close",
    "change_percent",
    "market_cap",
    "trailing_pe",
    "trailing_eps",
    "debt_to_equity",
    "gross_margin",
    "profit_margin",
    "operating_margin",
    "fifty_two_week_high",
    "fifty_two_week_low",
)


def _optional_float(value: object) -> float | None:
    """A payload figure as a float, or `None` — never a fabricated zero (CLAUDE.md)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _payload(kind: str, card: object) -> Artifact:
    """A card as JSON-safe primitives.

    `asdict` rather than a hand-written mapping per class, and the trade is stated: every field
    on every one of these dataclasses is a `str`, a number, `None`, or a dataclass of those, so
    the conversion is total and a field added to the engine appears in the payload without
    anyone remembering to add it. What it does *not* give is a guarantee that it stayed that way
    — so `tests/test_finance_tools.py` round-trips every card through `json.dumps`, which is the
    assertion `Context.as_payload`'s hand-written version gets for free.

    **Every sequence comes out as a `list`, and that is not cosmetic.** `asdict` preserves
    tuples; JSON has none, so a tuple here means the artifact a caller holds and the artifact it
    reads back out of the checkpoint compare unequal on every field that was a tuple.
    `Context.as_payload` hit exactly this and documents it: the citation register decides
    whether a rewrite is owed by comparing a rebuilt payload against the one it read back, and a
    shape that always looks changed makes that comparison useless. Nothing here compares
    payloads *yet* — `finance_cards` only reads — but a card that cannot round-trip is a
    landmine for whatever does, and the fix is one function.
    """
    body = _jsonable(asdict(card))  # type: ignore[call-overload]
    freshness = body.pop("freshness", {})
    return {"kind": kind, **body, "freshness": freshness}


def _jsonable(value: Any) -> Any:
    """`value` with every tuple turned into a list, recursively. See `_payload`."""
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _freshness_from(payload: Mapping[str, Any]) -> Freshness:
    """A card's freshness, defaulting to **fresh and ageless** when the payload has none.

    The honest absence for this field: a reply checkpointed before `Freshness` existed carried
    no staleness, so claiming it was stale would put a banner on a card nobody measured, and
    claiming an age would invent a number. "We recorded nothing about this" reads closest to
    `stale=False, age=0`, and the card's figures are what the answer above it was written from.
    """
    raw = payload.get("freshness") or {}
    return Freshness(
        stale=bool(raw.get("stale", False)), age_seconds=float(raw.get("age_seconds", 0.0))
    )


def finance_cards(messages: Iterable[AnyMessage]) -> tuple[FinanceCard, ...]:
    """Every finance-tool reply in `messages`, oldest first, as the card it renders as.

    The one reader of these tools' message protocol, for the same reason `search_results` is the
    one reader of `search_filings`': the agent's per-turn results and the UI both need it, and a
    protocol known independently in two files is one a change can half-update.

    A reply whose artifact did not survive — an older checkpoint, or a `None` — is **skipped**
    rather than rendered as an empty card. Unlike a search reply, whose mere existence proves
    the knowledge base was consulted, a finance reply with no artifact says nothing a reader can
    use: the figures were in the artifact.
    """
    cards: list[FinanceCard] = []
    for message in messages:
        if not isinstance(message, ToolMessage) or message.name not in FINANCE_TOOL_NAMES:
            continue
        artifact = message.artifact
        if not isinstance(artifact, Mapping):
            continue
        card_type = _CARDS.get(str(artifact.get("kind", "")))
        if card_type is None:
            continue
        cards.append(card_type.from_payload(artifact))
    return tuple(cards)


# --------------------------------------------------------------------------------------
# The tools
# --------------------------------------------------------------------------------------


def build_finance_tools(
    *,
    quote: Any = None,
    headlines: Any = None,
    now: datetime | None = None,
) -> list[BaseTool]:
    """The three finance tools, bound to their data sources.

    `quote` and `headlines` are the engine entry points, injectable for the same reason
    `retrieve()`'s `store` is: the suite drives the real tools over recorded fixtures with no
    network (spec seam 2 wants the agent real and the external data mocked), and a test can hand
    in a source that fails to exercise the refusal paths. Production passes neither and gets
    `finance.quotes.quote` / `finance.news.headlines`, which own the process-level caches.

    `now` is injectable because `get_recent_news` filters a window against it, and a test that
    passed a recorded feed through a real clock would go red the day after it was recorded.
    """
    fetch_quote = quote if quote is not None else quotes_engine.quote
    fetch_headlines = headlines if headlines is not None else news_engine.headlines

    @tool(
        STOCK_TOOL_NAME,
        description=GET_STOCK_DATA_DESCRIPTION,
        response_format="content_and_artifact",
    )
    def get_stock_data(ticker: str) -> tuple[str, Artifact]:
        try:
            resolved = resolve_ticker(ticker)
        except TickerRefused as refused:
            return _refused(STOCK_TOOL_NAME, QuoteCard.KIND, str(ticker), refused)
        try:
            fetched = fetch_quote(resolved)
        except Exception as exc:
            return _unavailable(
                STOCK_TOOL_NAME, QuoteCard.KIND, resolved, "Live market data", exc
            )
        card = QuoteCard(quote=fetched.value, freshness=Freshness.of(fetched))
        _log_call(STOCK_TOOL_NAME, resolved, card.freshness)
        return _with_staleness(_quote_text(card), card.freshness), card.as_payload()

    @tool(
        RATIOS_TOOL_NAME,
        description=CALCULATE_RATIOS_DESCRIPTION,
        response_format="content_and_artifact",
    )
    def calculate_ratios(ticker: str) -> tuple[str, Artifact]:
        try:
            resolved = resolve_ticker(ticker)
        except TickerRefused as refused:
            return _refused(RATIOS_TOOL_NAME, RatiosCard.KIND, str(ticker), refused)
        try:
            own = fetch_quote(resolved)
        except Exception as exc:
            return _unavailable(RATIOS_TOOL_NAME, RatiosCard.KIND, resolved, "Ratios", exc)
        # Peers resolve through the same cached path (ADR-0009: zero new API surface), and a
        # peer whose fetch fails is *skipped* rather than fatal: a comparison against one of two
        # peers is worth reporting as long as it says so, which `unavailable` does.
        peers: dict[str, Quote] = {}
        peer_fetches: list[Fetched[Quote]] = []
        for peer in PEERS[resolved]:
            try:
                peer_fetched = fetch_quote(peer)
            except Exception:  # noqa: BLE001 — reported through `unavailable`, not raised
                continue
            peers[peer] = peer_fetched.value
            peer_fetches.append(peer_fetched)
        card = RatiosCard(
            comparison=compare(own.value, peers),
            freshness=Freshness.of(own, *peer_fetches),
        )
        _log_call(RATIOS_TOOL_NAME, resolved, card.freshness, peers=len(peers))
        return _with_staleness(_ratios_text(card), card.freshness), card.as_payload()

    @tool(
        NEWS_TOOL_NAME,
        description=GET_RECENT_NEWS_DESCRIPTION,
        response_format="content_and_artifact",
    )
    def get_recent_news(ticker: str, days: int = NEWS_DEFAULT_DAYS) -> tuple[str, Artifact]:
        try:
            resolved = resolve_ticker(ticker)
        except TickerRefused as refused:
            return _refused(NEWS_TOOL_NAME, NewsCard.KIND, str(ticker), refused)
        # Clamped rather than refused, as the description says: a model asking for 90 days wants
        # "everything recent", and a refusal spends a round trip to teach it a number it will
        # then guess again. `int()` because a model may send "7".
        window = max(1, min(int(days), NEWS_MAX_DAYS))
        try:
            fetched = fetch_headlines(resolved)
        except Exception as exc:
            return _unavailable(NEWS_TOOL_NAME, NewsCard.KIND, resolved, "Recent news", exc)
        card = NewsCard(
            ticker=resolved,
            days=window,
            headlines=news_engine.within_days(fetched.value, window, now=now),
            freshness=Freshness.of(fetched),
        )
        _log_call(
            NEWS_TOOL_NAME, resolved, card.freshness, days=window, headlines=len(card.headlines)
        )
        return _with_staleness(_news_text(card), card.freshness), card.as_payload()

    return [get_stock_data, calculate_ratios, get_recent_news]


def _refused(
    tool_name: str, kind: str, raw: str, refused: TickerRefused
) -> tuple[str, Artifact]:
    """A validation refusal, as a result the model can act on. Nothing was fetched."""
    log_event(
        logger,
        "tool_refused",
        tool=tool_name,
        # A length, not the argument: this is the one field on this path that could carry
        # arbitrary text, and these lines are kept (`logging_setup`).
        argument_chars=len(raw),
        reason="not_in_universe" if len(raw) <= TICKER_MAX_CHARS else "too_long",
    )
    return refused.message, FailedCard(
        kind, raw[:TICKER_MAX_CHARS], refused.message
    ).as_payload()


def _unavailable(
    tool_name: str, kind: str, ticker: str, what: str, exc: Exception
) -> tuple[str, Artifact]:
    """A source that failed with nothing cached — the API tier of the tiered handling (PLAN §2).

    The retry and the stale fallback already happened inside `finance/cache.py`; reaching here
    means both were exhausted, so this is the tier where a *message* is the only thing left. The
    exception type is logged and the message never is: a client's error string can contain a
    request URL, and a request URL can contain a key.
    """
    message = unavailable_message(what, ticker)
    log_event(
        logger,
        "tool_unavailable",
        level=logging.WARNING,
        tool=tool_name,
        ticker=ticker,
        error=type(exc).__name__,
    )
    return message, FailedCard(kind, ticker, message).as_payload()


def _log_call(tool_name: str, ticker: str, freshness: Freshness, **fields: Any) -> None:
    """One line per successful call, for T10's tool-calling eval and the Phase-6 analysis."""
    log_event(
        logger,
        "tool_call",
        tool=tool_name,
        ticker=ticker,
        stale=freshness.stale,
        age_seconds=round(freshness.age_seconds),
        **fields,
    )


def _with_staleness(text: str, freshness: Freshness) -> str:
    """The tool's text, with `prompts.stale_notice` in front when it is serving stale data.

    In **front**, not appended: the model has to know how much to trust the figures before it
    reads them, and a caveat after a table of numbers is a caveat competing with the numbers.
    """
    if not freshness.stale:
        return text
    return f"{stale_notice(freshness.age_minutes)}\n\n{text}"


# --------------------------------------------------------------------------------------
# What the model reads
# --------------------------------------------------------------------------------------


def _quote_text(card: QuoteCard) -> str:
    """One quote as lines a model can quote from — every figure through `Unit.format`.

    Shared formatting with the UI card, so the prose and the card cannot disagree about a number
    (`finance/ratios.py`'s `Unit`). Absences read "not reported", which is what the description
    promises the model it will see.
    """
    quote = card.quote
    change = (
        "not reported"
        if quote.change_percent is None
        else f"{quote.change_percent:+.2f}% on the previous close"
    )
    lines = [
        f"{quote.ticker} — {quote.name} (prices in {quote.currency})",
        f"Last price: {Unit.PRICE.format(quote.price)} ({change})",
        f"Previous close: {Unit.PRICE.format(quote.previous_close)}",
        f"Market capitalisation: {Unit.MONEY.format(quote.market_cap)} {quote.currency}",
        f"P/E (trailing): {Unit.MULTIPLE.format(quote.trailing_pe)}",
        f"EPS (trailing): {Unit.PRICE.format(quote.trailing_eps)}",
        f"52-week range: {Unit.PRICE.format(quote.fifty_two_week_low)}"
        f"–{Unit.PRICE.format(quote.fifty_two_week_high)}",
    ]
    if card.freshness.age_minutes and not card.freshness.stale:
        lines.append(f"Quoted from cache, {card.freshness.age_minutes} minute(s) old.")
    return "\n".join(lines)


def _ratios_text(card: RatiosCard) -> str:
    """One peer comparison as lines a model can quote from, with ADR-0009's basis on top.

    The basis line comes **first** because it is the sentence the ADR requires be reported
    inline, and a model that read the metrics first would sometimes report them without it.
    """
    comparison = card.comparison
    lines = [
        f"{comparison.ticker} — {comparison.name}, {comparison.basis}",
    ]
    if comparison.unavailable:
        lines.append(
            f"Could not fetch: {', '.join(comparison.unavailable)} — the means below rest on "
            f"the remaining peers, so say how many."
        )
    for metric in comparison.metrics:
        detail = ""
        if metric.n_compared and metric.n_compared < comparison.n:
            detail = f" [only {metric.n_compared} of {comparison.n} peers reported this]"
        lines.append(f"{metric.label}: {metric.versus_peers}{detail}")
    return "\n".join(lines)


def _news_text(card: NewsCard) -> str:
    """The headlines, quarantined by `prompts.news_block` (ADR-0006).

    An empty window is a **fact**, not a failure, and says so in those terms: the feed answered
    and had nothing inside the days asked for, which is different from the feed not answering
    (`unavailable_message`). Conflating them has the model report an outage as quiet news.
    """
    if not card.headlines:
        return (
            f"No headlines for {card.ticker} in the last {card.days} day(s). The feed answered "
            f"and had nothing in that window — this is not a fetch failure. Say there is no "
            f"recent news rather than reaching for older news you remember."
        )
    heading = (
        f"{len(card.headlines)} headline(s) for {card.ticker} in the last {card.days} day(s)."
    )
    lines = [_headline_line(headline) for headline in card.headlines]
    return f"{heading}\n\n{news_block(lines)}"


def _headline_line(headline: Headline) -> str:
    """`- Title — publisher, 2026-07-28. Summary` — one headline, deliberately unnumbered.

    A `-` and not `1.`: the answer's `[n]` markers name filing chunks a reader can check against
    EDGAR, and a numbered headline list invites the model to cite a press release as `[3]`.
    """
    when = "" if headline.published is None else f", {headline.published[:10]}"
    summary = f" {headline.summary}" if headline.summary else ""
    return f"- {headline.title} — {headline.source}{when}.{summary}"
