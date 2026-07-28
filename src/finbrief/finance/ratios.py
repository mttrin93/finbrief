"""Peer-relative ratios: the one tool that computes rather than reports (ADR-0009).

`get_stock_data` and `get_recent_news` hand on what an endpoint said. This module does
arithmetic, which means it is the one place where an answer's numbers can be quietly wrong
rather than merely stale — so every choice it makes is stated here.

**Peers come from `config.PEERS` and nowhere else** (ADR-0009). Not a GICS lookup, not the
model's suggestion, not an argument: the map is derived from the single `UNIVERSE` declaration,
`tests/test_golden_set.py` asserts the golden set's copy against it, and `compare` reads it. Any
of those three choosing its own peers would be a comparison basis nobody agreed to.

**An absence is excluded from a mean, never averaged in as zero.** This is the routine case, not
the edge: JPM and BAC report no debt-to-equity at all, so a `banks` leverage comparison has one
peer figure in it. Counting the two absences as nought would report a peer mean a third of the
truth, and it would look like a measurement. So a metric carries the peers that **reported** it
(`peers_compared`) alongside the cluster's full peer set — "mean of 1 of 2 peers" and "mean of
2 peers" are different claims and the tool says which one it is making.

**The spread goes out with the mean.** An arithmetic mean of multiples is dominated by its
outlier, and the Universe has a bad one: Ford's two `autos` peers are TSLA at 286× and GM at
37×, so the mean of 162× describes neither company. ADR-0009 specifies a mean and this reports a
mean — the fix is not to quietly substitute a median but to send the low and the high with it,
so the sentence a reader quotes carries its own caveat. (A median or a harmonic mean is a
legitimate Tier-2 experiment: a change to the ADR, made in the open, not one made here.)

**Nothing here says whether a number is good.** "Cheap relative to its cluster" is the analyst's
call and "should you buy it" is the one this project refuses (user story 15), so a `Metric` has
no `higher_is_better` and no verdict. It has a figure, a peer mean, and the peers behind it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from finbrief.config import COMPANIES, PEERS, PeerCluster
from finbrief.finance.quotes import Quote


class Unit(StrEnum):
    """How a figure is written down — the one renderer both surfaces use.

    The tool's text (what the model reads and quotes) and the UI card (what the analyst reads)
    format every figure through here, because a number formatted twice is a number the answer
    and the card can disagree about, and the disagreement is invisible until a reviewer checks
    one against the other.
    """

    #: A multiple of something: `31.6x`. An `x` rather than `×` because this string reaches a
    #: model's prompt as well as a Streamlit card, and an ASCII multiplier survives every
    #: tokenizer and every terminal a reviewer might read a log line in.
    MULTIPLE = "multiple"
    #: A fraction rendered as a percentage: `0.74144995` → `74.1%`.
    PERCENT = "percent"
    #: A share price: `196.51`. Two decimals, which is how a quote is quoted; the currency is
    #: the caller's to add, because it belongs to the company and not to the figure.
    PRICE = "price"
    #: A large money figure, abbreviated: `4.76T`, `640.91B`, `58.50B`. Market capitalisation is
    #: the only one, and `4759668391936` on a card is a number a reader has to count digits on.
    MONEY = "money"

    @property
    def chart_scale(self) -> float:
        """What to multiply a figure by to plot it in the units the axis is labelled in.

        On `Unit` rather than at the chart, because the surface was already switching on this
        enum three times — to format a figure, to scale it, and to caption the axis (issue #9
        review) — and a fourth reader is a fourth chance for one of them to disagree. A
        percentage is stored as a fraction and plotted as a percentage; everything else plots
        as stored.
        """
        return 100.0 if self is Unit.PERCENT else 1.0

    @property
    def axis_label(self) -> str:
        """What to call an axis of these figures: `Percentages`, `Multiples`."""
        return "Percentages" if self is Unit.PERCENT else "Multiples"

    def format(self, value: float | None) -> str:
        """`value` as a reader would write it, or `not reported` when there is nothing to write.

        **Words for an absence, never a number.** `0.0%` on a bank's gross-margin row is a claim
        nobody made; a bare `None` reads as a bug. One decimal place for a ratio, because a P/E
        quoted to four is a false precision about a figure scraped from a free endpoint.
        """
        if value is None:
            return "not reported"
        if self is Unit.PERCENT:
            return f"{value * 100:.1f}%"
        if self is Unit.PRICE:
            return f"{value:,.2f}"
        if self is Unit.MONEY:
            return _abbreviated(value)
        return f"{value:.1f}x"


def _abbreviated(value: float) -> str:
    """`4759668391936` → `4.76T`. Falls back to grouped digits below a billion.

    Stops at billions rather than descending to millions and thousands: every Universe market
    cap is ≥ $58bn, so a smaller scale would be untested code, and `140,605,292,544` spelled out
    is the failure this exists to avoid.
    """
    for suffix, scale in (("T", 1e12), ("B", 1e9)):
        if abs(value) >= scale:
            return f"{value / scale:,.2f}{suffix}"
    return f"{value:,.0f}"


@dataclass(frozen=True, slots=True)
class PeerValue:
    """One peer's figure for one metric. Kept named so a card can show who is where."""

    ticker: str
    value: float


@dataclass(frozen=True, slots=True)
class Metric:
    """One ratio: the company's figure, and the peers' — with no verdict on either.

    The peers here are the ones that **reported** this metric, which may be fewer than the
    cluster's peer set; `PeerComparison.peers` is the full set and `n` its size, so the two
    facts a reader needs ("compared against whom" and "how many actually had a number") are both
    available and neither stands in for the other.
    """

    key: str
    #: What to call it on a card and in the tool's text — `P/E (trailing)`. One label, shared,
    #: for the same reason `Unit.format` is shared.
    label: str
    unit: Unit
    #: The company's own figure, or `None` when the endpoint did not report it. Ford has no
    #: trailing P/E, and the row still appears: "not reported" answers the analyst's question,
    #: where a dropped row silently narrows it.
    value: float | None
    #: Only the peers that reported this metric, in the cluster's declaration order.
    peer_values: tuple[PeerValue, ...]

    @property
    def peers_compared(self) -> tuple[str, ...]:
        """Which peers are actually behind `peer_mean`."""
        return tuple(peer.ticker for peer in self.peer_values)

    @property
    def n_compared(self) -> int:
        """How many peers reported this metric — never more than `PeerComparison.n`."""
        return len(self.peer_values)

    @property
    def peer_mean(self) -> float | None:
        """The arithmetic mean of the peers that reported. `None` when none did."""
        if not self.peer_values:
            return None
        return sum(peer.value for peer in self.peer_values) / len(self.peer_values)

    @property
    def peer_low(self) -> float | None:
        """The lowest peer figure — half of the spread the mean is reported with."""
        return min((peer.value for peer in self.peer_values), default=None)

    @property
    def peer_high(self) -> float | None:
        return max((peer.value for peer in self.peer_values), default=None)

    def coverage_note(self, peers: int) -> str:
        """`3 of 5 peers reported this`, or `""` when every peer did.

        One wording, and it lives here for the reason `PeerComparison.basis` does: the tool's
        text and the UI card were each deriving it, in two phrasings, from the same two numbers
        (issue #9 review) — which is exactly the disagreement `basis` was extracted to prevent.
        `peers` is passed in because the cluster's size belongs to the comparison, not to one
        metric.
        """
        if not self.n_compared or self.n_compared >= peers:
            return ""
        return f"{self.n_compared} of {peers} peers reported this"

    @property
    def versus_peers(self) -> str:
        """`42.6x vs. peer mean 24.9x (range 18.8x–30.9x)` — the comparison as one phrase.

        Assembled here rather than at each surface because it is the sentence ADR-0009's "report
        the peer set and n inline" is about, and the mean must not be renderable without its
        spread beside it. The peer set and `n` come from `PeerComparison`, which owns them.
        """
        own = self.unit.format(self.value)
        if self.peer_mean is None:
            return f"{own} vs. peers: none reported this"
        spread = ""
        if self.peer_low is not None and self.peer_low != self.peer_high:
            spread = (
                f" (range {self.unit.format(self.peer_low)}–{self.unit.format(self.peer_high)})"
            )
        return f"{own} vs. peer mean {self.unit.format(self.peer_mean)}{spread}"


#: The metrics `calculate_ratios` reports, in the order it reports them: PLAN §2's "P/E, D/E,
#: margins", with the three margins named individually because a card that said "margins: 74.1%"
#: would leave a reader guessing which one. Keyed on `Quote`'s field names, so a metric cannot
#: name a figure the quote does not carry.
_METRICS: tuple[tuple[str, str, Unit], ...] = (
    ("trailing_pe", "P/E (trailing)", Unit.MULTIPLE),
    ("debt_to_equity", "Debt / equity", Unit.MULTIPLE),
    ("gross_margin", "Gross margin", Unit.PERCENT),
    ("operating_margin", "Operating margin", Unit.PERCENT),
    ("profit_margin", "Net margin", Unit.PERCENT),
)


@dataclass(frozen=True, slots=True)
class PeerComparison:
    """One company's ratios against its cluster — the whole of `calculate_ratios`' answer."""

    ticker: str
    name: str
    cluster: PeerCluster
    #: `config.PEERS[ticker]`, in full and unfiltered — the **basis of the comparison**, which
    #: ADR-0009 requires be reported inline whatever the data coverage turned out to be. The
    #: golden set carries a copy and `tests/test_golden_set.py` binds it to the same map.
    peers: tuple[str, ...]
    metrics: tuple[Metric, ...]
    #: The peers whose quote could not be fetched at all — not the ones missing a figure. Named
    #: apart because the causes differ: one is a company that does not report a ratio, the other
    #: is an API that did not answer, and only the second one might work on a retry.
    unavailable: tuple[str, ...] = ()

    @property
    def n(self) -> int:
        """The peer set's size — 2 for a three-member cluster, 5 for `big_tech` (ADR-0009)."""
        return len(self.peers)

    @property
    def basis(self) -> str:
        """`vs. mean of 2 autos peers: F, GM` — ADR-0009's inline basis sentence, verbatim.

        The one place it is written. The tool's text and the UI card both read it, so the
        model's prose and the card cannot describe two different comparisons.
        """
        return f"vs. mean of {self.n} `{self.cluster.value}` peers: {', '.join(self.peers)}"

    @property
    def every_peer_quote_failed(self) -> bool:
        """Whether *no* peer quote was fetched, so there is no mean of anything to report."""
        return bool(self.unavailable) and len(self.unavailable) == self.n

    @property
    def unavailable_note(self) -> str:
        """What to say about peers whose quote could not be fetched. `""` when they all were.

        **Two sentences, because "some failed" and "all failed" are different facts** and the
        first wording served both: "the means below rest on the remaining peers" is true of a
        partial outage and false when there are no remaining peers — at which point every metric
        row reads "vs. peers: none reported this" and the banner above it claimed a mean that
        does not exist (issue #9 review). A reader resolving that has to guess which to trust.

        Here rather than at each surface for the reason `basis` and `coverage_note` are: the
        tool text and the card were each phrasing it, and a banner that disagreed with itself
        across two surfaces is the same defect one layer out.
        """
        if not self.unavailable:
            return ""
        named = ", ".join(self.unavailable)
        if self.every_peer_quote_failed:
            return (
                f"No quote could be fetched for any peer ({named}), so no peer mean is "
                f"reported below."
            )
        return f"No quote for {named}, so the means below rest on the remaining peers."


def compare(quote: Quote, peer_quotes: Mapping[str, Quote]) -> PeerComparison:
    """Compare `quote` against its Universe cluster, using whichever peer quotes are in hand.

    `peer_quotes` is a mapping the caller has already fetched — `tools/finance.py` resolves the
    cluster through the shared `QUOTE_CACHE`, which is what makes ADR-0009's "zero new API
    surface" true. Extra entries are ignored, so the caller may pass everything it has; a peer
    that is *missing* lands in `unavailable` rather than being silently dropped from `n`.

    Raises `KeyError` for a ticker outside the Universe. The whitelist is enforced at the tool
    boundary, but an engine that invented an empty cluster would turn "mean of 0 peers" into a
    number, so this refuses instead.
    """
    peers = PEERS[quote.ticker]
    available = [ticker for ticker in peers if ticker in peer_quotes]
    return PeerComparison(
        ticker=quote.ticker,
        name=quote.name,
        cluster=COMPANIES[quote.ticker].cluster,
        peers=peers,
        metrics=tuple(
            Metric(
                key=key,
                label=label,
                unit=unit,
                value=getattr(quote, key),
                peer_values=tuple(
                    PeerValue(ticker, value)
                    for ticker in available
                    if (value := getattr(peer_quotes[ticker], key)) is not None
                ),
            )
            for key, label, unit in _METRICS
        ),
        unavailable=tuple(ticker for ticker in peers if ticker not in peer_quotes),
    )
