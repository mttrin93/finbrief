"""Central configuration: the Universe, Peers, models, strategy switches, feature flags.

Every knob FinBrief has lives here, so the app and the evaluation harness read the same
switches (ADR-0003: one code path, config-selected strategy). Secrets are read from the
environment — `.env` locally, platform env vars / `st.secrets` when deployed — and are
never logged or reprinted.

`Settings.from_env()` takes an explicit mapping so configuration is a pure function of
its environment; `get_settings()` is the cached application entrypoint on top of it.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

from dotenv import load_dotenv


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or malformed."""


# --------------------------------------------------------------------------------------
# Universe and Peers (ADR-0009)
# --------------------------------------------------------------------------------------


class PeerCluster(StrEnum):
    """A curated same-sector peer cluster within the Universe.

    Doubles as the peer pool: `calculate_ratios` compares a company against the other
    Universe members of its cluster and never against an out-of-Universe ticker.

    Deliberately not named `Sector`: these are curated for ratio comparability, not drawn
    from a sector taxonomy. `big_tech` spans three GICS sectors, and AMZN sits apart from
    the autos despite sharing one with them — the cluster is the unit, GICS an input to
    curating it (ADR-0009).
    """

    BIG_TECH = "big_tech"
    AUTOS = "autos"
    BANKS = "banks"
    HEALTHCARE = "healthcare"

    @property
    def label(self) -> str:
        """Display name for the UI: `big_tech` -> "Big Tech"."""
        return self.value.replace("_", " ").title()


@dataclass(frozen=True, slots=True)
class Company:
    """One member of the Universe."""

    ticker: str
    name: str
    cluster: PeerCluster
    #: The shortened forms of `name` an analyst actually types, for the query-side entity
    #: normalisation in `retrieval/query_translation.py` (ADR-0004 amendment, T6). `name`
    #: itself is always matched, so these are only what it is *abbreviated* to.
    #:
    #: **Explicit data, not derived**, and the derivation was tried first: taking the leading
    #: word of each legal name produces `The` for Goldman Sachs (a token in 89.5% of the
    #: collection's chunks), `General` for GM (15 filers), `Bank` for BAC (10 filers), and
    #: `Amazon.com` for AMZN (which matches nothing, since the corpus tokenizes it as two
    #: terms). A heuristic that wrong on 4 of 15 members is not a heuristic.
    #:
    #: **Legal-name shortenings only** — deliberately not brands or subsidiaries (`Google`,
    #: `Facebook`). Resolving a brand to its filer is entity resolution, which the agent
    #: already does against the conversation (ADR-0003 amendment §1), and doing it here too
    #: would be a second resolver with no evaluation of its own.
    aliases: tuple[str, ...] = ()


#: The Universe: fixed at ingest time, curated as same-sector peer clusters so it can
#: serve triple duty — KB scope, demo cast, and peer pool (CONTEXT.md, ADR-0009).
#:
#: **Every member must file a 10-K.** ADR-0007 scopes the KB to Items 1/1A/7/7A of the
#: annual 10-K, and foreign private issuers file a 20-F, which has no such items — so a
#: 20-F filer here would silently produce zero Sections at ingest. `test_config.py` guards
#: this; ADR-0007 records it as a stated limitation.
UNIVERSE: tuple[Company, ...] = (
    Company("AAPL", "Apple Inc.", PeerCluster.BIG_TECH, ("Apple",)),
    Company("MSFT", "Microsoft Corporation", PeerCluster.BIG_TECH, ("Microsoft",)),
    Company("NVDA", "NVIDIA Corporation", PeerCluster.BIG_TECH, ("NVIDIA",)),
    Company("AMZN", "Amazon.com, Inc.", PeerCluster.BIG_TECH, ("Amazon.com", "Amazon")),
    Company("GOOGL", "Alphabet Inc.", PeerCluster.BIG_TECH, ("Alphabet",)),
    Company("META", "Meta Platforms, Inc.", PeerCluster.BIG_TECH, ("Meta Platforms", "Meta")),
    Company("TSLA", "Tesla, Inc.", PeerCluster.AUTOS, ("Tesla",)),
    Company("F", "Ford Motor Company", PeerCluster.AUTOS, ("Ford Motor", "Ford")),
    Company("GM", "General Motors Company", PeerCluster.AUTOS, ("General Motors",)),
    Company("JPM", "JPMorgan Chase & Co.", PeerCluster.BANKS, ("JPMorgan Chase", "JPMorgan")),
    Company("BAC", "Bank of America Corporation", PeerCluster.BANKS, ("Bank of America",)),
    Company(
        "GS",
        "The Goldman Sachs Group, Inc.",
        PeerCluster.BANKS,
        ("Goldman Sachs Group", "Goldman Sachs", "Goldman"),
    ),
    Company("JNJ", "Johnson & Johnson", PeerCluster.HEALTHCARE, ("J&J",)),
    Company("LLY", "Eli Lilly and Company", PeerCluster.HEALTHCARE, ("Eli Lilly", "Lilly")),
    Company("PFE", "Pfizer Inc.", PeerCluster.HEALTHCARE, ("Pfizer",)),
)

#: Foreign private issuers file a 20-F, not a 10-K, so they cannot supply the Sections
#: ADR-0007 scopes the KB to.
#:
#: **No longer the source of truth.** Ticket T2 (#3) put the real check in
#: `ingestion/gate.py`, which asks EDGAR for each company's most recent annual filing and
#: fails loudly unless it is in the 10-K family — that catches *any* foreign private
#: issuer, not merely one someone thought to write down.
#:
#: Kept anyway, in its demoted role, because it is free and offline: it fails in
#: `test_config.py` in milliseconds with no network, where the gate needs a live EDGAR
#: round trip and a full ingest run to say the same thing. A tripwire that catches the
#: plausible re-addition (the `eu_tech` cluster this replaced — SAP/ASML/STM — was all
#: three) before anyone waits for the gate. The invariant itself remains the `UNIVERSE`
#: docstring's "every member files a 10-K"; neither list nor test proves it, and the gate
#: is what enforces it.
_TWENTY_F_FILERS: frozenset[str] = frozenset(
    {
        # Tech / big_tech candidates
        "SAP", "ASML", "STM", "TSM", "BABA", "INFY", "SONY", "SHOP",
        # Autos candidates
        "TM", "HMC", "RACE", "STLA",
        # Banks candidates
        "HSBC", "UBS", "DB", "BCS", "MUFG", "RY", "TD",
        # Healthcare candidates
        "NVO", "AZN", "SNY", "GSK", "NVS", "TAK",
        # Energy / other frequently-suggested large caps
        "SHEL", "BP", "TTE", "PBR", "BHP", "RIO",
    }
)  # fmt: skip


#: The Universe filers hand-verified to answer Item 7A with a pointer into Item 7 — every
#: bank and every healthcare name, 6 of 15, and nobody else (ADR-0007 amendment; the rows
#: marked "incorporated by reference" in docs/verification/section-starts.md).
#:
#: A tripwire, not documentation: `ingestion/gate.py` fails any filing where the
#: incorporation-by-reference excusal fires for a ticker outside this set
#: (`pointer_filer_is_recorded`). Either the filer newly hands the Item off — verify
#: against the filing on EDGAR, then record it here — or the pointer detection misfired
#: on a broken parse. Both deserve a human; neither may be a silent drop.
ITEM_7A_POINTER_FILERS: frozenset[str] = frozenset({"BAC", "GS", "JNJ", "JPM", "LLY", "PFE"})


def _build_clusters(
    universe: tuple[Company, ...],
) -> Mapping[PeerCluster, tuple[str, ...]]:
    """Group the Universe by peer cluster, preserving declaration order."""
    by_cluster: dict[PeerCluster, list[str]] = {}
    for company in universe:
        by_cluster.setdefault(company.cluster, []).append(company.ticker)
    return MappingProxyType({cluster: tuple(t) for cluster, t in by_cluster.items()})


def _build_peers(
    clusters: Mapping[PeerCluster, tuple[str, ...]], universe: tuple[Company, ...]
) -> Mapping[str, tuple[str, ...]]:
    """Derive the static PEERS map from the Universe's peer clusters.

    ADR-0009 calls for a static map in `config.py`. Deriving it from the single Universe
    declaration keeps it static (computed once at import, no I/O) while making it
    impossible for the map to drift from the Universe it describes.
    """
    return MappingProxyType(
        {
            company.ticker: tuple(t for t in clusters[company.cluster] if t != company.ticker)
            for company in universe
        }
    )


COMPANIES: Mapping[str, Company] = MappingProxyType({c.ticker: c for c in UNIVERSE})
TICKERS: frozenset[str] = frozenset(COMPANIES)

#: The Universe filers that have an `Item 7A` Section of their own — the complement of
#: `ITEM_7A_POINTER_FILERS`. Derived rather than typed: "9 of 15" appears in the app's
#: grounding-scope panel, in the retrieval smoke check's rationale and in CONTEXT.md, and a
#: hand-typed copy of it is wrong the moment a filer starts or stops handing the Item off.
ITEM_7A_SECTION_FILERS: frozenset[str] = TICKERS - ITEM_7A_POINTER_FILERS

#: The fewest characters a ticker must have to be worth adding to a lexical query
#: (`retrieval/query_translation.py`'s entity normalisation, ADR-0004 amendment).
#:
#: **Two, and `F` is why.** Measured against the ingested collection: the token `f` appears in
#: 534 of 5,842 chunks (9.1%) and in chunks belonging to **five** filers — F, GM, GS, JPM, PFE —
#: because a single letter is also a footnote marker, a table label and a unit. `ford` appears
#: in 232 chunks (4.0%) belonging to **one**. So for Ford the company *name* is already the
#: sharper lexical identifier and the ticker would dilute it, which is the opposite of what
#: normalisation is for. Excluding it costs nothing precisely because the original is retained.
#:
#: Stated from the principle rather than tuned to the number: a one-character term is not an
#: identifier in any lexical index. The measurement corroborates it; a second single-letter
#: ticker joining the Universe would be excluded for the same reason without re-measuring.
#:
#: **What it suppresses, and what that costs — measured** (ADR-0004 amendment §8).
#: `normalised()` is the only reader of the map below and returns either the whole rewritten
#: query or `None`, so a filtered-out ticker means **no variant at all** — both retrievers lose
#: it, not just BM25. For Ford the cost is zero to negative: `Ford debt` retrieves 5/5 Ford
#: under vector search while the suppressed `F debt` retrieves 3/5 (two BAC chunks intrude, at
#: worse distances), and BM25's top-5 is identical either way.
#:
#: The real variable is how well a filer's **name** covers its own chunks: `ford` is a token in
#: 232 of Ford's 499 chunks and 0 elsewhere, against `tesla`'s 33 of 280. Ford has the
#: best-covered name in the Universe bar META, so it does not need normalisation and its
#: ticker is a poor embedding token — both facts point the same way. This constant is a
#: first-principles proxy for that, and it would be the *wrong* proxy for a future member with
#: a short ticker and a poorly-covered name.
MIN_LEXICAL_TICKER_CHARS = 2

#: Lower-cased company name form -> ticker, for query-side entity normalisation. Derived from
#: `UNIVERSE` so a 16th company is normalisable the moment it is declared, and *filtered* by
#: `MIN_LEXICAL_TICKER_CHARS` so a ticker that would be lexical noise is never substituted in.
#:
#: A company excluded here is not un-searchable: the retained original still carries whatever
#: the analyst typed (ADR-0004 — translation only ever adds).
TICKER_BY_COMPANY_NAME: Mapping[str, str] = MappingProxyType(
    {
        form.lower(): company.ticker
        for company in UNIVERSE
        if len(company.ticker) >= MIN_LEXICAL_TICKER_CHARS
        for form in (company.name, *company.aliases)
    }
)

#: peer cluster -> its tickers. The one place the grouping is computed: `_build_peers` and
#: the UI's Universe panel both read it rather than re-deriving it from `UNIVERSE`.
CLUSTERS: Mapping[PeerCluster, tuple[str, ...]] = _build_clusters(UNIVERSE)

#: ticker -> same-cluster Universe peers, excluding the ticker itself.
#: Every cluster holds at least three members, so no company is ever compared against a
#: single peer (a "peer mean" of one). Phase 3 (`calculate_ratios`) still reports the peer
#: set and n inline, per ADR-0009.
PEERS: Mapping[str, tuple[str, ...]] = _build_peers(CLUSTERS, UNIVERSE)


def thinnest_cluster_filer() -> Company:
    """The Universe member with the fewest peers — the crispest worked example, from the data.

    **One derivation, because two of them disagreed on screen.** The sidebar's Universe
    panel picks a company to illustrate "peers come only from this set", and
    `prompts.EXAMPLE_QUESTIONS` picks one for its peer-comparison button. Both were written
    as `min(UNIVERSE, key=...)` over the same idea with different tie-breaks — and five
    clusters tie at two members, so the panel said `TSLA` while the button asked about Bank
    of America, under a comment claiming the button was derived the way the panel is (code
    review of #13).

    `ticker` breaks the tie so the answer is deterministic across runs: on size alone it fell
    out of dictionary order, stable per build and not across curation changes.

    Here rather than in `prompts.py` because it is a fact about the Universe and not a
    prompt, and `prompts.py` already reads this module — the other direction is the import
    loop `GROUNDING_SCOPE` is careful about.
    """
    return min(UNIVERSE, key=lambda company: (len(PEERS[company.ticker]), company.ticker))


# --------------------------------------------------------------------------------------
# Finance tools: the free-tier budget and the input caps (T5, ADR-0009)
# --------------------------------------------------------------------------------------

#: How long a fetched quote is served without re-asking, in seconds.
#:
#: **Fifteen minutes, because the feed itself is fifteen minutes delayed.** Yahoo's free
#: quote data is delayed for most venues, so a TTL shorter than that delay spends a call to
#: re-fetch a number that cannot have changed — the cache would cost quota and buy nothing.
#: PLAN §4 names the same figure from the other direction (yfinance is unofficial: cache
#: aggressively), and ADR-0009 leans on this path for peers, so a `big_tech` ratio comparison
#: costs six quotes per window rather than six per question.
QUOTE_TTL_SECONDS = 900

#: How long a fetched headline list is served without re-asking, in seconds.
#:
#: The same figure for a different reason: the RSS feeds are free and uncapped, so this is
#: politeness to a public endpoint rather than budget, and a brief is not a ticker tape — a
#: headline that broke four minutes ago changes no answer this assistant is qualified to give.
NEWS_TTL_SECONDS = 900

#: Attempts per fetch, and the first backoff. `0.5 · 2ⁿ` between attempts, so three attempts
#: wait 0.5s then 1.0s and add **at most 1.5s** to a failing call.
#:
#: Calibrated against the characteristic yfinance failure, which is transient: a scrape of a
#: private endpoint returns an empty body or a 429 and then works. Three attempts is what turns
#: that into an invisible recovery instead of a stale banner; a fourth would spend 3.5s to
#: convert a persistent outage into a slower stale banner, which is the wrong trade against
#: ADR-0005's latency thinking.
FETCH_ATTEMPTS = 3
FETCH_BACKOFF_SECONDS = 0.5

#: The smallest response `finance/news.py` will accept as "a feed that answered and had nothing"
#: rather than as a soft failure to retry.
#:
#: **Because an empty feed and a throttled one arrive identically at the parser**, and reporting
#: one as the other is the absence-vs-measurement mistake in both directions: "no news this
#: week" is a claim about the company; "the feed did not answer" is a claim about the feed.
#: Until #9's review, *every* zero-entry response raised — so a genuinely quiet week reported as
#: an outage, which is the inverse of the rule the same module draws elsewhere.
#:
#: 400 bytes, calibrated against the recorded fixtures: Yahoo's channel preamble — copyright,
#: description, the `<image>` block — measures 541, 547 and 547 bytes in the three real feeds
#: under `tests/fixtures/market/`, so a zero-item feed from this publisher still weighs ~565
#: bytes. Anything materially under that is not a Yahoo channel.
#:
#: **What is measured and what is not**, because this is a threshold and thresholds invite
#: over-trust: the *populated* side is measured (three real feeds, three consistent preambles).
#: A real throttle response is **not** — nothing here has ever recorded one, so its size is an
#: assumption. That is why size is only the second test: `finance/news.py` also asks whether the
#: document parsed as a feed at all, which catches the HTML error page a blocked client gets
#: however large it is.
NEWS_MIN_FEED_BYTES = 400

#: Alpha Vantage's free-tier daily call budget, as *documented by them* — not a knob, and not
#: read by any fetch, because nothing calls Alpha Vantage (see `Settings.alphavantage_enabled`).
#:
#: It is here because it is a **number that appears in prose in four places** — this file,
#: `.env.example`, PLAN §4 and the README's "What the live figures are, and are not" — and it is
#: the whole argument for why the fallback is deferred. A budget typed four times is one that
#: will disagree with itself, and `tests/test_grounding_scope.py` binds the README's copy to
#: this one (issue #9 review).
ALPHAVANTAGE_FREE_TIER_CALLS_PER_DAY = 25

#: How much price history one quote fetch carries, at what interval, and how to say so in prose.
#:
#: `HISTORY_PERIOD` is yfinance's own period token, not a day count, and deliberately: `"1mo"`
#: means "the last calendar month" and yields the ~21 *trading* sessions in it, where `"30d"`
#: yields 30 rows over a span that shifts with the weekends and holidays inside it. The
#: sparkline `get_stock_data` renders (user story 13) wants sessions.
#:
#: `HISTORY_PERIOD_LABEL` is here rather than in the prompt because
#: `GET_STOCK_DATA_DESCRIPTION` states the window to the model, and `prompts.py`'s rule is that
#: a prompt's every count is derived and never typed — the first draft had "a month of daily
#: closes" as prose in the description and `"1mo"` in `finance/quotes.py` — two copies of one
#: fact nothing would have caught disagreeing (issue #9 review). A token unfit for prose and
#: prose unfit for an API are two names, but they are two names in **one place**.
#: `HISTORY_AUTO_ADJUST` is here for the same one-place reason, and it is not cosmetic: adjusted
#: closes are split- and dividend-adjusted, so the *same* company plotted both ways diverges
#: across any corporate action in the window. `scripts/record_market_fixtures.py` records the
#: closes every chart test asserts against, so a recorder that adjusted differently from the
#: fetcher would bake a silent mismatch into the fixtures — on the one path no test covers
#: (issue #9 review). It was typed twice before this.
HISTORY_PERIOD = "1mo"
HISTORY_PERIOD_LABEL = "one month"
HISTORY_INTERVAL = "1d"
HISTORY_AUTO_ADJUST = True

#: How long a **single HTTP request** on a finance fetch may hang before it is abandoned, in
#: seconds.
#:
#: An unofficial free endpoint that accepts a connection and then stalls is the failure a retry
#: cannot help with, and `TimedCache` holds its lock across the refresh — so without a ceiling
#: one stalled socket blocks every session's quotes for as long as the OS allows. Fifteen
#: seconds is generous for a JSON response.
#:
#: **Per request, not per fetch, and the difference is worth stating** because the first version
#: of this comment implied the latter. One `fetch_quote` makes two requests (`.info` and the
#: history), `FETCH_ATTEMPTS` is 3, and `TimedCache` holds its lock across all of it, so the
#: worst case a stalled endpoint can hold the quote lock for is `3 × 2 × 15s` plus backoff —
#: about 91 seconds, not 15. That is the honest ceiling; it is bounded and survivable, where
#: yfinance's own default of 30s per request made it ~181s, and *un*bounded before the clamp
#: reached the quote path at all (issue #9 review).
#:
#: `finance/news.py` passes this to `urlopen` directly. `finance/quotes.py` cannot: yfinance
#: passes `timeout=30` explicitly at every call site, so a session default is overridden and the
#: clamp has to sit on the session's `request` — see `_bounded_session` there.
FETCH_TIMEOUT_SECONDS = 15

#: The worst case `FETCH_TIMEOUT_SECONDS` actually permits on the quote path, derived rather
#: than typed so the docstring above cannot drift from it. Two requests per attempt.
QUOTE_FETCH_WORST_CASE_SECONDS = FETCH_ATTEMPTS * 2 * FETCH_TIMEOUT_SECONDS + sum(
    FETCH_BACKOFF_SECONDS * 2**attempt for attempt in range(FETCH_ATTEMPTS - 1)
)

#: The `days` window `get_recent_news` uses when the model names none, and the ceiling it
#: clamps to. A month is where "recent news" stops being recent; the default is a week because
#: that is the window a pre-earnings brief is about.
NEWS_DEFAULT_DAYS = 7
NEWS_MAX_DAYS = 30

#: The most headlines one call reports. A cap on the *prompt*, not on the feed: every headline
#: is text the model pays to read, and a brief that lists twenty is not a brief.
NEWS_MAX_HEADLINES = 8

#: How long a raw ticker argument may be before it is rejected unread (user story 21).
#:
#: The whitelist in `TICKERS` is the real validation — this is the length cap in front of it,
#: so a model (or an injection routed through a tool argument) cannot hand a lookup a kilobyte
#: of prose and have it echoed back inside an error message. Five is the longest Universe
#: ticker; the slack is for a suffix a model might append (`NVDA.US`) that is still worth a
#: clear "not in the Universe" rather than a length complaint.
TICKER_MAX_CHARS = 12

# --------------------------------------------------------------------------------------
# The security gate (T7, ADR-0006)
# --------------------------------------------------------------------------------------

#: What ADR-0006 **pre-registered** for the input gate's p50, in milliseconds, before anything
#: had measured what a classifier round trip costs.
#:
#: **Kept after being revised, which is the point of it.** The measured escalated p50 on
#: `Settings.classifier_model` is 738–1041 ms across eight passes — over this figure in six of
#: them — so the prediction *straddles* rather than holds, and a single run's verdict against it
#: is close to a coin toss. ADR-0006's T7 amendment §2 records the measurements and revises the
#: budget to `GATE_LATENCY_BUDGET_MS` below. Deleting this constant would turn a revised
#: pre-registration into a number that had always been met — the same reason ADR-0005's
#: pre-registered strategy survives its own A/B, and the reason `security/report.py` prints both
#: figures rather than only the one now being met.
GATE_LATENCY_BUDGET_PREREGISTERED_MS = 800

#: The p50 the input gate is judged against today — ADR-0006's T7 amendment §2, in one place.
#:
#: **A target to report against, not a timeout to enforce**, and the distinction is the whole
#: reason it is a constant here rather than a branch somewhere: the acceptance criterion is that
#: the gate *is* this fast, measured from the structured logs over a real run, and a gate that
#: enforced it by abandoning slow calls would satisfy the number by not doing the work.
#: `security/report.py` compares the measured p50 against this and the README quotes it, so the
#: prose and the verdict cannot disagree — bound by
#: `tests/test_grounding_scope.py::test_the_readme_names_both_latency_budgets`, as an **equality
#: on the millisecond figure**. That test asserted `GATE_LATENCY_BUDGET_MS // 1000` against the
#: string `"1 s"` until issue #8's review, which made 1000–1999 ms indistinguishable *and*
#: passed on the words "the Tier-1 spec" while the README named no revised figure — so this
#: sentence was false in the one place written to keep it true.
#:
#: One second rather than 800 ms, and revised for reasons rather than to fit: it is a figure the
#: gate cleared on every one of eight measured passes, where 800 ms was cleared on two. The gate
#: is 9–21% of the wait it sits inside (measured, three real turns), and the one model that
#: clears 800 ms at an equal attack catch rate blocked a legitimate analyst question — trading
#: catch quality for latency on a security control, which is the trade ADR-0006 exists to
#: refuse.
GATE_LATENCY_BUDGET_MS = 1000

#: ADR-0005's budget for the p50 latency **added by query translation**, in milliseconds.
#:
#: The other pre-registered latency figure, and it lives beside the gate's for the reason
#: CLAUDE.md gives about a pair split across two files: "the latency budgets" are named in the
#: single-source-of-truth list, and this one spent T10 as a `budget_ms: float = 1500.0` default
#: argument in `evaluation/latency.py` — an unbound literal, while the gate's twin was bound by
#: an equality in `tests/test_grounding_scope.py`. `evaluation/report.py` prints the measured
#: total beside this, and the artifact's verdict line is a comparison against it.
#:
#: **A budget dominance is judged *within*, not a timeout**, exactly like the gate's above:
#: ADR-0005's clause is that `hybrid + translation` dominates *and* costs no more than this, so
#: a run that exceeds it puts the default in question rather than aborting a retrieval.
TRANSLATION_LATENCY_BUDGET_MS = 1500

#: How long the classifier's single model call may take before it is abandoned, in seconds.
#:
#: Five, which is deliberately far above the p50 budget above and far below the answering path's
#: 60. It is a **circuit breaker, not a budget**: the budget is a median to report, and this is
#: the point past which a hung provider stops being a slow gate and starts being a hung app.
#: Tighter would convert ordinary jitter into a fail-open (see `classifier.classify`), which
#: trades a rare slow turn for a routinely-skipped layer.
GATE_TIMEOUT_SECONDS = 5

#: Retries on the classifier's call. **One attempt, no retry**, unlike every other fetch in this
#: codebase — because a retry multiplies the worst case inside a latency budget, and the failure
#: mode here is already survivable: the gate fails open onto three other layers, where a retried
#: fetch in `finance/` is the difference between a figure and no figure.
GATE_CLASSIFIER_ATTEMPTS = 1

#: How much of a **blocked** turn's normalised input the gate-trigger log line records.
#:
#: **The one place a line written by `log_event` may carry user-derived text**, and the
#: exception is narrow on purpose (CLAUDE.md; ADR-0006 requires the normalised input in the
#: gate-trigger record). Three bounds make it proportionate: only on a **block**, only the
#: **normalised** form, and only this many characters. A pass logs counts and verdicts like
#: every other event.
#:
#: **What the normalised form does and does not hide, stated precisely.** It is lowercased,
#: de-accented, de-homoglyphed, de-leetspeaked and stripped of punctuation, which destroys
#: *figures and identifiers* — `Item 1A` becomes `item ia`, `$5bn` becomes `ssbn` — but leaves
#: **the wording legible**: "Can you ignore the tax effects and just give me the gross margin?"
#: normalises to `can you ignore the tax effects and just give me the gross margin`. This
#: docstring said "useless as a question", and ADR-0006 §5 and CLAUDE.md said "unusable as a
#: question", and none of them was true (issue #8 review). It matters because a false positive
#: is exactly the case this records, so the mitigation has to be described as what it is: a
#: question stripped of its numbers and still perfectly readable, kept for 500 characters.
#:
#: The argument for logging it at all is that a denylist you cannot audit is a denylist you
#: cannot tune: reviewing a false positive means seeing what tripped it. The argument against is
#: that a false positive means an innocent question ends up in a kept log — a real cost, and
#: the reason for the three bounds rather than a reason to have no record. 500 characters is an
#: eighth of `MAX_QUESTION_CHARS` — enough for the payload in a long paste, not the paste.
GATE_LOGGED_INPUT_MAX_CHARS = 500

#: How much of an **unparseable** classifier reply `gate_classifier_unparsed` records.
#:
#: A logged-input cap, which CLAUDE.md puts here rather than beside the rule — and this one
#: had been an inline `token[:32]` in `security/classifier.py`, justified by a comment saying
#: the value was "the classifier's own output vocabulary, not the analyst's text". That is true
#: of a *compliant* model and false in the branch the field is logged from: the parser reaches
#: it only when the reply was **not** one of the two labels, which is exactly when a confused
#: or jailbroken classifier may be echoing the question back (issue #8 review). So it is
#: bounded as user-derived text, at one short token — enough to tell "the model answered a
#: sentence" from "the model answered nothing", which is all the field is for.
GATE_LOGGED_CLASSIFIER_TOKEN_MAX_CHARS = 32

#: The **answering** path's per-request ceiling and retry count — `llm.build_chat_model`'s
#: defaults.
#:
#: Here rather than in `llm.py` because `GATE_TIMEOUT_SECONDS` and `GATE_CLASSIFIER_ATTEMPTS`
#: above are the same knob for the other path, and a pair split across two files is a pair that
#: drifts (issue #8 review). Generous on purpose, and the contrast with the gate's five seconds
#: and no retry is the point: an answer is what the analyst is waiting for, so a retried 503 is
#: cheaper than a failed turn — where a slow *classifier* call holds a turn in front of the
#: refusal it was deciding about.
ANSWER_TIMEOUT_SECONDS = 60
ANSWER_MAX_RETRIES = 2

#: How long a question may be before the app declines to send it (user story 21).
#:
#: A cost and abuse bound, not a linguistic one: no analyst's question is 4,000 characters, and
#: what arrives at that length is a paste — often a document with instructions in it, which is
#: ADR-0006's problem and cheaper to refuse here than to classify. Enforced in the UI because
#: that is the only door a human types through; the agent's own inputs are bounded by the tool
#: signatures.
MAX_QUESTION_CHARS = 4000

#: How many questions one browser session may ask before the app stops answering (T12 item 6).
#:
#: **Cost and abuse limiting, and explicitly not a security control.** A refresh mints a new
#: `session_state` and therefore a new counter, so anyone who wants past this walks past it —
#: and saying so is the point rather than a caveat: ADR-0006's input gate is the security
#: boundary, and a session counter advertised as rate limiting would be a claim the code cannot
#: support. What it does buy, on a public demo deployment with the author's own key behind it,
#: is that one tab left open on a script cannot spend the budget for everyone.
#:
#: A real rate limit is keyed server-side on something the client does not choose — an account,
#: an IP, a token bucket in a shared store — and needs the auth Tier-2 defers (ADR-0010 §6).
#: That is a ticket, not a constant.
#:
#: Its sibling `MAX_QUESTION_CHARS` is a module constant rather than env-overridable and this
#: follows it, for the same reason: both are bounds on what the one human-facing door accepts,
#: and a deployment that wants a different answer is changing what the app *is* rather than
#: configuring it.
MAX_QUESTIONS_PER_SESSION = 40


# --------------------------------------------------------------------------------------
# Retrieval strategy switches (ADR-0004, ADR-0005)
# --------------------------------------------------------------------------------------


class RetrievalStrategy(StrEnum):
    """The retrieval strategies the A/B harness compares."""

    VECTOR = "vector"
    HYBRID = "hybrid"


#: Pre-registered shipping default (ADR-0005), fixed before any A/B data exists.
DEFAULT_STRATEGY = RetrievalStrategy.HYBRID
DEFAULT_TRANSLATION_ENABLED = True
#: ADR-0004's latency-driven ceiling on the planner's sub-queries. Named beside the other two
#: because it is the third thing the shipping default *is*, and `evaluation/arms.py` had a
#: hardcoded copy of the literal that `shipping_default_matches_config` could not see (code
#: review of #11). Still env-overridable through `FINBRIEF_MAX_SUB_QUERIES` — this is the
#: default, not a second knob.
DEFAULT_MAX_SUB_QUERIES = 3

#: Reciprocal Rank Fusion's rank-smoothing constant: a candidate list's vote for the chunk it
#: ranks `r`th is `1 / (RRF_K + r)` (`retrieval/hybrid.py`).
#:
#: **60 is the published default, stated and not tuned** — Cormack, Clarke & Buettcher (2009),
#: who introduced RRF and report it as insensitive over a wide range. Deliberately *not* a
#: `Settings` field, on the same reasoning as `CHUNK_SIZE_CHARS` below and for one extra one:
#: ADR-0005 pre-registers the shipping default before any A/B data exists so that the winner
#: cannot be picked after seeing the numbers, and a fusion constant somebody could sweep is
#: exactly the back door into that. A `hybrid` result obtained at the published constant is a
#: prediction that survived a test; one obtained at the best of several `RRF_K` values is a
#: number about the sweep. If it is ever changed, it is changed here, in one commit, with the
#: A/B re-run — never per environment.
RRF_K = 60

#: Maximum characters per chunk — the PLAN.md baseline for section-aware chunking
#: (RecursiveCharacterTextSplitter, 1000 chars with a 200-char overlap; the overlap
#: belongs to the chunker itself and lands with it in Phase 1).
#:
#: **Ticket T2 (KB ingest, #3) owns tuning this value**, since it lands with the chunker in
#: Phase 1 — Tier-1 work, not post-gate Tier-2 — and it is the single source of truth for
#: it: two things depend on it. `retrieval/embeddings.py` sends raw strings with no length-safe
#: splitting, so a chunk over the embedding model's 8191-token window would fail the
#: request outright — `tests/test_chunk_token_limit.py` imports this constant and asserts
#: the worst-case token count it implies stays under that window, so raising it here
#: re-checks the guarantee instead of silently voiding it.
#: Changing it also invalidates an existing index, which must be re-ingested.
#: A deliberately plain constant, not a `Settings` field: chunk size is not a Tier-1 A/B
#: axis, and an env override would let a running app disagree with the index on disk.
CHUNK_SIZE_CHARS = 1000


# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------

_ENV_FILE_HINT = (
    "Copy .env.example to .env and fill it in (or set it in the deployment environment)."
)

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


@dataclass(frozen=True, slots=True)
class Settings:
    """Resolved runtime configuration.

    `openrouter_api_key` is `repr=False` so the secret cannot leak into a traceback,
    a log line, or a debugger dump of this object.
    """

    openrouter_api_key: str = field(repr=False)
    openrouter_base_url: str
    chat_model: str
    #: The model the input gate's zero-shot classifier calls (ADR-0006 layer 3). Its own field
    #: rather than `chat_model` because the two are priced against different jobs: the gate pays
    #: for one YES/NO per turn and wants the cheapest model that can read a sentence, where the
    #: answering model is what the analyst's brief is worth. Raising one must not raise the
    #: other — which is exactly what a single field would do.
    classifier_model: str
    #: The model RAGAs scores with (T10, #11) — the **judge**, and deliberately not
    #: `chat_model`.
    #:
    #: ADR-0002 decision 1 separates the ground truth from the answering pipeline: candidate Q/A
    #: pairs were drafted by a different model and hand-verified, so the references cannot be
    #: circular. A judge that *is* the answering model puts the circularity back at the other
    #: end of the same measurement — the pipeline grading its own output — and the whole reason
    #: the golden set cost a day of reading filings is to avoid that. Its own field for the
    #: reason `classifier_model` has one: three roles, three prices, and raising one must not
    #: raise the others.
    #:
    #: The default is a *stronger* model than the answerer rather than a cheaper one, which is
    #: the opposite of the gate's choice and for the opposite reason: the gate pays for one
    #: YES/NO per turn, while a judge that misreads a filing passage silently moves every number
    #: in the report. Measured cost of the difference over a full six-arm run: about $1.28
    #: against $0.57 (docs/verification/evaluation.md records the arithmetic).
    judge_model: str
    embedding_model: str
    retrieval_strategy: RetrievalStrategy
    query_translation_enabled: bool
    retrieval_k: int
    max_sub_queries: int
    eval_mode: bool
    #: **Deliberately unread, and this is the record of that.** Alpha Vantage was scoped as a
    #: fundamentals fallback for when yfinance — an unofficial client for a private endpoint —
    #: returns nothing. It is deferred rather than shipped: its free tier is **25 calls a day**,
    #: which one `calculate_ratios` on a `big_tech` company would spend a quarter of, so a
    #: fallback that fired on a bad afternoon would exhaust the budget and then fail anyway.
    #:
    #: What survives the free tier instead is the path in `finance/cache.py`: a TTL window, a
    #: retry, and a **stale serve with its age** rather than an error. A second source would be
    #: a second thing to be down; the stale banner is honest about the same outage for free.
    #:
    #: So this flag exists, is validated, and is read by nothing — which a reader is entitled to
    #: find suspicious, hence this block. Written up in `.env.example` beside
    #: `ALPHAVANTAGE_API_KEY` and in PLAN §4 under what T5 deferred; flipping it on today
    #: changes no behaviour, and wiring it up is a ticket, not a config change (#9 review).
    alphavantage_enabled: bool
    sec_edgar_user_agent: str
    chroma_dir: str
    checkpoint_db: str
    #: What a million input / output tokens of `chat_model` cost, in dollars — or `None`, which
    #: is the default and means *unpriced* (T12 item 5).
    #:
    #: **Configuration rather than a table in the repo, and `None` rather than a guess.** This
    #: project reaches every model through OpenRouter, which fronts many upstreams and routes by
    #: availability, so the price of a call is not something this codebase can assert — a
    #: hardcoded figure would be a number nobody measured, going stale silently, in a panel
    #: whose whole subject is spend. Unset, the meter reports tokens and says the cost is
    #: unpriced; set, it multiplies. Two fields rather than one because input and output are
    #: priced differently everywhere, and one blended figure would have to be wrong for both.
    input_cost_per_mtok: float | None
    output_cost_per_mtok: float | None

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> Settings:
        """Build settings from an environment mapping, failing loudly on bad input."""
        return cls(
            openrouter_api_key=_required(env, "OPENROUTER_API_KEY"),
            openrouter_base_url=_string(
                env, "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
            chat_model=_string(env, "FINBRIEF_CHAT_MODEL", "openai/gpt-4o-mini"),
            classifier_model=_string(env, "FINBRIEF_CLASSIFIER_MODEL", "openai/gpt-4o-mini"),
            judge_model=_string(env, "FINBRIEF_JUDGE_MODEL", "openai/gpt-4.1-mini"),
            # Served by OpenRouter's /v1/embeddings, so it needs no key or base URL of
            # its own. One model for ingest and query — see retrieval/embeddings.py.
            embedding_model=_string(
                env, "FINBRIEF_EMBEDDING_MODEL", "openai/text-embedding-3-small"
            ),
            retrieval_strategy=_strategy(env, "FINBRIEF_RETRIEVAL_STRATEGY", DEFAULT_STRATEGY),
            query_translation_enabled=_boolean(
                env, "FINBRIEF_QUERY_TRANSLATION", DEFAULT_TRANSLATION_ENABLED
            ),
            retrieval_k=_integer(env, "FINBRIEF_RETRIEVAL_K", 5, minimum=1),
            # ADR-0004 caps translation at 3 sub-queries for latency. The cap is enforced,
            # not merely defaulted: the latency budget ADR-0005 judges dominance within
            # (<=1.5s p50 added by translation) assumes it, so an env override must not be
            # able to quietly invalidate the A/B result.
            max_sub_queries=_integer(
                env,
                "FINBRIEF_MAX_SUB_QUERIES",
                DEFAULT_MAX_SUB_QUERIES,
                minimum=0,
                maximum=DEFAULT_MAX_SUB_QUERIES,
            ),
            eval_mode=_boolean(env, "FINBRIEF_EVAL_MODE", False),
            alphavantage_enabled=_boolean(env, "FINBRIEF_ALPHAVANTAGE_ENABLED", False),
            sec_edgar_user_agent=resolve_sec_edgar_user_agent(env),
            # Where the persisted Chroma collections live. An override exists because a
            # deployment's writable path is not the repo's, and because the evaluation
            # harness builds throwaway indexes; the default keeps ingest and app pointed
            # at the same directory without configuration.
            chroma_dir=_string(env, "FINBRIEF_CHROMA_DIR", "data/chroma"),
            # The agent's memory of record (ADR-0008): a SQLite file, overridable for the
            # same reason `chroma_dir` is — a deployment's writable path is not the repo's.
            # Ephemeral by design on Streamlit Community Cloud; conversations are not durable
            # data, and losing them costs a conversation rather than the knowledge base.
            checkpoint_db=_string(env, "FINBRIEF_CHECKPOINT_DB", "data/checkpoints.sqlite"),
            input_cost_per_mtok=_price(env, "FINBRIEF_INPUT_COST_PER_MTOK"),
            output_cost_per_mtok=_price(env, "FINBRIEF_OUTPUT_COST_PER_MTOK"),
        )


def _raw(env: Mapping[str, str], name: str) -> str:
    """The trimmed value of `name`, or `""` when unset or blank.

    The single entry point the typed readers below share, so "unset" and "whitespace only"
    mean the same thing everywhere — see `test_blank_api_key_is_treated_as_missing`.
    """
    return env.get(name, "").strip()


def _required(env: Mapping[str, str], name: str) -> str:
    value = _raw(env, name)
    if not value:
        raise ConfigError(f"{name} is not set. {_ENV_FILE_HINT}")
    return value


def _string(env: Mapping[str, str], name: str, default: str) -> str:
    return _raw(env, name) or default


def _boolean(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = _raw(env, name).lower()
    if not raw:
        return default
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise ConfigError(
        f"{name}={env[name]!r} is not a boolean. Use one of: "
        f"{', '.join(sorted(_TRUE | _FALSE))}."
    )


def _integer(
    env: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    raw = _raw(env, name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name}={raw!r} is not an integer.") from exc
    if value < minimum:
        raise ConfigError(f"{name}={value} is below the minimum of {minimum}.")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{name}={value} is above the maximum of {maximum}.")
    return value


def _price(env: Mapping[str, str], name: str) -> float | None:
    """A dollars-per-million-tokens price, or `None` when unset — never a default figure.

    There is deliberately no default to fall back on: an unset price means the cost meter
    reports
    tokens and says it cannot price them, which is honest, where a stand-in would put a dollar
    figure on screen that no rate card produced.

    Negative is refused rather than clamped, on the same reasoning `_integer` takes a `minimum`:
    a negative price is a typo, and silently treating it as zero would report a spend of
    nothing.
    """
    raw = _raw(env, name)
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name}={raw!r} is not a number of dollars.") from exc
    if value < 0:
        raise ConfigError(f"{name}={value} is negative; a price cannot be below zero.")
    return value


def _strategy(
    env: Mapping[str, str], name: str, default: RetrievalStrategy
) -> RetrievalStrategy:
    raw = _raw(env, name).lower()
    if not raw:
        return default
    try:
        return RetrievalStrategy(raw)
    except ValueError as exc:
        valid = ", ".join(s.value for s in RetrievalStrategy)
        raise ConfigError(f"{name}={raw!r} is not a strategy. Use one of: {valid}.") from exc


@lru_cache(maxsize=1)
def load_env() -> None:
    """Load `.env` into the process environment, once.

    Existing environment variables win, so deployment env vars and `st.secrets`
    (which Streamlit also exposes as env vars) are never overwritten by a stray
    local `.env`.
    """
    load_dotenv(override=False)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The application's settings. Cached; call `get_settings.cache_clear()` in tests."""
    load_env()
    return Settings.from_env(os.environ)


def resolve_sec_edgar_user_agent(env: Mapping[str, str]) -> str:
    """Resolve `SEC_EDGAR_USER_AGENT` alone, without the rest of `Settings`.

    Deliberately independent of `Settings` for the same reason `resolve_log_level` is:
    a `--dry-run` ingest needs an EDGAR identity and no OpenRouter key, and
    `Settings.from_env` hard-requires the key. `Settings.from_env` reads the field
    through this same function, so the two resolutions cannot drift.
    """
    return _raw(env, "SEC_EDGAR_USER_AGENT")


def resolve_log_level(env: Mapping[str, str]) -> int:
    """Resolve `LOG_LEVEL` to a `logging` level.

    Deliberately independent of `Settings`: logging must be configurable before — and
    without — a valid API key, so a config failure can still be reported through it.
    Takes an explicit mapping for the same reason `Settings.from_env` does — one
    resolution path, and no hidden `.env` read on the way; the caller loads the
    environment (`load_env()`) and passes it in.
    """
    raw = _raw(env, "LOG_LEVEL").upper()
    if not raw:
        return logging.INFO
    # `getLevelNamesMapping()` includes NOTSET (0), which is not a threshold but "inherit
    # from the parent". Accepting it would set the package logger to 0 and — with
    # `propagate=False` — let its effective level fall back to root's WARNING, silently
    # dropping every INFO event Phase 6/7 reads back. A blackout is worse than a failure.
    levels = {
        name: level
        for name, level in logging.getLevelNamesMapping().items()
        if name != "NOTSET"
    }
    if raw not in levels:
        valid = ", ".join(sorted(levels))
        raise ConfigError(f"LOG_LEVEL={raw!r} is not a log level. Use one of: {valid}.")
    return levels[raw]


def resolve_log_file(env: Mapping[str, str]) -> Path | None:
    """Resolve `FINBRIEF_LOG_FILE` — where the JSON-lines sink appends, or `None` for nowhere.

    Independent of `Settings` for the same reason `resolve_log_level` is: the sink is installed
    before a key is validated, so a config failure is still recorded by the run that failed.

    **Unset means off, and that is load-bearing rather than timid.** `configure_logging` is
    called with no arguments by all four entry points, `tests/conftest.py` strips every
    `FINBRIEF_` variable, and a path-shaped default would therefore have the hermetic suite
    appending to a file in the working directory on every run. The cost of default-off is that
    the sink has to be *turned on* where the data matters — `.env.example`, the README's
    evaluation-run instructions and T11's demo walkthrough all name it, because a log nobody
    enables is an instrument that ships and never runs (T8, #10).

    No default path is offered here on purpose: naming one in code and another in `.env.example`
    is two sources of truth for one string. `.env.example` is the one a reader acts on — and it
    carries the recommendation **commented out**, so that acting on it (`cp .env.example .env`,
    which the README and `missing_key_message` both advise) does not silently enable a sink that
    retains blocked questions' normalised text. Default-off in code and default-on in the file a
    reader copies is not a default-off (issue #10 review).
    """
    raw = _raw(env, "FINBRIEF_LOG_FILE")
    return Path(raw) if raw else None
