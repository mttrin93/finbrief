"""The grounding-scope disclosure's arithmetic, against the committed ingest evidence.

`prompts.py` derives "54 of 60" from `config` so that a Universe change cannot leave a stale
number in the app's scope panel. That keeps the sentence *self*-consistent, which is not the
same as *true*: the subtraction assumes each pointer filer really does answer Item 7A by
reference, and `ITEM_7A_POINTER_FILERS` is a hand-maintained tripwire. The only witness to
what the knowledge base actually holds is `docs/verification/ingest-report.md`, written by
the ingest run itself.

The cross-check belongs here and not in `prompts.py`: the app must render its scope from
`config` alone, with no generated artifact on disk — a missing or half-written report should
never take the UI down, and importing one at startup would couple the page to the last run.
A test is where the two are allowed to disagree, loudly and in CI.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from finbrief.config import (
    ALPHAVANTAGE_FREE_TIER_CALLS_PER_DAY,
    CLUSTERS,
    PEERS,
    QUOTE_TTL_SECONDS,
    UNIVERSE,
)
from finbrief.ingestion.model import Section
from finbrief.prompts import (
    AGENT_SYSTEM_PROMPT,
    GROUNDING_SCOPE,
    GROUNDING_SCOPE_DETAILS,
    GROUNDING_SCOPE_VERIFY,
    LIVE_DATA_SCOPE,
    SEARCH_FILINGS_DESCRIPTION,
    SYSTEM_PROMPT,
    UNIVERSE_ROWS,
    query_translation_prompt,
)

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "docs" / "verification" / "ingest-report.md"
README = ROOT / "README.md"

#: The README's linked depth (T11 follow-up, #12). The README was a 1,709-line document that
#: could not be read in one pass; it is now a ten-minute tour and three files it links to.
#: **A binding does not weaken because its subject moved** — each assertion below targets the
#: file the text landed in, and the docstring says which. What would weaken it is retargeting
#: at "any of the four", which is why only the figure lists (whose figures are legitimately
#: spread across files) use `prose()`.
IMPLEMENTATION = ROOT / "docs" / "implementation.md"
FINDINGS = ROOT / "docs" / "findings.md"
LIMITATIONS = ROOT / "docs" / "limitations.md"
DEMO = ROOT / "docs" / "demo.md"
DOCUMENTS = (README, IMPLEMENTATION, FINDINGS, LIMITATIONS, DEMO)


def flat(text: str) -> str:
    """Whitespace-flattened, so a markdown table may wrap and a row still matches.

    Table rows are matched as `label | value` fragments rather than whole lines for the same
    reason `test_the_readme_states_the_same_scope_the_app_does` flattens: the alternative is a
    test that dictates the README's column widths.

    Blockquote markers are dropped for the same reason whitespace is. A quoted sentence that
    wraps inside a `>` block reads as `… for each of > the 15 companies …` once flattened, so a
    verbatim binding would be asserting the README's line breaks rather than its words.
    """
    return " ".join(word for line in text.splitlines() for word in line.lstrip(">").split())


def prose(*paths: Path) -> str:
    """The named documents, whitespace-flattened and concatenated.

    Defaults to all five. Flattened because these files wrap their prose and the panel and
    the artifacts do not, which is the same reason every README assertion here has always
    flattened first: the alternative is a test that dictates where a sentence may break.
    """
    return " ".join(flat(path.read_text(encoding="utf-8")) for path in (paths or DOCUMENTS))


#: A gate-table row: `JPM    FY2025    39,177    112,774    394,858    ->Item 7`. The
#: fiscal year is matched but not captured — it differs by filer (NVDA is FY2026), which is
#: itself one of the things the disclosure states.
_GATE_ROW = re.compile(r"^([A-Z]+)\s+FY\d{4}\s+\S.*$", re.MULTILINE)

#: `->Item 7` in the Item 7A column: the run's own record that this filer incorporated its
#: market-risk disclosure by reference, so the pair was lawfully not chunked (ADR-0007 §4).
_POINTER_MARKER = "->Item 7"

#: `1, 1A, 7 and 7A` — how a prose scope sentence reads the Items out. Rebuilt from the enum
#: here rather than imported from `prompts._ITEM_LABELS`, so this file checks the derivation
#: instead of restating it: importing the private constant would make both sides of the
#: assertion the same expression and it would pass however wrong that expression was.
_SECTIONS = tuple(Section)
_ITEM_LABELS_IN_PROSE = (
    ", ".join(section.item for section in _SECTIONS[:-1]) + f" and {_SECTIONS[-1].item}"
)


def report_text() -> str:
    assert REPORT.exists(), (
        f"{REPORT} is the evidence the disclosure is checked against and is committed to "
        f"the repo. Restore it, or re-run a full-Universe ingest to regenerate it."
    )
    return REPORT.read_text(encoding="utf-8")


def gate_rows(text: str) -> dict[str, str]:
    rows = {match.group(1): match.group(0) for match in _GATE_ROW.finditer(text)}
    assert rows, "the gate table's shape changed; this cross-check reads it"
    return rows


def test_the_disclosure_counts_what_the_ingest_run_actually_gated():
    # 15 x 4 = 60 slots, 6 of them answered by reference, 54 chunked. Every one of those
    # numbers is asserted against the run's own table rather than against `config`, which
    # is what `prompts.py` already derives them from.
    text = report_text()
    rows = gate_rows(text)
    slots = len(UNIVERSE) * len(Section)
    pointers = sum(1 for row in rows.values() if _POINTER_MARKER in row)

    assert len(rows) == len(UNIVERSE), "the run gated every company in the Universe"
    assert f"All {slots} company x Section checks passed." in text
    assert f"{pointers} Section(s) incorporated by reference into Item 7" in text
    # **In the panel, which is where the count now lives** (#13). It used to end
    # `GROUNDING_SCOPE`, the caption under the app's title, and moved when that caption was cut
    # to the sentence a reader can use before knowing what a Section is. Asserted against the
    # panel and not against "the page", which would pass wherever it had drifted to — including
    # back into a caption that is quoted by three prompts.
    assert f"{slots - pointers} of {slots} company × Section pairs" in " ".join(
        GROUNDING_SCOPE_DETAILS
    )
    assert f"{slots - pointers} of {slots}" not in GROUNDING_SCOPE, (
        "the caption states the scope; the panel states the arithmetic, and only one of them"
    )


def test_the_pointer_filers_named_on_screen_are_the_ones_the_run_found():
    # The sharpest sentence in the panel: a reader who does not know these six answer Item
    # 7A by reference reads "no Item 7A" as "no market-risk grounding". If the set on screen
    # and the set in the evidence ever diverge, the panel is telling a reader to look under
    # the wrong Item.
    #
    # **The list is asserted as one derived string, and the membership as an equality over
    # whole words** (#13). The old form tested `f"{ticker}," in panel` per ticker, which is
    # satisfied by six tickers in any order, in any sentence, separated by anything —
    # and would have passed on a panel that named five of them in the list and the sixth in a
    # footnote. Whole words because `F` is a Universe ticker and a substring of everything.
    rows = gate_rows(report_text())
    found = {ticker for ticker, row in rows.items() if _POINTER_MARKER in row}
    panel = " ".join(GROUNDING_SCOPE_DETAILS)

    named = {t for t in rows if re.search(rf"(?<![A-Za-z0-9]){t}(?![A-Za-z0-9])", panel)}
    assert named == found, "the disclosure names exactly the filers the run found"
    assert f"{', '.join(sorted(found))} answer Item 7A by pointing at Item 7" in panel
    assert f"all {len(UNIVERSE)} companies have market-risk grounding" in panel
    assert f"{len(UNIVERSE) - len(found)} have an `Item 7A` Section." in panel


def test_the_universe_table_is_two_columns_and_both_are_derived():
    """Ticker · Company, in `UNIVERSE` order, neither column composed here (#13).

    The table used to carry a third column — `Section` or `→ Item 7` per filer — and this file
    bound it, row by row, to the ingest run's own gate table. **That column is gone because it
    truncated the names**: three columns at the sidebar's width left `Company` too narrow to
    finish `Microsoft Corporation`, and a name a reader cannot tell from a prefix of itself
    discloses less than no column at all.

    What was lost is the row granularity, not the claim, and not its binding: the Item 7A
    disclosure is `GROUNDING_SCOPE_DETAILS`', which names the six filers and both counts, and
    `test_the_pointer_filers_named_on_screen_are_the_ones_the_run_found` above holds *that*
    sentence to what the run found. So the evidence tie survives the column; only the
    per-company rendering of it went.

    **Both columns asserted as whole-sequence equalities**, which is what keeps this from
    drifting into a shape check: a `len()` or an `in` would pass on a table that repeated one
    company fifteen times, and the ordering is load-bearing now that the panel's caption tells a
    reader the rows follow cluster order.
    """
    assert list(UNIVERSE_ROWS[0]) == ["Ticker", "Company"]
    assert [r["Ticker"] for r in UNIVERSE_ROWS] == [c.ticker for c in UNIVERSE]
    # The company column is the whole reason the table replaced a list of tickers, so it is the
    # legal name from `config` and not a ticker repeated or a shortened form composed here.
    assert [r["Company"] for r in UNIVERSE_ROWS] == [c.name for c in UNIVERSE]


#: What ADR-0007 obliges the panel to disclose, one entry per claim, as
#: `(what it is, a phrase that can only appear if the claim is being made)`.
#:
#: A list rather than one big assertion because the compression this guards is a *wording*
#: change (#13): the panel went from seven bullets to five, and the way that stops
#: being an edit and becomes a deletion is a claim quietly going with a bullet. Each phrase is
#: chosen to be unsatisfiable by prose that does not make the claim — "10-Q" cannot appear in a
#: panel that has stopped excluding quarterly filings.
SCOPE_PANEL_OBLIGATIONS = (
    ("the Item 8 financials are excluded", "Item 8"),
    ("10-Qs are excluded", "10-Q"),
    ("other Items are excluded", "every other Item"),
    ("table fidelity is a stated limitation", "lose its layout"),
    ("one filing per company, so fiscal years differ", "fiscal year differs by filer"),
    ("live figures are not from the filings", "never come from the filings"),
    ("a stale figure is shown with its age", "last cached figure with its age"),
    ("an unfetchable figure says so", "could not be fetched"),
    ("and neither is ever a placeholder", "never a placeholder"),
)


@pytest.mark.parametrize(("claim", "phrase"), SCOPE_PANEL_OBLIGATIONS)
def test_the_compressed_scope_panel_still_makes_every_claim_it_owes(claim, phrase):
    """ADR-0007's disclosure survived being shortened. The Items and filers are asserted by
    the two tests above; these are the claims that have no derived number to bind them."""
    panel = " ".join(GROUNDING_SCOPE_DETAILS)

    assert phrase in panel, f"the scope panel no longer says {claim}"


def test_the_scope_panel_is_five_short_lines_of_plain_language():
    """The shape of the compression, as an equality — a bound would pass on the seven-bullet
    panel this replaced (CLAUDE.md: prefer an equality over a bound).

    Two properties, both of which the old panel failed. **Five lines**, because "compress" that
    permits any number of bullets is not a constraint; a sixth claim belongs in the README, as
    the ingest-report provenance now is. And **no ADR numbers**, because this panel is read by
    an analyst mid-question: `(ADR-0009)` sent them looking for a document that is not in the
    app. The design record is still in `docs/adr/` and still cited from the code — this rule is
    about the copy on screen.
    """
    assert len(GROUNDING_SCOPE_DETAILS) == 5

    on_screen = " ".join((*GROUNDING_SCOPE_DETAILS, GROUNDING_SCOPE_VERIFY))
    assert not re.search(r"ADR-\d+", on_screen), "no ADR numbers in the user-facing copy"
    # One sentence each: the marker is a full stop with a word after it, so an abbreviation
    # ("10-K.") and the closing stop are both fine and a second sentence is not.
    for detail in GROUNDING_SCOPE_DETAILS:
        assert not re.search(r"\.\s+\S", detail), f"one sentence per line; got {detail!r}"


def test_the_ingest_report_provenance_is_stated_beside_the_counts():
    """Where the counts come from is a reviewer's question, so it is answered in prose and not
    in the app's panel. It left the panel when that panel was compressed (#13) and had to land
    somewhere — a claim dropped from one surface and added to none is the deletion a compression
    must not be.

    **It has since moved once more**, from the README to `docs/implementation.md`, with the
    disclosure whose counts it is the provenance *for* (T11 follow-up). That is the invariant
    worth binding: the provenance sentence sits beside the counts, wherever they are, and never
    in the panel."""
    disclosure = " ".join(IMPLEMENTATION.read_text(encoding="utf-8").split())
    panel = " ".join(GROUNDING_SCOPE_DETAILS)

    assert "docs/verification/ingest-report.md" in disclosure
    assert "not a live count of the index" in disclosure, "and what the counts are *not*"
    assert "ingest-report" not in panel, "the panel no longer carries it"


def test_every_scope_claim_in_a_prompt_is_derived_and_not_typed():
    # The surface the T4 review found unbound: `search_filings`' description opens with a
    # scope sentence, and it had been typed by hand — "Items 1, 1A, 7, 7A" and "the fifteen
    # Universe companies" as literals in a prompt the *model* reads and plans its searches
    # against. A 16th company or a fifth Section leaves that sentence quietly wrong, which is
    # the exact failure `prompts.py` exists to prevent for the sentence under the app's title.
    #
    # Asserted structurally rather than by string: every prompt that makes a scope claim must
    # spell the Items and the Universe size the way the enum and `config` currently do.
    items = ", ".join(section.item for section in tuple(Section)[:-1])
    for name, prompt in (
        ("SEARCH_FILINGS_DESCRIPTION", SEARCH_FILINGS_DESCRIPTION),
        ("SYSTEM_PROMPT", SYSTEM_PROMPT),
        ("AGENT_SYSTEM_PROMPT", AGENT_SYSTEM_PROMPT),
        # A prompt whether or not it is a constant: the planner reads the same scope sentence
        # and decides what to decompose against it, so a hand-typed count here misdirects
        # retrieval itself rather than only the reader (issue #6 review). The argument is the
        # enforced cap, not part of the scope claim.
        ("query_translation_prompt", query_translation_prompt(3)),
    ):
        assert f"Items {items}" in prompt, f"{name} lists the Items from the enum"
        assert str(len(UNIVERSE)) in prompt, f"{name} counts the Universe from config"
        assert "fifteen" not in prompt.lower(), (
            f"{name} spells a derived count as a word, which no longer tracks `config`"
        )


def test_the_tool_description_states_the_scope_the_app_does():
    # Same words, one source. The model is choosing whether a question is answerable from the
    # knowledge base, so a description that overstated the scope would have it search for
    # filings that were never ingested and report the empty result as the company's silence.
    assert f"Items {_ITEM_LABELS_IN_PROSE} of the latest annual 10-K" in GROUNDING_SCOPE
    assert f"Items {_ITEM_LABELS_IN_PROSE} of the latest annual 10-K" in (
        SEARCH_FILINGS_DESCRIPTION
    )
    assert f"{len(UNIVERSE)} companies in FinBrief's Universe" in SEARCH_FILINGS_DESCRIPTION


def test_the_readme_states_the_same_scope_the_app_does():
    # `prompts.py` names the README as a consumer of the scope details, which makes the
    # README a second copy of a sentence the module exists to keep singular. It cannot import
    # from `prompts`, being prose, so the binding is here: the load-bearing facts are
    # asserted rather than the whole string, since the README wraps its lines and the panel
    # does not. Add a Section or change the Universe and this fails until the prose follows.
    #
    # Whitespace is flattened first, so a sentence may wrap wherever it reads best — the
    # alternative is a test that dictates the README's line breaks.
    #
    # **Retargeted at `docs/implementation.md`** (T11 follow-up): the disclosure section moved
    # there when the README became a ten-minute tour. The README keeps a summary, and the two
    # claims it makes — the sentence itself and the pair count — are bound by
    # `test_the_readme_quotes_the_grounding_scope_sentence_the_app_renders` below. Everything
    # this test asserts is a *derived* fact that only the full disclosure states.
    readme = " ".join(IMPLEMENTATION.read_text(encoding="utf-8").split())
    rows = gate_rows(report_text())
    slots = len(UNIVERSE) * len(Section)
    pointers = {ticker for ticker, row in rows.items() if _POINTER_MARKER in row}

    assert f"**{slots - len(pointers)} of {slots}**" in readme
    assert f"{len(UNIVERSE)} companies in FinBrief's Universe" in readme
    assert f"Items {', '.join(s.item for s in tuple(Section)[:-1])}" in readme
    assert ", ".join(sorted(pointers)) in readme
    assert f"All {len(UNIVERSE)} companies have market-risk grounding" in readme
    assert f"{len(UNIVERSE) - len(pointers)} have an `Item 7A` Section" in readme


def test_the_readme_states_the_same_live_data_scope_the_app_does():
    # The *other* half of the scope, and it was unbound. T5 added a "What the live figures are,
    # and are not" section to the README that types every number `prompts.LIVE_DATA_SCOPE` and
    # `finance/ratios.py` derive — the 15-minute TTL, a peer set with its `n`, the coverage
    # sentence, the free-tier call budget. `git diff` showed this file untouched by that commit,
    # so the README became exactly the second copy `prompts.py` exists to prevent, and CLAUDE.md
    # names which copy loses: "the disagreeing copy is the one on screen" (issue #9 review).
    #
    # Bound to the *derivations*, not to a literal: change `QUOTE_TTL_SECONDS` and this fails
    # until the prose follows, which is the whole point.
    #
    # **Retargeted, not dropped** (T11 follow-up): the TTL sentence moved to
    # `docs/implementation.md`'s tool-calling section and the free-tier budget to
    # `docs/limitations.md`'s yfinance row. Each is asserted against the file it is in, so a
    # figure cannot satisfy this test from a document that does not make the claim.
    assert f"cached for {QUOTE_TTL_SECONDS // 60} minutes" in prose(IMPLEMENTATION)
    assert f"{ALPHAVANTAGE_FREE_TIER_CALLS_PER_DAY} calls a day" in prose(LIMITATIONS)
    # The app's own sentence agrees, so the two cannot drift apart in opposite directions.
    assert f"cached for {QUOTE_TTL_SECONDS // 60} minutes" in LIVE_DATA_SCOPE


def test_the_readmes_worked_peer_example_is_a_real_cluster_of_the_right_size():
    # The README works Ford's comparison through as an example — "vs. mean of 2 `autos` peers:
    # TSLA, GM" — which is `PeerComparison.basis`' sentence typed out by hand. A curation change
    # that moved Ford or renamed the cluster would leave a worked example on the front page
    # describing a comparison the tool does not make.
    readme = prose(IMPLEMENTATION)  # moved with the tool-calling section (T11 follow-up)
    peers = PEERS["F"]
    cluster = next(name for name, members in CLUSTERS.items() if "F" in members)

    assert f"mean of {len(peers)} `{cluster}` peers: {', '.join(peers)}" in readme
    assert len(peers) == 2, "the worked example is a three-member cluster, hence n = 2"


def test_the_readmes_coverage_sentence_is_the_one_the_tool_emits():
    # "1 of 2 peers reported this" is `Metric.coverage_note`'s wording, and the claim around it
    # is measured: JPM and BAC report no `debtToEquity`, so a `banks` leverage comparison rests
    # on GS alone. Both halves are bound — the count comes from the cluster, and the sentence
    # from the same f-string the card renders.
    readme = prose(IMPLEMENTATION)  # moved with the tool-calling section (T11 follow-up)
    bank_peers = PEERS["JPM"]

    assert f"1 of {len(bank_peers)} peers reported this" in readme
    assert "rests on GS alone" in readme, "the measured instance, named"


# --------------------------------------------------------------------------------------
# The security section's counts (T7, #8)
# --------------------------------------------------------------------------------------


def test_the_readmes_layer_counts_are_the_ones_the_code_has():
    """Every number in the marginal-contribution table is a count of something, so it is bound.

    The table is prose in a file that cannot import anything, which is the same problem
    `GROUNDING_SCOPE` has — and the same answer: a test where the two are allowed to disagree
    loudly. An eighth denylist rule that left the README saying seven would be a security claim
    that undercounts the mechanism.
    """
    from finbrief.security.advice import ADVICE_RULES
    from finbrief.security.corpus import BENIGN_QUESTIONS, PLANTED_PAYLOADS
    from finbrief.security.denylist import RULES

    # Moved with the prompt-injection section (T11 follow-up).
    readme = prose(IMPLEMENTATION)

    # Digits, not words, precisely so this binding is a substring check and not a translation
    # table: "five" and 5 are the same claim and only one of them can be compared to `len()`.
    assert f"{len(RULES)} rules" in readme
    assert f"{len(BENIGN_QUESTIONS)} real analyst questions" in readme
    # Wrapped prose, so the count and its noun can be a line apart.
    assert f"{len(PLANTED_PAYLOADS)} poisoned" in readme
    # Layer 4's rule count is in the generated artifact rather than the README; the binding here
    # is only that the README does not name a *different* number of rules for it.
    #
    # **Guarded so the assertion cannot disable itself.** Written as
    # `len(ADVICE_RULES) == len(RULES) or "…" not in readme`, the whole check evaporated on any
    # day the two counts happened to coincide (issue #8 review). Skipping it explicitly says so.
    if len(ADVICE_RULES) != len(RULES):
        assert f"{len(ADVICE_RULES)} rules" not in readme
    else:
        pytest.skip("the two rule counts coincide, so the substring cannot distinguish them")


def test_the_readme_names_both_latency_budgets():
    """Both, and from `config` — the revised one and the pre-registered one it did not meet.

    A README quoting only the budget now being met would turn a revised pre-registration into a
    number that had always held, which is the whole reason
    `GATE_LATENCY_BUDGET_PREREGISTERED_MS` still exists (ADR-0006 T7 amendment §2).

    **Both assertions are equalities on the millisecond figure, and the second one used not to
    be.** It read `f"{GATE_LATENCY_BUDGET_MS // 1000} s" in readme or "1s" in readme`, which is
    two defects in one line: the floor division made 1000–1999 ms indistinguishable, so the
    binding this test exists to enforce would have survived the budget moving to 1900; and the
    README did not name the revised figure at all — the assertion passed on the substring
    `"1 s"` inside "the Tier-**1 s**pec", an accident (issue #8 review). `config.py`'s claim
    that the README quotes this constant "so the prose and the verdict cannot disagree" was
    therefore false, in the one test written to keep it true. Prefer an equality over a bound
    (CLAUDE.md).
    """
    from finbrief.config import (
        GATE_LATENCY_BUDGET_MS,
        GATE_LATENCY_BUDGET_PREREGISTERED_MS,
    )

    # The gate's latency paragraph moved with its section (T11 follow-up). The README still
    # names the pre-registered figure in its limitations summary; the pair is asserted where
    # the pair is stated, since it is the *pairing* this test exists to protect.
    readme = prose(IMPLEMENTATION)

    assert f"{GATE_LATENCY_BUDGET_PREREGISTERED_MS} ms" in readme
    assert f"{GATE_LATENCY_BUDGET_MS} ms" in readme


def test_the_readme_names_the_gate_logging_cap_once_and_from_config():
    """The one user-derived field in the log, and the figure bounding it.

    Written after T8 added a *second* hardcoded `500` to the README — the switches list already
    carried one, and the "Recording a run" section repeated it. Two copies of a constant in
    prose is the shape that drifts, and a README figure is the copy a reader acts on. So the
    second mention names `config.GATE_LOGGED_INPUT_MAX_CHARS` instead of the number, and this
    test binds the remaining one (issue #10 review).
    """
    from finbrief.config import GATE_LOGGED_INPUT_MAX_CHARS

    # Moved with the logging section (T11 follow-up), and the count-of-one is asserted over
    # **all four** documents rather than one: the second copy this test forbids would be just
    # as harmful in a different file, and the split is exactly the event that could create one.
    readme = prose()

    # An equality on the figure, not a bound: `"500" in readme` would pass on any prose that
    # happened to contain the digits, which is the accident `test_the_readme_names_both_latency
    # _budgets` documents for the gate budget.
    assert f"{GATE_LOGGED_INPUT_MAX_CHARS} characters" in readme
    # And exactly once, so the copy that drifts cannot be reintroduced quietly.
    assert readme.count(f"{GATE_LOGGED_INPUT_MAX_CHARS} characters") == 1


#: The first two arguments of a `log_event` call — a logger, then the event name.
#:
#: A scan of the source rather than an import, because the names are literals at their call
#: sites and there is no registry to import: `log_event(logger, "retrieval", …)`. Which is the
#: point — the alternative to this test is a prose count nobody recomputes.
_LOG_EVENT_CALL = re.compile(r'log_event\(\s*[\w.()]+,\s*"([a-z_0-9]+)"')


def _event_names() -> set[str]:
    """Every event name the code emits, from the modules that emit one.

    `observability/logging_setup.py` is excluded and that is not a convenience: it *defines*
    `log_event` and its only match is the `chat_turn` example in the module docstring, which is
    documentation of the shape rather than a line any run writes.
    """
    root = Path(__file__).parents[1]
    sources = [
        path
        for path in [*(root / "src" / "finbrief").rglob("*.py"), *(root / "app").rglob("*.py")]
        if path.name != "logging_setup.py"
    ]
    return {
        name for path in sources for name in _LOG_EVENT_CALL.findall(path.read_text("utf-8"))
    }


def test_the_readme_states_the_number_of_event_types_the_code_emits():
    """The README counted **eleven**; the code emits 27 (code review of #12).

    A count in prose is the figure that rots first — every ticket that adds an instrument adds
    an event and none of them re-counts the sentence. So it is bound like every other derived
    figure here, and `implementation.md`'s selective table says it is selective rather than
    letting ten rows read as the whole set.
    """
    names = _event_names()

    # A sanity floor on the scan itself, so a regex that stopped matching reports as a broken
    # scan rather than as prose agreeing with zero.
    assert "retrieval" in names and "input_gate" in names, sorted(names)

    for document in (README, IMPLEMENTATION):
        assert f"**{len(names)}** event types" in prose(document), (
            f"{document.name} no longer states the {len(names)} event types `log_event` "
            f"emits. Recount from the source rather than editing this test: {sorted(names)}"
        )


EVALUATION = Path(__file__).parents[1] / "docs" / "verification" / "evaluation.md"


def _figures(text: str) -> str:
    """`text` with the spellings a number can legitimately differ by folded away.

    The README writes `−0.087` with a typographic minus and `8 of 8`; the artifact writes
    `-0.087` and `100% (8/8)`. Those are the same measurement in two registers, so the binding
    normalises rather than demanding one — which would be a style rule dressed as a check.
    """
    return text.replace("−", "-").replace("–", "-").replace(" of ", "/")


#: `(figure, the documents that must state it)` — the inventory of every copy.
#:
#: **Per document, not over the four concatenated**, and the difference is a mutation this
#: file's own author let through once: with the four joined, a stale `20/20` in
#: `implementation.md` was satisfied by the correct one in the README, so the detail file could
#: rot behind a right-looking summary. A figure that legitimately appears twice is bound twice,
#: and adding a copy means adding it here — which is the point, since an unlisted copy is
#: exactly the second copy this file exists to prevent.
EVALUATION_FIGURES = (
    ("-0.087", (README, FINDINGS)),  # H1's paired delta, the bucket hybrid exists to win
    ("3212", (README, FINDINGS, LIMITATIONS)),  # the p50 translation adds
    ("1500", (README, FINDINGS, LIMITATIONS)),  # ADR-0005's pre-registered budget
    ("4/18", (README, IMPLEMENTATION)),  # comparisons that carried a measurement
    ("8/8", (README, FINDINGS)),  # the agent-vs-original divergence rate
    # The determinism result, which is a headline claim and therefore has to be bound like every
    # other. `report.Determinism` emits these as table cells for this reason — a prose assertion
    # cannot be diffed against a measurement.
    ("168", (README, FINDINGS)),  # retrieval cells re-paid from scratch, all six arms
    ("672", (README, FINDINGS)),  # cells replayed on context-body keys, zero misses
    # Layer 4's residue never appears without its controls, so both are bound.
    ("6/6", (FINDINGS,)),  # recommendations not refused
    ("10/10", (FINDINGS,)),  # positive controls refused, which makes the 6/6 a measurement
    ("0.758", (README, IMPLEMENTATION)),  # RAGAs faithfulness, shipping default, 28 rows
    ("0.543", (README, IMPLEMENTATION)),  # context precision, same
    ("0.675", (README, IMPLEMENTATION)),  # context recall, same
    ("0.944", (IMPLEMENTATION,)),  # section recall — free, deterministic
    ("770", (README, IMPLEMENTATION)),  # cells replayed from cache on the run described
    ("0.900", (IMPLEMENTATION,)),  # leakage-free precision on the hybrid arms
    # Planner stability on the artifact's own run. The README quotes it too, in the footnote
    # separating the committed pass from the two that are only `findings.md`'s account.
    ("2/8", (README, FINDINGS)),
    ("100", (README, IMPLEMENTATION, FINDINGS)),  # tool selection, and the divergence rate
    # `1838` (the planner's p50) is deliberately absent: it was only ever quoted inside the
    # latency table, and that table is now single-copy in the artifact, so no prose quotes the
    # figure and a binding for it would assert against nothing (T11 follow-up).
)


@pytest.mark.parametrize(("figure", "documents"), EVALUATION_FIGURES)
def test_every_evaluation_figure_the_docs_quote_is_in_the_artifact(figure, documents):
    """A figure retyped into prose is a figure that will disagree with its source.

    `report.headline_section`'s own docstring says exactly that, and `test_grounding_scope.py`
    exists to bind README prose to derived values and committed evidence — yet the README's
    whole evaluation section was retyped from `evaluation.md` with nothing binding it (code
    review of #11). The next run moves these numbers and the README would keep asserting the old
    ones, in the section that states the project's headline conclusion.
    """
    artifact = _figures(EVALUATION.read_text(encoding="utf-8"))

    # **Matched as a whole number, not as a substring**, which is the difference between a
    # binding and a decoration: `"8" in text` is true of "18", "0.087" and every date, so a
    # bare-substring check on a short figure is a check that cannot fail — this repo's named
    # bug class, and it very nearly arrived inside the test written to prevent it.
    def quotes(text: str) -> bool:
        return re.search(rf"(?<![\d.\-]){re.escape(figure)}(?![\d])", text) is not None

    for document in documents:
        assert quotes(_figures(prose(document))), (
            f"{document.name} no longer quotes {figure}. If the copy moved, move it in "
            f"EVALUATION_FIGURES too; if it went, drop the entry and say why."
        )
    assert quotes(artifact), (
        f"the docs quote {figure} and the committed artifact does not. Re-run "
        f"`scripts/evaluate.py` and requote from the file it writes."
    )


#: The row of ADR-0004 §6's before/after table that describes the arm the app actually runs.
_SHIPPED_ARM_ROW = "normalisation + 3 sub-queries — **what ships**"


def test_no_document_credits_the_shipped_arm_with_the_ablations_rank():
    """`n=1`, and the one number in it that a summary is tempted to round up.

    #12's second comment is binding here — *"Don't claim a perfect ranking"* — and the README's
    summary claimed one: *"rank 5 → absent under hybrid alone → rank 1 once normalised"*. Rank 1
    is state 3 and state 5 of that table, both **ablation** arms; the arm that ships is state 4,
    at rank **2** with 5/5 entries on the right filer. `findings.md` reads the table correctly
    and said the opposite of the README — *"`vector + normalisation` put the target chunk at
    rank 1 against the default's rank 2"* — which is the drift a summary of a table introduces.

    Bound to the table rather than to a typed number: the rank is read out of
    `implementation.md`'s own row, so moving the measurement moves the expectation with it.
    """
    table = IMPLEMENTATION.read_text(encoding="utf-8")
    row = next(line for line in table.splitlines() if _SHIPPED_ARM_ROW in line)
    rank = [cell.strip() for cell in row.strip().strip("|").split("|")][4]

    assert rank == "2", (
        f"ADR-0004 §6's shipped-arm row now reports rank {rank!r}. Requote the README's "
        f"summary from it — this test holds the two together, not the value 2."
    )
    assert f"rank **{rank}** once normalised" in prose(README), (
        "the README's hybrid-search summary must name the shipped arm's rank, not the "
        "planner-off ablation's — the ablation is the one that reaches rank 1"
    )


#: The header of `findings.md`'s table of checks that could not fail.
_COULD_NOT_FAIL_HEADER = "| where | it asserted | what it had established |"

#: A row recording more than one occurrence of its defect, as tiktoken's `(twice)` does.
_ROW_MULTIPLICITY = re.compile(r"\((twice|three times)\)")

#: Spelled out, because the prose spells them out and a binding may not quietly accept digits.
_NUMBER_WORDS = {16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen"}


def _could_not_fail_rows() -> list[str]:
    lines = FINDINGS.read_text(encoding="utf-8").splitlines()
    start = lines.index(_COULD_NOT_FAIL_HEADER) + 2  # past the header and its separator
    end = next(n for n, line in enumerate(lines[start:], start) if not line.startswith("|"))
    return lines[start:end]


def test_the_table_of_checks_that_could_not_fail_counts_itself():
    """The count in the prose is the count in the table, rows *and* multiplicities.

    Three sentences said **seventeen** over a seventeen-row table whose first row reads
    "tiktoken's warm cache **(twice)**" — eighteen instances (code review of #12). The count is
    the section's whole argument, since the claim is that the instances look unrelated until
    they are listed, and a table that miscounts itself is that section's own bug class.

    Derived from the table rather than asserted against a constant: a nineteenth instance
    arrives as a row, and this is what makes the three sentences follow it.
    """
    rows = _could_not_fail_rows()
    instances = sum(2 if _ROW_MULTIPLICITY.search(row) else 1 for row in rows)

    assert (len(rows), instances) == (17, 18), (
        f"the table now holds {len(rows)} row(s) and {instances} instance(s). Update the "
        f"sentence above it and the README's two references to it, then update this equality "
        f"— it is here so a new row cannot leave three stale counts behind."
    )
    assert (
        f"**{_NUMBER_WORDS[instances]}** times in {_NUMBER_WORDS[len(rows)]} places"
        in prose(FINDINGS)
    ), "findings.md must state both counts, since they differ and the difference is the point"
    assert prose(README).count(f"{_NUMBER_WORDS[instances]}-instance table") == 2, (
        "the README names the table twice — in the document map and in Part 5 — and both "
        "namings carry the instance count"
    )


def test_the_translation_budget_is_pinned_by_an_equality_and_not_by_a_bound():
    """Its twin `GATE_LATENCY_BUDGET_MS` is bound by equalities in two places, which is the
    reason `config.py` cites for the move. This one's only coverage was `within_budget is True`
    at 1300 and `False` at 2200 — satisfied by any budget in [1300, 2200) (review of #11)."""
    from finbrief.config import TRANSLATION_LATENCY_BUDGET_MS

    assert TRANSLATION_LATENCY_BUDGET_MS == 1500.0
    assert f"{TRANSLATION_LATENCY_BUDGET_MS:.0f} ms" in README.read_text(encoding="utf-8")


#: The cited-marker deferral's three-way split, which README and artifact both lead with.
#:
#: Bound as one phrase rather than as three numbers: `22`, `40` and `8` are each too short to
#: match meaningfully on their own, and the *composition* is the finding anyway — the middle
#: bucket is the largest and is what neither the citation register nor whole-answer faithfulness
#: can see. A rate quoted without it reads as a simple failure rate, which it is not.
CITED_MARKER_COMPOSITION = "22 fully supported / 40 partly supported / 8 not supported"

#: The same composition in the register a table cell and a one-line summary use.
#:
#: Bound as well as the long form, and per document, because two spellings of one measurement
#: are two things that can drift apart — and the short one is what the README's limitations row
#: and `limitations.md` both carry, which is the copy a reviewer reads last and remembers.
CITED_MARKER_COMPOSITION_SHORT = "22 fully / 40 partly / 8 not supported"

#: `(spelling, the documents that must carry it)`, on `EVALUATION_FIGURES`' rule.
CITED_MARKER_SPELLINGS = (
    (CITED_MARKER_COMPOSITION, (README, IMPLEMENTATION)),
    (CITED_MARKER_COMPOSITION_SHORT, (README, LIMITATIONS)),
)


@pytest.mark.parametrize(("spelling", "documents"), CITED_MARKER_SPELLINGS)
def test_the_cited_marker_composition_is_quoted_the_same_way_everywhere(spelling, documents):
    """Every document that states the split states it as the artifact rendered it.

    **Per document, not over the five concatenated**, and the difference is the mutation this
    test used to allow: it asserted against `prose()`, so the README could drop the sentence
    entirely and `implementation.md`'s copy would satisfy the assertion written about the
    README — a check that cannot fail for the reason its own docstring named (CLAUDE.md).
    """
    artifact = EVALUATION.read_text(encoding="utf-8")

    assert CITED_MARKER_COMPOSITION in artifact, (
        f"the artifact no longer renders {CITED_MARKER_COMPOSITION!r}. If the run moved those "
        f"counts, requote the docs from it and update these constants."
    )
    for document in documents:
        assert spelling in prose(document), (
            f"{document.name} must lead with the composition rather than the derived rate: a "
            f"31% full-support rate hides that partial support is the largest bucket."
        )


# --------------------------------------------------------------------------------------
# T11 (#12): the README is the submission, so every table in it is bound to its source
# --------------------------------------------------------------------------------------
#
# The README grew from a status file into the review-facing document, which multiplies the one
# failure this file exists to prevent: a figure retyped into prose disagrees with its source,
# and the prose is the copy a reviewer acts on. Three kinds of figure arrive with T11 and each
# gets a binding of its own kind — the parameters and bounds tables are *derived* and bind to
# `config`; the Universe table and the suite counts are *measurements* and bind to the artifact
# that produced them; the per-bucket cells bind as whole cells, because listing forty short
# numbers individually is how a binding becomes a decoration.


def readme_flat() -> str:
    return flat(README.read_text(encoding="utf-8"))


#: `(row label, the value the README must state)` for every parameter it quotes from `config`.
#:
#: Bound as `label | value` and not as the bare value, which would be vacuous for `5` and `10`
#: (CLAUDE.md: prefer a check that exercises the thing). Assembled at call time so a constant
#: change fails here rather than at import.
def _derived_parameter_rows() -> tuple[tuple[str, str], ...]:
    from finbrief.agent.agent import MAX_AGENT_STEPS
    from finbrief.config import (
        ANSWER_MAX_RETRIES,
        ANSWER_TIMEOUT_SECONDS,
        CHUNK_SIZE_CHARS,
        DEFAULT_MAX_SUB_QUERIES,
        FETCH_ATTEMPTS,
        FETCH_BACKOFF_SECONDS,
        FETCH_TIMEOUT_SECONDS,
        GATE_CLASSIFIER_ATTEMPTS,
        GATE_TIMEOUT_SECONDS,
        MAX_QUESTION_CHARS,
        MAX_QUESTIONS_PER_SESSION,
        NEWS_DEFAULT_DAYS,
        NEWS_MAX_DAYS,
        NEWS_MAX_HEADLINES,
        NEWS_TTL_SECONDS,
        QUOTE_TTL_SECONDS,
        RRF_K,
        TICKER_MAX_CHARS,
        Settings,
    )
    from finbrief.ingestion.chunking import CHUNK_OVERLAP_CHARS

    k = Settings.from_env({"OPENROUTER_API_KEY": "sk-test"}).retrieval_k
    # One original + at most one ticker form + at most `max_sub_queries` (ADR-0004 amendment),
    # and twice that many candidate lists under hybrid. Derived here rather than typed, because
    # the variant budget is what ADR-0005's latency budget is judged against.
    variants = 1 + 1 + DEFAULT_MAX_SUB_QUERIES
    assert QUOTE_TTL_SECONDS == NEWS_TTL_SECONDS, "the README states one TTL for both caches"

    return (
        ("chunk size", f"{CHUNK_SIZE_CHARS} characters"),
        ("chunk overlap", f"{CHUNK_OVERLAP_CHARS} characters"),
        ("top-k after fusion", f"{k}"),
        ("candidate-list depth", f"{k}"),
        (
            "query variants",
            f"1 original + ≤1 ticker form + ≤{DEFAULT_MAX_SUB_QUERIES} "
            f"sub-queries = {variants}",
        ),
        ("candidate lists under hybrid", f"{variants * 2}"),
        ("RRF constant", f"{RRF_K}"),
        ("agent steps", f"{MAX_AGENT_STEPS}"),
        ("question length", f"{MAX_QUESTION_CHARS} characters"),
        ("questions per session", f"{MAX_QUESTIONS_PER_SESSION}"),
        ("ticker length", f"{TICKER_MAX_CHARS} characters"),
        ("news window", f"{NEWS_DEFAULT_DAYS} days by default, clamped to {NEWS_MAX_DAYS}"),
        ("headlines per card", f"{NEWS_MAX_HEADLINES}"),
        ("quote / news cache TTL", f"{QUOTE_TTL_SECONDS} seconds"),
        (
            "one HTTP request",
            f"{FETCH_TIMEOUT_SECONDS} seconds, {FETCH_ATTEMPTS} attempts, "
            f"{FETCH_BACKOFF_SECONDS} s backoff",
        ),
        (
            "gate classifier",
            f"{GATE_TIMEOUT_SECONDS} seconds, {GATE_CLASSIFIER_ATTEMPTS} attempt",
        ),
        ("answering call", f"{ANSWER_TIMEOUT_SECONDS} seconds, {ANSWER_MAX_RETRIES} retries"),
    )


def test_every_parameter_the_readme_tabulates_is_the_one_config_holds():
    """The retrieval parameters and every bound, as `label | value` rows bound to `config`.

    T11's README tabulates the knobs a reviewer would otherwise have to read the source for —
    chunk size, `k`, the variant budget, `RRF_K`, and the ten limits in *Technical
    implementation*. Each already has a single source of truth, so the README is a second copy
    by construction and this is where the two are allowed to disagree.
    """
    readme = prose(IMPLEMENTATION)  # the two tables moved (T11 follow-up)
    missing = [
        f"{label} | {value}"
        for label, value in _derived_parameter_rows()
        if f"{label} | {value}" not in readme
    ]

    assert not missing, (
        f"the README's tables no longer state these config-derived values: {missing}. "
        f"Requote them from `config.py` rather than editing this list."
    )


def test_the_readme_states_the_quote_worst_case_the_demo_has_to_warm_around():
    """`91.5 s`, and it is a walkthrough instruction rather than trivia (#12's third comment).

    Demo step 3 resolves six quotes through one cache whose lock is held across a retry
    sequence, so a cold take is the slowest possible take. The figure is `config`'s, so the
    prose cannot drift from the constant the risk register cites.
    """
    from finbrief.config import QUOTE_FETCH_WORST_CASE_SECONDS

    # The walkthrough moved to `docs/demo.md` (T11 follow-up); the figure is asserted where the
    # instruction is, and the README's summary of the walkthrough names it too.
    walkthrough = prose(DEMO)

    assert f"{QUOTE_FETCH_WORST_CASE_SECONDS} s" in walkthrough
    assert "warm the quote cache" in walkthrough.lower(), "and what to do about it"
    assert f"{QUOTE_FETCH_WORST_CASE_SECONDS} s" in prose(README), (
        "the README's walkthrough summary carries the number, since it is why the step exists"
    )


def test_the_readme_quotes_the_grounding_scope_sentence_the_app_renders():
    """T11's AC-2 is a UI obligation and this README documents it, so it quotes the constant.

    Verbatim, not paraphrased: `GROUNDING_SCOPE` is the sentence under the app's title *and* the
    opening of four prompts, and a README paraphrase of it is exactly the second copy
    `prompts.py` exists to prevent. The counts inside it are already bound by
    `test_the_readme_states_the_same_scope_the_app_does`; this binds the wording.
    """
    # **Both documents**, and that is deliberate: the README's summary and
    # `implementation.md`'s full section each quote the app's sentence, so each is bound to it.
    # A summary that paraphrased the disclosure would be the second wording this constant
    # exists to prevent — the failure is the same whether the copy is long or short.
    for document in (README, IMPLEMENTATION):
        assert flat(GROUNDING_SCOPE) in prose(document), (
            f"{document.name} must quote `prompts.GROUNDING_SCOPE` verbatim, not restate it"
        )
    # The README's summary also carries the arithmetic, which is the number a reviewer checks.
    slots = len(UNIVERSE) * len(Section)
    rows = gate_rows(report_text())
    pointers = sum(1 for row in rows.values() if _POINTER_MARKER in row)
    assert f"**{slots - pointers} of {slots}**" in prose(README)
    # And the constants are named, so a reader can find the one place the sentence lives.
    assert "GROUNDING_SCOPE" in prose(IMPLEMENTATION)
    assert "GROUNDING_SCOPE_DETAILS" in prose(IMPLEMENTATION)


def _universe_table_rows(text: str) -> dict[str, tuple[str, str]]:
    """`{ticker: (accession, chunks)}` from the ingest report's own chunk table.

    A different table from `_GATE_ROW`'s: that one is the gate's character counts, this one is
    what the run read back out of the persisted collection.
    """
    rows = re.findall(
        r"^\|\s*([A-Z]+)\s*\|\s*FY\d{4}\s*\|\s*`([\d-]+)`\s*\|\s*([\d,]+)\s*\|",
        text,
        re.MULTILINE,
    )
    assert len(rows) == len(UNIVERSE), "the ingest report's chunk table shape changed"
    return {ticker: (accession, chunks) for ticker, accession, chunks in rows}


def test_the_readmes_universe_table_is_the_ingest_runs_own_numbers():
    """Fifteen rows of ticker, company, cluster, fiscal year, accession and chunk count.

    Every cell of it is either `config`'s or the ingest run's, and none of it is derivable by
    eye — an accession is nineteen digits and a chunk count is a measurement. This is the
    largest block of retyped figures T11 adds, so it is bound cell by cell, not by a total.
    """
    readme = IMPLEMENTATION.read_text(encoding="utf-8")  # moved (T11 follow-up)
    evidence = _universe_table_rows(report_text())

    gate = report_text()
    years = {
        ticker: re.search(rf"^{ticker}\s+FY(\d{{4}})", gate, re.MULTILINE).group(1)
        for ticker in evidence
    }
    expected = [
        f"| {c.ticker} | {c.name} | {c.cluster.value} | FY{years[c.ticker]} | "
        f"`{evidence[c.ticker][0]}` | {evidence[c.ticker][1]} |"
        for c in UNIVERSE
    ]
    missing = [row for row in expected if flat(row) not in flat(readme)]

    assert not missing, (
        f"the README's Universe table disagrees with docs/verification/ingest-report.md on "
        f"{len(missing)} row(s): {missing[:2]}. Requote from the artifact."
    )


def test_the_readmes_chunk_total_is_the_sum_the_artifact_reports():
    """5,842 — asserted as the sum of the artifact's own rows, not as a string it also contains.

    Both the README and the report state the total, so a substring check would pass on two
    copies of the same stale number. Summing the rows is the check that the total is a total.
    """
    evidence = _universe_table_rows(report_text())
    total = sum(int(chunks.replace(",", "")) for _, chunks in evidence.values())

    assert f"{total:,}" in README.read_text(encoding="utf-8"), (
        f"the README no longer states the collection's {total:,} chunks"
    )


SECURITY = Path(__file__).parents[1] / "docs" / "verification" / "security-gate.md"


#: `(figure, the documents that must state it)`, on the rule `EVALUATION_FIGURES` explains.
SECURITY_FIGURES = (
    ("20/20", (README, IMPLEMENTATION)),  # attacks stopped by the *expected* layer
    ("28/28", (README, IMPLEMENTATION)),  # benign analyst questions allowed
    ("22/22", (README, IMPLEMENTATION)),  # answer verdicts correct, both directions
    ("5/5", (README, IMPLEMENTATION)),  # planted payloads retrieved *and* resisted
    ("564 ms", (IMPLEMENTATION,)),  # p50 over every screening
    ("670 ms", (README, IMPLEMENTATION)),  # p50 over the escalated screenings
)


@pytest.mark.parametrize(("figure", "documents"), SECURITY_FIGURES)
def test_every_security_figure_the_docs_quote_is_in_the_suites_artifact(figure, documents):
    """The gate's counts are a live run's, so the README quotes them and does not compute them.

    `security-gate.md` is regenerated by every suite run and the counts move with the corpus —
    widening the benign set changes the denominator *and* the latency median (ADR-0006 T7
    amendment §2). A README carrying last month's counts would be describing a gate that is no
    longer the one in the repo.
    """
    artifact = _figures(SECURITY.read_text(encoding="utf-8"))

    def quotes(text: str) -> bool:
        return re.search(rf"(?<![\d.\-]){re.escape(figure)}(?![\d])", text) is not None

    for document in documents:
        assert quotes(_figures(prose(document))), (
            f"{document.name} no longer quotes {figure}; update SECURITY_FIGURES too"
        )
    assert quotes(artifact), (
        f"the docs quote {figure} and docs/verification/security-gate.md does not. Re-run "
        f"`scripts/security_suite.py` and requote from the file it writes."
    )


#: `0.629 [0.200–1.000] n=7` — the shape every per-bucket mean is printed in.
#:
#: Matched as a whole cell precisely because its parts are short: `0.629` alone would be a
#: substring check of the kind amendment four of ADR-0011 records, and forty of them listed by
#: hand would be a list nobody maintains. A cell carries its mean, its spread and its `n`, which
#: is specific enough that a moved number cannot match by accident — and ADR-0005 requires the
#: spread beside the mean anyway, so binding the cell binds that obligation too.
_BUCKET_CELL = re.compile(r"\d\.\d{3} \[\d\.\d{3}[–-]\d\.\d{3}\] n=\d+")


#: The shipping-default arm, spelled as the artifact's own row label.
_SHIPPING_DEFAULT_ARM = "hybrid + translation (shipping default)"

#: `(the artifact heading the table sits under, how many of its metric columns the docs
#: reproduce)` — the two per-bucket tables AC-1 of #12 names by hand.
#:
#: The A/B table carries five metric columns and the docs reproduce the first three; the two
#: chunk-level columns stay artifact-only, and the prose beside the table says so. The RAGAs
#: table is reproduced whole.
PER_BUCKET_TABLES = (
    ("## Per-bucket A/B — the deterministic retrieval metrics", 3),
    ("## RAGAs — all four metrics, per bucket", 4),
)


def _artifact_bucket_rows(heading: str, columns: int) -> dict[str, tuple[str, ...]]:
    """`{bucket: the shipping-default arm's first `columns` cells}`, from one artifact table.

    Read out of `evaluation.md` rather than listed here, so the expectation is the measurement:
    a list of cells typed into this file would be a third copy, and the copy a test trusts.
    """
    artifact = EVALUATION.read_text(encoding="utf-8")
    start = artifact.index(heading)
    end = artifact.find("\n## ", start + len(heading))
    section = artifact[start : end if end != -1 else len(artifact)]

    rows = {}
    for line in section.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) > columns + 1 and cells[1] == _SHIPPING_DEFAULT_ARM:
            rows[cells[0]] = tuple(cells[2 : 2 + columns])

    assert len(rows) == 4, (
        f"expected the four stratified buckets under {heading!r} on the "
        f"{_SHIPPING_DEFAULT_ARM!r} arm; found {sorted(rows)}. If the artifact's table shape "
        f"changed, this reader has to change with it."
    )
    return rows


@pytest.mark.parametrize(("heading", "columns"), PER_BUCKET_TABLES)
def test_the_per_bucket_tables_the_docs_reproduce_are_the_artifacts_own(heading, columns):
    """Every `mean [min–max] n=` cell the docs print is the cell `evaluation.md` rendered.

    #12's AC-1 names these two tables — *"README includes architecture, per-bucket RAGAs/A-B
    tables, …"* — so they are reproduced rather than linked, and the staleness that argues for
    linking is answered here instead: each row is rebuilt from the artifact's own
    shipping-default row and asserted whole. A re-run that moves one cell fails this.

    **Bound as a whole row, not as a bag of cells** (T11 follow-up). An unordered membership
    check passes when two columns are transposed, which is the same wrong number in a
    right-looking table; asserting the row asserts the column order too.
    """
    reproduced = prose(IMPLEMENTATION)
    missing = [
        f"| {bucket} | {' | '.join(cells)} |"
        for bucket, cells in _artifact_bucket_rows(heading, columns).items()
        if flat(f"| {bucket} | {' | '.join(cells)} |") not in reproduced
    ]

    assert not missing, (
        f"implementation.md's copy of {heading!r} disagrees with "
        f"docs/verification/evaluation.md on {len(missing)} row(s): {missing[:1]}. Those "
        f"tables are rendered by `scripts/evaluate.py`; requote from the file it writes."
    )


def test_no_other_document_keeps_a_second_copy_of_a_per_bucket_cell():
    """The tables are reproduced **once**, in the document bound to the artifact above.

    The inverse half, and the one that survives the README's split into linked documents: a cell
    pasted into the README or `findings.md` is a copy nothing binds, and it is the copy a
    reviewer reads first. `implementation.md` is excluded here precisely because it is the file
    `test_the_per_bucket_tables_the_docs_reproduce_are_the_artifacts_own` covers.
    """
    found = {
        f"{path.name}: {cell}"
        for path in DOCUMENTS
        if path != IMPLEMENTATION
        for cell in _BUCKET_CELL.findall(path.read_text(encoding="utf-8"))
    }

    assert not found, (
        f"{len(found)} per-bucket cell(s) are quoted outside implementation.md: "
        f"{sorted(found)[:3]}. Link to docs/verification/evaluation.md rather than keeping a "
        f"copy that a re-run will silently outdate."
    )


def test_the_readme_labels_the_dollar_figure_as_an_estimate():
    """The one cost figure in the repo is a planning estimate, and the README may not launder
    it.

    `config.py` and `evaluation/cache.py` size a full judged run at ~$1.28 from #11's plan. No
    run wrote it, no artifact carries it, and nothing binds it — so the README states it beside
    the words that say so. This is the mirror of
    `test_the_readme_does_not_quote_a_price_as_though_it_were_measured`: that one forbids a rate
    card, this one forbids an unlabelled bill.
    """
    readme = readme_flat()

    assert "$1.28" in readme, "the estimate is informative and is kept"
    assert "**not a measurement**" in readme, "and labelled, in the same breath"
    assert "No dollar figure in this project comes from a committed artifact" in readme


def test_the_readme_states_the_session_cap_and_that_it_is_not_a_security_control():
    """T12 item 6. The number is bound; the disclaimer is bound; both for the same reason.

    ADR-0001's amendment records this reversal and puts the danger plainly: a reviewer who
    reads a session counter as rate limiting stops looking for the thing that is. So the README
    may not quote the cap without the sentence that a refresh resets it, and it may not quote a
    *stale* cap either — this is the file that binds prose to `config`.
    """
    from finbrief.config import MAX_QUESTIONS_PER_SESSION

    readme = prose(IMPLEMENTATION)  # moved with the rate-limiting section (T11 follow-up)

    # A whole number, not a substring: `40` inside `140` is the vacuous match this file's own
    # amendment-four finding is about.
    assert re.search(rf"(?<![\d.\-]){MAX_QUESTIONS_PER_SESSION}(?![\d])", readme), (
        f"the README no longer states the {MAX_QUESTIONS_PER_SESSION}-question session cap"
    )
    assert "Refreshing the page resets it" in readme
    assert "not a security control" in readme


def test_the_readme_does_not_quote_a_price_as_though_it_were_measured():
    """The absence that has to stay an absence, asserted from the other direction.

    Both cost knobs default to unset and the README's argument is that no rate card belongs in
    this repo — so the prose must not carry a dollar-per-million figure presented as FinBrief's
    cost. `.env.example` may show a *sample* value beside a commented switch; the README may not
    state one as fact, because that is the "figure nobody measured" the design rejects.
    """
    from finbrief.config import Settings

    settings = Settings.from_env({"OPENROUTER_API_KEY": "sk-test"})
    assert settings.input_cost_per_mtok is None, "unpriced is the default"
    assert settings.output_cost_per_mtok is None

    # The knobs and the no-rate-card argument moved with the spend section; the README's cost
    # section states the same absence in its own words and is bound by
    # `test_the_readme_labels_the_dollar_figure_as_an_estimate` (T11 follow-up).
    readme = prose(IMPLEMENTATION)
    assert "FINBRIEF_INPUT_COST_PER_MTOK" in readme, "the knob is documented"
    assert "no rate card" in readme.lower(), "and so is the reason there is no default"


def test_no_test_imports_through_the_tests_package():
    """`from fakes import …`, never `from tests.fakes import …` — and the difference is CI.

    There is no `tests/__init__.py`, so pytest puts *this directory* on `sys.path` and `fakes`
    resolves anywhere. The `tests.` prefix additionally needs the **repo root** on the path,
    which a local editable install happens to supply and a clean runner does not: one such
    import sat in `test_eval_pipeline.py`, passed on the author's machine, and failed CI with
    `ModuleNotFoundError: No module named 'tests'`.

    Forbidden by a scan rather than fixed once, because "green locally" and "green in CI" are
    different claims and this is the difference that made them differ. A convention followed by
    nine of ten importers is not a convention — it is a coin flip that has come up heads nine
    times.
    """
    offenders = sorted(
        f"{path.name}:{number}"
        for path in Path(__file__).resolve().parent.glob("*.py")
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if re.match(r"^\s*(?:from|import)\s+tests\.", line)
    )

    assert not offenders, (
        f"{offenders} import through the `tests.` package. Spell it `from fakes import …`: "
        f"the prefixed form needs the repo root on sys.path, which CI does not provide."
    )
