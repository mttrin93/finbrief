"""The grounding-scope disclosure's arithmetic, against the committed ingest evidence.

`prompts.py` derives "54 of 60" from `config` so that a Universe change cannot leave a stale
number under the app's title. That keeps the sentence *self*-consistent, which is not the
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

REPORT = Path(__file__).parents[1] / "docs" / "verification" / "ingest-report.md"
README = Path(__file__).parents[1] / "README.md"

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
    assert f"{slots - pointers} of {slots} company × Section pairs" in GROUNDING_SCOPE


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


def test_the_ingest_report_provenance_moved_to_the_readme():
    """Where the counts come from is a reviewer's question, and the README is where a reviewer
    reads. It left the panel when that panel was compressed (#13) and had to land somewhere —
    a claim dropped from one surface and added to none is the deletion a compression must
    not be."""
    readme = " ".join(README.read_text(encoding="utf-8").split())
    panel = " ".join(GROUNDING_SCOPE_DETAILS)

    assert "docs/verification/ingest-report.md" in readme
    assert "not a live count of the index" in readme, "and what the counts are *not*"
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
    readme = " ".join(README.read_text(encoding="utf-8").split())
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
    readme = " ".join(README.read_text(encoding="utf-8").split())

    assert f"cached for {QUOTE_TTL_SECONDS // 60} minutes" in readme
    assert f"{ALPHAVANTAGE_FREE_TIER_CALLS_PER_DAY} calls a day" in readme
    # The app's own sentence agrees, so the two cannot drift apart in opposite directions.
    assert f"cached for {QUOTE_TTL_SECONDS // 60} minutes" in LIVE_DATA_SCOPE


def test_the_readmes_worked_peer_example_is_a_real_cluster_of_the_right_size():
    # The README works Ford's comparison through as an example — "vs. mean of 2 `autos` peers:
    # TSLA, GM" — which is `PeerComparison.basis`' sentence typed out by hand. A curation change
    # that moved Ford or renamed the cluster would leave a worked example on the front page
    # describing a comparison the tool does not make.
    readme = " ".join(README.read_text(encoding="utf-8").split())
    peers = PEERS["F"]
    cluster = next(name for name, members in CLUSTERS.items() if "F" in members)

    assert f"mean of {len(peers)} `{cluster}` peers: {', '.join(peers)}" in readme
    assert len(peers) == 2, "the worked example is a three-member cluster, hence n = 2"


def test_the_readmes_coverage_sentence_is_the_one_the_tool_emits():
    # "1 of 2 peers reported this" is `Metric.coverage_note`'s wording, and the claim around it
    # is measured: JPM and BAC report no `debtToEquity`, so a `banks` leverage comparison rests
    # on GS alone. Both halves are bound — the count comes from the cluster, and the sentence
    # from the same f-string the card renders.
    readme = " ".join(README.read_text(encoding="utf-8").split())
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

    readme = README.read_text(encoding="utf-8")

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

    readme = README.read_text(encoding="utf-8")

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

    readme = README.read_text(encoding="utf-8")

    # An equality on the figure, not a bound: `"500" in readme` would pass on any prose that
    # happened to contain the digits, which is the accident `test_the_readme_names_both_latency
    # _budgets` documents for the gate budget.
    assert f"{GATE_LOGGED_INPUT_MAX_CHARS} characters" in readme
    # And exactly once, so the copy that drifts cannot be reintroduced quietly.
    assert readme.count(f"{GATE_LOGGED_INPUT_MAX_CHARS} characters") == 1


EVALUATION = Path(__file__).parents[1] / "docs" / "verification" / "evaluation.md"


def _figures(text: str) -> str:
    """`text` with the spellings a number can legitimately differ by folded away.

    The README writes `−0.087` with a typographic minus and `8 of 8`; the artifact writes
    `-0.087` and `100% (8/8)`. Those are the same measurement in two registers, so the binding
    normalises rather than demanding one — which would be a style rule dressed as a check.
    """
    return text.replace("−", "-").replace("–", "-").replace(" of ", "/")


@pytest.mark.parametrize(
    "figure",
    [
        "-0.087",  # H1's paired delta, the bucket hybrid exists to win
        "3212",  # the p50 translation adds
        "1500",  # ADR-0005's pre-registered budget
        "4/18",  # comparisons that carried a measurement
        "8/8",  # the agent-vs-original divergence rate
        # The determinism result, which is a headline claim in the README and therefore has to
        # be bound like every other. `report.Determinism` emits these as table cells for this
        # reason — a prose assertion cannot be diffed against a measurement, and the artifact
        # not carrying them parseably was the cheaper half of the problem to fix.
        "168",  # retrieval cells re-paid from scratch, all six arms
        "672",  # cells replayed on context-body keys, zero misses
        # Layer 4's residue never appears without its controls, so both are bound.
        "6/6",  # recommendations not refused
        "10/10",  # positive controls refused, which is what makes the 6/6 a measurement
    ],
)
def test_every_evaluation_figure_the_readme_quotes_is_in_the_artifact(figure):
    """A figure retyped into prose is a figure that will disagree with its source.

    `report.headline_section`'s own docstring says exactly that, and `test_grounding_scope.py`
    exists to bind README prose to derived values and committed evidence — yet the README's
    whole evaluation section was retyped from `evaluation.md` with nothing binding it (code
    review of #11). The next run moves these numbers and the README would keep asserting the old
    ones, in the section that states the project's headline conclusion.
    """
    readme = _figures(README.read_text(encoding="utf-8"))
    artifact = _figures(EVALUATION.read_text(encoding="utf-8"))

    # **Matched as a whole number, not as a substring**, which is the difference between a
    # binding and a decoration: `"8" in text` is true of "18", "0.087" and every date, so a
    # bare-substring check on a short figure is a check that cannot fail — this repo's named
    # bug class, and it very nearly arrived inside the test written to prevent it.
    def quotes(text: str) -> bool:
        return re.search(rf"(?<![\d.\-]){re.escape(figure)}(?![\d])", text) is not None

    assert quotes(readme), f"the README no longer quotes {figure}; update this list too"
    assert quotes(artifact), (
        f"the README quotes {figure} and the committed artifact does not. Re-run "
        f"`scripts/evaluate.py` and requote from the file it writes."
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


def test_the_cited_marker_composition_is_quoted_the_same_way_in_both():
    """The README leads with the split, and the artifact is where it comes from."""
    readme = README.read_text(encoding="utf-8")
    artifact = EVALUATION.read_text(encoding="utf-8")

    assert CITED_MARKER_COMPOSITION in artifact, (
        f"the artifact no longer renders {CITED_MARKER_COMPOSITION!r}. If the run moved those "
        f"counts, requote the README from it and update this constant."
    )
    assert CITED_MARKER_COMPOSITION in readme, (
        "the README must lead with the composition rather than the derived rate: a 31% "
        "full-support rate hides that partial support is the largest bucket."
    )


def test_the_readme_states_the_session_cap_and_that_it_is_not_a_security_control():
    """T12 item 6. The number is bound; the disclaimer is bound; both for the same reason.

    ADR-0001's amendment records this reversal and puts the danger plainly: a reviewer who
    reads a session counter as rate limiting stops looking for the thing that is. So the README
    may not quote the cap without the sentence that a refresh resets it, and it may not quote a
    *stale* cap either — this is the file that binds prose to `config`.
    """
    from finbrief.config import MAX_QUESTIONS_PER_SESSION

    readme = " ".join(README.read_text(encoding="utf-8").split())

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

    readme = " ".join(README.read_text(encoding="utf-8").split())
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
