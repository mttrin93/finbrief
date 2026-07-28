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
    LIVE_DATA_SCOPE,
    SEARCH_FILINGS_DESCRIPTION,
    SYSTEM_PROMPT,
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
    rows = gate_rows(report_text())
    found = {ticker for ticker, row in rows.items() if _POINTER_MARKER in row}
    panel = " ".join(GROUNDING_SCOPE_DETAILS)

    named = {ticker for ticker in rows if f"{ticker}," in panel or f"{ticker} answer" in panel}
    assert named == found, "the disclosure names exactly the filers the run found"
    assert f"{len(UNIVERSE) - len(found)} have an `Item 7A` Section." in panel


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
    assert len(ADVICE_RULES) == len(RULES) or f"{len(ADVICE_RULES)} rules" not in readme


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
