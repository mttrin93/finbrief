"""The citation-marker validator: does every `[n]` name a chunk this turn retrieved?

T3's second deferred finding (issue #5, recorded on #8), taken in T7 as the deterministic half
of the output side. What is asserted here is *resolution* and nothing else — whether the cited
chunk supports the claim is faithfulness, measured by T9's RAGAs and out of scope by
construction (`security/markers.py`).
"""

from __future__ import annotations

import json

from finbrief.observability.logging_setup import configure_logging
from finbrief.security.markers import log_markers, markers

RANKS = (1, 2, 3)


def test_an_answer_whose_markers_all_resolve_is_clean() -> None:
    report = markers("Tesla identifies supply chain risk [1] and competition [2].", ranks=RANKS)

    assert report.clean
    assert report.resolved == 2
    assert report.unresolved == ()


def test_a_compound_marker_is_two_markers() -> None:
    """`[1][3]` is the form `_ANSWER_RULES` asks for when two sources support one sentence."""
    report = markers("Both filings say so [1][3].", ranks=RANKS)

    assert report.resolved == 2
    assert report.clean


def test_a_number_naming_no_retrieved_chunk_is_unresolved() -> None:
    """The T3 finding itself: a model that cites `[6]` against three contexts.

    Silent before this — no exception, no warning, no log line — so a reader could not tell a
    hallucinated citation from a numbering slip.
    """
    report = markers("Margins improved [6].", ranks=RANKS)

    assert report.unresolved == (6,)
    assert report.clean is False


def test_a_non_numeric_bracket_is_reported_apart_from_an_unresolved_number() -> None:
    """The T5 finding: the model writes `[Yahoo Finance]` despite the prompt reserving brackets.

    Counted separately because it is a different defect. An unresolved number is a citation
    pointing nowhere; a non-numeric bracket is a *collision* with the syntax that makes any
    citation resolvable, and a reader seeing both cannot resolve either.
    """
    report = markers(
        "The price is $412 [Yahoo Finance], and the risk is stated [1].", ranks=RANKS
    )

    assert report.non_numeric == ("Yahoo Finance",)
    assert report.unresolved == ()
    assert report.resolved == 1
    assert report.clean is False


def test_each_offender_is_reported_once_however_often_it_appears() -> None:
    """A caption naming `[6]` three times is noise; the datum is which numbers, not how many."""
    report = markers("[6] and again [6] and [Reuters] and [Reuters].", ranks=RANKS)

    assert report.unresolved == (6,)
    assert report.non_numeric == ("Reuters",)


def test_unresolved_numbers_are_sorted() -> None:
    """So a log line and a caption are stable, and two reports of one answer compare equal."""
    assert markers("[9] then [4] then [7]", ranks=RANKS).unresolved == (4, 7, 9)


def test_a_grounded_answer_with_no_markers_at_all_is_clean_but_visibly_uncited() -> None:
    """Clean is about brackets that *are* there. `resolved == 0` is the fact T10 reports on.

    This is the shape T5's live run produced when the brief's headings were read as search
    terms: a paragraph about a company with no `[n]` anywhere. Nothing here can call it a
    violation — an answer may legitimately have no factual claim to cite — so it is reported as
    a count rather than judged.
    """
    report = markers("I could not find anything to ground that.", ranks=RANKS)

    assert report.clean
    assert report.resolved == 0


def test_a_stray_bracket_in_prose_cannot_swallow_the_rest_of_the_answer() -> None:
    """Bounded to one line and 40 characters, so an unclosed `[` matches nothing."""
    report = markers("The filing says [ and then a great deal more text follows.", ranks=RANKS)

    assert report.non_numeric == ()
    assert report.resolved == 0


def test_no_ranks_means_every_number_is_unresolved() -> None:
    """A turn that retrieved nothing and cited anyway — which is the answer-from-memory case."""
    assert markers("Revenue rose [1].", ranks=()).unresolved == (1,)


def test_the_log_line_carries_counts_and_numbers_but_no_prose() -> None:
    """`non_numeric` is a count, not the spans: the pattern admits 40 characters of anything.

    The spans reach the *reader*, on screen, where they are useful. These lines are kept, so
    what goes in them is this module's own vocabulary and the answer's own citation numbers.
    """
    import io

    stream = io.StringIO()
    configure_logging(stream=stream)
    report = markers("[6] and [Yahoo Finance] and [1]", ranks=RANKS)

    log_markers(report, thread_id="t-1", sources=3)

    (line,) = [
        json.loads(row)
        for row in stream.getvalue().splitlines()
        if json.loads(row).get("event") == "citation_markers"
    ]
    assert line["fields"] == {
        "thread_id": "t-1",
        "sources": 3,
        "resolved": 1,
        "unresolved": [6],
        "non_numeric": 1,
        "clean": False,
    }
