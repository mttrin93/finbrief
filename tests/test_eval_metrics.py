"""The free retrieval metrics, and the four places they refuse to report a number.

Seam: `Retrieval` in, ratios out — pure, so every case below is a hand-built retrieval rather
than a run. The tests that matter most are the negative ones: an absent target rank, an
unreachable recall ceiling, a trivial row, and a difference smaller than its own spread. Each
is a number this harness could have printed and would have been wrong to.
"""

from __future__ import annotations

import pytest

from finbrief.evaluation.loader import Bucket, load_golden_set
from finbrief.evaluation.metrics import (
    Summary,
    Verdict,
    compare,
    leakage_precision,
    score_row,
    summarise,
)
from finbrief.ingestion.model import Section
from finbrief.retrieval.retrieve import Context, Retrieval


def a_context(chunk_id: str, *, rank: int, ticker: str = "TSLA", section: str = "Item 1A"):
    return Context(
        chunk_id=chunk_id,
        body="filing text",
        ticker=ticker,
        filing_type="10-K",
        section=Section(section),
        fiscal_year=2025,
        accession=chunk_id.split(":")[0],
        distance=0.5,
        rank=rank,
        fused_score=1 / (60 + rank),
        provenance=(),
    )


def a_retrieval(*contexts) -> Retrieval:
    return Retrieval(contexts=tuple(contexts), variants=("q",), translated=False)


@pytest.fixture(scope="module")
def golden():
    return load_golden_set()


@pytest.fixture(scope="module")
def s1(golden):
    """S1: one section, six target chunks — so recall@5 cannot reach 1.0."""
    return golden.row("S1")


def targets_of(row, count: int, *, start: int = 0):
    return sorted(row.target_chunk_ids)[start : start + count]


# --- the ratios ----------------------------------------------------------------------


def test_precision_and_recall_over_a_partial_hit(s1):
    retrieved = targets_of(s1, 2) + [
        "0000000000-00-000000:Item 1:0",
        "x:Item 1:1",
        "y:Item 1:2",
    ]
    retrieval = a_retrieval(*[a_context(cid, rank=i) for i, cid in enumerate(retrieved, 1)])

    score = score_row(s1, retrieval, arm="vector", k=5)

    assert score.target_hits == 2
    assert score.precision_at_k == pytest.approx(2 / 5)
    assert score.chunk_recall == pytest.approx(2 / 6)
    assert score.hit is True


def test_the_recall_ceiling_says_how_much_headroom_there_was(s1):
    # Six targets at k=5: the retriever's best possible recall is 5/6. Reporting 0.83 against a
    # 1.0 expectation without the ceiling reads as a failure that was arithmetically impossible.
    retrieval = a_retrieval(
        *[a_context(cid, rank=i) for i, cid in enumerate(targets_of(s1, 5), 1)]
    )

    score = score_row(s1, retrieval, arm="vector", k=5)

    assert score.chunk_recall == pytest.approx(5 / 6)
    assert score.chunk_recall_ceiling == pytest.approx(5 / 6)
    assert score.chunk_recall == score.chunk_recall_ceiling, "the retriever did all it could"


def test_an_absent_target_has_no_rank_rather_than_a_stand_in(s1):
    # A `k + 1` or a `999` here would average into the rank column and read as a measurement
    # of a retrieval that missed entirely. `Context.distance` is optional for the same reason.
    retrieval = a_retrieval(a_context("nothing:Item 1:0", rank=1))

    score = score_row(s1, retrieval, arm="vector", k=5)

    assert score.target_rank is None
    assert score.hit is False
    assert score.chunk_recall == 0.0


def test_the_target_rank_is_the_best_one_not_the_first_seen(s1):
    wanted = targets_of(s1, 2)
    retrieval = a_retrieval(
        a_context("miss:Item 1:0", rank=1),
        a_context(wanted[1], rank=2),
        a_context(wanted[0], rank=3),
    )

    score = score_row(s1, retrieval, arm="vector", k=5)

    assert score.target_rank == 2


def test_an_empty_retrieval_has_no_precision_rather_than_zero(s1):
    # Emptiness is not wrongness. `0.0` would claim the retriever returned bad chunks where it
    # returned none — the same distinction `rag.answer_question` draws for its fallback.
    score = score_row(s1, a_retrieval(), arm="vector", k=5)

    assert score.precision_at_k is None
    assert score.filer_precision is None
    assert score.chunk_recall == 0.0


def test_filer_precision_counts_the_rows_own_filers(s1):
    # ADR-0004 §7 predicts hybrid positive here (3/5 -> 5/5 TSLA on issue #6's case) while its
    # contribution to target-chunk rank may be zero or negative. Two different columns.
    retrieval = a_retrieval(
        a_context("a:Item 1A:0", rank=1, ticker="TSLA"),
        a_context("b:Item 1A:1", rank=2, ticker="TSLA"),
        a_context("c:Item 7:0", rank=3, ticker="F"),
        a_context("d:Item 7:1", rank=4, ticker="F"),
        a_context("e:Item 7:2", rank=5, ticker="BAC"),
    )

    score = score_row(s1, retrieval, arm="hybrid", k=5)

    assert score.filer_precision == pytest.approx(2 / 5)


def test_section_recall_counts_sections_reached_not_chunks(golden):
    # The measure that matters for a cross-filer row: did the retrieval reach something from
    # each filer it needed? A chunk count cannot say that — five chunks from one filer would
    # look like a strong result on a three-filer question.
    row = golden.row("M3")
    reached = row.grounding[0]
    retrieval = a_retrieval(
        a_context(
            reached.chunk_ids[0], rank=1, ticker=reached.ticker, section=reached.section.value
        )
    )

    score = score_row(row, retrieval, arm="vector", k=5)

    # Ford and GM, one section each: reaching one filer is half the row, however many of that
    # filer's chunks came back.
    assert row.non_trivial_sections == ("F Item 1A", "GM Item 1A")
    assert score.section_recall == pytest.approx(1 / 2)


def test_a_trivial_only_row_reports_no_section_recall(golden):
    # S4's single target section holds 3 chunks, so a k=5 retrieval reaches it whatever the
    # retriever does. Averaging a guaranteed 1.0 into the column is the inflation ADR-0002's
    # amendment added the flag to prevent.
    row = golden.row("S4")
    retrieval = a_retrieval(a_context("anything:Item 1:0", rank=1))

    score = score_row(row, retrieval, arm="vector", k=5)

    assert row.recall_trivial is True
    assert score.section_recall is None


def test_a_partially_trivial_row_is_scored_on_its_non_trivial_sections_only(golden):
    # M2 is the three-filer 100bp comparison, and two of its three Item 7A sections hold <= 4
    # chunks (AAPL 4, MSFT 3 — counted from the collection, not assumed). So it contributes
    # META's section alone: the retriever is neither credited for the two it could not miss
    # nor dropped.
    row = golden.row("M2")

    assert row.recall_trivial is False
    assert row.target_sections == ("AAPL Item 7A", "MSFT Item 7A", "META Item 7A")
    assert row.non_trivial_sections == ("META Item 7A",)

    reached = a_retrieval(a_context("x:Item 7A:0", rank=1, ticker="META", section="Item 7A"))
    missed = a_retrieval(a_context("y:Item 7A:0", rank=1, ticker="AAPL", section="Item 7A"))

    # Reaching only a trivial section scores zero, which is the point of excluding them: the two
    # easy sections cannot carry the row.
    assert score_row(row, reached, arm="v", k=5).section_recall == pytest.approx(1.0)
    assert score_row(row, missed, arm="v", k=5).section_recall == pytest.approx(0.0)


def test_leaked_chunks_are_recorded_per_row(golden):
    # S2's known false positive is a correct lexical match for the question and wrong grounding.
    row = golden.row("S2")
    leaked = sorted(row.known_false_positives)[0]
    retrieval = a_retrieval(
        a_context(sorted(row.target_chunk_ids)[0], rank=1),
        a_context(leaked, rank=2, ticker="NVDA"),
    )

    score = score_row(row, retrieval, arm="hybrid", k=5)

    assert score.leaked_chunk_ids == (leaked,)


def test_leakage_precision_is_one_minus_the_leaked_share(golden):
    row = golden.row("S2")
    leaked = sorted(row.known_false_positives)[0]
    retrieval = a_retrieval(
        a_context("a:Item 1A:0", rank=1),
        a_context(leaked, rank=2, ticker="NVDA"),
        a_context("c:Item 1A:2", rank=3),
        a_context("d:Item 1A:3", rank=4),
    )
    score = score_row(row, retrieval, arm="hybrid", k=5)

    measured = leakage_precision([score], arm="hybrid")

    assert measured.leaked == 1
    assert measured.retrieved == 4
    assert measured.precision == pytest.approx(0.75)


def test_leakage_precision_over_no_probes_is_absent_rather_than_perfect():
    # A rate over nothing is not 1.0. Rows nobody enumerated false positives for contribute no
    # evidence, and counting them clean would dilute the measurement towards a flattering
    # number.
    assert leakage_precision([], arm="hybrid").precision is None


# --- summaries and the refusal to over-read a difference -----------------------------


def test_a_summary_carries_the_spread_and_the_denominator():
    summary = Summary.of([0.2, 0.4, 0.9])

    assert summary.n == 3
    assert summary.mean == pytest.approx(0.5)
    assert summary.spread == pytest.approx(0.7)
    assert summary.total == 3


def test_an_undefined_value_is_absent_and_never_averaged_as_zero():
    # The `observability.events.Samples` rule: a mean over `present` alone is a mean over a
    # denominator the caller never chose, so the absence is returned beside it.
    summary = Summary.of([1.0, None, 1.0])

    assert summary.mean == pytest.approx(1.0)
    assert summary.n == 2
    assert summary.absent == 1
    assert summary.total == 3


def test_a_summary_over_nothing_has_no_mean():
    summary = Summary.of([None, None])

    assert summary.mean is None
    assert summary.spread is None
    assert summary.n == 0
    assert summary.absent == 2


def test_summarise_can_exclude_whole_trivial_rows(golden):
    trivial = score_row(
        golden.row("S4"), a_retrieval(a_context("a:Item 1:0", rank=1)), arm="v", k=5
    )
    ordinary = score_row(
        golden.row("S1"), a_retrieval(a_context("b:Item 1A:0", rank=1)), arm="v", k=5
    )

    assert summarise([trivial, ordinary], "chunk_recall", exclude_trivial=True).total == 1
    assert summarise([trivial, ordinary], "chunk_recall").total == 2


def test_a_difference_inside_the_spread_is_not_called_a_win():
    # **The rule #11 states as "no threshold at four decimals".** These two arms differ by
    # 0.05 in the mean over questions that themselves range across 0.6, so 7 questions cannot
    # tell them apart — and #5 measured the embedding call as reproducible only to ~0.0009,
    # which is the floor under any comparison at all.
    baseline = Summary.of([0.2, 0.5, 0.8])
    candidate = Summary.of([0.25, 0.55, 0.85])

    result = compare(
        "chunk_recall",
        Bucket.SEMANTIC,
        baseline_arm="vector",
        candidate_arm="hybrid",
        baseline=baseline,
        candidate=candidate,
    )

    assert result.delta == pytest.approx(0.05)
    assert result.basis == pytest.approx(0.6)
    assert result.verdict is Verdict.WITHIN_SPREAD


def test_a_difference_larger_than_the_spread_is_a_direction():
    baseline = Summary.of([0.10, 0.12, 0.11])
    candidate = Summary.of([0.80, 0.82, 0.81])

    result = compare(
        "chunk_recall",
        Bucket.EXACT_IDENTIFIER,
        baseline_arm="vector",
        candidate_arm="hybrid",
        baseline=baseline,
        candidate=candidate,
    )

    assert result.verdict is Verdict.BETTER
    assert result.delta == pytest.approx(0.7)


def test_a_worse_candidate_is_named_worse():
    result = compare(
        "chunk_recall",
        Bucket.EXACT_IDENTIFIER,
        baseline_arm="vector",
        candidate_arm="hybrid",
        baseline=Summary.of([0.80, 0.82, 0.81]),
        candidate=Summary.of([0.10, 0.12, 0.11]),
    )

    assert result.verdict is Verdict.WORSE


def test_a_comparison_with_a_missing_side_is_undetermined_not_zero():
    # ADR-0005's clause must not fire because one arm produced no number: "we cannot say" and
    # "translation is worse" are different claims.
    result = compare(
        "chunk_recall",
        Bucket.SEMANTIC,
        baseline_arm="vector",
        candidate_arm="hybrid",
        baseline=Summary.of([None]),
        candidate=Summary.of([0.5]),
    )

    assert result.verdict is Verdict.UNDETERMINED
    assert result.delta is None


def test_section_precision_asks_whether_a_chunk_came_from_a_target_section(golden):
    # The precision counterpart to section recall, and the deterministic headline alongside it:
    # stricter than filer precision (right filer, wrong Item is a miss) and looser than chunk
    # identity (any chunk of the right Item counts).
    row = golden.row("S1")  # grounds in TSLA Item 1A
    retrieval = a_retrieval(
        a_context("x:Item 1A:99", rank=1, ticker="TSLA", section="Item 1A"),
        a_context("y:Item 7:1", rank=2, ticker="TSLA", section="Item 7"),
        a_context("z:Item 1A:2", rank=3, ticker="F", section="Item 1A"),
        a_context("w:Item 1A:3", rank=4, ticker="TSLA", section="Item 1A"),
    )

    score = score_row(row, retrieval, arm="hybrid", k=5)

    # Two of four sit in TSLA Item 1A. The right filer's wrong Item and the wrong filer's right
    # Item both miss, which is the distinction from filer precision.
    assert score.section_precision == pytest.approx(0.5)
    assert score.filer_precision == pytest.approx(0.75)


def test_section_precision_credits_a_trivial_section(golden):
    # `recall_trivial` says a section cannot be *missed*, a claim about recall. Retrieving from
    # it is still correct, and calling it a precision error would penalise the retriever for
    # the corpus's shape.
    row = golden.row("M2")
    retrieval = a_retrieval(a_context("x:Item 7A:0", rank=1, ticker="AAPL", section="Item 7A"))

    score = score_row(row, retrieval, arm="v", k=5)

    assert "AAPL Item 7A" in row.recall_trivial_sections
    assert score.section_precision == pytest.approx(1.0)
    assert score.section_recall == pytest.approx(0.0), "but it earns no recall credit"


def test_section_precision_is_absent_for_an_empty_retrieval(golden):
    assert score_row(golden.row("S1"), a_retrieval(), arm="v", k=5).section_precision is None
