"""The free retrieval metrics, and the four places they refuse to report a number.

Seam: `Retrieval` in, ratios out — pure, so every case below is a hand-built retrieval rather
than a run. The tests that matter most are the negative ones: an absent target rank, an
unreachable recall ceiling, a trivial row, and a difference the paired sample cannot resolve.
Each is a number this harness could have printed and would have been wrong to.

**One test here exists because the instrument it guards was tautological.** `compare`'s first
version judged a difference of means against the raw per-question *range*, which a difference of
means over shared questions cannot exceed —
`test_the_comparison_that_could_not_fail_now_fires` is the regression, built from the exact arm
summaries the first committed artifact reported for H1.
"""

from __future__ import annotations

import pytest

from finbrief.evaluation.loader import Bucket, load_golden_set
from finbrief.evaluation.metrics import (
    PAIRED_ALPHA,
    Summary,
    Verdict,
    compare,
    leakage_precision,
    minimum_detectable_pairs,
    paired,
    score_row,
    signed_rank_p,
    summarise,
    tolerated_minority_signs,
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


def a_comparison(baseline: list[float], candidate: list[float], *, metric="chunk_recall"):
    """Two arms over the *same* questions — the pairing `compare` is built on."""
    keys = [f"q{index}" for index in range(len(baseline))]
    return compare(
        metric,
        Bucket.SEMANTIC,
        baseline_arm="vector",
        candidate_arm="hybrid",
        baseline=Summary.of(baseline),
        candidate=Summary.of(candidate),
        pairs=paired(
            dict(zip(keys, baseline, strict=True)),
            dict(zip(keys, candidate, strict=True)),
        ),
    )


def test_a_difference_the_paired_sample_cannot_resolve_is_not_called_a_win():
    # **The rule #11 states as "no threshold at four decimals".** Seven questions whose paired
    # differences straddle zero cannot resolve a direction — and #5 measured the embedding call
    # as reproducible only to ~0.0009, which is the floor under any comparison at all.
    result = a_comparison(
        [0.2, 0.5, 0.8, 0.3, 0.6, 0.4, 0.7],
        [0.3, 0.4, 0.9, 0.2, 0.7, 0.3, 0.8],
    )

    assert result.verdict is Verdict.NOT_DETECTED
    assert result.p_value is not None and result.p_value > PAIRED_ALPHA
    assert result.pairs.n == 7


def test_the_comparison_that_could_not_fail_now_fires():
    """The regression for the defect ADR-0002's T10 amendment §3 records.

    These are H1's own arm summaries from the first committed artifact: means 0.858 and 0.771
    over per-question values spanning [0.500, 1.000] and [0.367, 1.000]. The old test compared
    |Δ| = 0.087 against a *range* of 0.633 and returned "within spread" — as it did for all 18
    pre-registered comparisons, because a difference of means over shared questions is bounded
    by that range and so could never exceed it. The paired instrument has to be able to reach a
    direction on data the old one could not, and this asserts it on a consistent within-arm
    shift the old basis would have swallowed whole.
    """
    baseline = [0.500, 0.667, 0.750, 0.858, 0.900, 1.000, 1.000]
    candidate = [value - 0.087 for value in baseline]

    result = a_comparison(baseline, candidate)

    # Every question moved the same way, so the direction is resolvable at n=7 — the old
    # range-based basis (0.633) exceeded the largest delta the data could produce and could not.
    assert result.verdict is Verdict.WORSE
    assert result.pairs.effective_n == 7
    assert result.delta == pytest.approx(-0.087)


def test_a_null_result_the_test_had_no_power_for_is_undetectable_not_undetected():
    # The third outcome, and the one the first artifact could not express: at three differing
    # questions the exact two-sided p has a floor of 2/2**3 = 0.25, so no arrangement of the
    # data could reach 0.05. "We could not have seen it" is not "the arms are the same".
    result = a_comparison([0.1, 0.2, 0.3], [0.9, 0.8, 0.7])

    assert result.verdict is Verdict.UNDETECTABLE
    assert result.pairs.detectable is False
    assert result.pairs.effective_n == 3
    # The effect is enormous and consistent; only the sample size stops it being a finding.
    assert result.delta == pytest.approx(0.6)


def test_detectability_turns_on_at_six_differing_questions():
    # An equality, not a bound: 2/2**5 = 0.0625 > 0.05 and 2/2**6 = 0.03125 <= 0.05, so six is
    # the exact point at which the instrument can return anything at all.
    five = paired({f"q{i}": 0.0 for i in range(5)}, {f"q{i}": 1.0 for i in range(5)})
    six = paired({f"q{i}": 0.0 for i in range(6)}, {f"q{i}": 1.0 for i in range(6)})

    assert five.detectable is False
    assert six.detectable is True


def test_zero_differences_are_excluded_from_the_test_and_counted():
    # A question both arms scored identically carries no sign, so it cannot enter a signed-rank
    # test — and `effective_n` is the denominator the p was computed against, so it is reported.
    pairs = paired(
        {"a": 0.5, "b": 0.5, "c": 0.5},
        {"a": 0.5, "b": 0.5, "c": 0.9},
    )

    assert pairs.n == 3
    assert pairs.effective_n == 1
    assert signed_rank_p(pairs.deltas) == pytest.approx(1.0)


def test_the_exact_p_matches_the_hand_computed_null():
    # Seven differences all positive: exactly one of the 2**7 sign assignments is at least as
    # extreme, so the two-sided p is 2/128. Asserted as an equality against arithmetic done by
    # hand, because a p-value from an unverified implementation is a number nobody can weigh.
    assert signed_rank_p([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]) == pytest.approx(2 / 128)
    # And the same magnitudes with mixed signs are not extreme at all.
    assert signed_rank_p([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]) > PAIRED_ALPHA


def test_pairing_drops_a_question_only_one_arm_scored_and_says_so():
    # A mean over one subset minus a mean over another is not a difference. The dropped count is
    # reported for the reason `Samples.absent` is.
    pairs = paired({"a": 0.5, "b": None, "c": 0.4}, {"a": 0.9, "b": 0.9, "c": None})

    assert pairs.question_ids == ("a",)
    assert pairs.dropped == 2


def test_a_worse_candidate_is_named_worse():
    result = a_comparison(
        [0.80, 0.82, 0.81, 0.79, 0.83, 0.78, 0.84],
        [0.10, 0.12, 0.11, 0.09, 0.13, 0.08, 0.14],
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
        pairs=paired({"a": None}, {"a": 0.5}),
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


def test_the_power_floor_is_derived_from_alpha_and_not_written_down():
    # Six at alpha=0.05, and derived: 2/2**5 = 0.0625 > 0.05 >= 2/2**6 = 0.03125. A typed
    # constant would stop matching PAIRED_ALPHA the day anyone moved it.
    floor = minimum_detectable_pairs()

    assert floor == 6
    assert 2 / 2 ** (floor - 1) > PAIRED_ALPHA
    assert 2 / 2**floor <= PAIRED_ALPHA


def test_at_the_power_floor_the_effect_must_be_perfectly_unanimous():
    """What the design can actually resolve, and it is bleaker than the p-floor alone says.

    At exactly `minimum_detectable_pairs()` differing questions **no** difference may point
    against the majority; at seven, exactly one may. So a bucket of ~7 questions resolves only
    a near-unanimous effect, at any effect size — the design consequence the artifact's power
    audit states, asserted here as equalities so it cannot drift.
    """
    assert tolerated_minority_signs(5) == -1, "no result is reachable below the floor"
    assert tolerated_minority_signs(6) == 0
    assert tolerated_minority_signs(7) == 1
    assert tolerated_minority_signs(8) == 2


def test_a_five_of_seven_effect_is_unresolvable_however_large_it_is():
    # The concrete consequence: a difference of 0.2 holding on five of seven questions is not
    # weakly supported, it is *unresolvable*. This is the case the audit's prose names.
    deltas = [0.2, 0.2, 0.2, 0.2, 0.2, -0.05, -0.05]

    assert signed_rank_p(deltas) > PAIRED_ALPHA
    assert tolerated_minority_signs(7) == 1, "two minority signs is one too many at n=7"
