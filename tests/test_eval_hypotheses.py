"""The pre-registered predictions, the two triggers, and the log refusals.

Seam: hand-built `Cell`s, so every branch of a clause that could drop the shipping default is
exercised without a run. The tests that matter are the ones asserting a clause does **not**
fire — ADR-0005's is deliberately narrow, and a clause that fires on the wrong evidence would
change the shipped default for a reason nobody registered.
"""

from __future__ import annotations

import json

import pytest

from finbrief.evaluation.hypotheses import (
    PREDICTIONS,
    FencedMetric,
    Prediction,
    as_report_entries,
    falsification_clause,
    outcomes,
    reexamination_trigger,
)
from finbrief.evaluation.judge import (
    CONTEXT_PRECISION,
    CONTEXT_RECALL,
    EXCLUDED_FROM_HYPOTHESES,
)
from finbrief.evaluation.latency import (
    NoSamples,
    SinkMissing,
    load_log,
    p50,
    token_spend,
    translation_cost,
)
from finbrief.evaluation.loader import Bucket
from finbrief.evaluation.metrics import RowScore, Verdict
from finbrief.evaluation.pipeline import Cell
from finbrief.observability.events import read_events
from finbrief.retrieval.retrieve import Retrieval


def a_cell(*, arm: str, bucket: Bucket, row: str, **judged) -> Cell:
    score = RowScore(
        question_id=row,
        bucket=bucket,
        arm=arm,
        k=5,
        retrieved_chunk_ids=(),
        target_hits=0,
        precision_at_k=None,
        chunk_recall=0.0,
        chunk_recall_ceiling=1.0,
        section_recall=None,
        section_precision=None,
        filer_precision=None,
        target_rank=None,
        leaked_chunk_ids=(),
        recall_trivial=False,
    )
    return Cell(
        question_id=row,
        bucket=bucket,
        arm=arm,
        retrieval=Retrieval(contexts=(), variants=("q",), translated=False),
        score=score,
        answer="an answer",
        judged=judged,
    )


def cells_for(bucket: Bucket, values: dict[str, list[float]], metric: str) -> list[Cell]:
    """One cell per (arm, value), so a bucket has a real spread to be judged against."""
    return [
        a_cell(arm=arm, bucket=bucket, row=f"{arm}-{index}", **{metric: value})
        for arm, series in values.items()
        for index, value in enumerate(series)
    ]


# --- the predictions -----------------------------------------------------------------


def test_every_prediction_names_a_source_and_a_settling_comparison():
    for prediction in PREDICTIONS:
        assert prediction.source, prediction.id
        assert prediction.baseline is not prediction.candidate
        assert prediction.metric not in EXCLUDED_FROM_HYPOTHESES


def test_a_prediction_cannot_be_written_on_the_fenced_metric():
    # The fence enforced at construction, not left to whoever writes the next prediction:
    # response relevancy moves between runs, so a delta in it cannot confirm or refute
    # anything.
    with pytest.raises(FencedMetric):
        Prediction(
            id="HX",
            text="relevancy improves",
            source="nowhere",
            bucket=Bucket.SEMANTIC,
            metric="answer_relevancy",
            baseline=PREDICTIONS[0].baseline,
            candidate=PREDICTIONS[0].candidate,
            expected=Verdict.BETTER,
        )


def test_a_clear_gain_confirms_a_better_prediction():
    prediction = PREDICTIONS[0]
    cells = cells_for(
        prediction.bucket,
        {
            prediction.baseline.name: [0.10, 0.11, 0.12],
            prediction.candidate.name: [0.80, 0.81, 0.82],
        },
        prediction.metric,
    )

    result = next(o for o in outcomes(cells) if o.prediction.id == prediction.id)

    assert result.measured is Verdict.BETTER
    assert result.confirmed is True
    assert result.verdict_text == "confirmed"


def test_a_within_spread_result_refutes_a_better_prediction_and_confirms_a_tie():
    # H1 predicts BETTER and H4 predicts WITHIN_SPREAD, so one set of flat numbers should
    # refute the first and confirm the second — which is what makes a predicted tie evidence
    # rather than an absence of one (ADR-0002 decision 4).
    flat = {"vector": [0.4, 0.5, 0.6], "hybrid": [0.4, 0.5, 0.6]}
    cells = cells_for(Bucket.EXACT_IDENTIFIER, flat, CONTEXT_PRECISION)
    cells += cells_for(
        Bucket.SEMANTIC,
        {"vector+translation": [0.4, 0.5, 0.6], "hybrid+translation": [0.4, 0.5, 0.6]},
        CONTEXT_PRECISION,
    )

    results = {o.prediction.id: o for o in outcomes(cells)}

    assert results["H1"].measured is Verdict.WITHIN_SPREAD
    assert results["H1"].confirmed is False, "a refuted prediction is a result, not a failure"
    assert results["H4"].confirmed is True


def test_a_prediction_the_run_could_not_settle_is_undetermined_not_refuted():
    # Three-valued on purpose: "the sample could not tell" and "the prediction was wrong" are
    # different claims, and collapsing them would report an absence as a refutation.
    result = next(o for o in outcomes([]) if o.prediction.id == "H1")

    assert result.measured is Verdict.UNDETERMINED
    assert result.confirmed is None
    assert result.verdict_text == "undetermined"


def test_the_report_entry_puts_the_prediction_before_the_measurement():
    entries = as_report_entries(outcomes([]))

    assert list(entries[0]) == ["prediction", "measurement", "verdict"]
    assert "ADR" in entries[0]["prediction"]


# --- ADR-0005's falsification clause -------------------------------------------------


def test_the_clause_fires_only_when_both_metrics_move_against_translation():
    cells = cells_for(
        Bucket.SEMANTIC,
        {"hybrid": [0.80, 0.81, 0.82], "hybrid+translation": [0.10, 0.11, 0.12]},
        CONTEXT_PRECISION,
    )
    cells += cells_for(
        Bucket.SEMANTIC,
        {"hybrid": [0.80, 0.81, 0.82], "hybrid+translation": [0.10, 0.11, 0.12]},
        CONTEXT_RECALL,
    )

    clause = next(c for c in falsification_clause(cells) if c.bucket is Bucket.SEMANTIC)

    assert clause.precision is Verdict.WORSE
    assert clause.recall is Verdict.WORSE
    assert clause.fires is True


def test_the_clause_does_not_fire_on_precision_alone():
    # **Directional and two-sided, deliberately.** ADR-0005 drops the default only when
    # translation is worse on *both*; firing on one would drop a pre-registered default on
    # half the evidence.
    cells = cells_for(
        Bucket.SEMANTIC,
        {"hybrid": [0.80, 0.81, 0.82], "hybrid+translation": [0.10, 0.11, 0.12]},
        CONTEXT_PRECISION,
    )
    cells += cells_for(
        Bucket.SEMANTIC,
        {"hybrid": [0.10, 0.11, 0.12], "hybrid+translation": [0.80, 0.81, 0.82]},
        CONTEXT_RECALL,
    )

    clause = next(c for c in falsification_clause(cells) if c.bucket is Bucket.SEMANTIC)

    assert clause.precision is Verdict.WORSE
    assert clause.recall is Verdict.BETTER
    assert clause.fires is False


def test_the_clause_does_not_fire_on_an_unmeasured_bucket():
    # An absent number must not drop the shipping default: "we cannot say" is not "translation
    # is worse".
    assert not any(clause.fires for clause in falsification_clause([]))


# --- ADR-0005 §4's re-examination trigger --------------------------------------------


def test_the_trigger_fires_when_hybrid_adds_nothing_anywhere():
    cells = []
    for bucket in Bucket:
        cells += cells_for(
            bucket,
            {"vector+translation": [0.4, 0.5, 0.6], "hybrid+translation": [0.4, 0.5, 0.6]},
            CONTEXT_PRECISION,
        )

    trigger = reexamination_trigger(cells)

    assert trigger.fires is True
    assert trigger.undetermined == ()


def test_the_trigger_does_not_fire_when_hybrid_wins_one_bucket():
    # It fires on an *absence of gain* on **every** bucket, so a single clear win keeps the
    # dominance argument alive.
    cells = []
    for bucket in Bucket:
        series = (
            {"vector+translation": [0.1, 0.1, 0.1], "hybrid+translation": [0.9, 0.9, 0.9]}
            if bucket is Bucket.EXACT_IDENTIFIER
            else {"vector+translation": [0.4, 0.5, 0.6], "hybrid+translation": [0.4, 0.5, 0.6]}
        )
        cells += cells_for(bucket, series, CONTEXT_PRECISION)

    assert reexamination_trigger(cells).fires is False


def test_an_all_undetermined_trigger_does_not_fire_and_names_the_gaps():
    trigger = reexamination_trigger([])

    assert trigger.fires is False
    assert set(trigger.undetermined) == set(Bucket)


# --- the log, and the refusals ------------------------------------------------------


def a_log(tmp_path, *events) -> str:
    path = tmp_path / "events.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"ts": "2026-07-29T10:00:00+00:00", "event": name, "fields": fields})
            for name, fields in events
        )
        + "\n",
        encoding="utf-8",
    )
    return str(path)


def test_an_unnamed_sink_raises_rather_than_reading_as_an_empty_run():
    # #11's non-negotiable 2. `.env.example` ships the sink commented out, so this is the
    # *likely* state of an evaluation run, and a p50 over nothing rendering as "budget met" is
    # the failure.
    with pytest.raises(SinkMissing, match="never set"):
        load_log(None)


def test_a_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_log(tmp_path / "nothing.jsonl")


def test_a_median_over_no_samples_raises_rather_than_returning_zero():
    with pytest.raises(NoSamples, match="not a budget met"):
        p50([], what="retrieval lines")


def test_the_added_latency_is_the_planner_round_plus_the_retrieval_delta(tmp_path):
    path = a_log(
        tmp_path,
        ("query_translation", {"latency_ms": 900, "input_tokens": 260, "output_tokens": 70}),
        ("retrieval", {"latency_ms": 1400, "translation": True}),
        ("retrieval", {"latency_ms": 1000, "translation": False}),
    )

    cost = translation_cost(read_events(path))

    assert cost.planner_p50_ms == 900
    assert cost.retrieval_delta_p50_ms == 400
    assert cost.added_p50_ms == 1300
    assert cost.within_budget is True


def test_a_replayed_planner_line_is_not_counted_as_translations_cost(tmp_path):
    # The subtlety the replay introduces: a stub-served `query_translation` line has no token
    # counts because no model was called, and counting its ~1ms as the planner's cost would
    # report translation as almost free. The filter is what makes the number the shipped
    # path's.
    path = a_log(
        tmp_path,
        ("query_translation", {"latency_ms": 1, "sub_queries": 3}),
        ("query_translation", {"latency_ms": 1800, "input_tokens": 260, "output_tokens": 70}),
        ("retrieval", {"latency_ms": 1400, "translation": True}),
        ("retrieval", {"latency_ms": 1000, "translation": False}),
    )

    cost = translation_cost(read_events(path))

    assert cost.planner_samples == 1
    assert cost.planner_p50_ms == 1800
    assert cost.within_budget is False, "1800 + 400 is over the 1.5s budget"


def test_a_log_with_no_untranslated_retrieval_refuses_rather_than_assuming_zero(tmp_path):
    # Substituting 0 for the half it could not measure would report a delta that is really a
    # total.
    path = a_log(
        tmp_path,
        ("query_translation", {"latency_ms": 900, "input_tokens": 260}),
        ("retrieval", {"latency_ms": 1400, "translation": True}),
    )

    with pytest.raises(NoSamples, match="translation off"):
        translation_cost(read_events(path))


def test_token_spend_counts_each_field_against_its_own_denominator(tmp_path):
    # `observability/tokens.py`'s rule: a provider returning half a pair must not put a
    # fabricated zero on the line, so the output mean is over the calls that reported output.
    path = a_log(
        tmp_path,
        ("rag_answer", {"input_tokens": 1000, "output_tokens": 200}),
        ("rag_answer", {"input_tokens": 1200}),
        ("rag_answer", {}),
    )

    spend = token_spend(read_events(path), "rag_answer")

    assert spend.input_tokens == 2200
    assert spend.input_calls == 2
    assert spend.output_tokens == 200
    assert spend.output_calls == 1
    assert spend.lines == 3
    assert spend.unmetered_lines == 1
