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


#: Seven values per arm, because seven is the golden set's bucket size and six is the point at
#: which the exact paired test can return anything at all (`metrics.Paired.detectable`). A
#: fixture with three would make every assertion below `UNDETECTABLE` and prove nothing.
BUCKET_N = 7


def cells_for(bucket: Bucket, values: dict[str, list[float]], metric: str) -> list[Cell]:
    """One cell per (arm, question), the arms sharing question ids so they pair.

    **Shared ids are the point.** The helper first gave each arm its own row ids, which no real
    run does — every arm scores the same golden-set rows — and under a paired instrument that
    fixture pairs nothing at all. A test double that cannot represent the thing being measured
    is how an instrument goes unchecked (`metrics`' module docstring).
    """
    return [
        a_cell(arm=arm, bucket=bucket, row=f"q{index}", **{metric: value})
        for arm, series in values.items()
        for index, value in enumerate(series)
    ]


def cells_with(bucket: Bucket, arms: dict[str, dict[str, list[float]]]) -> list[Cell]:
    """One cell per (arm, question) carrying **every** metric — the shape a real run produces.

    ADR-0005's clause reads two metrics off the same cell, so a fixture that emits one cell per
    metric gives each question two cells and the second silently shadows the first. That is not
    a shape any run produces, and under a paired instrument it pairs nothing.
    """
    cells: list[Cell] = []
    for arm, metrics in arms.items():
        length = len(next(iter(metrics.values())))
        for index in range(length):
            cells.append(
                a_cell(
                    arm=arm,
                    bucket=bucket,
                    row=f"q{index}",
                    **{metric: values[index] for metric, values in metrics.items()},
                )
            )
    return cells


def series(*, shift: float = 0.0) -> list[float]:
    """Seven scores spanning most of [0, 1], optionally shifted by a constant.

    A constant shift is the shape a paired test resolves and a range-based one cannot: every
    question moves the same way, so the per-question differences are unanimous while the raw
    spread is unchanged.
    """
    base = [0.30, 0.42, 0.51, 0.60, 0.68, 0.79, 0.88]
    return [round(value + shift, 4) for value in base]


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
            prediction.baseline.name: series(),
            prediction.candidate.name: series(shift=0.10),
        },
        prediction.metric,
    )

    result = next(o for o in outcomes(cells) if o.prediction.id == prediction.id)

    assert result.measured is Verdict.BETTER
    assert result.confirmed is True
    assert result.verdict_text == "confirmed"


def test_a_measured_loss_refutes_a_better_prediction():
    # **The only thing that refutes a directional prediction**: a difference resolved in the
    # other direction. H1 predicts hybrid > vector, so seven questions all moving the other way
    # is a refutation — and nothing weaker is.
    prediction = PREDICTIONS[0]
    cells = cells_for(
        prediction.bucket,
        {
            prediction.baseline.name: series(),
            prediction.candidate.name: series(shift=-0.10),
        },
        prediction.metric,
    )

    result = next(o for o in outcomes(cells) if o.prediction.id == prediction.id)

    assert result.measured is Verdict.WORSE
    assert result.confirmed is False, "a refuted prediction is a result, not a failure"
    assert result.verdict_text == "refuted"


def test_a_null_result_does_not_refute_a_better_prediction():
    """The defect ADR-0002's T10 amendment §3 records, as a regression test.

    H1, H2, H3 and H6 were published as **refuted** on null results by the first committed run:
    `Verdict`'s docstring said the arms may genuinely differ, and `Outcome.confirmed` collapsed
    that into `False` one line later. A prediction the sample cannot settle is `None` —
    neither confirmed nor refuted — and the verdict column has to say which.
    """
    flat = {"vector": series(), "hybrid": series(shift=0.004)}
    cells = cells_for(Bucket.EXACT_IDENTIFIER, flat, CONTEXT_PRECISION)

    result = next(o for o in outcomes(cells) if o.prediction.id == "H1")

    assert result.measured is Verdict.BETTER or result.confirmed is None
    # The shift is unanimous, so this fixture pins the *wording* rather than the direction:
    # whatever the verdict, it is never the word "refuted" without a resolved loss.
    if result.measured is not Verdict.WORSE:
        assert result.verdict_text != "refuted"


def test_an_undetectable_comparison_is_not_a_refutation_and_names_its_n():
    # Three questions cannot resolve a difference of any size, so a prediction of BETTER comes
    # back `None` with the effective n in the verdict — never "refuted".
    cells = cells_for(
        Bucket.EXACT_IDENTIFIER,
        {"vector": [0.1, 0.2, 0.3], "hybrid": [0.2, 0.3, 0.4]},
        CONTEXT_PRECISION,
    )

    result = next(o for o in outcomes(cells) if o.prediction.id == "H1")

    assert result.measured is Verdict.UNDETECTABLE
    assert result.confirmed is None
    assert result.verdict_text == "undetectable (effective n=3)"


def test_a_predicted_tie_is_consistent_with_a_null_rather_than_confirmed_by_it():
    # ADR-0002 decision 4 registered the tie as a prediction and it stays readable as one — but
    # a null result at n=7 does not confirm a tie, so the verdict column says "not detected"
    # and the measurement cell carries the nuance.
    cells = cells_for(
        Bucket.SEMANTIC,
        {
            "vector+translation": series(),
            "hybrid+translation": [0.30, 0.45, 0.48, 0.62, 0.66, 0.81, 0.85],
        },
        CONTEXT_PRECISION,
    )

    result = next(o for o in outcomes(cells) if o.prediction.id == "H4")
    entry = next(e for e in as_report_entries([result]))

    assert result.measured is Verdict.NOT_DETECTED
    assert result.verdict_text == "not detected (n=7)"
    assert result.confirmed is None
    assert "consistent" in entry["measurement"]


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


def test_the_measurement_cell_names_the_paired_delta_the_exact_p_and_both_ns():
    # The three numbers the verdict is a summary of. A p without its effective n cannot be
    # weighed, and a delta without the p is the figure the old instrument printed beside a
    # verdict it could not have reached.
    cells = cells_for(
        Bucket.EXACT_IDENTIFIER,
        {"vector": series(), "hybrid": series(shift=0.10)},
        CONTEXT_PRECISION,
    )

    entry = next(e for e in as_report_entries(outcomes(cells)) if "**H1**" in e["prediction"])

    assert "paired Δ +0.100" in entry["measurement"]
    assert "exact signed-rank p=" in entry["measurement"]
    assert "n=7 (7 differing)" in entry["measurement"]
    assert "spread" not in entry["measurement"]


# --- ADR-0005's falsification clause -------------------------------------------------


def test_the_clause_fires_only_when_both_metrics_move_against_translation():
    cells = cells_with(
        Bucket.SEMANTIC,
        {
            "hybrid": {
                CONTEXT_PRECISION: series(shift=0.10),
                CONTEXT_RECALL: series(shift=0.10),
            },
            "hybrid+translation": {
                CONTEXT_PRECISION: series(shift=-0.10),
                CONTEXT_RECALL: series(shift=-0.10),
            },
        },
    )

    clause = next(c for c in falsification_clause(cells) if c.bucket is Bucket.SEMANTIC)

    assert clause.precision.verdict is Verdict.WORSE
    assert clause.recall.verdict is Verdict.WORSE
    assert clause.could_fire is True
    assert clause.fires is True


def test_the_clause_does_not_fire_on_precision_alone():
    # **Directional and two-sided, deliberately.** ADR-0005 drops the default only when
    # translation is worse on *both*; firing on one would drop a pre-registered default on
    # half the evidence.
    cells = cells_with(
        Bucket.SEMANTIC,
        {
            "hybrid": {
                CONTEXT_PRECISION: series(shift=0.10),
                CONTEXT_RECALL: series(shift=-0.10),
            },
            "hybrid+translation": {
                CONTEXT_PRECISION: series(shift=-0.10),
                CONTEXT_RECALL: series(shift=0.10),
            },
        },
    )

    clause = next(c for c in falsification_clause(cells) if c.bucket is Bucket.SEMANTIC)

    assert clause.precision.verdict is Verdict.WORSE
    assert clause.recall.verdict is Verdict.BETTER
    assert clause.could_fire is True
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
            {
                "vector+translation": series(),
                # Differences in both directions, so each bucket has the power to resolve a
                # gain and resolves none — which is the evidence the trigger fires on.
                "hybrid+translation": [0.34, 0.38, 0.55, 0.56, 0.72, 0.75, 0.84],
            },
            CONTEXT_PRECISION,
        )

    trigger = reexamination_trigger(cells)

    assert trigger.evaluable is True
    assert trigger.undetectable == ()
    assert trigger.fires is True
    assert trigger.undetermined == ()


def test_the_trigger_cannot_be_evaluated_when_no_bucket_has_the_power():
    """The correction ADR-0002's T10 amendment §3 records, as a regression test.

    This trigger fires on an *absence* of gain, so a powerless instrument feeds it directly:
    under the range-based test every bucket returned a null whatever the data said, and the
    first committed run reported the trigger as **FIRING** on that tautology. Three questions
    per bucket cannot resolve a gain of any size, so there is no absence to fire on.
    """
    cells = []
    for bucket in Bucket:
        cells += cells_for(
            bucket,
            {"vector+translation": [0.1, 0.5, 0.9], "hybrid+translation": [0.2, 0.4, 0.8]},
            CONTEXT_PRECISION,
        )

    trigger = reexamination_trigger(cells)

    assert trigger.evaluable is False
    assert len(trigger.undetectable) == len(tuple(Bucket))
    assert trigger.fires is False, "an absence nobody could have detected is not an absence"


def test_an_undetectable_bucket_is_excluded_rather_than_counted_as_an_absence():
    # One bucket with power and a null; the rest powerless. The trigger fires on the evidence it
    # has rather than being blocked by, or inflated with, the buckets that carry none.
    cells = cells_for(
        Bucket.SEMANTIC,
        {
            "vector+translation": series(),
            "hybrid+translation": [0.34, 0.38, 0.55, 0.56, 0.72, 0.75, 0.84],
        },
        CONTEXT_PRECISION,
    )
    for bucket in Bucket:
        if bucket is not Bucket.SEMANTIC:
            cells += cells_for(
                bucket,
                {"vector+translation": [0.1, 0.5], "hybrid+translation": [0.2, 0.4]},
                CONTEXT_PRECISION,
            )

    trigger = reexamination_trigger(cells)

    assert Bucket.SEMANTIC not in trigger.undetectable
    assert trigger.evaluable is True
    assert trigger.fires is True


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
    """A sink holding `events`, with `turn_id` on the **envelope** where the formatter puts it.

    `JsonLinesFormatter` hoists `turn_id` out of the payload — a field named like an envelope
    key would otherwise collide silently (`observability/events.py`) — so a fixture leaving it
    in `fields` would build lines no emitter produces, and `Event.turn_id` would read `None` on
    every one. That is the shape of double that proves nothing.
    """
    path = tmp_path / "events.jsonl"
    lines = []
    for name, fields in events:
        payload = {"ts": "2026-07-29T10:00:00+00:00", "event": name}
        body = dict(fields)
        turn_id = body.pop("turn_id", None)
        if turn_id is not None:
            payload["turn_id"] = turn_id
        payload["fields"] = body
        lines.append(json.dumps(payload))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
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
        ("retrieval", {"latency_ms": 1400, "translation": True, "variants": 5}),
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
        ("retrieval", {"latency_ms": 1400, "translation": True, "variants": 5}),
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
        ("retrieval", {"latency_ms": 1400, "translation": True, "variants": 5}),
    )

    with pytest.raises(NoSamples, match="translation off"):
        translation_cost(read_events(path))


def test_a_config_disabled_planner_is_excluded_and_a_refusal_is_not(tmp_path):
    """The exclusion keys on the arm's configuration, not on how many variants came back.

    Four of the six arms carry `translation: true` and only two plan, so the ablations' cheaper
    retrievals must not be averaged under a budget meant for the planner's round. But an
    earlier version keyed on the *observed* variant count and dropped anything under three,
    which conflated "disabled by config" with "ran and refused" — and a refusal paid for a full
    chat round. Dropping refusals removes the cheap retrievals from a median of expensive ones,
    biasing the p50 **upward**, which makes the budget miss look worse than it is. So the key is
    the turn's own `max_sub_queries`, joined through `logging_setup.turn`.
    """
    path = a_log(
        tmp_path,
        ("query_translation", {"latency_ms": 900, "input_tokens": 260}),
        # A planning arm: planner enabled, three sub-queries.
        (
            "query_translation",
            {"max_sub_queries": 3, "sub_queries": 3, "turn_id": "hybrid+translation/S1"},
        ),
        (
            "retrieval",
            {
                "latency_ms": 1400,
                "translation": True,
                "variants": 5,
                "turn_id": "hybrid+translation/S1",
            },
        ),
        # A planning arm whose planner **refused**: enabled, zero sub-queries, one variant. The
        # chat round was paid for, so this belongs in the pool even though it looks cheap.
        (
            "query_translation",
            {"max_sub_queries": 3, "sub_queries": 0, "turn_id": "hybrid+translation/S2"},
        ),
        (
            "retrieval",
            {
                "latency_ms": 500,
                "translation": True,
                "variants": 1,
                "turn_id": "hybrid+translation/S2",
            },
        ),
        # An ablation arm: planner disabled by configuration, ticker form only.
        (
            "query_translation",
            {"max_sub_queries": 0, "sub_queries": 0, "turn_id": "hybrid+normalisation/S1"},
        ),
        (
            "retrieval",
            {
                "latency_ms": 450,
                "translation": True,
                "variants": 2,
                "turn_id": "hybrid+normalisation/S1",
            },
        ),
        ("retrieval", {"latency_ms": 1000, "translation": False, "variants": 1}),
    )

    cost = translation_cost(read_events(path))

    # The refusal is in, the config-disabled arm is out.
    assert cost.translated_samples == 2
    assert cost.planner_disabled_lines == 1
    assert cost.refusal_lines_kept == 1
    # Median of {1400, 500} = 950. Under the old variant-count floor the 500ms refusal was
    # dropped and the median was 1400 — the upward bias this test pins.
    assert cost.retrieval_translated_p50_ms == 950
    assert cost.retrieval_delta_p50_ms == -50


def test_an_unattributable_translated_retrieval_stays_in_the_pool(tmp_path):
    # A `retrieval` line with no turn scope cannot be shown to come from a disabled-planner arm,
    # and the ablation arms are the only ones that are — so the honest default is to keep it.
    # Excluding on absence of evidence is what the old floor did.
    path = a_log(
        tmp_path,
        ("query_translation", {"latency_ms": 900, "input_tokens": 260}),
        ("retrieval", {"latency_ms": 1400, "translation": True, "variants": 1}),
        ("retrieval", {"latency_ms": 1000, "translation": False}),
    )

    cost = translation_cost(read_events(path))

    assert cost.translated_samples == 1
    assert cost.planner_disabled_lines == 0


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
