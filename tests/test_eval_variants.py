"""Resolve-once, replay-everywhere — ADR-0004 §9's mechanism for making the A/B exact.

Seam: `retrieve()`'s injectable `model=`. What is asserted is the property §9 is *for* — that
two arms differing only in strategy retrieve over byte-identical variants — plus the check
that keeps the persisted file from drifting away from the parser that consumes it.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import HumanMessage

from finbrief.config import RetrievalStrategy, Settings
from finbrief.evaluation.arms import HYBRID_TRANSLATED, VECTOR_TRANSLATED
from finbrief.evaluation.loader import load_golden_set
from finbrief.evaluation.variants import (
    MissingPlan,
    PlannerAgreement,
    ReplayMismatch,
    ReplayPlanner,
    ResolvedPlan,
    VariantSet,
    agreement,
    load_variants,
    resolve_plan,
    save_variants,
    verify_replay,
)
from finbrief.retrieval.retrieve import retrieve

QUESTION = "How much debt does Tesla carry?"

#: A planner reply in the shape a real one arrives in: one per line, with list furniture the
#: parser is supposed to strip.
REPLY = "- Tesla total debt\n- Tesla liquidity and capital resources\n"


def a_plan(**overrides) -> ResolvedPlan:
    fields = {
        "question_id": "E1",
        "question": QUESTION,
        "reply": REPLY,
        "variants": (
            QUESTION,
            "How much debt does TSLA carry?",
            "Tesla total debt",
            "Tesla liquidity and capital resources",
        ),
        "planner_model": "openai/gpt-4o-mini",
        "max_sub_queries": 3,
    }
    return ResolvedPlan(**{**fields, **overrides})


def a_variant_set(*plans: ResolvedPlan) -> VariantSet:
    return VariantSet(
        plans={plan.question_id: plan for plan in plans},
        planner_model="openai/gpt-4o-mini",
        max_sub_queries=3,
        golden_set_schema_version=1,
    )


def a_settings(**overrides) -> Settings:
    return Settings.from_env({"OPENROUTER_API_KEY": "test-key", **overrides})


# --- the replay itself ---------------------------------------------------------------


def test_the_stub_hands_retrieve_the_recorded_reply(filings_store):
    # The mechanism, end to end against a real on-disk Chroma with a fake embedding: the
    # variants `retrieve()` runs over are the recorded ones, and the planner was actually
    # reached.
    planner = ReplayPlanner(reply=REPLY)

    retrieval = retrieve(
        QUESTION,
        strategy=RetrievalStrategy.VECTOR,
        translate=True,
        k=5,
        store=filings_store,
        settings=a_settings(),
        model=planner,
    )

    assert planner.invocations == 1, (
        "the replay was never reached, so the arm resampled nothing"
    )
    assert retrieval.variants[0] == QUESTION, "translation must only ever add (ADR-0004)"
    assert retrieval.sub_queries == (
        "Tesla total debt",
        "Tesla liquidity and capital resources",
    )


def test_two_arms_differing_only_in_strategy_retrieve_over_identical_variants(filings_store):
    # **The property ADR-0004 §9 exists to buy.** Without the replay these two arms each make
    # their own planner call, so a changed sub-query moves fused ranks and the matrix compares
    # strategy *plus* sampling. Asserted as an equality on the variant tuples, which is the
    # thing that has to be identical — not on the contexts, which are allowed to differ by
    # strategy.
    variants = a_variant_set(a_plan())

    retrievals = {
        arm.name: retrieve(
            QUESTION,
            strategy=arm.strategy,
            translate=arm.translate,
            k=5,
            store=filings_store,
            settings=a_settings(),
            model=variants.planner("E1"),
        )
        for arm in (VECTOR_TRANSLATED, HYBRID_TRANSLATED)
    }

    assert (
        retrievals["vector+translation"].variants
        == retrievals["hybrid+translation"].variants
        == variants.plan("E1").variants
    )


def test_a_replayed_arm_is_reproducible_across_two_runs(filings_store):
    # The claim the artifact makes about its own numbers: same inputs, same contexts, in order.
    def run():
        return retrieve(
            QUESTION,
            strategy=RetrievalStrategy.HYBRID,
            translate=True,
            k=5,
            store=filings_store,
            settings=a_settings(),
            model=ReplayPlanner(reply=REPLY),
        )

    first, second = run(), run()

    assert [c.chunk_id for c in first.contexts] == [c.chunk_id for c in second.contexts]
    assert [c.rank for c in first.contexts] == [c.rank for c in second.contexts]


def test_resolve_captures_the_reply_and_the_variants_it_parses_to():
    # One paid call per question at resolve time, and what is captured is the raw reply — the
    # parsed variants cannot be turned back into a reply, because the parser drops lines.
    plan = resolve_plan(
        "E1",
        QUESTION,
        model=ReplayPlanner(reply=REPLY),
        max_sub_queries=3,
        planner_model="openai/gpt-4o-mini",
    )

    assert plan.reply == REPLY
    assert plan.variants[0] == QUESTION
    assert plan.sub_queries == ("Tesla total debt", "Tesla liquidity and capital resources")
    assert plan.ticker_form == "How much debt does TSLA carry?"


def test_the_ticker_form_is_not_reported_as_a_sub_query():
    # ADR-0004's T6 amendment turns on which of the two additions earns the exact-identifier
    # bucket, so crediting the planner with a config lookup would corrupt the finding.
    plan = a_plan()

    assert plan.ticker_form not in plan.sub_queries


# --- keeping a fixed input honest ----------------------------------------------------


def test_a_reply_that_no_longer_parses_to_its_stored_variants_raises():
    # The parser can legitimately change — it has, twice (`_MARKER`'s decimal fix, `_REFUSAL`).
    # Any such change silently alters what every +translation arm retrieved over, so it has to
    # fail here rather than move a bucket number quietly.
    stale = a_plan(variants=(QUESTION, "something the parser would never produce"))

    with pytest.raises(ReplayMismatch, match="E1"):
        verify_replay((stale,))


def test_a_consistent_plan_verifies():
    verify_replay((a_plan(),))


def test_load_verifies_by_default(tmp_path):
    path = tmp_path / "golden_variants.json"
    save_variants(a_variant_set(a_plan(variants=(QUESTION, "invented"))), path)

    with pytest.raises(ReplayMismatch):
        load_variants(path)

    # And loadable without the check, for inspecting a file that has drifted.
    assert len(load_variants(path, verify=False)) == 1


def test_the_persisted_file_round_trips(tmp_path):
    path = tmp_path / "golden_variants.json"
    save_variants(a_variant_set(a_plan()), path)

    reloaded = load_variants(path)

    assert reloaded.plan("E1") == a_plan()
    assert reloaded.planner_model == "openai/gpt-4o-mini"
    assert reloaded.max_sub_queries == 3


def test_the_persisted_file_says_it_is_generated(tmp_path):
    # Every file under docs/verification/ carries that warning; this one is package data with
    # the same property, and `verify_replay` is the mechanism the note points at.
    path = tmp_path / "golden_variants.json"
    save_variants(a_variant_set(a_plan()), path)

    payload = json.loads(path.read_text())

    assert "hand-edited" in payload["note"]
    assert "ADR-0004 §9" in payload["note"]


def test_coverage_is_an_equality_over_the_golden_sets_ids():
    # A stale plan for a question the set no longer holds is as much a reason to re-resolve as a
    # missing one, so a subset check would pass over a file describing a different golden set.
    golden = load_golden_set()
    complete = a_variant_set(*[a_plan(question_id=row.id) for row in golden])

    assert complete.covers(golden)
    assert not a_variant_set(a_plan(question_id="S1")).covers(golden)

    extra = a_variant_set(
        *[a_plan(question_id=row.id) for row in golden], a_plan(question_id="GONE")
    )
    assert not extra.covers(golden)


def test_a_missing_plan_names_the_stage_that_produces_it():
    with pytest.raises(MissingPlan, match="--stage resolve"):
        a_variant_set(a_plan()).plan("S1")


# --- the planner's own variance, reported separately ---------------------------------


def test_identical_repeats_report_agreement():
    result = agreement("E1", [("a", "b"), ("a", "b"), ("a", "b")])

    assert result.identical is True
    assert result.modal_share == 1.0
    assert result.repeats == 3


def test_a_differing_repeat_is_visible_with_its_share():
    result = agreement("E1", [("a", "b"), ("a", "b"), ("a", "c")])

    assert result.identical is False
    assert result.distinct[0] == (("a", "b"), 2)
    assert result.modal_share == pytest.approx(2 / 3)


def test_reordered_sub_queries_are_not_called_identical():
    # Order matters: variants are retrieved in order and RRF breaks ties on chunk id, so two
    # orderings are not guaranteed to fuse the same way. Calling them identical would overstate
    # the planner's stability in the direction that flatters the harness.
    result = agreement("E1", [("a", "b"), ("b", "a")])

    assert result.identical is False


def test_agreement_over_no_repeats_reports_no_share_rather_than_dividing_by_zero():
    result = PlannerAgreement(question_id="E1", repeats=0, distinct=())

    assert result.modal_share == 0.0


def test_the_resolve_pass_goes_through_the_real_translate_so_its_call_is_logged(caplog):
    # **The hole this closes.** `resolve_plan` used to call the model itself and then replay
    # the reply, so the *paid* call happened outside `query_translation.translate` — the only
    # place a `query_translation` event is emitted. The log then held 322 of those lines and
    # **zero** with token counts, so `latency.translation_cost` refused to compute ADR-0005's
    # budget from it. Asserted on the event, because the event is what was missing.
    import logging

    from finbrief.evaluation.variants import RecordingPlanner

    inner = ReplayPlanner(reply=REPLY)
    recorder = RecordingPlanner(inner=inner)

    with caplog.at_level(logging.INFO, logger="finbrief.retrieval.query_translation"):
        plan = resolve_plan(
            "E1",
            QUESTION,
            model=recorder,
            max_sub_queries=3,
            planner_model="openai/gpt-4o-mini",
        )

    assert inner.invocations == 1, "the real planner was not reached"
    assert plan.reply == REPLY, "the reply was not recorded"
    emitted = [r for r in caplog.records if getattr(r, "event", None) == "query_translation"]
    assert len(emitted) == 1, (
        "the paid planner call emitted no query_translation event, so its latency and tokens "
        "are unmeasurable and ADR-0005's cost half cannot be computed"
    )


def test_the_recorder_returns_what_the_inner_model_returned():
    # It must not alter the reply on the way past: the persisted text is what the arms replay.
    from finbrief.evaluation.variants import RecordingPlanner

    recorder = RecordingPlanner(inner=ReplayPlanner(reply="- one\n- two\n"))

    reply = recorder.invoke([HumanMessage("anything")])

    assert reply.text == "- one\n- two\n"
    assert recorder.reply == "- one\n- two\n"


def test_a_plan_resolved_for_a_different_wording_is_refused_at_load():
    """`verify_replay` re-parses against the persisted wording, so it cannot see an edited row.

    The gap (code review of #11): `verify_replay` compares a reply against `plan.question` — the
    text stored beside it — and `covers` is an equality on *ids*, so nothing bound a plan to the
    row it is replayed for. Edit a golden row and re-run against a warm cache, and both
    `+translation` arms replay a reply produced for the previous question.
    """
    from dataclasses import replace

    from finbrief.evaluation.loader import load_golden_set
    from finbrief.evaluation.variants import ReplayMismatch, verify_questions

    golden = load_golden_set()
    row = golden.questions[0]
    variant_set = VariantSet(
        plans={
            row.id: ResolvedPlan(
                question_id=row.id,
                question=row.question,
                reply="a reply",
                variants=(row.question,),
                planner_model="openai/gpt-4o-mini",
                max_sub_queries=3,
            )
        },
        planner_model="openai/gpt-4o-mini",
        max_sub_queries=3,
        golden_set_schema_version=1,
    )

    verify_questions(variant_set, (row,))

    with pytest.raises(ReplayMismatch, match="now asks"):
        verify_questions(variant_set, (replace(row, question=row.question + " Why?"),))


def test_a_row_with_no_plan_is_left_to_the_missing_plan_error():
    """This check is about *drift*, not about coverage — `VariantSet.plan` owns the absence."""
    from finbrief.evaluation.loader import load_golden_set
    from finbrief.evaluation.variants import verify_questions

    row = load_golden_set().questions[0]
    empty = VariantSet(
        plans={},
        planner_model="openai/gpt-4o-mini",
        max_sub_queries=3,
        golden_set_schema_version=1,
    )

    verify_questions(empty, (row,))
