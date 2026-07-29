"""The arm matrix and the typed golden-set reader.

Seam: everything the harness reads before it spends anything. The invariant worth the most here
is the one that binds the *measured* default to the *shipped* default — a headline number
reported against a configuration the app does not run would be the ADR-0003 failure ("a number
is never reported against a configuration nobody selected") arriving from the eval side.
"""

from __future__ import annotations

import json

import pytest

from finbrief.config import (
    DEFAULT_STRATEGY,
    DEFAULT_TRANSLATION_ENABLED,
    RetrievalStrategy,
    Settings,
)
from finbrief.evaluation.arms import (
    ABLATION_ARMS,
    ALL_ARMS,
    SCORED_ARMS,
    SHIPPED_MAX_SUB_QUERIES,
    SHIPPING_DEFAULT,
    STRATEGY_CONTRAST,
    TRANSLATION_CONTRAST,
    arm,
    shipping_default_matches_config,
)
from finbrief.evaluation.loader import (
    Bucket,
    GoldenSet,
    UnverifiedGoldenSet,
    load_golden_set,
)
from finbrief.ingestion.model import Section


@pytest.fixture(scope="module")
def golden() -> GoldenSet:
    return load_golden_set()


# --- the arm matrix -------------------------------------------------------------------


def test_the_measured_default_is_the_shipped_default():
    # ADR-0005 pre-registers `hybrid + translation`, `config.py` ships it, and this harness
    # reports the headline generation numbers against it. If any of the three moves without the
    # others, the artifact describes a configuration nobody runs.
    assert shipping_default_matches_config()
    assert SHIPPING_DEFAULT.strategy is DEFAULT_STRATEGY
    assert SHIPPING_DEFAULT.translate is DEFAULT_TRANSLATION_ENABLED


def test_the_shipping_default_runs_at_the_configured_sub_query_cap():
    # The other half of the same fact, and the one a constant in `arms.py` could get wrong:
    # `Settings`' own default is what the app plans with, so an arm claiming to be the shipping
    # default has to plan with it too. An equality on the number, not a bound.
    settings = Settings.from_env({"OPENROUTER_API_KEY": "test-key"})

    assert settings.max_sub_queries == SHIPPED_MAX_SUB_QUERIES
    assert SHIPPING_DEFAULT.max_sub_queries == settings.max_sub_queries


def test_the_matrix_is_exactly_adr_0002s_four_cells():
    # Two axes, two values each: the four combinations and nothing else. A fifth scored arm
    # would be a number in the A/B table that ADR-0002's matrix does not describe.
    assert len(SCORED_ARMS) == 4
    assert {(a.strategy, a.translate) for a in SCORED_ARMS} == {
        (RetrievalStrategy.VECTOR, False),
        (RetrievalStrategy.VECTOR, True),
        (RetrievalStrategy.HYBRID, False),
        (RetrievalStrategy.HYBRID, True),
    }


def test_every_arm_has_a_distinct_name_and_label():
    # The name is a cache-key component and the label is a table heading; a duplicate of either
    # silently merges two configurations' results.
    assert len({a.name for a in ALL_ARMS}) == len(ALL_ARMS)
    assert len({a.label for a in ALL_ARMS}) == len(ALL_ARMS)


def test_both_ablation_arms_translate_with_the_planner_off():
    # ADR-0005 §2's channel is `FINBRIEF_MAX_SUB_QUERIES=0` *with translation on*: the ticker
    # form is still added, so the cell isolates what normalisation contributes. Translation off
    # would make it a `−translation` baseline and answer a different question.
    for ablation in ABLATION_ARMS:
        assert ablation.translate is True
        assert ablation.max_sub_queries == 0
        assert ablation.plans is False


def test_every_ablation_arm_is_deterministic_and_so_are_both_baselines():
    # ADR-0004 §9's table, as code. These four reach no sampled step, which is what lets the
    # falsification channel be exact rather than exact-up-to-sampling.
    deterministic = {a.name for a in ALL_ARMS if a.deterministic}

    assert deterministic == {
        "vector",
        "hybrid",
        "vector+normalisation",
        "hybrid+normalisation",
    }


def test_only_the_two_translated_arms_reach_the_planner():
    assert {a.name for a in ALL_ARMS if a.plans} == {"vector+translation", "hybrid+translation"}


def test_generation_metrics_run_on_the_matrix_and_not_on_the_ablations():
    # The ablations exist to attribute a *retrieval* result; faithfulness over an answer is not
    # evidence about which candidate list found a chunk.
    assert all(a.judged for a in SCORED_ARMS)
    assert not any(a.judged for a in ABLATION_ARMS)


def test_the_translation_contrast_holds_strategy_fixed():
    # ADR-0005's falsification clause is "translation is worse on both precision and recall
    # within a bucket". Its operands must differ in translation *only*, or the clause fires on a
    # strategy effect and drops the shipping default for the wrong reason.
    for baseline, translated in TRANSLATION_CONTRAST:
        assert baseline.strategy is translated.strategy
        assert baseline.translate is False and translated.translate is True


def test_the_strategy_contrast_holds_translation_fixed():
    # ADR-0005 §4's re-examination trigger is `hybrid − vector` **at equal translation**.
    for vector, hybrid in STRATEGY_CONTRAST:
        assert vector.translate is hybrid.translate
        assert vector.strategy is RetrievalStrategy.VECTOR
        assert hybrid.strategy is RetrievalStrategy.HYBRID


def test_an_unknown_arm_name_raises_and_lists_the_real_ones():
    with pytest.raises(KeyError) as caught:
        arm("hybrid+reranking")

    assert "vector+translation" in str(caught.value)


# --- the typed reader ----------------------------------------------------------------


def test_the_committed_set_loads_with_twenty_eight_rows(golden):
    assert len(golden) == 28
    assert golden.verified_against_edgar is True


def test_every_bucket_is_represented_in_declaration_order(golden):
    by_bucket = golden.by_bucket()

    assert list(by_bucket) == list(Bucket)
    assert all(len(rows) == 7 for rows in by_bucket.values())


def test_a_rows_target_chunk_ids_are_the_union_of_its_grounding(golden):
    row = golden.row("S1")

    assert row.target_chunk_ids == frozenset(row.grounding[0].chunk_ids)
    assert len(row.target_chunk_ids) == 6


def test_target_sections_are_keyed_the_way_section_chunk_counts_are(golden):
    # The join the recall denominator depends on: `Grounding.key` and the authored
    # `section_chunk_counts` keys have to be the same string, or the denominator silently
    # goes missing and recall is computed against nothing.
    for row in golden:
        assert set(row.target_sections) == set(row.section_chunk_counts)


def test_a_multi_filer_row_reports_every_filer_it_grounds_in(golden):
    row = golden.row("M3")

    # An equality rather than `> 1`: M3 is *the* multi-filer row and 2, 3 and 7 all satisfy a
    # bound, so a row that lost a filer would still pass one.
    assert row.tickers == {"F", "GM"}
    assert row.tickers == {entry.ticker for entry in row.grounding}


def test_non_trivial_sections_exclude_the_ones_a_k5_retrieval_cannot_miss(golden):
    # M2 is the partial case: two of its sections hold <= 4 chunks and the row is not
    # `recall_trivial` as a whole. Reporting its recall over all targets would credit the
    # retriever for sections it could not have missed.
    row = golden.row("M2")

    assert row.recall_trivial is False
    assert len(row.recall_trivial_sections) == 2
    assert set(row.non_trivial_sections) == set(row.target_sections) - set(
        row.recall_trivial_sections
    )


def test_the_wholly_trivial_row_has_no_non_trivial_section(golden):
    row = golden.row("S4")

    assert row.recall_trivial is True
    assert row.non_trivial_sections == ()


def test_a_tool_augmented_row_carries_its_tool_expectation(golden):
    row = golden.row("T2")

    assert row.tool_expectation is not None
    assert row.tool_expectation.name == "calculate_ratios"
    assert row.tool_expectation.peer_set


def test_a_retrieval_only_row_carries_none(golden):
    assert golden.row("S1").tool_expectation is None


def test_a_tool_without_peers_reports_no_peer_set_rather_than_an_empty_one(golden):
    # `get_stock_data` makes no peer comparison, so its `peer_set` is `None`. An empty tuple
    # would read as "compared against nobody", which is a claim about a comparison that never
    # happened — the same absence-is-not-zero rule `finance/quotes.py` applies to a figure.
    quote_row = golden.row("T1")
    ratio_row = golden.row("T2")

    assert quote_row.tool_expectation.peer_set is None
    assert quote_row.tool_expectation.n is None
    assert ratio_row.tool_expectation.peer_set == ("TSLA", "GM")
    assert ratio_row.tool_expectation.n == 2


def test_the_sections_are_read_as_section_members(golden):
    # A `Section` and not a string, so a row that named a section outside ADR-0007's four would
    # raise at load rather than compare unequal to every chunk's metadata at scoring time.
    assert golden.row("S1").grounding[0].section is Section.RISK_FACTORS


def test_an_unverified_set_is_refused(tmp_path, golden):
    # ADR-0002's flag is what stops a set being cited before the hand-verification pass. A
    # harness that read it and continued would make it decoration.
    payload = json.loads(load_golden_set.__globals__["GOLDEN_SET_PATH"].read_text())
    payload["provenance"]["verified_against_edgar"] = False
    draft = tmp_path / "draft.json"
    draft.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UnverifiedGoldenSet):
        load_golden_set(draft)

    # And loadable on purpose, for a set actually under authoring.
    assert len(load_golden_set(draft, require_verified=False)) == 28
