"""The artifact's own rules: the partial banner, the absence marker, the not-comparable cell.

Seam: `render_report` takes measurements and returns markdown, so every rule the artifact
enforces on itself is testable without a run. These are the checks that stop the *evidence*
from over-claiming — which is the same job `tests/test_security_report.py` does for the
security suite's artifact.
"""

from __future__ import annotations

from finbrief.evaluation.arms import ABLATION_ARMS, SCORED_ARMS, SHIPPING_DEFAULT
from finbrief.evaluation.cache import CacheStats
from finbrief.evaluation.loader import Bucket, load_golden_set
from finbrief.evaluation.metrics import RowScore, Summary
from finbrief.evaluation.pipeline import Cell
from finbrief.evaluation.report import (
    ALL_STAGES,
    Provenance,
    _relevancy_cell,
    cell,
    headline_section,
    is_partial,
    missing_coverage,
    recall_ceiling_note,
    render_report,
    trivial_rows_table,
)
from finbrief.retrieval.retrieve import Retrieval


def a_provenance(**overrides) -> Provenance:
    fields = {
        "generated_at": "2026-07-29 12:00 UTC",
        "judge_model": "openai/gpt-4.1-mini",
        "chat_model": "openai/gpt-4o-mini",
        "embedding_model": "openai/text-embedding-3-small",
        "planner_model": "openai/gpt-4o-mini",
        "ragas_version": "0.4.3",
        "k": 5,
        "collection_ingest_run": "2026-07-27 08:51 UTC, 5842 chunks",
        "collection_fingerprint": "c986299c02043ee8deadbeef",
        "golden_set_rows": 28,
        "golden_set_total": 28,
        "stages": ALL_STAGES,
        "cache": {"judge": CacheStats(hits=3, misses=7)},
    }
    return Provenance(**{**fields, **overrides})


def a_cell(
    *,
    bucket: Bucket,
    arm: str,
    row: str,
    recall_trivial: bool = False,
    chunk_recall_ceiling: float = 0.83,
    **judged,
) -> Cell:
    score = RowScore(
        question_id=row,
        bucket=bucket,
        arm=arm,
        k=5,
        retrieved_chunk_ids=("a:Item 1A:0",),
        target_hits=1,
        precision_at_k=0.2,
        chunk_recall=0.5,
        chunk_recall_ceiling=chunk_recall_ceiling,
        section_recall=1.0,
        section_precision=1.0,
        filer_precision=1.0,
        target_rank=1,
        leaked_chunk_ids=(),
        recall_trivial=recall_trivial,
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


def some_cells() -> list[Cell]:
    cells = []
    for arm in SCORED_ARMS:
        for index, bucket in enumerate(Bucket):
            cells.append(
                a_cell(
                    bucket=bucket,
                    arm=arm.name,
                    row=f"{arm.name}-{index}",
                    faithfulness=0.9,
                    answer_relevancy=0.0 if bucket is Bucket.TOOL_AUGMENTED else 0.85,
                    context_precision=0.5,
                    context_recall=0.4,
                )
            )
    return cells


def test_a_complete_run_carries_no_partial_banner():
    markdown = render_report(
        provenance=a_provenance(),
        golden=load_golden_set(),
        cells=some_cells(),
        arms=SCORED_ARMS,
    )

    assert "PARTIAL RUN" not in markdown
    assert not is_partial(a_provenance())


def test_a_staged_run_names_the_stages_it_skipped_above_the_numbers():
    # The `--gate-only` precedent: a partial run must not be committed as a full one, and the
    # signal has to be above the tables rather than inferable from a missing column.
    provenance = a_provenance(stages=("retrieve", "report"))

    markdown = render_report(
        provenance=provenance,
        golden=load_golden_set(),
        cells=some_cells(),
        arms=SCORED_ARMS,
    )

    assert is_partial(provenance)
    banner = markdown.split("## Per-bucket")[0]
    assert "PARTIAL RUN" in banner
    assert "resolve" in banner and "answer" in banner and "judge" in banner


def test_a_row_subset_is_flagged_partial_and_names_what_it_scored():
    """The `--rows` half of the banner gap, as a regression test.

    `--rows S1,T2` is the documented smoke mode and it wrote the committed artifact with every
    hypothesis, both pre-registered decisions and the headline block rendered over two questions
    and **no banner** — `is_partial` looked only at stages. `_resolve` already refused to let a
    partial run overwrite `golden_variants.json`; the artifact had no equivalent.
    """
    provenance = a_provenance(golden_set_rows=2, golden_set_total=28)

    markdown = render_report(
        provenance=provenance,
        golden=load_golden_set(),
        cells=some_cells(),
        arms=SCORED_ARMS,
    )

    assert is_partial(provenance)
    banner = markdown.split("## Per-bucket")[0]
    assert "PARTIAL RUN" in banner
    assert "2 of the golden set's 28 rows" in banner


def test_skipping_the_ablations_is_flagged_partial_and_names_the_absent_channels():
    # `--no-ablations` dropped ADR-0004 §7's ablation and ADR-0005 §2's falsification channel
    # and left no trace in the file at all: the section is simply not emitted, so a reader
    # cannot tell an unrun channel from one that was never asked for.
    provenance = a_provenance(ablations_run=False)

    markdown = render_report(
        provenance=provenance,
        golden=load_golden_set(),
        cells=some_cells(),
        arms=SCORED_ARMS,
        ablations=(),
    )

    assert is_partial(provenance)
    banner = markdown.split("## Per-bucket")[0]
    assert "PARTIAL RUN" in banner
    assert "ADR-0005 §2" in banner


def test_a_full_row_count_with_every_stage_and_the_ablations_is_not_partial():
    # The equality that keeps the three flags honest in the other direction: a complete run must
    # not be banner-flagged, or the banner stops meaning anything.
    assert not is_partial(a_provenance())
    assert missing_coverage(a_provenance()) == ()


def test_an_absent_metric_renders_as_a_dash_and_never_as_zero():
    # "We cannot say" and "it scored zero" are different claims, and a table of numbers is where
    # the difference disappears if the renderer does not keep it.
    assert cell(Summary.of([None, None])).startswith("—")
    assert "n=0 of 2" in cell(Summary.of([None, None]))
    assert "0.000" not in cell(Summary.of([None, None]))


def test_a_partially_absent_metric_says_how_many_rows_it_averaged():
    rendered = cell(Summary.of([1.0, None, 1.0]))

    assert "n=2" in rendered
    assert "1 absent" in rendered


def test_every_mean_is_rendered_with_its_spread():
    rendered = cell(Summary.of([0.2, 0.8]))

    assert "0.500" in rendered
    assert "[0.200–0.800]" in rendered
    assert "n=2" in rendered


def test_the_tool_augmented_relevancy_cell_is_not_a_number():
    # #11's addition 3. A chain answer that correctly says the filings carry no current share
    # price is scored ≈0 for being right, so the cell prints a noncommittal count instead of a
    # mean that would invite comparison with a semantic bucket's 0.9.
    markdown = render_report(
        provenance=a_provenance(),
        golden=load_golden_set(),
        cells=some_cells(),
        arms=SCORED_ARMS,
    )

    # Scoped to the RAGAs table: `| tool-augmented |` also opens rows in the free-metrics table,
    # where a precision number is perfectly comparable and should stay one.
    ragas_table = markdown.split("## RAGAs")[1].split("## ")[0]
    tool_rows = [
        line for line in ragas_table.split("\n") if line.startswith("| tool-augmented |")
    ]
    assert tool_rows
    assert all("not comparable" in row for row in tool_rows)
    assert all("noncommittal" in row for row in tool_rows)


def test_the_fenced_column_is_labelled_in_the_table_not_only_underneath():
    markdown = render_report(
        provenance=a_provenance(),
        golden=load_golden_set(),
        cells=some_cells(),
        arms=SCORED_ARMS,
    )

    header = next(line for line in markdown.split("\n") if "faithfulness" in line)
    assert "answer relevancy ⚠" in header


def test_the_artifact_says_it_is_generated_and_names_its_generator():
    markdown = render_report(
        provenance=a_provenance(),
        golden=load_golden_set(),
        cells=some_cells(),
        arms=SCORED_ARMS,
    )

    assert "GENERATED FILE" in markdown
    assert "scripts/evaluate.py" in markdown


def test_the_provenance_names_all_three_model_roles():
    # Three roles, three prices: a reader has to be able to tell which model scored the run from
    # which model produced the answers it scored.
    markdown = render_report(
        provenance=a_provenance(),
        golden=load_golden_set(),
        cells=some_cells(),
        arms=SCORED_ARMS,
    )

    assert "openai/gpt-4.1-mini" in markdown
    assert "openai/gpt-4o-mini" in markdown
    assert "ragas 0.4.3" in markdown


def test_the_ablation_table_is_rendered_separately_when_asked_for():
    markdown = render_report(
        provenance=a_provenance(),
        golden=load_golden_set(),
        cells=some_cells(),
        arms=SCORED_ARMS,
        ablations=ABLATION_ARMS,
    )

    assert "## Ablations" in markdown
    assert "ADR-0005 §2" in markdown


def test_the_headline_block_omits_the_fenced_metric():
    # Quoting a figure that moves between runs as a headline number is the thing the fence is
    # for.
    block = headline_section(
        some_cells(), default_arm=SHIPPING_DEFAULT, judge_calls=1344, cache_replayed=0
    )

    assert "faithfulness" in block
    assert "context precision" in block
    assert "1344" in block
    assert "deliberately absent" in block


def test_an_unmeasured_deferral_says_so_rather_than_being_omitted():
    # "An unscored criterion reads as a passed one" — #9's comment on the AC-1 pairing names
    # this exactly. All four are listed whether they ran or not, and an absent one is words
    # rather than a gap a reader has to notice.
    from finbrief.evaluation.report import DEFERRALS, deferrals_section

    rendered = deferrals_section()

    assert len(DEFERRALS) == 4
    for name, _, _ in DEFERRALS:
        assert name in rendered
    assert rendered.count("**not measured by this run**") == 4
    assert "too small to publish" in rendered


def test_the_bracket_deferral_points_at_the_page_that_made_it_readable():
    """AC-8's other half, which nothing verified (code review of #14).

    T13 asks that this note point at `app/pages/1_Analytics.py` as the surface where the
    app-only `citation_markers` instrument becomes readable. The committed
    `docs/verification/evaluation.md` does not carry the pointer yet and correctly cannot —
    every
    file under `docs/verification/` is generated, and this section is rendered by a paid
    `scripts/evaluate.py` run. So the artifact half of that criterion is verifiable only here,
    against the code that will write it, and it shipped with no test at all: prose asserting a
    pointer, which is exactly the shape that cannot fail.

    It must also keep saying what the pointer does **not** do. The rate the page shows is
    observational over whoever used the app, not the controlled measurement over a stratified
    set
    this row asks for, so naming the surface may not read as closing the deferral.
    """
    from finbrief.evaluation.report import deferrals_section

    rendered = deferrals_section()

    assert "app/pages/1_Analytics.py" in rendered, "the surface, by path"
    assert "observational over logged sessions" in rendered
    assert "does not close this deferral" in rendered
    # And the row is still one of the four reported as unmeasured by the run itself: a surface
    # that can read the instrument is not a run that measured it.
    assert rendered.count("**not measured by this run**") == 4


def test_a_measured_deferral_replaces_the_not_measured_wording():
    from finbrief.evaluation.report import DEFERRALS, deferrals_section

    name = DEFERRALS[0][0]

    rendered = deferrals_section({name: "0 verbatim of 14 searches"})

    assert "0 verbatim of 14 searches" in rendered
    assert rendered.count("**not measured by this run**") == 3


# --- the two claims the artifact used to make about itself --------------------------------


def test_the_trivial_rows_clause_now_has_a_table_under_it():
    """ "(reported separately)" was printed into the artifact and nothing reported them.

    `metrics.py`'s docstring says "a row that cannot miss is reported separately, not excluded
    and not folded in", and `grep -c recall_trivial report.py` found exactly one occurrence: the
    prose making the claim (code review of #11).
    """
    cells = some_cells() + [
        a_cell(
            bucket=Bucket.SEMANTIC,
            arm=arm.name,
            row="S4",
            recall_trivial=True,
            context_precision=0.5,
        )
        for arm in SCORED_ARMS
    ]

    markdown = render_report(
        provenance=a_provenance(), golden=load_golden_set(), cells=cells, arms=SCORED_ARMS
    )

    assert "reported separately in their own table below" in markdown
    assert "### The `recall_trivial` rows the columns above dropped" in markdown
    assert "**S4**" in markdown


def test_a_run_with_no_trivial_rows_says_so_rather_than_printing_an_empty_table():
    body = trivial_rows_table(some_cells(), SCORED_ARMS)

    assert "dropped nothing" in body
    assert "|---|" not in body


def test_the_recall_ceiling_is_rendered_beside_the_column_it_qualifies():
    """`RowScore.chunk_recall_ceiling` was computed per cell and rendered nowhere.

    `metrics.py`: "recall is reported against the real denominator, **with its ceiling beside
    it** … reporting the ceiling says how much headroom the retriever actually had."
    """
    note = recall_ceiling_note(some_cells(), k=5)

    assert "0.830" in note, "the mean ceiling, from the cells' own field"
    assert "cannot* reach 1.000" in note


def test_a_corpus_with_no_capped_row_says_that_instead_of_naming_none():
    cells = [
        a_cell(bucket=Bucket.SEMANTIC, arm="vector", row="S1", chunk_recall_ceiling=1.0),
    ]

    note = recall_ceiling_note(cells, k=5)

    assert "No row targets more chunks" in note


# --- the noncommittal flag is keyed on the score, not on the bucket -----------------------


def test_a_noncommittal_cell_outside_tool_augmented_is_flagged_and_excluded():
    """The bucket was standing in for the condition, and the condition fires elsewhere.

    Nine ≈0 relevancy cells sat in the first artifact's multi-hop bucket and were averaged into
    its four published means — including correct, on-topic answers scored 0.0 for declining to
    commit (code review of #11).
    """
    rows = [
        a_cell(bucket=Bucket.MULTI_HOP, arm="hybrid", row="M1", answer_relevancy=0.0),
        a_cell(bucket=Bucket.MULTI_HOP, arm="hybrid", row="M2", answer_relevancy=0.9),
        a_cell(bucket=Bucket.MULTI_HOP, arm="hybrid", row="M3", answer_relevancy=0.8),
    ]

    rendered = _relevancy_cell(Bucket.MULTI_HOP, rows)

    assert "(+1 noncommittal, excluded)" in rendered
    assert "0.850" in rendered, "the mean is over the two that committed, not over three"


def test_an_absent_relevancy_score_is_not_counted_as_noncommittal():
    """`(score or 0.0)` turned the honest `None` into a flag — a fabricated zero.

    `judge.score` returns `None` for a row ragas could not score, and every other summariser
    here routes that through `Summary.absent`.
    """
    rows = [
        a_cell(bucket=Bucket.MULTI_HOP, arm="hybrid", row="M1", answer_relevancy=None),
        a_cell(bucket=Bucket.MULTI_HOP, arm="hybrid", row="M2", answer_relevancy=0.9),
    ]

    rendered = _relevancy_cell(Bucket.MULTI_HOP, rows)

    assert "noncommittal" not in rendered
    assert "1 absent" in rendered


def test_the_tool_augmented_cell_counts_flags_and_not_absences():
    rows = [
        a_cell(bucket=Bucket.TOOL_AUGMENTED, arm="hybrid", row="T1", answer_relevancy=0.0),
        a_cell(bucket=Bucket.TOOL_AUGMENTED, arm="hybrid", row="T2", answer_relevancy=None),
    ]

    assert _relevancy_cell(Bucket.TOOL_AUGMENTED, rows) == "not comparable — 1/2 noncommittal"


# --- the finding sections, which no test rendered ------------------------------------------


def a_cost(**overrides):
    from finbrief.evaluation.latency import TranslationCost

    fields = {
        "planner_p50_ms": 900.0,
        "planner_samples": 8,
        "retrieval_translated_p50_ms": 1400.0,
        "retrieval_untranslated_p50_ms": 1000.0,
        "translated_samples": 14,
        "untranslated_samples": 14,
        "planner_disabled_lines": 14,
        "refusal_lines_kept": 1,
        "budget_ms": 1500.0,
    }
    return TranslationCost(**{**fields, **overrides})


def test_the_latency_verdict_word_follows_the_budget():
    from finbrief.evaluation.report import latency_section

    within = latency_section(a_cost(planner_p50_ms=900.0), [])
    over = latency_section(a_cost(planner_p50_ms=2900.0), [])

    assert "**Verdict: within budget.**" in within
    assert "**Verdict: **over budget**.**" in over


def test_a_replayed_window_discloses_that_these_figures_are_not_a_fresh_timing():
    """The disclosure is gated on `getattr(window, "replayed", False)`, so a renamed field would
    make it silently vanish — and the section would then imply this run timed something."""
    from finbrief.evaluation.latency import Window
    from finbrief.evaluation.report import latency_section

    fresh = Window(path="events.jsonl", offset=0, recorded_at="2026-07-29 12:00 UTC")
    replayed = Window(
        path="events.jsonl", offset=0, recorded_at="2026-07-29 12:00 UTC", replayed=True
    )

    assert "replayed" not in latency_section(a_cost(), [], window=fresh).split("\n")[0]
    body = latency_section(a_cost(), [], window=replayed)
    assert "**These figures are the measuring run of 2026-07-29 12:00 UTC's**, replayed" in body


def test_the_hypothesis_and_decision_sections_render_from_a_run(monkeypatch):
    """Neither was reachable from `render_report`, which takes `sections` as pre-rendered
    strings
    and every test passed none — so ~400 lines of the artifact's *findings* were unrendered in
    CI.
    """
    from finbrief.evaluation.hypotheses import (
        as_report_entries,
        falsification_clause,
        outcomes,
        reexamination_trigger,
    )
    from finbrief.evaluation.report import decisions_section, hypothesis_section

    cells = some_cells()
    rendered = hypothesis_section(as_report_entries(outcomes(cells)))

    assert "H1" in rendered and "H6" in rendered

    decisions = decisions_section(
        falsification_clause(cells),
        reexamination_trigger(cells),
        default_arm_label=SHIPPING_DEFAULT.label,
    )

    assert SHIPPING_DEFAULT.label in decisions


def test_the_power_audit_renders_its_table_body():
    from finbrief.evaluation.hypotheses import outcomes
    from finbrief.evaluation.report import power_audit_section

    labelled = tuple(
        (outcome.prediction.id, outcome.comparison) for outcome in outcomes(some_cells())
    )

    rendered = power_audit_section(labelled, bucket_floor=6)

    assert "| verdict | cells | how to read it |" in rendered
    assert f"— the {len(labelled)} cells of" in rendered
    assert f"of {len(labelled)} cells carry a measurement" in rendered
    # The audit's own finding, derived rather than asserted: at ADR-0002's bucket floor the
    # exact test tolerates zero minority signs, so the design can resolve only a near-unanimous
    # effect.
    assert "6" in rendered


def test_the_provenance_table_derives_the_edgar_claim_rather_than_printing_it():
    """`golden` was threaded into `_provenance_table` only to be `del`-ed one line later.

    The row said "verified against EDGAR" as a literal, so the artifact asserted the thing whose
    evidence it was handed and never read (code review of #11). `loader.py` raises on an
    unverified set, which is why deriving it costs nothing.
    """
    from dataclasses import replace

    golden = load_golden_set()
    common = {
        "provenance": a_provenance(),
        "cells": some_cells(),
        "arms": SCORED_ARMS,
    }

    verified = render_report(golden=golden, **common)
    unverified = render_report(golden=replace(golden, verified_against_edgar=False), **common)

    assert "rows, verified against EDGAR" in verified
    assert "**not verified against EDGAR**" in unverified
    assert f"golden set schema | v{golden.schema_version}" in verified
