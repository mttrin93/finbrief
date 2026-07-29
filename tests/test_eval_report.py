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
    cell,
    headline_section,
    is_partial,
    render_report,
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
        "stages": ALL_STAGES,
        "cache": {"judge": CacheStats(hits=3, misses=7)},
    }
    return Provenance(**{**fields, **overrides})


def a_cell(*, bucket: Bucket, arm: str, row: str, **judged) -> Cell:
    score = RowScore(
        question_id=row,
        bucket=bucket,
        arm=arm,
        k=5,
        retrieved_chunk_ids=("a:Item 1A:0",),
        target_hits=1,
        precision_at_k=0.2,
        chunk_recall=0.5,
        chunk_recall_ceiling=0.83,
        section_recall=1.0,
        filer_precision=1.0,
        target_rank=1,
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
