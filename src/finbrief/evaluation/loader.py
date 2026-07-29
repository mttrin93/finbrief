"""`golden_set.json` as typed objects — the one reader of the one hand-authored artifact.

`tests/test_golden_set.py` owns the schema and the ADR-0002/ADR-0004 sampling constraints, and
this module deliberately does not repeat them: it reads a set those tests have already declared
well-formed and gives the harness fields instead of dictionary lookups. One reader for the same
reason `observability/events.py` is the one reader of the log — a renamed field then breaks
loudly in one place rather than reading back as `None` and averaging as nothing.

**It refuses to load an unverified set by default, and that is the point of the flag.** ADR-0002
decision 1 hand-verifies every row against the primary source, and the `verified_against_edgar`
flag exists so that a set cannot be *cited* before that pass happens. A harness that read the
flag and shrugged would make the flag decoration; `load_golden_set` raises, so the refusal is at
the door rather than in a footnote under a published number.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from finbrief.ingestion.model import Section

#: The committed set, beside this module because it is package data and not a fixture.
GOLDEN_SET_PATH = Path(__file__).with_name("golden_set.json")


class UnverifiedGoldenSet(RuntimeError):
    """The set has not completed ADR-0002's hand-verification pass, so it may not be scored."""


class Bucket(StrEnum):
    """ADR-0002 decision 2's four strata. Guardrail/injection cases are excluded by decision.

    A `StrEnum` so a bucket can be a dict key, a report heading and a JSON value without a
    conversion at each boundary — and so an unknown stratum raises here rather than appearing as
    a fifth column nobody predicted.
    """

    SEMANTIC = "semantic"
    EXACT_IDENTIFIER = "exact-identifier"
    TOOL_AUGMENTED = "tool-augmented"
    MULTI_HOP = "multi-hop"


@dataclass(frozen=True, slots=True)
class Grounding:
    """One `(ticker, section)` a reference draws on — the filings half, per ADR-0002.

    `chunk_ids` is the recall numerator's target set and `passage` is the verbatim text the
    reference was authored from. Both are read from the collection at authoring time and
    hand-verified against EDGAR; nothing here re-checks them, because a hermetic test cannot
    open the ingested collection (CLAUDE.md).
    """

    ticker: str
    section: Section
    fiscal_year: int
    accession: str
    chunk_ids: tuple[str, ...]
    passage: str

    @property
    def key(self) -> str:
        """`TSLA Item 1A` — how `section_chunk_counts` names this target."""
        return f"{self.ticker} {self.section.value}"


@dataclass(frozen=True, slots=True)
class ToolExpectation:
    """The additional non-retrieval tool a `tool-augmented` row needs (ADR-0002 amendment).

    Scored by the tool-calling eval, pass/fail, and **never by RAGAs** — that split is the
    amendment's whole subject. `search_filings` is deliberately absent: every row in every
    bucket needs it, so naming it 28 times would be noise.
    """

    name: str
    args: Mapping[str, Any]
    #: `None` where a peer set does not apply, **not** an empty tuple — only `calculate_ratios`
    #: has peers, and `get_stock_data` having "no peers" would read as a comparison against
    #: nobody rather than as a tool that makes no comparison. The same honest-absence rule
    #: `finance/quotes.py` applies to a figure that was not reported.
    peer_set: tuple[str, ...] | None
    n: int | None


@dataclass(frozen=True, slots=True)
class GoldenQuestion:
    """One row: a question, its source-separated reference, and the flags T10 must honour."""

    id: str
    bucket: Bucket
    question: str
    reference: str
    grounding: tuple[Grounding, ...]
    labels: tuple[str, ...]
    #: Chunks that are correct lexical matches and wrong grounding (ADR-0004 §11). Reported as
    #: mention-leakage precision *beside* the bucket mean rather than inside it.
    known_false_positives: frozenset[str]
    #: `TSLA Item 1A` -> how many chunks the collection holds for it. The recall denominator.
    section_chunk_counts: Mapping[str, int]
    #: True when every target section holds <= 4 chunks, so a `k=5` retrieval cannot miss it.
    #: Reported separately: a question that cannot miss must not inflate the recall column.
    recall_trivial: bool
    #: The subset of targets that are trivial, for a row where only some are. A multi-section
    #: row is not all-or-nothing, and the whole-row flag alone cannot express it.
    recall_trivial_sections: tuple[str, ...]
    #: True when the row compares figures not measured on the same basis or at the same date.
    basis_mismatch: bool
    verified_against_edgar: bool
    tool_expectation: ToolExpectation | None

    @property
    def target_chunk_ids(self) -> frozenset[str]:
        """Every chunk the reference is grounded in — recall's numerator target."""
        return frozenset(chunk_id for entry in self.grounding for chunk_id in entry.chunk_ids)

    @property
    def target_sections(self) -> tuple[str, ...]:
        """`TSLA Item 1A`, once per grounding entry, in authored order."""
        return tuple(entry.key for entry in self.grounding)

    @property
    def tickers(self) -> frozenset[str]:
        """The filers a correct answer draws on — filer-level precision's target (§7)."""
        return frozenset(entry.ticker for entry in self.grounding)

    @property
    def non_trivial_sections(self) -> tuple[str, ...]:
        """Targets a `k=5` retrieval could actually miss."""
        trivial = frozenset(self.recall_trivial_sections)
        return tuple(key for key in self.target_sections if key not in trivial)


@dataclass(frozen=True, slots=True)
class GoldenSet:
    """The committed set, plus the provenance a report has to quote alongside its numbers."""

    questions: tuple[GoldenQuestion, ...]
    verified_against_edgar: bool
    collection_ingest_run: str
    schema_version: int

    def __len__(self) -> int:
        return len(self.questions)

    def __iter__(self):
        return iter(self.questions)

    def by_bucket(self) -> Mapping[Bucket, tuple[GoldenQuestion, ...]]:
        """Rows grouped by stratum, in `Bucket` declaration order.

        Declaration order rather than insertion order, so every table in the artifact lists the
        buckets the same way round and two tables can be read against each other.
        """
        grouped = {
            bucket: tuple(row for row in self.questions if row.bucket is bucket)
            for bucket in Bucket
        }
        return MappingProxyType(grouped)

    def row(self, row_id: str) -> GoldenQuestion:
        """The row with `row_id`, or `KeyError` naming what was asked for."""
        for question in self.questions:
            if question.id == row_id:
                return question
        raise KeyError(f"no golden-set row with id {row_id!r}")


def load_golden_set(
    path: Path | str | None = None, *, require_verified: bool = True
) -> GoldenSet:
    """Read the committed golden set into typed rows.

    `require_verified=False` exists for a *draft* set under authoring; the harness never passes
    it, because ADR-0002's flag is what stands between an unverified reference and a published
    number.
    """
    path = Path(path) if path is not None else GOLDEN_SET_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    verified = bool(payload["provenance"]["verified_against_edgar"])
    if require_verified and not verified:
        raise UnverifiedGoldenSet(
            f"{path} reports verified_against_edgar=false, so its references have not been "
            f"hand-checked against the primary source (ADR-0002 decision 1). Scoring it would "
            f"publish numbers over unverified ground truth."
        )
    return GoldenSet(
        questions=tuple(_question(row) for row in payload["questions"]),
        verified_against_edgar=verified,
        collection_ingest_run=str(payload["provenance"]["collection_ingest_run"]),
        schema_version=int(payload["schema_version"]),
    )


def _question(row: Mapping[str, Any]) -> GoldenQuestion:
    return GoldenQuestion(
        id=str(row["id"]),
        bucket=Bucket(row["bucket"]),
        question=str(row["question"]),
        reference=str(row["reference"]),
        grounding=tuple(_grounding(entry) for entry in row["grounding"]),
        labels=tuple(str(label) for label in row.get("labels", ())),
        known_false_positives=frozenset(
            str(chunk_id) for chunk_id in row.get("known_false_positives", ())
        ),
        section_chunk_counts=MappingProxyType(
            {str(key): int(count) for key, count in row["section_chunk_counts"].items()}
        ),
        recall_trivial=bool(row["recall_trivial"]),
        recall_trivial_sections=tuple(
            str(key) for key in row.get("recall_trivial_sections", ())
        ),
        basis_mismatch=bool(row["basis_mismatch"]),
        verified_against_edgar=bool(row["verified_against_edgar"]),
        tool_expectation=_tool_expectation(row.get("tool_expectation")),
    )


def _grounding(entry: Mapping[str, Any]) -> Grounding:
    return Grounding(
        ticker=str(entry["ticker"]),
        section=Section(entry["section"]),
        fiscal_year=int(entry["fiscal_year"]),
        accession=str(entry["accession"]),
        chunk_ids=tuple(str(chunk_id) for chunk_id in entry["chunk_ids"]),
        passage=str(entry["passage"]),
    )


def _tool_expectation(entry: Mapping[str, Any] | None) -> ToolExpectation | None:
    if entry is None:
        return None
    return ToolExpectation(
        name=str(entry["name"]),
        args=MappingProxyType(dict(entry.get("args", {}))),
        peer_set=(
            None
            if entry.get("peer_set") is None
            else tuple(str(peer) for peer in entry["peer_set"])
        ),
        n=None if entry.get("n") is None else int(entry["n"]),
    )
