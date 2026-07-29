"""The retrieval metrics that cost nothing: precision@k, recall@k, filer precision, leakage.

Every number here comes from a `Retrieval`'s own return value against the golden set's chunk ids
— no judge, no model, no spend. That is why they carry the A/B: ADR-0011's scope observation
("the per-question retrieval provenance is available to T10 without reading a single log line")
applies with equal force to the metrics themselves, and a free deterministic number is one a
reviewer can re-derive.

Four decisions in here are about *not* reporting a number, and each one is a rule this repo
already enforces elsewhere:

**The granularity was chosen after measuring, and the measurement is why.** The obvious
deterministic metric is chunk-identity precision/recall — did the retriever return the exact
chunks a reference was authored from? On this corpus that measures **section size**, not
retrieval quality: S1's reference is authored from 6 of TSLA Item 1A's **129** chunks, and the
first live run returned chunks 48-70 against targets 0, 1, 6, 34, 94 and 117 — chunk recall
0.000 and section recall 1.000, for a retrieval that found the right filer's right Item and
would satisfy an analyst. So **section-level recall and precision are the deterministic
headline**, with filer-level precision beside them (ADR-0004 §7's own prediction is about
filer precision), and chunk-level figures are kept for the rows where they mean something —
the small sections, where "the right chunks" and "the right section" nearly coincide. Reported
in that order, and the order is a finding rather than a preference.

- **An absent target chunk has no rank.** `target_rank` is `None` when nothing relevant was
  retrieved, never `k + 1` or `999`. A stand-in would average into a mean and read as a
  measurement of a retrieval that missed entirely (`Context.distance`'s reason for being
  optional).
- **Recall is reported against the real denominator, with its ceiling beside it.** Six of the
  golden set's rows target more chunks than `k=5` can return, so their recall *cannot* reach
  1.0. Dividing by `min(k, targets)` would hide that by construction; reporting the ceiling
  says how much headroom the retriever actually had.
- **A row that cannot miss is reported separately, not excluded and not folded in.**
  `recall_trivial` (ADR-0002 amendment) marks rows every one of whose sections holds ≤ 4 chunks.
  `recall_trivial_sections` marks the partial case, so a multi-section row contributes its
  non-trivial sections and is not credited for the ones it could not miss.
- **A difference smaller than the per-question spread is not a result.** `compare` returns
  `WITHIN_SPREAD` rather than a direction, because with 7 questions per bucket a mean difference
  inside the spread of its own inputs is not evidence — and because #5 measured the embedding
  call as reproducible only to ~0.0009, so a delta at four decimals is API noise wearing a
  finding's clothes.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from finbrief.evaluation.loader import Bucket, GoldenQuestion
from finbrief.retrieval.retrieve import Context, Retrieval


@dataclass(frozen=True, slots=True)
class RowScore:
    """Every free metric for one `(question, arm)` cell.

    Deliberately holds the retrieved chunk ids as well as the ratios computed from them: a
    surprising bucket number has to be inspectable without a re-run, which is the same complaint
    ADR-0004 §9 makes about un-recoverable sub-queries.
    """

    question_id: str
    bucket: Bucket
    arm: str
    #: The `k` requested, so a short retrieval is visible as one rather than inferred.
    k: int
    retrieved_chunk_ids: tuple[str, ...]
    #: How many retrieved chunks are in the row's grounding.
    target_hits: int
    #: `None` when nothing at all came back — an empty retrieval has no precision, and `0.0`
    #: would claim it retrieved wrongly rather than not at all.
    precision_at_k: float | None
    #: Retrieved targets over *all* targets. See the module docstring on the denominator.
    chunk_recall: float
    #: The most `chunk_recall` could have been at this `k`: `min(k, targets) / targets`.
    chunk_recall_ceiling: float
    #: Sections reached over sections needed, counting only sections a `k=5` retrieval could
    #: miss. `None` when every target section was trivial, because a row that cannot miss has no
    #: recall to report — it is reported in its own table instead.
    #:
    #: **The headline recall on this corpus**, for a reason established by measurement rather
    #: than assumed in advance — see the module docstring's note on granularity.
    section_recall: float | None
    #: Share of retrieved chunks that belong to a `(ticker, section)` the row grounds in — the
    #: precision counterpart to `section_recall`, and the headline precision for the same
    #: reason.
    #:
    #: Stricter than `filer_precision` and looser than `precision_at_k`: it asks "did this chunk
    #: come from somewhere the answer should be drawn from?", which is the question an analyst
    #: would ask of a source list, where chunk identity asks whether it is one of the six
    #: paragraphs a reference happened to be authored from.
    section_precision: float | None
    #: Share of retrieved chunks belonging to a filer the row grounds in — ADR-0004 §7's
    #: *positive* prediction for hybrid, and the other sign of §11's mention leakage.
    filer_precision: float | None
    #: Best (lowest) rank at which a target chunk appeared, or `None` for absent.
    target_rank: int | None
    #: Retrieved chunks the golden set labels as correct-lexical-match, wrong-grounding.
    leaked_chunk_ids: tuple[str, ...]
    recall_trivial: bool

    @property
    def hit(self) -> bool:
        """Whether the retrieval reached the grounding at all."""
        return self.target_hits > 0


def score_row(question: GoldenQuestion, retrieval: Retrieval, *, arm: str, k: int) -> RowScore:
    """Every free metric for one cell, from the retrieval's own return value."""
    contexts = retrieval.contexts
    retrieved_ids = tuple(context.chunk_id for context in contexts)
    targets = question.target_chunk_ids
    hits = tuple(context for context in contexts if context.chunk_id in targets)
    return RowScore(
        question_id=question.id,
        bucket=question.bucket,
        arm=arm,
        k=k,
        retrieved_chunk_ids=retrieved_ids,
        target_hits=len(hits),
        precision_at_k=(len(hits) / len(contexts) if contexts else None),
        chunk_recall=(len(hits) / len(targets) if targets else 0.0),
        chunk_recall_ceiling=(min(k, len(targets)) / len(targets) if targets else 0.0),
        section_recall=_section_recall(question, contexts),
        section_precision=_section_precision(question, contexts),
        filer_precision=_filer_precision(question, contexts),
        target_rank=(min(context.rank for context in hits) if hits else None),
        leaked_chunk_ids=tuple(
            chunk_id for chunk_id in retrieved_ids if chunk_id in question.known_false_positives
        ),
        recall_trivial=question.recall_trivial,
    )


def _section_recall(question: GoldenQuestion, contexts: Sequence[Context]) -> float | None:
    """Non-trivial target sections reached, over non-trivial target sections needed.

    `None` when the row has no non-trivial section: S4's single target holds 3 chunks, so a
    `k=5` retrieval reaches it whatever the retriever does, and averaging a guaranteed 1.0
    into the recall column is the inflation ADR-0002's amendment added the flag to prevent.
    """
    needed = question.non_trivial_sections
    if not needed:
        return None
    reached = {f"{context.ticker} {context.section.value}" for context in contexts}
    return sum(1 for key in needed if key in reached) / len(needed)


def _section_precision(question: GoldenQuestion, contexts: Sequence[Context]) -> float | None:
    """Retrieved chunks that sit in a target `(ticker, section)`, over all retrieved chunks.

    Counts **every** target section, trivial ones included — unlike `_section_recall`, which
    excludes them. A trivial section is one a `k=5` retrieval cannot *miss*, which is a
    statement about recall; retrieving from it is still correct, and calling it a precision
    error because it was easy to find would penalise the retriever for the corpus's shape.
    """
    if not contexts:
        return None
    wanted = frozenset(question.target_sections)
    return sum(
        1 for context in contexts if f"{context.ticker} {context.section.value}" in wanted
    ) / len(contexts)


def _filer_precision(question: GoldenQuestion, contexts: Sequence[Context]) -> float | None:
    if not contexts:
        return None
    wanted = question.tickers
    return sum(1 for context in contexts if context.ticker in wanted) / len(contexts)


@dataclass(frozen=True, slots=True)
class Summary:
    """A bucket's mean for one metric, with the spread and the denominator beside it.

    `absent` is not diagnostic detail, for the reason `observability.events.Samples` gives: a
    mean over `present` alone is a mean over a denominator the caller never chose. Here it
    counts the rows where the metric is *undefined* — an absent target rank, a trivial-only
    row's section recall — which must never be averaged as zero.

    Never rendered without `spread`: with 7 questions per bucket, a mean on its own invites a
    comparison the sample cannot support.
    """

    n: int
    absent: int
    mean: float | None
    minimum: float | None
    maximum: float | None

    @property
    def total(self) -> int:
        """Rows considered, defined or not — what a rate must be reported against."""
        return self.n + self.absent

    @property
    def spread(self) -> float | None:
        """max − min over the rows where the metric is defined."""
        if self.minimum is None or self.maximum is None:
            return None
        return self.maximum - self.minimum

    @classmethod
    def of(cls, values: Iterable[float | None]) -> Summary:
        """Summarise `values`, counting `None` as absent rather than as zero."""
        collected = list(values)
        present = [value for value in collected if value is not None]
        return cls(
            n=len(present),
            absent=len(collected) - len(present),
            mean=(statistics.fmean(present) if present else None),
            minimum=(min(present) if present else None),
            maximum=(max(present) if present else None),
        )


class Verdict(StrEnum):
    """What a difference between two arms is allowed to be called."""

    BETTER = "better"
    WORSE = "worse"
    #: The difference is no larger than the spread of the questions it is a mean over, so the
    #: sample cannot tell the two arms apart. **Not** "equal" and not "tied": the arms may
    #: genuinely differ, and this says only that these 7 questions do not show it.
    WITHIN_SPREAD = "within spread"
    #: One side has no number at all.
    UNDETERMINED = "undetermined"


@dataclass(frozen=True, slots=True)
class Comparison:
    """One arm against another on one metric in one bucket, with the verdict's own basis."""

    metric: str
    bucket: Bucket
    baseline_arm: str
    candidate_arm: str
    baseline: Summary
    candidate: Summary
    verdict: Verdict

    @property
    def delta(self) -> float | None:
        if self.baseline.mean is None or self.candidate.mean is None:
            return None
        return self.candidate.mean - self.baseline.mean

    @property
    def basis(self) -> float | None:
        """The spread the verdict was judged against — the larger of the two arms'."""
        spreads = [s for s in (self.baseline.spread, self.candidate.spread) if s is not None]
        return max(spreads) if spreads else None


def compare(
    metric: str,
    bucket: Bucket,
    *,
    baseline_arm: str,
    candidate_arm: str,
    baseline: Summary,
    candidate: Summary,
) -> Comparison:
    """Judge `candidate` against `baseline`, refusing to call a within-spread difference.

    The rule #11 fixes in one sentence — "no threshold at four decimals" — implemented as a
    comparison against the *measured* spread of the questions rather than against a constant
    nobody derived. A bucket whose 7 questions range over 0.4 cannot support a claim about a
    mean difference of 0.05, and saying so is the honest reading of ADR-0005's own reasoning
    for making its falsification clause directional ("with ~6 questions per bucket, a tight
    numeric margin would be false precision").
    """
    delta = (
        None
        if baseline.mean is None or candidate.mean is None
        else candidate.mean - baseline.mean
    )
    spreads = [s for s in (baseline.spread, candidate.spread) if s is not None]
    basis = max(spreads) if spreads else None
    if delta is None or basis is None:
        verdict = Verdict.UNDETERMINED
    elif abs(delta) <= basis:
        verdict = Verdict.WITHIN_SPREAD
    else:
        verdict = Verdict.BETTER if delta > 0 else Verdict.WORSE
    return Comparison(
        metric=metric,
        bucket=bucket,
        baseline_arm=baseline_arm,
        candidate_arm=candidate_arm,
        baseline=baseline,
        candidate=candidate,
        verdict=verdict,
    )


def summarise(
    scores: Iterable[RowScore], metric: str, *, exclude_trivial: bool = False
) -> Summary:
    """`metric` across `scores` as a `Summary`.

    `exclude_trivial` drops whole rows flagged `recall_trivial` — used for the recall columns,
    where a question that cannot miss would inflate the number, and *not* for precision, which a
    small target section does not make trivial.
    """
    selected = [score for score in scores if not (exclude_trivial and score.recall_trivial)]
    return Summary.of(getattr(score, metric) for score in selected)


@dataclass(frozen=True, slots=True)
class LeakagePrecision:
    """Mention-leakage precision over the labelled probes (ADR-0004 §11).

    Reported *beside* the bucket mean rather than inside it, which is what makes the cost
    computable at all: these chunks are correct lexical matches for the question and wrong
    grounding, so BM25 admitting them is a precision cost hybrid pays for its recall gain.
    """

    arm: str
    rows: int
    retrieved: int
    leaked: int

    @property
    def precision(self) -> float | None:
        """Share of retrieved chunks that are not known false positives."""
        if not self.retrieved:
            return None
        return 1 - self.leaked / self.retrieved


def leakage_precision(scores: Sequence[RowScore], *, arm: str) -> LeakagePrecision:
    """Leakage over the rows that carry labelled false positives, and only those.

    Rows with no measured leakage are excluded rather than counted as clean: a row nobody
    enumerated false positives for contributes no evidence either way, and including it would
    dilute the rate towards 1.0 with rows that were never probed.
    """
    return LeakagePrecision(
        arm=arm,
        rows=len(scores),
        retrieved=sum(len(score.retrieved_chunk_ids) for score in scores),
        leaked=sum(len(score.leaked_chunk_ids) for score in scores),
    )
