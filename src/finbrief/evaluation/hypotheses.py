"""The pre-registered predictions, as data — and the two triggers that fire on them.

Every prediction below was written down **before any A/B data existed**, in ADR-0002 decision
4 as revised by ADR-0004's T6 amendment §6/§7. They live here as data rather than as prose in
a report
for the reason ADR-0005 pre-registers anything: a prediction that can be reworded after
the measurement is not a prediction. `report.hypothesis_section` renders prediction first,
measurement second, verdict last, in that column order, so the artifact cannot be read as having
fitted one to the other.

**A refuted prediction is a result.** ADR-0004 §6 is the precedent — a pre-registered hypothesis
failed there ("BM25 on the retained original already nails exact-identifier"), the failure was
root-caused, and the ADR came out stronger with a mechanism attached to its replacement. Nothing
here is scored on how many predictions survived.

**Nothing rests on response relevancy.** It is non-reproducible by construction
(`judge.EXCLUDED_FROM_HYPOTHESES`), so `_check_metric` refuses to build a prediction on it
rather than trusting whoever writes the next one to remember.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from finbrief.evaluation.arms import (
    HYBRID_ONLY,
    HYBRID_TRANSLATED,
    VECTOR_ONLY,
    VECTOR_TRANSLATED,
    Arm,
)
from finbrief.evaluation.judge import (
    CONTEXT_PRECISION,
    CONTEXT_RECALL,
    EXCLUDED_FROM_HYPOTHESES,
)
from finbrief.evaluation.loader import Bucket
from finbrief.evaluation.metrics import Summary, Verdict, compare
from finbrief.evaluation.pipeline import Cell


class FencedMetric(ValueError):
    """A prediction was written on a metric no pre-registered decision may rest on."""


def _check_metric(metric: str) -> str:
    if metric in EXCLUDED_FROM_HYPOTHESES:
        raise FencedMetric(
            f"{metric!r} is excluded from every pre-registered decision "
            f"(judge.EXCLUDED_FROM_HYPOTHESES): it is non-reproducible by construction, so a "
            f"delta in it cannot confirm or refute anything."
        )
    return metric


@dataclass(frozen=True, slots=True)
class Prediction:
    """One pre-registered claim, and the exact comparison that settles it."""

    id: str
    #: The claim, in the words of the ADR that registered it.
    text: str
    source: str
    bucket: Bucket
    metric: str
    baseline: Arm
    candidate: Arm
    #: What the ADR predicted the comparison would show.
    expected: Verdict

    def __post_init__(self) -> None:
        _check_metric(self.metric)


#: ADR-0002 decision 4 as revised by ADR-0004 §6/§7 — every one written pre-data.
#:
#: Both metrics are the judged, text-comparing ones, deliberately: chunk-identity recall is a
#: near-lottery on this set's large sections (see `report`'s caveat), and ADR-0005's own clause
#: names context precision and context recall.
PREDICTIONS: tuple[Prediction, ...] = (
    Prediction(
        id="H1",
        text=(
            "hybrid > vector-only on `exact-identifier` — ADR-0002's original prediction, "
            "which "
            "ADR-0004 §7 narrows to *at equal translation off* and expects to be where the "
            "bucket's win comes from **least**"
        ),
        source="ADR-0002 decision 4; narrowed by ADR-0004 §7",
        bucket=Bucket.EXACT_IDENTIFIER,
        metric=CONTEXT_PRECISION,
        baseline=VECTOR_ONLY,
        candidate=HYBRID_ONLY,
        expected=Verdict.BETTER,
    ),
    Prediction(
        id="H2",
        text=(
            "translation is **positive** on `exact-identifier`, via entity normalisation — "
            "ADR-0004 §6 revising the earlier '≈ neutral', on one root-caused case where the "
            "target chunk moved from absent to rank 1"
        ),
        source="ADR-0004 §6 (T6 amendment)",
        bucket=Bucket.EXACT_IDENTIFIER,
        metric=CONTEXT_PRECISION,
        baseline=HYBRID_ONLY,
        candidate=HYBRID_TRANSLATED,
        expected=Verdict.BETTER,
    ),
    Prediction(
        id="H3",
        text=(
            "translation wins on `multi-hop` — decomposition gives the retrievers something "
            "a filing actually answers"
        ),
        source="ADR-0002 decision 4",
        bucket=Bucket.MULTI_HOP,
        metric=CONTEXT_RECALL,
        baseline=HYBRID_ONLY,
        candidate=HYBRID_TRANSLATED,
        expected=Verdict.BETTER,
    ),
    Prediction(
        id="H4",
        text=(
            "all configurations ≈ tie on `semantic`. **A predicted tie is evidence the "
            "experiment "
            "is sound, not a failure** (ADR-0002 decision 4)"
        ),
        source="ADR-0002 decision 4",
        bucket=Bucket.SEMANTIC,
        metric=CONTEXT_PRECISION,
        baseline=VECTOR_TRANSLATED,
        candidate=HYBRID_TRANSLATED,
        expected=Verdict.WITHIN_SPREAD,
    ),
    Prediction(
        id="H5",
        text=(
            "hybrid's marginal contribution over `vector + translation` on "
            "`exact-identifier` is "
            "**small** — ADR-0004 §7, because §6 measured the recovery as embedding-side and "
            "BM25's role in it as redundancy rather than recovery"
        ),
        source="ADR-0004 §7 (pre-data)",
        bucket=Bucket.EXACT_IDENTIFIER,
        metric=CONTEXT_PRECISION,
        baseline=VECTOR_TRANSLATED,
        candidate=HYBRID_TRANSLATED,
        expected=Verdict.WITHIN_SPREAD,
    ),
    Prediction(
        id="H6",
        text=(
            "hybrid's contribution may be **larger on `semantic`** than on the bucket it "
            "exists "
            "to win, after ADR-0004 §10 stopped BM25 admitting chunks on question-form terms"
        ),
        source="ADR-0004 §7/§10 (pre-data)",
        bucket=Bucket.SEMANTIC,
        metric=CONTEXT_PRECISION,
        baseline=VECTOR_TRANSLATED,
        candidate=HYBRID_TRANSLATED,
        expected=Verdict.BETTER,
    ),
)


def _summarise(cells: Sequence[Cell], arm: Arm, bucket: Bucket, metric: str) -> Summary:
    return Summary.of(
        cell.judged.get(metric)
        for cell in cells
        if cell.arm == arm.name and cell.bucket is bucket
    )


@dataclass(frozen=True, slots=True)
class Outcome:
    """A prediction, what the run measured, and whether it survived."""

    prediction: Prediction
    measured: Verdict
    delta: float | None
    spread: float | None
    baseline: Summary
    candidate: Summary

    @property
    def confirmed(self) -> bool | None:
        """`True` confirmed, `False` refuted, `None` when the run could not settle it.

        Three-valued on purpose: "the sample could not tell" is not "the prediction was wrong",
        and collapsing them would let an undetermined cell be reported as a refutation.
        """
        if self.measured is Verdict.UNDETERMINED:
            return None
        return self.measured is self.prediction.expected

    @property
    def verdict_text(self) -> str:
        if self.confirmed is None:
            return "undetermined"
        return "confirmed" if self.confirmed else "refuted"


def outcomes(cells: Sequence[Cell]) -> tuple[Outcome, ...]:
    """Each prediction against what this run measured."""
    results = []
    for prediction in PREDICTIONS:
        baseline = _summarise(cells, prediction.baseline, prediction.bucket, prediction.metric)
        candidate = _summarise(
            cells, prediction.candidate, prediction.bucket, prediction.metric
        )
        comparison = compare(
            prediction.metric,
            prediction.bucket,
            baseline_arm=prediction.baseline.name,
            candidate_arm=prediction.candidate.name,
            baseline=baseline,
            candidate=candidate,
        )
        results.append(
            Outcome(
                prediction=prediction,
                measured=comparison.verdict,
                delta=comparison.delta,
                spread=comparison.basis,
                baseline=baseline,
                candidate=candidate,
            )
        )
    return tuple(results)


def as_report_entries(results: Sequence[Outcome]) -> tuple[Mapping[str, Any], ...]:
    """`report.hypothesis_section`'s rows: prediction, then measurement, then verdict."""
    entries = []
    for outcome in results:
        prediction = outcome.prediction
        delta = "—" if outcome.delta is None else f"{outcome.delta:+.3f}"
        spread = "—" if outcome.spread is None else f"{outcome.spread:.3f}"
        entries.append(
            {
                "prediction": f"**{prediction.id}** {prediction.text} "
                f"<br>*{prediction.source}*",
                "measurement": (
                    f"{prediction.metric.replace('_', ' ')} on `{prediction.bucket.value}`, "
                    f"{prediction.candidate.name} − {prediction.baseline.name}: "
                    f"Δ {delta} against per-question spread {spread} → "
                    f"**{outcome.measured.value}** "
                    f"(baseline {_mean(outcome.baseline)}, "
                    f"candidate {_mean(outcome.candidate)})"
                ),
                "verdict": outcome.verdict_text,
            }
        )
    return tuple(entries)


def _mean(summary: Summary) -> str:
    return "—" if summary.mean is None else f"{summary.mean:.3f} n={summary.n}"


@dataclass(frozen=True, slots=True)
class FalsificationClause:
    """ADR-0005's directional test, per bucket, on both metrics at once.

    The clause: **if translation is worse on *both* context precision *and* context recall
    within any bucket, the default drops to `hybrid-only`** and the contradiction is written
    up as a finding. Directional and two-sided by design — with ~7 questions per bucket a
    tight numeric margin would be false precision — so this fires only when both metrics move
    against translation by more than their own spread.
    """

    bucket: Bucket
    precision: Verdict
    recall: Verdict

    @property
    def fires(self) -> bool:
        return self.precision is Verdict.WORSE and self.recall is Verdict.WORSE


def falsification_clause(cells: Sequence[Cell]) -> tuple[FalsificationClause, ...]:
    """ADR-0005's clause evaluated per bucket, at the shipping strategy.

    Held at `hybrid` on both sides, because the clause is about *translation*: comparing across
    strategies would let a strategy effect drop the default for translation's supposed sin.
    """
    results = []
    for bucket in Bucket:
        verdicts = {}
        for metric in (CONTEXT_PRECISION, CONTEXT_RECALL):
            verdicts[metric] = compare(
                metric,
                bucket,
                baseline_arm=HYBRID_ONLY.name,
                candidate_arm=HYBRID_TRANSLATED.name,
                baseline=_summarise(cells, HYBRID_ONLY, bucket, metric),
                candidate=_summarise(cells, HYBRID_TRANSLATED, bucket, metric),
            ).verdict
        results.append(
            FalsificationClause(
                bucket=bucket,
                precision=verdicts[CONTEXT_PRECISION],
                recall=verdicts[CONTEXT_RECALL],
            )
        )
    return tuple(results)


@dataclass(frozen=True, slots=True)
class ReexaminationTrigger:
    """ADR-0005 §4's trigger: `hybrid − vector` at equal translation, on every bucket.

    It fires on an **absence of gain** rather than on a loss: "if T10 finds `hybrid − vector`
    at equal translation to be within per-question spread on *every* bucket, the dominance
    argument is re-argued rather than defended, and `vector + translation` becomes a live
    candidate for the shipping default." Recorded before the numbers existed, because a
    default kept because its marginal component was never separately measured is p-hacking in
    the other direction.
    """

    per_bucket: Mapping[Bucket, Verdict]

    @property
    def fires(self) -> bool:
        settled = [v for v in self.per_bucket.values() if v is not Verdict.UNDETERMINED]
        if not settled:
            return False
        return all(v is Verdict.WITHIN_SPREAD for v in settled)

    @property
    def undetermined(self) -> tuple[Bucket, ...]:
        """Buckets the run could not settle — named, because they weaken the conclusion."""
        return tuple(
            bucket
            for bucket, verdict in self.per_bucket.items()
            if verdict is Verdict.UNDETERMINED
        )


def reexamination_trigger(cells: Sequence[Cell]) -> ReexaminationTrigger:
    """ADR-0005 §4, on context precision at equal translation (both `+translation`)."""
    per_bucket = {}
    for bucket in Bucket:
        per_bucket[bucket] = compare(
            CONTEXT_PRECISION,
            bucket,
            baseline_arm=VECTOR_TRANSLATED.name,
            candidate_arm=HYBRID_TRANSLATED.name,
            baseline=_summarise(cells, VECTOR_TRANSLATED, bucket, CONTEXT_PRECISION),
            candidate=_summarise(cells, HYBRID_TRANSLATED, bucket, CONTEXT_PRECISION),
        ).verdict
    return ReexaminationTrigger(per_bucket=per_bucket)
