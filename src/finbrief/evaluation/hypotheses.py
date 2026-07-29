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

**But a prediction the sample could not settle is not refuted, and that distinction cost this
module its first four verdicts.** "No effect detected at n=7" and "no effect" are different
claims, and `Outcome.confirmed` collapsed the first into the second — so H1, H2, H3 and H6 were
published as **refuted** on null results, H2 while its paired delta ran +0.103 in the predicted
direction and while ADR-0004 §6's live root-caused case stood unreconciled beside it. The
instrument underneath was worse than underpowered: it could not return anything else
(`metrics`' module docstring; ADR-0002's T10 amendment §3). The verdict vocabulary is now
four-valued — confirmed, refuted, `not detected (n=…)`, `undetectable (effective n=…)` — and a
directional prediction is refuted only by a measured difference in the *other* direction.

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
    SHIPPING_DEFAULT,
    STRATEGY_CONTRAST,
    TRANSLATION_CONTRAST,
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
from finbrief.evaluation.metrics import Comparison, Summary, Verdict, compare, paired
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
        expected=Verdict.NOT_DETECTED,
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
        expected=Verdict.NOT_DETECTED,
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


def _by_question(
    cells: Sequence[Cell], arm: Arm, bucket: Bucket, metric: str
) -> dict[str, float | None]:
    """One arm's scores in one bucket, keyed by question id — what `metrics.paired` aligns."""
    return {
        cell.question_id: cell.judged.get(metric)
        for cell in cells
        if cell.arm == arm.name and cell.bucket is bucket
    }


def _compare(
    cells: Sequence[Cell], baseline: Arm, candidate: Arm, bucket: Bucket, metric: str
) -> Comparison:
    """The one place a comparison is built, so every decision uses the same instrument."""
    return compare(
        metric,
        bucket,
        baseline_arm=baseline.name,
        candidate_arm=candidate.name,
        baseline=_summarise(cells, baseline, bucket, metric),
        candidate=_summarise(cells, candidate, bucket, metric),
        pairs=paired(
            _by_question(cells, baseline, bucket, metric),
            _by_question(cells, candidate, bucket, metric),
        ),
    )


@dataclass(frozen=True, slots=True)
class Outcome:
    """A prediction, what the run measured, and whether it survived."""

    prediction: Prediction
    comparison: Comparison

    @property
    def measured(self) -> Verdict:
        return self.comparison.verdict

    @property
    def confirmed(self) -> bool | None:
        """`True` confirmed, `False` refuted, `None` when the run could not settle it.

        Three-valued on purpose, and **`None` now covers the common case as well as the rare
        one**. It always covered `UNDETERMINED` — one side with no number at all — but the
        instrument's real "the sample could not tell" verdict is `NOT_DETECTED`, and collapsing
        that into `False` reported a null result as a refutation. It did so for four of six
        predictions in the first committed artifact (ADR-0002's T10 amendment §3): `Verdict`'s
        own docstring said the arms "may genuinely differ", and this property threw that away
        one line later. A directional prediction is refuted by a *measured difference in the
        other direction*, and by nothing else.
        """
        if self.measured in (
            Verdict.UNDETERMINED,
            Verdict.NOT_DETECTED,
            Verdict.UNDETECTABLE,
        ):
            return None
        return self.measured is self.prediction.expected

    @property
    def verdict_text(self) -> str:
        """The verdict column's word, with the denominator that earns it.

        `not detected (n=7)` carries its n because that is the claim's whole content: the same
        words at n=700 would be a strong result about an absent effect and at n=7 are a
        statement about the sample.
        """
        if self.measured is Verdict.UNDETERMINED:
            return "undetermined"
        if self.measured is Verdict.UNDETECTABLE:
            return f"undetectable (effective n={self.comparison.pairs.effective_n})"
        if self.measured is Verdict.NOT_DETECTED:
            return f"not detected (n={self.comparison.pairs.n})"
        return "confirmed" if self.confirmed else "refuted"


def outcomes(cells: Sequence[Cell]) -> tuple[Outcome, ...]:
    """Each prediction against what this run measured."""
    return tuple(
        Outcome(
            prediction=prediction,
            comparison=_compare(
                cells,
                prediction.baseline,
                prediction.candidate,
                prediction.bucket,
                prediction.metric,
            ),
        )
        for prediction in PREDICTIONS
    )


def as_report_entries(results: Sequence[Outcome]) -> tuple[Mapping[str, Any], ...]:
    """`report.hypothesis_section`'s rows: prediction, then measurement, then verdict.

    The measurement cell names the **paired** delta, the exact p, and the effective n, because
    those three are the claim. A prediction that expected no difference is annotated as
    *consistent* rather than confirmed: ADR-0002 decision 4 registered the tie as a prediction
    and it deserves to be readable as one, but a null result at n=7 does not confirm a tie — the
    verdict column keeps one vocabulary and the nuance sits with the numbers.
    """
    entries = []
    for outcome in results:
        prediction = outcome.prediction
        pairs = outcome.comparison.pairs
        delta = "—" if outcome.comparison.delta is None else f"{outcome.comparison.delta:+.3f}"
        span = pairs.delta_range
        p_value = outcome.comparison.p_value
        p_text = "—" if p_value is None else f"p={p_value:.3f}"
        measurement = (
            f"{prediction.metric.replace('_', ' ')} on `{prediction.bucket.value}`, "
            f"{prediction.candidate.name} − {prediction.baseline.name}: "
            f"paired Δ {delta}"
            + ("" if span is None else f" [{span[0]:+.3f}…{span[1]:+.3f}]")
            + f", exact signed-rank {p_text}, n={pairs.n} "
            f"({pairs.effective_n} differing) → **{outcome.measured.value}** "
            f"(baseline {_mean(outcome.comparison.baseline)}, "
            f"candidate {_mean(outcome.comparison.candidate)})"
        )
        if prediction.expected is Verdict.NOT_DETECTED and outcome.measured in (
            Verdict.NOT_DETECTED,
            Verdict.UNDETECTABLE,
        ):
            measurement += (
                " — the prediction was of *no* difference, so this is **consistent** with it "
                "rather than a confirmation of it"
            )
        entries.append(
            {
                "prediction": f"**{prediction.id}** {prediction.text} "
                f"<br>*{prediction.source}*",
                "measurement": measurement,
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
    tight numeric margin would be false precision — so this fires only when the paired exact
    test resolves both metrics as worse under translation.

    `could_fire` sits beside `fires` because the two are different claims and the first
    artifact reported only the second.
    """

    bucket: Bucket
    precision: Comparison
    recall: Comparison

    @property
    def fires(self) -> bool:
        return self.precision.verdict is Verdict.WORSE and self.recall.verdict is Verdict.WORSE

    @property
    def could_fire(self) -> bool:
        """Whether this bucket's sample could have fired the clause at all.

        **Reported beside `fires`, and that is the point.** A clause that does not fire on a
        bucket where it *could not have* fired is not evidence for the default it protects —
        which is what the first committed artifact reported on all four buckets (ADR-0002's T10
        amendment §3). Both metrics have to be able to resolve a difference, since the clause
        needs both to be `WORSE`.
        """
        return self.precision.pairs.detectable and self.recall.pairs.detectable


def _contrast_ending_at_default(
    contrast: tuple[tuple[Any, Any], ...], *, axis: str
) -> tuple[Any, Any]:
    """The pair in `contrast` whose second arm is the shipping default.

    **This is how `arms.TRANSLATION_CONTRAST` and `arms.STRATEGY_CONTRAST` come to be read**,
    and reading them is the point (code review of #11). `arms.py` declares both "as data because
    `hypotheses.py` reads them: a clause whose operands are written out in prose is one that can
    be applied to the wrong pair" — and this module wrote all five pairs out by hand, so nothing
    consulted them and `tests/test_eval_arms.py`'s axis-constancy assertions were about data no
    verdict depended on. A claim in a comment cannot fail; this makes the constants
    load-bearing.

    Selected by predicate and not by index, because "the pair whose translated (or hybrid) side
    is what we ship" is the thing ADR-0005 means, and a `[1]` would silently follow a
    reordering.
    """
    for pair in contrast:
        if pair[1] is SHIPPING_DEFAULT:
            return pair
    raise LookupError(
        f"no {axis} contrast in arms.py ends at the shipping default "
        f"({SHIPPING_DEFAULT.name}), so ADR-0005's comparison has no operands. Either the "
        f"default moved or the contrast pairs did."
    )


#: ADR-0005's clause is about *translation*, held at the shipping strategy on both sides —
#: comparing across strategies would let a strategy effect drop the default for translation's
#: supposed sin. That is `TRANSLATION_CONTRAST`'s hybrid pair, named by derivation.
CLAUSE_CONTRAST = _contrast_ending_at_default(TRANSLATION_CONTRAST, axis="translation")

#: ADR-0005 §4's trigger is `hybrid − vector` **at equal translation**, which is
#: `STRATEGY_CONTRAST`'s translated pair.
TRIGGER_CONTRAST = _contrast_ending_at_default(STRATEGY_CONTRAST, axis="strategy")


def falsification_clause(cells: Sequence[Cell]) -> tuple[FalsificationClause, ...]:
    """ADR-0005's clause evaluated per bucket, at the shipping strategy.

    Held at `hybrid` on both sides, because the clause is about *translation*: comparing across
    strategies would let a strategy effect drop the default for translation's supposed sin. The
    two arms come from `CLAUSE_CONTRAST`, so `arms.py`'s axis-constancy tests bind them.
    """
    baseline, translated = CLAUSE_CONTRAST
    return tuple(
        FalsificationClause(
            bucket=bucket,
            precision=_compare(cells, baseline, translated, bucket, CONTEXT_PRECISION),
            recall=_compare(cells, baseline, translated, bucket, CONTEXT_RECALL),
        )
        for bucket in Bucket
    )


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

    per_bucket: Mapping[Bucket, Comparison]

    @property
    def fires(self) -> bool:
        """Fires when every bucket that *could* resolve a gain failed to find one.

        **`UNDETECTABLE` buckets are excluded rather than counted as absences of gain**, and
        that is the correction ADR-0002's T10 amendment §3 records. This trigger fires on an
        absence, so an instrument with no power feeds it directly: under the old range-based
        test every bucket returned "within spread" no matter what the data said, and the trigger
        fired on a tautology. A bucket that could not have seen a gain is not evidence that
        there was none.
        """
        evidence = [
            comparison.verdict
            for comparison in self.per_bucket.values()
            if comparison.verdict not in (Verdict.UNDETERMINED, Verdict.UNDETECTABLE)
        ]
        if not evidence:
            return False
        return all(verdict is Verdict.NOT_DETECTED for verdict in evidence)

    @property
    def resolving(self) -> int:
        """How many buckets carried the power to resolve a gain — the `fires` denominator.

        Printed by `report.decisions_section` because it is the gap between ADR-0005 §4 as
        *registered* ("within per-question spread on **every** bucket") and as it can be
        applied: a bucket with no power is excluded rather than counted as an absence of gain,
        so "every" is every bucket that could have spoken. Firing on 1 of 4 and firing on 4 of 4
        are different strengths of evidence and the artifact has to say which (code review of
        #11).
        """
        return sum(
            1
            for comparison in self.per_bucket.values()
            if comparison.verdict not in (Verdict.UNDETERMINED, Verdict.UNDETECTABLE)
        )

    @property
    def evaluable(self) -> bool:
        """Whether any bucket carried enough power for the trigger to mean anything."""
        return any(comparison.pairs.detectable for comparison in self.per_bucket.values())

    @property
    def undetermined(self) -> tuple[Bucket, ...]:
        """Buckets the run could not settle — named, because they weaken the conclusion."""
        return tuple(
            bucket
            for bucket, comparison in self.per_bucket.items()
            if comparison.verdict is Verdict.UNDETERMINED
        )

    @property
    def undetectable(self) -> tuple[Bucket, ...]:
        """Buckets whose sample could not have resolved a gain of any size."""
        return tuple(
            bucket
            for bucket, comparison in self.per_bucket.items()
            if comparison.verdict is Verdict.UNDETECTABLE
        )


def reexamination_trigger(cells: Sequence[Cell]) -> ReexaminationTrigger:
    """ADR-0005 §4, on context precision at equal translation (both `+translation`).

    The two arms come from `TRIGGER_CONTRAST`, which is `arms.STRATEGY_CONTRAST`'s translated
    pair.
    """
    vector, hybrid = TRIGGER_CONTRAST
    return ReexaminationTrigger(
        per_bucket={
            bucket: _compare(cells, vector, hybrid, bucket, CONTEXT_PRECISION)
            for bucket in Bucket
        }
    )
