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
- **A difference the sample cannot resolve is not a result.** `compare` returns `NOT_DETECTED`
  rather than a direction, because with 7 questions per bucket a small mean difference is not
  evidence — and because #5 measured the embedding call as reproducible only to ~0.0009, so a
  delta at four decimals is API noise wearing a finding's clothes.

**The instrument reports when it could not have detected anything, and that is the whole
point of this module's second revision.** The first version compared a difference of *means*
against `basis` — the larger of the two arms' raw per-question **range**. Because each arm's
mean is bounded by that arm's own min and max, the largest delta the observed values permit is
`max(cand.max − base.min, base.max − cand.min)`; and when two arms score the *same* questions
under near-identical configurations, that quantity **equals the range**. The test was
`abs(delta) <= basis`, so it could not fail. All 18 pre-registered comparisons in the first
committed artifact — six hypotheses, eight falsification-clause cells, four re-examination
trigger cells — were rendered as verdicts by a test that was mathematically incapable of
returning anything else (ADR-0002's T10 amendment §3 records it). The pre-registration was
sound; the instrument judging it was not.

So this version does three things the first did not: it pairs on the **question** rather than
comparing two ranges, it settles direction with an **exact** test rather than a bound nobody
derived, and `Paired.detectable` states whether the test could have reached `PAIRED_ALPHA` at
this effective n **at all**. A verdict from an undetectable comparison is reported as
undetectable, because "we could not have seen it" is a third claim and neither of the other
two.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping, Sequence
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
    #: The paired test did not resolve a difference at `PAIRED_ALPHA`, so the sample cannot
    #: tell the two arms apart. **Not** "equal" and not "tied": the arms may genuinely differ,
    #: and this says only that these 7 questions do not show it.
    NOT_DETECTED = "not detected"
    #: The test could not have resolved a difference of *any* size at this effective n — the
    #: exact two-sided p has a floor of `2 / 2**effective_n`, above `PAIRED_ALPHA` for n < 6.
    #: Separate from `NOT_DETECTED` because a null result from an instrument that had no chance
    #: of firing is not evidence of absence, and reporting it as one is the defect this module's
    #: docstring records.
    UNDETECTABLE = "undetectable at this n"
    #: One side has no number at all.
    UNDETERMINED = "undetermined"


#: Two-sided significance level for the paired exact test.
#:
#: Not in `config.py` deliberately, on the same grounds as the ingestion thresholds: it is an
#: assertion about what this harness is willing to call a difference, calibrated against the
#: golden set's ~7 questions per bucket, and an env-overridable alpha is a knob for tuning a
#: verdict after seeing it.
PAIRED_ALPHA = 0.05


def minimum_detectable_pairs(alpha: float = PAIRED_ALPHA) -> int:
    """The fewest differing questions at which the exact test can reach `alpha` at all.

    Derived rather than written down: the two-sided exact p cannot go below `2 / 2**m`, so this
    walks m upward until it can. Six at α=0.05. A typed constant here would be a number that
    silently stops matching `PAIRED_ALPHA` the day anyone moves it.
    """
    m = 1
    while 2 / 2**m > alpha:
        m += 1
    return m


def tolerated_minority_signs(pairs: int, alpha: float = PAIRED_ALPHA) -> int:
    """How many of `pairs` differences may point *against* the majority and still reach `alpha`.

    **This is the number that says what the design can resolve**, and it is bleaker than the
    p-floor alone suggests: at exactly `minimum_detectable_pairs()` differing questions the
    answer is **zero** — the effect has to be perfectly unanimous — and at seven it is one. So a
    bucket of ~7 questions can only ever resolve a near-unanimous effect, whatever its size.
    ADR-0002 sized the buckets at ≥6 against a different kind of test, and this is the
    consequence, derived here so the artifact states it rather than leaving it to be inferred
    from a p-value.

    `-1` when no arrangement of `pairs` differences can reach `alpha`.
    """
    if 2 / 2**pairs > alpha:
        return -1
    tolerated = -1
    for flips in range(pairs + 1):
        # Flip the smallest-magnitude differences first: that is the arrangement most favourable
        # to significance, so it is the ceiling on what any real data could tolerate.
        deltas = [float(rank + 1) for rank in range(pairs)]
        for index in range(flips):
            deltas[index] = -deltas[index]
        p_value = signed_rank_p(deltas)
        if p_value is None or p_value > alpha:
            break
        tolerated = flips
    return tolerated


@dataclass(frozen=True, slots=True)
class Paired:
    """The same questions under two arms, aligned by question id — the unit of comparison.

    Paired rather than two independent summaries because the arms score the **same** golden-set
    rows: the per-question difference removes the question's own difficulty, which is the term
    that dominates the raw spread. A bucket whose questions range over 0.63 can still have
    every paired difference inside 0.05, and only the paired form can see that.

    `dropped` counts questions where either arm had no number, for the reason
    `observability.events.Samples` returns `absent`: a paired n the caller did not choose is a
    denominator they cannot weigh.
    """

    question_ids: tuple[str, ...]
    #: `candidate − baseline`, one per question both arms scored.
    deltas: tuple[float, ...]
    dropped: int

    @property
    def n(self) -> int:
        """Questions both arms scored."""
        return len(self.deltas)

    @property
    def effective_n(self) -> int:
        """Questions whose score actually *differed* — what the signed-rank test ranks.

        A zero difference carries no sign, so it contributes nothing to the test. Reported
        because it, and not `n`, is the denominator the p-value was computed against.
        """
        return sum(1 for delta in self.deltas if delta != 0.0)

    @property
    def mean_delta(self) -> float | None:
        return statistics.fmean(self.deltas) if self.deltas else None

    @property
    def delta_range(self) -> tuple[float, float] | None:
        """min and max of the paired differences — the effect size's own spread."""
        if not self.deltas:
            return None
        return (min(self.deltas), max(self.deltas))

    @property
    def detectable(self) -> bool:
        """Whether the exact test could reach `PAIRED_ALPHA` at this effective n at all.

        The two-sided exact p-value cannot go below `2 / 2**effective_n` — every rank on one
        side is the most extreme assignment there is — so below 6 differing questions no
        arrangement of the data can produce a significant result. **This property is the fix
        for the defect in the module docstring**: it makes "the instrument had no chance"
        representable instead of indistinguishable from "the arms are the same".
        """
        n = self.effective_n
        return n > 0 and 2 / 2**n <= PAIRED_ALPHA


def paired(
    baseline: Mapping[str, float | None], candidate: Mapping[str, float | None]
) -> Paired:
    """Align two arms' per-question scores, keeping only questions both arms scored.

    Keyed on question id rather than on position: the cells arrive per arm and a filtered or
    reordered arm would otherwise pair question 3's score with question 5's, which is a
    comparison of nothing that looks exactly like a comparison.
    """
    ids, deltas, dropped = [], [], 0
    for question_id in sorted(set(baseline) | set(candidate)):
        before, after = baseline.get(question_id), candidate.get(question_id)
        if before is None or after is None:
            dropped += 1
            continue
        ids.append(question_id)
        deltas.append(after - before)
    return Paired(question_ids=tuple(ids), deltas=tuple(deltas), dropped=dropped)


def _midranks(values: Sequence[float]) -> tuple[float, ...]:
    """Ranks of `values`, ties sharing their average rank. Always multiples of 0.5."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        shared = (position + end) / 2 + 1
        for index in order[position : end + 1]:
            ranks[index] = shared
        position = end + 1
    return tuple(ranks)


def signed_rank_p(deltas: Sequence[float]) -> float | None:
    """Exact two-sided Wilcoxon signed-rank p for `deltas`, or `None` when none differ.

    **Exact, by enumerating the null distribution rather than approximating it.** At n=7 a
    normal approximation is not defensible — that was the review's constraint — so the null is
    built by convolution over the signed ranks: under the null every sign assignment is equally
    likely, so the distribution of `W+` is the count of assignments reaching each rank sum. That
    is O(n · ΣR) and exact for every n this harness sees.

    Two honest caveats, stated because a p-value's assumptions are part of the measurement:
    tied `|delta|` values take midranks, which makes the null mildly approximate (the standard
    treatment, and the alternative — dropping ties — discards data at an n that cannot spare
    it); and zero differences are excluded, which is why `Paired.effective_n` is reported
    beside every p.
    """
    nonzero = [delta for delta in deltas if delta != 0.0]
    if not nonzero:
        return None
    # Doubled so midranks (always multiples of 0.5) index an integer convolution exactly.
    ranks = [int(round(rank * 2)) for rank in _midranks([abs(d) for d in nonzero])]
    counts = {0: 1}
    for rank in ranks:
        nxt: dict[int, int] = {}
        for total, count in counts.items():
            nxt[total] = nxt.get(total, 0) + count  # this rank signed negative
            nxt[total + rank] = nxt.get(total + rank, 0) + count  # signed positive
        counts = nxt
    observed = sum(rank for rank, delta in zip(ranks, nonzero, strict=True) if delta > 0)
    universe = 2 ** len(ranks)
    at_or_below = sum(c for total, c in counts.items() if total <= observed) / universe
    at_or_above = sum(c for total, c in counts.items() if total >= observed) / universe
    return min(1.0, 2 * min(at_or_below, at_or_above))


@dataclass(frozen=True, slots=True)
class Comparison:
    """One arm against another on one metric in one bucket, with the test that settled it."""

    metric: str
    bucket: Bucket
    baseline_arm: str
    candidate_arm: str
    baseline: Summary
    candidate: Summary
    pairs: Paired
    #: `None` when no question's score differed, so there was nothing to rank.
    p_value: float | None
    verdict: Verdict

    @property
    def delta(self) -> float | None:
        """The paired mean difference — the effect size the verdict is about.

        The mean of the per-question differences, which for a complete pairing equals the
        difference of the two means and for a partial one is the honest version of it: the
        difference of means over *different* question subsets is not a difference.
        """
        return self.pairs.mean_delta


def compare(
    metric: str,
    bucket: Bucket,
    *,
    baseline_arm: str,
    candidate_arm: str,
    baseline: Summary,
    candidate: Summary,
    pairs: Paired,
) -> Comparison:
    """Judge `candidate` against `baseline` on the paired per-question differences.

    Three outcomes and not two, which is the correction this module's docstring records. The
    exact signed-rank test settles direction; `Paired.detectable` decides whether a null result
    is `NOT_DETECTED` ("these questions do not show it") or `UNDETECTABLE` ("no arrangement of
    this many differing questions could have shown it"). `baseline` and `candidate` are still
    carried so the artifact can print each arm's own mean and spread beside the delta, but no
    verdict rests on them — a difference of means judged against a range of raw observations is
    the test that could not fail.
    """
    if not pairs.n:
        verdict = Verdict.UNDETERMINED
        p_value = None
    else:
        p_value = signed_rank_p(pairs.deltas)
        mean_delta = pairs.mean_delta or 0.0
        if not pairs.detectable:
            verdict = Verdict.UNDETECTABLE
        elif p_value is not None and p_value <= PAIRED_ALPHA:
            verdict = Verdict.BETTER if mean_delta > 0 else Verdict.WORSE
        else:
            verdict = Verdict.NOT_DETECTED
    return Comparison(
        metric=metric,
        bucket=bucket,
        baseline_arm=baseline_arm,
        candidate_arm=candidate_arm,
        baseline=baseline,
        candidate=candidate,
        pairs=pairs,
        p_value=p_value,
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
