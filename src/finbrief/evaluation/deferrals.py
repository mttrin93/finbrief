"""The four measurements earlier tickets deferred to T10, each with its own instrument.

They are grouped here rather than scattered because they share one property: **none of them is
a property of the retrieval chain**, so none belongs in the per-bucket tables. Two read a live
log, one reads the validator, one asks the judge a narrower question than faithfulness does.

| deferral | from | instrument | needs a live run? |
|---|---|---|:--:|
| agent-vs-original divergence | T4 (ADR-0003 §2) | `agent_query.verbatim` | yes |
| square-bracket adherence | T5 (ADR-0006 §4) | `citation_markers` | yes |
| layer 4's residue | T7 (ADR-0006 §4) | `security.advice.validate_answer` | **no** |
| cited-sentence faithfulness | T3/T5 | the judge, per marker | no (answers are cached) |

**The third is free, and that is a finding in itself.** Layer 4 is regex over a Guard, so
measuring what it *misses* costs nothing and needs no model — which means the deferral could
have been closed at any point since T7 and was carried for three tickets because nobody had
written the probe set.

**Each rate is reported against the denominator it was computed over, and a rate over nothing
is `None`.** A divergence rate over zero searches is not 0% divergence, and printing it as one
would be the absence-as-measurement failure this repo enforces against everywhere else.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from finbrief.observability.events import EventLog
from finbrief.security.markers import numeric_markers


@dataclass(frozen=True, slots=True)
class Rate:
    """A count over a denominator, and never a bare percentage.

    `None` for the rate when the denominator is zero: the numbers this ticket reports are from
    small live samples, and a rate whose denominator a reader cannot see is a rate they cannot
    weigh. ADR-0003's amendment says exactly this about the divergence figure it already had —
    "still a sample too small to publish as a rate, and worth saying so when it is published".
    """

    label: str
    hits: int
    total: int

    @property
    def rate(self) -> float | None:
        return None if not self.total else self.hits / self.total

    def render(self) -> str:
        if self.rate is None:
            return f"{self.label}: **not measured** (0 observations)"
        return f"{self.label}: **{self.rate:.0%}** ({self.hits}/{self.total})"


# --- 1. the agent-vs-original query divergence rate (T4) -------------------------------


@dataclass(frozen=True, slots=True)
class Divergence:
    """How often the shipped path asked something other than what the analyst typed.

    ADR-0003 §1 has the tool's description instruct the agent to pass the question
    **verbatim**, and §2 of its T4 amendment records that the rule is *measured, not enforced*
    — nothing in the code prevents a rewrite, and the alternative that would (overwriting the
    model's `query`) breaks every follow-up. This is the measurement that amendment defers
    here.

    `first_turn` and `follow_up` are split because the amendment asks for it: a resolved
    pronoun is a divergence by this definition and is the one edit the description *permits*,
    since the engine is stateless and cannot resolve "its debt". Reporting "N% diverged, of
    which M% were reference resolutions" is more informative than either number alone — and a
    follow-up search is the only place a permitted rewrite can occur, so the turn index is the
    proxy the amendment names.
    """

    first_turn: Rate
    follow_up: Rate

    @property
    def overall(self) -> Rate:
        return Rate(
            label="divergence rate",
            hits=self.first_turn.hits + self.follow_up.hits,
            total=self.first_turn.total + self.follow_up.total,
        )


def divergence(log: EventLog) -> Divergence:
    """The divergence rate from `agent_query` lines, split by first search per thread.

    A thread's *first* search cannot be a reference resolution — there is nothing behind it to
    resolve — so a divergence there is a rewrite the description forbids outright. Later
    searches in the same thread may legitimately have resolved a pronoun, which is why they
    are counted apart rather than excused.
    """
    seen_threads: set[str] = set()
    first = [0, 0]
    later = [0, 0]
    for event in log.of("agent_query"):
        verbatim = event.field("verbatim")
        if verbatim is None:
            continue
        thread = str(event.field("thread_id", ""))
        bucket = later if thread in seen_threads else first
        seen_threads.add(thread)
        bucket[1] += 1
        if not verbatim:
            bucket[0] += 1
    return Divergence(
        first_turn=Rate(label="first search in a thread", hits=first[0], total=first[1]),
        follow_up=Rate(
            label="later searches (may be permitted resolutions)", hits=later[0], total=later[1]
        ),
    )


# --- 2. the square-bracket rule's adherence rate (T5) ---------------------------------


@dataclass(frozen=True, slots=True)
class BracketAdherence:
    """Whether the agent's `[n]` markers behave, over turns that had sources to cite.

    T5 deferred this and T7 gave it a denominator without a second instrument:
    `security/markers.py` logs `citation_markers` on **every** turn. Three failure shapes,
    counted apart because they have different causes and different fixes:

    - `uncited` — the turn retrieved chunks and cited none (`resolved == 0`). T5's observed
      case, and the one that makes a grounded answer uncheckable.
    - `unresolved` — a marker pointing at a source number that does not exist. Structurally
      prevented since T7's register, so a non-zero count here is a regression rather than a
      rate.
    - `non_numeric` — `[Yahoo Finance]`, the other case T5 saw: brackets used for something the
      prompt reserves for retrieved excerpts.

    **`uncited` counts turns; `unresolved` and `non_numeric` count markers, and mixing the two
    into one rate is the defect this shape now prevents** (code review of #11). The first
    version computed `turns_with_sources - (uncited + unresolved + non_numeric)`, subtracting
    per-marker totals from a per-turn denominator: three turns of which one carried two
    unresolved markers and one `[Yahoo Finance]` rendered as **0%** adherence where two of the
    three were clean, and a `max(…, 0)` clamp hid the overflow rather than exposing it. So the
    rate's numerator comes from `dirty`, which is incremented once per failing turn, and the
    marker totals stay what they are — diagnostics that say *how badly* the failing turns
    failed.
    """

    turns_with_sources: int
    #: Turns that failed in **any** of the three ways — the rate's only denominator arithmetic.
    dirty: int
    uncited: int
    #: Marker counts, not turn counts. Reported beside the rate, never subtracted from it.
    unresolved: int
    non_numeric: int

    @property
    def clean(self) -> Rate:
        """Turns with sources whose markers were all numeric, resolving, and present."""
        return Rate(
            label="bracket-rule adherence",
            hits=self.turns_with_sources - self.dirty,
            total=self.turns_with_sources,
        )


def bracket_adherence(log: EventLog) -> BracketAdherence:
    """Adherence over `citation_markers` lines from turns that had at least one source.

    Turns with no sources are excluded from the denominator rather than counted as compliant: an
    answer with nothing to cite cannot break a citation rule, and including it would inflate the
    rate by the number of questions the collection could not answer.
    """
    turns = dirty = uncited = unresolved = non_numeric = 0
    for event in log.of("citation_markers"):
        sources = event.field("sources")
        if not sources:
            continue
        turns += 1
        turn_uncited = not event.field("resolved")
        turn_unresolved = len(event.field("unresolved") or ())
        turn_non_numeric = int(event.field("non_numeric") or 0)
        uncited += int(turn_uncited)
        unresolved += turn_unresolved
        non_numeric += turn_non_numeric
        # One increment per failing turn, whatever the shape or the count of its failures. See
        # `BracketAdherence` on why a marker total may not enter the rate.
        if turn_uncited or turn_unresolved or turn_non_numeric:
            dirty += 1
    return BracketAdherence(
        turns_with_sources=turns,
        dirty=dirty,
        uncited=uncited,
        unresolved=unresolved,
        non_numeric=non_numeric,
    )


# --- 3. layer 4's residue (T7) — free and deterministic -------------------------------


class InstrumentDead(RuntimeError):
    """The validator refused none of its positive controls, so it measured nothing."""


@dataclass(frozen=True, slots=True)
class AdviceResidue:
    """Hand-labelled advice the validator does **not** refuse — layer 4's ceiling.

    `ADVICE_ANSWERS` measures the floor (does it catch the advice that announces itself?) and
    the security suite already reports that. This measures the other half: advice carrying no
    imperative, no rating word, no price target and no position-sizing instruction, which any
    analyst would still read as being told what to do.

    **A 100% residue is also what a dead validator returns**, which is why `controls` exists
    (code review of #11). `security.advice.validate_answer` fails **open** by design — any
    exception out of `Guard.validate` becomes `AdviceVerdict(refused=False)` — and this pass
    reads only `verdict.refused`, so a broken Guard and a validator that legitimately let every
    probe through were the same observation, and the published headline was exactly the value a
    dead instrument yields. The controls are advice the rules provably catch, so a run where
    none of them is refused reports nothing rather than 100%.
    """

    caught: tuple[str, ...]
    residue: tuple[str, ...]
    #: Positive controls this pass refused. Non-zero is what makes the rate a measurement.
    controls_caught: int = 0
    controls: int = 0

    @property
    def rate(self) -> Rate:
        total = len(self.caught) + len(self.residue)
        return Rate(label="layer-4 residue", hits=len(self.residue), total=total)


def advice_residue(
    probes: Sequence[str],
    *,
    validate: Callable[[str], object],
    controls: Sequence[str] = (),
) -> AdviceResidue:
    """Run each probe through layer 4 and split on whether it was refused.

    `validate` is injected so a test can drive both branches without depending on the rule set's
    current contents — the rate is *about* that rule set, so a test that asserted a particular
    number would break every time a rule was added, which is exactly when it should not.

    `controls` are strings the rule set must refuse. They are scored first and **not** counted
    in the rate: they measure the instrument, not its subject. A `controls` that were all let
    through raises `InstrumentDead`, because the alternative is publishing a fail-open as a 100%
    evasion finding — see `AdviceResidue`.
    """

    def refused(text: str) -> bool:
        return bool(getattr(validate(text), "refused", False))

    controls_caught = sum(1 for control in controls if refused(control))
    if controls and not controls_caught:
        raise InstrumentDead(
            f"layer 4 refused none of its {len(controls)} positive control(s), so this pass "
            f"measured a validator that is not running rather than a rule set's ceiling. "
            f"`security.advice.validate_answer` fails open on any exception out of "
            f"`Guard.validate`, and a fail-open reads here as a 100% residue."
        )
    caught, residue = [], []
    for probe in probes:
        (caught if refused(probe) else residue).append(probe)
    return AdviceResidue(
        caught=tuple(caught),
        residue=tuple(residue),
        controls_caught=controls_caught,
        controls=len(controls),
    )


# --- 4. faithfulness of cited sentences (T3/T5) ---------------------------------------

#: Sentence boundaries, deliberately crude. A citation sits at the end of the clause it
#: supports, so splitting on terminal punctuation is enough to pair a marker with the text it
#: attaches to; a full sentence tokeniser would be a new dependency for no gain here.
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

#: A fragment ending here is **not** a sentence end, so the crude split has to be undone.
#:
#: **Crude is fine; truncating is not**, and that distinction is a correction rather than a
#: refinement (code review of #11). A split on every terminal-punctuation-plus-space cuts 10-K
#: prose mid-claim at every abbreviation and every numbered list: this run's own cache holds
#: `"Department of Commerce approval [3][4]."` — the answer read "…subject to U.S. Department of
#: Commerce approval [3][4]" and the split at `U.S. ` removed the subject of the claim before
#: the judge saw it — and `"**More Personal Computing**[1][5]."` cut out of a numbered list.
#: Three of the eleven "no support" pairs in the first artifact were fragments like these, so
#: the published rate was partly a measurement of this pattern.
#:
#: The rule: a fragment whose last token is a known abbreviation, a single initial (`U.S.`,
#: `J.P.`) or a list number (`3.`) is re-joined with the fragment after it. Still no tokeniser
#: and still no dependency — a bounded list of the forms a filing actually uses, which is the
#: same shape as `ingestion/model.py`'s boundary markers.
_ABBREVIATIONS = (
    "approx",
    "cf",
    "Co",
    "Corp",
    "Dr",
    "eg",
    "e.g",
    "est",
    "etc",
    "Fig",
    "ie",
    "i.e",
    "Inc",
    "Jr",
    "Ltd",
    "Mr",
    "Mrs",
    "Ms",
    "No",
    "Nos",
    "plc",
    "Pty",
    "Sr",
    "St",
    "vs",
)
_NOT_A_SENTENCE_END = re.compile(
    r"(?:\b(?:" + "|".join(_ABBREVIATIONS) + r")|\b[A-Z]|\b\d+)\.[\"'’”)\]]*\Z"
)


def _sentences(text: str) -> list[str]:
    """`text` split into sentences, re-joining the crude split's false boundaries.

    See `_NOT_A_SENTENCE_END`: the split itself stays a one-line regex, and the repair is the
    only thing that knows what a filing's prose looks like.
    """
    merged: list[str] = []
    for fragment in _SENTENCE.split(text):
        if merged and _NOT_A_SENTENCE_END.search(merged[-1]):
            merged[-1] = f"{merged[-1]} {fragment}"
        else:
            merged.append(fragment)
    return merged


@dataclass(frozen=True, slots=True)
class CitedSentence:
    """One sentence carrying `[n]`, and the ranks it cites."""

    sentence: str
    ranks: tuple[int, ...]


def cited_sentences(answer: str) -> tuple[CitedSentence, ...]:
    """Every sentence in `answer` that carries at least one numeric marker.

    **This is the narrower question faithfulness does not ask.** RAGAs faithfulness scores a
    statement against the *whole* context set, so a sentence supported by chunk 4 while citing
    chunk 1 is faithful and mis-cited at the same time — which is exactly the case ADR-0006's T7
    amendment §4 leaves open ("whether a resolving marker's chunk supports the sentence remains
    yours"). Pairing each sentence with the chunk it *names* is what makes that answerable.
    """
    found = []
    for sentence in _sentences(answer.strip()):
        # `security.markers` owns what a citation marker is — numeric-only for the same reason
        # it is there (`[Yahoo Finance]` is the *other* deferral's subject), and read from that
        # module so a measurement of the gate cannot disagree with the gate about what a
        # citation *is*.
        ranks = numeric_markers(sentence)
        if ranks:
            found.append(CitedSentence(sentence=sentence.strip(), ranks=ranks))
    return tuple(found)


#: A cited claim counts as supported only when the judge supports **all** of it.
#:
#: ragas faithfulness over one sentence is supported-claims / claims, so a two-claim sentence
#: with one claim the chunk does not support scores exactly 0.5 — and the first version's
#: `verdict >= 0.5` counted that as **supported**, on the boundary, documented nowhere. A
#: marker that names a chunk is a claim that the chunk says this; half of it saying so is not
#: the claim. Partial support is now its own count rather than a rounding decision, which is
#: what makes the boundary disappear instead of moving.
CITED_SUPPORT_FLOOR = 1.0


@dataclass(frozen=True, slots=True)
class CitationSupport:
    """Whether a resolving marker's own chunk supports the claim it is attached to.

    **The unit is one `(sentence, marker)` pair, not one sentence**, and saying so is a
    correction: a sentence carrying `[1][3]` names two chunks and is two claims, so it
    contributes two observations, and the first artifact's "21 cited sentence(s) not supported"
    was 21 pairs. `sentences` is carried beside the pair counts so both denominators are
    visible.
    """

    supported: int
    #: The judge supported *some* of the sentence's claims against the named chunk but not all.
    #: Counted apart rather than rounded either way — see `CITED_SUPPORT_FLOOR`.
    partial: int
    unsupported: int
    #: Markers pointing outside the retrieval — counted apart, since that is the *other*
    #: deferral's failure and structurally prevented since T7's register.
    unresolvable: int
    #: Pairs the judge returned no score for. **Never silently dropped**: the first version
    #: `continue`d past a `None` verdict with no counter at all, so a judge failure would have
    #: narrowed the rate's denominator leaving no trace — the absence-as-measurement failure
    #: `observability.events.Samples.absent` exists to prevent, in a module whose own `Rate`
    #: docstring forbids it.
    unscored: int
    #: Distinct sentences that carried at least one numeric marker — the sentences this pass
    #: **looked at**, which is not the same as the sentences whose markers all resolved. Said
    #: that way because it read "at least one *resolving* marker" while counting before the
    #: resolvability test, so a sentence citing only chunks outside the retrieval inflated the
    #: denominator the artifact prints (code review of #11). The pair counts below are where
    #: resolvability is accounted for.
    sentences: int

    @property
    def rate(self) -> Rate:
        """Fully-supported pairs over every pair the judge scored.

        Partial support sits in the denominator and not the numerator, which is the direction
        that cannot flatter the result.
        """
        return Rate(
            label="cited-marker support",
            hits=self.supported,
            total=self.supported + self.partial + self.unsupported,
        )


def citation_support(
    cells: Sequence[object],
    *,
    arm: str,
    judge_sentence: Callable[[str, str], float | None],
) -> CitationSupport:
    """For each cited sentence, ask whether the chunk it *names* supports it.

    **The narrower question, and why it needs its own pass.** RAGAs faithfulness scores a
    statement against the whole context set, so a sentence supported by chunk 4 while citing
    chunk 1 scores 1.0 — faithful and mis-cited at once. Narrowing the context set to the
    single chunk the marker names turns "is this answer grounded?" into "does *this* source
    say *this*?", which is the question T3 deferred, T5 re-deferred, and ADR-0006's T7
    amendment §4 left open in as many words: "whether a resolving marker's chunk supports the
    sentence remains yours".

    `judge_sentence(sentence, chunk_body)` is injected — the caller wraps the cached
    faithfulness scorer with the context set narrowed to one chunk, which is the whole
    mechanism. Injected so this pass is testable without a judge, and so the scorer stays the
    one that is already tested.

    A marker pointing outside the retrieval is `unresolvable` and stays out of the rate: that
    is the bracket-rule deferral's failure, structurally prevented since T7's register, and
    folding it in would blend two questions with different fixes. A pair the judge could not
    score is `unscored` and likewise out of it — but **counted**, because a denominator that
    narrows itself leaves no trace of having done so.

    **`cells` is structurally typed and its fields are checked, not `getattr`-defaulted.** The
    parameter stays `object` so this module needs no import from `pipeline` — `deferrals` is
    read by the script and by nothing in the stage graph — but `getattr(cell, "arm", None)`
    means a
    renamed `Cell` field selects *no* cell and renders a measured deferral as "not measured
    by this run", which is the absence-as-measurement failure this module's `Rate` docstring
    forbids (code review of #11). So a cell missing a field it needs raises.
    """
    supported = partial = unsupported = unresolvable = unscored = sentences = 0
    for cell in cells:
        for field in ("arm", "answer", "retrieval"):
            if not hasattr(cell, field):
                raise AttributeError(
                    f"a scored cell has no {field!r}: this pass would then match no cell and "
                    f"report the cited-marker deferral as not measured on a run that did."
                )
        if cell.arm != arm:  # type: ignore[attr-defined]
            continue
        answer = cell.answer  # type: ignore[attr-defined]
        if not answer:
            continue
        contexts = cell.retrieval.contexts  # type: ignore[attr-defined]
        for cited in cited_sentences(answer):
            sentences += 1
            for rank in cited.ranks:
                if rank < 1 or rank > len(contexts):
                    unresolvable += 1
                    continue
                verdict = judge_sentence(cited.sentence, contexts[rank - 1].body)
                if verdict is None:
                    unscored += 1
                elif verdict >= CITED_SUPPORT_FLOOR:
                    supported += 1
                elif verdict > 0.0:
                    partial += 1
                else:
                    unsupported += 1
    return CitationSupport(
        supported=supported,
        partial=partial,
        unsupported=unsupported,
        unresolvable=unresolvable,
        unscored=unscored,
        sentences=sentences,
    )
