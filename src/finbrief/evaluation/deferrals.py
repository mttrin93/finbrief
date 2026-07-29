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
    """

    turns_with_sources: int
    uncited: int
    unresolved: int
    non_numeric: int

    @property
    def clean(self) -> Rate:
        """Turns with sources whose markers were all numeric, resolving, and present."""
        bad = self.uncited + self.unresolved + self.non_numeric
        return Rate(
            label="bracket-rule adherence",
            hits=max(self.turns_with_sources - bad, 0),
            total=self.turns_with_sources,
        )


def bracket_adherence(log: EventLog) -> BracketAdherence:
    """Adherence over `citation_markers` lines from turns that had at least one source.

    Turns with no sources are excluded from the denominator rather than counted as compliant: an
    answer with nothing to cite cannot break a citation rule, and including it would inflate the
    rate by the number of questions the collection could not answer.
    """
    turns = uncited = unresolved = non_numeric = 0
    for event in log.of("citation_markers"):
        sources = event.field("sources")
        if not sources:
            continue
        turns += 1
        if not event.field("resolved"):
            uncited += 1
        unresolved += len(event.field("unresolved") or ())
        non_numeric += int(event.field("non_numeric") or 0)
    return BracketAdherence(
        turns_with_sources=turns,
        uncited=uncited,
        unresolved=unresolved,
        non_numeric=non_numeric,
    )


# --- 3. layer 4's residue (T7) — free and deterministic -------------------------------


@dataclass(frozen=True, slots=True)
class AdviceResidue:
    """Hand-labelled advice the validator does **not** refuse — layer 4's ceiling.

    `ADVICE_ANSWERS` measures the floor (does it catch the advice that announces itself?) and
    the security suite already reports that. This measures the other half: advice carrying no
    imperative, no rating word, no price target and no position-sizing instruction, which any
    analyst would still read as being told what to do.
    """

    caught: tuple[str, ...]
    residue: tuple[str, ...]

    @property
    def rate(self) -> Rate:
        total = len(self.caught) + len(self.residue)
        return Rate(label="layer-4 residue", hits=len(self.residue), total=total)


def advice_residue(
    probes: Sequence[str], *, validate: Callable[[str], object]
) -> AdviceResidue:
    """Run each probe through layer 4 and split on whether it was refused.

    `validate` is injected so a test can drive both branches without depending on the rule set's
    current contents — the rate is *about* that rule set, so a test that asserted a particular
    number would break every time a rule was added, which is exactly when it should not.
    """
    caught, residue = [], []
    for probe in probes:
        verdict = validate(probe)
        (caught if getattr(verdict, "refused", False) else residue).append(probe)
    return AdviceResidue(caught=tuple(caught), residue=tuple(residue))


# --- 4. faithfulness of cited sentences (T3/T5) ---------------------------------------

#: An inline citation marker: `[1]`, `[12]`. The same shape `security/markers.py` parses, and
#: numeric-only for the same reason — `[Yahoo Finance]` is the *other* deferral's subject.
_MARKER = re.compile(r"\[(\d+)\]")

#: Sentence boundaries, deliberately crude. A citation sits at the end of the clause it
#: supports, so splitting on terminal punctuation is enough to pair a marker with the text it
#: attaches to; a full sentence tokeniser would be a new dependency for no gain here.
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


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
    for sentence in _SENTENCE.split(answer.strip()):
        ranks = tuple(int(match) for match in _MARKER.findall(sentence))
        if ranks:
            found.append(CitedSentence(sentence=sentence.strip(), ranks=ranks))
    return tuple(found)


@dataclass(frozen=True, slots=True)
class CitationSupport:
    """How often a resolving marker's own chunk supports the sentence carrying it."""

    supported: int
    unsupported: int
    #: Markers pointing outside the retrieval — counted apart, since that is the *other*
    #: deferral's failure and structurally prevented since T7's register.
    unresolvable: int

    @property
    def rate(self) -> Rate:
        return Rate(
            label="cited-sentence support",
            hits=self.supported,
            total=self.supported + self.unsupported,
        )
