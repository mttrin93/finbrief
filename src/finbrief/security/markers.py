"""The back door's deterministic half: does every `[n]` in an answer name a retrieved source?

**T3's second deferred finding, taken here** (issue #5, recorded on #8). Half the citation
contract is structural and holds without help: `Context.rank` is assigned once — by
`retrieve()`, renumbered once per thread by `agent/citations.py` — and read by exactly two
surfaces, the prompt's `<sources>` block and the app's sources panel, both iterating the same
tuple. So every *panel* entry `[n]` names exactly one chunk and keeps naming it across reruns.

The **citing** half was not enforced. Nothing parsed the markers out of the answer, so a model
that wrote `[6]` against five contexts produced a marker pointing at no entry: no exception, no
warning, no log line, and a reader who tried to check the claim could not tell a hallucinated
citation from a numbering slip. This module is that parse. It costs no model call, which is why
it fits inside a ticket whose latency budget is about the *input* gate.

**What a violation does: it is reported, not repaired.** Three options were on the table — strip
the dangling marker, refuse the answer, log only — and the choice is to *surface* it beside the
answer and log it.

- Refusing a whole answer over one bad marker throws away work that is mostly correct, and it
  would make a numbering slip indistinguishable to the reader from the advice refusal layer 4
  gives, which is a different thing entirely.
- Stripping the marker silently is worse than leaving it: the reader loses the *evidence*
  that the
  answer cited something unverifiable, which is precisely the fact they need. CLAUDE.md's rule
  is that an absence must not be reported as a measurement, and a quietly-cleaned answer reports
  "every claim here is cited" about prose where one was not.
- So the marker stays, and a caption under the answer says which numbers resolve to nothing.

**Two failures, counted apart, because they are different defects.** An `unresolved` marker is a
number naming no retrieved chunk — the T3 finding. A `non_numeric` bracket is `[Yahoo Finance]`,
which T5 measured the model writing in prose despite `AGENT_SYSTEM_PROMPT` reserving square
brackets for filing excerpts: not a resolution failure but a *collision* with the syntax that
makes resolution possible, and a reader who sees both cannot resolve either.

**What this does not check, deliberately: whether the cited chunk supports the claim.** A marker
that resolves can still sit at the end of a paragraph of tool figures the chunk says nothing
about — T5 recorded the model doing exactly that. That is **faithfulness**, not marker
resolution, and it is out of scope here by construction rather than by omission: deciding
whether a chunk supports a sentence needs a judge model and ground truth, which is ADR-0002's
golden set and RAGAs faithfulness (T9, #4). The bracket-rule adherence *rate* is T10's (#11),
beside the verbatim-query divergence it already reports. Stated as a limitation in the README
rather than left implied.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass

from finbrief.observability.logging_setup import log_event

logger = logging.getLogger(__name__)

#: Any bracketed span, numeric or not. Deliberately **not** `\[(\d+)\]` alone: the non-numeric
#: case is half of what this module reports, and a pattern that only matched numbers could not
#: see it. Bounded to 40 characters so a stray `[` in prose cannot swallow a paragraph, and to a
#: single line for the same reason.
_BRACKETED = re.compile(r"\[([^\[\]\n]{1,40})\]")

#: A marker's contents when it is a citation: one or more digits, nothing else. `[1][3]` is two
#: markers by this reading, which is the compound form `_ANSWER_RULES` asks for.
_NUMERIC = re.compile(r"\A\d+\Z")


@dataclass(frozen=True, slots=True)
class MarkerReport:
    """What an answer's square brackets resolve to.

    Sorted tuples rather than sets, so a log line and a caption are stable across runs and two
    reports of the same answer compare equal.
    """

    #: Numbers the answer cited that name no retrieved chunk — the T3 finding.
    unresolved: tuple[int, ...] = ()
    #: Bracketed spans that are not numbers at all, in first-appearance order — the T5 finding.
    non_numeric: tuple[str, ...] = ()
    #: How many well-formed, resolving markers the answer carried. Reported because "two
    #: markers, one unresolved" and "eleven markers, one unresolved" are different-sized
    #: problems, and because a grounded answer with **zero** markers is its own finding for T10.
    resolved: int = 0

    @property
    def clean(self) -> bool:
        """Whether every bracket in the answer resolves to a source a reader can check."""
        return not self.unresolved and not self.non_numeric


def markers(answer: str, *, ranks: Iterable[int]) -> MarkerReport:
    """Read the answer's brackets against the ranks this turn actually retrieved. Pure.

    `ranks` is the caller's — `{context.rank for context in turn.contexts}` on the agent path,
    `GroundedAnswer.contexts` on the chain's. Passed in rather than derived here because
    `agent/citations.py` is the only place a citation number is assigned, and a module that
    recomputed the valid set would be a second opinion about it.

    Note what "resolves" means and does not: that the number names a chunk this turn retrieved.
    It says nothing about whether that chunk supports the sentence the marker is attached to —
    see the module docstring.
    """
    valid = set(ranks)
    unresolved: list[int] = []
    non_numeric: list[str] = []
    resolved = 0
    for match in _BRACKETED.finditer(answer):
        inner = match.group(1).strip()
        if not _NUMERIC.match(inner):
            if inner not in non_numeric:
                non_numeric.append(inner)
            continue
        number = int(inner)
        if number in valid:
            resolved += 1
        elif number not in unresolved:
            unresolved.append(number)
    return MarkerReport(
        unresolved=tuple(sorted(unresolved)),
        non_numeric=tuple(non_numeric),
        resolved=resolved,
    )


def log_markers(report: MarkerReport, *, thread_id: str, sources: int) -> None:
    """Record one turn's citation-marker verdict, for T10 (#11) to report a rate from.

    Emitted on **every** turn that retrieved something, not only on a violation: a rate needs a
    denominator, and a log that only carried the failures would let a reader compute "how often
    is a marker unresolved" only against a count from somewhere else.

    The numbers are the answer's own citation numbers and this module's counts — no prose, so
    nothing here is user content. `non_numeric` is the one field carrying model-written text,
    and it is a *count* here rather than the spans: `[Yahoo Finance]` is harmless, but the
    pattern admits up to 40 characters of anything and these lines are kept. The spans reach the
    reader on screen, where they belong.
    """
    log_event(
        logger,
        "citation_markers",
        thread_id=thread_id,
        sources=sources,
        resolved=report.resolved,
        unresolved=list(report.unresolved),
        non_numeric=len(report.non_numeric),
        clean=report.clean,
    )
