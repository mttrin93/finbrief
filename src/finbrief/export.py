"""Exporting a conversation the analyst can keep — JSON and CSV (T12 item 4, #13).

**The source is the display transcript, never the checkpointer**, and that is the decision the
rest of this module follows from: the export is what the analyst *saw*. The two differ in both
directions and neither difference is cosmetic. A question the input gate blocked never reached
`answer()`, so the checkpointer has no memory of it while the page shows the exchange
(`app/Home.py`); and an answer layer 4 refused is *in* the checkpointer while the page shows the
refusal that replaced it (ADR-0006's stated consequence). Exporting the agent's memory would
therefore hand a reader an answer that was withheld from them and omit a refusal they were
given. An export that disagrees with the screen it was taken from is worse than no export.

**Every `[n]` resolves against the file's own source list, and the list is the conversation's
rather than the turn's.** `agent/citations.py` numbers a thread's sources in one running
sequence, so a follow-up may legitimately cite `[1]` from a chunk an *earlier* turn retrieved —
which is the same reason `app/Home.py`'s `issued_ranks` validates markers against the whole
transcript. Scoped per turn, an export would render that citation unresolvable in a file whose
own answers cite it, so `Transcript.sources` is one table keyed by rank and each turn names the
ranks it retrieved. `tests/test_export.py` carries the cross-turn case as the invariant.

**Absences stay absent.** A refusal has no `AgentTurn` behind it, so whether it searched is
*unknown* and not `False`, and no key is written for it — the per-field absence rule
`observability/tokens.py` states, in the module where it is easiest to break, because a
spreadsheet averages a column without asking what its blanks meant. A chunk BM25 recovered has
no vector distance (ADR-0004 §3) and exports `null`, never `0.0`. And nothing here is invented:
the fields are the ones the sources panel renders, because that is what "what the analyst saw"
means.

**No PDF.** It needs a new dependency, and three of the libraries this project added for
quality or safety shipped a telemetry path enabled by default (`guardrails-ai`, `uvloop`,
`ragas` — the README's *Three libraries, three that phone home by default*). A rendering library
is a worse bet than those three, not a better one: it would be added for *presentation*, which
buys none of the argument that made the other three worth their switches. JSON and CSV need no
dependency at all — `json` and `csv` are stdlib — so the export ships with the same egress
surface the page already had.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from finbrief.observability.logging_setup import log_event
from finbrief.retrieval.retrieve import Context

#: What the JSON calls itself. A reader that finds a file in a downloads folder six months later
#: has the shape's name and the version in the file rather than in a filename someone renamed.
EXPORT_FORMAT = "finbrief-transcript"

#: Bumped when a consumer would have to change. Additive fields do not move it — a reader of
#: version 1 tolerates a new key for the same reason every `from_payload` on the checkpoint path
#: does.
EXPORT_VERSION = 1

#: The CSV's columns, in order. **One shape: a row per (turn, source that turn retrieved.)**
#:
#: The alternative was a row per turn with the sources packed into a cell, and this is the
#: better trade for the one property that matters: every source is a row carrying its own
#: `rank`, so an `[n]` resolves by scanning one column instead of by parsing a list out of a
#: cell — in a format whose whole point is that a spreadsheet can read it. The cost is that a
#: turn's answer repeats across its source rows, which is the ordinary price of a long format
#: and is stated here rather than discovered.
#:
#: A turn that retrieved nothing still gets a row, with the source columns *empty*. It said
#: something, so it belongs in the file; and an empty cell is an absence where `0` would be a
#: measurement.
CSV_COLUMNS: tuple[str, ...] = (
    "turn_index",
    "role",
    "content",
    "searched",
    "source_rank",
    "citation",
    "chunk_id",
    "ticker",
    "section",
    "fiscal_year",
    "distance",
    "retrievers",
    "body",
)

#: The characters a spreadsheet executes when they open a cell. Excel and Sheets both evaluate
#: `=`, `+`, `-` and `@` on open, and every text cell here is either model output or a filer's
#: prose — neither of which this repo writes. Prefixed rather than stripped, so the text
#: survives intact and still round-trips through `csv.reader`, which is the consumer this
#: format is specified against.
_FORMULA_LEADERS = frozenset("=+-@")

#: How much of a `thread_id` stands in for the conversation in something a human reads.
#:
#: **One constant for two surfaces**, because they are the same fact: the sidebar's `Thread
#: 3294dcff` caption and this module's file names have to name a conversation the same way, or a
#: file in a downloads folder cannot be matched to the handle the page showed — or to the
#: `turn_id` prefix on its log lines, which is the whole reason the handle is displayed at all.
THREAD_HANDLE_CHARS = 8

logger = logging.getLogger(__name__)


def file_name(thread_id: str, suffix: str) -> str:
    """`finbrief-3294dcff.json` — what the download is called on disk.

    Prefixed with the conversation's handle so two tabs' exports do not overwrite each other in
    a downloads folder, and so a file found later can be matched to the conversation the sidebar
    named and the log lines carry.

    Here rather than inline in `app/Home.py` because `AppTest` cannot see a download button's
    file name — it is not on the proto, since the bytes are served over a URL — so a name built
    in the app is a claim no test can reach. `tests/test_export.py` binds it.
    """
    return f"finbrief-{thread_id[:THREAD_HANDLE_CHARS]}.{suffix}"


@dataclass(frozen=True, slots=True)
class ExportedSource:
    """One entry of the export's resolution table — a chunk, as the sources panel showed it.

    The fields are the panel's, deliberately: `citation` and `body` are what a reader checks an
    `[n]` against, and `chunk_id`, `distance` and `retrievers` are the provenance printed beside
    it. Nothing is added that the analyst did not see, and `fused_score` and the per-variant
    `provenance` rows are left out for the same reason — they belong to *How I answered*, which
    is a different artifact from the one this replaces.
    """

    rank: int
    citation: str
    chunk_id: str
    ticker: str
    section: str
    fiscal_year: int
    distance: float | None
    retrievers: tuple[str, ...]
    body: str

    @classmethod
    def of(cls, context: Context) -> ExportedSource:
        """From a `Context`, flattening every enum to the string it prints as.

        **Enums are converted here rather than left to `json.dumps`**, which is the trap
        `tools/finance.py` records for a card's artifact: a `StrEnum` *is* a `str`, so a
        `Section` serialises without complaint and comes back as a bare string that no
        longer round-trips to the enum. Converting at the boundary makes the leaf type the
        declared one.
        """
        return cls(
            rank=context.rank,
            citation=context.citation,
            chunk_id=context.chunk_id,
            ticker=context.ticker,
            section=str(context.section.value),
            fiscal_year=int(context.fiscal_year),
            distance=context.distance,
            retrievers=tuple(str(one.value) for one in context.retrievers),
            body=context.body,
        )


@dataclass(frozen=True, slots=True)
class ExportedTurn:
    """One row of the transcript as it was rendered.

    `searched` and `retrieved_ranks` are `None` for a turn there is nothing to say it about — a
    refusal the agent never produced, or a row written by a build that stored neither. `None`
    here means *omitted* from the output, not `false` and not `[]`: "the agent did not search"
    and "we cannot say whether it searched" are different claims, and only the first is a fact
    about a turn.
    """

    role: str
    content: str
    searched: bool | None = None
    retrieved_ranks: tuple[int, ...] | None = None

    def as_payload(self) -> dict[str, Any]:
        """The JSON object for this turn, carrying no key it has no value for."""
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.searched is not None:
            payload["searched"] = self.searched
        if self.retrieved_ranks is not None:
            payload["retrieved_ranks"] = list(self.retrieved_ranks)
        return payload


@dataclass(frozen=True, slots=True)
class Transcript:
    """A conversation ready to render: the turns, and the table their `[n]`s resolve into."""

    thread_id: str
    turns: tuple[ExportedTurn, ...]
    sources: tuple[ExportedSource, ...]

    @classmethod
    def of(cls, messages: Iterable[Mapping[str, Any]], *, thread_id: str) -> Transcript:
        """Build from `st.session_state.messages`.

        **The row shapes and their precedence are `app/Home.py`'s replay loop's**, read in the
        same order for the same reason: a live session's transcript outlives a deploy, so
        rows written by an older build are still in `session_state` on the first rerun after
        one. If the export read them in a different order the file and the screen would
        describe the same row differently, which is the class of disagreement this repo keeps
        finding.
        """
        turns: list[ExportedTurn] = []
        # Keyed by rank so a chunk two turns both surfaced is one source with one number: two
        # entries under one rank would make the table ambiguous at exactly the rank being looked
        # up. First writer wins, which is the turn that was numbered first.
        sources: dict[int, ExportedSource] = {}
        for message in messages:
            role = str(message.get("role", ""))
            content = str(message.get("content", ""))
            contexts, searched = _grounding(message)
            for context in contexts or ():
                sources.setdefault(context.rank, ExportedSource.of(context))
            turns.append(
                ExportedTurn(
                    role=role,
                    content=content,
                    searched=searched,
                    retrieved_ranks=(
                        None if contexts is None else tuple(c.rank for c in contexts)
                    ),
                )
            )
        return cls(
            thread_id=thread_id,
            turns=tuple(turns),
            # Sorted by the number that cites them: the table is a lookup, and a reader
            # resolving `[7]` scans for 7. Insertion order is search order and usually agrees,
            # but a follow-up citing an earlier chunk touches them out of sequence.
            sources=tuple(sources[rank] for rank in sorted(sources)),
        )


def _grounding(
    message: Mapping[str, Any],
) -> tuple[tuple[Context, ...] | None, bool | None]:
    """What a transcript row says it retrieved, and whether it searched — or `None` for neither.

    The three shapes `app/Home.py` renders, in its order. `getattr` on the turn for the reason
    every reader on that path is tolerant: a row holding an `AgentTurn` built before a field
    existed replays here too.
    """
    if message.get("role") != "assistant":
        return None, None
    turn = message.get("turn")
    if turn is not None:
        return tuple(getattr(turn, "contexts", ())), bool(getattr(turn, "searched", False))
    contexts = message.get("contexts")
    if contexts is not None:
        # `searched` defaults True exactly as the replay loop defaults it: before the agent
        # existed contexts were always retrieved, so an empty tuple in such a row really does
        # mean an empty collection.
        return tuple(contexts), bool(message.get("searched", True))
    return None, None


def as_json(transcript: Transcript) -> str:
    """The conversation as one JSON object, indented for a human to open and read."""
    payload = {
        "format": EXPORT_FORMAT,
        "version": EXPORT_VERSION,
        "thread_id": transcript.thread_id,
        "turns": [turn.as_payload() for turn in transcript.turns],
        # One list, at the top level, holding every body once — which is what makes it the
        # resolution table rather than a per-turn copy that could disagree with itself.
        "sources": [
            {
                "rank": source.rank,
                "citation": source.citation,
                "chunk_id": source.chunk_id,
                "ticker": source.ticker,
                "section": source.section,
                "fiscal_year": source.fiscal_year,
                "distance": source.distance,
                "retrievers": list(source.retrievers),
                "body": source.body,
            }
            for source in transcript.sources
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def as_csv(transcript: Transcript) -> str:
    """The conversation as CSV — one row per (turn, source it retrieved). See `CSV_COLUMNS`.

    Written through `csv.writer` rather than by joining strings, which is not a style
    preference: a filing body carries commas by the hundred, `"` around defined terms and
    blank lines between paragraphs, and a hand-rolled join corrupts the file on the first one
    of those.
    `\\r\\n` is the dialect's own line terminator and `csv.reader` handles it on every platform.
    """
    out = io.StringIO(newline="")
    writer = csv.writer(out)
    writer.writerow(CSV_COLUMNS)
    by_rank = {source.rank: source for source in transcript.sources}
    for index, turn in enumerate(transcript.turns):
        retrieved = [by_rank[rank] for rank in (turn.retrieved_ranks or ()) if rank in by_rank]
        # A turn with no sources still gets one row: `[None]` is the "and nothing to say about
        # its sources" case, not a row that is skipped.
        for source in retrieved or [None]:
            writer.writerow(_row(index, turn, source))
    return out.getvalue()


def _row(index: int, turn: ExportedTurn, source: ExportedSource | None) -> list[str]:
    """One CSV record. Every absent value is `""` — an empty cell, never a stand-in."""
    return [
        str(index),
        turn.role,
        _text(turn.content),
        "" if turn.searched is None else str(turn.searched).lower(),
        *(
            ["", "", "", "", "", "", "", "", ""]
            if source is None
            else [
                str(source.rank),
                _text(source.citation),
                source.chunk_id,
                source.ticker,
                source.section,
                str(source.fiscal_year),
                "" if source.distance is None else f"{source.distance:.6f}",
                " + ".join(source.retrievers),
                _text(source.body),
            ]
        ),
    ]


def _text(value: str) -> str:
    """`value`, guarded against a spreadsheet reading it as a formula (`_FORMULA_LEADERS`)."""
    return f"'{value}" if value[:1] in _FORMULA_LEADERS else value


def log_export(transcript: Transcript, *, fmt: str, size: int) -> None:
    """Record that an export happened — **a count and a format, and never the payload.**

    The transcript is the analyst's own questions and the filer's prose, which is exactly what
    `log_event` may not carry: these lines are kept, and the no-user-content rule has one
    bounded exception and it is a blocked question's normalised text (ADR-0006, ADR-0011). So
    what goes
    on the line is how many turns and sources went out, in which format, at what size — enough
    to answer "is anyone using this, and how big do these get" and nothing that reconstructs a
    conversation.
    """
    log_event(
        logger,
        "transcript_export",
        format=fmt,
        turns=len(transcript.turns),
        sources=len(transcript.sources),
        bytes=size,
    )
