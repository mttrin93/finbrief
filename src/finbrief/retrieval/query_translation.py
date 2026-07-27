"""Query translation: the analyst's question, plus up to three sub-queries (ADR-0004).

One rule, and everything here exists to hold it: **translation only ever adds.** The original
question is always the first variant, so BM25 always sees the raw identifiers the analyst typed
— a ticker, `Item 1A`, a ratio name. A rewrite that *replaced* the question could strip exactly
those, degrading the `exact-identifier` bucket hybrid search is in the pipeline to serve, and
ADR-0004 rejects that asymmetry on those grounds.

What translation is *for* is the other end of the spectrum: "is Tesla in trouble?" has no good
nearest neighbour, because no passage in a 10-K is about being in trouble. Decomposing it into
the questions a filing does answer — risk factors, liquidity, results — is what gives the
retrievers something to match, and ADR-0004 predicts that as the `multi-hop` bucket's win.

**A failed translation raises.** It does not degrade to no-translation, which is the instinct.
`±translation` is one axis of the measured A/B (ADR-0002), so a silent fallback would report a
translation-enabled number for a retrieval where translation never ran — the same failure
ADR-0005 refuses when it makes `hybrid` raise rather than serve vector results under a hybrid
label. Phase 5 owns turning it into a graceful UI failure; what it must not become is an
invisible one.

The chat model is injected, never built here: `llm.py` is the only chat-model constructor
(CLAUDE.md), and the prompt is `prompts.py`'s, which owns everything the model reads.
"""

from __future__ import annotations

import logging
import re
import time

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from finbrief.observability.logging_setup import log_event
from finbrief.prompts import query_translation_prompt

logger = logging.getLogger(__name__)

#: Leading list furniture a model adds however plainly it was asked not to: `- `, `* `, `• `,
#: `1. `, `1) `. Stripped rather than prompted away, because what is left of it gets embedded
#: and BM25-indexed — a query beginning `1. ` carries a term the corpus does not.
_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def sub_queries(reply: str, *, question: str, limit: int) -> tuple[str, ...]:
    """A model's free text as at most `limit` sub-queries, in the order it offered them.

    Pure, so the parsing can be tested against every plausible formatting of "one per line"
    without a model in the way. Four things are dropped, each because it would otherwise be
    embedded, retrieved and fused as though the analyst had asked it:

    - **list furniture and blank lines** (see `_MARKER`);
    - **a line ending in a colon** — "Here are three sub-queries:" is a preamble, and a genuine
      query does not end in one. Narrow on purpose;
    - **a repeat of the question**, compared case-insensitively on stripped text: neither casing
      nor whitespace is a translation, and the retained original already retrieves it. Keeping
      it would cost a second embedding *and* let RRF count one candidate list twice, inflating
      those chunks' scores on no additional evidence;
    - **a repeat of an earlier sub-query**, on the same reasoning.
    """
    seen = {question.strip().casefold()}
    kept: list[str] = []
    for line in reply.splitlines():
        candidate = _MARKER.sub("", line).strip()
        if not candidate or candidate.endswith(":"):
            continue
        fingerprint = candidate.casefold()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        kept.append(candidate)
        if len(kept) == limit:
            break
    return tuple(kept)


def translate(question: str, *, model: BaseChatModel, max_sub_queries: int) -> tuple[str, ...]:
    """The query variants to retrieve over: `question` first, then its sub-queries.

    The first element is always `question`, unchanged — the invariant this module exists for.
    Returning it alone is a legal outcome and means translation added nothing, which is what an
    already-atomic question deserves.

    `max_sub_queries=0` skips the generation entirely rather than paying for one whose every
    line is then discarded; `config.py` permits that value, so it has to mean something.

    Logged by count. A sub-query is derived from the analyst's question and is therefore user
    content, and these lines are kept (`observability/logging_setup.py`).
    """
    if max_sub_queries <= 0:
        return (question,)
    started = time.perf_counter()
    reply = model.invoke(
        [SystemMessage(query_translation_prompt(max_sub_queries)), HumanMessage(question)]
    )
    added = sub_queries(reply.text, question=question, limit=max_sub_queries)
    log_event(
        logger,
        "query_translation",
        sub_queries=len(added),
        max_sub_queries=max_sub_queries,
        # Sizes, never text — see the docstring. The chars let Phase 7 see whether the model
        # is decomposing or merely restating, without recording either question.
        question_chars=len(question),
        sub_query_chars=[len(sub_query) for sub_query in added],
        latency_ms=round((time.perf_counter() - started) * 1000),
    )
    return (question, *added)
