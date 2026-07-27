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
from collections.abc import Iterable, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from finbrief.config import TICKER_BY_COMPANY_NAME
from finbrief.observability.logging_setup import log_event
from finbrief.prompts import query_translation_prompt

logger = logging.getLogger(__name__)

#: Leading list furniture a model adds however plainly it was asked not to: `- `, `* `, `• `,
#: `1. `, `1) `. Stripped rather than prompted away, because what is left of it gets embedded
#: and BM25-indexed — a query beginning `1. ` carries a term the corpus does not.
_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")

#: Every Universe company name form, longest first, as one alternation.
#:
#: **Longest first is load-bearing.** `Goldman`, `Goldman Sachs` and `The Goldman Sachs Group,
#: Inc.` all map to `GS`; matching the shortest first would rewrite the legal name to `GS Sachs
#: Group, Inc.` — a query naming a company that does not exist, then embedded and BM25-indexed
#: as such.
#:
#: `(?<!\w)` / `(?!\w)` rather than `\b`, because several forms begin or end on a non-word
#: character (`Amazon.com, Inc.`, `J&J`) and `\b` is defined relative to word characters, so it
#: asserts the wrong thing at those edges. The lookarounds are what stop `Meta` firing inside
#: `metabolism`.
_COMPANY_NAME = re.compile(
    r"(?<!\w)(?:"
    + "|".join(
        re.escape(form) for form in sorted(TICKER_BY_COMPANY_NAME, key=len, reverse=True)
    )
    + r")(?!\w)",
    re.IGNORECASE,
)


def normalised(question: str) -> str | None:
    """`How much debt does Tesla carry?` → `…does TSLA carry?`, or `None` if nothing matched.

    The query side of ADR-0004's exact-identifier claim, and the T6 amendment's reason for
    existing. A chunk's provenance header carries the **ticker**: `tsla` is a token on all 280
    of Tesla's chunks, while `tesla` is a token on 34 of them — body mentions only, because a
    filer writes "we". So a lexical retriever handed the *name* sees almost none of the filer,
    which is why hybrid alone made issue #6's case worse rather than better, and why the ticker
    form has to be put in front of BM25 explicitly.

    **Deterministic, and a lookup rather than a model call.** `config.TICKER_BY_COMPANY_NAME` is
    derived from the Universe's own declarations, so nothing is inferred: a form is substituted
    if and only if the Universe declares it. That is what keeps this out of entity-recogniser
    territory, and what lets the `+translation` arm of the A/B gain a variant without gaining
    model variance.

    **One pass over the whole question**, so a comparison naming three filers still produces one
    extra variant rather than three — the added retrieval cost stays flat in the question's
    breadth. Returns `None` when the result would equal the input, since a duplicate variant
    costs an embedding and lets RRF count one candidate list twice.
    """
    rewritten = _COMPANY_NAME.sub(
        lambda match: TICKER_BY_COMPANY_NAME[match.group(0).lower()], question
    )
    return rewritten if rewritten != question else None


def added_variants(variants: Sequence[str]) -> tuple[str | None, tuple[str, ...]]:
    """Split what translation added into `(ticker_form, sub_queries)`.

    `translate` returns one flat tuple, because that is what the retrievers consume and because
    a flat list of strings is the only thing that has to survive the checkpoint. But the two
    kinds of addition are *not* the same fact and must not be reported as one: one is a
    deterministic surface form of the analyst's own words, the other is a model's paraphrase.
    Labelling the ticker form "sub-query 1" in the panel would credit the planner for a lookup,
    and the T6 amendment's finding is about which of the two earns the exact-identifier bucket.

    Recovered by **recomputing** `normalised` on the retained original rather than by carrying a
    tag through the payload. That is sound precisely because normalisation is deterministic and
    depends only on `config` — the property this module is built on — and it keeps the wire form
    a plain list of strings. A reply checkpointed before an alias was added could be mislabelled
    on replay after that deploy; the cost of being wrong is one word in a table.
    """
    if not variants:
        return None, ()
    added = tuple(variants[1:])
    ticker_form = normalised(variants[0])
    if ticker_form is not None and ticker_form in added:
        return ticker_form, tuple(v for v in added if v != ticker_form)
    return None, added


def sub_queries(reply: str, *, known: Iterable[str], limit: int) -> tuple[str, ...]:
    """A model's free text as at most `limit` sub-queries, in the order it offered them.

    Pure, so the parsing can be tested against every plausible formatting of "one per line"
    without a model in the way. Four things are dropped, each because it would otherwise be
    embedded, retrieved and fused as though the analyst had asked it:

    - **list furniture and blank lines** (see `_MARKER`);
    - **a line ending in a colon** — "Here are three sub-queries:" is a preamble, and a genuine
      query does not end in one. Narrow on purpose;
    - **a repeat of anything in `known`**, compared case-insensitively on stripped text: neither
      casing nor whitespace is a translation, and a variant already in play retrieves it
      already. Keeping it would cost a second embedding *and* let RRF count one candidate list
      twice, inflating those chunks' scores on no additional evidence. `known` is every variant
      so far — the original *and* its normalised form — rather than just the question, because a
      planner that echoes the ticker form back would otherwise be retrieved twice;
    - **a repeat of an earlier sub-query**, on the same reasoning.

    `limit=0` keeps nothing, and is checked before the loop rather than inside it: the cap used
    to be tested *after* appending, so `len(kept) == 0` was never true and a zero cap kept
    **every** line the model offered — the exact inverse of this function's contract, in the one
    configuration (`FINBRIEF_MAX_SUB_QUERIES=0`) whose whole point is that the planner
    contributes nothing (issue #6 review). `translate`'s own guard hides that from production;
    this one makes the contract true of the function.
    """
    if limit <= 0:
        return ()
    seen = {variant.strip().casefold() for variant in known}
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
        if len(kept) >= limit:
            break
    return tuple(kept)


def translate(question: str, *, model: BaseChatModel, max_sub_queries: int) -> tuple[str, ...]:
    """The query variants to retrieve over: `question`, its ticker form, then its sub-queries.

    The first element is always `question`, unchanged — the invariant this module exists for.
    Returning it alone is a legal outcome and means translation added nothing, which is what an
    already-atomic question with no Universe name in it deserves.

    **Two kinds of addition, and only one of them costs a generation.** `normalised` is a
    deterministic lookup and runs whenever a Universe company is named; the planner's
    sub-queries are the model's and are capped. So `max_sub_queries=0` still normalises — and
    that combination is an A/B cell in its own right, "hybrid + normalisation, no planner".

    **Variant budget: 1 original + at most 1 normalised + at most `max_sub_queries`.** ADR-0004
    writes the cap as "up to 3 sub-queries", and the normalised form is a fourth thing under
    that arithmetic — recorded in the T6 amendment rather than smuggled in, because ADR-0005's
    latency budget is judged against this count. It is bounded at one however many companies the
    question names, and it adds a retrieval round rather than a chat round.

    Logged by count. A sub-query is derived from the analyst's question and is therefore user
    content, and these lines are kept (`observability/logging_setup.py`).
    """
    started = time.perf_counter()
    variants = [question]
    ticker_form = normalised(question)
    if ticker_form is not None:
        variants.append(ticker_form)
    added: tuple[str, ...] = ()
    if max_sub_queries > 0:
        reply = model.invoke(
            [SystemMessage(query_translation_prompt(max_sub_queries)), HumanMessage(question)]
        )
        added = sub_queries(reply.text, known=variants, limit=max_sub_queries)
        variants.extend(added)
    log_event(
        logger,
        "query_translation",
        # Separated, because they are separately caused: `normalised` is a config lookup that
        # cannot fail, `sub_queries` is a model's output that routinely returns nothing usable.
        # One count would make "the planner is idle" indistinguishable from "no company named".
        normalised=ticker_form is not None,
        sub_queries=len(added),
        max_sub_queries=max_sub_queries,
        variants=len(variants),
        # Sizes, never text — see the docstring. The chars let Phase 7 see whether the model
        # is decomposing or merely restating, without recording either question.
        question_chars=len(question),
        sub_query_chars=[len(sub_query) for sub_query in added],
        latency_ms=round((time.perf_counter() - started) * 1000),
    )
    return tuple(variants)
