"""Layer 3 of the input gate: one zero-shot LLM classifier (ADR-0006).

**What this layer contributes.** Novel phrasings — the payloads layer 2 has no rule for, because
nobody has written them down yet. It is the only layer that can catch a wording invented after
this file was, which is why a denylist pass *always* escalates here rather than being treated as
a verdict.

**One model call per turn, on its own cheap model.** `Settings.classifier_model` is a separate
field from `chat_model` for the reason its docstring gives: the gate pays for one YES/NO per
turn and the answering model is what a brief is worth. The call is narrowed at both ends — a
5-second ceiling and no retry (`config.GATE_TIMEOUT_SECONDS`, `GATE_CLASSIFIER_ATTEMPTS`) —
because it sits inside a p50 budget the answering path does not have
(`config.GATE_LATENCY_BUDGET_MS`).

**It fails open, and that is a decision with a cost.** A provider outage, a timeout, or a reply
this module cannot parse produces `Verdict.UNDECIDED`, which `input_gate.screen` allows while
logging a warning. Failing *closed* would turn an OpenRouter blip into an assistant that refuses
every question and looks, to the analyst, like censorship rather than an outage. Failing open
leaves three layers standing — normalisation and the denylist have already run, the persona's
boundaries are in the system prompt, and the output validator still guards the exit — so the
loss is the *novel-phrasing* coverage of one turn, which is a proportionate amount to lose to a
503. It is a stated limitation, not an accident: a gate that is silently absent whenever its
provider is having a bad afternoon is worth knowing about.
"""

from __future__ import annotations

import logging
from enum import StrEnum

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from finbrief.caching import build_once
from finbrief.config import (
    GATE_CLASSIFIER_ATTEMPTS,
    GATE_LOGGED_CLASSIFIER_TOKEN_MAX_CHARS,
    GATE_TIMEOUT_SECONDS,
    Settings,
    get_settings,
)
from finbrief.llm import build_chat_model
from finbrief.observability.logging_setup import log_event
from finbrief.prompts import (
    CLASSIFIER_INJECTION_LABEL,
    CLASSIFIER_SAFE_LABEL,
    INJECTION_CLASSIFIER_PROMPT,
    classifier_input,
)

logger = logging.getLogger(__name__)


class Verdict(StrEnum):
    """What the classifier concluded — three values, because "no answer" is not "safe".

    `UNDECIDED` exists so the fail-open decision is *visible* in the return type and in the log,
    rather than being a `False` that reads identically to a model saying the input is fine. A
    caller that wanted to fail closed one day only has to change how it treats this member.
    """

    INJECTION = "injection"
    SAFE = "safe"
    UNDECIDED = "undecided"


def _classifier_model(settings: Settings) -> BaseChatModel:
    """The gate's chat model, built once per process (`caching.build_once`).

    Through `build_once` rather than `lru_cache` for the reason CLAUDE.md records about the
    other process-level singletons on this path: `lru_cache` is atomic about its bookkeeping and
    says nothing about the constructor it wraps, and two Streamlit session threads screening
    their first question at the same time both miss a cold key.
    """
    return build_chat_model(
        settings,
        model=settings.classifier_model,
        timeout=GATE_TIMEOUT_SECONDS,
        # `max_retries` is the count of *retries*, so one attempt means none.
        max_retries=GATE_CLASSIFIER_ATTEMPTS - 1,
    )


#: The process-level handle. Keyed on the frozen `Settings`, like the others.
classifier_model = build_once(_classifier_model)


def classify(
    question: str,
    *,
    model: BaseChatModel | None = None,
    settings: Settings | None = None,
) -> Verdict:
    """Ask the classifier whether `question` is an injection attempt. Never raises.

    `model` is injectable so the suite can drive this layer with a scripted reply (spec seam 4:
    "the classifier layer via mocked responses in unit runs, live only in the cached security-
    suite evals"). Production passes neither argument.

    Every failure path lands on `UNDECIDED` — including a model that answers a sentence instead
    of a word, which is the one this module cannot distinguish from a jailbroken classifier and
    therefore must not read charitably. Nothing about the reply is logged except whether it
    parsed: the reply is one token and the *question* is user content.
    """
    settings = settings or get_settings()
    chat = model if model is not None else classifier_model(settings)
    try:
        reply = chat.invoke(
            [
                SystemMessage(INJECTION_CLASSIFIER_PROMPT),
                HumanMessage(classifier_input(question)),
            ]
        )
    except Exception as exc:  # noqa: BLE001 — every provider failure is one fail-open decision
        # The exception *type*, never its message: a client error string can carry a request
        # URL, and a request URL can carry an API key (`observability/logging_setup.py`).
        log_event(
            logger,
            "gate_classifier_unavailable",
            level=logging.WARNING,
            error=type(exc).__name__,
            question_chars=len(question),
        )
        return Verdict.UNDECIDED
    return _verdict(reply.text)


def _verdict(reply: str) -> Verdict:
    """Read one of the two labels off the model's reply, or `UNDECIDED`.

    **The first alphabetic token, not a substring search.** `CLASSIFIER_SAFE_LABEL in reply`
    says `NO` about "this is NOT an injection", and `"YES" in reply` says injection about "the
    answer is not YES" — a parser that can be talked out of its verdict by the shape of a
    sentence. The prompt asks for exactly one word; a reply that is not one word is a reply this
    module does not understand, and an unparsed reply is `UNDECIDED` rather than a guess.

    **Reduced to its letters, not stripped of a list of punctuation.** The strip list was
    `.,:;!?"'`, so `**YES**`, `` `YES` `` and `- NO` all landed on `UNDECIDED` — and because
    this layer fails open, a model that merely *bolds* its one-word answer disables layer 3 for
    every turn, silently, with nothing but a warning in the log (issue #8 review). Keeping only
    the alphabetic characters, and skipping a purely non-alphabetic lead token, cannot make the
    parser more permissive about *which* word it read: the comparison is still against one whole
    token and still against exactly two labels.
    """
    # The first token that has any letters in it — a leading `-`, `*` or `1.` is a bullet, not
    # the model's answer. Tokens after that one are never considered, which is the discipline
    # that keeps "the answer is not YES" unparsed rather than read as a verdict.
    token = next(
        (
            letters.upper()
            for word in reply.split()
            if (letters := "".join(c for c in word if c.isalpha()))
        ),
        "",
    )
    if token == CLASSIFIER_INJECTION_LABEL:
        return Verdict.INJECTION
    if token == CLASSIFIER_SAFE_LABEL:
        return Verdict.SAFE
    log_event(
        logger,
        "gate_classifier_unparsed",
        level=logging.WARNING,
        # **Bounded as user-derived text, not exempted as the model's own vocabulary.** The
        # comment here used to say the token was "the classifier's own output vocabulary, not
        # the analyst's text" — true of a compliant model, and false in this branch, which is
        # reached only when the reply was *not* one of the two labels. That is precisely when a
        # confused classifier may be echoing the question back (issue #8 review). It is kept
        # because it is the only way to tell a model that ignored the format from one that was
        # never asked, and capped in `config` like every other logged input.
        token=token[:GATE_LOGGED_CLASSIFIER_TOKEN_MAX_CHARS],
        reply_chars=len(reply),
    )
    return Verdict.UNDECIDED
