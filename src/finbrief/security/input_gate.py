"""The front door: normalise → denylist → classifier, cheap-first (ADR-0006).

`screen()` is the seam. It is the whole of the input side of the gate, and the *composition* is
what it owns — each layer is somebody else's module, and this file is the argument for the order
they run in:

1. **Normalisation is not a verdict**, it is what the next layer matches against. It never
   blocks anything on its own, which is why `Layer` has no member for it and why its
   contribution has
   to be measured at its own seam (`tests/test_normalization.py`) rather than through a verdict.
2. **A denylist catch exits early.** It cost nothing, it is certain, and paying for a model call
   to confirm a match on `ignore all previous instructions` is spending money to learn nothing.
3. **A denylist pass always escalates.** This is the load-bearing half of the ADR and the
   easiest thing to get wrong: a pass is not "safe", it is "not one of the payloads we wrote
   down". The escalation is unconditional — there is no confidence score, no length threshold,
   no short-circuit on a question that looks fine — because every one of those would make the
   denylist the effective gate and layer 3 an optimisation.

**One model call per turn, and the budget is reported rather than enforced.** `latency_ms` on
every screening is what `docs/verification/security-gate.md` computes the ≤800ms p50 from
(`config.GATE_LATENCY_BUDGET_MS`). Nothing here abandons a slow call to make the number look
better; the only ceiling is the classifier's own circuit breaker.

**Where this runs.** `app/Home.py`, before the question reaches `agent.answer` — the same door
`config.MAX_QUESTION_CHARS` is checked at, and the only one a human types through. Deliberately
*not* inside `rag.answer_question` or the agent loop: ADR-0003 keeps the measured chain callable
with nothing in the way, and a gate in the agent would screen the model's own tool arguments as
if an analyst had typed them.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import StrEnum

from langchain_core.language_models import BaseChatModel

from finbrief.config import GATE_LOGGED_INPUT_MAX_CHARS, Settings
from finbrief.observability.logging_setup import log_event
from finbrief.security.classifier import Verdict, classify
from finbrief.security.denylist import denylisted
from finbrief.security.normalize import normalise

logger = logging.getLogger(__name__)


class Layer(StrEnum):
    """A layer that can *decide* a screening.

    Two members, not three: normalisation decides nothing — it is the transform layer 2 matches
    against — so a `Layer` naming it would be a value no screening could ever carry. Its
    contribution is real and is measured where it happens (`normalize.py`).
    """

    DENYLIST = "denylist"
    CLASSIFIER = "classifier"


@dataclass(frozen=True, slots=True)
class Screening:
    """What the gate decided about one question, and what it cost.

    Carries no user-derived text, deliberately: the normalised form exists inside `screen()`
    long enough to be matched and — on a block only — logged, and never travels out on this
    object. That keeps it off every surface by construction rather than by every surface
    remembering.
    """

    blocked: bool
    #: Which layer decided to block. `None` when the question was allowed — including when the
    #: classifier was `UNDECIDED`, because no layer decided anything then.
    layer: Layer | None = None
    #: The `denylist.Rule.id` that fired, when `layer` is `DENYLIST`.
    rule: str | None = None
    #: Whether layer 3 was reached at all. `False` on a denylist catch — which is the
    #: cheap-first property, and the number the marginal-contribution table's "caught for free"
    #: column is.
    classifier_ran: bool = False
    #: Layer 3's verdict when it ran, `UNDECIDED` included, so a fail-open is visible to a
    #: reader of the result and not only to a reader of the log.
    classifier_verdict: Verdict | None = None
    #: Wall-clock for the whole screening. The ≤800ms p50 is computed from these.
    latency_ms: int = 0


def screen(
    question: str,
    *,
    model: BaseChatModel | None = None,
    settings: Settings | None = None,
) -> Screening:
    """Screen one analyst question. Never raises; a layer that fails, fails open.

    `model` is the classifier's, injectable for the same reason `translation_model` is: the
    suite drives layer 3 with a scripted reply rather than a paid call (spec seam 4).
    """
    started = time.perf_counter()
    normalised = normalise(question)

    if (rule := denylisted(normalised)) is not None:
        screening = Screening(
            blocked=True,
            layer=Layer.DENYLIST,
            rule=rule.id,
            latency_ms=_elapsed_ms(started),
        )
    else:
        verdict = classify(question, model=model, settings=settings)
        screening = Screening(
            blocked=verdict is Verdict.INJECTION,
            layer=Layer.CLASSIFIER if verdict is Verdict.INJECTION else None,
            classifier_ran=True,
            classifier_verdict=verdict,
            latency_ms=_elapsed_ms(started),
        )

    _log(screening, question=question, normalised=normalised.text)
    return screening


def _elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


def _log(screening: Screening, *, question: str, normalised: str) -> None:
    """Emit the gate-trigger record ADR-0006 asks for: layer, pattern, normalised input, time.

    The timestamp is the JSON envelope's `ts` — `logging_setup` puts one on every line, so a
    field here would be a second clock to disagree with the first.

    **The `normalised` field is the one exception to `log_event`'s no-user-content rule**, and
    it is bounded three ways: only on a block, only the normalised form, and only
    `config.GATE_LOGGED_INPUT_MAX_CHARS` of it. The reasoning for the exception and against it
    is in that constant's docstring, and CLAUDE.md records it as an exception rather than
    leaving a reader to find a question in a log line and conclude the rule was never true.

    An allowed turn logs counts and verdicts like every other event: the vast majority of
    screenings are of ordinary questions, and there is no security question their text answers.
    """
    fields = {
        "blocked": screening.blocked,
        "layer": None if screening.layer is None else screening.layer.value,
        "rule": screening.rule,
        "classifier_ran": screening.classifier_ran,
        "classifier_verdict": (
            None if screening.classifier_verdict is None else screening.classifier_verdict.value
        ),
        "question_chars": len(question),
        "normalised_chars": len(normalised),
        "latency_ms": screening.latency_ms,
    }
    if screening.blocked:
        fields["normalised"] = normalised[:GATE_LOGGED_INPUT_MAX_CHARS]
    log_event(logger, "input_gate", **fields)
