"""Layer 4, the back door: no personalised investment advice, via Guardrails AI (ADR-0006).

**What this layer contributes that the input layers cannot: consequences.** The three front-door
layers judge what the analyst *asked*; this one judges what the model *produced*. Those come
apart in both directions. A perfectly ordinary question — "how does its valuation compare to its
fundamentals?" — can be answered with a recommendation nobody asked for, and a *successful*
indirect injection, arriving through retrieved text that never went past the front door at all
(user story 17), shows up here or nowhere. It is the only layer whose input the attacker does
not choose.

**Why it exists at all when the persona already refuses.** This repo has four recorded cases of
the model declining a stated prompt instruction — the verbatim-query contract (T4), sub-query
refusals (T6), the square-bracket rule and the section headings read as search terms (T5). A
boundary written in a prompt is a request; a boundary written in code is a boundary. That is the
whole argument for a structural check here rather than a firmer sentence in
`AGENT_SYSTEM_PROMPT`.

**What Guardrails AI supplies, precisely.** The `Guard`/`Validator` framework, the `on_fail`
semantics, and one standard error shape (`ValidationError`) for a failed check. It does **not**
supply the rule: there is no hub validator for "no investment advice", and a financial-tone or
topic-restriction validator would be measuring something adjacent. So the rule is a registered
custom validator — which is the documented way to express a domain constraint in Guardrails —
and the rule itself is the deterministic pattern set below. ADR-0006 dropped Guardrails'
`DetectJailbreak` deliberately: it would have been a second, redundant *input* detector, and
redundancy on one axis is what that ADR exists to remove.

**The rule is deterministic, and that is a choice with a cost.** An LLM-judge validator would
catch advice phrased in a way nothing here matches, at the price of a second model call on every
answer and a check whose own verdict is nondeterministic. The pattern set is cheap, testable in
both directions against a fixed corpus, and free of a second failure mode in front of every
reply — and what it misses is stated rather than implied: novel phrasings of advice are this
layer's blind spot exactly as they are layer 2's, and there is no layer 5. ADR-0002's
RAGAs faithfulness run (T10, #11), over T9's golden set, is what measures the residue
statistically.

**`on_fail=EXCEPTION`, turned into a result.** ADR-0006 asks for `on_fail="exception"` →
graceful refusal, and the two halves live in different places: raising is Guardrails' contract,
and converting it into an `AdviceVerdict` the caller renders is this module's — the same
refusals-as-results shape `tools/finance.py` uses, and for the same reason. A `ValidationError`
reaching `app/Home.py` would be a traceback where user story 15 asks for a disclaimer.

**Telemetry is switched off, and it had to be.** Guardrails posts anonymous validation metrics
to its own endpoint by default (`enable_metrics=True` unless `~/.guardrailsrc` says otherwise),
which means a library added for a security control would, unconfigured, send a record of every
validated answer to a third party. Measured: with the suite's egress guard installed,
`Guard.validate` resolved `hty0gc1ok3.execute-api.us-east-1.amazonaws.com`. `_build_guard`
disables it in the one order that holds — see there — and `tests/test_hermetic_suite.py` carries
the per-backend test that keeps it disabled rather than a comment claiming it is.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

from guardrails import Guard, OnFailAction
from guardrails.classes.validation.validation_result import (
    FailResult,
    PassResult,
    ValidationResult,
)
from guardrails.errors import ValidationError
from guardrails.validator_base import Validator, register_validator

from finbrief.caching import build_once
from finbrief.observability.logging_setup import log_event

logger = logging.getLogger(__name__)

# **Validate synchronously, and this is not a performance preference.**
# `guardrails.validator_service.get_loop` runs on *every* `Guard.validate` and calls
# `asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())` — replacing the whole process's
# event loop policy as a side effect of validating one answer. In a Streamlit app that already
# has loops of its own (and `chromadb`/`langchain` async paths beside them), a library swapping
# the policy mid-process is a change nobody asked for; it also moves DNS resolution into libuv,
# outside the reach of the suite's egress guard (`tests/conftest.py`'s
# `_block_uvloop_resolution`, which exists anyway — a hole is a hole whether today's code walks
# through it).
#
# `GUARDRAILS_RUN_SYNC` is guardrails' only switch for it, and it costs nothing: with one
# validator, `SequentialValidatorService` is what runs either way — the async path already falls
# back to it here, warning "Could not obtain an event loop", because Streamlit's script threads
# have none. So this makes the behaviour explicit and removes the side effect and the warning
# together. Measured both ways: without it the policy goes `asyncio.unix_events` -> `uvloop` on
# the first `Guard.validate`; with it, it does not move.
#
# **At import rather than inside `_build_guard`** (issue #8 review): a process-wide environment
# mutation hidden in a `build_once` constructor fires at an unpredictable moment — whenever some
# caller happens to validate first — and anything that built its own `Guard` before that would
# already have swapped the policy. Module import is the earliest deterministic point, and
# guardrails reads the variable per-validate, so setting it here is in time for every path.
os.environ["GUARDRAILS_RUN_SYNC"] = "true"

#: The validator's name in Guardrails' registry. Namespaced, because the registry is global.
VALIDATOR_NAME = "finbrief/no-investment-advice"

#: How far back a negatable rule looks for a denial, in characters.
#:
#: **An assertion about a sentence, not a knob** (CLAUDE.md's exemption list). Sixty characters
#: is about a clause: it reaches "I can't give a recommendation or a **price target**" and
#: stops short of the previous sentence, which is the distance that matters — a denial two
#: sentences back does not govern this one. The search is additionally bounded to the current
#: clause (`_CLAUSE_START`), so this is a ceiling rather than the mechanism —
#: `tests/test_advice_validator.py` pins the two apart.
NEGATION_WINDOW_CHARS = 60

#: The denials that make a *mention* of advice not an instance of it.
_NEGATION = re.compile(
    r"\b(?:not|never|no|cannot|can't|won't|wouldn't|don't|doesn't|isn't|aren't|unable|"
    r"decline|refuse|avoid)\b"
)

#: Where the clause a match sits in begins, for bounding the negation lookback to it.
#:
#: Sentence punctuation, an em/en dash, and the contrastive conjunctions — because those are
#: the three ways the hedge "I can't do X **but** here is X" joins its two halves, and a denial
#: on the far side of any of them governs nothing on this side. A plain comma is deliberately
#: absent: "I can't give a recommendation, a price target or a rating" is one denial over a
#: list.
_CLAUSE_START = re.compile(
    r"[.!?;–—]|\b(?:but|however|although|though|that said|still|nonetheless"
    r"|nevertheless|regardless)\b"
)


@dataclass(frozen=True, slots=True)
class AdviceRule:
    """One way an answer can be personalised investment advice.

    `negatable` is the line this module draws and the reason it is per-rule rather than global.
    A rule that matches a **noun** — "price target", "your portfolio" — can fire on a sentence
    that *denies* giving one, so it yields to a denial in the same clause. A rule that matches
    an **act** — "you should buy", "I recommend", an imperative — cannot: "I would not buy Ford"
    is a sell view, and a validator that read the "not" as a disclaimer would pass the advice
    and refuse the refusal. A blanket negation guard is the hole this avoids.
    """

    id: str
    #: What this rule is for, in the words the README's table prints.
    catches: str
    pattern: re.Pattern[str]
    negatable: bool = False


ADVICE_RULES: tuple[AdviceRule, ...] = (
    AdviceRule(
        id="directive-trade",
        catches="Telling the reader to trade — 'you should buy', 'I would sell'.",
        # `\b…\b` around the verb, so `buybacks` and `holdings` — ordinary capital-allocation
        # vocabulary that appears in real filings — are not trades.
        #
        # **The modal is required, and it was optional** (issue #8 review). Optional, every
        # element between pronoun and verb could be skipped, so a bare `we buy` matched —
        # and a bare `we buy` is what a 10-K says in the filer's own first person. Measured:
        # `Tesla states in Item 1A that "we buy raw materials from a limited number of
        # suppliers"` was refused, and a refusal here destroys the answer *and* its panels
        # (`app/Home.py`). Quoting a filer is the single most characteristic thing this
        # application writes, so the rule may not fire on it.
        #
        # What that costs, stated rather than hidden: an unhedged first-person recommendation
        # with no modal at all — "We buy Ford here." — now reaches the reader unless it also
        # trips `imperative-trade` or `rating`. That is the same blind spot this layer already
        # declares for novel phrasing, and it is the right side of the trade: a miss here leaves
        # a disclaimer under the answer, where a false positive replaces a grounded brief with
        # an accusation.
        pattern=re.compile(
            r"\b(?:you|i|we)\b"
            r"(?:\s*'(?:d|ll)|\s+(?:should|would|could|ought to|will|might want to|need to|"
            r"do not|don't|wouldn't|shouldn't))"
            r"\s*(?:not\s+)?\s*"
            r"\b(?:buy|sell|hold|short|accumulate|divest|trim|exit|avoid|add to)\b",
            re.IGNORECASE,
        ),
    ),
    AdviceRule(
        id="recommendation",
        catches="Issuing a recommendation in the first person.",
        # `recommend\b` does not match `recommendation` — `\b` needs a non-word character after
        # it, and "recommendation" continues with `a`. That is what lets "I can't give a
        # recommendation" through without needing the negation guard, which this rule must not
        # have: "I do not recommend Tesla" is a recommendation.
        pattern=re.compile(
            r"\b(?:i|we)\b[^.;!?]{0,15}\brecommend(?:s|ed|ing)?\b", re.IGNORECASE
        ),
    ),
    AdviceRule(
        id="imperative-trade",
        catches="An imperative trade instruction opening a sentence — 'Hold Pfizer for now.'",
        # `short` carries a negative lookahead because "Short interest stands at 3.1% of float"
        # is ordinary equity-research vocabulary that happens to open a sentence with the verb
        # (issue #8 review). The other verbs need no such guard — there is no noun phrase
        # "buy interest" a brief would write sentence-initially.
        pattern=re.compile(
            r"(?:\A|[.!?]\s+|\n\s*[-*]?\s*)"
            r"(?:buy|sell|hold|short(?!\s+(?:interest|sellers?|positions?|float|squeeze))"
            r"|accumulate|avoid|divest|add to|take profits)\s",
            re.IGNORECASE,
        ),
    ),
    AdviceRule(
        id="rating",
        catches="A sell-side rating on the company — 'Rating: BUY', 'a strong buy'.",
        pattern=re.compile(
            r"\brating\s*[:=]\s*(?:buy|sell|hold|overweight|underweight|neutral)\b"
            r"|\b(?:strong|clear|screaming|outright)\s+(?:buy|sell)\b"
            r"|\b(?:over|under)weight\s+(?:it|the\s+(?:stock|shares|name))\b",
            re.IGNORECASE,
        ),
    ),
    AdviceRule(
        id="price-target",
        catches="A price target or a forecast of where the price goes.",
        pattern=re.compile(
            r"\bprice target\b|\bfair value (?:of|is|sits at)\b|\btarget price\b",
            re.IGNORECASE,
        ),
        negatable=True,
    ),
    AdviceRule(
        id="personalised-allocation",
        catches="Advice about the reader's own money — position sizing, portfolio weight.",
        pattern=re.compile(
            r"\b(?:of|in|into|from)\s+your\s+(?:portfolio|holdings|position|capital|money)\b"
            r"|\bgiven your\s+(?:time horizon|risk tolerance|circumstances|situation)\b"
            r"|\bposition siz(?:e|ing)\b",
            re.IGNORECASE,
        ),
        negatable=True,
    ),
)


def advice_hits(answer: str) -> tuple[str, ...]:
    """Every `AdviceRule` id `answer` trips, in rule order. Pure.

    **Every** rule rather than the first, unlike `denylist.denylisted`: the denylist's caller
    exits on a catch and logs one rule, where a refused answer is refused once and the
    interesting datum is *how many ways* it was advice — which is what tells a reader of the log
    whether the model slipped once or wrote a research note as a trade ticket.

    **Every *occurrence* too, for a negatable rule, and that is the fix for the shape this layer
    exists to catch.** This iterated `search` — the *first* match only — so a rule whose first
    occurrence sat inside a denial was dropped for the whole answer, and every later un-denied
    one was never looked at. "I can't give a price target. My price target is $260." therefore
    passed, which is both the most natural hedged wording a model produces and exactly what an
    obeyed indirect injection looks like (issue #8 review). A negatable rule now fires unless
    **every** occurrence is denied.
    """
    return tuple(
        rule.id
        for rule in ADVICE_RULES
        if any(
            not (rule.negatable and _denied(answer, match.start()))
            for match in rule.pattern.finditer(answer)
        )
    )


def _denied(answer: str, at: int) -> bool:
    """Whether the clause ending at `at` denies what the match after it says.

    Bounded twice — to `NEGATION_WINDOW_CHARS` and to the current clause — because a denial in
    the *previous* clause governs nothing here: "I can't do that. My price target is $260."
    contains a denial and a price target, and only one of them is about the target. The same
    holds across a contrastive conjunction, which is why `_CLAUSE_START` is more than sentence
    punctuation: "I can't give a price target, but fair value is $260" is the hedge, not a
    denial.
    """
    window = answer[max(0, at - NEGATION_WINDOW_CHARS) : at]
    # Everything after the last clause break in the window is the clause the match sits in.
    breaks = list(_CLAUSE_START.finditer(window))
    clause = window[breaks[-1].end() :] if breaks else window
    return _NEGATION.search(clause) is not None


@register_validator(name=VALIDATOR_NAME, data_type="string")
class NoInvestmentAdvice(Validator):
    """The Guardrails validator around `advice_hits`.

    Thin on purpose: the rule is a pure function so it can be tested and reported on without a
    `Guard` in the way, and this class is the adapter that gives it Guardrails' `on_fail`
    semantics. The `error_message` names the rule ids so the raised `ValidationError` reads as
    something rather than as "validation failed" — `validate_answer` does **not** parse it back
    out, it re-runs `advice_hits`, because a log line derived from a library's error string is a
    log line that changes when the library does.
    """

    def validate(self, value: str, metadata: dict | None = None) -> ValidationResult:  # noqa: ARG002 — the Validator signature
        hits = advice_hits(value)
        if hits:
            return FailResult(error_message=f"investment advice: {', '.join(hits)}")
        return PassResult()


@dataclass(frozen=True, slots=True)
class AdviceVerdict:
    """What layer 4 decided about one answer.

    `rules` is empty on a pass and names every rule on a refusal, so the log line says *why* and
    a reader of the result can tell a one-phrase slip from a note written as a trade ticket.
    """

    refused: bool
    rules: tuple[str, ...] = ()


def _build_guard() -> Guard:
    """The one `Guard` this process has, with telemetry off.

    **The order is the whole function**, and it is not the order the obvious reading suggests:

    1. `Guard()` — which does *not* read `~/.guardrailsrc`.
    2. `configure(allow_metrics_collection=False)` — which does, and then sets the process-wide
       `HubTelemetry` singleton's `_enabled` to `False`. It must come before step 3, because it
       *overwrites* `settings.rc` from the file on its way.
    3. Replace `settings.rc` outright. `Validator.__init__` reads `enable_metrics` to
       decide whether to build a tracer of its own, and `use_remote_inferencing` to decide
       whether a validator may call a remote inference endpoint. Both default to `True`, from a
       file this repo does not ship and cannot rely on.
    4. Only now instantiate the validator, so it reads the settings from step 3.

    **Which of the two switches does the work, measured** rather than assumed, because "we set
    two flags" is not a description of a mechanism. With the egress guard installed and step 2
    removed, `Guard.validate` reaches the network; with step 3 removed instead, it does not.
    Step 2 is the load-bearing one — it disables the shared `HubTelemetry` singleton that step
    4's validator then finds already off. Step 3 is kept anyway and for stated reasons rather
    than superstition: it is what would matter if a *hub* validator were ever added beside this
    one (`use_remote_inferencing` is what sends a validator's input to a remote inference
    endpoint), and it removes this module's dependence on the order in which something else in
    the process first touches that singleton. Reordering 2 and 3 silently reinstates the call,
    which is why `tests/test_hermetic_suite.py` exercises the shipped `validate_answer` — a
    comment asserting telemetry is off cannot fail.
    """
    from guardrails.classes.rc import RC
    from guardrails.settings import settings

    guard = Guard()
    guard.configure(allow_metrics_collection=False)
    settings.rc = RC(enable_metrics=False, use_remote_inferencing=False)
    return guard.use(NoInvestmentAdvice(on_fail=OnFailAction.EXCEPTION))


#: The process-level handle, for the reason the other **five** singletons use `build_once`:
#: Guardrails' `Guard`, its validator registry and its telemetry singleton are process state,
#: and `lru_cache` is atomic about its bookkeeping and not about the constructor it wraps.
advice_guard = build_once(_build_guard)


def validate_answer(answer: str) -> AdviceVerdict:
    """Run layer 4 over one answer. Never raises.

    Goes through the `Guard` rather than calling `advice_hits` directly, and the difference is
    not ceremony: what ADR-0006 chose was Guardrails' `on_fail` contract, so the shipped path
    has to exercise it — a `validate_answer` that skipped the Guard would leave the library in
    the dependency list and the decision untested.

    **"Never raises" means every exception, not only `ValidationError`.** The caller is the
    `else:` clause of `app/Home.py`'s turn, and Python does not route an `else:` exception to
    that `try`'s handler — so anything *other* than a failed validation coming out of
    `Guard.validate` (a guardrails internal, or a changed error type: the pin is `>=0.10.2`,
    unbounded) reached the analyst as a raw traceback where user story 15 asks for a disclaimer
    (issue #8 review). Those fail **open**, like the classifier's provider failures and for the
    same reason, and are logged as a warning rather than swallowed.
    """
    try:
        advice_guard().validate(answer)
    except ValidationError as exc:
        rules = advice_hits(answer)
        log_event(
            logger,
            "output_validator",
            refused=True,
            rules=list(rules),
            # Lengths and verdicts, never the answer: an answer is derived from user content and
            # these lines are kept. The rule ids are this module's own vocabulary.
            answer_chars=len(answer),
            # Carried because a `ValidationError` with no rules behind it would mean the Guard
            # and `advice_hits` disagree, which is a defect worth seeing in the log.
            error=type(exc).__name__,
        )
        return AdviceVerdict(refused=True, rules=rules)
    except Exception as exc:  # noqa: BLE001 — every other library failure is one fail-open
        # The exception *type*, never its message, for `classifier.classify`'s reason: a client
        # error string can carry a request URL and a request URL can carry an API key.
        log_event(
            logger,
            "output_validator_unavailable",
            level=logging.WARNING,
            error=type(exc).__name__,
            answer_chars=len(answer),
        )
        return AdviceVerdict(refused=False)
    return AdviceVerdict(refused=False)
