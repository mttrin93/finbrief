"""Layer 4, the back door: the no-investment-advice validator (spec seam 4, ADR-0006).

Driven by `security.corpus`'s two answer sets, and the *second* one is the point. A validator
that refuses everything satisfies "should I buy X is refused"; what makes this one defensible is
that it lets through the answers FinBrief exists to write — including the ones that use the
vocabulary of advice while giving none, which is where a keyword check refuses the assistant's
own correct output.

The line drawn here is stated in `advice.py`: a **noun** can be mentioned in a refusal ("I can't
give a price target"), so those rules yield to a negation in the same sentence. An **act**
cannot be performed in a refusal — "I do not recommend Tesla" is a recommendation — so those
rules never yield. Both halves are asserted below.
"""

from __future__ import annotations

import pytest

from finbrief.security import corpus
from finbrief.security.advice import ADVICE_RULES, advice_hits, validate_answer


@pytest.mark.parametrize("answer", corpus.ADVICE_ANSWERS)
def test_an_advice_shaped_answer_is_refused(answer: str) -> None:
    verdict = validate_answer(answer)

    assert verdict.refused, answer
    assert verdict.rules, "a refusal has to name what it fired on, for the log line"


@pytest.mark.parametrize("answer", corpus.RESEARCH_ANSWERS)
def test_the_research_this_assistant_writes_is_allowed(answer: str) -> None:
    """The false-positive floor. A refusal here is FinBrief refusing its own correct answer."""
    verdict = validate_answer(answer)

    assert verdict.refused is False, f"{answer}\nfired: {verdict.rules}"


def test_a_negatable_rule_yields_to_a_denial_in_the_same_sentence() -> None:
    """ "I can't give you a price target" mentions one; it does not give one."""
    assert advice_hits("My price target for NVDA is $260.")
    assert advice_hits("I can't give a price target, and the filing carries none either.") == ()


@pytest.mark.parametrize(
    "answer",
    [
        "I can't give a price target. My price target for NVDA is $260.",
        "I can't give a price target, but fair value is $260 on my model.",
        "I won't advise on position sizing — put 5% of your portfolio into Apple.",
        "I do not give price targets. However, the target price is $300.",
    ],
)
def test_disclaiming_first_does_not_buy_a_pass_for_advising_second(answer: str) -> None:
    """The shape a hedged model writes, and the shape an obeyed indirect injection writes.

    `advice_hits` examined only a negatable rule's **first** match, so a denial in the opening
    clause suppressed the rule for the whole answer and every later, un-denied occurrence went
    unexamined (issue #8 review). All four of these returned `()` and `refused=False`.

    Four joins rather than one, because the fix is about where a *clause* ends: a full stop, a
    contrastive conjunction, a dash, and a conjunction opening the second sentence.
    """
    assert advice_hits(answer) != ()
    assert validate_answer(answer).refused


def test_the_clause_clamp_is_what_makes_the_previous_test_pass_and_not_the_window() -> None:
    """The two bounds on the negation lookback, asserted apart.

    `_denied` clamps to `NEGATION_WINDOW_CHARS` *and* to the current clause. Only the second one
    separates "I can't do that. My price target is $260." from "I can't give a recommendation or
    a price target." — the denial is within 60 characters in both. Replacing `_denied` with a
    plain character window left every test green (issue #8 review), so the two cases are pinned
    against each other here.
    """
    assert advice_hits("I can't do that. My price target is $260.") == ("price-target",)
    assert advice_hits("I can't give a recommendation or a price target.") == ()


def test_a_denial_further_back_than_the_window_does_not_reach_the_match() -> None:
    """The other bound: `NEGATION_WINDOW_CHARS`, in the same clause, exceeded.

    Pinned from above as well as below — 60, 200 and 5000 were all indistinguishable to the
    suite, so the constant could be widened into a blanket negation guard without a red test.
    """
    from finbrief.security.advice import NEGATION_WINDOW_CHARS

    filler = "and the filing carries a great deal of detail on the matter " * 3
    assert len(filler) > NEGATION_WINDOW_CHARS
    assert advice_hits(f"I cannot {filler} give a price target") == ("price-target",)


def test_an_act_rule_does_not_yield_to_a_denial() -> None:
    """A negated recommendation is still a recommendation — the direction is the only change.

    This is the hole a blanket negation guard would open, and the reason the guard is per-rule:
    "I would not buy Ford here" is a sell view stated politely, and a validator that read the
    "not" as a disclaimer would wave it through.
    """
    assert advice_hits("I would not buy Ford at this price.")
    assert advice_hits("You should not hold Tesla through the print.")


def test_the_word_buy_inside_another_word_is_not_advice() -> None:
    """`buybacks` is ordinary capital-allocation vocabulary and appears in real filings."""
    assert advice_hits("Management prioritises buybacks over dividends [1].") == ()
    assert advice_hits("The company's buyback programme was expanded [2].") == ()


def test_reporting_that_analysts_have_ratings_is_not_giving_one() -> None:
    """Naming the *existence* of sell-side ratings is research; issuing one is not."""
    assert advice_hits("Analysts' buy and sell ratings are not something FinBrief holds.") == ()


def test_a_refusal_names_every_rule_it_fired_on() -> None:
    """Plural because one answer can be advice several ways, and the log wants all of them."""
    hits = advice_hits("Rating: BUY. My price target is $300, so you should buy it now.")

    assert len(hits) > 1


def test_every_rule_states_what_it_catches() -> None:
    """`catches` feeds the README's marginal-contribution table, like the denylist's does."""
    assert all(rule.catches for rule in ADVICE_RULES)


def test_rule_ids_are_unique() -> None:
    assert len({rule.id for rule in ADVICE_RULES}) == len(ADVICE_RULES)


def test_an_empty_answer_is_not_advice() -> None:
    """A generation that produced nothing is a different failure, and not this layer's."""
    assert validate_answer("").refused is False


def test_the_guard_is_built_once_per_process() -> None:
    """Guardrails' `Guard`, its validator registry and its telemetry are process state.

    Built through `caching.build_once` for the reason CLAUDE.md records about the other
    process-level singletons: two Streamlit session threads validating their first answer at the
    same time both miss a cold `lru_cache` key and both construct, and Guardrails' registry is
    not a thing to construct twice.
    """
    from finbrief.security.advice import advice_guard

    assert advice_guard() is advice_guard()


def gate_lines(captured, event: str) -> list[dict]:
    import json

    return [
        json.loads(line)
        for line in captured.getvalue().splitlines()
        if json.loads(line).get("event") == event
    ]


@pytest.fixture
def capsys_stream():
    """The JSON-lines handler pointed at a buffer, so a test can read the events back."""
    import io

    from finbrief.observability.logging_setup import configure_logging

    stream = io.StringIO()
    configure_logging(stream=stream)
    return stream


def test_a_refusal_logs_lengths_and_verdicts_and_never_the_answer(capsys_stream) -> None:
    """The privacy claim on `output_validator`, asserted rather than commented.

    An answer is derived from user content and these lines are kept, so the record carries
    `answer_chars` and the rule ids — this module's own vocabulary — and nothing else. Adding
    `answer=answer` left the whole suite green (issue #8 review), which is why the field *set*
    is pinned rather than a sample of it.
    """
    answer = "You should buy Tesla — the multiple is fair [1]."
    validate_answer(answer)

    (line,) = gate_lines(capsys_stream, "output_validator")
    assert set(line["fields"]) == {"refused", "rules", "answer_chars", "error"}
    assert line["fields"]["answer_chars"] == len(answer)
    assert answer not in capsys_stream.getvalue()


def test_a_library_failure_that_is_not_a_validation_error_fails_open(capsys_stream) -> None:
    """ "Never raises" means every exception, which it did not.

    The caller is the `else:` clause of `app/Home.py`'s turn, and Python does not route an
    `else:` exception to that `try`'s handler — so anything but a `ValidationError` out of
    `Guard.validate` reached the analyst as a traceback (issue #8 review). Fails open, like the
    classifier, and logs the exception *type* only for the same reason: an error string can
    carry a request URL and a request URL can carry an API key.
    """
    import finbrief.security.advice as advice

    class Broken:
        def validate(self, answer):
            raise RuntimeError("https://api.example/v1?key=sk-secret exploded")

    original = advice.advice_guard
    advice.advice_guard = lambda: Broken()
    try:
        verdict = validate_answer("Tesla identifies supply chain concentration [1].")
    finally:
        advice.advice_guard = original

    assert verdict.refused is False
    (line,) = gate_lines(capsys_stream, "output_validator_unavailable")
    assert line["fields"]["error"] == "RuntimeError"
    assert "sk-secret" not in capsys_stream.getvalue()


def test_validating_does_not_move_the_processes_event_loop_policy() -> None:
    """`GUARDRAILS_RUN_SYNC`, measured at the seam its comment describes.

    `guardrails.validator_service.get_loop` calls `asyncio.set_event_loop_policy(uvloop…)` on
    every `Guard.validate` unless the variable is set — a library replacing the whole process's
    policy as a side effect of validating one answer. `tests/test_hermetic_suite.py` covers the
    *egress* consequence; the side effect itself was described and never asserted, so it could
    return silently.
    """
    import asyncio

    before = type(asyncio.get_event_loop_policy())
    validate_answer("Tesla identifies supply chain concentration [1].")

    assert type(asyncio.get_event_loop_policy()) is before
