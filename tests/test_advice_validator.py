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
