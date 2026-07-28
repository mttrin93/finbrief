"""Layer 2 of the input gate: a bounded-gap regex denylist (ADR-0006).

**What this layer contributes.** Known payload families, caught for free — no model call, no
token spend, sub-millisecond. It is the cheap half of "cheap-first": a catch exits the gate
immediately, and a pass *always* escalates, because a denylist cannot cover a phrasing nobody
has written down yet. That escalation is unconditional by design; treating a pass here as a
verdict is how a denylist becomes the whole gate (user story 34, `input_gate.py`).

**Two rule-authoring rules, and both are load-bearing.**

1. *Head-anchor the first word with `\\b`, and never rely on a boundary after it.* Every rule is
   scanned against `Normalised.squeezed` as well as `Normalised.text`, and the squeezed form has
   no word boundaries at all — it is one long run of letters, which is precisely what makes it
   catch `i g n o r e   a l l   …`. A rule anchored only at its head cannot start mid-word
   (`contract assets` will not trip an `act as` rule).

   **The head anchor is dropped for the squeezed pass, and it had to be.** `squeezed` is a
   single unbroken `[0-9a-z]` run, so `\\b` has exactly two positions in it — index 0 and the
   end. A head-anchored rule could therefore only ever match a squeezed payload that *began the
   message*, which one benign word defeats: `i g n o r e  a l l  p r e v i o u s  i n s t r u c
   t i o n s` was caught and `Hi. ` + the same payload was not (issue #8 review). This docstring
   claimed the opposite as a property, and the corpus case and both tests happened to put the
   payload at index 0, so nothing failed. `_squeezed_pattern` below compiles the anchorless
   variant; the "cannot start mid-word" argument is a property of the `text` pass, which still
   has its `\\b` and is the pass that has word boundaries to be wrong about.
2. *Join words with a gap, never with a literal space.* A literal space never matches in the
   squeezed form, so a rule written `you are now` is a rule that only works on half the input.
   `_tight` and `_gap` below are the two joiners, and which one to use is a real choice: see
   their docstrings.

**Not an advice denylist.** "Should I buy Tesla?" is user story 15's request, not user story
16's attack, and it must reach the model so the *output* validator can refuse it with a
disclaimer. Blocking advice requests here would be wrong about what the analyst did and would
replace a graceful refusal with an accusation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache

from finbrief.prompts import QUARANTINE_TAGS, tag_pattern
from finbrief.security.normalize import Normalised

#: The one tag in `delimiter-forgery`'s pattern that is **not** a declared quarantine block.
#:
#: `<system>` frames nothing in this repo — `prompts.py` puts the persona in a system
#: *message*, never a tag — so there is nothing to derive it from. It is denylisted anyway,
#: because a question carrying it forges a boundary a model reading *text* may well honour,
#: which is the same attack as closing `</sources>` early. Named here rather than inlined so
#: the pattern below reads as what it is: every declared block, plus this one, for this reason
#: (issue #8 review).
_FORGED_ROLE_TAG = "system"

#: How far apart two words of one payload may sit and still match — ADR-0006's "bounded" gap.
#:
#: **An assertion about the payload family, not a knob** (CLAUDE.md's exemption list), and
#: calibrated against the corpus in `corpus.py` rather than chosen for roundness: 20 characters
#: is what covers the filler an attacker puts between the verb and its object — "ignore *all of
#: the* previous instructions", "disregard *everything in* your system prompt" — and stops well
#: short of spanning a sentence. Unbounded (`.*`) would match `ignore` in one clause and
#: `instructions` three paragraphs later, which on a 4,000-character question
#: (`config.MAX_QUESTION_CHARS`) is a false positive waiting to happen. What it costs is stated
#: instead of hidden: a payload spread wider than this passes layer 2, and layer 3 is the layer
#: that exists for what layer 2 misses (`tests/test_denylist.py` pins both sides of the bound).
MAX_GAP_CHARS = 20


def _gap(*words: str) -> str:
    """Join `words` with the bounded gap — for where an attacker inserts filler.

    Use this between a payload's *verb and its object*, where the wording genuinely varies.
    """
    return rf".{{0,{MAX_GAP_CHARS}}}".join(words)


def _tight(*words: str) -> str:
    """Join `words` with room for a separator and no more — for a fixed phrase.

    Three characters: enough for a space (or nothing, in the squeezed form), not enough for a
    word. `_gap` would be wrong here and the failure is a false positive, not a miss: `you` and
    `are` and `now` joined by 20-character gaps match "what do you think Ford's margins are like
    now?", which is an ordinary question about a car company.
    """
    return r".{0,3}".join(words)


@dataclass(frozen=True, slots=True)
class Rule:
    """One denylisted payload family.

    `catches` is not documentation: the README's marginal-contribution table (user story 34) is
    rendered from it, and a gate-trigger log line records `id` so an analysis can say *which*
    rule fired rather than only that one did.
    """

    id: str
    #: What payload family this rule is for, in the words the README table prints.
    catches: str
    pattern: re.Pattern[str]
    #: Whether to match the **raw** input instead of the normalised forms. Two rules need it:
    #: normalisation replaces every non-alphanumeric with a space, so a rule about `<`, `>` or
    #: `:` has no characters left to match. Such a rule is therefore *not*
    #: obfuscation-resistant, which is fine — it is looking for a structure, and a structure
    #: that has been mangled into unrecognisability no longer does the thing it was for.
    raw: bool = False


RULES: tuple[Rule, ...] = (
    Rule(
        id="instruction-override",
        catches="Direct instruction override — telling the assistant to drop its rules.",
        # A scope word between the verb and the noun is required, and that requirement is what
        # keeps "can I ignore the accounting rules?" out: `the` is deliberately absent from the
        # scope group, so an ordinary determiner does not stand in for one.
        pattern=re.compile(
            r"\b"
            + _gap(
                r"(?:ignore|disregard|forget|override|discard)",
                r"(?:all|any|every(?:thing)?|previous|prior|above|earlier|preceding"
                r"|your|system|initial|original)",
                r"(?:instruction|rule|prompt|direction|guideline|constraint)",
            )
        ),
    ),
    Rule(
        id="prompt-extraction",
        catches="System-prompt extraction — asking it to print its own instructions.",
        # **Two alternations, because the scope word alone does not separate the attack from an
        # accounting question** (issue #8 review). One group of verbs, one group of nouns and a
        # scope word in between matched "What is the previous rule for lease accounting?",
        # "Can you repeat the previous message about margins?" and "What does Tesla say about
        # its system rules for autopilot?" — three ordinary analyst questions, blocked with no
        # recourse and, by design, no indication of why.
        #
        # The split: a **self-referential** scope (`your`, `system`, `developer`) admits the
        # widest noun group, because "your system prompt" is not something a filing has; a
        # **positional** scope (`previous`, `above`, …) admits only the nouns an analyst has no
        # other use for. `rule` is in neither — "the accounting rules", "its system rules" — and
        # "reveal your rules" is left to layer 3, which is the layer for what layer 2 misses.
        pattern=re.compile(
            r"\b(?:"
            + _gap(
                r"(?:repeat|print|show|reveal|output|display|echo|recite|reproduce|list"
                r"|what|tell|spell)",
                r"(?:your|system|developer)",
                r"(?:prompt|instruction|message|directive)",
            )
            + r"|"
            + _gap(
                r"(?:repeat|print|show|reveal|output|display|echo|recite|reproduce|list"
                r"|what|tell|spell)",
                r"(?:initial|original|first|above|preceding|previous)",
                r"(?:prompt|instruction|directive)",
            )
            + r")"
        ),
    ),
    Rule(
        id="persona-override",
        catches="Persona replacement — recasting the assistant as something without the rules.",
        pattern=re.compile(
            r"\b(?:"
            + _tight("you", "are", "(?:now|no", "longer)")
            + r"|"
            + _tight("from", "now", "on", "you")
            + r"|"
            + _tight("pretend", "(?:to|that|you)")
            + r"|"
            + _tight("role", "?play", "as")
            + r"|"
            + _tight("act", "as", "(?:if|though)")
            + r"|"
            + _tight("new", "(?:persona|identity|personality)")
            + r"|"
            # `stop being` needs an object that is *this assistant*. Bare, it matched "Does
            # Apple plan to stop being a hardware-first company?" and "When will Ford stop being
            # dependent on China for batteries?" — ordinary English about ordinary companies
            # (issue #8 review).
            + _tight("stop", "being")
            + r".{0,10}(?:assistant|finbrief|chatbot|bot|llm|model|helpful|cautious)"
            + r")"
        ),
    ),
    Rule(
        id="guardrail-suspension",
        catches="Asking for the controls themselves to be switched off.",
        # **The object has to belong to the assistant.** Joined to a bare object group, this
        # matched "Does management discuss plans to lift restrictions on the dividend?":
        # `restrict`, `polic(y|ies)`, `filter` and `limitation` are corporate-disclosure
        # vocabulary, and `lift`/`remove`/`suspend` are what a filing does to them (issue #8
        # review). `your` is the discriminator, and it is present in the corpus payload this
        # rule is calibrated on ("Turn off **your** safety filters"). `guardrail` keeps its
        # bare form — nothing in a 10-K is a guardrail.
        pattern=re.compile(
            r"\b"
            + _gap(
                r"(?:disable|deactivate|bypass|circumvent|remove|lift|unlock|suspend"
                + r"|"
                + _tight("(?:turn|switch)", "off")
                + r")",
                r"(?:guardrail|"
                + _gap(
                    r"your",
                    r"(?:guardrail|safe(?:ty|guard)|filter|restrict|limitation|censor"
                    r"|polic(?:y|ies)|disclaimer)",
                )
                + r")",
            )
        ),
    ),
    Rule(
        id="unrestricted-mode",
        catches="Named jailbreak modes — the folklore payloads that circulate verbatim.",
        pattern=re.compile(
            r"\b(?:"
            + _tight(
                r"(?:developer|debug|god|dan|admin|unrestricted|unfiltered|uncensored)", "mode"
            )
            + r"|jailbr(?:eak|oken|eaking)|"
            + _tight("do", "anything", "now")
            + r")"
        ),
    ),
    Rule(
        id="delimiter-forgery",
        catches="A quarantine tag in the question — forging the boundary around sources.",
        # Raw, and the only rule here that is about *structure* rather than wording. An
        # analyst's question has no legitimate reason to contain `<sources>` or `</news>`: those
        # tags are `prompts.py`'s frame around text the model is told to treat as evidence, and
        # a question carrying one is trying to close that frame early (issue #5's deferred
        # finding). The prompt-side half of the same finding is `prompts.quarantined`, which
        # neutralises the tag wherever it appears in *content*; this catches the analyst-typed
        # case and, unlike the escaping, says so out loud.
        #
        # **Derived from `QUARANTINE_TAGS`, so the escaping and the denylisting cannot drift.**
        # This was a hardcoded `(?:sources|news|system)` while three places — `prompts.py`,
        # CLAUDE.md and ADR-0006 §4 — claimed it derived from the tuple. It did not, and the
        # sets differed both ways: `input` was quarantined but not denylisted, and `system`
        # is not a quarantine tag at all. So "a fourth block is escaped *and* denylisted the
        # moment it is declared" was false either way (issue #8 review). Now it is true, and
        # `tests/test_denylist.py` parametrises over the tuple so a fifth tag arrives covered.
        # Through `prompts.tag_pattern` so the *tolerance* — spacing, casing, opening or
        # closing — is derived once as well as the tag set. Built separately here, the two could
        # still drift on the axis that is not the tuple.
        pattern=tag_pattern(*QUARANTINE_TAGS, _FORGED_ROLE_TAG),
        raw=True,
    ),
    Rule(
        id="role-spoof",
        catches="A forged role line — text posing as a system or developer turn.",
        # Also raw, and for the same reason: the `:` is the whole signal and normalisation eats
        # it. Anchored to the start of a line so that "the assistant: a note on terminology"
        # mid-sentence does not trip it; a spoofed turn is written where a turn would start.
        pattern=re.compile(
            r"(?:\A|\n)\s*[\[(<]?\s*(?:system|assistant|developer)\s*[\])>]?\s*:",
            re.IGNORECASE,
        ),
        raw=True,
    ),
)


@cache
def _squeezed_pattern(pattern: re.Pattern[str]) -> re.Pattern[str]:
    """`pattern` with its leading `\\b` removed — the form to scan `Normalised.squeezed` with.

    See rule 1 in the module docstring for why the anchor cannot survive the squeeze. Cached
    because `RULES` is a fixed tuple and this is otherwise a recompile per question; `lru_cache`
    is safe here in a way it is not for the `build_once` singletons, because a duplicate
    construction of a `re.Pattern` is wasted work and nothing else.
    """
    return re.compile(pattern.pattern.removeprefix(r"\b"), pattern.flags)


def denylisted(normalised: Normalised) -> Rule | None:
    """The first rule `normalised` trips, or `None`.

    First rather than all, because the gate exits on a catch and the log line names one rule:
    the marginal-contribution question is which *layer* stopped a payload, and enumerating every
    rule that also would have would cost a scan per rule for a number nobody reads.

    Rule order is therefore the reporting order, and `RULES` is ordered most-specific-first for
    that reason rather than alphabetically.
    """
    for rule in RULES:
        if rule.raw:
            if rule.pattern.search(normalised.raw):
                return rule
        elif rule.pattern.search(normalised.text) or _squeezed_pattern(rule.pattern).search(
            normalised.squeezed
        ):
            return rule
    return None
