"""Layer 1 of the input gate: fold an obfuscated payload into one surface form (ADR-0006).

**What this layer contributes, and why it is not the denylist's job.** A regex denylist matches
strings. `ignore all previous instructions` has an unbounded number of spellings — case,
accents, leetspeak, zero-width joiners, fullwidth Latin, Cyrillic lookalikes — and encoding each
one as a rule is a losing race the attacker sets the pace of. Normalisation collapses the
spellings so **one** rule covers all of them. That is the marginal contribution the README's
table claims for this layer (user story 34), and `tests/test_normalization.py` asserts it at
*this* seam rather than through a verdict, so the claim cannot be satisfied by the rule instead.

**The output is for matching and for nothing else.** It is deliberately lossy — de-leetspeak
turns `Item 1A` into `item ia` — so it must never be embedded, retrieved on, or shown to a
reader. `input_gate.py` is the only caller, and the one place any of it is *recorded* is a
blocked turn's log line (see `config.GATE_LOGGED_INPUT_MAX_CHARS`).

**Two forms, because one cannot do both jobs.** `text` keeps word separators, so a rule can use
`\\b` to avoid matching inside an unrelated word (`contract assets` must not trip an `act as`
rule). `squeezed` removes them, which is the only thing that catches letter-by-letter spacing
(`i g n o r e   a l l   …`), where every letter is its own token and no word-level rule can see
the words at all. `denylist.py` scans both with one rule set; see its head-anchoring rule for
what that costs a rule author.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

#: Confusables that Unicode compatibility normalisation does **not** fold, because they are
#: distinct letters rather than compatibility variants: `а` (Cyrillic) and `a` (Latin) are
#: different characters that happen to render identically, so NFKC leaves both alone.
#:
#: Latin-skeleton only, and lowercase only — `normalise` casefolds before it gets here. The set
#: is the Cyrillic and Greek letters whose lowercase glyph is a Latin lookalike in a normal
#: font, which is what an attacker pasting from a homoglyph generator gets. It is deliberately
#: **not** the full Unicode confusables table (~6,000 entries): that table is maintained for
#: security *display* decisions, most of its mappings are for scripts no analyst will type, and
#: shipping a copy of it here would be a dependency on data nothing in this repo can re-verify.
#: What it means is stated rather than hidden: a lookalike outside this map survives
#: normalisation and reaches layer 3, which is the layer that exists for what layers 1 and 2
#: miss.
_HOMOGLYPHS = str.maketrans(
    {
        # Cyrillic
        "а": "a", "в": "b", "с": "c", "е": "e", "ѕ": "s", "һ": "h", "і": "i", "ј": "j",
        "к": "k", "м": "m", "н": "h", "о": "o", "р": "p", "т": "t", "у": "y", "х": "x",
        # Greek
        "α": "a", "β": "b", "ε": "e", "ι": "i", "κ": "k", "μ": "u", "ν": "v", "ο": "o",
        "ρ": "p", "σ": "o", "τ": "t", "υ": "u", "χ": "x",
    }
)  # fmt: skip

#: Digit- and symbol-for-letter substitutions, the cheapest obfuscation there is.
#:
#: **Lossy, and the loss is accepted here rather than hedged.** `1` → `i` turns `Item 1A` into
#: `item ia` and `$5bn` into `ssbn`, which would be unusable for anything except matching — and
#: matching is all this text is for. The alternative (substituting only where the result is a
#: dictionary word) needs a dictionary and would still be guessing at intent.
_LEETSPEAK = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t",
                            "@": "a", "$": "s", "!": "i"})  # fmt: skip

#: Anything that is not a letter or a digit. Replaced with a space rather than removed, so
#: `ignore-all` becomes two words in `text` and one run in `squeezed`.
_NOT_ALPHANUMERIC = re.compile(r"[^0-9a-z]+")

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class Normalised:
    """One input in every form a denylist rule is scanned against.

    Two normalised forms, not one, for the reason the module docstring gives: `text` keeps the
    word boundaries a rule needs to *not* match, and `squeezed` is the only form in which
    letter-by-letter spacing still contains its words. The invariant between them —
    `squeezed == text.replace(" ", "")` — is asserted rather than assumed, because it is what
    lets one rule set scan both.

    **`raw` is carried too**, which reads oddly on a class called `Normalised` and is the honest
    shape anyway: two of the denylist's rules are *structural* and match characters this module
    deliberately destroys — `<` and `>` for a forged `</sources>` delimiter, `:` for a spoofed
    `system:` role line. Those rules cannot be written against a normalised form at all, so the
    object a rule is handed has to carry the original. See `denylist.Rule.raw`.
    """

    #: What the caller passed in, untouched.
    raw: str
    #: Lowercase, de-accented, de-homoglyphed, de-leetspeaked words, single-space separated.
    text: str
    #: `text` with the separators gone.
    squeezed: str


def normalise(raw: str) -> Normalised:
    """Fold `raw` into the two matching forms. Pure, and idempotent on its own `text`.

    **Order is load-bearing at three points.**

    1. *Format characters go first.* A zero-width joiner inside `ig<ZWJ>nore` is invisible to a
       reader and splits the word for every step after this one, so removing category `Cf`
       (which covers ZWSP/ZWNJ/ZWJ, the BOM and the soft hyphen) has to precede everything.
    2. *NFKC before casefold, then NFKD.* NFKC folds the compatibility forms an attacker reaches
       for — fullwidth Latin, mathematical alphanumerics, ligatures — into ASCII letters, so
       casefold and the homoglyph map below see letters rather than variants. The NFKD pass that
       follows decomposes accents so the combining marks can be dropped; doing it before
       casefold would leave a casefold on characters that are about to change again.
    3. *Homoglyphs before leetspeak.* Cyrillic `о` becomes Latin `o`, which the leet map then
       leaves alone — where the reverse order would have the leet map see a character it has no
       entry for and pass it to the denylist as a non-Latin letter.
    """
    without_invisibles = "".join(
        character for character in raw if unicodedata.category(character) != "Cf"
    )
    folded = unicodedata.normalize("NFKC", without_invisibles).casefold()
    decomposed = unicodedata.normalize("NFKD", folded)
    unaccented = "".join(
        character
        for character in decomposed
        # `Mn` is a non-spacing combining mark — the accent itself, once NFKD has split it off
        # its base letter. Dropping it is what turns `é` into `e`.
        if unicodedata.category(character) != "Mn"
    )
    substituted = unaccented.translate(_HOMOGLYPHS).translate(_LEETSPEAK)
    text = _WHITESPACE.sub(" ", _NOT_ALPHANUMERIC.sub(" ", substituted)).strip()
    return Normalised(raw=raw, text=text, squeezed=text.replace(" ", ""))
