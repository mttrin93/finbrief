"""The security gate: four layers, each with a job the others do not do (ADR-0006).

Three of the four live here; the fourth is a framing this package assumes rather than owns.

    normalize.normalise()      layer 1 — folds obfuscation into one surface form
    denylist.denylisted()      layer 2 — known payloads, free, no model call
    classifier.classify()      layer 3 — novel phrasings, one cheap model call
    input_gate.screen()        the front door: the three above, cheap-first
    advice.validate_answer()   the back door — Guardrails AI, no investment advice
    markers.markers()          the back door's deterministic half — `[n]` naming no source

`prompts.py` owns the fourth: retrieved text is quarantined as data, which is what makes the
back door necessary rather than sufficient (ADR-0006 amendment, ticket T3).

**`screen` and `validate_answer` are the two seams**; everything else is a layer one of them
composes. Nothing is re-exported here, on the repo's convention that a package `__init__` is
documentation: a layer imported from the module that owns it reads as the layer it is, which is
what a marginal-contribution claim has to be argued about (user story 34).
"""
