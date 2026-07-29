"""The configuration matrix: ADR-0002's four arms, plus the two ablation cells §7 needs.

An `Arm` is the *only* place a strategy and a translation switch are paired with a name, so a
number in the artifact can always be traced to a configuration that ran. That is the same rule
`retrieve()` enforces from the other side by refusing to read either switch from configuration:
"a number is never reported against a configuration nobody selected" (ADR-0003 amendment).

**Four scored arms and two ablations, and the ablations are not an afterthought.** ADR-0005 §2
names `FINBRIEF_MAX_SUB_QUERIES=0` as the channel that refutes its own revised hypothesis —
"if the exact-identifier win does not survive with the planner off, then normalisation is not
what earned it" — and ADR-0004 §7's `vector + normalisation` ablation is the other half of the
same question. Both are *deterministic*: translation is on, so the ticker form is still added,
but the planner is never invoked. They cost embeddings and no chat round.

**Generation-side metrics run on the four, not the six.** The ablations exist to attribute a
*retrieval* result, and faithfulness over an answer is not evidence about which candidate list
found a chunk. `judged` is what says so, per arm, rather than a condition buried in the runner.
"""

from __future__ import annotations

from dataclasses import dataclass

from finbrief.config import (
    DEFAULT_MAX_SUB_QUERIES,
    DEFAULT_STRATEGY,
    DEFAULT_TRANSLATION_ENABLED,
    RetrievalStrategy,
)

#: The sub-query cap the shipping default runs at (`Settings.max_sub_queries`' own default, and
#: ADR-0004's latency-driven ceiling). Named here so the arms below cannot drift from it — and
#: read from `config.py` rather than retyped, because it was a hardcoded `3` that
#: `shipping_default_matches_config` did not check (code review of #11).
SHIPPED_MAX_SUB_QUERIES = DEFAULT_MAX_SUB_QUERIES


@dataclass(frozen=True, slots=True)
class Arm:
    """One configuration of `retrieve()`, named once.

    `strategy` and `translate` are the two axes ADR-0002's matrix compares. `max_sub_queries` is
    not a third axis — it is `0` only on the two ablation cells, where it removes the planner
    call outright rather than truncating its output (ADR-0004 §9).
    """

    name: str
    strategy: RetrievalStrategy
    translate: bool
    max_sub_queries: int
    #: Whether the generation-side metrics (faithfulness, response relevancy) run on this arm.
    judged: bool
    #: What the artifact's row header says. Written out rather than derived from the fields so a
    #: table heading reads as prose, and asserted against the fields by the tests.
    label: str

    @property
    def plans(self) -> bool:
        """Whether the sub-query planner runs at all — `translate` **and** a non-zero cap.

        The distinction `Retrieval.planned` draws, for the same reason: "no sub-query came back"
        is a fact about the configuration on an ablation arm and a fact about the model on a
        scored one.
        """
        return self.translate and self.max_sub_queries > 0

    @property
    def deterministic(self) -> bool:
        """Whether this arm reaches no sampled step at all (ADR-0004 §9).

        True for the two `−translation` baselines *and* for both ablations, which is what makes
        ADR-0005 §2's falsification channel exact. The two `+translation` arms are exact only
        because their variants are replayed through a stub (`evaluation/variants.py`); this
        property is about the configuration, not about the replay.
        """
        return not self.plans


VECTOR_ONLY = Arm(
    name="vector",
    strategy=RetrievalStrategy.VECTOR,
    translate=False,
    max_sub_queries=0,
    judged=True,
    label="vector, no translation",
)
VECTOR_TRANSLATED = Arm(
    name="vector+translation",
    strategy=RetrievalStrategy.VECTOR,
    translate=True,
    max_sub_queries=SHIPPED_MAX_SUB_QUERIES,
    judged=True,
    label="vector + translation",
)
HYBRID_ONLY = Arm(
    name="hybrid",
    strategy=RetrievalStrategy.HYBRID,
    translate=False,
    max_sub_queries=0,
    judged=True,
    label="hybrid, no translation",
)
HYBRID_TRANSLATED = Arm(
    name="hybrid+translation",
    strategy=RetrievalStrategy.HYBRID,
    translate=True,
    max_sub_queries=SHIPPED_MAX_SUB_QUERIES,
    judged=True,
    label="hybrid + translation (shipping default)",
)
VECTOR_NORMALISED = Arm(
    name="vector+normalisation",
    strategy=RetrievalStrategy.VECTOR,
    translate=True,
    max_sub_queries=0,
    judged=False,
    label="vector + normalisation, planner off (ADR-0004 §7 ablation)",
)
HYBRID_NORMALISED = Arm(
    name="hybrid+normalisation",
    strategy=RetrievalStrategy.HYBRID,
    translate=True,
    max_sub_queries=0,
    judged=False,
    label="hybrid + normalisation, planner off (ADR-0005 §2 channel)",
)

#: ADR-0002's matrix: the four cells the per-bucket A/B compares, in `vector → hybrid` order so
#: a table's rows read as "add BM25" then "add translation".
SCORED_ARMS: tuple[Arm, ...] = (
    VECTOR_ONLY,
    VECTOR_TRANSLATED,
    HYBRID_ONLY,
    HYBRID_TRANSLATED,
)

#: The two planner-off cells. Not part of the matrix, and reported in their own table.
ABLATION_ARMS: tuple[Arm, ...] = (VECTOR_NORMALISED, HYBRID_NORMALISED)

ALL_ARMS: tuple[Arm, ...] = SCORED_ARMS + ABLATION_ARMS

#: The pre-registered shipping default (ADR-0005), and the arm every deferral measurement and
#: every generation-side headline number is reported against.
SHIPPING_DEFAULT = HYBRID_TRANSLATED

#: The two arms ADR-0005's falsification clause differences, and the two its §4 re-examination
#: trigger differences. Named as data because `hypotheses.py` reads them: a clause whose
#: operands are written out in prose is one that can be applied to the wrong pair.
#:
#: **It reads them now.** That sentence was here while `hypotheses.py` wrote all five pairs out
#: by hand and consulted neither constant, so `tests/test_eval_arms.py`'s axis-constancy
#: assertions guarded data no verdict depended on (code review of #11).
#: `hypotheses.CLAUSE_CONTRAST` and `TRIGGER_CONTRAST` now select from these by predicate — the
#: pair whose second arm is `SHIPPING_DEFAULT` — which is what makes those tests bind the
#: operands they are about.
TRANSLATION_CONTRAST: tuple[tuple[Arm, Arm], ...] = (
    (VECTOR_ONLY, VECTOR_TRANSLATED),
    (HYBRID_ONLY, HYBRID_TRANSLATED),
)
STRATEGY_CONTRAST: tuple[tuple[Arm, Arm], ...] = (
    (VECTOR_ONLY, HYBRID_ONLY),
    (VECTOR_TRANSLATED, HYBRID_TRANSLATED),
)


def shipping_default_matches_config() -> bool:
    """Whether `SHIPPING_DEFAULT` is still the configuration `config.py` ships.

    A function rather than an assertion at import: `config`'s defaults are the shipped ones and
    this module's job is to *measure* them, so the two must agree or the headline arm is not the
    arm the app runs. `tests/test_eval_arms.py` is what fails when they part.
    """
    return (
        SHIPPING_DEFAULT.strategy is DEFAULT_STRATEGY
        and SHIPPING_DEFAULT.translate is DEFAULT_TRANSLATION_ENABLED
        # **The cap too, and it was the one axis this did not check** (code review of #11).
        # `settings_for` forces every arm to its own `max_sub_queries`, so with
        # `FINBRIEF_MAX_SUB_QUERIES` set to anything but 3 the resolve pass records plans at the
        # env's cap while the arms retrieve at this one — the persisted `variants` then do not
        # describe the cell keyed on them. At 0 it is worse: the planner is never invoked, the
        # empty reply is persisted, and both `+translation` arms retrieve identically to their
        # own ablations while the artifact labels one "the shipping default".
        and SHIPPED_MAX_SUB_QUERIES == DEFAULT_MAX_SUB_QUERIES
    )
