"""Resolve each question's query variants once, persist them, replay them (ADR-0004 §9).

Exactly one step inside `retrieve()` samples: the sub-query planner's chat completion. Left
alone, that makes two of ADR-0002's four arms reproducible only up to temperature-0 sampling,
and **the shipping default is one of the two** — so a re-run of this ticket could report a
different per-bucket number with no code change. ADR-0004 §9 pre-registered the fix before any
A/B data existed, and this module is it:

1. resolve every question's variants **once**, against the real planner;
2. persist them beside the golden set as a fixed input, versioned with it;
3. replay them through a stub model on every arm, so the four arms differ **only** by `strategy`
   and `translate` — which is what ADR-0002's matrix claims to compare;
4. report the planner's own variance separately, as an n-repeat (`agreement` below), because it
   is a property of the shipped path and not of the fusion strategy. Folding it into a
   strategy's error bars would make the strategy comparison inherit noise from a component it
   is not about.

**What is persisted is the planner's raw reply, not the parsed sub-queries.** Replaying the
reply puts `query_translation.sub_queries` — list-furniture stripping, refusal detection, the
duplicate guard — back in the path, so what the arms retrieve over is what the *code* produces
from what the *model* said, rather than what a previous version of the parser produced. It
also answers ADR-0004 §9's second complaint directly: "a scored run's variants cannot be
inspected after the fact". They can now; they are in a committed file, with the reply that
produced them.

**And the file is checked against the code that consumes it, on load.** `verify_replay` re-runs
the real `translate()` over each persisted reply and compares the result with the persisted
variants. A parser change that would silently alter what every arm retrieved over fails there
instead — which matters because this file is an *input* to a measurement, and an input nothing
checks is an input that drifts.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from finbrief.evaluation.loader import GoldenSet
from finbrief.retrieval import query_translation

#: The persisted variants, beside the golden set because ADR-0004 §9 makes them the same kind of
#: thing: a fixed input to the measurement, hand-checkable, versioned with the questions.
VARIANTS_PATH = Path(__file__).with_name("golden_variants.json")

SCHEMA_VERSION = 1


class ReplayMismatch(RuntimeError):
    """A persisted reply no longer parses to the variants persisted beside it."""


class MissingPlan(KeyError):
    """A question has no persisted variants, so its `+translation` arms cannot be replayed."""


@dataclass(frozen=True, slots=True)
class ResolvedPlan:
    """One question's planner call, as a replayable record.

    `variants` is what `translate()` returned at resolve time. It is stored **as well as** the
    reply so that `verify_replay` has something to compare against — the reply alone would make
    a parser change invisible, which is the failure this file exists to prevent.
    """

    question_id: str
    question: str
    #: The planner's reply, verbatim. Replayed through `ReplayPlanner` so the real parser runs.
    reply: str
    variants: tuple[str, ...]
    planner_model: str
    max_sub_queries: int

    @property
    def sub_queries(self) -> tuple[str, ...]:
        """What the *planner* added — the deterministic ticker form excluded."""
        return query_translation.added_variants(self.variants)[1]

    @property
    def ticker_form(self) -> str | None:
        """The deterministic ticker form, if a Universe company was named."""
        return query_translation.added_variants(self.variants)[0]

    def as_payload(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "reply": self.reply,
            "variants": list(self.variants),
            "planner_model": self.planner_model,
            "max_sub_queries": self.max_sub_queries,
        }

    @classmethod
    def from_payload(cls, question_id: str, payload: Mapping[str, Any]) -> ResolvedPlan:
        return cls(
            question_id=question_id,
            question=str(payload["question"]),
            reply=str(payload["reply"]),
            variants=tuple(str(variant) for variant in payload["variants"]),
            planner_model=str(payload["planner_model"]),
            max_sub_queries=int(payload["max_sub_queries"]),
        )


class RecordingPlanner(BaseChatModel):
    """Delegates to the real planner and keeps its reply — so the resolve pass logs properly.

    **The fix for a hole this harness created.** `resolve_plan` first called `model.invoke`
    itself and then re-ran `translate()` over the reply through a `ReplayPlanner`. That
    captured the reply but made the **paid** planner call outside
    `query_translation.translate`, which is the only place a `query_translation` event is
    emitted — so the real call's latency and token counts were never recorded, and the only
    lines in the log came from replays, which report no spend. The log then carried 322
    `query_translation` lines and **zero** with token counts, and `latency.translation_cost`
    correctly refused to compute ADR-0005's budget from it.

    Wrapping instead of bypassing means the resolve pass runs the shipped code path:
    `translate()` invokes this, this invokes the real model, and the event `translate()`
    writes is a measurement of the actual call. One code path, which is what CLAUDE.md asks
    for everywhere else and what the first version quietly gave up.
    """

    inner: BaseChatModel
    reply: str = ""

    @property
    def _llm_type(self) -> str:
        return "finbrief-recording-planner"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        result = self.inner._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        # Kept so `resolve_plan` can persist it. Assigning to a pydantic field on a model that
        # is not frozen, which is the same mechanism `ReplayPlanner.invocations` uses.
        self.reply = result.generations[0].message.text()
        return result


class ReplayPlanner(BaseChatModel):
    """A chat model that returns one recorded reply, so an arm's variants are not resampled.

    The whole of ADR-0004 §9's step 2. `retrieve()` already takes an injectable `model=`, so no
    production code changes to make the A/B exact — which is the property that made §9's design
    cheap enough to pre-register.

    It records how many times it was invoked, because "the replay ran" and "the replay was
    reached" are different claims: an arm that never called the planner would produce the
    `−translation` variants and look like a quieter version of the same result.
    """

    reply: str
    invocations: int = 0

    @property
    def _llm_type(self) -> str:
        return "finbrief-replay-planner"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.invocations += 1
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])


@dataclass(frozen=True, slots=True)
class VariantSet:
    """Every question's persisted plan, plus the provenance the artifact has to quote."""

    plans: Mapping[str, ResolvedPlan]
    planner_model: str
    max_sub_queries: int
    golden_set_schema_version: int
    schema_version: int = SCHEMA_VERSION

    def __len__(self) -> int:
        return len(self.plans)

    def plan(self, question_id: str) -> ResolvedPlan:
        try:
            return self.plans[question_id]
        except KeyError as exc:
            raise MissingPlan(
                f"no persisted variants for {question_id!r}. Run the resolve stage "
                f"(scripts/evaluate.py --stage resolve) before scoring a +translation arm: "
                f"the replay is what makes the arms differ only by strategy and translation."
            ) from exc

    def planner(self, question_id: str) -> ReplayPlanner:
        """A stub model that will hand `retrieve()` this question's recorded reply."""
        return ReplayPlanner(reply=self.plan(question_id).reply)

    def covers(self, golden: GoldenSet) -> bool:
        """Whether every row has a plan and no plan belongs to a row that is gone.

        An **equality** on the id sets rather than a subset check: a stale plan for a question
        the set no longer holds is as much a reason to re-resolve as a missing one, and a
        superset would pass a bound while the file described a different golden set.
        """
        return set(self.plans) == {row.id for row in golden}

    def as_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "planner_model": self.planner_model,
            "max_sub_queries": self.max_sub_queries,
            "golden_set_schema_version": self.golden_set_schema_version,
            "note": (
                "Generated by scripts/evaluate.py --stage resolve. ADR-0004 §9: a question's "
                "variants are resolved once and replayed through a stub model on every arm, so "
                "the A/B's four arms differ only by strategy and translation. Never "
                "hand-edited — `finbrief.evaluation.variants.verify_replay` re-parses every "
                "reply and fails if it no longer yields the variants stored beside it."
            ),
            "plans": {
                question_id: plan.as_payload()
                for question_id, plan in sorted(self.plans.items())
            },
        }


def resolve_plan(
    question_id: str,
    question: str,
    *,
    model: BaseChatModel,
    max_sub_queries: int,
    planner_model: str,
) -> ResolvedPlan:
    """One paid planner call, captured as a replayable record.

    The reply is captured by asking the planner directly rather than by calling `translate()`
    and reconstructing it: a reply cannot be recovered from the variants it produced, since
    the parser drops lines.
    """
    recorder = RecordingPlanner(inner=model)
    variants = query_translation.translate(
        question, model=recorder, max_sub_queries=max_sub_queries
    )
    text = recorder.reply
    return ResolvedPlan(
        question_id=question_id,
        question=question,
        reply=text,
        variants=variants,
        planner_model=planner_model,
        max_sub_queries=max_sub_queries,
    )


def verify_replay(plans: Sequence[ResolvedPlan]) -> None:
    """Re-parse every persisted reply and raise unless it yields the variants stored with it.

    The check that keeps a *fixed input* honest. `query_translation.sub_queries` strips list
    furniture, drops prose refusals and de-duplicates against what is already in play — all of
    which can legitimately change — and any of those changes would silently alter what every
    `+translation` arm retrieved over. This turns that into a failure at load.
    """
    for plan in plans:
        replayed = query_translation.translate(
            plan.question,
            model=ReplayPlanner(reply=plan.reply),
            max_sub_queries=plan.max_sub_queries,
        )
        if replayed != plan.variants:
            raise ReplayMismatch(
                f"{plan.question_id}: the persisted reply now parses to {replayed!r}, but the "
                f"variants stored beside it are {plan.variants!r}. Either "
                f"query_translation.sub_queries changed — in which case re-run the resolve "
                f"stage and re-run the A/B, since every +translation arm retrieved over the "
                f"old variants — "
                f"or golden_variants.json was hand-edited, which it must never be."
            )


def load_variants(path: Path | str | None = None, *, verify: bool = True) -> VariantSet:
    """Read the persisted variants, checking each reply still parses to what is stored."""
    path = Path(path) if path is not None else VARIANTS_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    plans = {
        question_id: ResolvedPlan.from_payload(question_id, plan)
        for question_id, plan in payload["plans"].items()
    }
    variant_set = VariantSet(
        plans=plans,
        planner_model=str(payload["planner_model"]),
        max_sub_queries=int(payload["max_sub_queries"]),
        golden_set_schema_version=int(payload["golden_set_schema_version"]),
        schema_version=int(payload["schema_version"]),
    )
    if verify:
        verify_replay(tuple(plans.values()))
    return variant_set


def save_variants(variant_set: VariantSet, path: Path | str | None = None) -> Path:
    """Write the persisted variants. Committed, so the file is stable and sorted."""
    path = Path(path) if path is not None else VARIANTS_PATH
    path.write_text(
        json.dumps(variant_set.as_payload(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


@dataclass(frozen=True, slots=True)
class PlannerAgreement:
    """How often n repeats of the planner returned the same sub-queries for one question.

    ADR-0004 §9 step 3, and it is reported on its own rather than as an error bar on any arm.
    §9 also names what would make it a non-finding: "if the n-repeat finds the planner returns
    identical sub-queries across runs on this Universe and this model, step 2 was unnecessary
    caution". That would be a good outcome and is not one to assume — which is why this is
    measured rather than argued.
    """

    question_id: str
    repeats: int
    #: The distinct sub-query tuples seen, most frequent first, with their counts.
    distinct: tuple[tuple[tuple[str, ...], int], ...]

    @property
    def identical(self) -> bool:
        """Whether every repeat produced the same sub-queries."""
        return len(self.distinct) == 1

    @property
    def modal_share(self) -> float:
        """The share of repeats that produced the most common result."""
        if not self.repeats:
            return 0.0
        return self.distinct[0][1] / self.repeats


def agreement(question_id: str, sub_query_sets: Sequence[tuple[str, ...]]) -> PlannerAgreement:
    """Summarise n repeats' sub-queries for one question.

    Order-sensitive, deliberately: the variants are retrieved in order and RRF's ties break on
    chunk id, so two orderings of the same sub-queries are not guaranteed to fuse identically.
    Calling them "the same" would overstate the planner's stability in exactly the direction
    that flatters the harness.
    """
    counts: dict[tuple[str, ...], int] = {}
    for candidate in sub_query_sets:
        counts[candidate] = counts.get(candidate, 0) + 1
    ranked = tuple(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
    return PlannerAgreement(
        question_id=question_id, repeats=len(sub_query_sets), distinct=ranked
    )
