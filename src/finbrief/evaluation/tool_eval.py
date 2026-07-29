"""Tool-selection accuracy over scripted prompts (spec US-29), and the AC-1 pairing metric.

This is the **other** half of ADR-0003's split. Every table in `report.py` measures the chain;
this measures the agent's *selection layer* on top of it — a different and nondeterministic
instrument, reported beside those tables and never inside them.

**Pass/fail against the golden set's expectations, plus negative controls.** Seven cases come
from `tool_expectation` (ADR-0002's amendment: the additional non-retrieval tool a row needs,
with the arguments the call must carry). Three are controls the golden set cannot express,
because it holds no row whose right answer is *not to call a tool*:

- a retrieval-only question must call `search_filings` and **no** finance tool — a model that
  reaches for a quote on a risk-factors question is wrong in a way accuracy over positive cases
  cannot see;
- an out-of-Universe ticker must be refused as a *result* rather than answered (user story 21,
  ADR-0009's whitelist);
- an advice question must not send the loop to price the recommendation — layer 4 owns the
  refusal and the security suite reports it, so what is scored here is the *selection*: no
  finance tool.

**A control needs something it can fail on**, and C3 had nothing (code review of #11). With
`expected`, `forbidden` and `args` all empty, `missing`, `forbidden_called` and `wrong_args`
were all empty for every possible agent behaviour, so `passed` was `True` whether the agent
called no
tool, one, or all three — a green cell inside a published accuracy rate, which is this repo's
named bug class. Its own note said it was "scored on whether a tool fired" and nothing looked.

**The pairing hole #9 recorded is measured separately, and deliberately not by reshaping the
golden set.** `AGENT_SYSTEM_PROMPT` says a valuation question wants the quote *and* the peer
comparison; `tool_expectation` records one tool per row. So the pairing is its own rate over the
valuation cases (`pairing_rate`), leaving the committed, hand-verified artifact alone — ADR-0002
describes that field as recording *the additional* tool, and rewriting reference data to carry a
measurement is the wrong direction. Option 3 from #9's comment (leave it unmeasured and say so)
is explicitly not taken.

**Nothing here is scored on the answer's text.** Whether the agent called the right tool with
the right arguments is a fact about the loop; whether what it then said is faithful is the
chain's metric, and conflating them is the category error ADR-0003 exists to prevent.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from finbrief.evaluation.loader import Bucket, GoldenSet
from finbrief.tools.finance import FINANCE_TOOL_NAMES, RATIOS_TOOL_NAME, STOCK_TOOL_NAME
from finbrief.tools.search_filings import TOOL_NAME as SEARCH_TOOL_NAME


@dataclass(frozen=True, slots=True)
class ToolCase:
    """One scripted prompt, and what the selection layer has to do with it."""

    id: str
    question: str
    #: Tools that must be called. Empty for a case whose right answer calls no finance tool.
    expected: frozenset[str] = frozenset()
    #: Tools that must **not** be called — how a negative control is expressed.
    forbidden: frozenset[str] = frozenset()
    #: Arguments the expected call must carry, e.g. `{"ticker": "F"}`.
    args: Mapping[str, str] = field(default_factory=dict)
    #: Why this case is here. Rendered into the artifact, so a reader can see what it probes.
    note: str = ""
    #: `True` for the three controls, so accuracy can be reported with and without them.
    control: bool = False


#: The three cases the golden set cannot express, because every row in it expects a tool.
CONTROL_CASES: tuple[ToolCase, ...] = (
    ToolCase(
        id="C1",
        question="What are the main risk factors for Tesla?",
        expected=frozenset({SEARCH_TOOL_NAME}),
        forbidden=FINANCE_TOOL_NAMES,
        note=(
            "retrieval-only: reaching for a quote on a risk-factors question is an error "
            "accuracy over positive cases cannot see"
        ),
        control=True,
    ),
    ToolCase(
        id="C2",
        question="What is Nestle's dividend policy?",
        forbidden=FINANCE_TOOL_NAMES,
        note=(
            "out-of-Universe: the whitelist should make this a refusal-as-result rather than a "
            "fetch (user story 21, ADR-0009)"
        ),
        control=True,
    ),
    ToolCase(
        id="C3",
        question="Should I buy Ford shares today?",
        forbidden=FINANCE_TOOL_NAMES,
        note=(
            "advice-shaped: the selection layer must not go and price the recommendation. The "
            "refusal itself is layer 4's and the security suite's; what is scored here is that "
            "no market-data fetch was made in service of one"
        ),
        control=True,
    ),
)


def cases_from(golden: GoldenSet) -> tuple[ToolCase, ...]:
    """The seven `tool-augmented` rows as cases, then the three controls.

    Built from `tool_expectation` rather than retyped, so a change to the golden set changes
    what is scored — the same reason `peer_set` is asserted against `config.PEERS` rather than
    copied.
    """
    rows = golden.by_bucket()[Bucket.TOOL_AUGMENTED]
    scripted = tuple(
        ToolCase(
            id=row.id,
            question=row.question,
            expected=frozenset({row.tool_expectation.name, SEARCH_TOOL_NAME}),
            args=dict(row.tool_expectation.args),
            note=f"golden-set row {row.id}",
        )
        for row in rows
        if row.tool_expectation is not None
    )
    return scripted + CONTROL_CASES


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    """What one case's live turn actually called, and whether that satisfied the case."""

    case: ToolCase
    #: `(tool, ticker)` per call, in the order the agent made them.
    calls: tuple[tuple[str, str | None], ...]
    #: `None` when the turn raised — a failed turn is not a failed selection, and saying it is
    #: would blame the model for an outage.
    failed: str | None = None

    @property
    def tools(self) -> frozenset[str]:
        return frozenset(tool for tool, _ in self.calls)

    @property
    def missing(self) -> frozenset[str]:
        return self.case.expected - self.tools

    @property
    def forbidden_called(self) -> frozenset[str]:
        return self.case.forbidden & self.tools

    @property
    def wrong_args(self) -> tuple[str, ...]:
        """Expected tools whose call did not carry the expected ticker.

        Only the ticker is checked, because it is the only argument `Step` carries — a `days` on
        `get_recent_news` is unconstrained by the golden set anyway (`field_notes`).
        """
        wanted = self.case.args.get("ticker")
        if wanted is None:
            return ()
        return tuple(
            tool
            for tool in sorted(self.case.expected & self.tools)
            if tool != SEARCH_TOOL_NAME
            and wanted not in {ticker for called, ticker in self.calls if called == tool}
        )

    @property
    def passed(self) -> bool | None:
        """`None` when the turn failed — absent, not a failure of the thing being measured."""
        if self.failed is not None:
            return None
        return not (self.missing or self.forbidden_called or self.wrong_args)

    @property
    def reason(self) -> str:
        if self.failed is not None:
            return f"turn failed: {self.failed}"
        problems = []
        if self.missing:
            problems.append(f"never called {', '.join(sorted(self.missing))}")
        if self.forbidden_called:
            problems.append(f"called forbidden {', '.join(sorted(self.forbidden_called))}")
        if self.wrong_args:
            expected = self.case.args.get("ticker")
            problems.append(f"{', '.join(self.wrong_args)} not called with {expected}")
        return "; ".join(problems) or "as expected"


def run_case(
    case: ToolCase,
    *,
    ask: Callable[[str, list[tuple[str, str | None]]], None],
) -> ToolOutcome:
    """Drive one case through `ask`, which records the steps the agent took.

    `ask` takes the question and a list to append `(tool, ticker)` to — the shape
    `agent.answer`'s `on_step` callback produces. Injected rather than imported so a hermetic
    test can script the selection layer, since this module's whole subject is what a *live*
    model does.
    """
    calls: list[tuple[str, str | None]] = []
    try:
        ask(case.question, calls)
    except Exception as exc:  # noqa: BLE001 — an outage is not a selection failure
        return ToolOutcome(case=case, calls=tuple(calls), failed=f"{type(exc).__name__}: {exc}")
    return ToolOutcome(case=case, calls=tuple(calls))


@dataclass(frozen=True, slots=True)
class ToolReport:
    """The rates, each with the denominator it was computed over."""

    outcomes: tuple[ToolOutcome, ...]

    @property
    def scored(self) -> tuple[ToolOutcome, ...]:
        """Outcomes whose turn completed — the only ones a rate may be computed over."""
        return tuple(outcome for outcome in self.outcomes if outcome.passed is not None)

    @property
    def failed_turns(self) -> int:
        return len(self.outcomes) - len(self.scored)

    def accuracy(self, *, include_controls: bool = True) -> float | None:
        """Share of scored cases that selected correctly, or `None` over an empty
        denominator."""
        selected = [
            outcome for outcome in self.scored if include_controls or not outcome.case.control
        ]
        if not selected:
            return None
        return sum(1 for outcome in selected if outcome.passed) / len(selected)

    @property
    def pairing_rate(self) -> tuple[int, int]:
        """`(paired, valuation cases)` — #9's AC-1 hole, measured as its own rate.

        A valuation case is one expecting `calculate_ratios`; the prompt asks for the quote
        beside it. Returned as a fraction rather than a float so a reader sees the
        denominator, which on this set is **2**.
        """
        valuation = [
            outcome for outcome in self.scored if RATIOS_TOOL_NAME in outcome.case.expected
        ]
        paired = sum(
            1 for outcome in valuation if {RATIOS_TOOL_NAME, STOCK_TOOL_NAME} <= outcome.tools
        )
        return paired, len(valuation)


def render(report: ToolReport) -> str:
    """The tool-calling section of the artifact."""
    paired, valuation = report.pairing_rate
    overall = report.accuracy()
    without = report.accuracy(include_controls=False)
    lines = [
        "ADR-0003's *other* half: every table above measures the deterministic chain, and this "
        "measures the agent's selection layer on top of it. A different instrument, and a "
        "nondeterministic one — so it is reported beside those tables and never inside them.",
        "",
        f"**Tool-selection accuracy: {_pct(overall)} over {len(report.scored)} scored cases** "
        f"({_pct(without)} over the seven golden-set rows alone; the other three are controls "
        f"the golden set cannot express, because every row in it expects a tool).",
        "",
        "**Every case in that denominator can fail, which was not true when this rate was "
        "first published.** Control C3 declared no expected tool, no forbidden tool and no "
        "argument, so its `passed` was `True` for every possible agent behaviour and the rate "
        "was a claim about nine cases wearing a denominator of ten. Each control now forbids "
        "the finance tools its note describes, and "
        "`tests/test_eval_tool_eval.py::test_every_control_can_fail` scores each of them "
        "against an agent scripted to call all three — a control whose `passed` survives that "
        "fails the suite. The rate is unchanged; what changed is that it is now a rate.",
        "",
        f"**Valuation pairing (#9's AC-1 hole): {paired}/{valuation}.** "
        f"`AGENT_SYSTEM_PROMPT` says a valuation question wants the quote *and* the peer "
        f"comparison, while `tool_expectation` records one tool per row — so this is measured "
        f"as "
        f"its own rate rather than by reshaping the committed golden set. Measured, not "
        f"enforced.",
        "",
        "| case | question | expected | called | verdict | note |",
        "|---|---|---|---|---|---|",
    ]
    for outcome in report.outcomes:
        expected = ", ".join(sorted(outcome.case.expected)) or "no finance tool"
        called = ", ".join(f"{tool}({ticker or ''})" for tool, ticker in outcome.calls) or "—"
        verdict = (
            "—" if outcome.passed is None else ("**pass**" if outcome.passed else "**fail**")
        )
        lines.append(
            f"| {outcome.case.id} | {outcome.case.question} | {expected} | {called} "
            f"| {verdict} | {outcome.reason if not outcome.passed else outcome.case.note} |"
        )
    if report.failed_turns:
        lines += [
            "",
            f"{report.failed_turns} turn(s) failed outright and are **excluded from the rate "
            f"rather than counted as failures** — an outage is not a selection error, and "
            f"counting it as one would blame the model for the network.",
        ]
    return "\n".join(lines)


def _pct(value: float | None) -> str:
    return "not measured" if value is None else f"{value:.0%}"


def scripted_asker(
    answer: Callable[..., object], *, agent: object, thread_prefix: str
) -> Callable[[str, list[tuple[str, str | None]]], None]:
    """An `ask` that drives the real agent, one fresh thread per case.

    A fresh `thread_id` per case, because the checkpointer is the agent's memory (ADR-0008)
    and a shared thread would let case N's tool calls be answered from case N−1's conversation
    — which would measure the memory rather than the selection.
    """
    counter = {"n": 0}

    def ask(question: str, calls: list[tuple[str, str | None]]) -> None:
        counter["n"] += 1
        answer(
            question,
            thread_id=f"{thread_prefix}-{counter['n']}",
            agent=agent,
            on_step=lambda step: calls.append((step.tool, step.ticker)),
        )

    return ask


def run_all(
    cases: Sequence[ToolCase],
    *,
    ask: Callable[[str, list[tuple[str, str | None]]], None],
) -> ToolReport:
    """Every case through `ask`, in order."""
    return ToolReport(outcomes=tuple(run_case(case, ask=ask) for case in cases))
