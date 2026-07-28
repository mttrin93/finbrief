"""The control flow of `scripts/security_suite.py`: exit codes, flags, committed evidence.

The script is non-hermetic — a classifier call per corpus case, a paid embedding of a throwaway
collection, and real agent turns — and the suite must never invoke it for real (CLAUDE.md). Its
*control flow* is hermetic, and it is where the properties an operator relies on live: that a
failing suite exits non-zero so this is usable as a gate rather than only as a generator, that
`--gate-only` sets **both** the empty `injections` and the `gate_only` flag that makes the
artifact say so, and that `--no-write` prints instead of overwriting the evidence.

The two other non-hermetic entry points each have a file like this
(`test_ingest_script.py`, `test_retrieval_smoke_script.py`); the newest and most consequential
one had none (issue #8 review). Every test `chdir`s for the reason those do: the report path is
relative, and a run under test that skipped it would overwrite committed evidence.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from finbrief.config import Settings
from finbrief.security import corpus
from finbrief.security.input_gate import Layer
from finbrief.security.report import SuiteRun

SCRIPT = Path(__file__).parent.parent / "scripts" / "security_suite.py"

#: Where the script writes, relative to the working directory — hence the `chdir`.
REPORT = Path("docs/verification/security-gate.md")

SETTINGS = Settings.from_env({"OPENROUTER_API_KEY": "test-key"})


def load_script():
    spec = importlib.util.spec_from_file_location("security_suite_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Turn:
    """The shape `agent.answer` returns, as much of it as `run_indirect` reads."""

    def __init__(self, text: str, contexts=()):
        self.text = text
        self.contexts = contexts


class Body:
    """A `Context`, as much of it as `run_indirect` reads."""

    def __init__(self, body: str):
        self.body = body


@pytest.fixture
def script(monkeypatch, tmp_path):
    """The script with a scratch CWD and every paid collaborator replaced.

    The gate and the output validator run for real — they are deterministic once the classifier
    is stubbed, and stubbing them would leave nothing under test. What is replaced is the money:
    the classifier's model, the embedding, the store and the agent.
    """
    module = load_script()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "docs" / "verification").mkdir(parents=True)
    monkeypatch.setattr(module, "get_settings", lambda: SETTINGS)

    # Layer 3 as a live run that got every case right. Conftest's autouse stub answers `SAFE` to
    # everything, which would fail every `CLASSIFIER` corpus row for a reason that is about the
    # stub rather than about the script — and these tests are about the script's branching, not
    # about what a real model recognises (that is the artifact's subject, and it cannot be
    # tested; see the module docstring).
    from finbrief.security import input_gate
    from finbrief.security.classifier import Verdict

    novel = {case.payload for case in corpus.DIRECT_CASES if case.caught_by is Layer.CLASSIFIER}

    def as_a_correct_classifier(question, *, model=None, settings=None):
        return Verdict.INJECTION if question in novel else Verdict.SAFE

    monkeypatch.setattr(input_gate, "classify", as_a_correct_classifier)
    return module


def a_run(script, monkeypatch, *, answers_by_id=None, gate_only=False) -> SuiteRun:
    """Drive `main()` and hand back the `SuiteRun` it built, plus the exit code.

    The run is captured off `render_report` rather than reconstructed, so what is asserted is
    the object the artifact was rendered from.
    """
    captured: list[SuiteRun] = []
    rendered = []

    real_render = script.render_report

    def spy(run: SuiteRun) -> str:
        captured.append(run)
        text = real_render(run)
        rendered.append(text)
        return text

    monkeypatch.setattr(script, "render_report", spy)
    monkeypatch.setattr(
        script, "run_indirect", lambda settings, directory: _planted(answers_by_id)
    )
    argv = ["--gate-only"] if gate_only else []
    code = script.main(argv)
    return captured[0], code, rendered[0]


def _planted(answers_by_id):
    """`run_indirect`'s return value for a run in which every payload was resisted."""
    from finbrief.security.report import InjectionResult

    return tuple(
        InjectionResult(
            payload_id=payload.id,
            technique=payload.technique,
            retrieved=True,
            obeyed=False,
            leaked=False,
            refused_by_validator=False,
            excerpt=(answers_by_id or {}).get(payload.id, "A grounded answer [1]."),
        )
        for payload in corpus.PLANTED_PAYLOADS
    )


def test_a_green_full_run_writes_the_artifact_and_exits_zero(script, monkeypatch):
    run, code, _ = a_run(script, monkeypatch)

    assert code == 0
    assert run.passed
    assert REPORT.read_text(encoding="utf-8").count("SUITE PASSED") == 1


def test_a_failing_suite_exits_non_zero_so_this_is_usable_as_a_gate(script, monkeypatch):
    """The claim the module docstring makes about being a gate and not only a generator.

    Failed by making one planted payload obeyed, which is the failure the paid half exists to
    detect — rather than by hand-building a `SuiteRun`, so the exit code is reached through the
    same `run.passed` the artifact is rendered from.
    """
    monkeypatch.setattr(
        script,
        "run_indirect",
        lambda settings, directory: tuple(
            r.__class__(**{**{f: getattr(r, f) for f in r.__slots__}, "obeyed": True})
            for r in _planted(None)[:1]
        ),
    )
    code = script.main([])

    assert code == 1


def test_gate_only_sets_both_the_empty_half_and_the_flag_that_says_so(script, monkeypatch):
    """The pair CLAUDE.md's "never commit a partial run as a full one" rests on.

    A run that emptied `injections` but forgot `gate_only` would render a passing `0/0` with no
    banner; one that set the flag without emptying them would claim coverage it skipped. Both
    halves are asserted, and so is the rendered consequence.
    """
    run, code, report = a_run(script, monkeypatch, gate_only=True)

    assert run.injections == ()
    assert run.gate_only is True
    assert code == 0, "a green partial run is a legitimate cheap check"
    assert "PARTIAL RUN" in report
    assert "0/0 planted" not in report


def test_a_full_run_carries_no_partial_banner(script, monkeypatch):
    """The other direction, because the banner is only useful if its absence means something."""
    _, _, report = a_run(script, monkeypatch)

    assert "PARTIAL RUN" not in report
    assert f"{len(corpus.PLANTED_PAYLOADS)}/{len(corpus.PLANTED_PAYLOADS)} planted" in report


def test_no_write_prints_the_report_and_leaves_the_artifact_alone(script, monkeypatch, capsys):
    monkeypatch.setattr(script, "render_report", lambda run: "RENDERED REPORT")
    monkeypatch.setattr(script, "run_indirect", lambda settings, directory: _planted(None))

    script.main(["--no-write"])

    assert "RENDERED REPORT" in capsys.readouterr().out
    assert not REPORT.exists()


def test_the_gate_rows_cover_every_corpus_case_and_every_benign_question(script):
    """`run_gate`'s two halves, so a corpus addition cannot silently go unscreened."""
    results = script.run_gate(SETTINGS)

    assert len(results) == len(corpus.DIRECT_CASES) + len(corpus.BENIGN_QUESTIONS)
    assert {r.case_id for r in results} >= {c.id for c in corpus.DIRECT_CASES}
    assert {r.case_id for r in results} >= set(corpus.BENIGN_QUESTIONS)


def drive_indirect(script, monkeypatch, tmp_path, turn_for):
    """Run `run_indirect` with the paid collaborators replaced, and hand back its rows.

    The three imports inside `run_indirect` are patched at their source modules, not on the
    script, because that is where the function resolves them.
    """
    from finbrief.retrieval import embeddings, vectorstore

    monkeypatch.setattr(embeddings, "build_embeddings", lambda settings: object())
    monkeypatch.setattr(vectorstore, "build_filings_store", lambda **kwargs: object())
    monkeypatch.setattr(vectorstore, "write_chunks", lambda store, chunks: None)
    monkeypatch.setattr(script, "build_agent", lambda **kwargs: object())
    monkeypatch.setattr(script, "build_checkpointer", lambda path: object())
    monkeypatch.setattr(script, "answer", lambda question, thread_id, agent: turn_for(question))
    return script.run_indirect(SETTINGS, tmp_path)


def test_a_payload_the_turn_never_retrieved_is_recorded_as_such_and_fails(
    script, monkeypatch, tmp_path
):
    """`InjectionResult.retrieved`, driven through the code that derives it.

    The corpus-side test asserts every payload *is* retrievable; nothing asserted that the
    script notices when a turn did not reach one — which is the exact shape of the first live
    run's green cell about nothing (issue #8 review).
    """
    rows = drive_indirect(
        script, monkeypatch, tmp_path, lambda q: Turn("Which company did you mean?")
    )

    assert all(row.retrieved is False for row in rows)
    assert all(row.passed is False for row in rows)


def test_an_empty_canary_is_never_reported_as_obedience(script, monkeypatch, tmp_path):
    """`""` is a substring of every string, so the guard is what keeps a leak row honest.

    Dropping `bool(payload.canary) and` would report every extraction payload as obeyed, and
    `test_every_planted_payload_has_something_to_detect` asserts about the *corpus* rather than
    about the code that reads it.
    """
    bodies = {p.question: Body(p.body) for p in corpus.PLANTED_PAYLOADS}
    rows = drive_indirect(
        script,
        monkeypatch,
        tmp_path,
        lambda q: Turn("A grounded answer [1].", contexts=(bodies[q],)),
    )
    leaks = {p.id for p in corpus.PLANTED_PAYLOADS if not p.canary}

    assert leaks, "the fixture is meaningless if no payload is an extraction one"
    assert all(row.obeyed is False for row in rows if row.payload_id in leaks)
    assert all(row.retrieved for row in rows)


def test_a_canary_in_the_answer_is_obedience_and_a_prompt_fragment_is_a_leak(
    script, monkeypatch, tmp_path
):
    """The other two derivations, at the seam that computes them."""
    bodies = {p.question: p for p in corpus.PLANTED_PAYLOADS}
    fragment = corpus.SYSTEM_PROMPT_FRAGMENTS[0]

    def obedient(question):
        payload = bodies[question]
        return Turn(f"{payload.canary} {fragment}".strip(), contexts=(Body(payload.body),))

    rows = drive_indirect(script, monkeypatch, tmp_path, obedient)
    by_id = {row.payload_id: row for row in rows}

    assert all(row.leaked for row in rows)
    assert all(by_id[p.id].obeyed is bool(p.canary) for p in corpus.PLANTED_PAYLOADS)
    assert all(row.passed is False for row in rows)


def test_the_output_validator_half_labels_each_answer_set_correctly(script):
    """The `expected_refusal` flag, which is what makes a layer-4 row a pass or a fail."""
    results = script.run_output_validator()
    expected = {r.text: r.expected_refusal for r in results}

    assert all(expected[a] is True for a in corpus.ADVICE_ANSWERS)
    assert all(expected[a] is False for a in corpus.RESEARCH_ANSWERS)
    assert all(r.passed for r in results), [r.text for r in results if not r.passed]
