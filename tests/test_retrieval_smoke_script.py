"""The control flow of `scripts/retrieval_smoke.py`: exit codes and committed evidence.

The script is non-hermetic — it embeds five queries through the paid model and reads the
persisted collection — and the suite must never invoke it for real (CLAUDE.md). Its *control
flow* is hermetic though, and it is where the properties an operator relies on live: that an
un-ingested collection is refused **before** anything is spent on embeddings, that a wrong
top hit exits non-zero rather than quietly writing a green artifact, and that `--no-write`
prints the report it renders instead of paying for a run and discarding it.

`docs/verification/retrieval-smoke.md` is committed evidence of a paid run, so every test
here `chdir`s first: the script's report path is relative, and a run under test that skipped
the `chdir` would overwrite that evidence.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from fakes import a_context

from finbrief.retrieval.retrieve import Retrieval
from finbrief.retrieval.smoke import SMOKE_QUERIES

SCRIPT = Path(__file__).parent.parent / "scripts" / "retrieval_smoke.py"

#: Where the script writes, relative to the working directory — hence the `chdir`.
REPORT = Path("docs/verification/retrieval-smoke.md")


def load_script():
    spec = importlib.util.spec_from_file_location("retrieval_smoke_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_contexts(question: str) -> Retrieval:
    """What a run that got every query right would retrieve, keyed off the query itself.

    A `Retrieval`, because that is what `retrieve()` returns — and the script reads `.contexts`
    off it. `variants` is the question alone: this check runs the plain vector path with
    translation off (`SMOKE_STRATEGY`), so there is never a second variant.
    """
    (query,) = [q for q in SMOKE_QUERIES if q.question == question]
    if query.is_control:
        contexts = (a_context(1, ticker="PFE", distance=1.07),)
    else:
        contexts = (a_context(1, ticker=query.expect_ticker, section=query.expect_section),)
    return Retrieval(contexts=contexts, variants=(question,), translated=False)


def wire(script, monkeypatch, tmp_path, store, retrieve=None):
    """A scratch CWD, a key, a store the script does not have to build, and a fake retrieval.

    `retrieve` defaults to the all-correct run. It is stubbed rather than driven through a
    real fixture collection because what these tests are about is the script's own branching
    — `tests/test_retrieve.py` owns retrieval against a real store.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(script, "build_filings_store", lambda settings=None: store)
    monkeypatch.setattr(
        script,
        "retrieve",
        retrieve or (lambda question, *, strategy, k, **kwargs: expected_contexts(question)),
    )
    # `default_filings_store` is the shared handle production reads; the script builds its own,
    # so patching only `build_filings_store` would leave a second constructor reachable.
    monkeypatch.setattr(script, "build_filings_store", lambda settings=None: store)


def test_an_un_ingested_collection_exits_2_before_spending_anything_on_embeddings(
    monkeypatch, tmp_path, empty_filings_store, capsys
):
    # Every verdict would be "retrieved nothing", which reads as a retrieval bug rather than
    # as the missing ingest it is — and five paid query embeddings would have bought it.
    script = load_script()
    retrieved = []
    wire(
        script,
        monkeypatch,
        tmp_path,
        empty_filings_store,
        retrieve=lambda *a, **k: (
            retrieved.append(a) or Retrieval(contexts=(), variants=(), translated=False)
        ),
    )

    assert script.main([]) == 2

    assert retrieved == [], "the guard must come before the first embedding"
    assert not (tmp_path / REPORT).exists()
    assert "ingest_filings.py" in capsys.readouterr().err


def test_a_run_where_every_query_lands_correctly_exits_0_and_writes_the_evidence(
    monkeypatch, tmp_path, filings_store
):
    script = load_script()
    wire(script, monkeypatch, tmp_path, filings_store)

    assert script.main([]) == 0

    report = (tmp_path / REPORT).read_text(encoding="utf-8")
    assert "SMOKE PASSED" in report
    assert f"{len(SMOKE_QUERIES) - 1}/{len(SMOKE_QUERIES) - 1} check(s) passed" in report


def test_a_wrong_top_hit_exits_1_and_still_writes_the_evidence(
    monkeypatch, tmp_path, filings_store, capsys
):
    # A failing smoke run is exactly the run whose report someone needs to read, so the
    # artifact is written either way — and the exit code is what CI or an operator sees.
    script = load_script()

    def retrieve_tesla_from_the_wrong_filer(question, *, strategy, k, **kwargs):
        if question.startswith("What are the main risk factors"):
            return Retrieval(
                contexts=(a_context(1, ticker="F"),), variants=(question,), translated=False
            )
        return expected_contexts(question)

    wire(script, monkeypatch, tmp_path, filings_store, retrieve_tesla_from_the_wrong_filer)

    assert script.main([]) == 1

    assert "SMOKE FAILED" in (tmp_path / REPORT).read_text(encoding="utf-8")
    assert "wiring failure, not a quality score" in capsys.readouterr().err


def test_the_out_of_kb_control_alone_never_fails_the_run(monkeypatch, tmp_path, filings_store):
    # The control retrieves a far-away chunk from some unrelated filer by definition. If that
    # counted as a wrong top hit, the smoke check would exit 1 on every healthy run — the
    # regression `Verdict.CONTROL` exists to prevent.
    script = load_script()
    wire(script, monkeypatch, tmp_path, filings_store)

    assert script.main([]) == 0

    report = (tmp_path / REPORT).read_text(encoding="utf-8")
    assert "1 control recorded" in report


def test_no_write_prints_the_report_and_leaves_the_committed_artifact_alone(
    monkeypatch, tmp_path, filings_store, capsys
):
    # Both halves matter. Not writing is the flag's purpose; printing is what makes the run
    # worth its five query embeddings, and the distance table is the calibration input the
    # report exists to carry.
    script = load_script()
    wire(script, monkeypatch, tmp_path, filings_store)
    (tmp_path / REPORT).parent.mkdir(parents=True)
    (tmp_path / REPORT).write_text("the last paid run", encoding="utf-8")

    assert script.main(["--no-write"]) == 0

    assert (tmp_path / REPORT).read_text(encoding="utf-8") == "the last paid run"
    out = capsys.readouterr().out
    assert "# Retrieval smoke check" in out, "the rendered report, not just a note about it"
    assert "## Observed distance ranges" in out
    assert "left as the last run wrote it" in out


def test_a_first_ever_run_creates_the_evidence_directory(monkeypatch, tmp_path, filings_store):
    script = load_script()
    wire(script, monkeypatch, tmp_path, filings_store)
    assert not (tmp_path / REPORT).parent.exists()

    assert script.main([]) == 0

    assert (tmp_path / REPORT).exists()


@pytest.mark.parametrize("argv", [[], ["--no-write"]])
def test_the_run_checks_the_strategy_the_app_actually_ships(
    monkeypatch, tmp_path, filings_store, argv
):
    # This is a *wiring* check — "is retrieval reading the collection ingest wrote, embedded by
    # the model that wrote it?" — so it deliberately runs the plain vector path rather than
    # the shipped `hybrid + translation`. Every part of the shipped configuration it added
    # would be a part that can absorb the failure it exists to catch: BM25 matches lexically
    # and would find the right filing even if the embedding model had drifted, which is the
    # drift this script is here to notice. `settings.retrieval_strategy` must therefore *not*
    # be what it follows.
    script = load_script()
    monkeypatch.setenv("FINBRIEF_RETRIEVAL_STRATEGY", "hybrid")
    monkeypatch.setenv("FINBRIEF_QUERY_TRANSLATION", "true")
    ran = []

    def record(question, *, strategy, k, translate=False, **kwargs):
        ran.append((strategy, translate))
        return expected_contexts(question)

    wire(script, monkeypatch, tmp_path, filings_store, record)

    assert script.main(argv) == 0

    assert ran == [(script.SMOKE_STRATEGY, script.SMOKE_TRANSLATION)] * len(SMOKE_QUERIES)
    assert script.SMOKE_TRANSLATION is False, "no paid chat call inside a wiring check"
