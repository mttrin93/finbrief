"""`scripts/evaluate.py`'s control flow: the sink pre-flight, the window, `--rows`, `--stage`.

The script is the fifth and most expensive non-hermetic entry point — ~1,600 judge calls, 112
generations, ~450 embedding requests — and the suite must never invoke `run()` (CLAUDE.md). Its
*control flow* is hermetic, and it is where the properties an operator relies on live:

- the sink pre-flight, which exists so a 30-minute paid run does not end with ADR-0005's latency
  half unmeasurable — and its `--allow-missing-sink` escape hatch, which turns the refusal into
  a warning and must therefore be exercised rather than described;
- the measuring-run-marks / report-only-replays split, including the case a mark was recorded
  against a **different** sink, where applying its offset would slice an unrelated stream;
- `--rows`' unknown-id refusal, which is the difference between a smoke run and a silently
  narrower one;
- the honest-absence latency body, which is the artifact's refusal path;
- the paid-judge headline, which summed only the kind named `judge` and reported "0" on a run
  that paid for four.

The other four non-hermetic entry points each have a file like this (`test_ingest_script.py`,
`test_retrieval_smoke_script.py`, `test_security_suite_script.py`); the newest and most
claim-dense one had none (code review of #11), and two of the bugs this branch fixed lived here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from finbrief.evaluation import latency
from finbrief.evaluation.cache import Cache, CacheStats

SCRIPT = Path(__file__).parent.parent / "scripts" / "evaluate.py"

#: Where the script writes, relative to the working directory — hence the `chdir` in the one
#: test that reaches `main`.
REPORT = Path("docs/verification/evaluation.md")


def load_script():
    spec = importlib.util.spec_from_file_location("evaluate_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script():
    return load_script()


# --- the sink pre-flight ------------------------------------------------------------------


def test_an_unset_sink_refuses_before_anything_is_spent(script, monkeypatch):
    monkeypatch.delenv("FINBRIEF_LOG_FILE", raising=False)

    with pytest.raises(script.SinkNotEnabled, match="could not be measured"):
        script.require_sink(allow_missing=False)


def test_the_escape_hatch_proceeds_and_returns_no_sink(script, monkeypatch, caplog):
    """A documented hard stop with an undocumented override is a hard stop nobody can trust."""
    monkeypatch.delenv("FINBRIEF_LOG_FILE", raising=False)

    with caplog.at_level("WARNING"):
        assert script.require_sink(allow_missing=True) is None

    assert "records no latency" in caplog.text


def test_a_named_sink_is_returned(script, monkeypatch, tmp_path):
    monkeypatch.setenv("FINBRIEF_LOG_FILE", str(tmp_path / "events.jsonl"))

    assert script.require_sink(allow_missing=False) == tmp_path / "events.jsonl"


# --- the window: which slice of the sink an artifact's numbers come from -------------------


def test_a_measuring_run_marks_the_sink_and_persists_the_mark(script, tmp_path):
    sink = tmp_path / "events.jsonl"
    sink.write_text('{"event": "retrieval"}\n', encoding="utf-8")

    window = script._log_window(
        sink, cache_dir=tmp_path / "cache", stages=("retrieve", "report")
    )

    assert window.offset == sink.stat().st_size
    assert window.replayed is False
    assert latency.load_window(tmp_path / "cache").offset == window.offset


def test_a_report_only_run_appends_nothing_and_replays_the_recorded_mark(script, tmp_path):
    """The case `Window` exists for: a re-render replays every cell, so its own window is empty.

    Without the persisted mark, the latency section refuses on a run whose numbers it is
    otherwise reproducing exactly.
    """
    sink = tmp_path / "events.jsonl"
    sink.write_text('{"event": "retrieval"}\n', encoding="utf-8")
    cache_dir = tmp_path / "cache"
    latency.save_window(
        cache_dir, latency.Window(path=str(sink), offset=7, recorded_at="2026-07-29 12:00 UTC")
    )

    window = script._log_window(sink, cache_dir=cache_dir, stages=("report",))

    assert (window.offset, window.replayed) == (7, True)
    assert window.recorded_at == "2026-07-29 12:00 UTC"


def test_a_report_only_run_with_no_recorded_mark_reports_no_window(script, tmp_path, caplog):
    sink = tmp_path / "events.jsonl"
    sink.write_text("", encoding="utf-8")

    with caplog.at_level("WARNING"):
        window = script._log_window(sink, cache_dir=tmp_path / "cache", stages=("report",))

    assert window is None

    assert "cannot attribute latency" in caplog.text


def test_a_mark_taken_against_another_sink_is_refused_rather_than_applied(
    script, tmp_path, caplog
):
    """An offset only means something against the file it came from.

    `Window.path` was persisted, loaded and never compared, so a rotated or renamed sink had the
    artifact slicing an unrelated stream and captioning it "the measuring run of
    {recorded_at}'s" — the attribution failure the class exists to eliminate (review of #11).
    """
    sink = tmp_path / "events.jsonl"
    sink.write_text("", encoding="utf-8")
    rotated = tmp_path / "events.jsonl.1"
    rotated.write_text("", encoding="utf-8")
    cache_dir = tmp_path / "cache"
    latency.save_window(
        cache_dir, latency.Window(path=str(rotated), offset=99, recorded_at="then")
    )

    with caplog.at_level("WARNING"):
        assert script._log_window(sink, cache_dir=cache_dir, stages=("report",)) is None

    assert "does not describe this file" in caplog.text


def test_only_report_is_a_non_measuring_stage(script):
    """A run of anything else appends lines a statistic can be taken over."""
    assert (
        frozenset({"resolve", "retrieve", "answer", "judge", "agent"})
        == script.MEASURING_STAGES
    )


# --- `--rows` -----------------------------------------------------------------------------


def test_selected_rows_defaults_to_the_whole_set(script):
    from finbrief.evaluation.loader import load_golden_set

    golden = load_golden_set()

    assert script.selected_rows(golden, "") == tuple(golden)
    assert script.selected_rows(golden, "   ") == tuple(golden)


def test_selected_rows_follows_the_sets_order_and_not_the_flags(script):
    from finbrief.evaluation.loader import load_golden_set

    golden = load_golden_set()
    first, second = golden.questions[0].id, golden.questions[1].id

    assert [row.id for row in script.selected_rows(golden, f"{second},{first}")] == [
        first,
        second,
    ]


def test_an_unknown_row_id_raises_rather_than_silently_narrowing_the_run(script):
    from finbrief.evaluation.loader import load_golden_set

    with pytest.raises(KeyError, match="ZZ9"):
        script.selected_rows(load_golden_set(), "ZZ9")


# --- the artifact's refusal paths ---------------------------------------------------------


def test_the_latency_body_refuses_when_the_sink_was_never_enabled(script):
    body = script._latency_body(None, None)

    assert "Not measured, and therefore not met" in body
    assert "no measuring run recorded one" in body


def test_the_latency_body_refuses_when_a_warm_run_appended_nothing(script, tmp_path):
    """A stage served entirely from cache issues no calls, so its window is empty.

    A latency figure in the artifact now means the stage behind it actually ran — and the
    refusal says so, rather than the section vanishing (which reads as a run with no latency).
    """
    sink = tmp_path / "events.jsonl"
    sink.write_text('{"ts": "t", "event": "retrieval", "fields": {}}\n', encoding="utf-8")
    window = latency.Window(path=str(sink), offset=sink.stat().st_size, recorded_at="now")

    body = script._latency_body(sink, window)

    assert "Not measured, and therefore not met" in body
    assert "served from cache" in body


# --- the headline's paid-call arithmetic ---------------------------------------------------


def test_every_judging_kind_counts_toward_the_paid_judge_calls(script, tmp_path):
    """This summed only the kind named `judge` and reported "0" on a run that paid for four.

    The citation pass scores `(sentence, chunk)` pairs through the same scorer under its own
    kind, so a headline over one kind understates spend.
    """
    assert set(script.JUDGING_KINDS) == {"judge", "cited_sentence"}

    cache = Cache(tmp_path / "cache")
    cache._stats.update(
        {
            "judge": CacheStats(hits=10, misses=3),
            "cited_sentence": CacheStats(hits=1, misses=4),
            "retrieval": CacheStats(hits=5, misses=0),
        }
    )

    _, body = script._headline([], cache)

    assert "7" in body, "3 judge misses + 4 cited-sentence misses"


def test_the_bucket_floor_is_adr_0002s_and_not_a_second_opinion_about_it(script):
    """CLAUDE.md names this a single source of truth; `tests/test_golden_set.py` asserts
    `n >= 6` independently, and the two 6s were unbound."""
    from collections import Counter

    from finbrief.evaluation.loader import load_golden_set

    counts = Counter(row.bucket for row in load_golden_set())

    assert script.BUCKET_FLOOR == 6
    assert min(counts.values()) >= script.BUCKET_FLOOR


# --- the committed artifact ----------------------------------------------------------------


def test_the_committed_artifact_is_a_full_run(script):
    """The evidence of record, bound to the one claim a reader takes from it at a glance.

    `security-gate.md` has the same test and for the same reason: the counts are deliberately
    not asserted — they are the last paid run's measurements — but what must **never** be true
    of a committed artifact is that it came from a `--stage`, `--rows` or `--no-ablations` run.
    """
    text = (Path(__file__).parents[1] / REPORT).read_text(encoding="utf-8")

    assert "PARTIAL RUN" not in text, "a partial run was committed as a whole one"
    assert "GENERATED FILE" in text
    assert f"stages run | {', '.join(script.report.ALL_STAGES)}" in text
