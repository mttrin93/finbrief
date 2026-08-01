"""Reading the JSON-lines events back — the half T10 (#11) consumes.

The round-trip tests are the point of this file. `log_event` is the only emitter and
`read_events` is the only reader, so the two drifting apart is a whole class of silent
failure: a renamed field reads back as an absence, and an absence read as a zero is a
measurement reported about nothing.
"""

import io
import json
import logging

import pytest

from finbrief.observability.events import Event, read_events
from finbrief.observability.logging_setup import configure_logging, log_event, turn


@pytest.fixture
def sink(tmp_path):
    """A configured logger writing to a real file, and that file's path."""
    path = tmp_path / "events.jsonl"
    logger = configure_logging(logging.DEBUG, stream=io.StringIO(), path=path)
    return logger, path


# --- The round trip: emitter and reader cannot drift --------------------------------------


def test_an_event_survives_the_round_trip_intact(sink):
    logger, path = sink
    log_event(
        logger,
        "retrieval",
        strategy="hybrid",
        translation=True,
        k=5,
        latency_ms=812,
        per_list_hits=[5, 5, 0],
        provenance=[{"chunk_id": "TSLA-1A-0", "score": 0.0164}],
    )

    log = read_events(path)
    (event,) = log.events
    assert event.event == "retrieval"
    assert event.level == "INFO"
    assert event.logger == "finbrief"
    assert event.ts.tzinfo is not None, "the envelope's timestamp is UTC-aware"
    # The whole field set, not a sampled key: a field the emitter adds and the reader drops
    # is exactly the drift this test exists to forbid.
    assert event.fields == {
        "strategy": "hybrid",
        "translation": True,
        "k": 5,
        "latency_ms": 812,
        "per_list_hits": [5, 5, 0],
        "provenance": [{"chunk_id": "TSLA-1A-0", "score": 0.0164}],
    }


def test_an_absent_field_stays_absent_across_the_round_trip(sink):
    logger, path = sink
    # The shape a real log has: `rag_answer` carries token counts when the provider reported
    # usage and carries no such key when it did not (`observability/tokens.py`).
    log_event(logger, "rag_answer", contexts=5, input_tokens=1_204, output_tokens=317)
    log_event(logger, "rag_answer", contexts=5)

    log = read_events(path)
    with_usage, without_usage = log.of("rag_answer")
    assert with_usage.field("input_tokens") == 1_204
    assert "input_tokens" not in without_usage.fields
    assert without_usage.field("input_tokens") is None
    # Not `0`. A run whose provider reported no usage spent tokens nobody counted; reading
    # that back as zero would put a fabricated number in a cost table.
    # A default is available, but it is the *caller's* to choose and never this module's.
    assert without_usage.field("input_tokens", default=0) == 0


def test_the_denominator_of_a_sample_is_reported_not_silently_narrowed(sink):
    logger, path = sink
    log_event(logger, "rag_answer", input_tokens=1_000)
    log_event(logger, "rag_answer")
    log_event(logger, "rag_answer", input_tokens=1_400)

    samples = read_events(path).samples("rag_answer", "input_tokens")
    assert samples.present == (1_000, 1_400)
    # The one line that carried no usage is *counted*, not dropped: a mean over 2 of 3 lines
    # reported as a mean over the run is the "no silent caps" failure (CLAUDE.md).
    assert samples.absent == 1
    assert samples.total == 3


def test_the_turn_id_survives_the_round_trip_and_groups_a_questions_lines(sink):
    # The join, through both halves: `logging_setup.turn` writes it into the envelope and
    # `read_events` has to read it off the envelope, not out of `fields`. Emitter and reader
    # were written in the same commit, which is exactly when a key can be spelled two ways.
    logger, path = sink
    with turn("golden-multi-hop-02:hybrid+translation"):
        log_event(logger, "query_translation", sub_queries=3)
        log_event(logger, "retrieval", hits=5)
    log_event(logger, "retrieval", hits=5)  # a line from outside any turn

    log = read_events(path)
    grouped = log.by_turn()
    assert set(grouped) == {"golden-multi-hop-02:hybrid+translation", None}
    assert [event.event for event in grouped["golden-multi-hop-02:hybrid+translation"]] == [
        "query_translation",
        "retrieval",
    ]
    # And it is *not* smuggled into the payload, where a field of the same name could shadow it.
    assert all("turn_id" not in event.fields for event in log.events)


def test_a_line_from_before_the_turn_id_existed_reads_as_no_turn(tmp_path):
    # The checkpoint-compatibility rule again: the sink is append-only, so a file can hold
    # lines written by a deploy that had no `turn_id` at all. Absence, not a `KeyError`.
    path = tmp_path / "events.jsonl"
    path.write_text(
        '{"ts": "2026-07-29T10:00:00.000+00:00", "level": "INFO", "logger": "finbrief",'
        ' "event": "retrieval", "fields": {"hits": 5}}\n',
        encoding="utf-8",
    )

    (event,) = read_events(path).events
    assert event.turn_id is None


def test_an_event_with_no_fields_at_all_round_trips(sink):
    logger, path = sink
    log_event(logger, "gate_classifier_unparsed")

    (event,) = read_events(path).events
    # The formatter omits `fields` entirely when it is empty, so this is the one envelope
    # shape the reader sees without the key at all.
    assert event.fields == {}


# --- Malformed input, and the difference between empty and unknown ------------------------


def test_a_truncated_last_line_is_skipped_and_counted(tmp_path):
    # What a process killed mid-write leaves behind. A reader that raises here loses a whole
    # run's samples to its final half-line; one that skips silently loses them invisibly.
    path = tmp_path / "events.jsonl"
    path.write_text(
        '{"ts": "2026-07-29T10:00:00.000+00:00", "level": "INFO", "logger": "finbrief",'
        ' "event": "retrieval", "fields": {"k": 5}}\n'
        '{"ts": "2026-07-29T10:00:01.000+00:00", "level": "INFO", "logger": "finbr',
        encoding="utf-8",
    )

    log = read_events(path)
    assert [event.event for event in log.events] == ["retrieval"]
    assert log.malformed == 1


@pytest.mark.parametrize(
    "line",
    [
        "not json at all",
        "[1, 2, 3]",  # valid JSON, not an object
        "null",
        '{"ts": "2026-07-29T10:00:00.000+00:00", "level": "INFO"}',  # no `event` key
        '{"event": "retrieval", "ts": "half past ten", "level": "INFO", "logger": "f"}',
    ],
)
def test_a_line_that_is_not_an_event_is_counted_rather_than_read(tmp_path, line):
    path = tmp_path / "events.jsonl"
    path.write_text(f"{line}\n", encoding="utf-8")

    log = read_events(path)
    assert log.events == ()
    assert log.malformed == 1


def test_blank_lines_are_not_malformed(tmp_path):
    # A file that ends in a newline is the normal case, not a damaged one.
    path = tmp_path / "events.jsonl"
    path.write_text(
        '{"ts": "2026-07-29T10:00:00.000+00:00", "level": "INFO", "logger": "finbrief",'
        ' "event": "retrieval", "fields": {}}\n\n',
        encoding="utf-8",
    )

    log = read_events(path)
    assert len(log.events) == 1
    assert log.malformed == 0


def test_a_missing_sink_is_an_error_not_an_empty_log(tmp_path):
    # The distinction the app's `barren_variants` makes, applied to the log: "the sink was
    # never enabled" is not "this run emitted nothing". Reading a missing file as empty would
    # let T10 report a p50 over zero samples as a budget met.
    with pytest.raises(FileNotFoundError):
        read_events(tmp_path / "never-enabled.jsonl")


def test_an_empty_sink_reads_as_an_empty_log(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text("", encoding="utf-8")

    log = read_events(path)
    assert log.events == ()
    assert log.samples("retrieval", "latency_ms").total == 0


# --- Selectors ----------------------------------------------------------------------------


def test_events_can_be_selected_by_name(sink):
    logger, path = sink
    log_event(logger, "retrieval", latency_ms=100)
    log_event(logger, "rag_answer", latency_ms=900)
    log_event(logger, "retrieval", latency_ms=120)

    log = read_events(path)
    assert [e.field("latency_ms") for e in log.of("retrieval")] == [100, 120]
    assert len(log.of("retrieval", "rag_answer")) == 3
    assert log.of("no_such_event") == ()


def test_the_reader_yields_samples_and_leaves_the_statistics_to_its_caller(sink):
    # Deliberately no `p50` here: ADR-0005's ≤1.5s budget and ADR-0006's ≤1s budget are
    # T10's arithmetic over these samples, and `security/report.py` already owns its own
    # median over in-process screenings. A third median in this module would be a third
    # place the definition could drift.
    logger, path = sink
    for latency in (100, 900, 300):
        log_event(logger, "retrieval", latency_ms=latency)

    assert read_events(path).samples("retrieval", "latency_ms").present == (100, 900, 300)


def test_an_event_read_from_a_line_written_before_a_field_existed(tmp_path):
    # The checkpoint-compatibility rule, applied to logs: a file written by yesterday's deploy
    # is read by today's code, and a field added since must be an absence rather than a
    # `KeyError` on the first line.
    path = tmp_path / "events.jsonl"
    path.write_text(
        '{"ts": "2026-07-29T10:00:00.000+00:00", "level": "INFO", "logger": "finbrief",'
        ' "event": "retrieval"}\n',
        encoding="utf-8",
    )

    (event,) = read_events(path).events
    assert isinstance(event, Event)
    assert event.fields == {}
    assert event.field("provenance", default=()) == ()


def test_the_model_that_answered_survives_the_round_trip_in_both_fields(sink):
    """T14 (#15) — the picker's record, through the one emitter and the one reader.

    A model slug is configuration and not user content, so it needs no exception to the
    no-user-content rule; what it does need is the same round trip every other field gets,
    because a field the emitter writes and the reader drops reads back as `None` and `None`
    attributes a turn's tokens to nobody.

    Both fields, because they can differ: `model` is what was requested, `model_reported` what
    the reply said. A test asserting only the first would pass on an implementation that never
    wrote the second.
    """
    logger, path = sink
    log_event(
        logger,
        "agent_turn",
        model="anthropic/claude-3.5-haiku",
        model_reported="anthropic/claude-3-5-haiku-20241022",
        input_tokens=1_204,
        output_tokens=317,
    )

    (event,) = read_events(path).of("agent_turn")
    assert event.field("model") == "anthropic/claude-3.5-haiku"
    assert event.field("model_reported") == "anthropic/claude-3-5-haiku-20241022"


def test_a_turn_logged_before_the_model_field_existed_reads_as_unattributed(tmp_path):
    # Every line already in a developer's sink was written before T14, and the analytics page
    # reads the whole file. An older turn is *unattributed* — not a turn on the default model,
    # which is a claim nothing recorded.
    path = tmp_path / "events.jsonl"
    path.write_text(
        '{"ts": "2026-07-29T10:00:00.000+00:00", "level": "INFO", "logger": "finbrief",'
        ' "event": "agent_turn", "fields": {"searches": 1, "input_tokens": 900}}\n',
        encoding="utf-8",
    )

    (event,) = read_events(path).of("agent_turn")
    assert event.field("model") is None
    assert "model" not in event.fields
    assert event.field("input_tokens") == 900, "and the rest of the line still reads"


def test_a_turn_that_named_no_model_records_the_absence_rather_than_omitting_it(sink):
    # `answer()` writes `model=None` when no caller named one, which the emitter serialises as a
    # JSON `null`. Distinguishable from the line above only by `"model" in fields` — and both
    # read back as `None`, which is the point: the two absences mean the same thing to a reader
    # and neither may become a default.
    logger, path = sink
    log_event(logger, "agent_turn", model=None, model_reported=None, input_tokens=900)

    (event,) = read_events(path).of("agent_turn")
    assert event.field("model") is None
    assert event.fields["model"] is None


def test_the_reader_does_not_reformat_what_it_read(sink):
    """Whatever the emitter serialised is what the reader hands back, verbatim."""
    logger, path = sink
    log_event(logger, "retrieval", distances=[1.0404, None])

    (event,) = read_events(path).events
    (raw,) = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert event.fields == raw["fields"]


# --- run-scoping: a statistic over a shared sink is a statistic over every run ---------


def test_reading_from_a_mark_returns_only_what_was_appended_after_it(tmp_path):
    """The defect ADR-0011's T10 amendment records, as a regression test.

    The sink is append-only across runs, so a median over the whole file is a median over every
    run that ever shared it — which is how the first committed evaluation artifact reported a
    planner p50 from a pool holding 13 appended runs, the pre-fix ones included. `sink_offset`
    marks the file before a run and `read_events` reads forward from the mark.
    """
    from finbrief.observability.events import read_events, sink_offset

    path = tmp_path / "events.jsonl"
    path.write_text(
        '{"event": "retrieval", "ts": "2026-07-29T10:00:00+00:00",'
        ' "fields": {"latency_ms": 9}}\n',
        encoding="utf-8",
    )

    mark = sink_offset(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            '{"event": "retrieval", "ts": "2026-07-29T11:00:00+00:00",'
            ' "fields": {"latency_ms": 1}}\n'
        )

    assert len(read_events(path).of("retrieval")) == 2, "the whole file still reads whole"
    scoped = read_events(path, start_offset=mark).of("retrieval")
    assert [event.field("latency_ms") for event in scoped] == [1]


def test_a_mark_on_a_sink_that_does_not_exist_yet_is_zero(tmp_path):
    # `configure_logging()` may not have created the file, and a run that then writes it must
    # read all of its own lines rather than none.
    from finbrief.observability.events import sink_offset

    assert sink_offset(tmp_path / "absent.jsonl") == 0


def test_a_fully_replayed_stage_leaves_an_empty_window_and_the_p50_refuses(tmp_path):
    """The interaction between run-scoping and the cache, asserted rather than assumed.

    A warm run issues no calls, so it appends no lines, so its window is empty — and the honest
    outcome is a refusal, not the previous run's median served as this run's. This is what makes
    a latency figure in the artifact mean the stage behind it actually ran.
    """
    from finbrief.evaluation.latency import NoSamples, load_log, translation_cost
    from finbrief.observability.events import sink_offset

    path = tmp_path / "events.jsonl"
    path.write_text(
        '{"event": "query_translation", "ts": "2026-07-29T10:00:00+00:00",'
        ' "fields": {"latency_ms": 1518, "input_tokens": 271}}\n'
        '{"event": "retrieval", "ts": "2026-07-29T10:00:01+00:00",'
        ' "fields": {"latency_ms": 1576, "translation": true, "variants": 5}}\n'
        '{"event": "retrieval", "ts": "2026-07-29T10:00:02+00:00",'
        ' "fields": {"latency_ms": 328, "translation": false, "variants": 1}}\n',
        encoding="utf-8",
    )

    # Unscoped, the previous run's lines are a complete measurement.
    assert translation_cost(load_log(path)).planner_p50_ms == 1518

    warm = sink_offset(path)
    with pytest.raises(NoSamples):
        translation_cost(load_log(path, start_offset=warm))
