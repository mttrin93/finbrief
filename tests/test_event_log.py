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


def test_the_reader_does_not_reformat_what_it_read(sink):
    """Whatever the emitter serialised is what the reader hands back, verbatim."""
    logger, path = sink
    log_event(logger, "retrieval", distances=[1.0404, None])

    (event,) = read_events(path).events
    (raw,) = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert event.fields == raw["fields"]
