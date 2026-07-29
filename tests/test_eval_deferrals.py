"""The four deferred measurements, and the denominators each refuses to fake.

Seam: the log for two of them, the injected validator for the third, plain text for the
fourth. The assertions that matter are about what is *excluded* from a denominator — a turn
with nothing to cite cannot break a citation rule, and counting it as compliant would inflate
the rate by the number of questions the collection could not answer.
"""

from __future__ import annotations

import json
import re

import pytest

from finbrief.evaluation.deferrals import (
    AdviceResidue,
    BracketAdherence,
    Rate,
    advice_residue,
    bracket_adherence,
    cited_sentences,
    divergence,
)
from finbrief.observability.events import read_events


def a_log(tmp_path, *events) -> object:
    path = tmp_path / "events.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"ts": "2026-07-29T10:00:00+00:00", "event": name, "fields": fields})
            for name, fields in events
        )
        + "\n",
        encoding="utf-8",
    )
    return read_events(path)


# --- the rate type -------------------------------------------------------------------


def test_a_rate_over_nothing_is_absent_rather_than_zero():
    # A divergence rate over zero searches is not 0% divergence.
    empty = Rate(label="x", hits=0, total=0)

    assert empty.rate is None
    assert "not measured" in empty.render()


def test_a_rate_renders_its_denominator():
    assert "(2/5)" in Rate(label="x", hits=2, total=5).render()


# --- 1. divergence -------------------------------------------------------------------


def test_divergence_splits_a_threads_first_search_from_its_later_ones(tmp_path):
    # ADR-0003's amendment asks for the split: a resolved pronoun is the one rewrite the tool's
    # description permits, and it can only occur after the first search — where there is nothing
    # behind it to resolve.
    log = a_log(
        tmp_path,
        ("agent_query", {"thread_id": "t1", "verbatim": True}),
        ("agent_query", {"thread_id": "t1", "verbatim": False}),
        ("agent_query", {"thread_id": "t2", "verbatim": False}),
    )

    result = divergence(log)

    assert result.first_turn.hits == 1 and result.first_turn.total == 2
    assert result.follow_up.hits == 1 and result.follow_up.total == 1
    assert result.overall.hits == 2 and result.overall.total == 3


def test_a_line_without_a_verdict_is_skipped_rather_than_counted_either_way(tmp_path):
    # A `agent_query` line written before the field existed says nothing about divergence, and
    # defaulting it to verbatim would flatter the rate.
    log = a_log(
        tmp_path,
        ("agent_query", {"thread_id": "t1"}),
        ("agent_query", {"thread_id": "t2", "verbatim": False}),
    )

    assert divergence(log).overall.total == 1


def test_divergence_over_an_empty_log_is_absent(tmp_path):
    assert divergence(a_log(tmp_path, ("retrieval", {}))).overall.rate is None


# --- 2. bracket adherence ------------------------------------------------------------


def test_a_turn_with_no_sources_is_excluded_from_the_denominator(tmp_path):
    # An answer with nothing to cite cannot break a citation rule. Counting it as compliant
    # would inflate the rate by the number of questions the collection could not answer.
    log = a_log(
        tmp_path,
        ("citation_markers", {"sources": 0, "resolved": 0, "unresolved": [], "non_numeric": 0}),
        ("citation_markers", {"sources": 5, "resolved": 2, "unresolved": [], "non_numeric": 0}),
    )

    result = bracket_adherence(log)

    assert result.turns_with_sources == 1
    assert result.clean.total == 1
    assert result.clean.rate == pytest.approx(1.0)


def test_a_turn_that_retrieved_and_cited_nothing_is_the_uncited_case(tmp_path):
    # T5's observed failure: a grounded answer with no markers is one nobody can check.
    log = a_log(
        tmp_path,
        ("citation_markers", {"sources": 5, "resolved": 0, "unresolved": [], "non_numeric": 0}),
    )

    result = bracket_adherence(log)

    assert result.uncited == 1
    assert result.clean.rate == pytest.approx(0.0)


def test_non_numeric_and_unresolved_markers_are_counted_apart(tmp_path):
    # Different causes, different fixes: `[Yahoo Finance]` is a prompt-adherence failure,
    # while an unresolvable number is a register regression T7 made structurally impossible.
    log = a_log(
        tmp_path,
        (
            "citation_markers",
            {"sources": 5, "resolved": 2, "unresolved": [9], "non_numeric": 1},
        ),
    )

    result = bracket_adherence(log)

    assert result.unresolved == 1
    assert result.non_numeric == 1
    assert result.uncited == 0


def test_adherence_never_goes_negative_when_one_turn_fails_several_ways(tmp_path):
    log = a_log(
        tmp_path,
        (
            "citation_markers",
            {"sources": 5, "resolved": 0, "unresolved": [9, 10], "non_numeric": 2},
        ),
    )

    assert bracket_adherence(log).clean.hits == 0


def test_adherence_over_no_turns_is_absent():
    assert BracketAdherence(0, 0, 0, 0).clean.rate is None


# --- 3. layer 4's residue ------------------------------------------------------------


class Verdict:
    def __init__(self, refused: bool) -> None:
        self.refused = refused


def test_residue_is_the_share_the_validator_lets_through():
    # Injected validator, because the rate is *about* the live rule set: asserting a particular
    # number here would break whenever a rule was added, which is exactly when it should not.
    residue = advice_residue(
        ["caught", "missed", "missed too"],
        validate=lambda probe: Verdict(refused=probe == "caught"),
    )

    assert residue.residue == ("missed", "missed too")
    assert residue.caught == ("caught",)
    assert residue.rate.rate == pytest.approx(2 / 3)


def test_a_rule_set_that_catches_everything_reports_no_residue():
    residue = advice_residue(["a", "b"], validate=lambda _: Verdict(refused=True))

    assert residue.residue == ()
    assert residue.rate.rate == pytest.approx(0.0)


def test_residue_over_no_probes_is_absent():
    assert AdviceResidue(caught=(), residue=()).rate.rate is None


def test_the_committed_probes_are_advice_shaped_and_carry_no_rating_word():
    # The probe set's whole point: none of them uses the vocabulary the rules look for, so a
    # non-zero residue is a statement about the rules' *generality* rather than about their
    # floor.
    from finbrief.security.corpus import ADVICE_RESIDUE_PROBES

    assert len(ADVICE_RESIDUE_PROBES) >= 5
    # Whole words, not substrings: a *rating* is `hold` standing alone, as in "Rating: HOLD" —
    # `holding` is a verb of ownership and a 10-K uses it constantly, which is exactly why
    # `RESEARCH_ANSWERS` carries the filer's own first person and why this check has to be
    # narrower than a keyword scan.
    shouted = re.compile(
        r"\b(?:buy|sell|hold|rating|overweight|underweight|price target)\b", re.IGNORECASE
    )
    for probe in ADVICE_RESIDUE_PROBES:
        assert not shouted.search(probe), probe


def test_the_residue_probes_are_measurable_without_a_model():
    # Free and deterministic — which is why this deferral needed no live run and could have been
    # closed at any point since T7.
    from finbrief.security.advice import validate_answer
    from finbrief.security.corpus import ADVICE_RESIDUE_PROBES

    measured = advice_residue(ADVICE_RESIDUE_PROBES, validate=validate_answer)

    assert measured.rate.total == len(ADVICE_RESIDUE_PROBES)


# --- 4. cited sentences --------------------------------------------------------------


def test_only_sentences_carrying_a_marker_are_returned():
    answer = "Tesla flags supplier risk [1]. It also mentions tariffs. Margins fell [2][3]."

    found = cited_sentences(answer)

    assert [sentence.ranks for sentence in found] == [(1,), (2, 3)]
    assert "tariffs" not in " ".join(sentence.sentence for sentence in found)


def test_a_non_numeric_marker_is_not_a_citation():
    # `[Yahoo Finance]` is the bracket-rule deferral's subject, not this one's.
    assert cited_sentences("Ford recalled 1.2m cars [Yahoo Finance].") == ()


def test_an_answer_with_no_markers_yields_nothing():
    assert cited_sentences("The filings do not give a current share price.") == ()


class _Retrieval:
    def __init__(self, bodies):
        self.contexts = tuple(
            type("C", (), {"chunk_id": f"c{i}", "body": body})()
            for i, body in enumerate(bodies)
        )


class _Cell:
    def __init__(self, answer, bodies, arm="hybrid+translation"):
        self.arm = arm
        self.answer = answer
        self.retrieval = _Retrieval(bodies)


def test_a_sentence_is_judged_against_the_chunk_it_names_not_the_whole_set():
    # **The point of the whole pass.** Chunk 2 supports the sentence and chunk 1 does not;
    # whole-set faithfulness would score this 1.0 and call the answer grounded. Asked of the
    # *cited* chunk, it is unsupported — faithful and mis-cited at the same time.
    from finbrief.evaluation.deferrals import citation_support

    seen: list[str] = []

    def judge(sentence, body):
        seen.append(body)
        return 1.0 if "tariffs" in body else 0.0

    result = citation_support(
        [
            _Cell(
                "Tesla flags tariff cost increases [1].", ["supplier concentration", "tariffs"]
            )
        ],
        arm="hybrid+translation",
        judge_sentence=judge,
    )

    assert seen == ["supplier concentration"], "it judged against the wrong chunk"
    assert result.unsupported == 1
    assert result.rate.rate == pytest.approx(0.0)


def test_a_supported_citation_counts_towards_the_rate():
    from finbrief.evaluation.deferrals import citation_support

    result = citation_support(
        [_Cell("Tesla flags supplier concentration [1].", ["supplier concentration"])],
        arm="hybrid+translation",
        judge_sentence=lambda sentence, body: 1.0,
    )

    assert result.supported == 1
    assert result.rate.rate == pytest.approx(1.0)


def test_a_marker_outside_the_retrieval_is_unresolvable_and_stays_out_of_the_rate():
    # The bracket-rule deferral's failure, not this one's — structurally prevented since T7's
    # register. Folding it in would blend two questions with different fixes.
    from finbrief.evaluation.deferrals import citation_support

    result = citation_support(
        [_Cell("A claim [9].", ["only one chunk"])],
        arm="hybrid+translation",
        judge_sentence=lambda sentence, body: pytest.fail("should not be judged"),
    )

    assert result.unresolvable == 1
    assert result.rate.rate is None


def test_only_the_named_arm_is_scored():
    from finbrief.evaluation.deferrals import citation_support

    result = citation_support(
        [_Cell("A claim [1].", ["body"], arm="vector")],
        arm="hybrid+translation",
        judge_sentence=lambda sentence, body: 1.0,
    )

    assert result.rate.rate is None


def test_an_unscoreable_sentence_is_skipped_rather_than_counted_either_way():
    # The judge returning `None` means it could not score the pair; counting it as unsupported
    # would blame the pipeline for the judge's failure.
    from finbrief.evaluation.deferrals import citation_support

    result = citation_support(
        [_Cell("A claim [1].", ["body"])],
        arm="hybrid+translation",
        judge_sentence=lambda sentence, body: None,
    )

    assert (result.supported, result.unsupported) == (0, 0)
    assert result.rate.rate is None
