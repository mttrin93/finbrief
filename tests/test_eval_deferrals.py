"""The four deferred measurements, and the denominators each refuses to fake.

Seam: the log for two of them, the injected validator for the third, plain text for the
fourth. The assertions that matter are about what is *excluded* from a denominator — a turn
with nothing to cite cannot break a citation rule, and counting it as compliant would inflate
the rate by the number of questions the collection could not answer.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

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

    # An equality, not a bound: the committed headline is a rate over this exact count, so a
    # probe silently added or dropped changes the denominator the artifact publishes.
    assert len(ADVICE_RESIDUE_PROBES) == 6
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
    """Free and deterministic, and asserted on the **split** rather than on the total.

    `measured.rate.total == len(probes)` was the whole assertion, and it is true by construction
    of `advice_residue` whatever the validator did — the caught/residue split is the measurement
    and it went unchecked (code review of #11). It matters here more than usual because
    `advice_residue` reads the verdict through `getattr(verdict, "refused", False)`: rename
    `AdviceVerdict.refused` and every probe silently lands in the residue, reporting a 100%
    evasion rate that is really a missing attribute. The committed headline is 6/6, so the
    failure would have looked exactly like the result.
    """
    from finbrief.security.advice import validate_answer
    from finbrief.security.corpus import ADVICE_RESIDUE_PROBES

    measured = advice_residue(ADVICE_RESIDUE_PROBES, validate=validate_answer)

    assert measured.rate.total == len(ADVICE_RESIDUE_PROBES)
    # The attribute the rate is read through exists and is a bool on a real verdict — the thing
    # `getattr(..., False)` would swallow.
    verdict = validate_answer(ADVICE_RESIDUE_PROBES[0])
    assert isinstance(verdict.refused, bool)
    # And the split is the validator's, not the helper's: an injected validator that refuses
    # everything must produce an empty residue over the same probes.
    all_refused = advice_residue(
        ADVICE_RESIDUE_PROBES, validate=lambda _: SimpleNamespace(refused=True)
    )
    assert all_refused.residue == ()
    assert all_refused.caught == tuple(ADVICE_RESIDUE_PROBES)
    assert all_refused.rate.hits == 0


def test_a_probe_set_the_validator_cannot_be_read_from_is_not_a_zero_residue():
    # The `getattr` fallback is a fail-*open* on the measurement: a verdict object with no
    # `refused` attribute reads as "not refused" and inflates the residue to 100%. Pinned so the
    # behaviour is a decision rather than an accident — if it should raise instead, this test is
    # where that choice gets made.
    measured = advice_residue(("some advice",), validate=lambda _: object())

    assert measured.residue == ("some advice",)
    assert measured.rate.rate == pytest.approx(1.0), (
        "an unreadable verdict currently reads as unrefused — see the comment"
    )


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


def test_an_unscoreable_pair_is_kept_out_of_the_rate_and_still_counted():
    # The judge returning `None` means it could not score the pair; counting it as unsupported
    # would blame the pipeline for the judge's failure. **But it is counted**: the first version
    # `continue`d past it with no field at all, so a judge failure narrowed the rate's
    # denominator and left nothing in the artifact to say it had — the absence-as-measurement
    # failure `Samples.absent` exists to prevent.
    from finbrief.evaluation.deferrals import citation_support

    result = citation_support(
        [_Cell("A claim [1].", ["body"])],
        arm="hybrid+translation",
        judge_sentence=lambda sentence, body: None,
    )

    assert (result.supported, result.partial, result.unsupported) == (0, 0, 0)
    assert result.unscored == 1
    assert result.rate.rate is None


def test_a_half_supported_sentence_is_not_counted_as_supported():
    """The boundary the first artifact's 70% sat on, as a regression test.

    ragas faithfulness over one sentence is supported-claims / claims, so a two-claim sentence
    with one claim the chunk does not support scores exactly 0.5 — and `verdict >= 0.5` counted
    that as **supported**, on the boundary, documented nowhere. Partial support is its own count
    and sits in the denominator, which is the direction that cannot flatter the rate.
    """
    from finbrief.evaluation.deferrals import citation_support

    result = citation_support(
        [_Cell("Two claims, one of them grounded [1].", ["body"])],
        arm="hybrid+translation",
        judge_sentence=lambda sentence, body: 0.5,
    )

    assert result.supported == 0
    assert result.partial == 1
    assert result.rate.rate == pytest.approx(0.0)
    assert result.rate.total == 1


def test_full_support_is_the_floor_for_supported():
    # An equality against the constant rather than a bound, so a moved floor fails here.
    from finbrief.evaluation.deferrals import CITED_SUPPORT_FLOOR, citation_support

    assert CITED_SUPPORT_FLOOR == 1.0
    result = citation_support(
        [_Cell("A claim [1].", ["body"])],
        arm="hybrid+translation",
        judge_sentence=lambda sentence, body: CITED_SUPPORT_FLOOR,
    )

    assert (result.supported, result.partial) == (1, 0)


def test_the_unit_is_one_sentence_marker_pair_not_one_sentence():
    """The mislabel in the first artifact: "21 cited sentence(s)" was 21 pairs.

    A sentence citing two chunks makes two claims and is judged twice, so the rate's denominator
    counts pairs. Both denominators are carried, because a rate whose unit a reader has to infer
    is a rate they cannot weigh.
    """
    from finbrief.evaluation.deferrals import citation_support

    result = citation_support(
        [_Cell("One sentence naming two chunks [1][2].", ["grounded", "not grounded"])],
        arm="hybrid+translation",
        judge_sentence=lambda sentence, body: 1.0 if body == "grounded" else 0.0,
    )

    assert result.sentences == 1
    assert result.rate.total == 2, "two markers on one sentence are two observations"
    assert (result.supported, result.unsupported) == (1, 1)
    assert result.rate.label == "cited-marker support"
