"""The judge's two invariants that hold before a single paid call: the switch and the fence."""

from __future__ import annotations

from finbrief.config import Settings
from finbrief.evaluation.judge import EXCLUDED_FROM_HYPOTHESES, _silence_ragas_telemetry


def a_settings(**overrides) -> Settings:
    env = {"OPENROUTER_API_KEY": "test-key", **overrides}
    return Settings.from_env(env)


def test_the_telemetry_switch_is_set_on_the_mapping_it_is_given():
    # The function, not the import side effect: `conftest` sets the same variable for the suite,
    # so asserting on `os.environ` here would pass whether this code ran or not — which is the
    # vacuous-assertion shape CLAUDE.md names. The script path has no conftest, and this is what
    # covers it.
    env: dict[str, str] = {}

    _silence_ragas_telemetry(env)

    assert env == {"RAGAS_DO_NOT_TRACK": "true"}


def test_the_telemetry_switch_overrides_an_exported_false():
    # `setdefault` would leave a developer who exported the variable running with tracking live.
    env = {"RAGAS_DO_NOT_TRACK": "false"}

    _silence_ragas_telemetry(env)

    assert env["RAGAS_DO_NOT_TRACK"] == "true"


def test_response_relevancy_is_fenced_off_from_every_hypothesis():
    # ragas forces temperature 0.3 at n>1 and asks for n=3, so this column moves between runs.
    # The fence is data because the report renders it and the hypothesis table asserts on it.
    assert "answer_relevancy" in EXCLUDED_FROM_HYPOTHESES


def test_the_two_metrics_the_pre_registered_decisions_rest_on_are_not_fenced():
    # ADR-0005's falsification clause and its §4 trigger are both about context precision and
    # context recall. If either were ever added to the fence, both pre-registered decisions
    # would become unanswerable and this ticket's headline claim would have no basis.
    assert not EXCLUDED_FROM_HYPOTHESES & {"context_precision", "context_recall"}


def test_the_judge_is_not_the_answering_model_by_default():
    # ADR-0002 decision 1's anti-circularity, at the other end of the measurement: a judge that
    # is the answering model grades its own output. An equality on both values, not just an
    # inequality between them, so that changing either default has to come here and say so.
    settings = a_settings()

    assert settings.judge_model == "openai/gpt-4.1-mini"
    assert settings.chat_model == "openai/gpt-4o-mini"
    assert settings.judge_model != settings.chat_model


def test_the_judge_model_is_overridable_on_its_own():
    # Its own field, so raising the judge does not raise the answerer or the gate.
    settings = a_settings(FINBRIEF_JUDGE_MODEL="openai/gpt-5-mini")

    assert settings.judge_model == "openai/gpt-5-mini"
    assert settings.chat_model == "openai/gpt-4o-mini"
    assert settings.classifier_model == "openai/gpt-4o-mini"
