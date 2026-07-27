"""The Streamlit layer via AppTest: rendering and the round-trip wiring only.

The agent is stubbed entirely — no LLM calls, no retrieval (spec, testing seam 3).
Session/thread_id behaviour arrives in Phase 3 as `test_app_state.py` (ADR-0008); this file
is the CI smoke test that the page renders what a grounded answer needs — the answer, its
citations' sources, the disclaimer, and the grounding-scope disclosure — and that a message
reaches the agent seam.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from finbrief.agent import agent
from finbrief.ingestion.model import Section
from finbrief.prompts import DISCLAIMER, NO_CONTEXT_FALLBACK
from finbrief.rag import GroundedAnswer
from finbrief.retrieval.retrieve import Context

APP = str(Path(__file__).parents[1] / "app" / "Home.py")


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    return AppTest.from_file(APP, default_timeout=10)


def a_context(rank: int) -> Context:
    return Context(
        chunk_id=f"0001628280-26-003952:Item 1A:{rank}",
        body=f"Tesla's risk factor number {rank}, in the filer's own words.",
        ticker="TSLA",
        filing_type="10-K",
        section=Section.RISK_FACTORS,
        fiscal_year=2025,
        accession="0001628280-26-003952",
        distance=0.1 * rank,
        rank=rank,
    )


def a_grounded_answer(text="Tesla identifies supply-chain concentration [1][2].", contexts=2):
    return GroundedAnswer(
        text=text, contexts=tuple(a_context(rank) for rank in range(1, contexts + 1))
    )


def stub_answer(monkeypatch, answer=None):
    """Replace the agent seam with a fixed grounded answer, recording the questions."""
    asked = []
    result = answer if answer is not None else a_grounded_answer()

    def fake_answer(question):
        asked.append(question)
        return result

    monkeypatch.setattr(agent, "answer", fake_answer)
    return asked


def test_the_page_renders_the_universe_and_a_chat_input(app):
    app.run()

    assert not app.exception
    assert app.title[0].value == "FinBrief"
    sidebar_text = " ".join(md.value for md in app.sidebar.markdown)
    assert "TSLA, F, GM" in sidebar_text  # the autos peer cluster (ADR-0009)
    assert app.chat_input


def test_the_page_states_what_the_answers_are_grounded_in(app):
    # User story 18 and ADR-0007's UI obligation: the scope is declared before a reader can
    # mistake an out-of-scope guess for a grounded answer, not buried in a README.
    app.run()

    page = " ".join(
        element.value for element in [*app.caption, *app.markdown, *app.sidebar.markdown]
    )
    assert "Items 1, 1A, 7 and 7A" in page
    assert "15 companies" in page
    assert "54 of 60" in page


def test_the_page_says_where_the_six_pointer_filers_market_risk_lives(app):
    # The KB's sharpest edge (ADR-0007 amendment §4): those six answer Item 7A by
    # incorporating Item 7, so their market-risk text is in the KB labelled `Item 7`. A
    # reader who does not know that reads "no Item 7A" as "no market-risk grounding".
    app.run()

    scope = " ".join(element.value for element in app.sidebar.markdown)
    for ticker in ("BAC", "GS", "JNJ", "JPM", "LLY", "PFE"):
        assert ticker in scope
    assert "Item 7" in scope


def test_an_answer_renders_with_its_sources_and_the_disclaimer(app, monkeypatch):
    stub_answer(monkeypatch)
    app.run()

    app.chat_input[0].set_value("What are Tesla's risk factors?").run()

    assert not app.exception
    assistant = app.chat_message[1]
    assert "Tesla identifies supply-chain concentration [1][2]." in [
        md.value for md in assistant.markdown
    ]
    # The sources panel: one entry per retrieved chunk, each carrying the metadata user
    # story 3 asks for — ticker, Section and fiscal year — so an inline `[n]` can be
    # checked against the primary source.
    (sources,) = assistant.expander
    assert "2" in sources.label, "the panel states how many chunks grounded the answer"
    panel = " ".join(md.value for md in sources.markdown)
    assert panel.count("TSLA 10-K FY2025, Item 1A") == 2
    assert "[1]" in panel and "[2]" in panel
    assert "Tesla's risk factor number 1, in the filer's own words." in panel
    # The disclaimer is rendered by the app, not asked of the model (user story 15).
    assert DISCLAIMER in [caption.value for caption in app.caption]


def test_the_sources_panel_survives_the_next_turn(app, monkeypatch):
    # The transcript is replayed from `st.session_state` on every rerun, so citations that
    # are only rendered on the turn they arrive lose their sources the moment anything else
    # happens — and an uncheckable `[1]` is worse than no citation.
    stub_answer(monkeypatch)
    app.run()

    app.chat_input[0].set_value("What are Tesla's risk factors?").run()
    app.chat_input[0].set_value("And its competition?").run()

    assert not app.exception
    assert [message.name for message in app.chat_message] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert len(app.chat_message[1].expander) == 1
    assert len(app.chat_message[3].expander) == 1


def test_an_ungrounded_answer_renders_no_empty_sources_panel(app, monkeypatch):
    # `retrieve()` found nothing, so `rag` returned the fallback (spec §Tools, tiered error
    # handling). An empty "Sources (0)" panel would suggest the answer was grounded.
    stub_answer(monkeypatch, GroundedAnswer(text=NO_CONTEXT_FALLBACK, contexts=()))
    app.run()

    app.chat_input[0].set_value("What is Nestle's dividend?").run()

    assert not app.exception
    assistant = app.chat_message[1]
    assert NO_CONTEXT_FALLBACK in [md.value for md in assistant.markdown]
    assert not assistant.expander


def test_sending_a_message_reaches_the_agent_seam(app, monkeypatch):
    asked = stub_answer(monkeypatch)
    app.run()

    app.chat_input[0].set_value("What are Tesla's risk factors?").run()

    assert asked == ["What are Tesla's risk factors?"]
    assert [m["role"] for m in app.session_state.messages] == ["user", "assistant"]


def test_a_missing_key_reports_a_clear_error_and_stops(app, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")

    app.run()

    assert not app.exception
    assert "OPENROUTER_API_KEY" in app.error[0].value
    assert ".env.example" in app.error[0].value
    assert not app.chat_input, "the app should stop before offering a chat input"


def test_a_bad_log_level_reports_a_clear_error_and_stops(app, monkeypatch):
    # LOG_LEVEL is configuration too: `resolve_log_level` raises ConfigError, so it must
    # reach the same banner as a missing key rather than a raw traceback.
    monkeypatch.setenv("LOG_LEVEL", "chatty")

    app.run()

    assert not app.exception
    assert "LOG_LEVEL" in app.error[0].value
    assert not app.chat_input, "the app should stop before offering a chat input"


def test_a_failing_model_call_is_reported_not_raised(app, monkeypatch):
    def boom(question):
        raise RuntimeError("upstream refused")

    monkeypatch.setattr(agent, "answer", boom)
    app.run()

    app.chat_input[0].set_value("What are Tesla's risks?").run()

    assert not app.exception
    assert "upstream refused" in app.error[0].value
    # A failed turn must not leave a phantom assistant message in the transcript.
    assert [m["role"] for m in app.session_state.messages] == ["user"]
