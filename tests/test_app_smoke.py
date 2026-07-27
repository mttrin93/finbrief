"""The Streamlit layer via AppTest: rendering and the round-trip wiring only.

The agent is stubbed entirely — no LLM calls, no retrieval, no checkpoint file (spec, testing
seam 3). This file is the CI smoke test that the page renders what a grounded answer needs —
the answer, its citations' sources, the disclaimer, and the grounding-scope disclosure — and
that a message reaches the agent seam. `test_app_state.py` owns session and thread_id
behaviour (ADR-0008).
"""

from pathlib import Path

import pytest
from fakes import a_context
from streamlit.testing.v1 import AppTest

from finbrief.agent import agent
from finbrief.agent.agent import AgentTurn, Search
from finbrief.ingestion.model import Section
from finbrief.prompts import DISCLAIMER, NO_CONTEXT_FALLBACK
from finbrief.retrieval.hybrid import Retriever, Surfaced

APP = str(Path(__file__).parents[1] / "app" / "Home.py")


@pytest.fixture(autouse=True)
def never_build_a_real_agent(agent_builds):
    """The app builds its agent to answer a message; here nothing may build a real one."""


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    return AppTest.from_file(APP, default_timeout=10)


def a_tesla_risk_context(rank: int):
    return a_context(
        rank,
        body=f"Tesla's risk factor number {rank}, in the filer's own words.",
        distance=0.1 * rank,
    )


def a_turn(
    text="Tesla identifies supply-chain concentration [1][2].", contexts=2, *, query=None
):
    """A turn that searched once and got `contexts` chunks back."""
    return AgentTurn(
        text=text,
        searches=(
            Search(
                query=query or "What are Tesla's risk factors?",
                contexts=tuple(a_tesla_risk_context(rank) for rank in range(1, contexts + 1)),
            ),
        ),
    )


def a_turn_without_searching(text="Two risks, briefly: […]"):
    """A turn answered from the conversation — no search, so nothing to cite and no banner."""
    return AgentTurn(text=text, searches=())


def stub_answer(monkeypatch, answer=None):
    """Replace the agent seam with a fixed turn, recording the questions."""
    asked = []
    result = answer if answer is not None else a_turn()

    def fake_answer(question, *, thread_id, agent):  # noqa: ARG001 — state is seam 3's other file
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


def sidebar_captions(app) -> str:
    return " ".join(caption.value for caption in app.sidebar.caption)


def sources_panel(assistant):
    """The `Sources (n)` expander of one assistant turn, or `None` if it has none.

    Found by label rather than by position: an answer now renders two expanders — its sources
    and the RAG-viz panel — and a positional lookup would silently start asserting about the
    wrong one the day their order changed.
    """
    panels = [panel for panel in assistant.expander if "Sources" in panel.label]
    return panels[0] if panels else None


def how_i_answered(assistant):
    """The RAG-visualization expander of one assistant turn, or `None` if it has none."""
    panels = [panel for panel in assistant.expander if "How I answered" in panel.label]
    return panels[0] if panels else None


def test_the_configuration_panel_states_the_shipping_default_out_of_the_box(app):
    # `config.DEFAULT_STRATEGY` is the pre-registered `hybrid + translation` (ADR-0005), and as
    # of Phase 4 it is also what answers — so the panel names one value, not two. Until Phase 4
    # this test asserted the opposite (`vector`, plus a caption explaining the gap), because the
    # configured default was a strategy `retrieve()` refused.
    app.run()

    panel = " ".join(md.value for md in app.sidebar.markdown)
    assert "**Strategy** `hybrid + translation`" in panel
    assert "lands in Phase 4" not in sidebar_captions(app), "the gap it described is closed"


def test_the_configuration_panel_follows_the_switches_it_does_not_restate_them(
    app, monkeypatch
):
    # Both switches move independently (ADR-0002's A/B axes), and the panel is the only place a
    # reviewer checks which configuration produced the answers above it. `agent.build_agent`
    # reads these same two settings, which is what makes the panel a report rather than a claim.
    monkeypatch.setenv("FINBRIEF_RETRIEVAL_STRATEGY", "vector")
    monkeypatch.setenv("FINBRIEF_QUERY_TRANSLATION", "false")
    app.run()

    panel = " ".join(md.value for md in app.sidebar.markdown)
    assert "**Strategy** `vector`" in panel
    assert "translation" not in panel


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
    # checked against the primary source. It is the *first* expander; the RAG-viz panel
    # ("How I answered") sits beside it and has its own tests below.
    sources = sources_panel(assistant)
    assert "2" in sources.label, "the panel states how many chunks grounded the answer"
    panel = " ".join(md.value for md in sources.markdown)
    assert panel.count("TSLA 10-K FY2025, Item 1A") == 2
    assert "[1]" in panel and "[2]" in panel
    assert "Tesla's risk factor number 1, in the filer's own words." in [
        text.value for text in sources.text
    ]
    # The disclaimer is rendered by the app, not asked of the model (user story 15).
    assert DISCLAIMER in [caption.value for caption in app.caption]


def a_dollar_bearing_body(recorded_filing) -> str:
    """Apple's real Item 7 around its first dollar figure, whole paragraphs, verbatim.

    Real text, not invented text, because the characters that break rendering are the
    filer's own and no hand-written fixture would think to include them — segment figures
    run together as `Americas$178,353 7 %$167,045`, and the tables sit between blank lines.

    Sliced on paragraph boundaries rather than by character offset so it is shaped like a
    body `retrieve()` actually produces: `_as_context` splits on the same blank line and
    every one of the recorded filing's chunks arrives already stripped, which matters
    because `st.text` strips what it is given. The interior is what this guards.
    """
    paragraphs = recorded_filing.sections[Section.MDA].split("\n\n")
    first_figure = next(index for index, text in enumerate(paragraphs) if "$" in text)
    body = "\n\n".join(paragraphs[first_figure - 2 : first_figure + 3])
    assert body.count("$") > 1, "the slice must carry the figures KaTeX would swallow"
    assert "\n\n" in body, "and a paragraph break, which a blockquote would end at"
    assert body == body.strip(), "and no outer whitespace, as a real body has none"
    return body


def test_a_source_body_renders_the_filers_words_character_identical(
    app, monkeypatch, recorded_filing
):
    # The panel is where an inline `[n]` gets checked against the primary source, so the
    # body has to survive rendering exactly. A Markdown blockquote does not: Streamlit
    # parses `$…$` as KaTeX, which swallows the segment figures a valuation question is
    # asked *about*, and the quote silently ends at the filing's first blank line.
    body = a_dollar_bearing_body(recorded_filing)
    stub_answer(
        monkeypatch,
        AgentTurn(
            text="Apple reports net sales by segment [1].",
            searches=(
                Search(
                    query="How did Apple's segments perform?",
                    contexts=(a_context(1, ticker="AAPL", section=Section.MDA, body=body),),
                ),
            ),
        ),
    )
    app.run()

    app.chat_input[0].set_value("How did Apple's segments perform?").run()

    assert not app.exception
    assert [text.value for text in sources_panel(app.chat_message[1]).text] == [body]


def test_an_answers_dollar_figures_survive_rendering(app, monkeypatch):
    # The same KaTeX hazard as the sources panel, on the surface it costs the most: this is
    # the sentence a reader takes the number from, and the persona is asked to be
    # quantitative where the source is. Unescaped, `$416,161 million from $` is parsed as a
    # maths expression and both figures vanish from the answer.
    figures = "Net sales rose to $416,161 million from $391,035 million [1]."
    stub_answer(monkeypatch, a_turn(text=figures))
    app.run()

    app.chat_input[0].set_value("How did Apple's net sales move?").run()

    assert not app.exception
    rendered = [md.value for md in app.chat_message[1].markdown]
    assert r"Net sales rose to \$416,161 million from \$391,035 million [1]." in rendered
    assert figures not in rendered, "an unescaped `$` is the whole defect"


def test_a_dollar_figure_in_the_question_survives_the_echo(app, monkeypatch):
    # The user's own words are echoed through the same `st.markdown`, so a question about a
    # threshold loses it before the answer is even asked for.
    stub_answer(monkeypatch)
    app.run()

    app.chat_input[0].set_value("Did Tesla's revenue pass $100 billion in $USD?").run()

    assert not app.exception
    assert r"Did Tesla's revenue pass \$100 billion in \$USD?" in [
        md.value for md in app.chat_message[0].markdown
    ]


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
    assert sources_panel(app.chat_message[1]) is not None
    assert sources_panel(app.chat_message[3]) is not None


def test_an_ungrounded_answer_renders_no_empty_sources_panel(app, monkeypatch):
    # The search returned nothing, so the answer is the fallback (spec §Tools, tiered error
    # handling). An empty "Sources (0)" panel would suggest the answer was grounded.
    stub_answer(monkeypatch, a_turn(text=NO_CONTEXT_FALLBACK, contexts=0))
    app.run()

    app.chat_input[0].set_value("What is Nestle's dividend?").run()

    assert not app.exception
    assistant = app.chat_message[1]
    assert NO_CONTEXT_FALLBACK in [md.value for md in assistant.markdown]
    assert not assistant.expander


def test_retrieving_nothing_names_the_un_ingested_collection_as_the_cause(app, monkeypatch):
    # A populated Chroma always returns top-k, so an empty result has one cause: nobody has
    # ingested, or the app is pointed at the wrong directory. The fallback text alone reads
    # as "your question was out of scope" and sends a reviewer who simply has not run ingest
    # looking for a retrieval bug (issue #5 review).
    stub_answer(monkeypatch, a_turn(text=NO_CONTEXT_FALLBACK, contexts=0))
    app.run()

    app.chat_input[0].set_value("What are Tesla's risk factors?").run()

    assert not app.exception
    (warning,) = app.warning
    assert "ingest_filings.py" in warning.value
    assert "data/chroma" in warning.value, "and which collection it looked in"


def test_a_grounded_answer_raises_no_setup_banner(app, monkeypatch):
    # The other half: a banner on every healthy turn is a banner nobody reads.
    stub_answer(monkeypatch)
    app.run()

    app.chat_input[0].set_value("What are Tesla's risk factors?").run()

    assert not app.warning


def test_a_turn_answered_from_the_conversation_shows_no_panel_and_no_banner(app, monkeypatch):
    # The agent decides whether to retrieve, so "summarise that" is answered from the
    # conversation: no sources, and nothing wrong. Treated as an empty retrieval it would fire
    # the un-ingested banner and send a reviewer to re-run a paid ingest because a follow-up
    # worked; given an empty panel it would claim citations the answer does not make.
    stub_answer(monkeypatch, a_turn_without_searching())
    app.run()

    app.chat_input[0].set_value("Summarise that in two lines.").run()

    assert not app.exception
    assistant = app.chat_message[1]
    assert "Two risks, briefly: […]" in [md.value for md in assistant.markdown]
    assert not assistant.expander
    assert not app.warning
    # The disclaimer is still owed — it is not a property of having retrieved something.
    assert DISCLAIMER in [caption.value for caption in assistant.caption]


def test_a_turn_that_did_not_search_replays_without_a_banner(app, monkeypatch):
    # And it has to survive the next rerun as itself: without `searched` in the transcript row
    # the replay cannot tell it from an empty retrieval, so the banner appears one turn late.
    stub_answer(monkeypatch, a_turn_without_searching())
    app.run()

    app.chat_input[0].set_value("Summarise that in two lines.").run()
    app.run()

    assert not app.exception
    assert not app.warning
    assert app.session_state.messages[1]["turn"].searched is False


def test_a_transcript_row_from_an_older_shape_replays(app, monkeypatch):
    # A live session's transcript outlives a code reload, so rows written before `contexts`
    # existed are still in `st.session_state` on the first rerun after a deploy. They must
    # replay without their sources, not `KeyError` the whole page.
    stub_answer(monkeypatch)
    app.run()
    app.session_state.messages = [
        {"role": "user", "content": "What are Tesla's risk factors?"},
        {"role": "assistant", "content": "An answer written before contexts were stored."},
    ]

    app.run()

    assert not app.exception
    assistant = app.chat_message[1]
    assert "An answer written before contexts were stored." in [
        md.value for md in assistant.markdown
    ]
    assert not assistant.expander
    # And no banner: "this row predates contexts" is not "your collection is empty". Sending
    # a reviewer to re-run a paid ingest because an old answer replayed is the worse failure
    # of the two, and it fires on every rerun until the row scrolls out of the transcript.
    assert not app.warning
    assert DISCLAIMER in [caption.value for caption in assistant.caption]


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
    def boom(question, *, thread_id, agent):  # noqa: ARG001
        raise RuntimeError("upstream refused")

    monkeypatch.setattr(agent, "answer", boom)
    app.run()

    app.chat_input[0].set_value("What are Tesla's risks?").run()

    assert not app.exception
    assert "upstream refused" in app.error[0].value
    # A failed turn must not leave a phantom assistant message in the transcript.
    assert [m["role"] for m in app.session_state.messages] == ["user"]


# --------------------------------------------------------------------------------------
# The RAG-visualization panel (user story 5, ADR-0004)
# --------------------------------------------------------------------------------------


def a_translated_turn():
    """A turn whose search translated the question and fused two retrievers over both variants.

    Built by hand rather than retrieved, because seam 3 stubs the agent entirely: what is under
    test here is what the page renders from a turn, not what a turn contains.
    """
    variants = ("What are Tesla's risk factors?", "Tesla supply chain concentration")
    surfaced_by_both = a_context(
        1,
        # The chunk's distance is the nearest of its vector rows, so the one vector row below
        # carries it — the coherence a real `fuse` would produce (`hybrid.fuse`).
        distance=0.91,
        provenance=(
            Surfaced(
                variant=variants[0],
                retriever=Retriever.VECTOR,
                rank=3,
                contribution=1 / 63,
                distance=0.91,
            ),
            Surfaced(
                variant=variants[1], retriever=Retriever.BM25, rank=1, contribution=1 / 61
            ),
        ),
    )
    lexical_only = a_context(
        2,
        distance=None,
        provenance=(
            Surfaced(
                variant=variants[0], retriever=Retriever.BM25, rank=2, contribution=1 / 62
            ),
        ),
    )
    return AgentTurn(
        text="Tesla identifies supply-chain concentration [1][2].",
        searches=(
            Search(
                query=variants[0],
                contexts=(surfaced_by_both, lexical_only),
                variants=variants,
                translated=True,
            ),
        ),
    )


def test_the_panel_shows_the_original_query_and_the_sub_queries_it_added(app, monkeypatch):
    # User story 5, and ADR-0004's invariant made visible: the original is variant 1 and
    # translation only ever *added* to it. A reader who cannot see the original cannot tell a
    # decomposition from a replacement, which is the failure mode the ADR is written against.
    stub_answer(monkeypatch, a_translated_turn())
    app.run()

    app.chat_input[0].set_value("What are Tesla's risk factors?").run()

    panel = how_i_answered(app.chat_message[1])
    text = " ".join(md.value for md in panel.markdown)
    assert "`original`" in text
    assert "What are Tesla's risk factors?" in text
    assert "`sub-query 1`" in text
    assert "Tesla supply chain concentration" in text


def test_the_panel_shows_which_variant_and_retriever_surfaced_each_chunk(app, monkeypatch):
    # The provenance ADR-0004 asks for, on the surface it was asked for: variant × retriever ×
    # rank × RRF contribution, per chunk. This is what makes "*why* hybrid wins" checkable
    # against a single answer rather than only in an aggregate table.
    stub_answer(monkeypatch, a_translated_turn())
    app.run()

    app.chat_input[0].set_value("What are Tesla's risk factors?").run()

    text = " ".join(md.value for md in how_i_answered(app.chat_message[1]).markdown)
    assert "| query | retriever | rank | RRF contribution | distance |" in text
    assert "| original | vector | 3 |" in text
    assert "| sub-query 1 | bm25 | 1 |" in text
    assert f"{1 / 61:.6f}" in text, "the contribution that was summed, not a recomputation"


def test_the_panel_shows_each_rows_own_distance_and_an_em_dash_for_a_bm25_row(app, monkeypatch):
    # ADR-0004 §7's mechanism argument is a *per-variant* distance comparison — the same chunk
    # nearer under one surface form than another — so the row carries its own rather than the
    # chunk's nearest. BM25 has no distance of its own (§3), and a stand-in would print a number
    # no measurement produced.
    stub_answer(monkeypatch, a_translated_turn())
    app.run()

    app.chat_input[0].set_value("What are Tesla's risk factors?").run()

    text = " ".join(md.value for md in how_i_answered(app.chat_message[1]).markdown)
    assert "| original | vector | 3 | 0.015873 | 0.9100 |" in text
    assert "| sub-query 1 | bm25 | 1 | 0.016393 | — |" in text


def a_turn_with_a_variant_that_surfaced_nothing():
    """Three variants, one of which no returned chunk credits — ADR-0004 §1's negative datum.

    The case that justifies `retrieve()` returning a `Retrieval` at all: `sub-query 2` ran
    through both retrievers and nothing it found survived fusion, which is a fact about the
    retrieval that no chunk's provenance can carry.
    """
    variants = (
        "What are Tesla's risk factors?",
        "Tesla supply chain concentration",
        "Tesla executive compensation",
    )
    fused = a_context(
        1,
        provenance=(
            Surfaced(
                variant=variants[0], retriever=Retriever.VECTOR, rank=1, contribution=1 / 61
            ),
            Surfaced(
                variant=variants[1], retriever=Retriever.BM25, rank=2, contribution=1 / 62
            ),
        ),
    )
    return AgentTurn(
        text="Tesla identifies supply-chain concentration [1].",
        searches=(
            Search(query=variants[0], contexts=(fused,), variants=variants, translated=True),
        ),
    )


def test_the_panel_says_which_variant_surfaced_nothing(app, monkeypatch):
    # ADR-0004 amendment §1 makes this datum the entire justification for widening the return
    # shape: "the RAG-viz panel has to show a sub-query that surfaced **nothing**, and that is
    # exactly the datum no chunk's provenance can carry". Listing the variant among the queries
    # put it on screen without reporting it — a reader had to diff that list against every
    # provenance table to notice (issue #6 review).
    stub_answer(monkeypatch, a_turn_with_a_variant_that_surfaced_nothing())
    app.run()

    app.chat_input[0].set_value("What are Tesla's risk factors?").run()

    panel = how_i_answered(app.chat_message[1])
    captions = " ".join(caption.value for caption in panel.caption)
    assert "Surfaced no chunk in the top-1" in captions
    assert "`sub-query 2`" in captions
    # And it does not accuse the variants that did contribute.
    assert "`original`" not in captions
    assert "`sub-query 1`" not in captions


def test_a_reply_with_no_provenance_is_not_reported_as_every_variant_surfacing_nothing(
    app, monkeypatch
):
    # "This query found nothing" and "we cannot say what this query found" are different
    # claims. A reply checkpointed before Phase 4 carries variants with no provenance rows
    # behind them, and reporting all of them as barren would invent a finding.
    turn = AgentTurn(
        text="Tesla identifies supply-chain concentration [1].",
        searches=(
            Search(
                query="What are Tesla's risk factors?",
                contexts=(a_context(1, provenance=()),),
                variants=("What are Tesla's risk factors?", "Tesla supply chain"),
                translated=True,
            ),
        ),
    )
    stub_answer(monkeypatch, turn)
    app.run()

    app.chat_input[0].set_value("What are Tesla's risk factors?").run()

    captions = " ".join(c.value for c in how_i_answered(app.chat_message[1]).caption)
    assert "Surfaced no chunk" not in captions
    assert "Provenance was not recorded for this chunk." in captions


def test_the_panel_says_when_a_chunk_has_no_vector_distance_rather_than_inventing_one(
    app, monkeypatch
):
    # The exact-identifier case: a chunk BM25 recovered is one vector search did not return, so
    # it has no distance. A stand-in would put a number on screen that no measurement produced,
    # and `distance None` reads as a bug.
    stub_answer(monkeypatch, a_translated_turn())
    app.run()

    app.chat_input[0].set_value("What are Tesla's risk factors?").run()

    assistant = app.chat_message[1]
    rendered = " ".join(
        [
            *(md.value for md in how_i_answered(assistant).markdown),
            *(caption.value for caption in sources_panel(assistant).caption),
        ]
    )
    assert "no vector distance" in rendered
    assert "distance None" not in rendered


def test_the_sources_panel_names_the_retrievers_that_found_each_chunk(app, monkeypatch):
    # The one-glance version of the same fact, where a reader already is: beside the citation.
    stub_answer(monkeypatch, a_translated_turn())
    app.run()

    app.chat_input[0].set_value("What are Tesla's risk factors?").run()

    captions = " ".join(c.value for c in sources_panel(app.chat_message[1]).caption)
    assert "vector + BM25" in captions


def test_the_panel_survives_the_next_turn_like_the_sources_do(app, monkeypatch):
    # The transcript is replayed from `session_state` on every rerun, so a panel rendered only
    # on the turn it arrives is one a reader cannot go back to — the same defect the sources
    # panel was fixed for in T3.
    stub_answer(monkeypatch, a_translated_turn())
    app.run()

    app.chat_input[0].set_value("What are Tesla's risk factors?").run()
    app.chat_input[0].set_value("And its competition?").run()

    assert not app.exception
    assert how_i_answered(app.chat_message[1]) is not None
    assert how_i_answered(app.chat_message[3]) is not None


def test_a_turn_that_did_not_search_renders_no_panel(app, monkeypatch):
    # "Summarise that" was answered from the conversation. There is no retrieval to explain, and
    # an empty panel promising an explanation is worse than no panel.
    stub_answer(monkeypatch, a_turn_without_searching())
    app.run()

    app.chat_input[0].set_value("Summarise that in two lines.").run()

    assert how_i_answered(app.chat_message[1]) is None


def test_a_reply_whose_provenance_did_not_survive_renders_no_panel(app, monkeypatch):
    # A checkpoint written before Phase 4 replays with chunks and no provenance. The sources
    # panel still works — the citations are checkable — and the RAG-viz panel has nothing
    # truthful to say, so it says nothing.
    stub_answer(
        monkeypatch,
        AgentTurn(
            text="An answer from before provenance was recorded [1].",
            searches=(
                Search(query="Tesla risk factors", contexts=(a_context(1, provenance=()),)),
            ),
        ),
    )
    app.run()

    app.chat_input[0].set_value("What are Tesla's risk factors?").run()

    assert not app.exception
    assert sources_panel(app.chat_message[1]) is not None
    assert how_i_answered(app.chat_message[1]) is None


def test_the_panel_names_the_ticker_form_apart_from_the_planners_sub_queries(app, monkeypatch):
    # The T6 finding depends on telling the two additions apart (ADR-0004 amendment): one is a
    # deterministic lookup in the Universe, the other is a model's paraphrase. Labelling the
    # ticker form "sub-query 1" would credit the planner for a lookup — and the panel is where a
    # reviewer decides which of the two earned the exact-identifier win.
    variants = ("Tesla debt", "TSLA debt", "Tesla liquidity and capital resources")
    stub_answer(
        monkeypatch,
        AgentTurn(
            text="Tesla reports $8.18 billion of indebtedness [1].",
            searches=(
                Search(
                    query=variants[0],
                    contexts=(
                        a_context(
                            1,
                            provenance=(
                                Surfaced(
                                    variant=variants[1],
                                    retriever=Retriever.BM25,
                                    rank=1,
                                    contribution=1 / 61,
                                ),
                            ),
                        ),
                    ),
                    variants=variants,
                    translated=True,
                ),
            ),
        ),
    )
    app.run()

    app.chat_input[0].set_value("Tesla debt").run()

    assistant = app.chat_message[1]
    text = " ".join(md.value for md in how_i_answered(assistant).markdown)
    assert "`original`" in text and "`ticker form`" in text and "`sub-query 1`" in text
    assert "| ticker form | bm25 | 1 |" in text, "and the table names it too"
    # And the panel says *why* a ticker form exists, since it is the least obvious of the three.
    captions = " ".join(c.value for c in how_i_answered(assistant).caption)
    assert "deterministically" in captions


def a_search(**kwargs) -> Search:
    """A search whose only interesting facts are the translation switches it ran under."""
    return Search(
        query="Tesla debt",
        contexts=(a_context(1),),
        variants=("Tesla debt", "TSLA debt"),
        **kwargs,
    )


def captions_for(app, monkeypatch, search: Search) -> str:
    """The *newest* answer's panel captions, so several searches can be asked of one app.

    Indexed from the end because the transcript accumulates: a second question leaves the first
    answer at `chat_message[1]`, and reading that one would assert against the previous case.
    """
    stub_answer(monkeypatch, AgentTurn(text="Tesla reports debt [1].", searches=(search,)))
    app.run()
    app.chat_input[0].set_value("Tesla debt").run()
    return " ".join(c.value for c in how_i_answered(app.chat_message[-1]).caption)


def test_the_panel_distinguishes_a_planner_that_found_nothing_from_one_that_never_ran(
    app, monkeypatch
):
    # Three states, three captions, because the causes are different and only one of them is the
    # model's. At `FINBRIEF_MAX_SUB_QUERIES=0` translation is *on* — the ticker form is still
    # added — and no chat call is made, so "the question was already one specific thing a filing
    # answers" describes a generation that never happened. That cell is ADR-0004 §6's own
    # falsification channel for the planner's contribution, so the surface a reviewer reads it
    # off is the last place to misreport it (issue #6 review).
    ran = captions_for(app, monkeypatch, a_search(translated=True, planned=True))
    assert "planner added nothing" in ran
    assert "switched off" not in ran

    off = captions_for(app, monkeypatch, a_search(translated=True, planned=False))
    assert "FINBRIEF_MAX_SUB_QUERIES=0" in off
    assert "planner added nothing" not in off

    none = captions_for(app, monkeypatch, a_search(translated=False))
    assert "translation was off" in none.lower()
    assert "planner" not in none


def test_the_panel_reports_the_variants_of_a_search_that_returned_no_chunks(app, monkeypatch):
    # The empty-collection case *under the shipping default*: the queries ran, and every one of
    # them surfaced nothing. The variants are the whole reason `retrieve()` returns a
    # `Retrieval` rather than a bare sequence of contexts, so a search with no chunks is exactly
    # when the panel has something no chunk could carry — and `render_sources` raises its own
    # "re-run ingest" banner beside it.
    stub_answer(
        monkeypatch,
        AgentTurn(
            text=NO_CONTEXT_FALLBACK,
            searches=(
                Search(
                    query="Tesla debt",
                    contexts=(),
                    variants=("Tesla debt", "TSLA debt"),
                    translated=True,
                    planned=True,
                ),
            ),
        ),
    )
    app.run()

    app.chat_input[0].set_value("Tesla debt").run()

    assistant = app.chat_message[1]
    panel = how_i_answered(assistant)
    assert panel is not None, "variants ran, so there is something to explain"
    text = " ".join(md.value for md in panel.markdown)
    assert "`original`" in text and "`ticker form`" in text
    # Not "every variant was barren": with no provenance recorded anywhere, `barren_variants`
    # says nothing at all, and the empty collection gets its own banner instead.
    assert "Surfaced no chunk" not in " ".join(c.value for c in panel.caption)
    assert sources_panel(assistant) is None
