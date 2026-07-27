"""RRF fusion and the BM25 index — the two halves hybrid search is made of (ADR-0004).

Below seam 1, deliberately. `test_retrieve.py` owns what a *retrieval* means and asserts the
composition through `retrieve()`; what is asserted here is the arithmetic and the lexical
matching that composition rests on, because a fusion bug and a BM25 bug are
indistinguishable when you can only see the top-k that came out.

The BM25 tests are the closest thing the hermetic suite has to ADR-0004's central claim: a
chunk that contains a literal identifier beats one that merely discusses the same topic. The
paid embedding model is what makes the claim interesting, and the fake one cannot stand in for
it — so these tests pin the *lexical* half on its own, and the live before/after on issue #6 is
what tests the claim.
"""

from __future__ import annotations

from langchain_core.documents import Document

from finbrief.config import RRF_K
from finbrief.retrieval.hybrid import (
    BM25Index,
    CandidateList,
    Retriever,
    Surfaced,
    bm25_index,
    fuse,
    query_terms,
    tokenize,
)

ORIGINAL = "How much debt does Tesla carry?"
SUB_QUERY = "Tesla total indebtedness"


def a_document(chunk_id: str, text: str = "filing text") -> Document:
    return Document(id=chunk_id, page_content=text, metadata={"ticker": "TSLA"})


def a_vector_list(*hits: tuple[str, float], variant: str = ORIGINAL) -> CandidateList:
    """A vector candidate list, nearest first — each hit a chunk id and its distance."""
    return CandidateList(
        variant=variant,
        retriever=Retriever.VECTOR,
        hits=tuple((a_document(chunk_id), distance) for chunk_id, distance in hits),
    )


def a_bm25_list(*chunk_ids: str, variant: str = ORIGINAL) -> CandidateList:
    """A BM25 candidate list, best first. Lexical scores are deliberately not carried."""
    return CandidateList(
        variant=variant,
        retriever=Retriever.BM25,
        hits=tuple((a_document(chunk_id), None) for chunk_id in chunk_ids),
    )


def ids(fused) -> list[str]:
    return [item.document.id for item in fused]


# --------------------------------------------------------------------------------------
# Reciprocal Rank Fusion
# --------------------------------------------------------------------------------------


def test_one_candidate_list_fuses_to_its_own_order():
    # The invariant that lets `vector` without translation keep returning exactly what T3
    # returned: RRF over a single list is strictly decreasing in rank, so fusion is a no-op
    # on order. Without this, adding hybrid search silently moves the baseline the A/B
    # compares against.
    fused = fuse([a_vector_list(("c-1", 0.1), ("c-2", 0.2), ("c-3", 0.3))], limit=3)

    assert ids(fused) == ["c-1", "c-2", "c-3"]
    assert [item.distance for item in fused] == [0.1, 0.2, 0.3]


def test_fusion_truncates_to_the_limit_after_fusing_not_before():
    # Truncating each list first would throw away the agreement that RRF exists to find.
    fused = fuse(
        [
            a_vector_list(("c-1", 0.1), ("c-2", 0.2), ("c-3", 0.3)),
            a_bm25_list("c-3", "c-2", "c-1"),
        ],
        limit=2,
    )

    assert len(fused) == 2


def test_a_chunk_two_retrievers_agree_on_outranks_one_only_a_single_list_found():
    # ADR-0004's whole mechanism, in the smallest case that shows it. `c-9` is nobody's best
    # hit; it is the only chunk *both* retrievers returned, and that is what RRF rewards.
    fused = fuse(
        [
            a_vector_list(("c-1", 0.1), ("c-2", 0.2), ("c-9", 0.9)),
            a_bm25_list("c-8", "c-7", "c-9"),
        ],
        limit=4,
    )

    assert ids(fused)[0] == "c-9"
    assert fused[0].score == 2 / (RRF_K + 3)


def test_a_chunk_is_deduplicated_but_every_list_that_surfaced_it_is_recorded():
    # Dedup by chunk id (ADR-0004), and the provenance is what survives it: the panel and the
    # A/B analysis need "which variant × which retriever", so a dedup that dropped the losing
    # row would leave "why hybrid wins" unanswerable.
    fused = fuse(
        [
            a_vector_list(("c-1", 0.4)),
            a_bm25_list("c-1"),
            a_vector_list(("c-1", 0.2), variant=SUB_QUERY),
        ],
        limit=5,
    )

    (item,) = fused
    assert item.provenance == (
        Surfaced(
            variant=ORIGINAL,
            retriever=Retriever.VECTOR,
            rank=1,
            contribution=1 / 61,
            distance=0.4,
        ),
        Surfaced(
            variant=ORIGINAL,
            retriever=Retriever.BM25,
            rank=1,
            contribution=1 / 61,
            distance=None,
        ),
        Surfaced(
            variant=SUB_QUERY,
            retriever=Retriever.VECTOR,
            rank=1,
            contribution=1 / 61,
            distance=0.2,
        ),
    )
    assert item.score == 3 / 61


def test_a_contribution_is_the_published_reciprocal_rank_and_they_sum_to_the_score():
    fused = fuse([a_vector_list(("c-1", 0.1), ("c-2", 0.2))], limit=2)

    assert [item.score for item in fused] == [1 / (RRF_K + 1), 1 / (RRF_K + 2)]
    assert all(item.score == sum(row.contribution for row in item.provenance) for item in fused)


def test_ties_are_broken_by_chunk_id_so_a_retrieval_is_reproducible():
    # ADR-0003 stakes the A/B on determinism, and two chunks at the same rank in different
    # lists tie exactly. Dict insertion order would make the winner depend on which retriever
    # happened to be iterated first — reproducible within a process and not across a refactor.
    fused = fuse([a_vector_list(("c-b", 0.1)), a_bm25_list("c-a")], limit=2)

    assert ids(fused) == ["c-a", "c-b"]
    assert fused[0].score == fused[1].score


def test_a_chunk_only_bm25_found_has_no_vector_distance_rather_than_a_made_up_one():
    # The exact-identifier case ADR-0004 is built on: a chunk BM25 pulls up on a literal
    # token was, by construction, not in the vector list. There is no distance for it, and
    # inventing one (0.0, or `inf`) would put a number in the sources panel that no
    # measurement produced.
    fused = fuse([a_vector_list(("c-1", 0.1)), a_bm25_list("c-2")], limit=2)

    (vector_only,) = [item for item in fused if item.document.id == "c-1"]
    (bm25_only,) = [item for item in fused if item.document.id == "c-2"]
    assert vector_only.distance == 0.1
    assert bm25_only.distance is None


def test_a_chunk_several_variants_reached_keeps_the_nearest_distance():
    # One chunk, two vector lists, two distances. The nearest is the honest one to show: it is
    # the best evidence any query variant produced that this chunk is relevant.
    fused = fuse(
        [
            a_vector_list(("c-1", 0.9)),
            a_vector_list(("c-1", 0.3), variant=SUB_QUERY),
        ],
        limit=1,
    )

    assert fused[0].distance == 0.3


def test_each_provenance_row_keeps_its_own_variants_distance_not_the_chunks_nearest():
    # ADR-0004 §7's pre-registration is a *per-variant* distance comparison — "the ticker form
    # ranks the chunk 5th under vector search too, at distance 0.6778 against the original's
    # 1.0406" — and that is the number the prediction has to be refuted with. Folding the rows
    # to `Fused.distance`'s minimum destroys it, which is why §6's own table had to come from a
    # scratchpad script re-running each variant separately (issue #6 review).
    fused = fuse(
        [
            a_vector_list(("c-1", 1.0406)),
            a_vector_list(("c-1", 0.6778), variant=SUB_QUERY),
            a_bm25_list("c-1", variant=SUB_QUERY),
        ],
        limit=1,
    )

    (item,) = fused
    assert [(row.variant, row.distance) for row in item.provenance] == [
        (ORIGINAL, 1.0406),
        (SUB_QUERY, 0.6778),
        (SUB_QUERY, None),
    ]
    # And the chunk-level fold is unchanged, so both readings are available at once.
    assert item.distance == 0.6778


def test_a_provenance_row_survives_a_payload_round_trip_with_its_distance():
    row = Surfaced(
        variant=SUB_QUERY,
        retriever=Retriever.VECTOR,
        rank=4,
        contribution=1 / 64,
        distance=0.6778,
    )

    assert Surfaced.from_payload(row.as_payload()) == row


def test_a_provenance_row_with_no_distance_recorded_reads_back_as_none():
    # The repo-wide rule for a checkpointed payload: a reader tolerates a shape written before
    # the current one, with an honest absence value and never an invented number. `Surfaced` has
    # no deployed older shape of its own (the class arrives whole in Phase 4) — this pins the
    # absence value, which is also what every BM25 row carries.
    older = {"variant": ORIGINAL, "retriever": "vector", "rank": 2, "contribution": 1 / 62}

    assert Surfaced.from_payload(older) == Surfaced(
        variant=ORIGINAL, retriever=Retriever.VECTOR, rank=2, contribution=1 / 62, distance=None
    )


def test_fusing_nothing_returns_nothing():
    assert fuse([], limit=5) == ()
    assert fuse([a_vector_list()], limit=5) == ()


# --------------------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------------------

TSLA_CHUNK = Document(
    id="tsla-7-3",
    page_content=(
        "TSLA | FY2025 10-K | Item 7. Management's Discussion and Analysis\n\n"
        "Tesla's total debt, excluding vehicle and energy product financing, was "
        "$5.7 billion as of December 31, 2025."
    ),
    metadata={"ticker": "TSLA", "section": "Item 7"},
)

#: Four peer chunks that discuss indebtedness more, and in more words, than the one chunk that
#: names Tesla — the shape issue #6 measured. Deliberately worded differently from each other:
#: an identical corpus makes every IDF degenerate, and a test that only holds on a degenerate
#: corpus is not evidence about the real one.
FORD_CHUNKS = (
    Document(
        id="f-7a-0",
        page_content=(
            "F | FY2025 10-K | Item 7A. Quantitative and Qualitative Disclosures\n\n"
            "Our debt obligations expose us to interest rate risk. Total debt outstanding "
            "and the maturity profile of that debt are managed centrally."
        ),
        metadata={"ticker": "F", "section": "Item 7A"},
    ),
    Document(
        id="f-7-1",
        page_content=(
            "F | FY2025 10-K | Item 7. Management's Discussion and Analysis\n\n"
            "Ford Credit debt was $119 billion at year end. Company excluding Ford Credit "
            "debt was $19 billion, and consolidated debt therefore rose."
        ),
        metadata={"ticker": "F", "section": "Item 7"},
    ),
    Document(
        id="f-7-2",
        page_content=(
            "F | FY2025 10-K | Item 7. Management's Discussion and Analysis\n\n"
            "Our indebtedness could adversely affect liquidity. Servicing that debt "
            "consumes cash flow that would otherwise fund capital spending."
        ),
        metadata={"ticker": "F", "section": "Item 7"},
    ),
    Document(
        id="f-7-3",
        page_content=(
            "F | FY2025 10-K | Item 7. Management's Discussion and Analysis\n\n"
            "Unsecured debt issuance and securitization transactions funded the finance "
            "receivables portfolio; committed credit lines remained undrawn."
        ),
        metadata={"ticker": "F", "section": "Item 7"},
    ),
)


def test_bm25_ranks_the_chunk_naming_the_company_above_four_peers_discussing_debt():
    # Issue #6's failure, transposed into the lexical half. Four Ford chunks say "debt" more
    # often than the one Tesla chunk does, and vector search prefers them for exactly that
    # reason. BM25 does not, because `tesla` appears in one chunk out of five and its IDF says
    # so — which is the mechanism ADR-0004 predicts moves the Tesla chunk up.
    index = BM25Index.over([*FORD_CHUNKS, TSLA_CHUNK])

    hits = index.nearest("Tesla debt", 5)

    assert hits[0].id == "tsla-7-3"


def test_bm25_matches_a_section_literal_the_provenance_header_carries():
    # The chunk-side counterpart in ADR-0004: the header is indexed so "Item 7" matches every
    # chunk of a Section, not only the one the splitter left the heading in.
    #
    # Asserted on `f-7-3`, whose *body* holds neither "item" nor any digit, so the only way it
    # can be a hit is through its header. A bare `assert index.nearest(...)` passed with every
    # header stripped, because `$5.7 billion` tokenizes to a `7` — the test was true of a corpus
    # that did not index the header at all (issue #6 review).
    index = BM25Index.over([*FORD_CHUNKS, TSLA_CHUNK])

    assert "f-7-3" in {hit.id for hit in index.nearest("Item 7", 5)}


def test_bm25_returns_at_most_k_hits_best_first_and_the_same_order_every_time():
    index = BM25Index.over([*FORD_CHUNKS, TSLA_CHUNK])

    first = index.nearest("Tesla debt", 3)
    second = index.nearest("Tesla debt", 3)

    assert len(first) == 3
    assert [hit.id for hit in first] == [hit.id for hit in second]


def test_bm25_over_an_empty_collection_returns_nothing_rather_than_raising():
    # `rank_bm25` divides by the corpus's average document length, so an empty corpus is a
    # `ZeroDivisionError` waiting for the first query against an un-ingested collection.
    index = BM25Index.over([])

    assert index.nearest("anything", 5) == ()


def test_bm25_returns_nothing_for_a_query_sharing_no_term_with_the_collection():
    # A hit list padded out to `k` with chunks that matched nothing is worse than a short one:
    # RRF would give each of them a rank, and fuse them as if a retriever had voted for them.
    index = BM25Index.over([TSLA_CHUNK])

    assert index.nearest("cryptocurrency mining hashrate", 5) == ()


def test_a_chunk_holding_only_a_common_query_term_is_still_a_hit():
    # Why membership is a term-set intersection and not `score > 0`. BM25 Okapi floors the IDF
    # of a term held by most of the corpus, so a chunk matching only `debt` can score at or
    # below zero while genuinely containing it — and those four chunks are the comparison issue
    # #6 is measured against. Thresholding the score would have made the case unmeasurable.
    index = BM25Index.over([*FORD_CHUNKS, TSLA_CHUNK])

    hits = index.nearest("Tesla debt", 5)

    assert len(hits) == 5
    assert hits[0].id == "tsla-7-3", "the rare term still decides the ranking"


#: The observed leak in miniature (ADR-0004 §10). Repeats "main"/"are"/"the"/"for" and holds
#: none of `risk`/`factors`/`tesla` — the shape of GOOGL's `Item 7:21`, which a live
#: `what are the main risk factors for Tesla?` returned at BM25 **rank 1** over the whole
#: 5,842-chunk corpus, on `main` alone (df 0.1%, idf 6.6568) for 10.6016 of its 16.4742.
SCAFFOLDING_ONLY_CHUNK = Document(
    id="googl-7-21",
    page_content=(
        "GOOGL | FY2025 10-K | Item 7. Management's Discussion and Analysis\n\n"
        "The main components of our research and development expenses are: depreciation "
        "expense for technical infrastructure; and the main compensation expenses for the "
        "engineering employees responsible for the main development programmes."
    ),
    metadata={"ticker": "GOOGL", "section": "Item 7"},
)


def test_a_question_is_scored_on_its_topic_terms_not_its_question_form():
    # The lexicon, pinned as a whole. `main` is the one word here that no published stopword
    # list carries, and it is the whole finding: filers write "principal", so `main` is rare in
    # the corpus and IDF therefore weights it *above the filer's own name*.
    assert query_terms("What are the main risk factors for Tesla?") == [
        "risk",
        "factors",
        "tesla",
    ]


def test_a_chunk_matching_only_a_questions_scaffolding_is_not_a_hit():
    # ADR-0004 §10's observed leak. Scaffolding is dropped before the membership test as well as
    # before the scoring, so this chunk earns no rank — and therefore no RRF vote — rather than
    # merely scoring lower. It shares `main`, `the`, `are` and `for` with the question and not
    # one of `risk`, `factors`, `tesla`.
    index = BM25Index.over([*FORD_CHUNKS, TSLA_CHUNK, SCAFFOLDING_ONLY_CHUNK])

    hits = index.nearest("What are the main risk factors for Tesla?", 5)

    assert "googl-7-21" not in {hit.id for hit in hits}
    assert hits[0].id == "tsla-7-3"


def test_the_corpus_keeps_every_term_the_query_drops():
    # Query-side only, which is what leaves `df`, `idf` and `avgdl` exactly as they were and the
    # baseline byte-identical for any query carrying no scaffolding. In a filing, "the main
    # components" is content; in a question, "the main" is how an analyst asks for salience.
    assert tokenize("The main components are") == ["the", "main", "components", "are"]
    assert query_terms("The main components are") == ["components"]


def test_the_before_case_and_the_exact_identifier_bucket_keep_every_term():
    # ADR-0004 §6's before-case and the bucket BM25 exists to serve carry no scaffolding, so
    # they are untouched by construction — the property that made this safe to land on the
    # measured path. Verified against the real corpus too: all five rows unchanged (§10).
    assert query_terms("Tesla debt") == ["tesla", "debt"]
    assert query_terms("TSLA Item 1A") == ["tsla", "item", "1a"]

    index = BM25Index.over([*FORD_CHUNKS, TSLA_CHUNK, SCAFFOLDING_ONLY_CHUNK])

    assert index.nearest("Tesla debt", 5)[0].id == "tsla-7-3"


def test_a_query_that_is_all_scaffolding_scores_nothing_rather_than_falling_back():
    # Falling back to the unfiltered terms would re-admit exactly the chunks the filter exists
    # to exclude. An empty term list is already "no lexical hits", which `fuse` accepts from a
    # short list — and the vector half still answers the question.
    index = BM25Index.over([*FORD_CHUNKS, TSLA_CHUNK, SCAFFOLDING_ONLY_CHUNK])

    # One content word is still a query; scaffolding alone is not.
    assert query_terms("What are the main ones?") == ["ones"]
    assert query_terms("And what are the main?") == []
    assert index.nearest("And what are the main?", 5) == ()


def test_tokenizing_keeps_the_identifiers_the_exact_identifier_bucket_turns_on():
    # `Item 1A`, `10-K` and a ticker have to survive tokenization on both sides — the query's
    # and the chunk's — or the bucket hybrid search exists to win is decided by punctuation.
    assert tokenize("Item 1A of TSLA's FY2025 10-K") == [
        "item",
        "1a",
        "of",
        "tsla",
        "s",
        "fy2025",
        "10",
        "k",
    ]


# --------------------------------------------------------------------------------------
# The production accessor
# --------------------------------------------------------------------------------------


def test_the_index_over_a_collection_is_built_once_per_store(filings_store):
    # Building it reads every chunk out of Chroma and tokenizes it. Affordable once at the first
    # query, absurd per query — and the reason `vectorstore.default_filings_store` caches the
    # handle this is keyed on.
    bm25_index.cache_clear()

    first = bm25_index(filings_store)
    second = bm25_index(filings_store)

    assert first is second
    assert len(first) > 1, "the real fixture filing chunks into more than one chunk"


def test_the_index_matches_a_literal_the_fixture_filing_actually_contains(filings_store):
    # End to end over a real collection: the chunks were written by `write_chunks`, read back by
    # `all_chunks`, and are matchable by the header literal ADR-0004 put there.
    bm25_index.cache_clear()

    hits = bm25_index(filings_store).nearest("AAPL Item 1A", 5)

    assert hits
    assert all(hit.metadata["ticker"] == "AAPL" for hit in hits)
    assert hits[0].metadata["section"] == "Item 1A"


def test_the_index_over_an_un_ingested_collection_answers_nothing(empty_filings_store):
    # `BM25Okapi` divides by the corpus's average document length, so an empty collection is a
    # `ZeroDivisionError` waiting for the first query rather than the empty result the tiered
    # fallback expects.
    bm25_index.cache_clear()

    assert bm25_index(empty_filings_store).nearest("anything", 5) == ()
