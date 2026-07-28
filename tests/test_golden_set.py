"""The golden set's schema and the ADR-0002 / ADR-0004 sampling constraints it encodes.

Seam: the artifact itself. `golden_set.json` is hand-authored data that three Tier-1 pillars
read (RAGAs, the per-bucket A/B, precision/recall), so a malformed or self-inconsistent row
does not fail loudly at eval time — it silently reports a number against a question nobody
can defend. These tests are what makes it fail loudly instead.

**Hermetic, and that is what bounds them.** CLAUDE.md forbids pointing a test at the ingested
`data/chroma`: it is only searchable by the paid model that wrote it. So nothing here can
check that a `chunk_ids` entry *exists*, that a `passage` is really that chunk's text, or that
a `section_chunk_counts` number is right — those are cross-checks against the collection, run
at authoring time and re-run by the hand-verification pass (`verified_against_edgar`). What
these tests do own is everything derivable without the collection: the shape, the internal
consistency of the derived flags, and agreement with `config` — which is where the
pointer-filer rule, the Universe and the peer map live.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from finbrief.config import COMPANIES, ITEM_7A_POINTER_FILERS, PEERS, TICKERS
from finbrief.ingestion.model import Section

GOLDEN_SET_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "finbrief" / "evaluation" / "golden_set.json"
)

#: The four strata, and only these (ADR-0002 decision 2). Guardrail/injection cases are
#: excluded by decision, not by omission — faithfulness against a refusal is undefined.
BUCKETS = frozenset({"semantic", "exact-identifier", "tool-augmented", "multi-hop"})

#: The tools `create_agent` binds (ADR-0008). `search_filings` is deliberately absent: every
#: row in every bucket needs it to reach its filings half, so a `tool_expectation` naming it
#: would be noise. The field records the *additional* tool a tool-augmented row needs.
NON_RETRIEVAL_TOOLS = frozenset({"get_stock_data", "calculate_ratios", "get_recent_news"})

#: The filers whose own name is a poor lexical identifier for their own chunks, so a question
#: naming them by name is where entity normalisation does the work ADR-0004's amendment credits
#: it with. Measured on the ingested corpus, leading declared alias, 5,842 chunks (issue #4):
#: JNJ 0% (`j&j` in 0 of 215), LLY 6%, AMZN 10%, TSLA 12%, NVDA 13%. Lives here rather than in
#: `config.py` because it is a property of the *corpus*, not a knob — the same reason the
#: ingestion thresholds sit next to their rules.
LOW_NAME_COVERAGE_FILERS = frozenset({"JNJ", "LLY", "AMZN", "TSLA", "NVDA"})

#: The two labelled control cells the `exact-identifier` bucket must carry exactly one of each
#: (issue #4, T6 sampling constraint 4). `F` is the filer normalisation never fires for
#: (`MIN_LEXICAL_TICKER_CHARS` excludes a one-character ticker), so it measures the
#: un-normalised path; `META`'s name is a token in 100% of its own chunks and 0 elsewhere, so
#: normalisation is a no-op in effect. Both must be labelled rather than excluded, so #11 can
#: report the bucket with and without them instead of discovering the confound afterwards.
CONTROL_LABELS = {"F": "control:unnormalised-path", "META": "control:already-covered-path"}

#: `accession:Item N:index` — the chunk id ingestion assigns, which is what makes re-ingest
#: idempotent and what a reference has to name to be checkable.
CHUNK_ID = re.compile(r"^\d{10}-\d{2}-\d{6}:Item \d+A?:\d+$")

REQUIRED_KEYS = frozenset(
    {
        "id",
        "bucket",
        "question",
        "reference",
        "grounding",
        "labels",
        "known_false_positives",
        "section_chunk_counts",
        "recall_trivial",
        "recall_trivial_sections",
        "basis_mismatch",
        "tool_expectation",
        "verified_against_edgar",
    }
)


@pytest.fixture(scope="module")
def golden_set() -> dict:
    return json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rows(golden_set: dict) -> list[dict]:
    return golden_set["questions"]


# --- ADR-0002 decisions 2 and 5: the strata and the size ------------------------------


def test_size_is_in_adr_0002s_range(rows):
    # 15-20 across the buckets was too thin for a directional per-bucket claim.
    assert 24 <= len(rows) <= 28


def test_every_bucket_has_at_least_six_questions(rows):
    by_bucket = Counter(row["bucket"] for row in rows)
    assert set(by_bucket) == BUCKETS
    for bucket, n in by_bucket.items():
        assert n >= 6, f"{bucket} has {n} questions, too thin for a per-bucket claim"


def test_ids_are_unique(rows):
    ids = [row["id"] for row in rows]
    assert len(set(ids)) == len(ids)


def test_every_row_carries_every_field(rows):
    for row in rows:
        missing = REQUIRED_KEYS - set(row)
        assert not missing, f"{row['id']} is missing {sorted(missing)}"
        assert row["question"].strip()
        assert row["reference"].strip(), f"{row['id']} has no ground-truth answer"
        assert row["grounding"], f"{row['id']} has no filings-half grounding"


def test_every_grounding_entry_is_in_scope(rows):
    sections = {section.value for section in Section}
    for row in rows:
        for entry in row["grounding"]:
            assert entry["ticker"] in TICKERS, f"{row['id']} cites a non-Universe filer"
            assert entry["section"] in sections, f"{row['id']} cites an un-ingested Section"
            assert entry["chunk_ids"], f"{row['id']} cites a Section but no chunk"
            assert entry["passage"].strip(), f"{row['id']} cites chunks but quotes nothing"
            for chunk_id in entry["chunk_ids"]:
                assert CHUNK_ID.match(chunk_id), f"{row['id']}: malformed chunk id {chunk_id}"
                assert chunk_id.startswith(f"{entry['accession']}:{entry['section']}:"), (
                    f"{row['id']}: {chunk_id} does not belong to the declared "
                    f"{entry['accession']} {entry['section']}"
                )


# --- The KB's own shape: one filing per filer, and six filers with no Item 7A ----------


def test_a_filer_never_appears_with_two_fiscal_years_or_two_accessions(rows):
    # The KB holds one filing per company, the most recent, so "latest 10-K" is not one
    # fiscal year: NVDA is FY2026 and the other 14 are FY2025. Asserted as a function
    # rather than a typed table, so it catches an FY2025 NVDA row without a second copy
    # of the year map drifting from the ingest report.
    by_ticker: dict[str, set[tuple[int, str]]] = {}
    for row in rows:
        for entry in row["grounding"]:
            by_ticker.setdefault(entry["ticker"], set()).add(
                (entry["fiscal_year"], entry["accession"])
            )
    for ticker, filings in by_ticker.items():
        assert len(filings) == 1, f"{ticker} is cited as two different filings: {filings}"


def test_no_row_expects_item_7a_from_a_pointer_filer(rows):
    # The six pointer filers answer Item 7A by incorporating Item 7 by reference; the
    # pointer is lawful and deliberately not ingested, so they have no Item 7A chunks at
    # all. A row expecting Item 7A for one of them would score as a retrieval miss when
    # retrieval was correct — it would measure the authoring error, not the chain.
    for row in rows:
        for entry in row["grounding"]:
            if entry["ticker"] in ITEM_7A_POINTER_FILERS:
                assert entry["section"] != Section.MARKET_RISK.value, (
                    f"{row['id']} expects {entry['ticker']} Item 7A, which the KB does not "
                    "hold — its market-risk content is under the Item 7 label"
                )


def test_at_least_one_row_grounds_a_pointer_filers_market_risk_in_item_7(rows):
    # The converse of the rule above, and the reason it matters: the set has to actually
    # exercise the Item-7-labelled market-risk path, not merely avoid tripping on it.
    grounded = {
        entry["ticker"]
        for row in rows
        if "pointer-filer" in row["labels"]
        for entry in row["grounding"]
        if entry["section"] == Section.MDA.value
    }
    assert grounded & ITEM_7A_POINTER_FILERS


# --- Derived flags: recall_trivial, and the section-count bookkeeping ------------------


def test_section_chunk_counts_covers_exactly_the_sections_a_row_grounds_in(rows):
    for row in rows:
        grounded = {f"{e['ticker']} {e['section']}" for e in row["grounding"]}
        assert set(row["section_chunk_counts"]) == grounded, (
            f"{row['id']}: section_chunk_counts {sorted(row['section_chunk_counts'])} "
            f"does not match the sections it grounds in {sorted(grounded)}"
        )
        for count in row["section_chunk_counts"].values():
            assert isinstance(count, int) and count > 0


def test_recall_trivial_sections_are_exactly_the_sections_holding_four_chunks_or_fewer(rows):
    # k is 5 in eval mode, so a Section holding <= 4 chunks cannot be missed: the question
    # scores recall 1.0 whatever the retriever does. T10 reports these separately rather
    # than letting a question that cannot miss inflate the recall column.
    for row in rows:
        trivial = sorted(k for k, v in row["section_chunk_counts"].items() if v <= 4)
        assert sorted(row["recall_trivial_sections"]) == trivial, (
            f"{row['id']}: recall_trivial_sections {sorted(row['recall_trivial_sections'])} "
            f"!= the sections holding <= 4 chunks {trivial}"
        )


def test_recall_trivial_is_true_only_when_every_target_section_is_trivial(rows):
    # A multi-section row is not all-or-nothing: M2 grounds in AAPL Item 7A (4 chunks) and
    # MSFT Item 7A (3) but also META Item 7A (8), so recall on it can genuinely miss.
    # `recall_trivial` is the whole-row claim; `recall_trivial_sections` carries the partial
    # case, and conflating them would let a partially-trivial row be reported as immune.
    for row in rows:
        every = all(v <= 4 for v in row["section_chunk_counts"].values())
        assert row["recall_trivial"] is every, (
            f"{row['id']}: recall_trivial={row['recall_trivial']} but "
            f"{'every' if every else 'not every'} target Section holds <= 4 chunks"
        )
        if row["recall_trivial"]:
            assert sorted(row["recall_trivial_sections"]) == sorted(row["section_chunk_counts"])


# --- ADR-0004 §11: the mention-leakage probes and their known false positives ----------


def test_known_false_positives_are_populated_for_every_mention_leakage_probe(rows):
    probes = [row for row in rows if "probe:mention-leakage" in row["labels"]]
    assert len(probes) >= 2, "ADR-0004 §11 asks for the Microsoft case and the Apple case"
    for row in probes:
        assert row["known_false_positives"], (
            f"{row['id']} is labelled a mention-leakage probe but names no false positive, "
            "so #11 cannot compute mention-leakage precision for it"
        )


def test_a_known_false_positive_is_never_the_rows_own_filing(rows):
    # The point of the label is that the chunk is a correct lexical match filed by the
    # wrong company. A false positive drawn from the row's own accession would be a
    # grounding chunk mislabelled, which would silently deflate the probe's precision.
    for row in rows:
        own = {entry["accession"] for entry in row["grounding"]}
        for chunk_id in row["known_false_positives"]:
            assert CHUNK_ID.match(chunk_id), f"{row['id']}: malformed fp id {chunk_id}"
            accession = chunk_id.split(":", 1)[0]
            assert accession not in own, (
                f"{row['id']}: {chunk_id} is from the row's own filing {accession}"
            )


def test_no_chunk_is_both_grounding_and_a_known_false_positive(rows):
    for row in rows:
        grounding = {cid for entry in row["grounding"] for cid in entry["chunk_ids"]}
        overlap = grounding & set(row["known_false_positives"])
        assert not overlap, f"{row['id']}: {sorted(overlap)} is cited as both"


def test_only_labelled_probes_carry_known_false_positives(rows):
    for row in rows:
        if row["known_false_positives"]:
            assert any(label.startswith("probe:") for label in row["labels"]), (
                f"{row['id']} names false positives but is not labelled a probe, so #11 "
                "would fold them into the bucket mean"
            )


def test_the_question_form_probe_is_present_and_keeps_its_scaffolding(rows):
    # ADR-0004 §10: `main` has df 7/5842, so IDF weighted it above the filer's own name and
    # BM25 ranked a GOOGL chunk into slot [5] of this query. The bucket average hides it —
    # one slot of five on one question is invisible in a mean but is a displaced rank in
    # context precision — so the instance has to be labelled and its wording preserved.
    probes = [row for row in rows if "probe:question-form" in row["labels"]]
    assert probes, "the semantic bucket needs at least one question-form probe"
    assert any("main" in row["question"].lower() for row in probes)
    assert all(row["bucket"] == "semantic" for row in probes)


# --- ADR-0004 sampling: low-name-coverage filers, and the two labelled controls --------


def test_exact_identifier_bucket_samples_every_low_name_coverage_filer(rows):
    sampled = {
        entry["ticker"]
        for row in rows
        if row["bucket"] == "exact-identifier"
        for entry in row["grounding"]
    }
    missing = LOW_NAME_COVERAGE_FILERS - sampled
    assert not missing, (
        f"{sorted(missing)} unsampled — these are where a question naming the company by "
        "name fails to reach the filer's chunks lexically, so an unsampled one means the "
        "bucket does not exercise entity normalisation at all"
    )


def test_the_two_control_cells_appear_exactly_once_each_and_are_labelled(rows):
    for ticker, label in CONTROL_LABELS.items():
        matching = [
            row
            for row in rows
            if row["bucket"] == "exact-identifier"
            and any(entry["ticker"] == ticker for entry in row["grounding"])
        ]
        assert len(matching) == 1, (
            f"{ticker} appears in {len(matching)} exact-identifier rows, expected exactly 1 — "
            "several would make +translation look like it adds nothing on this bucket, and "
            "do it invisibly, because the questions would all pass"
        )
        assert label in matching[0]["labels"], f"{matching[0]['id']} must carry {label}"


# --- ADR-0002 amendment: the tool-augmented rows' two halves --------------------------


def test_tool_expectation_is_present_exactly_on_the_tool_augmented_bucket(rows):
    for row in rows:
        if row["bucket"] == "tool-augmented":
            assert row["tool_expectation"] is not None, (
                f"{row['id']} is tool-augmented but names no tool, so the tool-calling eval "
                "has nothing to score"
            )
        else:
            assert row["tool_expectation"] is None, (
                f"{row['id']} is {row['bucket']} but names a tool"
            )


def test_every_tool_expectation_names_a_bound_tool_and_the_rows_own_filer(rows):
    for row in rows:
        expectation = row["tool_expectation"]
        if expectation is None:
            continue
        assert expectation["name"] in NON_RETRIEVAL_TOOLS, (
            f"{row['id']} expects {expectation['name']!r}, which the agent does not bind"
        )
        ticker = expectation["args"]["ticker"]
        assert ticker in TICKERS
        grounded = {entry["ticker"] for entry in row["grounding"]}
        assert ticker in grounded, (
            f"{row['id']} calls a tool for {ticker} but grounds in {sorted(grounded)} — the "
            "two halves of the answer would be about different companies"
        )


def test_peer_set_and_n_are_config_peers_and_nothing_else(rows):
    # ADR-0009: peers are drawn exclusively from the Universe, never external tickers, and
    # never selected by GICS sector. A hand-typed peer set in the golden set would be the
    # one place that could disagree with the map the tool actually resolves against.
    for row in rows:
        expectation = row["tool_expectation"]
        if expectation is None:
            continue
        ticker = expectation["args"]["ticker"]
        if expectation["name"] == "calculate_ratios":
            assert tuple(expectation["peer_set"]) == PEERS[ticker], (
                f"{row['id']}: peer_set {expectation['peer_set']} != config.PEERS[{ticker!r}] "
                f"{list(PEERS[ticker])}"
            )
            assert expectation["n"] == len(PEERS[ticker])
        else:
            assert expectation["peer_set"] is None
            assert expectation["n"] is None


# --- Flag ①'s resolution: a compared figure states its basis ---------------------------


def test_a_basis_mismatch_row_compares_filers_and_names_each_one(rows):
    # The comparison being non-trivial is what makes it a good multi-hop row, but the
    # reference must not imply the numbers are directly comparable. It cannot state each
    # basis without naming each filer it states one for.
    for row in rows:
        if not row["basis_mismatch"]:
            continue
        tickers = {entry["ticker"] for entry in row["grounding"]}
        assert len(tickers) >= 2, f"{row['id']}: basis_mismatch on a single-filer row"
        for ticker in tickers:
            # A reference is prose an analyst reads, so it names the company the way one is
            # named — `Apple`, not `AAPL`. Any form the Universe declares counts; what must
            # not happen is a filer's figure appearing with no filer attached to it.
            company = COMPANIES[ticker]
            forms = (ticker, company.name, *company.aliases)
            assert any(form in row["reference"] for form in forms), (
                f"{row['id']}: reference names {ticker} by none of {forms}, so it cannot be "
                "stating that filer's basis and date"
            )


# --- The hand-verification gate (ADR-0002 decision 1) ---------------------------------


def test_every_row_declares_whether_it_has_been_verified_against_the_primary_source(rows):
    for row in rows:
        assert isinstance(row["verified_against_edgar"], bool)


def test_the_sets_verified_flag_is_the_conjunction_of_its_rows(golden_set, rows):
    # Candidates are hand-verified against the primary source, never self-certified. The
    # top-level flag is what a report may cite, so it may not read true while any row is
    # unverified — ingestion could mis-parse a Section and the mis-parse would propagate
    # straight into ground truth.
    assert golden_set["provenance"]["verified_against_edgar"] is all(
        row["verified_against_edgar"] for row in rows
    )
