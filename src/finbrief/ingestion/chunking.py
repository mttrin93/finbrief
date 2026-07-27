"""Section-aware chunking: one Filing's Sections into the units retrieval scores.

Each Section is split on its own, so no chunk straddles a boundary and every chunk's
`section` metadata is true of all of its text. Size comes from `config.CHUNK_SIZE_CHARS` —
the single source of truth `tests/test_chunk_token_limit.py` holds under the embedding
window (`retrieval/embeddings.py` sends raw strings, so nothing splits an over-long one).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from finbrief.config import CHUNK_SIZE_CHARS
from finbrief.ingestion.model import (
    ExtractedFiling,
    Section,
    is_incorporated_by_reference,
)

#: Characters each chunk repeats from its predecessor. The overlap belongs to the chunker
#: rather than to `config` (PLAN.md): unlike the chunk size it constrains nothing outside
#: this module — no embedding window, no index compatibility — so exporting it would
#: invite an env override with nothing to protect.
CHUNK_OVERLAP_CHARS = 200

#: Split on the largest natural boundary that fits, narrowing only as needed. Paragraphs
#: first, because filing prose is paragraph-structured and a risk factor cut mid-sentence
#: embeds as two half-thoughts.
_SEPARATORS = ("\n\n", "\n", ". ", " ", "")


@dataclass(frozen=True, slots=True)
class Chunk:
    """One unit of retrieval, with the provenance a citation and a ratio both need."""

    id: str
    body: str
    ticker: str
    filing_type: str
    section: Section
    fiscal_year: int
    accession: str
    content_hash: str

    @property
    def text(self) -> str:
        """What gets embedded and BM25-indexed: a provenance header, then the body.

        The header exists for BM25, not for the reader. ADR-0004 stakes the
        `exact-identifier` bucket on lexical matching of literals like "Item 1A" and
        "AAPL", and BM25 scores *text* — it never sees `metadata`. Without the header
        only the one chunk that happens to contain the section heading could match a
        query naming the section, and the bucket hybrid search exists to win would be
        decided by where the splitter happened to cut.

        The body is carried verbatim underneath. Nothing here rewrites a word.
        """
        return (
            f"{self.ticker} | FY{self.fiscal_year} {self.filing_type} | "
            f"{self.section.value}. {self.section.heading}\n\n{self.body}"
        )

    @property
    def metadata(self) -> dict[str, str | int]:
        """Chroma metadata. Primitives only — Chroma rejects anything else at write time.

        `section` is deliberately `.value` (`"Item 1A"`) and not the enum member: it is
        what a metadata filter is written against and what a citation displays.

        `content_hash` is not for retrieval at all — it is how a re-run tells a filing it
        already holds from one whose *extraction* has changed since (`content_hash`
        below, `pipeline.ingest`).
        """
        return {
            "ticker": self.ticker,
            "filing_type": self.filing_type,
            "section": self.section.value,
            "fiscal_year": self.fiscal_year,
            "accession": self.accession,
            "content_hash": self.content_hash,
        }


def chunk_filing(filing: ExtractedFiling) -> tuple[Chunk, ...]:
    """Split one extracted Filing into chunks, in Section order.

    Call this only on a filing the gate has passed. Chunking a misdetected Section is how
    a table-of-contents row becomes a citation.

    A Section the filer incorporated by reference is skipped rather than chunked: it is a
    sentence pointing at Item 7, and as a chunk it would retrieve for every market-risk
    question and ground none of them (ADR-0007 amendment).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_CHARS,
        chunk_overlap=CHUNK_OVERLAP_CHARS,
        separators=list(_SEPARATORS),
        keep_separator=True,
    )

    fingerprint = content_hash(filing)
    chunks: list[Chunk] = []
    for section in Section:
        text = filing.sections.get(section)
        if not text or is_incorporated_by_reference(section, text):
            continue
        for index, body in enumerate(splitter.split_text(text)):
            chunks.append(
                Chunk(
                    id=chunk_id(filing.ref.accession, section, index),
                    body=body,
                    ticker=filing.ref.ticker,
                    filing_type=filing.ref.form,
                    section=section,
                    fiscal_year=filing.ref.fiscal_year,
                    accession=filing.ref.accession,
                    content_hash=fingerprint,
                )
            )
    return tuple(chunks)


#: Enough of the digest to make a collision a non-event: sixteen hex characters is 64 bits
#: over a corpus of fifteen filings. Short because it is carried on every chunk's metadata.
_CONTENT_HASH_CHARS = 16


def content_hash(filing: ExtractedFiling) -> str:
    """A digest of everything about this filing that decides what gets stored.

    The reason it exists: the accession number is EDGAR's identity for a filing, so it
    answers "is this the same document?" — and `pipeline.ingest` needs the answer to a
    different question, "is what we hold what this run would write?". Those came apart the
    first time the extractor was fixed: commit `a0614de` moved eleven Sections' boundaries
    without a single accession changing, so an accession-only skip left the knowledge base
    holding chunks of pre-repair text while the run reported `skipped (already ingested)`
    and the verification artifact attested to the new text (issue #3 review).

    Covers exactly what would go into the store and nothing else: the provenance header
    `Chunk.text` prepends, the Section texts `chunk_filing` actually chunks — a Section
    incorporated by reference is skipped here for the same reason it is skipped there —
    and the splitter's parameters, so a chunk-size change no longer needs `--force` to be
    remembered. It cannot see the *embedding model*, which leaves no trace in the text;
    that one is still `--force`'s job (`retrieval/embeddings.py`).
    """
    digest = hashlib.sha256()
    digest.update(f"{CHUNK_SIZE_CHARS}\0{CHUNK_OVERLAP_CHARS}\0".encode())
    digest.update(
        f"{filing.ref.ticker}\0{filing.ref.form}\0{filing.ref.fiscal_year}\0".encode()
    )
    for section in Section:
        text = filing.sections.get(section)
        if not text or is_incorporated_by_reference(section, text):
            continue
        digest.update(f"{section.value}\0{text}\0".encode())
    return digest.hexdigest()[:_CONTENT_HASH_CHARS]


def chunk_id(accession: str, section: Section, index: int) -> str:
    """`0000320193-25-000079:Item 1A:7` — stable across runs, unique across filings.

    Derived rather than random, and rooted in the accession, so it does double duty:
    re-ingesting the same filing upserts onto its own ids instead of duplicating the
    knowledge base (ADR-0007's idempotency), and ADR-0004's fusion can dedup by it.
    """
    return f"{accession}:{section.value}:{index}"
