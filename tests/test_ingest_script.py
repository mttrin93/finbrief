"""The pure-logic contract of `scripts/ingest_filings.py`: exits, evidence, and plumbing.

The script itself is non-hermetic — it talks to EDGAR and pays for embeddings — but its
control flow is not: ticker validation happens before any network identity is needed, and
a `--dry-run` must work on a machine that has an EDGAR identity and **no** OpenRouter key.
That last property is the point of deferring `get_settings()`: these tests run with
`OPENROUTER_API_KEY` unset (the hermetic env guarantees it), so a regression that moves
the settings read back before the dry-run branch fails here with the ConfigError it would
inflict on a user.

The write path is covered too, against a real on-disk Chroma with a fake embedding. What
that buys is the argument plumbing no unit test of `ingest` or `render_ingest_report` can
see: that `--force` reaches `ingest`, that the report's chunk counts are read back from
the store rather than taken from the run's own writes, and that `--section-starts` lands
where it was asked to.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
from langchain_core.embeddings import FakeEmbeddings

from finbrief.config import UNIVERSE
from finbrief.ingestion.model import ExtractedFiling, FilingRef, Section
from finbrief.retrieval.vectorstore import build_filings_store

SCRIPT = Path(__file__).parent.parent / "scripts" / "ingest_filings.py"

BODY = "The Company designs and sells devices to customers worldwide. " * 40

REPORT = Path("docs/verification/ingest-report.md")


def load_script():
    spec = importlib.util.spec_from_file_location("ingest_filings_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def a_filing(ticker, *, sections=None):
    return ExtractedFiling(
        ref=FilingRef(
            ticker=ticker,
            form="10-K",
            accession=f"0000000000-25-{sum(ticker.encode()):06d}",
            fiscal_year=2025,
            filing_date="2025-10-31",
        ),
        latest_annual_form="10-K",
        sections=sections
        if sections is not None
        else {s: f"{s.value}. {s.heading}\n\n{BODY}" for s in Section},
    )


def wire_for_a_dry_run(script, monkeypatch, tmp_path, fetch):
    """An EDGAR identity, a fake `edgar` module, a fetch stub, and a scratch CWD.

    Deliberately no OPENROUTER_API_KEY: proving the dry run never resolves the full
    Settings is what these tests are for. The `chdir` is not cosmetic either — the report
    path is relative, so without it a run under test writes into the repo's own committed
    evidence.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "FinBrief test@example.com")
    fake_edgar = types.ModuleType("edgar")
    fake_edgar.set_identity = lambda identity: None
    monkeypatch.setitem(sys.modules, "edgar", fake_edgar)
    monkeypatch.setattr(script, "fetch_filing", fetch)


def wire_for_a_write(script, monkeypatch, tmp_path, fetch):
    """Everything a dry run needs, plus a paid key and a real Chroma with a fake model."""
    wire_for_a_dry_run(script, monkeypatch, tmp_path, fetch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        script,
        "build_filings_store",
        lambda settings=None: build_filings_store(
            persist_directory=str(tmp_path / "chroma"), embeddings=FakeEmbeddings(size=32)
        ),
    )


def test_a_ticker_outside_the_universe_exits_2_before_touching_anything(monkeypatch, tmp_path):
    # The KB scope *is* the Universe; a typo'd ticker must not reach EDGAR. No identity,
    # no key, no network is configured here — validation has to come first. The `chdir`
    # makes "before touching anything" checkable: the regression this guards against is
    # the report write moving ahead of the ticker check, which would pass on the exit
    # code alone while overwriting the committed evidence.
    monkeypatch.chdir(tmp_path)
    script = load_script()

    assert script.main(["--tickers", "ZZZZ"]) == 2
    assert not (tmp_path / REPORT).exists()


def test_a_dry_run_needs_an_edgar_identity_but_no_openrouter_key(monkeypatch, tmp_path):
    script = load_script()
    wire_for_a_dry_run(script, monkeypatch, tmp_path, lambda ticker: a_filing(ticker))

    assert script.main(["--tickers", "AAPL", "--dry-run"]) == 0


def test_a_dry_run_leaves_the_committed_evidence_of_the_last_full_run_alone(
    monkeypatch, tmp_path
):
    # The report's claim is "all fifteen ingest, and here is what the collection holds".
    # A dry run knows neither, so overwriting the file with a store-less stub would
    # destroy the evidence — and only show up later as an unexpected `git diff`.
    script = load_script()
    wire_for_a_dry_run(script, monkeypatch, tmp_path, lambda ticker: a_filing(ticker))
    (tmp_path / REPORT).parent.mkdir(parents=True)
    (tmp_path / REPORT).write_text("the last full run", encoding="utf-8")

    assert script.main(["--dry-run"]) == 0

    assert (tmp_path / REPORT).read_text(encoding="utf-8") == "the last full run"


def test_a_partial_run_leaves_the_committed_evidence_alone_too(monkeypatch, tmp_path):
    script = load_script()
    wire_for_a_write(script, monkeypatch, tmp_path, lambda ticker: a_filing(ticker))
    (tmp_path / REPORT).parent.mkdir(parents=True)
    (tmp_path / REPORT).write_text("the last full run", encoding="utf-8")

    assert script.main(["--tickers", "AAPL"]) == 0

    assert (tmp_path / REPORT).read_text(encoding="utf-8") == "the last full run"


def test_a_failed_gate_exits_1_and_still_writes_the_evidence(monkeypatch, tmp_path):
    # A full run that fails is exactly when the evidence matters most, so the report is
    # rewritten here even though nothing was ingested.
    script = load_script()
    broken = {s: f"{s.value}. {s.heading}\n\n{BODY}" for s in Section}
    del broken[Section.MDA]
    wire_for_a_dry_run(
        script, monkeypatch, tmp_path, lambda ticker: a_filing(ticker, sections=broken)
    )

    assert script.main([]) == 1

    report = (tmp_path / REPORT).read_text(encoding="utf-8")
    assert "GATE FAILED" in report and "section_found" in report


def test_a_full_run_writes_the_chunks_and_reports_what_the_store_holds(monkeypatch, tmp_path):
    script = load_script()
    wire_for_a_write(script, monkeypatch, tmp_path, lambda ticker: a_filing(ticker))

    assert script.main([]) == 0

    report = (tmp_path / REPORT).read_text(encoding="utf-8")
    assert "GATE PASSED" in report
    assert "## Chunks in the collection" in report
    # Every Universe company, and the counts are the store's own read-back, not the run's
    # write tally — the distinction `render_ingest_report` exists to preserve.
    for company in UNIVERSE:
        assert f"| {company.ticker} |" in report
    assert "wrote " in report


def test_an_idempotent_rerun_reports_holdings_it_did_not_write(monkeypatch, tmp_path):
    script = load_script()
    wire_for_a_write(script, monkeypatch, tmp_path, lambda ticker: a_filing(ticker))
    script.main([])

    assert script.main([]) == 0

    report = (tmp_path / REPORT).read_text(encoding="utf-8")
    assert "skipped (already ingested)" in report
    assert "wrote " not in report, "a skipped run writes nothing"
    # The whole point: zero written, and the collection still proves what it holds.
    assert "**Total**" in report and "| **0** |" not in report


def test_force_reaches_the_pipeline_so_a_rerun_re_embeds(monkeypatch, tmp_path):
    # `--force` is pure plumbing: nothing between the flag and `ingest(force=...)` can be
    # observed from the outside, so a flag that quietly stopped being passed would ship.
    script = load_script()
    wire_for_a_write(script, monkeypatch, tmp_path, lambda ticker: a_filing(ticker))
    script.main([])

    assert script.main(["--force"]) == 0

    report = (tmp_path / REPORT).read_text(encoding="utf-8")
    assert "wrote " in report
    assert "skipped (already ingested)" not in report


def test_section_starts_writes_the_checklist_where_it_was_asked_to(monkeypatch, tmp_path):
    script = load_script()
    wire_for_a_dry_run(script, monkeypatch, tmp_path, lambda ticker: a_filing(ticker))

    assert script.main(["--tickers", "AAPL", "--dry-run", "--section-starts", "out/c.md"]) == 0

    checklist = (tmp_path / "out/c.md").read_text(encoding="utf-8")
    assert "Hand-verification" in checklist
    assert checklist.count("- [ ]") == len(Section)


def test_a_subset_run_refuses_to_delete_the_other_companies_verified_rows(
    monkeypatch, tmp_path, capsys
):
    """`--tickers JPM --section-starts docs/verification/…` must not truncate the artifact.

    `render_section_starts` emits rows for the filings it is handed and no others, and the
    carry-forward cannot save what the run never fetched — so a one-company re-render of a
    sixty-row checklist deletes fifty-six hand-earned ticks and every note attached to them.
    That is the one loss the artifact's whole design exists to prevent, and it was reachable
    from the command most likely to be run while iterating on one company (ADR-0007 §7).
    """
    script = load_script()
    wire_for_a_dry_run(script, monkeypatch, tmp_path, lambda ticker: a_filing(ticker))
    script.main(["--dry-run", "--section-starts", "c.md"])
    full = (tmp_path / "c.md").read_text(encoding="utf-8")
    verified = full.replace("- [ ]", "- [x]").replace(
        "```text", "**Finding:** check pp.46-160.\n\n```text", 1
    )
    (tmp_path / "c.md").write_text(verified, encoding="utf-8")

    assert script.main(["--tickers", "AAPL", "--dry-run", "--section-starts", "c.md"]) == 0

    assert (tmp_path / "c.md").read_text(encoding="utf-8") == verified
    assert "did not fetch" in capsys.readouterr().err


def test_a_full_run_may_still_re_render_the_committed_checklist(monkeypatch, tmp_path):
    # The guard is about coverage, not about writing: a full run — dry or not — is exactly
    # who is allowed to rewrite the artifact, and the ticks it carries forward.
    script = load_script()
    wire_for_a_dry_run(script, monkeypatch, tmp_path, lambda ticker: a_filing(ticker))
    script.main(["--dry-run", "--section-starts", "c.md"])
    verified = (tmp_path / "c.md").read_text(encoding="utf-8").replace("- [ ]", "- [x]")
    (tmp_path / "c.md").write_text(verified, encoding="utf-8")

    assert script.main(["--dry-run", "--section-starts", "c.md"]) == 0

    rerendered = (tmp_path / "c.md").read_text(encoding="utf-8")
    assert rerendered.count("- [x]") == len(UNIVERSE) * len(Section)
    assert "**CHANGED**" not in rerendered


@pytest.mark.parametrize("flag", ["--dry-run", "--force"])
def test_the_checklist_is_written_even_when_the_gate_fails(monkeypatch, tmp_path, flag):
    # The checklist is the human's to-do list, and a gate failure is precisely when
    # someone needs to look at the text. Writing it after the findings check would hide
    # the excerpts in the run that most needs them.
    script = load_script()
    broken = {s: f"{s.value}. {s.heading}\n\n{BODY}" for s in Section}
    del broken[Section.MDA]
    wire_for_a_dry_run(
        script, monkeypatch, tmp_path, lambda ticker: a_filing(ticker, sections=broken)
    )

    assert script.main(["--tickers", "AAPL", flag, "--section-starts", "c.md"]) == 1

    assert "NOT EXTRACTED" in (tmp_path / "c.md").read_text(encoding="utf-8")
