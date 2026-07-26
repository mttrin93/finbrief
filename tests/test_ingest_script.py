"""The pure-logic contract of `scripts/ingest_filings.py`: exits, and what a run costs.

The script itself is non-hermetic — it talks to EDGAR and pays for embeddings — but its
control flow is not: ticker validation happens before any network identity is needed, and
a `--dry-run` must work on a machine that has an EDGAR identity and **no** OpenRouter key.
That last property is the point of deferring `get_settings()`: these tests run with
`OPENROUTER_API_KEY` unset (the hermetic env guarantees it), so a regression that moves
the settings read back before the dry-run branch fails here with the ConfigError it would
inflict on a user.
"""

import importlib.util
import sys
import types
from pathlib import Path

from finbrief.ingestion.model import ExtractedFiling, FilingRef, Section

SCRIPT = Path(__file__).parent.parent / "scripts" / "ingest_filings.py"

BODY = "The Company designs and sells devices to customers worldwide. " * 40


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
    Settings is what these tests are for.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "FinBrief test@example.com")
    fake_edgar = types.ModuleType("edgar")
    fake_edgar.set_identity = lambda identity: None
    monkeypatch.setitem(sys.modules, "edgar", fake_edgar)
    monkeypatch.setattr(script, "fetch_filing", fetch)


def test_a_ticker_outside_the_universe_exits_2_before_touching_anything():
    # The KB scope *is* the Universe; a typo'd ticker must not reach EDGAR. No identity,
    # no key, no network is configured here — validation has to come first.
    script = load_script()

    assert script.main(["--tickers", "ZZZZ"]) == 2


def test_a_dry_run_needs_an_edgar_identity_but_no_openrouter_key(monkeypatch, tmp_path):
    script = load_script()
    wire_for_a_dry_run(script, monkeypatch, tmp_path, lambda ticker: a_filing(ticker))

    assert script.main(["--tickers", "AAPL", "--dry-run"]) == 0

    report = (tmp_path / "docs/verification/ingest-report.md").read_text(encoding="utf-8")
    assert "GATE PASSED" in report and "dry run" in report.lower()


def test_a_failed_gate_exits_1_and_still_writes_the_evidence(monkeypatch, tmp_path):
    script = load_script()
    broken = {s: f"{s.value}. {s.heading}\n\n{BODY}" for s in Section}
    del broken[Section.MDA]
    wire_for_a_dry_run(
        script, monkeypatch, tmp_path, lambda ticker: a_filing(ticker, sections=broken)
    )

    assert script.main(["--tickers", "AAPL"]) == 1

    report = (tmp_path / "docs/verification/ingest-report.md").read_text(encoding="utf-8")
    assert "GATE FAILED" in report and "section_found" in report
