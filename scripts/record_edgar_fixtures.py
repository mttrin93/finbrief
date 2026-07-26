"""Record real EDGAR extractions as fixtures, so the suite tests reality without a network.

Run by hand when a fixture needs refreshing; never by the test suite.
"""

import gzip
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "src")

from finbrief.ingestion.edgar import configure_edgar, fetch_filing  # noqa: E402
from finbrief.ingestion.model import Section  # noqa: E402

os.environ.setdefault("SEC_EDGAR_USER_AGENT", "FinBrief rinaldim1993@gmail.com")
configure_edgar()

OUT = Path("tests/fixtures/edgar")
OUT.mkdir(parents=True, exist_ok=True)


def as_dict(filing):
    return {
        "ref": {
            "ticker": filing.ref.ticker,
            "form": filing.ref.form,
            "accession": filing.ref.accession,
            "fiscal_year": filing.ref.fiscal_year,
            "filing_date": filing.ref.filing_date,
            "url": filing.ref.url,
        },
        "latest_annual_form": filing.latest_annual_form,
        "sections": {s.value: t for s, t in filing.sections.items()},
    }


# 1. A whole clean filing — the real-token measurement and the gate's happy path.
aapl = fetch_filing("AAPL")
path = OUT / "aapl-fy2025.json.gz"
path.write_bytes(gzip.compress(json.dumps(as_dict(aapl), indent=1).encode()))
print(f"{path}  {path.stat().st_size:,} bytes")

# 2. GM's Item 7A — a real Item 7/8 boundary miss, small enough to keep verbatim.
gm = fetch_filing("GM")
sample = {
    "gm_item_7a_spill": gm.sections[Section.MARKET_RISK],
}
# 3. The six real incorporation-by-reference pointers.
for ticker in ["JPM", "BAC", "GS", "JNJ", "LLY", "PFE"]:
    filing = gm if ticker == "GM" else fetch_filing(ticker)
    sample[f"{ticker.lower()}_item_7a_pointer"] = filing.sections[Section.MARKET_RISK]

path = OUT / "section-samples.json"
path.write_text(json.dumps(sample, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"{path}  {path.stat().st_size:,} bytes")
