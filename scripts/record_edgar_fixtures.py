"""Record real EDGAR extractions as fixtures, so the suite tests reality without a network.

    uv run python scripts/record_edgar_fixtures.py

Run by hand when a fixture needs refreshing; never by the test suite — the suite is
hermetic by contract (CLAUDE.md) and reads what this leaves behind.

The recorded cases are the ones nobody would think to invent: a whole clean filing, six
different wordings of "incorporated by reference", and a real Item 7/Item 8 boundary miss.
"""

import gzip
import json
from pathlib import Path

from finbrief.ingestion.edgar import configure_edgar, fetch_filing
from finbrief.ingestion.model import Section

# Reads SEC_EDGAR_USER_AGENT from the environment and fails loudly without it, rather than
# defaulting to a contact address — a committed default would put one person's email in
# every request anyone ever makes with this script.
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
    sample[f"{ticker.lower()}_item_7a_pointer"] = fetch_filing(ticker).sections[
        Section.MARKET_RISK
    ]

path = OUT / "section-samples.json"
path.write_text(json.dumps(sample, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"{path}  {path.stat().st_size:,} bytes")
