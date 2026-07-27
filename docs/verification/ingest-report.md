# Ingest report

Machine evidence of the most recent ingestion run — rewritten by every
full-Universe `scripts/ingest_filings.py` run, and by those only, since a
`--tickers` subset or a `--dry-run` cannot speak to what the collection holds.
Chunk counts are read back from the
persisted collection after the run, not taken from the run's own writes, so an
idempotent re-run that writes nothing still shows what the knowledge base holds.

- Generated: 2026-07-27 08:51 UTC · `scripts/ingest_filings.py`
- Outcome: **GATE PASSED**

## Section-detection gate

```text
Section-detection gate (ADR-0007) — characters extracted per Section
                      Item 1     Item 1A      Item 7     Item 7A
----------------------------------------------------------------
AAPL   FY2025         16,051      68,160      17,996       3,029
MSFT   FY2025         48,645      68,794      48,342       1,995
NVDA   FY2026         48,195     114,728      34,094       4,245
AMZN   FY2025         13,490      60,639      46,206       6,099
GOOGL  FY2025         23,809      85,135      52,417       7,975
META   FY2025         24,525     195,448      60,251       5,944
TSLA   FY2025         45,410      83,599      55,375       1,626
F      FY2025         74,955      93,694     159,080      16,423
GM     FY2025         49,890      67,101      83,698      11,788
JPM    FY2025         39,177     112,774     394,858    ->Item 7
BAC    FY2025         38,036     117,080     293,458    ->Item 7
GS     FY2025        152,312     142,080     309,071    ->Item 7
JNJ    FY2025         34,625      43,416      65,797    ->Item 7
LLY    FY2025         84,138      77,548      48,886    ->Item 7
PFE    FY2025         92,008      87,480     105,475    ->Item 7

6 Section(s) incorporated by reference into Item 7 — lawful, not ingested, listed in the verification artifact (ADR-0007).
All 60 company x Section checks passed.
```

## Chunks in the collection

| Ticker | FY | Accession | Chunks | This run |
|---|---|---|---:|---|
| AAPL | FY2025 | `0000320193-25-000079` | 152 | wrote 152 |
| MSFT | FY2025 | `0000950170-25-100235` | 237 | wrote 237 |
| NVDA | FY2026 | `0001045810-26-000021` | 298 | wrote 298 |
| AMZN | FY2025 | `0001018724-26-000004` | 192 | wrote 192 |
| GOOGL | FY2025 | `0001652044-26-000018` | 239 | wrote 239 |
| META | FY2025 | `0001628280-26-003942` | 422 | wrote 422 |
| TSLA | FY2025 | `0001628280-26-003952` | 280 | wrote 280 |
| F | FY2025 | `0000037996-26-000015` | 499 | wrote 499 |
| GM | FY2025 | `0001467858-26-000013` | 311 | wrote 311 |
| JPM | FY2025 | `0001628280-26-008131` | 735 | wrote 735 |
| BAC | FY2025 | `0000070858-26-000157` | 643 | wrote 643 |
| GS | FY2025 | `0000886982-26-000091` | 875 | wrote 875 |
| JNJ | FY2025 | `0000200406-26-000016` | 215 | wrote 215 |
| LLY | FY2025 | `0000059478-26-000013` | 315 | wrote 315 |
| PFE | FY2025 | `0000078003-26-000026` | 429 | wrote 429 |
| **Total** | | | **5,842** | |
