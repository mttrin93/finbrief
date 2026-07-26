# ADR-0009: Peers drawn exclusively from the Universe

`calculate_ratios` compares a company's ratios against "peer averages." Peers from outside
the Universe would require fetching fundamentals for arbitrary tickers on every call —
multiplying API load against yfinance flakiness and Alpha Vantage's 25-calls/day cap — and
would need a separate cached snapshot table.

**Decision.** Peers are drawn **exclusively from the Universe** — no out-of-Universe peers,
no snapshot table.

- The **Universe is curated as same-sector peer clusters** (big tech / autos / banks /
  healthcare), so it serves triple duty: KB scope, demo cast, and peer pool.
- Every `calculate_ratios` call resolves peers via the **same TTL-cached `get_stock_data`
  path**, so peer averaging adds **zero new API surface**.
- `PEERS` is a **static map in `config.py`**. Selection rule: *same curated peer cluster
  within the Universe*.

  Not *same GICS sector* — an earlier wording of this rule that the clusters never
  implemented. `big_tech` spans three GICS sectors (Information Technology for
  AAPL/MSFT/NVDA, Consumer Discretionary for AMZN, Communication Services for
  GOOGL/META), and AMZN sits apart from TSLA/F/GM despite sharing a GICS sector with
  them. That is deliberate: the clusters are curated for **ratio comparability**, which
  is also the ground on which this ADR rejects grouping TSLA with mega-cap tech below —
  a GICS rule would have forced exactly that pairing. The cluster is the unit; GICS is
  an input to curating it, not the rule.
- Tool output reports the peer set and n inline — e.g. "vs. mean of 3 sector peers: F, GM".

**Considered and rejected.**
- *TSLA grouped with mega-cap tech* — rejected on ratio-comparability grounds (autos and
  tech have structurally different margin/multiple profiles).
- *Out-of-Universe peers + snapshot table* — rejected; reintroduces the API surface the
  in-Universe rule eliminates.
- *Company-vs-own-history instead of peers* — deferred to Tier-2; dropping peers would kill
  demo step 2 for a saving the in-Universe rule already achieves.
