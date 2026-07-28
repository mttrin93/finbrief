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
- Tool output reports the peer set and n inline — e.g. for TSLA, "vs. mean of 2 `autos`
  peers: F, GM" (n comes from the cluster: 2 for the three-member clusters, 5 for
  `big_tech`).

**Considered and rejected.**
- *TSLA grouped with mega-cap tech* — rejected on ratio-comparability grounds (autos and
  tech have structurally different margin/multiple profiles).
- *Out-of-Universe peers + snapshot table* — rejected; reintroduces the API surface the
  in-Universe rule eliminates.
- *Company-vs-own-history instead of peers* — deferred to Tier-2; dropping peers would kill
  demo step 2 for a saving the in-Universe rule already achieves.

---

## Amendment (T5, #9) — what "the mean" reports beside it

Implementing `calculate_ratios` against real data left the decision above intact and added two
things it did not anticipate. Both are consequences of the same fact: **the data is not complete,
and a mean over two peers is fragile.**

**1. The peer set and `n` stay the cluster's; per-metric coverage is reported separately.**
The decision says the tool reports the peer set and n inline, and it does — from `config.PEERS`,
in full, whatever the data turned out to hold. But a *metric* can be thinner than the cluster:
in the recorded snapshot JPM and BAC report no `debtToEquity` at all, so a `banks` leverage
comparison rests on GS alone. Averaging the two absences in as zero would report a peer mean a
third of the truth and it would look like a measurement, so an absent figure is **excluded** from
the mean and the tool says "1 of 2 peers reported this". "Mean of 1 of 2 peers" and "mean of 2
peers" are different claims; the basis sentence makes the first one.

A peer whose *quote* could not be fetched is reported apart from a peer with no figure
(`PeerComparison.unavailable`), because the causes differ and only one of them might work on a
retry.

**2. The range travels with the mean.** An arithmetic mean of multiples is dominated by its
outlier and the Universe has a bad one: Ford's two `autos` peers are TSLA at 286× and GM at 37×,
so a peer mean P/E of 162× describes neither company — and it is the number a reader would quote.
This ADR specifies a mean and the tool reports a mean, **with the low and the high beside it**, so
the sentence that gets quoted carries its own caveat.

Substituting a median or a harmonic mean would be a better estimator and is *not* done here: it
would be a change to this decision, made in the open, and it is a legitimate Tier-2 experiment
rather than a fix to apply quietly at implementation time.

**Also measured, and worth recording for T10.** `grossMargins` arrives as exactly `0.0` for JPM
and BAC beside operating margins of 50% and 38%. Gross profit is by construction at least
operating profit, so that zero is arithmetically impossible — it is the endpoint's absence marker,
and `finance/quotes.py` reads it as one. That is the only coercion on the path, it is bounded to
margins and to exactly zero, and Ford's genuine −3.2% net margin is untouched by it.
