# PMXT cross-venue research monitor

This integration is a bounded, read-only research monitor for possible
Kalshi/Polymarket identity pairs. PMXT is candidate discovery only: its
mappings, IDs, prices, confidence scores, and relationship labels are never
authoritative market semantics, venue identifiers, native books, executable
evidence, or proof of profit. Local semantic verification and evidence read
directly from each venue remain authoritative.

The monitor is alert-only. It has no order, cancel, RFQ, position, balance,
funding, signing, authorization, or automated-execution surface. It does not
deploy a daemon or scheduler, use paid data, or broaden credential scope.
Every emitted row has `live_eligible=false`.

## Candidate discovery

The PMXT credential is read only from the process environment as
`PMXT_API_KEY`. It is not accepted as a command-line argument, logged, or
persisted. If it is missing, discovery fails before constructing a PMXT client,
making any network request, or creating an artifact directory.

```powershell
uv run main.py pmxt-monitor `
  --limit 25 `
  --min-confidence 0.80 `
  --threshold 0.05
```

One discovery invocation makes exactly one PMXT Router request. A successful
response is capped at 5,000,000 decoded bytes before JSON parsing. The query is
fixed to:

- relation: `identity`;
- minimum confidence: at least `0.80`;
- maximum result limit: `25`;
- minimum distinct venues: `2`;
- venues: Kalshi and Polymarket;
- raw pairwise matches: required.

Non-identity relations and missing direct raw match records are rejected. The
artifact field `raw_edge_present` refers only to that direct PMXT identity-match
record; it does not mean a price discrepancy, executable quote, or trading
edge. Pathological cluster fan-out, excess clusters, and candidate overflow
fail closed within the configured bounds before venue-native reads. The older
`pmxt-sync` command remains available for candidate-only snapshots and does not
perform semantic verification or native-book analysis.

Normalized outcomes retain PMXT's displayed `best_bid` and `best_ask` fields
only as catalog diagnostics. When `price`, `best_bid`, and `best_ask` are all
valid, `catalog_quote_coherence_status` records whether the catalog price lies
within that same record's displayed spread. A candidate-level
`catalog_price_conflicts_with_displayed_spread` flag is raised if any outcome is
outside its displayed spread. Neither field rejects an identity proposal or
creates an edge calculation: PMXT catalog prices and displayed quotes remain
unsynchronized, depth-free, fee-free, and non-executable.

## Native-only continuation

An immutable candidate set can be re-evaluated against fresh public
venue-native evidence without another PMXT request and without inspecting or
requiring `PMXT_API_KEY`:

```powershell
uv run main.py pmxt-monitor `
  --source-run-id <source_run_id> `
  --limit 25 `
  --min-confidence 0.80 `
  --threshold 0.05
```

Continuation makes exactly zero PMXT requests. Before returning any source
candidate, the loader verifies containment, schema and safety flags, every
manifest-listed artifact's byte size and SHA-256, the embedded raw-payload
hash, the bounded identity-only query, the candidate count, and every
candidate's source-run binding. Symlinks, path traversal, hash drift, duplicate
IDs, unsupported artifacts, and out-of-envelope source rows fail closed before
a new run is reserved. The continuation manifest records the source run,
source manifest hash, source artifact hashes, raw payload hash, continuation
time, and `pmxt_network_requests=0`.

## Venue-native semantic evidence

Each direct PMXT proposal is independently reverse-resolved from explicit
venue hints. PMXT catalog UUIDs are never sent to a venue API. Missing,
ambiguous, or mismatched native identifiers fail closed.

The native clients are public and GET-only. They set `trust_env=false`, so
proxy credentials, proxy routing, and other HTTP settings are not inherited
from the process environment. Every completed HTTP acquisition
requests identity encoding and retains the raw HTTP entity bytes as base64,
their SHA-256, parsed response, an explicit provenance-header allowlist (both
normalized and ordered forms), HTTP status, final URL, body-completeness flag,
request ID and cache headers, wall-clock request/response bounds, monotonic
bounds, and measured round-trip time. Unexpected content encoding fails closed.
Sensitive or irrelevant response headers such as cookies are not persisted,
and each client cookie jar is cleared after every response or error so a server
cookie cannot be replayed on a later request. Redirects for the official Kalshi
fee document fail closed rather than carrying state to another request.
The semantic verifier recomputes raw-response and normalized-rule hashes and
checks the acquisition envelope rather than trusting producer-supplied digests.

Kalshi evidence includes the native market, event metadata, event, and series,
including ticker relationships, binary/MVE structure, timing, outcome labels,
settlement fields, and complete normalized rules. Polymarket evidence includes
the Gamma market and nested event, condition ID, distinct CLOB YES/NO token IDs,
negative-risk/group structure, timing, outcome labels, settlement fields, and
complete normalized rules. Native market/event/series/condition/outcome IDs,
raw responses, acquisition timestamps, and rule hashes are preserved.

The verifier compares:

- proposition meaning and explicit YES/NO polarity;
- native outcome labels and distinct native outcome IDs;
- binary market and mutually-exclusive/negative-risk group structure;
- close/deadline instant, expiration horizon, and timezone offset;
- settlement authority and resolution source;
- resolution criteria and rule text;
- void/cancel behavior and material edge cases;
- settlement delay.

Every proposal becomes exactly one of `VERIFIED_EQUIVALENT`, `REJECTED`, or
`NEEDS_REVIEW`, with explicit reason codes, comparisons, native summaries, raw
hashes, rule hashes, and request-body hashes. Inverted polarity, different
deadlines or timezones, differing settlement sources or rules, ambiguous token
IDs, non-binary/MVE products, and incompatible group structure are never
promoted as identity. Missing or insufficient evidence produces
`NEEDS_REVIEW`; it is not inferred away from PMXT confidence.
Matching grouped-market booleans alone do not establish that candidate sets or
an `Other` resolution path are equivalent, so grouped pairs remain
`NEEDS_REVIEW` until that structure is proven from venue-native evidence.

## Native books, freshness, and fees

Only `VERIFIED_EQUIVALENT` proposals reach book acquisition. The two venue legs
are requested concurrently to reduce capture skew. Kalshi L2 asks are derived
from the opposite native bid ladder; Polymarket YES and NO token books are read
from the CLOB. These are public top-of-book/depth snapshots only—there is no
websocket, FIX, L3, order, or private account activity.

Freshness is strict. A book must contain a venue-source timestamp, valid
request/response and monotonic bounds, acceptable age, acceptable cross-venue
skew, and enough non-crossed ask depth for the full configured size. Local
receipt time is evidence of acquisition, not a substitute for a venue-source
timestamp. Kalshi's current REST order-book response does not provide a native
book timestamp, so that leg is labeled `LOCAL_RECEIPT_BOUNDED` and fails the
`VENUE_SOURCE_TIMESTAMP` gate. Missing, stale, skewed, crossed, malformed, or
shallow evidence cannot emit an alert.

Fee evidence is venue-native and fail-closed:

- Kalshi: the client captures the exact bytes and SHA-256 of the canonical
  official fee-schedule PDF, requires the final URL, HTTP status, PDF content
  type, and PDF magic to match, and accepts formula constants only from a
  deliberately reviewed digest-to-formula binding. It also reads the current
  series fee tuple, event override, complete series fee history, and event fee
  history, and resolves the formula effective at the observation time. Unknown
  PDF bytes, changed or paginated histories, unresolved effective times,
  unsupported fee types, non-$1/binary contracts, and MVE/combo products fail
  closed. The shipped reviewed-digest allowlist is intentionally empty; a
  current official document must be independently reviewed and hash-bound
  before a real Kalshi fee row can become `VALID`.
- Polymarket: the client requests `/fee-rate` independently for each native
  outcome token, but retains the returned `base_fee` values only as estimates.
  CLOB V2 fees are condition-level and dynamic; `/fee-rate` alone does not bind
  the complete `r`/`e`/`to` parameters or an effective timestamp. The evidence
  is therefore labeled `POLYMARKET_FEE_ESTIMATE_ONLY` and `FAIL_CLOSED`, and is
  never used to calculate or alert on a shadow residual.

The Kalshi shadow calculator uses only the effective coefficient, multiplier,
trade rounding quantum, and balance-precision upper bound carried through the
reviewed official-document binding chain. It explicitly labels the result a
conservative upper bound, not an exact account charge, because account-class
and fee-accumulator inputs are unavailable. It assumes taker liquidity and
never assumes rebates, passive fills, or queue priority.

## Shadow residual and alerts

For both complementary two-leg directions, the monitor walks the immediately
available aggressive ask depth for the entire requested size. Partial-size
results are rejected. The calculation records gross payout, per-leg fills,
VWAP, observed spread, slippage contained in the aggressive fill cost, native
fees, settlement timing, capital-lock cost, and five separately configured
per-unit residual buffers:

- `--explicit-slippage-buffer-per-unit`;
- `--timestamp-skew-buffer-per-unit`;
- `--settlement-divergence-buffer-per-unit`;
- `--collateral-basis-buffer-per-unit`;
- `--rebalancing-withdrawal-allowance-per-unit`.

The five buffers default to `0.0` so a run's chosen assumptions remain explicit
in its manifest. The annual capital rate is configurable; lock duration is
derived from native close time plus settlement delay. A past, missing, or
incompatible settlement horizon fails closed.

An alert requires all semantic, structure, provenance, fee, freshness, skew,
and full-depth gates to pass and requires
`net_residual_per_unit >= threshold`. The initial research threshold is `0.05`.
An alert is still a shadow observation, not evidence of fillability, realized
profit, or permission to trade. `NO_VERIFIED_CANDIDATES` and
`NO_EXECUTABLE_SHADOW_EDGE` are expected valid outcomes. Missing or
under-specified fees are reported separately as `FEE_EVIDENCE_UNAVAILABLE`,
not as evidence that no economic edge exists.

## Immutable run artifacts

Each handled run is persisted in a newly and exclusively reserved directory:

```text
data/pmxt/runs/<run_id>/
  raw_pmxt.json
  candidates.jsonl
  raw_native_metadata.jsonl
  semantic_decisions.jsonl
  rejections.jsonl
  native_books.jsonl
  calculations.jsonl
  alerts.jsonl
  supporting_evidence.json   # present when run-level evidence was captured
  manifest.json
```

`supporting_evidence.json` preserves run-level evidence such as the exact
official Kalshi fee-schedule acquisition and reviewed-binding result. All
artifact byte sizes, SHA-256 hashes, and row counts are recorded in the
manifest, which is written last. The manifest also records configuration,
stage counts, terminal status, `live_eligible=false`, and
`no_order_actions=true`.

An existing run ID or artifact is never reopened for writing. If persistence
cannot complete, only the exact newly reserved run directory is removed;
sibling and prior runs are untouched. A failure before persistence begins can
correctly leave no artifact.

Valid terminal results include `NO_VERIFIED_CANDIDATES`,
`FEE_EVIDENCE_UNAVAILABLE`, `NO_EXECUTABLE_SHADOW_EDGE`, and `SHADOW_ALERTS`.
None establishes profitability. No code path in this integration sends an
order or grants live authorization.

## Frozen offline adjudication of the bounded candidate run

The bounded native-only continuation run
`20260829T222258601245Z_18398390` was reviewed offline against its hash-bound
venue-native metadata and rule clauses. The original run was not reopened or
modified. The derived immutable artifact is:

```text
data/pmxt/semantic_adjudication/runs/
  20260829T222258601245Z_18398390_adjudication_v1/
```

Its manifest SHA-256 is
`72f013cb68ba604083153c7e81b03cb2e977232f66e25f5a7c405440321ffad6`.
The result is 25 reviewed proposals: 11 `REJECTED`, 14 `NEEDS_REVIEW`, and
zero `VERIFIED_EQUIVALENT`. The offline adjudicator can only preserve or
downgrade a proposal to those two fail-closed outcomes; it cannot promote a
candidate to verified status.

The 11 rejections have explicit, hash-bound clauses from both venues proving
a material non-identity. The 14 unresolved proposals consist of nine sports
contracts missing a captured Kalshi terminal no-winner clause, three
presidential-nominee pairs missing full external terms and terminal/group
behavior, and two governor pairs whose captured Kalshi text points to
uncaptured full rules for the exact accelerated-resolution trigger. Missing
terms are treated as unknown, never inferred.

This adjudication was offline: zero network or credential reads, zero native
book requests, zero economics calculations, and zero orders. Because no
candidate is verified, no shadow residual or alert was eligible to be
calculated. The result establishes neither an executable edge nor
profitability.

### Superseding official-rule review

The 14 unresolved rows were subsequently reviewed against a separate,
immutable official-source capture:

```text
data/pmxt/rule_evidence/runs/
  20260829T235900000000Z_rule_evidence_v1/
```

That capture contains eight exact official resources, their raw entity bytes,
allowlisted provenance headers, receipt-time bounds, native identifiers, and
hash-bound candidate/resource indexes. Its manifest SHA-256 is
`90ae288640416381ca515e4cb30f8da4862497c4d2b9c32213a0155b8e2119f7`.
The offline terminal review is:

```text
data/pmxt/rule_review/runs/
  20260830T002000000000Z_rule_review_v1/
```

Its manifest SHA-256 is
`790e8e37fe5bae2491f71ed4fe1613ce916747335a62534ef978e33b79c678e6`.
It carries forward the 11 prior rejections and rejects all 14 reviewed rows,
leaving 25 `REJECTED`, zero `NEEDS_REVIEW`, and zero
`VERIFIED_EQUIVALENT`. The material mismatches are nine sports payout and
terminal-treatment mismatches, three strict native-close mismatches of 15
hours, and two governor resolution-source, deadline, and fallback mismatches.
Unresolved nominee Rulebook, authority, and grouped-`Other` details remain
recorded as evidence gaps; they cannot rescue the already-dispositive native
close mismatch under the frozen strict-identity policy.

This superseding review was also offline and requested no PMXT data, books,
fees, positions, or credentials. Since every proposal is rejected, native-book
capture and shadow economics are ineligible and correctly skipped. The sealed
review status is `OFFLINE_RULE_REVIEW_COMPLETE_ALL_CANDIDATES_REJECTED`; its
equivalent monitor outcome is `NO_VERIFIED_CANDIDATES`, not
`NO_EXECUTABLE_SHADOW_EDGE`. Neither establishes profitability.

### Candidate-quality boundary

The candidate-quality pass does not promote PMXT catalog fields into semantic
authority. Before venue-native evidence exists, the only hard filters are
structural: the pair must be Kalshi/Polymarket, have one unambiguous direct raw
match, use the exact `identity` relation, and meet the bounded confidence gate.
Conflicting PMXT titles, descriptions, outcome labels, resolution dates, or
source metadata remain non-authoritative diagnostics; the proposal stays
`UNVERIFIED` and `PENDING_REVIEW`.

Polarity, deadline and timezone, settlement authority, resolution source,
resolution criteria, void/cancel treatment, and material edge cases are gated
only after both native records and their provenance have been validated. The
semantic verifier accumulates all supported mismatch reasons. Invalid or
incomplete provenance takes precedence over an apparent mismatch and yields
`NEEDS_REVIEW`, never a provenance-free rejection. Books remain prohibited
unless the complete verifier returns `VERIFIED_EQUIVALENT`.

The terminal 25-row cohort also has a separate, derived rejection taxonomy at
`data/pmxt/rejection_taxonomy/runs/`
`20260830T0118521342262Z_rejection_taxonomy_v1/`. Its manifest SHA-256 is
`c5ed3c614f82091ff4f042f8d1bb285121ab82ce09d04f4edc7f98585d705366`.
It duplicates no native response bodies and binds the monitor,
offline-adjudication, official-rule-evidence, and terminal-review manifests and
rows by path, size, and SHA-256. Its mutually exclusive categories are
reporting diagnostics, not reusable semantic truth or permission to reject a
future market without fresh venue-native verification. The derived run is
offline, exclusive-create, non-overwriting, and keeps every row
`live_eligible=false` with zero books, economics, orders, and profitability
claims. That exclusive-create guarantee covers a trusted local workspace and
ordinary cooperative concurrency, including a pre-existing final path or link.
It is not an adversarial shared-filesystem guarantee: the writer does not defend
against another process swapping an already-validated parent directory for a
symlink or junction during sealing. Do not select an untrusted or concurrently
administered output root.

For this frozen cohort, the mutually exclusive reporting categories are nine
sports terminal/no-winner payout mismatches, four presidential race-call versus
inauguration mismatches, three Republican nominee strict-close mismatches, two
Israel prime-minister alternate-election/cutoff mismatches, two House-control
determination mismatches, two MLB-award multiple-winner tie-break mismatches,
two governor trigger/deadline/fallback mismatches, and one Brazil
settlement-source fallback mismatch. The counts sum to 25. Reason-code
incidence is reported separately because one candidate can have multiple
supported reasons; the three nominee rows also retain unresolved rule,
authority, and grouped-`Other` evidence gaps despite their dispositive strict
close-time mismatch.

A synthetic offline replay guards the future funnel with seven proposals: two
fully provenance-bound identity fixtures in opposite venue orientations, one
direct `overlap` relation rejected at normalization, three independently bound
native mismatches for deadline, settlement authority, and resolution source,
and one derived polarity flip whose unchanged raw native outcomes force
`NEEDS_REVIEW` rather than a provenance-free rejection. The replay deliberately
stops after semantic verification; its verified synthetic fixtures do not
request books or fees and cannot create calculations or alerts. It is
regression evidence for control flow only, not evidence that any real
proposition is equivalent or profitable.

The older `semantic_equivalence/runs/capture_001` remains immutable but is not
an input to this terminal review. It predates the strict response-header
allowlist and its frozen source hash was invalidated by the subsequent client
hardening (`trust_env=false`, header filtering, and response-cookie clearing).
It must not be treated as current authoritative evidence. The legacy module is
mechanically quarantined: production network capture is disabled, it is not a
supported export or CLI, and generic indexer discovery imports only modules
that statically declare an `Indexer` subclass. Its parser remains usable only
with exact in-memory fixture transports for regression tests.

Capture accounting includes one disclosed boundary incident. Before the
controlled official-rule capture, a fixture helper was mistakenly run outside
pytest and made four unauthenticated public GETs: the three Kalshi rule PDFs
and one Polymarket Gamma event. It stopped on a synthetic membership mismatch;
no PMXT request, credential, book, fee, position, order, or production artifact
was involved. Its temporary output was removed. The governed rule-evidence
capture was still invoked exactly once for its eight declared resources, but
the total official-source GET count for the work session is therefore twelve,
not eight. The four out-of-protocol calls are a process-boundary violation and
are not presented as governed capture evidence.

## Isolated acquisition-control prototype

`src/indexers/pmxt/acquisition.py` is explicitly `PROTOTYPE_ONLY=True` and
`PRODUCTION_READY=False`. It is intentionally not imported or exported by
`pmxt-sync`, `pmxt-monitor`, native-only continuation, or the rule-evidence
capture. It must not be wired into any of those paths without a separate
security and durability review.

The prototype only validates caller-supplied local documents and writes local
claim/receipt files. Those checks are not network or live authorization, are
not provider-authoritative billing or zero-dollar evidence, do not control the
actual HTTP transport at most once, and do not prove crash-safe state
transitions. In particular, its current filesystem model does not close all
claim/root mutation, duplicate-permit, transition-race, or crash-durability
gaps. Its metadata receipts also cannot prove which request bytes were sent or
which response bytes were received. A prototype test passing is therefore not
an operational go/no-go decision and does not authorize a PMXT request.

## Primary venue references

- [Kalshi market-data quick start](https://docs.kalshi.com/getting_started/quick_start_market_data)
- [Kalshi series metadata](https://docs.kalshi.com/api-reference/market/get-series)
- [Kalshi series fee changes](https://docs.kalshi.com/api-reference/exchange/get-series-fee-changes)
- [Kalshi event metadata](https://docs.kalshi.com/api-reference/events/get-event)
- [Kalshi event fee changes](https://docs.kalshi.com/api-reference/events/get-event-fee-changes)
- [Kalshi order-book updates](https://docs.kalshi.com/websockets/orderbook-updates)
- [Kalshi official fee schedule](https://kalshi.com/docs/kalshi-fee-schedule.pdf)
- [Kalshi fee rounding](https://docs.kalshi.com/getting_started/fee_rounding)
- [Polymarket fee rate](https://docs.polymarket.com/api-reference/market-data/get-fee-rate)
- [Polymarket CLOB market info](https://docs.polymarket.com/api-reference/markets/get-clob-market-info)
- [Polymarket CLOB V2 migration](https://docs.polymarket.com/v2-migration)
- [Polymarket fees](https://docs.polymarket.com/trading/fees)
- [Polymarket order books](https://docs.polymarket.com/api-reference/market-data/get-order-books-request-body)
