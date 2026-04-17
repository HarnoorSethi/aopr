# AOPR Engine

**Adjusted OPR** — cross-event, season-wide FRC rating system.

Estimates each team's offensive contribution after factoring out the effect of defensive play. Built on a weighted sparse least-squares solver with adaptive ridge regularisation, MAD-based noise modelling, per-team match history, and a full audit trail.

---

## Quick start

```bash
# 1. Install Python deps
pip install -r requirements.txt

# 2. Set your TBA API key
cp .env.example .env
# Edit .env: set TBA_AUTH_KEY=your_key_here
# Get a key at https://www.thebluealliance.com/account

# 3. Start the backend
cd backend
uvicorn main:app --port 8000

# 4. Open the dashboard
open http://localhost:8000

# API docs (auto-generated)
open http://localhost:8000/docs
```

The first request starts a background pipeline run. Data refreshes automatically every 10 minutes. Hit **↻ Refresh** in the UI for an immediate re-solve.

---

## Architecture

```
aopr/
├── backend/
│   ├── config.py           Season config (year, thresholds, decay half-life)
│   ├── cache.py            Thread-safe SQLite (ETag cache, audit, snapshots)
│   ├── tba_client.py       TBA API client — ETag/304, stale fallback
│   ├── match_normalizer.py Replay deduplication, DQ/surrogate exclusion
│   ├── matrix_builder.py   Sparse design matrices + per-row weights
│   ├── solver.py           Weighted lsqr + adaptive ridge damping
│   ├── metrics.py          Residuals, MAD noise σ, breaker detection, variability
│   ├── refund_engine.py    Defender detection + defensive refund computation
│   ├── pipeline.py         Full season pipeline orchestrator
│   ├── models.py           Pydantic response models
│   └── main.py             FastAPI app — all routes + background refresh
├── frontend/
│   └── index.html          Dashboard (AG Grid, canvas sparklines, CDN deps)
├── tests/
│   └── test_core.py        84 unit + integration tests (no network needed)
├── requirements.txt
└── .env.example
```

---

## API reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health` | Status, stale flag, last solve timestamp |
| GET | `/api/v1/meta` | Solver diagnostics, season aggregates |
| GET | `/api/v1/season/current` | Season summary |
| POST | `/api/v1/season/refresh` | Trigger manual pipeline refresh |
| GET | `/api/v1/events` | All events, sorted by week |
| GET | `/api/v1/event/{key}/stats` | Event-level team rankings + aggregates |
| GET | `/api/v1/event/{key}/matches` | All match rows for an event |
| GET | `/api/v1/teams` | All team stats (filterable by event, role, match count) |
| GET | `/api/v1/team/{n}` | Individual team summary |
| GET | `/api/v1/team/{n}/matches` | Full match-by-match history |
| GET | `/api/v1/audit` | Match-level audit rows (refunds, residuals) |

Interactive docs at `http://localhost:8000/docs`.

---

## Metrics

| Metric | Definition |
|--------|-----------|
| **OPR** | `argmin ‖W^½(Ax − y)‖² + λ‖x‖²` — weighted sparse least-squares offensive rating |
| **DPR** | Same solve against opponent scores — measures suppression |
| **AOPR** | Re-solve on `y + refunds` — OPR after defensive credit |
| **Δ Delta** | `AOPR − OPR` — positive means the team's offense was suppressed |
| **Residual** | `(A·opr) − y` per row — positive = underperformed vs expectation |
| **Refund** | `min(DPR_surplus × multiplier × split, residual)` — capped at positive residual |
| **Defender** | `DPR > 1.5 × OPR`, ≥6 matches, above 10% of season max DPR |
| **Variability** | `std(residuals for team's rows)` — match-to-match instability |
| **Breaker** | Row with `residual > 2σ` — downweighted to 10% automatically |

---

## Weighting model

```
w = w_time × w_stage × w_event_size × w_quality

w_time       = 2^(−Δt / 21)          # 21-day half-life, clamped to max 1.0
w_stage      = 1.0 / 1.35 / 1.5      # quals / elims / finals
w_event_size = clamp(√(teams/40), 0.85, 1.35)
w_quality    = 1.0 / 0.1 / 0.0       # normal / breaker / excluded
```

---

## Solver

- **Matrix**: 2 rows per match (one per alliance perspective). Teams are columns.
- **Regularisation**: adaptive ridge via `scipy.sparse.linalg.lsqr(damp=…)`
  - Condition < 1e6 → damp = 0
  - Condition 1e6–1e8 → damp = 0.05
  - Condition ≥ 1e8 → damp = 0.2
- **Noise**: σ = 1.4826 × MAD(residuals)
- **NaN guard**: all solver outputs pass through `_safe(v)` before storage — NaN/Inf → 0.0

---

## Configuration (`backend/config.py`)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `season_year` | current year | TBA season to pull |
| `refresh_interval_minutes` | 10 | Background refresh cadence |
| `time_decay_half_life_days` | 21 | Older matches decay exponentially |
| `min_matches_to_rank` | 6 | Below this → low-match warning |
| `defender_threshold_multiplier` | 1.5 | DPR/OPR ratio to flag defender |
| `oppr_breaker_sigma` | 2.0 | Breaker downweight threshold |

---

## Running tests

```bash
cd aopr
python3 tests/test_core.py          # no network or API key needed
# or, with pytest installed:
python -m pytest tests/ -v
```

84 tests across 13 test classes:

| Class | Tests | Covers |
|-------|-------|--------|
| `TestSolver` | 4 | lsqr solve, adaptive damping, zero-weight rows |
| `TestMetrics` | 7 | residuals, MAD sigma, breaker mask, variability, match counts |
| `TestDefenderDetection` | 5 | threshold, significance floor, match count gate |
| `TestRefundEngine` | 5 | no-defender case, residual cap, negative residual, audit fields |
| `TestMatchNormalizer` | 7 | replay dedup, DQ/surrogate exclusion, score validation |
| `TestMatrixBuilder` | 8 | shape, complement property, excluded/breaker weights |
| `TestMiniPipeline` | 2 | AOPR ≥ OPR for suppressed teams, determinism |
| `TestTimeWeightClamp` | 6 | future clamp, half-life, monotone decay |
| `TestMatchRowIndex` | 7 | per-team history, timestamps, side labels, refunds, defenders |
| `TestSafeFloat` | 9 | NaN, Inf, None, normal values |
| `TestCacheThreadSafety` | 6 | read/write concurrency, snapshot pruning |
| `TestRefreshJitter` | 2 | jitter range, worker spread |
| `TestMainHelpers` | 7 | event filter, rank rows, aggregates, dedup |
| `TestEndToEndPipeline` | 9 | full numeric pipeline, all outputs finite, deterministic |

---

## Thread safety

- SQLite in WAL mode (concurrent readers, serialised writers)
- `threading.Lock` wraps all writes via `_writer()` context manager
- `busy_timeout=30000` in SQLite + 30-second httpx timeout
- Background refresh uses `asyncio.Lock` to prevent overlapping pipeline runs
- Multi-worker jitter: `random.uniform(0, 60)` seconds before first refresh

---

## Known limitations

- `team_match_rows` (per-team match history) is rebuilt in memory on every pipeline run and is not persisted to SQLite. It returns `[]` before the first pipeline completes after a cold start.
- Defender detection requires a naturally unbalanced schedule to produce DPR ≠ OPR. On very small or perfectly round-robin synthetic datasets the solver produces OPR ≡ DPR.
- The global season model does not partition by district in v1. Use the `event_key` filter on `/api/v1/teams` for per-event views.
