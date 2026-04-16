# Takion StatArb

**Statistical Arbitrage Trading Bot — `dataSrv` TCP Client**

This project is a Python-based statistical arbitrage trading system that connects to the `dataSrv` Takion OMS extension via a TCP/JSON protocol. It implements a daily-gap mean-reversion strategy on groups of correlated equities, executing orders through the Takion OMS in real time.

- **Platform:** Python 3, asyncio, Windows/macOS
- **OMS Backend:** Takion via `dataSrv` TCP gateway (see server-side documentation)
- **Strategy Class:** Statistical Arbitrage on correlated equity groups (SADaily)

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Repository Layout](#2-repository-layout)
3. [Architecture](#3-architecture)
4. [Strategy Description](#4-strategy-description)
5. [TCP Protocol (Client Side)](#5-tcp-protocol-client-side)
6. [Configuration Reference](#6-configuration-reference)
7. [Data Pipeline](#7-data-pipeline)
8. [Running the Bot](#8-running-the-bot)
9. [Dependencies](#9-dependencies)
10. [Known Limitations](#10-known-limitations)
11. [Quick Reference](#11-quick-reference)

---

## 1. System Overview

The bot operates in two distinct phases:

### 1.1 Offline Phase (Nightly / Pre-session)

Before the trading session, the system:

1. Downloads and refreshes daily OHLCV data from Stooq.
2. Computes daily gap patterns and inter-symbol correlations across a 2 000-symbol universe.
3. Simulates the SADaily strategy over the historical window and scores each candidate group by Sharpe ratio and expected P&L.
4. Persists the selected groups to disk — one JSON file per group — for use by the live session.

### 1.2 Live Phase (Trading Session)

During the session, the bot:

1. Connects to `dataSrv` on the configured TCP port (default `11111`).
2. Authenticates with a logon message and obtains a session key.
3. Subscribes to real-time L1 market data for all symbols in the selected groups.
4. Runs a thread per group; each thread monitors RSI-based signals and submits limit orders through `dataSrv` when entry conditions are met.
5. Manages exits on P&L targets, time-based rules, or end-of-session drainage.

---

## 2. Repository Layout

```
takionStatArb/
└── src/
    └── prototyping/
        └── statarbongroups_bot/
            ├── main.py            Entry point; CLI action dispatcher
            ├── trader.py          Trader class: market-data state, order callbacks, strategy loader
            ├── startt.py          Async TCP client; dataSrv protocol handler
            ├── s_sadaily.py       SADaily strategy runner and group thread (~950 lines)
            ├── s_common.py        GroupData shared data structure
            ├── analysis.py        Symbol universe filtering and group selection algorithm
            ├── mdaccess.py        Offline data loading: Stooq CSV → Parquet
            ├── ymd.py             Real-time quote provider (yfinance wrapper)
            ├── tkconsts.py        dataSrv protocol constants (message IDs, field names)
            ├── tkmessages.py      JSON message builder helpers
            ├── basesettings.py    Strategy hyperparameters and universe settings
            ├── localsettings.py   Local filesystem paths (data root, strategy root)
            ├── common_math.py     Vectorised mathematical utilities (Numba JIT)
            ├── time_util.py       Timezone and trading-hours helpers
            └── reporting.py       Console colour/formatting utilities
```

---

## 3. Architecture

### 3.1 Runtime Threads

```
┌──────────────────────────────────────────────────────┐
│  Main Thread                                         │
│  asyncio event loop                                  │
│  startt.start_interaction()                          │
│    ├─ logon / subscribe                              │
│    ├─ recv loop → trader.process_md_message()        │
│    └─ keep-alive heartbeat every 30 s                │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  SADailyStrategyRunner  (Thread)                     │
│  Polls all groups every SADAILY_UPDATE_INTERVAL (20s)│
│  ├─ SADailyStrategyGroup-1  (Thread)                 │
│  ├─ SADailyStrategyGroup-2  (Thread)                 │
│  └─ …                                                │
└──────────────────────────────────────────────────────┘
```

`trader.py` is the bridge: it maintains a live price cache populated by the asyncio receiver, and exposes `get_realtime_price()` / `get_previous_close()` to the strategy threads. Order submission is dispatched back to the asyncio loop from strategy threads via a thread-safe queue.

### 3.2 Key Classes

| Class | File | Purpose |
|---|---|---|
| `Trader` | `trader.py` | Central state hub. Holds price cache, processes market data and order callbacks, loads and starts the strategy runner. |
| `start_interaction()` | `startt.py` | Async coroutine. Manages the full TCP session lifecycle with `dataSrv`. |
| `SADailyStrategyRunner` | `s_sadaily.py` | Thread supervisor. Loads group definitions from disk; spawns and monitors `SADailyStrategyGroup` threads. |
| `SADailyStrategyGroup` | `s_sadaily.py` | Per-group strategy thread. Implements the state machine: `preSession → inSession → postSession`. |
| `GroupData` | `s_common.py` | Shared mutable state for one group (positions, P&L, RSI values). |
| `select_symbols_daily()` | `analysis.py` | Offline function. Runs the full universe scan and persists selected groups. |
| `MDProviderImpl` | `ymd.py` | yfinance adapter. Provides real-time quotes and symbol metadata (sector, short %). |
| `load_stooq_daily()` | `mdaccess.py` | Ingests Stooq CSV files and converts them to Parquet. |

---

## 4. Strategy Description

### 4.1 Group Construction

Groups are constructed from the daily gap time series:

```
gap[t] = open[t] - close[t-1]
rsi[t] = gap[t] / rolling_amplitude[t]
```

Symbols are grouped by pairwise correlation of their normalised gap series. A group must contain between `MIN_GROUP_SIZE` (5) and `MAX_GROUP_SIZE` (35) symbols. Groups are scored in simulation by:

- **Sharpe ratio** of the strategy P&L over the lookback window
- **Expected P&L** per session (mean of simulated session returns)

Overlapping groups (sharing more than a threshold of members) are deduplicated. The final selection is written under `GSA_ROOT/<date>/<var>/`.

### 4.2 Pre-Session Signal

At session open (or in the pre-market window), each group computes the current RSI deviation for every member:

```
z_score[sym] = (rsi[sym] - mean(rsi[group])) / std(rsi[group])
```

A symbol is a candidate to **short** when `z_score > +SADAILY_FLUCTUATION` (overbought relative to group) and a candidate to **buy** when `z_score < -SADAILY_FLUCTUATION` (oversold).

### 4.3 Execution Modes

Two modes are selected per group at group-construction time:

| Mode | Label | Logic |
|---|---|---|
| Multi-stock | `'A'` | Enter up to 4 extreme symbols independently (long the weakest, short the strongest). Each position is managed separately. |
| Pair | `'B'` | Strictly long the single weakest symbol and short the single strongest. Exit when the spread converges to a fraction of the predicted target. |

### 4.4 Exit Rules

| Trigger | Action |
|---|---|
| `pnl >= target × TARGET_PNL_MIN (0.25)` | Reduce position to lock in partial gain |
| `pnl >= target × TARGET_PNL_RED (0.80)` | Close position entirely |
| End-of-session (`postSession` state) | Market-close all remaining positions |
| P&L stop-loss threshold | Close group and mark as inactive for the day |

### 4.5 State Persistence

Each group's intraday state (positions, entry prices, realised P&L, current mode) is serialised to `<group>_state.json` after every update cycle. On restart the runner reloads this file and resumes from the last saved state without re-entering already-filled positions.

---

## 5. TCP Protocol (Client Side)

Communication with `dataSrv` uses JSON messages delimited by `\n\n`. All constants are in `tkconsts.py`; message builders are in `tkmessages.py`.

### 5.1 Session Lifecycle

```
Client                             dataSrv
  │── logon ────────────────────────▶ │
  │◀─ logon (sessionKey) ─────────── │
  │── subscribe (symbols[]) ────────▶ │
  │◀─ marketData (snapshots) ──────── │  ← repeated on every quote update
  │── orderRequest ──────────────────▶ │
  │◀─ orderResponse (OK / FAIL) ───── │
  │◀─ orderReport (fill) ──────────── │
  │── keepAlive ────────────────────▶ │  ← every 30 s
  │◀─ heartbit ────────────────────── │
```

### 5.2 Logon

```json
{
  "messageId": "logon",
  "seqNo": 1,
  "accountId": "WBTE",
  "clientVersion": "1.2.3.4"
}
```

Server responds with a `sessionKey` that must accompany all subsequent messages.

### 5.3 Subscribe

```json
{
  "messageId": "subscribe",
  "seqNo": 2,
  "sessionKey": "...",
  "symbols": ["AAPL", "TSLA", "NVDA"]
}
```

### 5.4 Order Request

```json
{
  "messageId": "orderRequest",
  "seqNo": 100,
  "sessionKey": "...",
  "order": {
    "cid": "order-42",
    "data": {
      "symbol":    "AAPL",
      "side":      "B",
      "size":      200,
      "limit":     189.44,
      "TIF":       53,
      "accountId": "WBTE"
    }
  }
}
```

> **Note:** sell-side uses `"H"` (not `"S"`) as the `side` value — see `tkconsts.py`. TIF `53` is the extended-hours constant.

### 5.5 Incoming Messages Handled by `trader.py`

| `messageId` | Handler | Action |
|---|---|---|
| `marketData` | `process_md_message()` | Updates the price cache for all symbols in the payload. |
| `orderReport` | `process_order_report()` | Records fill price, updates realised P&L, unlocks the waiting strategy thread. |
| `orderResponse` | (inline in startt.py) | Logs OK/FAIL; maps `sendId` back to `cid`. |
| `news` | (logged only) | Not currently used for signal generation. |

---

## 6. Configuration Reference

### 6.1 `basesettings.py` — Strategy Hyperparameters

| Constant | Default | Description |
|---|---|---|
| `N_ACTIVE_SYMBOLS` | `2000` | Universe size used during group selection. |
| `MIN_GROUP_SIZE` | `5` | Minimum symbols per group. |
| `MAX_GROUP_SIZE` | `35` | Maximum symbols per group. |
| `GROUP_Q_LIST` | `[0.55, 0.75, 0.90, 0.95, 0.975]` | Correlation quantile thresholds for grouping. |
| `DATA_LEN` | `756` | Lookback window in trading days (3 years). |
| `SADAILY_UPDATE_INTERVAL` | `20` | Strategy poll interval in seconds. |
| `SADAILY_FLUCTUATION` | `0.65` | Z-score threshold for entry signal. |
| `TARGET_PNL_MIN` | `0.25` | Fractional target for partial exit. |
| `TARGET_PNL_RED` | `0.80` | Fractional target for full exit. |
| `EQ_WHITE` | `[...]` | Symbol whitelist that bypasses normal scoring filters. |
| `SYMBOL_BLACKLIST` | `['nail', 'bitf', 'rnam']` | Symbols excluded from all groups. |

### 6.2 `localsettings.py` — Filesystem Paths

| Variable | Default | Description |
|---|---|---|
| `MD_ROOT` | `~/dev/ita/data` | Root data directory. |
| `MD_DAILY` | `~/dev/ita/data/daily` | Daily OHLCV Parquet files. |
| `STRATEGY_ROOT` | `~/dev/ita/data/strategies` | Parent directory for strategy output. |
| `GSA_ROOT` | `~/dev/ita/data/strategies/gsa` | Selected groups and intraday state. |

### 6.3 `tkconsts.py` — Connection Settings

| Constant | Default | Description |
|---|---|---|
| `ACCOUNT_ID` | `'WBTE'` | Takion account ID sent in logon and order requests. |
| `VERSION_OF_APP` | `'1.2.3.4'` | Client version string sent at logon. |
| `EXTENDED` | `53` | TIF code for extended-hours orders. |

The `dataSrv` host and port are configured in `startt.py` (`host` parameter, default port `11111`).

---

## 7. Data Pipeline

### 7.1 Load Daily Data

Downloads and converts Stooq CSV files to Parquet:

```bash
python main.py -A load_stooq_daily
```

Output: per-symbol Parquet files under `MD_DAILY/`.

### 7.2 Read Symbol Metadata

Fetches sector, industry, quote type, and short interest from yfinance and caches as JSON:

```bash
python main.py -A read_symbol_info -R 2026-04-14
```

### 7.3 Select Groups

Runs the full correlation scan and simulation over the universe:

```bash
python main.py -A select_symbols_daily -R 2026-04-14 -S v1
```

Output: group definitions under `GSA_ROOT/2026-04-14/v1/`.

### 7.4 Group Directory Structure

```
GSA_ROOT/<date>/<var>/
  meta.json                     # List of selected group IDs and scores
  <leader>_<nc>/                # Per-group directory
    group.json                  # Group definition: symbol list, mode, parameters
    <leader>_<nc>_state.json    # Intraday state (positions, P&L) — written during session
  I/                            # Intraday session files
  P/                            # Pre-market session files
```

---

## 8. Running the Bot

### 8.1 Prerequisites

1. `dataSrv` must be running in Takion and listening on the configured port.
2. Daily data must be current (`load_stooq_daily`).
3. Groups must have been selected for today's date (`select_symbols_daily`).

### 8.2 Start the Live Session

```bash
python main.py -A run_strategy_daily -R 2026-04-14 -S v1 -O <options>
```

The bot will:

1. Load group definitions from `GSA_ROOT/2026-04-14/v1/`.
2. Connect to `dataSrv`, log on, and subscribe to all required symbols.
3. Spawn strategy threads and begin monitoring signals.
4. Log all activity to stdout with colour-coded output.

### 8.3 CLI Arguments

| Flag | Name | Description |
|---|---|---|
| `-A` | `--action` | Required. One of `load_stooq_daily`, `select_symbols_daily`, `run_strategy_daily`, `read_symbol_info`. |
| `-R` | `--date` | Reference date in `YYYY-MM-DD` format. |
| `-S` | `--strategy-variant` | Strategy variant label (e.g. `v1`). Used as sub-directory name under `GSA_ROOT/<date>/`. |
| `-O` | `--options` | Freeform options string passed to the strategy runner. |

---

## 9. Dependencies

### 9.1 Python Packages

| Package | Purpose |
|---|---|
| `pandas` | DataFrame operations for OHLCV data and group construction. |
| `numpy` | Array maths, correlation computation. |
| `numba` | JIT compilation of inner loops in gap/RSI calculations. |
| `yfinance` | Real-time quotes and symbol metadata (sector, short %). |
| `pytz` / `python-dateutil` | Timezone-aware datetime handling. |
| `matplotlib` | Visualisation (debugging and backtesting charts). |

### 9.2 Custom / Internal Packages

| Package | Used For |
|---|---|
| `pytrade.db.calendar` | Trading calendar (market open/close days). |
| `pycommon.time.helper` | Time utility functions. |

These must be installed separately from this repository.

### 9.3 Infrastructure (Optional / Future)

The codebase references RabbitMQ (`localhost:5672`) and Redis (`localhost:6379`) in `tkconsts.py`, but neither is used in the current main execution path.

---

## 10. Known Limitations

### 10.1 Single dataSrv Account

The bot sends all orders under the single `ACCOUNT_ID` configured in `tkconsts.py`. Multi-account routing is not implemented.

### 10.2 No Reconnect Logic

If the TCP connection to `dataSrv` drops mid-session, the asyncio loop exits. The bot must be restarted manually. Intraday state is persisted to disk and will be reloaded on restart, but any fills that arrived during the outage will not be replayed.

### 10.3 yfinance Rate Limiting

Real-time price lookups via `MDProviderImpl` are rate-limited internally, but yfinance may throttle or block requests during high-frequency polling. The primary price source during a live session is the `marketData` stream from `dataSrv`; yfinance is a fallback for symbols not yet subscribed.

### 10.4 Stooq Data Dependency

The offline pipeline depends on Stooq CSV files being available at `MD_ROOT`. There is no automated download — files must be placed manually or via a separate ingestion script.

### 10.5 No TLS on TCP Channel

Communication with `dataSrv` is plain TCP with no encryption or strong authentication. The session key provides correlation, not security. Do not expose the `dataSrv` port to untrusted networks.

### 10.6 Sell Side Value

The `side` field for sell orders is `"H"`, not `"S"` (see `tkconsts.py`). This is specific to the Takion OMS convention. Using `"S"` will cause the order to be rejected.

---

## 11. Quick Reference

### CLI Cheat Sheet

```bash
# 1. Refresh daily OHLCV data
python main.py -A load_stooq_daily

# 2. Cache symbol metadata
python main.py -A read_symbol_info -R $(date +%F)

# 3. Select trading groups for today
python main.py -A select_symbols_daily -R $(date +%F) -S v1

# 4. Run live strategy
python main.py -A run_strategy_daily -R $(date +%F) -S v1
```

### Key File Map

| Need to... | Look at... |
|---|---|
| Change the trading account | `tkconsts.py` → `ACCOUNT_ID` |
| Change the dataSrv port | `startt.py` → `start_interaction(host, port)` |
| Tune entry threshold | `basesettings.py` → `SADAILY_FLUCTUATION` |
| Tune exit targets | `basesettings.py` → `TARGET_PNL_MIN`, `TARGET_PNL_RED` |
| Change data paths | `localsettings.py` |
| Add a symbol to the whitelist | `basesettings.py` → `EQ_WHITE` |
| Understand order JSON fields | `tkconsts.py`, `tkmessages.py` |
| Understand how fills are processed | `trader.py` → `process_order_report()` |
| Understand group state machine | `s_sadaily.py` → `SADailyStrategyGroup` |
| Understand the TCP session | `startt.py` → `start_interaction()` |

---

*Takion StatArb · dataSrv TCP Client · April 2026*
