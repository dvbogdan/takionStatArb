# CLAUDE.md — takionStatArb

Python client side of the dataSrv ↔ takionStatArb TCP/JSON link. Statistical-arb trading bot (SADaily) over correlated equity groups. README.md has the full onboarding doc; this is the short version for working sessions.

## What is this repo

Single Python package, prototyping-grade. Two phases:

- **Offline** — pulls Stooq/Polygon daily OHLCV, runs correlation/Sharpe selection, persists groups to disk.
- **Live** — connects to `dataSrv` on `:11111`, subscribes to L1 for selected symbols, sends limit orders via SADaily strategy.

Source layout is flat under [src/prototyping/statarbongroups_bot/](src/prototyping/statarbongroups_bot/). No tests, no package metadata — entry point is [main.py](src/prototyping/statarbongroups_bot/main.py) via argparse.

## Key files

| Path | What's in it |
|---|---|
| [src/prototyping/statarbongroups_bot/main.py](src/prototyping/statarbongroups_bot/main.py) | argparse entry — `-A action` switches between offline jobs and `run_strategy_daily` |
| [src/prototyping/statarbongroups_bot/trader.py](src/prototyping/statarbongroups_bot/trader.py) | `MainTrader` singleton, owns strategies, MD state, drives `start_interaction(host)` |
| [src/prototyping/statarbongroups_bot/startt.py](src/prototyping/statarbongroups_bot/startt.py) | asyncio TCP client — logon, subscribe, keepalive, `handle_server` read loop |
| [src/prototyping/statarbongroups_bot/tkmessages.py](src/prototyping/statarbongroups_bot/tkmessages.py) | JSON message factories (logon, subscribe, order_request, …) |
| [src/prototyping/statarbongroups_bot/tkconsts.py](src/prototyping/statarbongroups_bot/tkconsts.py) | All wire-protocol field/value constants — single source of truth on client side |
| [src/prototyping/statarbongroups_bot/s_sadaily.py](src/prototyping/statarbongroups_bot/s_sadaily.py) | `SADailyStrategyRunner` — entry/exit/RSI logic, group state |
| [src/prototyping/statarbongroups_bot/analysis.py](src/prototyping/statarbongroups_bot/analysis.py) | Offline group selection / scoring |
| [src/prototyping/statarbongroups_bot/mdaccess.py](src/prototyping/statarbongroups_bot/mdaccess.py) | Stooq/Polygon downloaders, symbol info |
| [src/prototyping/statarbongroups_bot/common_math.py](src/prototyping/statarbongroups_bot/common_math.py) | 3.7k lines of math/indicator helpers — don't read whole, grep for what you need |
| [src/prototyping/statarbongroups_bot/localsettings.py](src/prototyping/statarbongroups_bot/localsettings.py) | Local overrides — don't commit secrets here |

## Protocol contract (must match takionTCP/dataSrv)

Reference doc: [../takionTCP/CLAUDE.md](../takionTCP/CLAUDE.md).

- Frame delimiter: `b'\n\n'` after each JSON object.
- Client sends `messageId`: `logon`, `keepAlive`, `subscribe`, `orderRequest`.
- Client receives `messageId`: `logon`, `keepAlive`, `subscribe`, `marketData`, `orderResponse`, `orderReport`, `orderStatus`, `news`.
- Reply matching: server echoes `refNo == client.seqNo`. Client today does **not** verify refNo.
- Symbols sent as UPPERCASE (`sym.upper()` in [trader.py:56](src/prototyping/statarbongroups_bot/trader.py)); server keys by exact string.

## Branches in flight (none merged)

- `fix/startt-tcp-framin` — **open, ready to merge into `main`**. Replaces byte-by-byte `read(1)` busy-loop with `readuntil(b'\n\n')`, raises `asyncio.open_connection(limit=4 MiB)`, adds try/except in `reply()`, drops dead `exit()` + call to nonexistent `Trader.update_account_information`. Independent of server-side changes — can merge first.
  - Note: branch name has a typo (`framin` vs `framing`). Rename before push or accept as-is.

## House rules

- Python 3, asyncio. No type-check pipeline; types are advisory.
- Wire-protocol strings live in [tkconsts.py](src/prototyping/statarbongroups_bot/tkconsts.py) — **never inline JSON keys** in handlers. If you add a field, add the constant.
- All TCP writes go through `send_tcp_message` ([startt.py:48](src/prototyping/statarbongroups_bot/startt.py)) so timeout/framing is uniform.
- Don't add new dependencies casually — this is "prototyping" code shipped as-is; install steps are manual.
- `print()` is the logger. No structured logging yet.
- `localsettings.py` is for local overrides; never commit credentials/IPs there casually.
- Commit messages: descriptive sentences, no conventional-commit prefixes (`git log main`).

## Known booby-traps

- `trader.send_order_log_to_mq(...)` is called from [startt.py:81](src/prototyping/statarbongroups_bot/startt.py) but **doesn't exist** on `Trader` — first order today raises `AttributeError`. Add a stub before exercising the order path.
- `Trader.update_account_information` was called from `handle_logon` on `main` and also doesn't exist. Already removed on `fix/startt-tcp-framin`.
- `SELL = 'H'` in [tkconsts.py:81](src/prototyping/statarbongroups_bot/tkconsts.py) is wrong — server treats `'S'` as sell/short (preBorrow path). Anything routed through SELL today goes long-side codepath.
- Host hardcoded twice: [startt.py:239](src/prototyping/statarbongroups_bot/startt.py) and [trader.py:96](src/prototyping/statarbongroups_bot/trader.py). Keep in sync until extracted.
- `send_keepalive` calls `start_connection()` on failure ([startt.py:159, 163](src/prototyping/statarbongroups_bot/startt.py)) which itself starts a new `asyncio.gather(handle_server, send_subscribe, send_keepalive)` — recursive nesting per reconnect.
- `start_connection` doesn't `return` after all `N_ATTEMPTS` fail — falls through to `send_logon(None)` and crashes ungracefully.
- Symbol-case mismatch: client upper-cases on subscribe but doesn't upper-case `item['symbol']` on receive ([trader.py:22-32](src/prototyping/statarbongroups_bot/trader.py)). `previousClose` may never fill if server echoes a different case.

## Don't-touch list

- [common_math.py](src/prototyping/statarbongroups_bot/common_math.py) — large legacy file, only edit if you really need a specific function changed.
- Historical Stooq/Polygon download caches (off-repo) — never commit raw market data.

## Work plan (open items, priority order)

Numbering coordinates with [../takionTCP/CLAUDE.md](../takionTCP/CLAUDE.md).

### P0 — protocol/contract breakage

1. **Merge `fix/startt-tcp-framin` into `main`.** Independent of server work; unblocks pipelined-message reception once the server-side liveness branch lands.
2. **Fix `SELL = 'H'` → `'S'`** in [tkconsts.py:81](src/prototyping/statarbongroups_bot/tkconsts.py). Audit all uses; `SELL_CODE='84'` is the ASCII code for `'T'` so likely also wrong — verify against server `simple_side()` mapping.
3. **Heartbeat name alignment.** Currently client sends `keepAlive` only; server has a separate `heartbit` handler. Decide with server team: drop `heartbit` server-side, or have client send both. Update [tkconsts.py](src/prototyping/statarbongroups_bot/tkconsts.py) `KEEP_ALIVE` accordingly.
4. **Honour multi-account.** Once server reads `accountId` from logon, make sure the bot sends a real account ID, not the constant `ACCOUNT_ID = 'WBTE'` from [tkconsts.py:9](src/prototyping/statarbongroups_bot/tkconsts.py).

### P1 — client bugs

5. **Add missing Trader methods.** `send_order_log_to_mq`, and any other called-but-undefined methods. Grep `startt.py` for `trader\.` and verify each exists on `Trader`.
6. **Fix `start_connection` reconnect logic.** No recursion from `send_keepalive`; centralise reconnect in a single outer loop. Return early if all attempts fail.
7. **Symbol case consistency.** Either upper-case at ingress in `process_md_message`, or stop upper-casing at egress — pick one and document.
8. **Extract host to config.** Move `10.101.3.83` out of both `startt.py` and `trader.py` into `tkconsts.py` or `localsettings.py`.
9. **Verify `refNo` on responses** at least for `logon` (single-shot) — silent mismatches today are invisible.

### P2 — hardening

10. Replace `print` with `logging` (stdlib) — gives levels and timestamps.
11. Bump `VERSION_OF_APP` whenever a wire-protocol-affecting change ships so server logs are diagnosable.
12. Consider a tiny `pytest` smoke that round-trips the message factories in `tkmessages.py` against a fixture of expected JSON shapes.

## Cross-repo coordination

When changing the wire protocol:

- Update [tkconsts.py](src/prototyping/statarbongroups_bot/tkconsts.py) and [tkmessages.py](src/prototyping/statarbongroups_bot/tkmessages.py) in the same PR as any server-side change in [../takionTCP/](../takionTCP/).
- Re-check [../takionTCP/CLAUDE.md](../takionTCP/CLAUDE.md) §"Protocol contract" — the two files must agree on `messageId` names, field names, and case rules.
- Bump `VERSION_OF_APP` so [server-side req_proc.cpp](../takionTCP/dataSrv/Ultimate/src/req_proc.cpp) logon log shows the new contract version.
