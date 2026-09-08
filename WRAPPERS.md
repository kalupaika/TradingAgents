# Wrapper tools

Fork-local tooling that wraps the core TradingAgents pipeline to make its output
usable day to day: a one-screen brief, a deterministic fact-check gate, a
watchlist / ensemble runner, and a packager that hands a run to a Claude.ai chat
for a web-grounded second opinion.

A fifth tool, [`fundamentals.py`](scripts/fundamentals.py), is a standalone
counterpoint to the whole pipeline — see the last section.

The wrappers live in [`scripts/`](scripts/), depend only on what the project
already installs, and share the same conventions:

- positional `[TICKER] [DATE]` (default: the most recent run), or `--path <run dir>`
- runs are read from `$TRADINGAGENTS_RESULTS_DIR`, else `$TRADINGAGENTS_HOME/logs`,
  else `~/.tradingagents/logs`
- `--json` for machine output, `--all` to sweep every run
- non-zero exit on a hard failure, so they drop into CI or a nightly job

Run them from the project venv:

```bash
source .venv/bin/activate
python scripts/digest.py --help
```

## How they fit together

```
batch.py ── runs the pipeline over a watchlist and/or several model configs
   │
   ├─ digest.py     one brief per run: decision, trader-vs-risk override, data flags
   ├─ factcheck.py  deterministic gate: indicators recomputed from OHLCV,
   │                look-ahead scan, internal-consistency checks
   └─ bundle.py     packages reports + digest + fact-check into one prompt for a
                    Claude.ai chat (audit the claims, or a second-opinion call)
```

A typical cycle:

```bash
# overnight — research a watchlist locally (Ollama, no API cost)
python scripts/batch.py --watchlist watchlist.txt --repeat 2

# next morning — triage
python scripts/digest.py --all
python scripts/factcheck.py --all

# for anything that clears the gate, get a web-grounded second opinion
python scripts/bundle.py NVDA --mode both      # doc is on the clipboard; paste into Claude.ai
```

---

## digest.py

Collapses the seven markdown reports a run produces into a five-line brief — the
part you actually read.

```bash
python scripts/digest.py                 # most recent run
python scripts/digest.py NVDA 2026-01-15
python scripts/digest.py --all           # one line per run
python scripts/digest.py NVDA --json
```

```
AAPL — 2026-09-07  ·  run ~48m
DECISION: HOLD  (Trader: BUY @ 328.0 stop 325.0 10% of portfolio — overridden by risk team)
MANAGER: Hold
WHY: The evidence presented by the analysts is materially conflicting and ambiguous …
FLAGS: ⚠ technical-analysis node produced no usable indicators (model parse failure)
       ⚠ sentiment confidence only medium
       ⚠ FRED macro data unavailable this run
```

The `FLAGS` are heuristic data-quality warnings drawn from the reports and
`message_tool.log` (ungrounded technical report, weak/sparse sentiment, FRED
disabled, rate-limited sources, missing report files).

No network.

## factcheck.py

Deterministic validation — no LLM calls. Recomputes the checkable facts from raw
OHLCV and diffs them against what the reports claim.

```bash
python scripts/factcheck.py                  # most recent run
python scripts/factcheck.py NVDA 2026-01-15
python scripts/factcheck.py --all
python scripts/factcheck.py NVDA --json
```

```
AAPL — 2026-09-07   fact-check: FAIL
  truth @ 2026-09-04 (290 bars): close 319.97  RSI-14 53.95  MACD 2.64/1.09
    50SMA 314.99  200SMA 283.44  ATR 7.63  60d range 273.51–344.27
  ✗ [grounding] market_report.md cites no indicator values — technical analyst
    produced nothing checkable (model likely failed to parse the OHLCV input)
```

Checks, against market data:

- **look-ahead** — any report referencing a date after the analysis date
- **indicators** — RSI / MACD / MACD-signal / 50-SMA / 200-SMA claims vs values
  recomputed as of the analysis date
- **overbought / oversold** language vs the actual RSI
- **price levels** — dollar figures vs the real trailing trading range
- **entry price** — the trader's entry vs the last close
- **grounding** — a technical report that cites no indicator numbers at all

And internal consistency, from the reports alone (no network):

- **signal-alignment** — final BUY/SELL vs the net directional lean of the analysts
- **calibration** — "high conviction" language over a heavily hedged rationale
- **traceability** — a final decision that never engages the bull/bear debate, or
  overrides the trader without addressing the technical case

Exit status is non-zero on any `ERROR`-level finding. Offline, the price/indicator
checks are skipped and the consistency checks still run.

Uses `yfinance` + `stockstats` (both already project deps).

## batch.py

Runs the full pipeline across many tickers, optionally across several model
profiles and/or repeated per profile.

```bash
python scripts/batch.py NVDA AAPL TSLA
python scripts/batch.py --watchlist watchlist.txt --date 2026-09-05
python scripts/batch.py NVDA --profiles qwen3-8b,qwen3-14b      # ensemble
python scripts/batch.py NVDA --repeat 3                          # stability
python scripts/batch.py --watchlist wl.txt --dry-run
python scripts/batch.py NVDA --profiles qwen3-8b,qwen3-14b --factcheck
```

- **one profile** (default: current env / `DEFAULT_CONFIG`) — one run per ticker
- **several `--profiles`** — per-ticker agreement: unanimous / split / consensus vote
- **`--repeat N`** — run-to-run stability (LLMs are non-deterministic; if two runs
  of the same config disagree on direction, the call isn't stable enough to act on)

Runs are sequential (a local Ollama model holds RAM; parallel runs thrash). Every
finished run is flushed to the output JSON immediately, so Ctrl-C or a crash keeps
everything completed so far. Built-in profiles: `qwen3-8b`, `qwen3-14b`,
`qwen3-30b`; add more with `--profile-file` (a JSON object of `{name: config}`).

```
Batch 2026-09-05  ·  3 ticker(s) × 2 profile(s) × 1 = 6 runs  ·  5 ok, 1 failed  ·  4h12m

TICKER    qwen3-8b        qwen3-14b       consensus
AAPL      HOLD            SELL            ✗ HOLD (1/2)
NVDA      BUY             BUY             ✓ BUY (2/2)
TSLA      BUY             ERROR           — BUY (1/1)
```

## bundle.py

The local pipeline (a small model such as `qwen3:8b`) does the research; this
packages a run for a Claude.ai chat — no API key — where Claude can web-search to
verify the claims or give an independent decision.

```bash
python scripts/bundle.py NVDA 2026-09-07          # audit prompt (default)
python scripts/bundle.py NVDA --mode decide       # portfolio-manager prompt
python scripts/bundle.py NVDA --mode both
python scripts/bundle.py NVDA -o nvda.md --stdout
```

The bundle (copied to the clipboard via `pbcopy` / `xclip` / `wl-copy`) contains:

1. an instruction header — `audit` (verify each claim against current web sources,
   flag fabrication and look-ahead) or `decide` (independent Buy/Hold/Sell with
   conviction and the contradictions between reports)
2. `factcheck.py`'s findings plus the recomputed indicator values, so Claude sees
   what the local technical analyst missed
3. `digest.py`'s summary of the pipeline's own decision
4. all six analyst reports, delimited

The `audit` header pins the as-of date so Claude treats every claim as
point-in-time and flags post-date information as look-ahead. Claude.ai web search
returns *today's* data, so it is most reliable for durable facts (did an event
happen, are these fundamentals real) — the numeric price/indicator claims are
already covered deterministically by `factcheck.py`.

---

## fundamentals.py — the deterministic counterpoint

Not a wrapper around a run — a different way of answering the same question.
Where TradingAgents runs a multi-agent LLM debate and lands on a Buy/Hold/Sell,
`fundamentals.py` pulls the reported financials, computes the standard ratios
itself, and renders a multi-tab HTML report that *characterises* the company as
of its last filing. No LLM, no forecast, no recommendation — every number shows
its inputs and source period, and a Data Quality tab lists what's missing or
distorted (negative-equity ROE, stale statements, a market cap yfinance dropped).

```bash
python scripts/fundamentals.py AAPL                  # writes AAPL_fundamentals.html
python scripts/fundamentals.py AAPL --mode quick     # compact text table
python scripts/fundamentals.py AAPL MSFT NVDA --mode compare
python scripts/fundamentals.py RELIANCE.NS -o r.html # US + Indian tickers
```

Tabs: Overview · Valuation (P/E, PEG, EV/EBITDA, FCF yield, …) · Profitability
(margins, ROE/ROA/ROIC) · Growth (3y/5y CAGR from the annual statements) ·
Financial health (liquidity, leverage, **Altman Z-Score** and **Piotroski
F-Score** with every component shown) · Capital returns · Data quality.

Run it next to a TradingAgents analysis of the same ticker to compare the two
approaches directly: the LLM's `fundamentals_report.md` narrative against the
computed evidence, and the pipeline's decision against a scorecard that refuses
to make one.

Data source: Yahoo Finance (`yfinance`) — annual statements + key stats.

## Local, no-API workflow

These tools were built around running the pipeline entirely on a local model via
Ollama, with a Claude.ai subscription (not the API) as the reasoning/verification
layer:

```
# .env
TRADINGAGENTS_LLM_PROVIDER=ollama
TRADINGAGENTS_QUICK_THINK_LLM=qwen3:8b     # analysts / research gathering
TRADINGAGENTS_DEEP_THINK_LLM=qwen3:14b     # manager / trader / risk / decide
```

`batch.py` gathers research overnight for free, `factcheck.py` gates the output,
`bundle.py` hands the survivors to Claude.ai for a web-grounded check. The known
weak link is the technical analyst on a small model — `factcheck.py` will fail
the run when it produces no grounded indicators.

## Tests

```bash
python -m pytest tests/test_digest.py tests/test_factcheck.py \
                 tests/test_batch.py tests/test_bundle.py -q
```

The pipeline itself is never invoked by the tests — `batch.py` takes an injected
fake runner, `factcheck.py`'s ground-truth call is not exercised, and the rest
run on fixture report directories.
