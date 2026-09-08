# Wrapper scripts

Tools that sit around the core pipeline. All are stdlib + the project's existing
deps, take `[TICKER] [DATE]` (default: most recent run) or `--path <run dir>`,
and read runs from `$TRADINGAGENTS_RESULTS_DIR` / `~/.tradingagents/logs`.

Run them from the project venv:

```bash
source .venv/bin/activate
python scripts/<tool>.py --help
```

## The loop

```
batch.py   run the pipeline over a watchlist / several model configs
   │
   ├─ digest.py     one-screen brief: decision, trader vs risk override, flags
   ├─ factcheck.py  deterministic gate: indicators recomputed from OHLCV,
   │                look-ahead scan, internal-consistency checks
   └─ bundle.py     package the reports + digest + fact-check into one prompt
                    for a Claude.ai chat (audit the claims / second-opinion call)
```

| Tool | What it does | Network |
|---|---|---|
| **digest.py** | Collapses the seven report files into a 5-line brief. `--all` for one line per run, `--json` for machine output. | no |
| **factcheck.py** | Recomputes RSI / MACD / SMA / ATR / price range as of the analysis date and diffs them against the reports; scans every report for dates after the analysis date; cross-reads the reports for signal-vs-decision divergence, over-confident language, and un-synthesised debates. Exit non-zero on any ERROR. | yfinance |
| **batch.py** | Runs the full pipeline across many tickers and/or model profiles. `--profiles a,b` reports per-ticker agreement; `--repeat N` reports run-to-run stability. Flushes after every run so Ctrl-C is safe. `--dry-run` to preview. | runs the pipeline |
| **bundle.py** | Assembles the reports + digest + fact-check into a markdown doc with an instruction header, copied to the clipboard. `--mode audit` (verify claims, web-search, flag look-ahead) or `--mode decide` (independent Buy/Hold/Sell). | via factcheck |

## Typical use

```bash
# overnight: research a watchlist locally
python scripts/batch.py --watchlist watchlist.txt --repeat 2

# next morning: triage
python scripts/digest.py --all
python scripts/factcheck.py --all

# for anything that passed the gate, get a second opinion from Claude.ai
python scripts/bundle.py NVDA --mode both      # now paste into a Claude chat
```
