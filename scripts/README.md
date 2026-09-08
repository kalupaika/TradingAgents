# Wrapper scripts

`digest.py`, `factcheck.py`, `batch.py`, and `bundle.py` wrap the core pipeline.
Full documentation — what each does, how they chain, and the local no-API
workflow — is in [../WRAPPERS.md](../WRAPPERS.md).

Quick reference:

```bash
source .venv/bin/activate

python scripts/digest.py [TICKER] [DATE]      # one-screen brief of a run
python scripts/factcheck.py [TICKER] [DATE]   # deterministic validation gate
python scripts/batch.py --watchlist wl.txt    # run a watchlist / ensemble
python scripts/bundle.py [TICKER] --mode both # package for a Claude.ai chat
```

Each takes `[TICKER] [DATE]` (default: latest run) or `--path <run dir>`, plus
`--all` and `--json`. Run with `--help` for the rest.
