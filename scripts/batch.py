#!/usr/bin/env python3
"""Batch / ensemble runner — many tickers, optionally many model configs.

Runs the full agent pipeline across a watchlist, and — when you pass more than
one profile or --repeat > 1 — reports where the calls agree and how stable they
are run to run. Agreement across independent configs is a cheap proxy for
confidence; disagreement is your read-manually queue.

Runs are sequential (a local Ollama model holds RAM; parallel runs thrash).
Each finished run is flushed to the output JSON immediately, so a crash or
Ctrl-C keeps everything completed so far.

Usage:
  python scripts/batch.py NVDA AAPL TSLA
  python scripts/batch.py --watchlist watchlist.txt --date 2026-09-05
  python scripts/batch.py NVDA --profiles qwen3-8b,qwen3-14b        # ensemble
  python scripts/batch.py NVDA --repeat 3                            # stability
  python scripts/batch.py --watchlist wl.txt --dry-run
  python scripts/batch.py NVDA --profiles qwen3-8b,qwen3-14b --factcheck

Profiles: built-ins (qwen3-8b / qwen3-14b / qwen3-30b) plus any in a JSON file
passed via --profile-file ({"name": {"llm_provider": ..., "deep_think_llm": ...}}).
With no --profiles, one run per ticker using the current env / default config.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import digest  # noqa: E402  (sibling script)

_OLLAMA = {"llm_provider": "ollama", "backend_url": "http://localhost:11434/v1"}
BUILTIN_PROFILES: dict[str, dict] = {
    "qwen3-8b": {**_OLLAMA, "deep_think_llm": "qwen3:8b", "quick_think_llm": "qwen3:8b"},
    "qwen3-14b": {**_OLLAMA, "deep_think_llm": "qwen3:14b", "quick_think_llm": "qwen3:14b"},
    "qwen3-30b": {**_OLLAMA, "deep_think_llm": "qwen3:30b", "quick_think_llm": "qwen3:30b"},
}

# 5-tier rating -> coarse side for agreement math.
_SIDE = {
    "Buy": "BUY", "Overweight": "BUY",
    "Hold": "HOLD", "REVIEW": "HOLD", "Review": "HOLD",
    "Underweight": "SELL", "Sell": "SELL",
}


def tier_to_side(rating: str | None) -> str:
    return _SIDE.get((rating or "").strip(), "HOLD")


def _hms(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def load_watchlist(path: str) -> list[str]:
    out: list[str] = []
    for line in Path(path).expanduser().read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.extend(tok.strip().upper() for tok in line.replace(",", " ").split())
    return list(dict.fromkeys(out))  # de-dupe, keep order


def resolve_profiles(names: str | None, profile_file: str | None) -> dict[str, dict]:
    table = dict(BUILTIN_PROFILES)
    if profile_file:
        loaded = json.loads(Path(profile_file).expanduser().read_text())
        if not isinstance(loaded, dict):
            raise SystemExit("--profile-file must be a JSON object of {name: config}")
        table.update(loaded)
    if not names:
        return {"env": {}}  # current environment / default config, one run per ticker
    chosen = {}
    for n in [x.strip() for x in names.split(",") if x.strip()]:
        if n not in table:
            raise SystemExit(f"unknown profile {n!r}; known: {', '.join(sorted(table))}")
        chosen[n] = table[n]
    return chosen


def build_plan(tickers, profiles: dict[str, dict], repeat: int):
    plan = []
    for ticker in tickers:
        for pname, pcfg in profiles.items():
            for r in range(1, repeat + 1):
                plan.append({"ticker": ticker, "profile": pname, "config": pcfg, "rep": r})
    return plan


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #
def run_one(ticker: str, date: str, profile_cfg: dict, results_dir: Path) -> dict:
    """Execute one pipeline run. Returns a result dict; never raises."""
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    cfg = DEFAULT_CONFIG.copy()
    cfg.update(profile_cfg)
    cfg["results_dir"] = str(results_dir)

    started = time.time()
    try:
        ta = TradingAgentsGraph(config=cfg)
        _, signal = ta.propagate(ticker, date)
        run_dir = results_dir / ticker / date
        if not (run_dir / "reports").is_dir():  # ticker sanitised for the path
            hits = sorted(results_dir.glob(f"*/{date}/reports"))
            if len(hits) == 1:
                run_dir = hits[0].parent
        return {
            "rating": signal,
            "side": tier_to_side(signal),
            "seconds": round(time.time() - started, 1),
            "run_dir": str(run_dir),
            "error": None,
        }
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001 — batch must survive one bad run
        return {
            "rating": None, "side": None,
            "seconds": round(time.time() - started, 1),
            "run_dir": None,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=3),
        }


def execute(plan, date: str, results_dir: Path, out_path: Path,
            runner=None, echo=print) -> list[dict]:
    runner = runner or run_one  # resolved at call time so tests can monkeypatch
    results: list[dict] = []
    total = len(plan)
    show_rep = any(p["rep"] > 1 for p in plan)
    for i, item in enumerate(plan, 1):
        tag = f"{item['ticker']}/{item['profile']}" + (f"#{item['rep']}" if show_rep else "")
        echo(f"[{i}/{total}] {tag} …", flush=True)
        res = runner(item["ticker"], date, item["config"], results_dir)
        row = {**{k: item[k] for k in ("ticker", "profile", "rep")}, **res}
        results.append(row)
        _flush(out_path, {"date": date, "results_dir": str(results_dir), "runs": results})
        mark = res["error"] or f"{res['rating']} ({res['side']})"
        echo(f"      -> {mark}  [{_hms(res['seconds'])}]", flush=True)
    return results


def _flush(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def summarize(runs: list[dict]) -> dict:
    by_ticker: dict[str, list[dict]] = {}
    for r in runs:
        by_ticker.setdefault(r["ticker"], []).append(r)

    tickers = {}
    for ticker, rows in by_ticker.items():
        ok = [r for r in rows if not r["error"]]
        sides = [r["side"] for r in ok]
        profiles = {}
        for r in rows:
            profiles.setdefault(r["profile"], []).append(r)

        # stability: per profile, do repeats land on the same side?
        stability = {}
        for pname, prows in profiles.items():
            psides = [r["side"] for r in prows if not r["error"]]
            if len(psides) > 1:
                modal, n = Counter(psides).most_common(1)[0]
                stability[pname] = {"modal": modal, "agree": n, "of": len(psides)}

        # ensemble: majority side across all successful runs. "unanimous" and
        # "split" only mean something with >= 2 successful runs to compare.
        consensus, vote = (None, 0)
        if sides:
            consensus, vote = Counter(sides).most_common(1)[0]
        comparable = len(sides) >= 2
        split = comparable and vote < len(sides)

        tickers[ticker] = {
            "profiles": {
                p: [
                    {"rep": r["rep"], "rating": r["rating"], "side": r["side"], "error": r["error"]}
                    for r in rows_
                ]
                for p, rows_ in profiles.items()
            },
            "sides": sides,
            "consensus": consensus,
            "consensus_votes": f"{vote}/{len(sides)}" if sides else "0/0",
            "unanimous": comparable and not split,
            "split": split,
            "stability": stability,
            "failures": [r["profile"] + (f"#{r['rep']}" if r["rep"] > 1 else "")
                         for r in rows if r["error"]],
        }

    return {
        "runs": len(runs),
        "ok": sum(1 for r in runs if not r["error"]),
        "failed": sum(1 for r in runs if r["error"]),
        "seconds": round(sum(r["seconds"] for r in runs), 1),
        "profiles": list(dict.fromkeys(r["profile"] for r in runs)),
        "tickers": tickers,
    }


def render_summary(date: str, plan_shape: str, summ: dict) -> str:
    lines = [
        f"Batch {date}  ·  {plan_shape}  ·  "
        f"{summ['ok']} ok, {summ['failed']} failed  ·  {_hms(summ['seconds'])}",
        "",
    ]
    profile_names = summ.get("profiles") or sorted(
        {p for t in summ["tickers"].values() for p in t["profiles"]}
    )
    multi = len(profile_names) > 1

    if multi:
        header = f"{'TICKER':<10}" + "".join(f"{p:<16}" for p in profile_names) + "consensus"
        lines.append(header)
        for ticker, t in sorted(summ["tickers"].items()):
            cells = []
            for p in profile_names:
                rows = t["profiles"].get(p, [])
                if not rows:
                    cells.append("·")
                    continue
                sides = [r["side"] for r in rows if not r["error"]]
                if not sides:
                    cells.append("ERROR")
                elif len(set(sides)) == 1:
                    cells.append(sides[0] + (f"×{len(sides)}" if len(sides) > 1 else ""))
                else:
                    cells.append("/".join(sides))
            mark = "✓" if t["unanimous"] else ("✗" if t["split"] else "—")
            cons = f"{mark} {t['consensus'] or 'n/a'} ({t['consensus_votes']})"
            lines.append(f"{ticker:<10}" + "".join(f"{c:<16}" for c in cells) + cons)
    else:
        p = profile_names[0] if profile_names else "env"
        lines.append(f"{'TICKER':<10}{'RATING':<14}{'SIDE':<8}stability")
        for ticker, t in sorted(summ["tickers"].items()):
            rows = t["profiles"].get(p, [])
            ok = [r for r in rows if not r["error"]]
            rating = ok[-1]["rating"] if ok else "ERROR"
            side = ok[-1]["side"] if ok else "—"
            st = t["stability"].get(p)
            stx = f"{st['agree']}/{st['of']} on {st['modal']}" if st else ""
            lines.append(f"{ticker:<10}{str(rating):<14}{str(side):<8}{stx}")

    fails = {f"{tk}:{f}" for tk, t in summ["tickers"].items() for f in t["failures"]}
    if fails:
        lines += ["", "failures: " + ", ".join(sorted(fails))]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("tickers", nargs="*", help="ticker symbols (or use --watchlist)")
    ap.add_argument("--watchlist", help="file with tickers (one per line, # comments)")
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                    help="analysis date YYYY-MM-DD (default: today)")
    ap.add_argument("--profiles", help="comma-separated profile names (default: current env)")
    ap.add_argument("--profile-file", help="JSON file of extra {name: config} profiles")
    ap.add_argument("--repeat", type=int, default=1, help="runs per ticker/profile (stability)")
    ap.add_argument("--results-dir", help="base logs dir (default: ~/.tradingagents/logs)")
    ap.add_argument("--out", help="summary JSON path (default: <results-dir>/_batch/…)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument("--json", action="store_true", help="print the summary as JSON")
    ap.add_argument("--digest", action="store_true", help="print digest.py brief after each run")
    ap.add_argument("--factcheck", action="store_true", help="print factcheck.py after each run")
    args = ap.parse_args(argv)

    # Best-effort .env load so OLLAMA_BASE_URL / API keys are present.
    try:
        from dotenv import load_dotenv

        load_dotenv(Path.cwd() / ".env")
    except Exception:  # noqa: BLE001
        pass

    tickers = list(dict.fromkeys(
        [t.upper() for t in args.tickers]
        + (load_watchlist(args.watchlist) if args.watchlist else [])
    ))
    if not tickers:
        ap.error("no tickers given (positional args or --watchlist)")
    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        ap.error(f"--date {args.date!r} is not YYYY-MM-DD")
    if args.repeat < 1:
        ap.error("--repeat must be >= 1")

    profiles = resolve_profiles(args.profiles, args.profile_file)
    plan = build_plan(tickers, profiles, args.repeat)
    shape = (f"{len(tickers)} ticker(s) × {len(profiles)} profile(s) "
             f"× {args.repeat} = {len(plan)} runs")

    results_dir = digest.resolve_results_dir(args.results_dir)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = Path(args.out).expanduser() if args.out else (
        results_dir / "_batch" / f"batch-{args.date}-{stamp}.json"
    )

    if args.dry_run:
        print(f"plan: {shape}\ndate: {args.date}\nprofiles: {', '.join(profiles)}")
        for item in plan:
            print(f"  {item['ticker']:<8} {item['profile']}"
                  + (f" #{item['rep']}" if args.repeat > 1 else ""))
        print(f"output: {out_path}")
        return 0

    print(f"{shape}  →  {out_path}\n")
    results = execute(plan, args.date, results_dir, out_path)

    if args.digest or args.factcheck:
        _post_reports(results, results_dir, args.digest, args.factcheck)

    summ = summarize(results)
    _flush(out_path, {
        "date": args.date, "results_dir": str(results_dir),
        "plan": shape, "runs": results, "summary": summ,
    })

    print()
    print(json.dumps(summ, indent=2) if args.json else render_summary(args.date, shape, summ))
    print(f"\nfull results: {out_path}")
    return 0 if summ["failed"] == 0 else 1


def _post_reports(results, results_dir, do_digest, do_factcheck):
    seen = set()
    for r in results:
        if r["error"] or (r["ticker"], r.get("run_dir")) in seen:
            continue
        seen.add((r["ticker"], r.get("run_dir")))
        run_dir = Path(r["run_dir"]) if r.get("run_dir") else results_dir / r["ticker"]
        if do_digest:
            try:
                print("\n" + digest.render(digest.build_digest(run_dir)))
            except Exception as exc:  # noqa: BLE001
                print(f"digest failed for {r['ticker']}: {exc}")
        if do_factcheck:
            try:
                import factcheck

                print("\n" + factcheck.render(factcheck.run_factcheck(run_dir)))
            except Exception as exc:  # noqa: BLE001
                print(f"factcheck failed for {r['ticker']}: {exc}")


if __name__ == "__main__":
    sys.exit(main())
