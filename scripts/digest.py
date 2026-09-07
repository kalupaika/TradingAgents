#!/usr/bin/env python3
"""Decision digest — collapse a run's multi-page agent transcript into a short brief.

A completed TradingAgents run writes seven markdown reports under
``<results_dir>/<TICKER>/<DATE>/reports/``. This reads that folder and prints the
part you actually act on: the final decision, what the trader proposed (and
whether the risk team overrode it), a one-line rationale, and data-quality flags
so you know how much to trust the call.

Usage:
    python scripts/digest.py                     # most recent run, any ticker
    python scripts/digest.py NVDA                 # most recent NVDA run
    python scripts/digest.py NVDA 2026-01-15      # a specific run
    python scripts/digest.py --path ~/.tradingagents/logs/NVDA/2026-01-15
    python scripts/digest.py --all               # one-line summary of every run
    python scripts/digest.py NVDA --json         # machine-readable

Results dir resolution: --results-dir, then $TRADINGAGENTS_RESULTS_DIR, then
$TRADINGAGENTS_HOME/logs, then ~/.tradingagents/logs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

REPORT_FILES = (
    "market_report.md",
    "sentiment_report.md",
    "news_report.md",
    "fundamentals_report.md",
    "investment_plan.md",
    "trader_investment_plan.md",
    "final_trade_decision.md",
)

# Canonical trade directions the various agents emit, longest first so
# "Strong Buy" wins over "Buy" when both could match.
_DIRECTIONS = ("STRONG BUY", "STRONG SELL", "BUY", "SELL", "HOLD")

_CHATTY_MARKERS = (
    "let me know if",
    "what specific analysis",
    "questions to clarify",
    "i'd be happy to",
    "would you like me to",
    "feel free to ask",
    "add all dates",  # placeholder code the model emitted instead of analysis
)
_INDICATORS = ("macd", "rsi", "sma", "ema", "bollinger", "atr", "adx", "vwap", "moving average")


def resolve_results_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get("TRADINGAGENTS_RESULTS_DIR")
    if env:
        return Path(env).expanduser()
    home = os.environ.get("TRADINGAGENTS_HOME")
    if home:
        return Path(home).expanduser() / "logs"
    return Path.home() / ".tradingagents" / "logs"


def _norm_direction(text: str | None) -> str | None:
    if not text:
        return None
    up = text.upper()
    for d in _DIRECTIONS:
        if d in up:
            return d
    return None


def _first(pattern: str, text: str, flags: int = re.I) -> str | None:
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None


@dataclass
class Digest:
    ticker: str
    date: str
    run_dir: Path
    decision: str | None = None
    rating: str | None = None
    manager_reco: str | None = None
    trader_action: str | None = None
    trader_entry: str | None = None
    trader_stop: str | None = None
    trader_size: str | None = None
    overridden: bool = False
    why: str | None = None
    duration_min: int | None = None
    flags: list[str] = field(default_factory=list)
    missing_reports: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["run_dir"] = str(self.run_dir)
        return d


def _read(reports: Path, name: str) -> str:
    p = reports / name
    return p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""


def _extract_why(final_text: str) -> str | None:
    """Pull the first substantive sentence of the final decision's rationale."""
    body = final_text
    m = re.search(r"(?:rationale|reasoning)\s*:?\s*\*{0,2}\s*(.+)", final_text, re.I | re.S)
    if m:
        body = m.group(1)
    # Drop markdown emphasis / headers / list bullets, collapse whitespace.
    body = re.sub(r"[*_#>`]", "", body)
    body = re.sub(r"^\s*[-\d.]+\s+", " ", body, flags=re.M)
    body = re.sub(r"\s+", " ", body).strip()
    for sentence in re.split(r"(?<=[.!?])\s+", body):
        s = sentence.strip()
        if len(s) >= 40:
            return s if s.endswith((".", "!", "?")) else s + "."
    return body[:240] or None


def _duration_minutes(run_dir: Path) -> int | None:
    log = run_dir / "message_tool.log"
    if not log.is_file():
        return None
    stamps = re.findall(r"^(\d{2}:\d{2}:\d{2})\b", log.read_text(errors="replace"), re.M)
    if len(stamps) < 2:
        return None
    fmt = "%H:%M:%S"
    try:
        t0, t1 = datetime.strptime(stamps[0], fmt), datetime.strptime(stamps[-1], fmt)
    except ValueError:
        return None
    delta = (t1 - t0).total_seconds()
    if delta < 0:  # ran past midnight; not worth guessing the day
        return None
    return round(delta / 60)


def _quality_flags(reports: Path, run_dir: Path) -> list[str]:
    flags: list[str] = []

    market = _read(reports, "market_report.md")
    if market:
        low = market.lower()
        has_indicator_number = any(
            re.search(rf"{ind}\W{{0,20}}\d", low) for ind in _INDICATORS
        )
        if any(marker in low for marker in _CHATTY_MARKERS) or not has_indicator_number:
            flags.append("technical-analysis node produced no usable indicators (model parse failure)")

    sentiment = _read(reports, "sentiment_report.md")
    conf = _first(r"confidence\s*:?\s*\*{0,2}\s*(low|medium|high)", sentiment)
    if conf and conf.lower() in ("low", "medium"):
        flags.append(f"sentiment confidence only {conf.lower()}")
    if sentiment and sentiment.lower().count("no posts") + sentiment.lower().count("no aapl") >= 2:
        flags.append("sentiment built on sparse social data")

    news = _read(reports, "news_report.md")
    nlow = news.lower()
    if news and ("exclude" in nlow and "fred" in nlow or "tool limitation" in nlow):
        flags.append("FRED macro data unavailable this run")

    log = run_dir / "message_tool.log"
    if log.is_file():
        text = log.read_text(errors="replace").lower()
        if "no data" in text or "error" in text and "get_stock_data" in text and "rows" not in text:
            pass  # deliberately conservative; price-data failures show up in market flag
        if "429" in text or "too many requests" in text:
            flags.append("some data sources rate-limited (HTTP 429) during the run")

    return flags


def build_digest(run_dir: Path) -> Digest:
    run_dir = run_dir.expanduser().resolve()
    reports = run_dir / "reports"
    if not reports.is_dir() and (run_dir / "final_trade_decision.md").is_file():
        reports = run_dir  # caller pointed straight at the reports folder
        run_dir = run_dir.parent

    date = run_dir.name
    ticker = run_dir.parent.name
    dg = Digest(ticker=ticker, date=date, run_dir=run_dir)

    dg.missing_reports = [f for f in REPORT_FILES if not (reports / f).is_file()]

    final_text = _read(reports, "final_trade_decision.md")
    dg.decision = _norm_direction(
        _first(r"final trading decision\s*:?\s*\*{0,2}\s*([A-Za-z ]+)", final_text)
        or _first(r"final transaction proposal\s*:?\s*\*{0,2}\s*([A-Za-z ]+)", final_text)
        or final_text[:400]
    )
    dg.rating = _first(r"rating\s*:?\s*\*{0,2}\s*([A-Za-z][A-Za-z ]*)", final_text)
    if dg.rating:
        dg.rating = dg.rating.split("\n")[0].strip()
    dg.why = _extract_why(final_text)

    trader_text = _read(reports, "trader_investment_plan.md")
    # Anchor to line start so the labelled fields win over the same words
    # appearing mid-sentence in the trader's free-text "Reasoning".
    dg.trader_action = _norm_direction(
        _first(r"(?:^|\n)\s*\*{0,2}action\*{0,2}\s*:\s*\*{0,2}\s*([A-Za-z ]+)", trader_text)
        or _first(r"final transaction proposal\s*:?\s*\*{0,2}\s*([A-Za-z ]+)", trader_text)
    )
    dg.trader_entry = _first(r"(?:^|\n)\s*\*{0,2}entry price\*{0,2}\s*:\s*\*{0,2}\s*\$?([\d,.]+)", trader_text)
    dg.trader_stop = _first(r"(?:^|\n)\s*\*{0,2}stop[- ]?loss\*{0,2}\s*:\s*\*{0,2}\s*\$?([\d,.]+)", trader_text)
    dg.trader_size = _first(
        r"(?:^|\n)\s*\*{0,2}position siz(?:e|ing)\*{0,2}\s*:\s*\*{0,2}\s*([^\n*]+)", trader_text
    )

    plan_text = _read(reports, "investment_plan.md")
    dg.manager_reco = _first(r"\*{0,2}recommendation\*{0,2}\s*:?\s*\*{0,2}\s*([A-Za-z ]+)", plan_text)
    if dg.manager_reco:
        dg.manager_reco = dg.manager_reco.split("\n")[0].strip()

    dg.overridden = bool(
        dg.decision and dg.trader_action and dg.decision != dg.trader_action
    )

    dg.duration_min = _duration_minutes(run_dir)
    dg.flags = _quality_flags(reports, run_dir)
    return dg


def find_runs(results_dir: Path, ticker: str | None = None):
    """Yield (ticker, date, run_dir) for every run that has a final decision."""
    if not results_dir.is_dir():
        return
    tickers = [results_dir / ticker.upper()] if ticker else sorted(results_dir.iterdir())
    for tdir in tickers:
        if not tdir.is_dir():
            continue
        for ddir in sorted(tdir.iterdir()):
            final = ddir / "reports" / "final_trade_decision.md"
            if final.is_file():
                yield tdir.name, ddir.name, ddir


def _latest(results_dir: Path, ticker: str | None, date: str | None) -> Path:
    runs = list(find_runs(results_dir, ticker))
    if date:
        runs = [r for r in runs if r[1] == date]
    if not runs:
        where = f"{ticker or 'any ticker'}" + (f" on {date}" if date else "")
        raise SystemExit(f"digest: no completed run found for {where} under {results_dir}")
    return max(runs, key=lambda r: (r[2] / "reports" / "final_trade_decision.md").stat().st_mtime)[2]


def render(dg: Digest) -> str:
    head = f"{dg.ticker} — {dg.date}"
    if dg.duration_min:
        head += f"  ·  run ~{dg.duration_min}m"

    decision = dg.decision or "UNKNOWN"
    if dg.rating:
        decision += f" ({dg.rating})"

    if dg.trader_action:
        bits = [f"Trader: {dg.trader_action}"]
        if dg.trader_entry:
            bits.append(f"@ {dg.trader_entry}")
        if dg.trader_stop:
            bits.append(f"stop {dg.trader_stop}")
        if dg.trader_size:
            bits.append(dg.trader_size.strip(" ,."))
        tail = " — overridden by risk team" if dg.overridden else ""
        trader_line = "  (" + " ".join(bits) + tail + ")"
    else:
        trader_line = ""

    lines = [
        head,
        f"DECISION: {decision}{trader_line}",
    ]
    if dg.manager_reco:
        lines.append(f"MANAGER: {dg.manager_reco}")
    lines.append(f"WHY: {dg.why or 'n/a'}")
    if dg.missing_reports:
        lines.append(f"MISSING: {', '.join(dg.missing_reports)}")
    if dg.flags:
        lines.append("FLAGS: " + f"\n{' ' * 7}".join(f"⚠ {f}" for f in dg.flags))
    else:
        lines.append("FLAGS: none")
    return "\n".join(lines)


def _one_liner(dg: Digest) -> str:
    d = (dg.decision or "?").ljust(10)
    r = f" {dg.rating}" if dg.rating else ""
    ov = " (overrode trader)" if dg.overridden else ""
    warn = f"  ⚠x{len(dg.flags)}" if dg.flags else ""
    return f"{dg.ticker:<8} {dg.date}  {d}{r}{ov}{warn}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ticker", nargs="?", help="ticker symbol (default: most recent run)")
    ap.add_argument("date", nargs="?", help="run date YYYY-MM-DD (default: most recent)")
    ap.add_argument("--path", help="explicit run dir or reports/ dir; overrides ticker/date")
    ap.add_argument("--results-dir", help="base logs dir (default: $TRADINGAGENTS_RESULTS_DIR or ~/.tradingagents/logs)")
    ap.add_argument("--all", action="store_true", help="one-line summary of every run found")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of the text brief")
    args = ap.parse_args(argv)

    results_dir = resolve_results_dir(args.results_dir)

    if args.all:
        digs = [build_digest(rd) for _, _, rd in find_runs(results_dir, args.ticker)]
        digs.sort(key=lambda d: (d.date, d.ticker))
        if args.json:
            print(json.dumps([d.to_dict() for d in digs], indent=2))
        elif not digs:
            print(f"digest: no completed runs under {results_dir}")
            return 1
        else:
            for d in digs:
                print(_one_liner(d))
        return 0

    run_dir = Path(args.path) if args.path else _latest(results_dir, args.ticker, args.date)

    dg = build_digest(run_dir)
    print(json.dumps(dg.to_dict(), indent=2) if args.json else render(dg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
