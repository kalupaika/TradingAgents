#!/usr/bin/env python3
"""Bundle a run's reports into one prompt-ready document for Claude.ai.

The local pipeline (qwen) does the research; you take the result into a
Claude.ai chat — no API key — to either audit the claims (Claude can web-search
in real time) or produce a second-opinion decision. This assembles the six
analyst reports plus the deterministic digest / fact-check output into a single
markdown doc with an instruction header, and copies it to the clipboard.

Usage:
  python scripts/bundle.py NVDA 2026-09-07              # audit prompt (default)
  python scripts/bundle.py NVDA --mode decide           # portfolio-manager prompt
  python scripts/bundle.py NVDA --mode both
  python scripts/bundle.py NVDA -o nvda_bundle.md       # also write a file
  python scripts/bundle.py NVDA --no-clip --stdout

Modes:
  audit   — ask Claude to verify each factual claim against current web sources,
            flag fabrication and look-ahead, and say which reports to trust
  decide  — ask Claude for an independent Buy/Hold/Sell with conviction and the
            key contradictions between reports
  both    — audit section followed by decide section
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import digest  # noqa: E402
import factcheck  # noqa: E402

_REPORT_TITLES = {
    "market_report.md": "MARKET / TECHNICAL ANALYST",
    "sentiment_report.md": "SOCIAL SENTIMENT ANALYST",
    "news_report.md": "NEWS / MACRO ANALYST",
    "fundamentals_report.md": "FUNDAMENTALS ANALYST",
    "investment_plan.md": "RESEARCH MANAGER (bull/bear debate synthesis)",
    "trader_investment_plan.md": "TRADER",
    "final_trade_decision.md": "PORTFOLIO MANAGER (pipeline's final decision)",
}

_AUDIT_HEADER = """\
You are auditing a multi-agent stock-research pipeline that ran on a small local
model and is known to be error-prone. Ticker: {ticker}. Analysis ("as-of") date:
{date}.

CRITICAL: treat every claim as point-in-time to {date}. Do not credit or penalise
the reports using information published after {date}; instead, flag any claim that
could only have been known after {date} as a look-ahead violation.

A deterministic checker (recomputed indicators from raw OHLCV, scanned for
look-ahead) already ran. Its findings:

{factcheck}

The pipeline's own summarised decision:

{digest}

YOUR TASK — using web search where it helps:
1. For each material factual claim (financials, price levels, indicator values,
   named news events, macro figures, competitive claims), mark it SUPPORTED /
   UNSUPPORTED / FABRICATED and cite a source.
2. Call out every look-ahead violation.
3. Say which of the reports below are trustworthy and which should be discarded.
4. State whether the pipeline's final decision survives the audit.

The six reports follow.
"""

_DECIDE_HEADER = """\
You are a portfolio manager. Below are the analyst reports for {ticker}
(analysis date {date}) from a multi-agent research pipeline run on a small local
model. The market/technical report is frequently unreliable on this setup —
weight it accordingly.

A deterministic fact-check flagged:

{factcheck}

The pipeline concluded:

{digest}

Give me, in this order:
- Direction: Buy / Hold / Sell, with conviction (low / medium / high)
- The three strongest points FOR and the three strongest AGAINST
- Where the reports contradict each other
- Your final call and reasoning, and whether you agree with the pipeline

The reports follow.
"""


def _load_reports(reports_dir: Path) -> list[tuple[str, str]]:
    out = []
    for name in digest.REPORT_FILES:
        p = reports_dir / name
        if p.is_file():
            text = p.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                out.append((_REPORT_TITLES.get(name, name), text))
    return out


def _tool_block(run_dir: Path, offline: bool) -> tuple[str, str]:
    try:
        dg_text = digest.render(digest.build_digest(run_dir))
    except Exception as exc:  # noqa: BLE001
        dg_text = f"(digest unavailable: {exc})"
    if offline:
        fc_text = "(fact-check skipped: --offline)"
    else:
        try:
            fc_text = factcheck.render(factcheck.run_factcheck(run_dir))
        except Exception as exc:  # noqa: BLE001
            fc_text = f"(fact-check unavailable: {exc})"
    return dg_text, fc_text


def build_bundle(run_dir: Path, mode: str, offline: bool = False) -> str:
    run_dir = run_dir.expanduser().resolve()
    reports_dir = run_dir / "reports"
    if not reports_dir.is_dir() and (run_dir / "final_trade_decision.md").is_file():
        reports_dir, run_dir = run_dir, run_dir.parent

    ticker, date = run_dir.parent.name, run_dir.name
    reports = _load_reports(reports_dir)
    if not reports:
        raise SystemExit(f"bundle: no report files under {reports_dir}")

    dg_text, fc_text = _tool_block(run_dir, offline)
    fmt = {"ticker": ticker, "date": date,
           "digest": dg_text.strip(), "factcheck": fc_text.strip()}

    sections: list[str] = []
    if mode in ("audit", "both"):
        sections.append(_AUDIT_HEADER.format(**fmt))
    if mode == "both":
        sections.append("\n" + "=" * 70 + "\nSECOND TASK — INDEPENDENT DECISION\n" + "=" * 70 + "\n")
    if mode in ("decide", "both"):
        sections.append(_DECIDE_HEADER.format(**fmt))

    body = [f"# {ticker} — analysis date {date}", ""]
    body.append("\n".join(sections))
    for title, text in reports:
        body.append(f"\n\n{'-' * 70}\n## {title}\n{'-' * 70}\n\n{text}")
    return "\n".join(body).strip() + "\n"


def _to_clipboard(text: str) -> bool:
    for cmd in (["pbcopy"], ["xclip", "-selection", "clipboard"], ["wl-copy"]):
        try:
            subprocess.run(cmd, input=text.encode(), check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("ticker", nargs="?")
    ap.add_argument("date", nargs="?")
    ap.add_argument("--path", help="explicit run dir or reports/ dir")
    ap.add_argument("--results-dir")
    ap.add_argument("--mode", choices=("audit", "decide", "both"), default="audit")
    ap.add_argument("--offline", action="store_true", help="skip the network fact-check")
    ap.add_argument("-o", "--out", help="also write the bundle to this file")
    ap.add_argument("--stdout", action="store_true", help="print the bundle")
    ap.add_argument("--no-clip", action="store_true", help="don't copy to the clipboard")
    args = ap.parse_args(argv)

    results_dir = digest.resolve_results_dir(args.results_dir)
    run_dir = Path(args.path) if args.path else digest._latest(results_dir, args.ticker, args.date)

    text = build_bundle(run_dir, args.mode, offline=args.offline)

    if args.out:
        Path(args.out).expanduser().write_text(text, encoding="utf-8")
    if args.stdout:
        sys.stdout.write(text)

    clipped = False if args.no_clip else _to_clipboard(text)
    words = len(text.split())
    where = []
    if clipped:
        where.append("clipboard")
    if args.out:
        where.append(args.out)
    dest = ", ".join(where) or "stdout only"
    print(f"bundle: {args.mode} mode, {words} words -> {dest}", file=sys.stderr)
    if not clipped and not args.no_clip and not args.out and not args.stdout:
        print("(no clipboard tool found; re-run with --stdout or -o FILE)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
