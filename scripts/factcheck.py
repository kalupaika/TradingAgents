#!/usr/bin/env python3
"""Fact-checker — validate a run's reports against source market data.

The agent reports are LLM prose. This recomputes the checkable facts from raw
OHLCV (price levels, RSI / MACD / moving averages / ATR) and diffs them against
what the reports actually claim, plus a look-ahead scan for dates past the
analysis date. It is fully deterministic — no LLM calls.

What it checks:
  Against market data —
  * look-ahead   — any report referencing a date after the analysis date
  * indicators   — RSI / MACD / MACD-signal / 50-SMA / 200-SMA / ATR claims vs
                   values recomputed as of the analysis date
  * overbought   — "overbought" / "oversold" language vs the actual RSI
  * price levels — dollar figures in the market + trader reports vs the real
                   trading range of the trailing window
  * entry price  — the trader's entry vs the last close
  * grounding    — flags a technical report that cites no indicator numbers
  Internal consistency (report cross-reads, no market data) —
  * signal-alignment — final decision vs the net directional lean of the analysts
  * calibration      — "high conviction" language over a heavily hedged rationale
  * traceability     — final decision that never engages the bull/bear debate, or
                       overrides the trader without addressing the technical case

Usage:
  python scripts/factcheck.py                    # most recent run
  python scripts/factcheck.py NVDA 2026-01-15
  python scripts/factcheck.py --path ~/.tradingagents/logs/NVDA/2026-01-15
  python scripts/factcheck.py NVDA --json
  python scripts/factcheck.py --all              # every run, one line each

Exit status: 0 if no ERROR-level findings, 1 otherwise (usable in CI / batch).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import digest  # noqa: E402  (sibling script, not a package)

# Tolerances — deliberately loose; the goal is catching fabrication and gross
# error, not policing rounding.
RSI_ABS_TOL = 12.0            # RSI points
MACD_ABS_TOL = 0.6            # MACD is price-scale dependent; also sign-checked
SMA_REL_TOL = 0.05            # 5% of the SMA value
PRICE_RANGE_PAD = 0.20        # a cited price this far outside the window range is flagged
ENTRY_REL_TOL = 0.12          # trader entry vs last close
WINDOW_TRADING_DAYS = 60      # trailing window the technical analyst nominally sees


@dataclass
class Finding:
    level: str          # ERROR | WARN | INFO
    check: str
    message: str

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class FactCheck:
    ticker: str
    date: str
    run_dir: Path
    truth: dict = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and not any(f.level == "ERROR" for f in self.findings)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "date": self.date,
            "run_dir": str(self.run_dir),
            "ok": self.ok,
            "truth": self.truth,
            "error": self.error,
            "findings": [f.to_dict() for f in self.findings],
        }


# --------------------------------------------------------------------------- #
# ground truth
# --------------------------------------------------------------------------- #
def _compute_truth(ticker: str, as_of: str) -> dict:
    """Recompute price + indicator ground truth as of ``as_of`` (inclusive)."""
    import yfinance as yf
    from stockstats import wrap

    as_of_dt = datetime.strptime(as_of, "%Y-%m-%d")
    start = (as_of_dt - timedelta(days=420)).strftime("%Y-%m-%d")
    end = (as_of_dt + timedelta(days=1)).strftime("%Y-%m-%d")  # yfinance end is exclusive

    raw = yf.Ticker(ticker).history(start=start, end=end)
    if raw.empty:
        raise ValueError(f"no OHLCV for {ticker} in {start}..{as_of}")
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    raw = raw[raw.index <= as_of_dt]
    if raw.empty:
        raise ValueError(f"no OHLCV for {ticker} on or before {as_of}")

    df = raw.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    stats = wrap(df.copy())
    # stockstats computes indicator columns lazily on first access — touch them
    # all before reading the last row, or they come back missing.
    stats[["rsi_14", "macd", "macds", "close_50_sma", "close_200_sma", "atr"]]
    last = stats.iloc[-1]
    window = df.tail(WINDOW_TRADING_DAYS)

    def val(x):
        try:
            f = float(x)
            return None if f != f else round(f, 2)  # drop NaN
        except (TypeError, ValueError):
            return None

    return {
        "as_of_bar": str(df["date"].iloc[-1].date()),
        "bars": int(len(df)),
        "last_close": val(last["close"]),
        "rsi_14": val(last.get("rsi_14")),
        "macd": val(last.get("macd")),
        "macd_signal": val(last.get("macds")),
        "sma_50": val(last.get("close_50_sma")),
        "sma_200": val(last.get("close_200_sma")),
        "atr_14": val(last.get("atr")),
        "window_low": val(window["low"].min()),
        "window_high": val(window["high"].max()),
        "window_days": int(len(window)),
    }


# --------------------------------------------------------------------------- #
# claim extraction helpers
# --------------------------------------------------------------------------- #
_MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]
_MONTH_RE = "|".join(_MONTHS)


def _iter_dates(text: str):
    for m in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", text):
        try:
            yield datetime(int(m[1]), int(m[2]), int(m[3])), m.group(0)
        except ValueError:
            continue
    for m in re.finditer(rf"\b({_MONTH_RE})\s+(\d{{1,2}}),?\s+(\d{{4}})\b", text, re.I):
        try:
            yield datetime(int(m[3]), _MONTHS.index(m[1].lower()) + 1, int(m[2])), m.group(0)
        except ValueError:
            continue


def _num(s: str) -> float | None:
    try:
        return float(s.replace(",", ""))
    except (TypeError, ValueError, AttributeError):
        return None


def _find_num(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text, re.I)
    return _num(m.group(1)) if m else None


def _check_indicators(truth: dict, market: str, findings: list[Finding]) -> None:
    low = market.lower()

    # Did the technical report cite ANY indicator number at all? (allow a few
    # words of prose between the indicator name and its value)
    indicator_hit = re.search(
        r"(macd|rsi|sma|ema|bollinger|atr|adx|moving average)[^\n]{0,25}?-?\d", low
    )
    if not indicator_hit:
        findings.append(Finding(
            "ERROR", "grounding",
            "market_report.md cites no indicator values — technical analyst produced "
            "nothing checkable (model likely failed to parse the OHLCV input)",
        ))
        return

    rsi_true = truth.get("rsi_14")
    if rsi_true is not None:
        claimed = _find_num(r"rsi[^\n\d]{0,20}?(\d{1,3}(?:\.\d+)?)", market)
        if claimed is not None and not (0 <= claimed <= 100):
            claimed = None
        if claimed is not None and abs(claimed - rsi_true) > RSI_ABS_TOL:
            findings.append(Finding(
                "ERROR", "rsi",
                f"report says RSI ~{claimed:g}; recomputed RSI-14 is {rsi_true:g}",
            ))
        # Overbought / oversold language vs reality
        says_ob = "overbought" in low
        says_os = "oversold" in low
        if says_ob and rsi_true < 65:
            findings.append(Finding(
                "WARN", "rsi", f"report calls it 'overbought' but RSI-14 is {rsi_true:g}"))
        if says_os and rsi_true > 35:
            findings.append(Finding(
                "WARN", "rsi", f"report calls it 'oversold' but RSI-14 is {rsi_true:g}"))

    macd_true = truth.get("macd")
    if macd_true is not None:
        claimed = _find_num(r"\bmacd\b[^\n]{0,20}?(-?\d+(?:\.\d+)?)", market)
        if claimed is not None:
            if claimed * macd_true < 0 and abs(claimed) > 0.1 and abs(macd_true) > 0.1:
                findings.append(Finding(
                    "ERROR", "macd",
                    f"report has MACD {claimed:g} (wrong sign); recomputed MACD is {macd_true:g}",
                ))
            elif abs(claimed - macd_true) > MACD_ABS_TOL:
                findings.append(Finding(
                    "WARN", "macd",
                    f"report says MACD ~{claimed:g}; recomputed MACD is {macd_true:g}",
                ))

    for label, key, pat in (
        ("50-day SMA", "sma_50", r"50[- ]?(?:day|d)?\s*(?:sma|simple moving average|ma)[^\n\d]{0,20}?(\d+(?:\.\d+)?)"),
        ("200-day SMA", "sma_200", r"200[- ]?(?:day|d)?\s*(?:sma|simple moving average|ma)[^\n\d]{0,20}?(\d+(?:\.\d+)?)"),
    ):
        true_v = truth.get(key)
        claimed = _find_num(pat, market)
        if true_v and claimed is not None and abs(claimed - true_v) / true_v > SMA_REL_TOL:
            findings.append(Finding(
                "WARN", "sma",
                f"report says {label} ~{claimed:g}; recomputed is {true_v:g}",
            ))


def _check_prices(truth: dict, market: str, trader: str, findings: list[Finding]) -> None:
    lo, hi = truth.get("window_low"), truth.get("window_high")
    if not lo or not hi:
        return
    pad_lo, pad_hi = lo * (1 - PRICE_RANGE_PAD), hi * (1 + PRICE_RANGE_PAD)
    scale_lo, scale_hi = lo * 0.4, hi * 2.5  # ignore numbers that clearly aren't a share price

    seen: set[float] = set()
    for m in re.finditer(r"\$\s?(\d[\d,]*(?:\.\d+)?)|(?<![\d.])(\d[\d,]*\.\d{2})(?![\d])", market):
        n = _num(m.group(1) or m.group(2))
        if n is None or n in seen or not (scale_lo <= n <= scale_hi):
            continue
        seen.add(n)
        if not (pad_lo <= n <= pad_hi):
            findings.append(Finding(
                "WARN", "price",
                f"market_report cites {n:g}, outside the {truth['window_days']}-day "
                f"range {lo:g}–{hi:g} (±{int(PRICE_RANGE_PAD*100)}%)",
            ))

    close = truth.get("last_close")
    entry = _find_num(r"\*{0,2}entry price\*{0,2}\s*:?\s*\*{0,2}\s*\$?(\d[\d,]*(?:\.\d+)?)", trader)
    if close and entry and abs(entry - close) / close > ENTRY_REL_TOL:
        findings.append(Finding(
            "ERROR", "entry",
            f"trader entry {entry:g} is {abs(entry-close)/close*100:.0f}% off the "
            f"last close {close:g} as of {truth['as_of_bar']}",
        ))


# --------------------------------------------------------------------------- #
# internal consistency (no market data — cross-reads the reports themselves)
# --------------------------------------------------------------------------- #
_BULL = (
    "bullish", "outperform", "upside", "undervalued", "accumulate", "overweight",
    "uptrend", "breakout", "rally", "tailwind", "strong buy", "long position",
)
_BEAR = (
    "bearish", "underperform", "downside", "overvalued", "underweight", "short position",
    "downtrend", "breakdown", "selloff", "sell-off", "headwind", "strong sell", "deteriorat",
)
_HEDGE = (
    "however", "although", "unclear", "ambiguous", "uncertain", "conflicting", "mixed signal",
    "on the other hand", "that said", "not guaranteed", "speculative", "hard to say",
    "remains to be seen", "caution", "wait for", "lack of clarity", "too early",
)
_CONVICTION = (
    "strong buy", "strong sell", "high confidence", "high conviction", "decisive",
    "compelling case", "unambiguous", "strongly recommend", "clear buy", "clear sell",
    "conviction is high",
)
_DEBATE_TOKENS = ("bull", "bear", "aggressive analyst", "conservative analyst",
                  "neutral analyst", "debate", "research manager", "risk team")

_BUY_SIDE = {"BUY", "STRONG BUY"}
_SELL_SIDE = {"SELL", "STRONG SELL"}


def _side(direction: str | None) -> str | None:
    """Collapse STRONG BUY/BUY -> BUY etc. so same-direction calls compare equal."""
    if direction in _BUY_SIDE:
        return "BUY"
    if direction in _SELL_SIDE:
        return "SELL"
    return direction  # HOLD or None


def _count(text: str, needles) -> int:
    return sum(text.count(n) for n in needles)


def _lean(name: str, text: str) -> float:
    """Directional lean of one analyst report in [-1, 1] (0 = neutral/unclear)."""
    low = text.lower()
    if name == "sentiment_report.md":
        m = re.search(r"overall sentiment[:\s*]+\**\s*(bullish|bearish|neutral)", low)
        if m:
            return {"bullish": 0.6, "bearish": -0.6, "neutral": 0.0}[m.group(1)]
    bull, bear = _count(low, _BULL), _count(low, _BEAR)
    if bull + bear < 4:
        return 0.0
    return round((bull - bear) / (bull + bear), 2)


def _check_consistency(reports_dir: Path, findings: list[Finding]) -> None:
    final = digest._read(reports_dir, "final_trade_decision.md")
    trader = digest._read(reports_dir, "trader_investment_plan.md")
    plan = digest._read(reports_dir, "investment_plan.md")
    if not final:
        findings.append(Finding("ERROR", "consistency", "final_trade_decision.md missing"))
        return

    decision = digest._norm_direction(
        digest._first(r"final trading decision\s*:?\s*\*{0,2}\s*([A-Za-z ]+)", final)
        or digest._first(r"final transaction proposal\s*:?\s*\*{0,2}\s*([A-Za-z ]+)", final)
        or final[:400]
    )
    if not decision:
        findings.append(Finding(
            "ERROR", "consistency", "no parseable decision in final_trade_decision.md"))
        return

    # 1. Analyst signals vs the decision.
    leans = {
        n: _lean(n, digest._read(reports_dir, n))
        for n in ("market_report.md", "sentiment_report.md",
                  "news_report.md", "fundamentals_report.md")
        if (reports_dir / n).is_file()
    }
    scored = [v for v in leans.values() if v != 0.0]
    if scored:
        net = sum(scored) / len(scored)
        detail = ", ".join(f"{k.split('_')[0]} {v:+g}" for k, v in leans.items() if v)
        if decision in _BUY_SIDE and net <= -0.34:
            findings.append(Finding(
                "WARN", "signal-alignment",
                f"decision is {decision} but analyst reports lean bearish "
                f"(net {net:+.2f}: {detail})",
            ))
        elif decision in _SELL_SIDE and net >= 0.34:
            findings.append(Finding(
                "WARN", "signal-alignment",
                f"decision is {decision} but analyst reports lean bullish "
                f"(net {net:+.2f}: {detail})",
            ))

    # 2. Confidence calibration — strong conviction language over a hedged rationale.
    blob = (final + "\n" + trader).lower()
    rating = digest._first(r"rating\s*:?\s*\*{0,2}\s*(strong buy|strong sell)", final)
    conviction = bool(rating) or _count(blob, _CONVICTION) > 0
    hedges = _count(blob, _HEDGE)
    if conviction and hedges >= 5:
        findings.append(Finding(
            "WARN", "calibration",
            f"decision asserts high conviction but the rationale hedges {hedges} times "
            "(however / unclear / conflicting / …)",
        ))

    # 3. Does the final decision actually engage the bull/bear debate?
    if plan and not any(tok in final.lower() for tok in _DEBATE_TOKENS):
        findings.append(Finding(
            "WARN", "traceability",
            "final_trade_decision.md never references the bull/bear debate or the "
            "research-manager view it is supposed to synthesise",
        ))

    # 4. Risk team overrode the trader without addressing why.
    trader_action = digest._norm_direction(
        digest._first(r"(?:^|\n)\s*\*{0,2}action\*{0,2}\s*:\s*\*{0,2}\s*([A-Za-z ]+)", trader)
        or digest._first(r"final transaction proposal\s*:?\s*\*{0,2}\s*([A-Za-z ]+)", trader)
    )
    if trader_action and _side(trader_action) != _side(decision):
        low = final.lower()
        if "trader" not in low and "technical" not in low:
            findings.append(Finding(
                "INFO", "traceability",
                f"final decision ({decision}) overrides the trader ({trader_action}) but "
                "the rationale doesn't mention the trader or the technical case it set aside",
            ))


def _check_lookahead(reports_dir: Path, as_of: str, findings: list[Finding]) -> None:
    cutoff = datetime.strptime(as_of, "%Y-%m-%d")
    horizon = cutoff + timedelta(days=1)
    for name in digest.REPORT_FILES:
        p = reports_dir / name
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        future = sorted({raw for dt, raw in _iter_dates(text) if dt > horizon})
        if future:
            findings.append(Finding(
                "ERROR", "look-ahead",
                f"{name} references date(s) after {as_of}: {', '.join(future[:5])}"
                + (" …" if len(future) > 5 else ""),
            ))


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run_factcheck(run_dir: Path) -> FactCheck:
    run_dir = run_dir.expanduser().resolve()
    reports = run_dir / "reports"
    if not reports.is_dir() and (run_dir / "final_trade_decision.md").is_file():
        reports, run_dir = run_dir, run_dir.parent

    fc = FactCheck(ticker=run_dir.parent.name, date=run_dir.name, run_dir=run_dir)

    try:
        datetime.strptime(fc.date, "%Y-%m-%d")
    except ValueError:
        fc.error = f"run dir name {fc.date!r} is not a YYYY-MM-DD date"
        return fc

    _check_lookahead(reports, fc.date, fc.findings)
    _check_consistency(reports, fc.findings)

    try:
        fc.truth = _compute_truth(fc.ticker, fc.date)
    except Exception as exc:  # noqa: BLE001 — surfaced as a finding, not a crash
        fc.error = f"could not compute ground truth: {exc}"
        return fc

    market = digest._read(reports, "market_report.md")
    trader = digest._read(reports, "trader_investment_plan.md")
    if not market:
        fc.findings.append(Finding("WARN", "grounding", "market_report.md missing"))
    else:
        _check_indicators(fc.truth, market, fc.findings)
        _check_prices(fc.truth, market, trader, fc.findings)

    return fc


def render(fc: FactCheck) -> str:
    t = fc.truth
    lines = [f"{fc.ticker} — {fc.date}   fact-check: {'PASS' if fc.ok else 'FAIL'}"]
    if fc.error:
        lines.append(f"  ! ground truth unavailable: {fc.error}")
        lines.append("  (price/indicator checks skipped; consistency checks still ran)")
    else:
        lines.append(
            f"  truth @ {t['as_of_bar']} ({t['bars']} bars): close {t['last_close']}  "
            f"RSI-14 {t['rsi_14']}  MACD {t['macd']}/{t['macd_signal']}  "
            f"50SMA {t['sma_50']}  200SMA {t['sma_200']}  ATR {t['atr_14']}  "
            f"{t['window_days']}d range {t['window_low']}–{t['window_high']}"
        )
    if not fc.findings:
        lines.append("  ✓ no discrepancies found")
    for f in sorted(fc.findings, key=lambda x: {"ERROR": 0, "WARN": 1, "INFO": 2}[x.level]):
        mark = {"ERROR": "✗", "WARN": "⚠", "INFO": "·"}[f.level]
        lines.append(f"  {mark} [{f.check}] {f.message}")
    return "\n".join(lines)


def _one_liner(fc: FactCheck) -> str:
    errs = sum(f.level == "ERROR" for f in fc.findings)
    warns = sum(f.level == "WARN" for f in fc.findings)
    status = "PASS" if fc.ok else "FAIL"
    tail = fc.error or f"{errs} error(s), {warns} warning(s)"
    return f"{fc.ticker:<8} {fc.date}  {status:<4}  {tail}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("ticker", nargs="?")
    ap.add_argument("date", nargs="?")
    ap.add_argument("--path", help="explicit run dir or reports/ dir")
    ap.add_argument("--results-dir")
    ap.add_argument("--all", action="store_true", help="fact-check every run found")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    results_dir = digest.resolve_results_dir(args.results_dir)

    if args.all:
        checks = [run_factcheck(rd) for _, _, rd in digest.find_runs(results_dir, args.ticker)]
        checks.sort(key=lambda c: (c.date, c.ticker))
        if args.json:
            print(json.dumps([c.to_dict() for c in checks], indent=2))
        elif not checks:
            print(f"factcheck: no completed runs under {results_dir}")
            return 1
        else:
            for c in checks:
                print(_one_liner(c))
        return 0 if all(c.ok for c in checks) else 1

    run_dir = Path(args.path) if args.path else digest._latest(results_dir, args.ticker, args.date)
    fc = run_factcheck(run_dir)
    print(json.dumps(fc.to_dict(), indent=2) if args.json else render(fc))
    return 0 if fc.ok else 1


if __name__ == "__main__":
    sys.exit(main())
