"""Unit tests for scripts/factcheck.py — deterministic report validation.

The ground-truth computation (`_compute_truth`) hits yfinance and is not
exercised here; these tests drive the pure claim-checking logic with a
synthetic truth dict.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

_SPEC = importlib.util.spec_from_file_location("factcheck", _SCRIPTS / "factcheck.py")
factcheck = importlib.util.module_from_spec(_SPEC)
sys.modules["factcheck"] = factcheck
_SPEC.loader.exec_module(factcheck)

pytestmark = pytest.mark.unit


TRUTH = {
    "as_of_bar": "2026-01-14",
    "bars": 290,
    "last_close": 320.0,
    "rsi_14": 54.0,
    "macd": 2.6,
    "macd_signal": 1.1,
    "sma_50": 315.0,
    "sma_200": 283.0,
    "atr_14": 7.6,
    "window_low": 300.0,
    "window_high": 340.0,
    "window_days": 60,
}


def _levels(findings, check):
    return [f.level for f in findings if f.check == check]


def test_grounding_flag_when_no_indicator_numbers():
    findings = []
    factcheck._check_indicators(TRUTH, "Prices drifted sideways. Let me know if you want more.", findings)
    assert _levels(findings, "grounding") == ["ERROR"]


def test_no_grounding_flag_when_indicators_present():
    findings = []
    factcheck._check_indicators(TRUTH, "RSI-14 sits at 55 and MACD is 2.5 above signal.", findings)
    assert not _levels(findings, "grounding")


def test_rsi_value_mismatch_is_error():
    findings = []
    factcheck._check_indicators(TRUTH, "The RSI is 82, deep in overbought territory.", findings)
    assert "ERROR" in _levels(findings, "rsi")


def test_rsi_value_close_enough_passes():
    findings = []
    factcheck._check_indicators(TRUTH, "RSI reads about 58 here, MACD 2.7.", findings)
    assert not _levels(findings, "rsi")


def test_overbought_language_without_high_rsi_warns():
    findings = []
    factcheck._check_indicators(TRUTH, "Momentum looks overbought. RSI near 54. MACD 2.6.", findings)
    assert "WARN" in _levels(findings, "rsi")


def test_macd_wrong_sign_is_error():
    findings = []
    factcheck._check_indicators(TRUTH, "MACD has crossed to -3.0, RSI 54.", findings)
    assert "ERROR" in _levels(findings, "macd")


def test_sma_relative_mismatch_warns():
    findings = []
    factcheck._check_indicators(TRUTH, "Price rides the 50-day SMA at 402.5. RSI 54. MACD 2.6.", findings)
    assert "WARN" in _levels(findings, "sma")


def test_price_outside_window_range_warns():
    findings = []
    factcheck._check_prices(TRUTH, "A key level sits at $455.00 after the run.", "", findings)
    assert "WARN" in _levels(findings, "price")


def test_price_inside_range_ok():
    findings = []
    factcheck._check_prices(TRUTH, "Support near $305.00, resistance at $335.00.", "", findings)
    assert not _levels(findings, "price")


def test_entry_far_from_last_close_is_error():
    trader = "**Entry Price**: 250.0\n**Stop Loss**: 240.0\n"
    findings = []
    factcheck._check_prices(TRUTH, "", trader, findings)
    assert "ERROR" in _levels(findings, "entry")


def test_lookahead_detects_future_dates(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "news_report.md").write_text(
        "Outlook into 2026-02-01 and March 3, 2026 remains strong.", encoding="utf-8"
    )
    (reports / "market_report.md").write_text("As of 2026-01-10 the trend is up.", encoding="utf-8")
    findings = []
    factcheck._check_lookahead(reports, "2026-01-14", findings)
    checks = [f for f in findings if f.check == "look-ahead"]
    assert len(checks) == 1 and checks[0].level == "ERROR"
    assert "news_report.md" in checks[0].message


def test_iter_dates_parses_both_formats():
    got = {raw for _, raw in factcheck._iter_dates("see 2026-01-15 and February 2, 2026")}
    assert got == {"2026-01-15", "February 2, 2026"}


def test_factcheck_ok_property():
    fc = factcheck.FactCheck(ticker="X", date="2026-01-14", run_dir=Path("/tmp/x"))
    assert fc.ok
    fc.findings.append(factcheck.Finding("WARN", "price", "minor"))
    assert fc.ok
    fc.findings.append(factcheck.Finding("ERROR", "rsi", "bad"))
    assert not fc.ok


def test_render_shape():
    fc = factcheck.FactCheck(ticker="AAPL", date="2026-01-14", run_dir=Path("/tmp/x"), truth=TRUTH)
    fc.findings.append(factcheck.Finding("ERROR", "grounding", "no indicators"))
    out = factcheck.render(fc)
    assert out.startswith("AAPL — 2026-01-14   fact-check: FAIL")
    assert "RSI-14 54.0" in out
    assert "✗ [grounding]" in out


def test_render_keeps_findings_when_truth_unavailable():
    fc = factcheck.FactCheck(ticker="X", date="2026-01-14", run_dir=Path("/tmp/x"),
                             error="network down")
    fc.findings.append(factcheck.Finding("WARN", "calibration", "hedged"))
    out = factcheck.render(fc)
    assert "ground truth unavailable" in out
    assert "[calibration]" in out


# --- internal-consistency checks ------------------------------------------- #

def _consistency_run(tmp_path, **files):
    reports = tmp_path / "reports"
    reports.mkdir()
    for name, text in files.items():
        (reports / f"{name}.md").write_text(text, encoding="utf-8")
    findings = []
    factcheck._check_consistency(reports, findings)
    return findings


def test_lean_uses_explicit_sentiment_marker():
    assert factcheck._lean("sentiment_report.md", "Overall Sentiment: **Bearish** (3/10)") < 0
    assert factcheck._lean("sentiment_report.md", "Overall Sentiment: Bullish") > 0


def test_lean_keyword_tally_needs_minimum_signal():
    assert factcheck._lean("news_report.md", "one bearish note") == 0.0
    assert factcheck._lean(
        "news_report.md", "bearish downside headwind underperform overvalued breakdown"
    ) < 0


def test_signal_alignment_flags_buy_against_bearish_analysts(tmp_path):
    findings = _consistency_run(
        tmp_path,
        final_trade_decision="**Final Trading Decision: Buy**\nThe bull case wins on balance.",
        trader_investment_plan="**Action**: Buy\n",
        investment_plan="**Recommendation**: Buy\n",
        market_report="bearish breakdown downtrend selloff headwind underperform",
        sentiment_report="Overall Sentiment: Bearish",
        news_report="bearish downside headwind underperform overvalued",
        fundamentals_report="bearish overvalued downside deteriorat",
    )
    assert any(f.check == "signal-alignment" and f.level == "WARN" for f in findings)


def test_no_signal_alignment_flag_when_aligned(tmp_path):
    findings = _consistency_run(
        tmp_path,
        final_trade_decision="**Final Trading Decision: Buy**\nThe bull case is strong.",
        trader_investment_plan="**Action**: Buy\n",
        investment_plan="**Recommendation**: Buy\n",
        market_report="bullish breakout uptrend rally accumulate outperform",
        sentiment_report="Overall Sentiment: Bullish",
    )
    assert not any(f.check == "signal-alignment" for f in findings)


def test_calibration_flags_conviction_over_hedged_rationale(tmp_path):
    findings = _consistency_run(
        tmp_path,
        final_trade_decision=(
            "**Final Trading Decision: Strong Buy**\nA compelling case. However unclear, "
            "conflicting, uncertain, mixed signal, too early, caution warranted, that said."
        ),
        trader_investment_plan="**Action**: Buy\n",
        investment_plan="**Recommendation**: Buy\n",
        market_report="bullish breakout",
    )
    assert any(f.check == "calibration" for f in findings)


def test_traceability_flags_decision_ignoring_debate(tmp_path):
    findings = _consistency_run(
        tmp_path,
        final_trade_decision="**Final Trading Decision: Hold**\nMacro is murky so we wait.",
        investment_plan="**Recommendation**: Hold\n\nThe bull and bear debated at length.",
        market_report="bullish breakout",
    )
    assert any(f.check == "traceability" and f.level == "WARN" for f in findings)


def test_traceability_ok_when_debate_referenced(tmp_path):
    findings = _consistency_run(
        tmp_path,
        final_trade_decision=(
            "**Final Trading Decision: Hold**\nThe Aggressive Analyst and Conservative "
            "Analyst disagree; the research manager's Hold stands."
        ),
        investment_plan="**Recommendation**: Hold\n",
        market_report="bullish breakout",
    )
    assert not any(f.check == "traceability" for f in findings)


def test_strong_buy_not_treated_as_override_of_buy(tmp_path):
    findings = _consistency_run(
        tmp_path,
        final_trade_decision="**Final Trading Decision: Strong Buy**\nBull and bear both weighed.",
        trader_investment_plan="**Action**: Buy\n",
        investment_plan="**Recommendation**: Buy\n",
    )
    assert not any(f.check == "traceability" and f.level == "INFO" for f in findings)


def test_unparseable_decision_is_error(tmp_path):
    findings = _consistency_run(tmp_path, final_trade_decision="the committee could not agree")
    assert findings and findings[0].level == "ERROR" and findings[0].check == "consistency"
