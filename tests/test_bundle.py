"""Unit tests for scripts/bundle.py — the Claude.ai prompt bundler."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location("bundle", _SCRIPTS / "bundle.py")
bundle = importlib.util.module_from_spec(_SPEC)
sys.modules["bundle"] = bundle
_SPEC.loader.exec_module(bundle)

pytestmark = pytest.mark.unit

FINAL = "**Final Trading Decision: Hold**\n\n**Rationale:** conflicting evidence, stay cautious for now.\n"
TRADER = "**Action**: Buy\n**Entry Price**: 328.0\n**Stop Loss**: 325.0\n**Position Sizing**: 10% of portfolio\n"
PLAN = "**Recommendation**: Hold\n"
MARKET = "MACD is 1.2, RSI 63, 50-day SMA 320.\n"


def _run(tmp_path, ticker="NVDA", date="2026-01-15", **extra):
    reports = tmp_path / ticker / date / "reports"
    reports.mkdir(parents=True)
    files = {
        "final_trade_decision.md": FINAL,
        "trader_investment_plan.md": TRADER,
        "investment_plan.md": PLAN,
        "market_report.md": MARKET,
        "sentiment_report.md": "Overall Sentiment: Bullish\nConfidence: High\n",
        "news_report.md": "Macro stable.\n",
        "fundamentals_report.md": "Revenue up 16% YoY.\n",
        **extra,
    }
    for name, text in files.items():
        (reports / name).write_text(text, encoding="utf-8")
    return tmp_path / ticker / date


def test_audit_bundle_structure(tmp_path):
    out = bundle.build_bundle(_run(tmp_path), "audit", offline=True)
    assert out.startswith("# NVDA — analysis date 2026-01-15")
    assert "You are auditing a multi-agent stock-research pipeline" in out
    assert "point-in-time to 2026-01-15" in out
    assert "(fact-check skipped: --offline)" in out
    # all seven sections present, in canonical order
    idx = [out.index(t) for t in (
        "MARKET / TECHNICAL ANALYST", "SOCIAL SENTIMENT ANALYST", "NEWS / MACRO ANALYST",
        "FUNDAMENTALS ANALYST", "RESEARCH MANAGER", "TRADER",
        "PORTFOLIO MANAGER (pipeline's final decision)",
    )]
    assert idx == sorted(idx)


def test_decide_bundle_uses_pm_prompt(tmp_path):
    out = bundle.build_bundle(_run(tmp_path), "decide", offline=True)
    assert "You are a portfolio manager." in out
    assert "You are auditing" not in out
    assert "DECISION: HOLD" in out  # digest brief embedded


def test_both_mode_has_two_headers(tmp_path):
    out = bundle.build_bundle(_run(tmp_path), "both", offline=True)
    assert "You are auditing" in out
    assert "You are a portfolio manager." in out
    assert "SECOND TASK — INDEPENDENT DECISION" in out


def test_missing_reports_errors(tmp_path):
    empty = tmp_path / "NVDA" / "2026-01-15" / "reports"
    empty.mkdir(parents=True)
    with pytest.raises(SystemExit):
        bundle.build_bundle(tmp_path / "NVDA" / "2026-01-15", "audit", offline=True)


def test_reports_dir_passed_directly(tmp_path):
    run = _run(tmp_path)
    out = bundle.build_bundle(run / "reports", "audit", offline=True)
    assert "# NVDA — analysis date 2026-01-15" in out


def test_clipboard_returns_false_without_tool(monkeypatch):
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    assert bundle._to_clipboard("x") is False


def test_main_stdout(tmp_path, capsys):
    run = _run(tmp_path)
    rc = bundle.main([
        "--path", str(run), "--mode", "audit", "--offline", "--stdout", "--no-clip",
    ])
    assert rc == 0
    out = capsys.readouterr()
    assert "You are auditing" in out.out
    assert "audit mode" in out.err


def test_main_writes_file(tmp_path):
    run = _run(tmp_path)
    dest = tmp_path / "b.md"
    bundle.main(["--path", str(run), "--offline", "--no-clip", "-o", str(dest)])
    assert dest.read_text().startswith("# NVDA — analysis date 2026-01-15")
