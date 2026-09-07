"""Unit tests for scripts/digest.py — the decision-digest wrapper."""

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "digest", Path(__file__).resolve().parents[1] / "scripts" / "digest.py"
)
digest = importlib.util.module_from_spec(_SPEC)
sys.modules["digest"] = digest  # let dataclasses resolve the module during class build
_SPEC.loader.exec_module(digest)

pytestmark = pytest.mark.unit


FINAL_HOLD = """**Final Trading Decision: Hold**

**Rationale:**
The evidence presented by the analysts is materially conflicting and ambiguous,
warranting a cautious stance rather than a directional bet. Preserve capital.
"""

FINAL_BUY_RATED = """**Final Trading Decision: Buy**
**Rating: Overweight**
**SNDK (NMS)**

---

**Rationale:** Strong quantifiable fundamentals and structural leadership.
"""

TRADER_BUY = """**Action**: Buy

**Reasoning**: Price stabilised; position sizing 5% aligns with the plan.

**Entry Price**: 328.0

**Stop Loss**: 325.0

**Position Sizing**: 10% of portfolio

FINAL TRANSACTION PROPOSAL: **BUY**
"""

PLAN_HOLD = "**Recommendation**: Hold\n\n**Rationale**: balanced risks.\n"

MARKET_GOOD = "MACD is 1.2 and RSI 63 with the 50-day SMA at 320.5 trending up.\n"
MARKET_CHATTY = "Let me know if you'd like further analysis or adjustments!\n"


def _make_run(tmp_path, ticker="NVDA", date="2026-01-15", *, files):
    reports = tmp_path / ticker / date / "reports"
    reports.mkdir(parents=True)
    for name, text in files.items():
        (reports / name).write_text(text, encoding="utf-8")
    return tmp_path / ticker / date


def test_parses_decision_trader_and_override(tmp_path):
    run = _make_run(
        tmp_path,
        files={
            "final_trade_decision.md": FINAL_HOLD,
            "trader_investment_plan.md": TRADER_BUY,
            "investment_plan.md": PLAN_HOLD,
            "market_report.md": MARKET_GOOD,
        },
    )
    dg = digest.build_digest(run)
    assert dg.ticker == "NVDA"
    assert dg.decision == "HOLD"
    assert dg.trader_action == "BUY"
    assert dg.trader_entry == "328.0"
    assert dg.trader_stop == "325.0"
    assert dg.trader_size == "10% of portfolio"  # labelled field, not the "5%" in Reasoning
    assert dg.manager_reco == "Hold"
    assert dg.overridden is True
    assert "conflicting and ambiguous" in dg.why


def test_rating_and_no_override(tmp_path):
    run = _make_run(
        tmp_path,
        ticker="SNDK",
        files={
            "final_trade_decision.md": FINAL_BUY_RATED,
            "trader_investment_plan.md": TRADER_BUY,
            "market_report.md": MARKET_GOOD,
        },
    )
    dg = digest.build_digest(run)
    assert dg.decision == "BUY"
    assert dg.rating == "Overweight"
    assert dg.overridden is False
    assert "MARKET" not in digest.render(dg)  # no investment_plan.md -> no MANAGER line
    assert "investment_plan.md" in dg.missing_reports


def test_technical_parse_failure_flagged(tmp_path):
    run = _make_run(
        tmp_path,
        files={
            "final_trade_decision.md": FINAL_HOLD,
            "market_report.md": MARKET_CHATTY,
        },
    )
    dg = digest.build_digest(run)
    assert any("technical-analysis node" in f for f in dg.flags)


def test_clean_technical_report_not_flagged(tmp_path):
    run = _make_run(
        tmp_path,
        files={
            "final_trade_decision.md": FINAL_HOLD,
            "market_report.md": MARKET_GOOD,
            "sentiment_report.md": "Overall Sentiment: Bullish\nConfidence: High\n",
            "news_report.md": "Macro backdrop is stable per FRED series fed_funds_rate.\n",
        },
    )
    dg = digest.build_digest(run)
    assert dg.flags == []


def test_latest_picks_newest_mtime(tmp_path):
    old = _make_run(tmp_path, date="2026-01-10", files={"final_trade_decision.md": FINAL_HOLD})
    new = _make_run(tmp_path, date="2026-01-20", files={"final_trade_decision.md": FINAL_BUY_RATED})
    import os
    import time

    now = time.time()
    os.utime(old / "reports" / "final_trade_decision.md", (now - 1000, now - 1000))
    os.utime(new / "reports" / "final_trade_decision.md", (now, now))
    picked = digest._latest(tmp_path, "NVDA", None)
    assert picked == new


def test_find_runs_skips_incomplete(tmp_path):
    _make_run(tmp_path, date="2026-01-10", files={"final_trade_decision.md": FINAL_HOLD})
    incomplete = tmp_path / "NVDA" / "2026-01-11" / "reports"
    incomplete.mkdir(parents=True)
    (incomplete / "market_report.md").write_text("partial", encoding="utf-8")
    runs = list(digest.find_runs(tmp_path, "NVDA"))
    assert [r[1] for r in runs] == ["2026-01-10"]


def test_render_shape(tmp_path):
    run = _make_run(
        tmp_path,
        files={
            "final_trade_decision.md": FINAL_HOLD,
            "trader_investment_plan.md": TRADER_BUY,
            "investment_plan.md": PLAN_HOLD,
            "market_report.md": MARKET_GOOD,
        },
    )
    out = digest.render(digest.build_digest(run))
    assert out.startswith("NVDA — 2026-01-15")
    assert "DECISION: HOLD" in out
    assert "overridden by risk team" in out
    assert out.count("\n") >= 3
