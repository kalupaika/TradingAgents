"""Unit tests for scripts/fundamentals.py — the deterministic analyzer.

``build_report`` hits yfinance and is not exercised; tests cover the pure
computation and rendering.
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location("fundamentals", _SCRIPTS / "fundamentals.py")
fundamentals = importlib.util.module_from_spec(_SPEC)
sys.modules["fundamentals"] = fundamentals
_SPEC.loader.exec_module(fundamentals)

M = fundamentals.Metric
pytestmark = pytest.mark.unit


def test_metric_formatting():
    assert M("x", None).fmt() == "—"
    assert M("x", 0.1234, "pct").fmt() == "12.3%"
    assert M("x", 45.0, "pct").fmt() == "45.0%"        # already a percentage
    assert M("x", 2.5, "x").fmt() == "2.50×"
    assert M("x", 1_500_000_000, "money").fmt() == "1.50B"
    assert M("x", 1234567, "int").fmt() == "1,234,567"
    assert M("x", "grey", "text").fmt() == "grey"


def test_money_scales():
    assert fundamentals._money(2.3e12) == "2.30T"
    assert fundamentals._money(5e8) == "500.00M"
    assert fundamentals._money(-4.1e9) == "-4.10B"


def test_div_yield_prefers_rate_over_price():
    assert fundamentals._div_yield({"dividendRate": 1.08, "currentPrice": 320.0}) == pytest.approx(0.003375)
    # newer API: dividendYield is a percentage number
    assert fundamentals._div_yield({"dividendYield": 3.0}) == pytest.approx(0.03)
    # older API: already a fraction
    assert fundamentals._div_yield({"dividendYield": 0.03}) == pytest.approx(0.03)
    assert fundamentals._div_yield({}) is None


def test_altman_z_known_value():
    # WC/TA .2, RE/TA .3, EBIT/TA .1, MCap/TL 2.0, Rev/TA 1.0
    z, expr, zone = fundamentals._altman_z(
        wc=200, ta=1000, re_=300, ebit=100, mcap=2000, tl=1000, rev=1000
    )
    # 1.2*.2 + 1.4*.3 + 3.3*.1 + 0.6*2 + 1.0*1 = .24+.42+.33+1.2+1 = 3.19
    assert z == pytest.approx(3.19, abs=0.01)
    assert "safe zone" in zone
    assert "WC/TA" in expr


def test_altman_z_incomplete():
    z, _, zone = fundamentals._altman_z(None, 1000, 300, 100, 2000, 1000, 1000)
    assert z is None and "incomplete" in zone


def _df(rows: dict, ncols=2):
    cols = [pd.Timestamp(f"202{5 - i}-09-30") for i in range(ncols)]
    return pd.DataFrame(rows, index=list(rows), columns=cols).T.T if False else pd.DataFrame(
        {c: [rows[r][i] for r in rows] for i, c in enumerate(cols)}, index=list(rows)
    )


def test_piotroski_strong_company():
    inc = _df({
        "Net Income": [120, 90],
        "Total Revenue": [1000, 900],
        "Gross Profit": [500, 430],
    })
    bs = _df({
        "Total Assets": [800, 820],
        "Long Term Debt": [100, 140],
        "Current Assets": [300, 260],
        "Current Liabilities": [200, 210],
        "Ordinary Shares Number": [1000, 1000],
    })
    cf = _df({"Operating Cash Flow": [150, 110]})
    score, detail, band = fundamentals._piotroski(inc, bs, cf, ["2025-09-30", "2024-09-30"])
    assert score == "9 / 9"
    assert band == "strong (7–9)"
    assert detail.count("✓") == 9


def test_piotroski_needs_two_years():
    score, _, band = fundamentals._piotroski(None, None, None, ["2025-09-30"])
    assert score == "—" and "insufficient" in band


def test_cagr_reports_yoy_and_multiyear():
    df = _df({"Total Revenue": [200, 180, 150, 120, 100, 90]}, ncols=6)
    m = fundamentals._cagr("Revenue", df, "Total Revenue",
                           [f"y{i}" for i in range(6)])
    assert m.label == "Revenue growth"
    assert m.value == pytest.approx(200 / 180 - 1)
    assert "3y" in m.inputs and "5y" in m.inputs


def _mini_report():
    rep = fundamentals.Report(ticker="ZZZ", name="Zeta Co", sector="Tech",
                              price=100.0, generated="2026-01-01 00:00 UTC",
                              statement_date="2025-12-31")
    rep.tabs = {
        "Overview": [M("Market cap", 5e9, "money")],
        "Valuation": [M("P/E (trailing)", 20.0, "x", "price / EPS")],
        "Financial health": [M("Altman Z-Score", 3.5, "num", "…", note="safe zone (>2.99)")],
        "Data quality": [M("EBITDA", None, "text", "missing from income stmt", flag="missing")],
    }
    rep.flags = ["EBITDA: missing"]
    return rep


def test_render_html_has_tabs_and_disclaimer():
    out = fundamentals.render_html(_mini_report())
    assert "<title>ZZZ fundamentals</title>" in out
    assert out.count("<button data-t=") == 4
    assert "not investment advice, no recommendation" in out
    assert "safe zone (&gt;2.99)" in out          # note rendered + escaped
    assert "⚠ missing" in out


def test_render_quick_lists_flags():
    out = fundamentals.render_quick(_mini_report())
    assert "Zeta Co (ZZZ)" in out
    assert "data-quality flags:" in out
    assert "⚠ EBITDA: missing" in out


def test_render_compare_side_by_side():
    a, b = _mini_report(), _mini_report()
    b.ticker = "YYY"
    out = fundamentals.render_compare([a, b])
    assert "<th>ZZZ</th><th>YYY</th>" in out
    assert "P/E (trailing)" in out


def test_main_compare_needs_two_tickers():
    with pytest.raises(SystemExit):
        fundamentals.main(["AAPL", "--mode", "compare"])
