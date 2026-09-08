#!/usr/bin/env python3
"""Deterministic fundamental analyzer — the anti-TradingAgents.

Where the core pipeline runs a multi-agent LLM debate and ends on a Buy/Hold/Sell,
this pulls the reported financials, computes the standard ratios itself, and
renders a multi-tab HTML report that *characterises* the company as of the last
filing. No LLM, no forecast, no recommendation — every number carries its inputs
and its source period, and a Data Quality tab lists what's missing or distorted.

Built to sit next to a TradingAgents run for the same ticker so the two
approaches — LLM narrative vs. computed evidence — can be compared directly.

Usage:
  python scripts/fundamentals.py AAPL                 # writes AAPL_fundamentals.html
  python scripts/fundamentals.py AAPL --mode quick    # compact text table
  python scripts/fundamentals.py AAPL MSFT NVDA --mode compare
  python scripts/fundamentals.py RELIANCE.NS -o r.html

Data source: Yahoo Finance (yfinance) — annual statements + key stats.
"""

from __future__ import annotations

import argparse
import html
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


# --------------------------------------------------------------------------- #
# metric container
# --------------------------------------------------------------------------- #
@dataclass
class Metric:
    label: str
    value: float | None
    kind: str = "num"          # num | x | pct | money | int | text
    inputs: str = ""           # the arithmetic, shown for auditability
    period: str = ""           # as-of date / fiscal year the inputs came from
    flag: str | None = None    # data-quality caveat (amber; collected on Data Quality tab)
    note: str | None = None    # neutral result annotation (e.g. an Altman zone)

    def fmt(self) -> str:
        v = self.value
        if v is None:
            return "—"
        if self.kind == "text":
            return str(v)
        if self.kind == "pct":
            return f"{v * 100:.1f}%" if abs(v) < 5 else f"{v:.1f}%"
        if self.kind == "x":
            return f"{v:.2f}×"
        if self.kind == "int":
            return f"{int(v):,}"
        if self.kind == "money":
            return _money(v)
        return f"{v:,.2f}"


def _money(v: float) -> str:
    a = abs(v)
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            return f"{v / div:.2f}{suf}"
    return f"{v:.0f}"


@dataclass
class Report:
    ticker: str
    name: str = ""
    sector: str = ""
    industry: str = ""
    currency: str = ""
    price: float | None = None
    generated: str = ""
    statement_date: str = ""
    tabs: dict[str, list[Metric]] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# fetch + compute
# --------------------------------------------------------------------------- #
def _row(df, *names, col=0):
    """First matching row label, at column `col` (0 = latest fiscal year)."""
    if df is None or df.empty or df.shape[1] <= col:
        return None
    for n in names:
        if n in df.index:
            try:
                val = float(df.loc[n].iloc[col])
            except (TypeError, ValueError):
                return None
            return None if val != val else val
    return None


def build_report(ticker: str) -> Report:
    import yfinance as yf

    t = yf.Ticker(ticker)
    info = t.info or {}
    if not info.get("regularMarketPrice") and not info.get("currentPrice"):
        raise SystemExit(f"fundamentals: no data for {ticker!r} on Yahoo Finance")

    inc, bs, cf = t.income_stmt, t.balance_sheet, t.cashflow
    fy = [c.strftime("%Y-%m-%d") for c in (inc.columns if inc is not None else [])]
    fy0 = fy[0] if fy else "n/a"
    fy1 = fy[1] if len(fy) > 1 else None

    rep = Report(
        ticker=ticker.upper(),
        name=info.get("longName") or info.get("shortName") or ticker.upper(),
        sector=info.get("sector", ""),
        industry=info.get("industry", ""),
        currency=info.get("financialCurrency") or info.get("currency", ""),
        price=info.get("currentPrice") or info.get("regularMarketPrice"),
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        statement_date=fy0,
    )

    mcap = info.get("marketCap")
    ev = info.get("enterpriseValue")
    revenue = _row(inc, "Total Revenue") or info.get("totalRevenue")
    ebitda = _row(inc, "EBITDA", "Normalized EBITDA") or info.get("ebitda")
    ebit = _row(inc, "EBIT", "Operating Income")
    net_income = _row(inc, "Net Income Common Stockholders", "Net Income")
    gross = _row(inc, "Gross Profit")
    op_income = _row(inc, "Operating Income", "Total Operating Income As Reported")
    pretax = _row(inc, "Pretax Income")
    tax = _row(inc, "Tax Provision")
    rnd = _row(inc, "Research And Development")
    dil_eps = _row(inc, "Diluted EPS")

    total_assets = _row(bs, "Total Assets")
    total_liab = _row(bs, "Total Liabilities Net Minority Interest")
    equity = _row(bs, "Stockholders Equity", "Common Stock Equity")
    retained = _row(bs, "Retained Earnings")
    working_cap = _row(bs, "Working Capital")
    total_debt = _row(bs, "Total Debt") or info.get("totalDebt")
    cash = _row(bs, "Cash And Cash Equivalents",
                "Cash Cash Equivalents And Short Term Investments") or info.get("totalCash")
    invested_capital = _row(bs, "Invested Capital")
    cur_assets = _row(bs, "Current Assets")
    cur_liab = _row(bs, "Current Liabilities")
    inventory = _row(bs, "Inventory")
    shares_now = _row(bs, "Ordinary Shares Number", "Share Issued")

    # yfinance .info sometimes omits marketCap for non-US listings — reconstruct
    # it from shares × price so downstream ratios (Altman D-term, yields) still work.
    if mcap is None and shares_now and rep.price:
        mcap = shares_now * rep.price
    if ev is None and mcap is not None:
        ev = mcap + (info.get("totalDebt") or 0) - (info.get("totalCash") or 0)

    ocf = _row(cf, "Operating Cash Flow", "Cash Flow From Continuing Operating Activities")
    capex = _row(cf, "Capital Expenditure")
    fcf = _row(cf, "Free Cash Flow") or info.get("freeCashflow")
    buyback = _row(cf, "Repurchase Of Capital Stock")
    dividends_paid = _row(cf, "Cash Dividends Paid", "Common Stock Dividend Paid")

    def d(a, b):
        return a / b if (a is not None and b not in (None, 0)) else None

    # ---- Overview -------------------------------------------------------- #
    rep.tabs["Overview"] = [
        Metric("Price", rep.price, "num"),
        Metric("Market cap", mcap, "money", "yfinance marketCap"),
        Metric("Enterprise value", ev, "money", "yfinance enterpriseValue"),
        Metric("52-week range",
               None if not info.get("fiftyTwoWeekLow") else
               f"{info['fiftyTwoWeekLow']:.2f} – {info['fiftyTwoWeekHigh']:.2f}", "text"),
        Metric("Revenue (FY)", revenue, "money", "income: Total Revenue", fy0),
        Metric("Net income (FY)", net_income, "money", "income: Net Income", fy0),
        Metric("Diluted EPS (FY)", dil_eps, "num", "income: Diluted EPS", fy0),
        Metric("Shares outstanding", shares_now or info.get("sharesOutstanding"), "int",
               "balance: Ordinary Shares Number", fy0),
        Metric("Beta", info.get("beta"), "num", "yfinance beta"),
    ]

    # ---- Valuation ------------------------------------------------------ #
    fcf_yield = d(fcf, mcap)
    earn_yield = d(net_income, mcap)
    div_yield = _div_yield(info)
    rep.tabs["Valuation"] = [
        Metric("P/E (trailing)", info.get("trailingPE"), "x", "price / trailing EPS"),
        Metric("P/E (forward)", info.get("forwardPE"), "x", "price / forward EPS est."),
        Metric("PEG (trailing)", info.get("trailingPegRatio") or info.get("pegRatio"), "x",
               "P/E ÷ earnings growth"),
        Metric("Price / sales", info.get("priceToSalesTrailing12Months"), "x", "mcap / TTM revenue"),
        Metric("Price / book", info.get("priceToBook"), "x", "price / book value per share"),
        Metric("EV / EBITDA", info.get("enterpriseToEbitda") or d(ev, ebitda), "x",
               "EV / EBITDA", fy0),
        Metric("EV / Sales", info.get("enterpriseToRevenue") or d(ev, revenue), "x",
               "EV / revenue", fy0),
        Metric("FCF yield", fcf_yield, "pct", "free cash flow / market cap", fy0),
        Metric("Earnings yield", earn_yield, "pct", "net income / market cap", fy0),
        Metric("Dividend yield", div_yield, "pct", "annual dividend / price"),
    ]

    # ---- Profitability ------------------------------------------------- #
    roe = d(net_income, equity)
    roa = d(net_income, total_assets)
    nopat = ebit * (1 - tax / pretax) if (ebit and pretax not in (None, 0) and tax is not None) else None
    ic = invested_capital or (
        (equity or 0) + (total_debt or 0) - (cash or 0) if equity is not None else None
    )
    roic = d(nopat, ic)
    roe_flag = "equity near zero / negative — ROE distorted" if (
        equity is not None and total_assets and abs(equity) < 0.05 * total_assets
    ) else None
    rep.tabs["Profitability"] = [
        Metric("Gross margin", d(gross, revenue), "pct", "gross profit / revenue", fy0),
        Metric("Operating margin", d(op_income, revenue), "pct", "operating income / revenue", fy0),
        Metric("Net margin", d(net_income, revenue), "pct", "net income / revenue", fy0),
        Metric("Return on equity", roe, "pct", "net income / shareholders' equity", fy0, roe_flag),
        Metric("Return on assets", roa, "pct", "net income / total assets", fy0),
        Metric("Return on invested capital", roic, "pct",
               "NOPAT / invested capital", fy0,
               None if invested_capital else "invested capital estimated (equity+debt−cash)"),
        Metric("R&D intensity", d(rnd, revenue), "pct", "R&D expense / revenue", fy0),
        Metric("Effective tax rate", d(tax, pretax), "pct", "tax provision / pretax income", fy0),
    ]

    # ---- Growth (CAGR from annual statements) -------------------------- #
    rep.tabs["Growth"] = [
        _cagr("Revenue", inc, "Total Revenue", fy),
        _cagr("Net income", inc, "Net Income Common Stockholders", fy, alt=("Net Income",)),
        _cagr("Diluted EPS", inc, "Diluted EPS", fy),
        _cagr("Operating cash flow", cf, "Operating Cash Flow", fy),
        _cagr("Free cash flow", cf, "Free Cash Flow", fy),
    ]

    # ---- Financial health -------------------------------------------- #
    z = _altman_z(working_cap, total_assets, retained, ebit, mcap, total_liab, revenue)
    piotroski = _piotroski(inc, bs, cf, fy)
    net_debt_ebitda = d((total_debt or 0) - (cash or 0), ebitda)
    int_exp = _row(inc, "Interest Expense", "Interest Expense Non Operating")
    rep.tabs["Financial health"] = [
        Metric("Current ratio", d(cur_assets, cur_liab) or info.get("currentRatio"), "x",
               "current assets / current liabilities", fy0),
        Metric("Quick ratio",
               d((cur_assets or 0) - (inventory or 0), cur_liab) or info.get("quickRatio"), "x",
               "(current assets − inventory) / current liabilities", fy0),
        Metric("Debt / equity", d(total_debt, equity) or _norm_de(info.get("debtToEquity")), "x",
               "total debt / shareholders' equity", fy0),
        Metric("Net debt / EBITDA", net_debt_ebitda, "x", "(total debt − cash) / EBITDA", fy0),
        Metric("Interest coverage", d(ebit, abs(int_exp)) if int_exp else None, "x",
               "EBIT / interest expense", fy0),
        Metric("Altman Z-Score", z[0], "num", z[1], fy0,
               flag=None if z[0] is not None else "inputs incomplete", note=z[2]),
        Metric("Piotroski F-Score", piotroski[0], "text", piotroski[1],
               f"{fy1} → {fy0}" if fy1 else fy0,
               flag=None if len(fy) >= 2 else "needs two fiscal years", note=piotroski[2]),
    ]

    # ---- Capital returns -------------------------------------------- #
    buyback_yield = d(abs(buyback), mcap) if buyback else None
    shareholder_yield = None
    if div_yield is not None or buyback_yield is not None:
        shareholder_yield = (div_yield or 0) + (buyback_yield or 0)
    rep.tabs["Capital returns"] = [
        Metric("Dividend yield", div_yield, "pct", "yfinance dividendYield"),
        Metric("Payout ratio", info.get("payoutRatio"), "pct", "dividends / net income"),
        Metric("Dividends paid (FY)", abs(dividends_paid) if dividends_paid else None, "money",
               "cashflow: Cash Dividends Paid", fy0),
        Metric("Buyback yield", buyback_yield, "pct", "share repurchases / market cap", fy0),
        Metric("Shareholder yield", shareholder_yield, "pct", "dividend yield + buyback yield", fy0),
        Metric("Capex / operating cash flow", d(abs(capex), ocf) if capex else None, "pct",
               "capital expenditure / operating cash flow", fy0),
    ]

    # ---- Data quality --------------------------------------------- #
    dq: list[Metric] = []
    if fy:
        age_days = (datetime.now(timezone.utc).date() - datetime.strptime(fy0, "%Y-%m-%d").date()).days
        dq.append(Metric("Latest annual statement", fy0, "text",
                         f"{age_days} days old", flag=None if age_days < 400 else
                         "statement is over 13 months old — interim results not reflected"))
        dq.append(Metric("Fiscal years available", len(fy), "int", ", ".join(fy)))
    for label, val, src in (
        ("Market cap", mcap, "yfinance"), ("Enterprise value", ev, "yfinance"),
        ("EBITDA", ebitda, "income stmt"), ("Total debt", total_debt, "balance sheet"),
        ("Free cash flow", fcf, "cash flow"), ("Shareholders' equity", equity, "balance sheet"),
    ):
        if val is None:
            dq.append(Metric(label, None, "text", f"missing from {src}", flag="missing"))
    if roe_flag:
        dq.append(Metric("ROE", roe, "pct", "", fy0, roe_flag))
    for tab in rep.tabs.values():
        for m in tab:
            if m.flag and m.label not in {x.label for x in dq}:
                dq.append(Metric(m.label, m.value, m.kind, m.inputs, m.period, m.flag))
    if not dq:
        dq.append(Metric("No data-quality issues detected", None, "text"))
    rep.tabs["Data quality"] = dq
    rep.flags = [f"{m.label}: {m.flag}" for m in dq if m.flag]

    return rep


def _div_yield(info: dict):
    """Return dividend yield as a fraction. yfinance's ``dividendYield`` is now a
    percentage number (0.34 -> 0.34%), so prefer computing rate/price."""
    rate = info.get("dividendRate") or info.get("trailingAnnualDividendRate")
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    if rate and price:
        return rate / price
    dy = info.get("dividendYield")
    if dy is None:
        return None
    return dy / 100 if dy > 1 else dy  # newer API: percentage; older: fraction


def _norm_de(v):
    return v / 100 if v is not None else None  # debtToEquity comes as a percentage


def _cagr(label: str, df, name: str, fy: list[str], alt: tuple[str, ...] = ()) -> Metric:
    if df is None or df.empty or name not in df.index and not any(a in df.index for a in alt):
        return Metric(f"{label} — 3y CAGR", None, "pct")
    key = name if name in df.index else next(a for a in alt if a in df.index)
    series = [float(x) for x in df.loc[key].tolist()]
    parts = []
    for span in (3, 5):
        if len(series) > span and series[span] and series[0] and series[span] > 0:
            cagr = (series[0] / series[span]) ** (1 / span) - 1
            parts.append(f"{span}y {cagr * 100:+.1f}%")
    yoy = None
    if len(series) > 1 and series[1]:
        yoy = series[0] / series[1] - 1
    label_txt = f"{label} growth"
    return Metric(
        label_txt,
        yoy,
        "pct",
        ("  ·  ".join(parts) or "insufficient history") + f"   (YoY shown; {fy[0] if fy else ''})",
        fy[0] if fy else "",
    )


def _altman_z(wc, ta, re_, ebit, mcap, tl, rev):
    if None in (wc, ta, re_, ebit, mcap, tl, rev) or ta == 0 or tl == 0:
        return (None, "needs working capital, total assets, retained earnings, EBIT, "
                "market cap, total liabilities, revenue", "inputs incomplete")
    a, b, c = wc / ta, re_ / ta, ebit / ta
    dd, e = mcap / tl, rev / ta
    z = 1.2 * a + 1.4 * b + 3.3 * c + 0.6 * dd + 1.0 * e
    zone = "distress zone (<1.81)" if z < 1.81 else (
        "grey zone (1.81–2.99)" if z < 2.99 else "safe zone (>2.99)")
    return (round(z, 2),
            f"1.2·(WC/TA {a:.2f}) + 1.4·(RE/TA {b:.2f}) + 3.3·(EBIT/TA {c:.2f}) "
            f"+ 0.6·(MCap/TL {dd:.2f}) + 1.0·(Rev/TA {e:.2f})",
            zone)


def _piotroski(inc, bs, cf, fy):
    if len(fy) < 2:
        return ("—", "needs two fiscal years", "insufficient history")

    def g(df, name, col, *alts):
        for n in (name, *alts):
            if df is not None and n in df.index:
                try:
                    v = float(df.loc[n].iloc[col])
                    return None if v != v else v
                except (TypeError, ValueError):
                    return None
        return None

    ni0, ni1 = g(inc, "Net Income", 0, "Net Income Common Stockholders"), g(inc, "Net Income", 1, "Net Income Common Stockholders")
    ta0, ta1 = g(bs, "Total Assets", 0), g(bs, "Total Assets", 1)
    ocf0 = g(cf, "Operating Cash Flow", 0, "Cash Flow From Continuing Operating Activities")
    ltd0, ltd1 = g(bs, "Long Term Debt", 0), g(bs, "Long Term Debt", 1)
    ca0, ca1 = g(bs, "Current Assets", 0), g(bs, "Current Assets", 1)
    cl0, cl1 = g(bs, "Current Liabilities", 0), g(bs, "Current Liabilities", 1)
    sh0, sh1 = g(bs, "Ordinary Shares Number", 0, "Share Issued"), g(bs, "Ordinary Shares Number", 1, "Share Issued")
    gp0, gp1 = g(inc, "Gross Profit", 0), g(inc, "Gross Profit", 1)
    rev0, rev1 = g(inc, "Total Revenue", 0), g(inc, "Total Revenue", 1)

    roa0 = ni0 / ta0 if ni0 is not None and ta0 else None
    roa1 = ni1 / ta1 if ni1 is not None and ta1 else None
    checks = [
        ("ROA > 0", roa0 is not None and roa0 > 0),
        ("Operating cash flow > 0", ocf0 is not None and ocf0 > 0),
        ("ROA improved YoY", roa0 is not None and roa1 is not None and roa0 > roa1),
        ("OCF > net income (quality)", ocf0 is not None and ni0 is not None and ocf0 > ni0),
        ("Long-term debt ratio fell",
         ltd0 is not None and ltd1 is not None and ta0 and ta1 and (ltd0 / ta0) < (ltd1 / ta1)),
        ("Current ratio improved",
         ca0 and cl0 and ca1 and cl1 and (ca0 / cl0) > (ca1 / cl1)),
        ("No share dilution", sh0 is not None and sh1 is not None and sh0 <= sh1 * 1.001),
        ("Gross margin improved",
         gp0 is not None and gp1 is not None and rev0 and rev1 and (gp0 / rev0) > (gp1 / rev1)),
        ("Asset turnover improved",
         rev0 and rev1 and ta0 and ta1 and (rev0 / ta0) > (rev1 / ta1)),
    ]
    score = sum(1 for _, ok in checks if ok)
    detail = " · ".join(f"{'✓' if ok else '✗'} {name}" for name, ok in checks)
    band = "weak (0–3)" if score <= 3 else ("middling (4–6)" if score <= 6 else "strong (7–9)")
    return (f"{score} / 9", detail, band)


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #
_DISCLAIMER = (
    "Characterisation of reported fundamentals as of the latest annual filing. "
    "Computed from Yahoo Finance data — not audited, not investment advice, no "
    "recommendation, no forward projection. Verify against primary filings before acting."
)

_CSS = """
*{box-sizing:border-box}body{margin:0;font:14px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;color:#1a1a1a;background:#fafafa}
.wrap{max-width:960px;margin:0 auto;padding:24px}
h1{font-size:22px;margin:0 0 2px}.sub{color:#666;font-size:13px;margin-bottom:18px}
.tabs{display:flex;flex-wrap:wrap;gap:4px;border-bottom:2px solid #e3e3e3;margin-bottom:0}
.tabs button{border:0;background:none;padding:9px 14px;font:inherit;cursor:pointer;color:#666;border-bottom:2px solid transparent;margin-bottom:-2px}
.tabs button.on{color:#0b5;border-bottom-color:#0b5;font-weight:600}
.panel{display:none;padding:18px 0}.panel.on{display:block}
table{width:100%;border-collapse:collapse}
td,th{text-align:left;padding:8px 10px;border-bottom:1px solid #eee;vertical-align:top}
th{font-size:12px;text-transform:uppercase;letter-spacing:.03em;color:#888}
td.v{font-variant-numeric:tabular-nums;font-weight:600;white-space:nowrap}
td.i{color:#777;font-size:12px}
.flag{color:#b45309;font-size:12px;margin-top:2px}.note{color:#666;font-size:12px;margin-top:2px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin-bottom:8px}
.card{background:#fff;border:1px solid #eee;border-radius:8px;padding:10px 12px}
.card .k{font-size:11px;color:#888;text-transform:uppercase;letter-spacing:.03em}
.card .val{font-size:18px;font-weight:700;font-variant-numeric:tabular-nums}
footer{margin-top:24px;padding-top:14px;border-top:1px solid #e3e3e3;color:#888;font-size:12px}
"""

_JS = """
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{
 document.querySelectorAll('.tabs button,.panel').forEach(x=>x.classList.remove('on'));
 b.classList.add('on');document.getElementById(b.dataset.t).classList.add('on');
});
"""


def _esc(s) -> str:
    return html.escape(str(s))


def _rows_html(metrics: list[Metric]) -> str:
    out = ["<table><tr><th>Metric</th><th>Value</th><th>Inputs / source</th><th>Period</th></tr>"]
    for m in metrics:
        ann = ""
        if m.note:
            ann += f"<div class='note'>{_esc(m.note)}</div>"
        if m.flag:
            ann += f"<div class='flag'>⚠ {_esc(m.flag)}</div>"
        out.append(
            f"<tr><td>{_esc(m.label)}{ann}</td><td class='v'>{_esc(m.fmt())}</td>"
            f"<td class='i'>{_esc(m.inputs)}</td><td class='i'>{_esc(m.period)}</td></tr>"
        )
    out.append("</table>")
    return "".join(out)


def render_html(rep: Report) -> str:
    tab_ids = {name: f"t{i}" for i, name in enumerate(rep.tabs)}
    buttons = "".join(
        f"<button data-t='{tid}' class='{'on' if i == 0 else ''}'>{_esc(name)}</button>"
        for i, (name, tid) in enumerate(tab_ids.items())
    )
    panels = []
    for i, (name, metrics) in enumerate(rep.tabs.items()):
        extra = ""
        if name == "Overview":
            cards = "".join(
                f"<div class='card'><div class='k'>{_esc(m.label)}</div>"
                f"<div class='val'>{_esc(m.fmt())}</div></div>"
                for m in metrics if m.value is not None
            )
            extra = f"<div class='cards'>{cards}</div>"
        panels.append(
            f"<div id='{tab_ids[name]}' class='panel {'on' if i == 0 else ''}'>"
            f"{extra}{_rows_html(metrics)}</div>"
        )
    subtitle = " · ".join(x for x in (rep.sector, rep.industry, rep.currency) if x)
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(rep.ticker)} fundamentals</title><style>{_CSS}</style></head><body><div class="wrap">
<h1>{_esc(rep.name)} <span style="color:#999">({_esc(rep.ticker)})</span></h1>
<div class="sub">{_esc(subtitle)}{' · ' if subtitle else ''}price {_esc(rep.price)} ·
statements to {_esc(rep.statement_date)} · generated {_esc(rep.generated)}</div>
<div class="tabs">{buttons}</div>{''.join(panels)}
<footer>{_esc(_DISCLAIMER)}</footer></div><script>{_JS}</script></body></html>"""


def render_quick(rep: Report) -> str:
    lines = [f"{rep.name} ({rep.ticker}) — {rep.sector}/{rep.industry} — "
             f"price {rep.price} {rep.currency} — statements to {rep.statement_date}", ""]
    for name in ("Valuation", "Profitability", "Growth", "Financial health"):
        lines.append(f"[{name}]")
        for m in rep.tabs.get(name, []):
            tail = f"   {m.note}" if m.note else ""
            tail += f"   ⚠ {m.flag}" if m.flag else ""
            lines.append(f"  {m.label:<28} {m.fmt():>12}{tail}")
        lines.append("")
    if rep.flags:
        lines.append("data-quality flags:")
        lines += [f"  ⚠ {f}" for f in rep.flags]
        lines.append("")
    lines.append(_DISCLAIMER)
    return "\n".join(lines)


def render_compare(reps: list[Report]) -> str:
    metric_order: list[str] = []
    for name in ("Valuation", "Profitability", "Growth", "Financial health", "Capital returns"):
        for m in reps[0].tabs.get(name, []):
            metric_order.append((name, m.label))

    head = "".join(f"<th>{_esc(r.ticker)}</th>" for r in reps)
    body = []
    last_section = None
    for section, label in metric_order:
        if section != last_section:
            body.append(f"<tr><td class='sec' colspan='{len(reps) + 1}'>{_esc(section)}</td></tr>")
            last_section = section
        cells = []
        for r in reps:
            m = next((x for x in sum(r.tabs.values(), []) if x.label == label), None)
            cells.append(f"<td class='v'>{_esc(m.fmt()) if m else '—'}</td>")
        body.append(f"<tr><td>{_esc(label)}</td>{''.join(cells)}</tr>")
    names = ", ".join(f"{_esc(r.name)} ({_esc(r.ticker)})" for r in reps)
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Compare: {' vs '.join(_esc(r.ticker) for r in reps)}</title><style>{_CSS}
td.sec{{font-weight:700;background:#f2f2f2;text-transform:uppercase;font-size:12px;letter-spacing:.03em}}
</style></head><body><div class="wrap"><h1>Fundamental comparison</h1>
<div class="sub">{names} · generated {_esc(reps[0].generated)}</div>
<table><tr><th>Metric</th>{head}</tr>{''.join(body)}</table>
<footer>{_esc(_DISCLAIMER)}</footer></div></body></html>"""


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("tickers", nargs="+")
    ap.add_argument("--mode", choices=("deep", "quick", "compare"), default="deep")
    ap.add_argument("-o", "--out", help="output HTML path (deep/compare)")
    args = ap.parse_args(argv)

    if args.mode == "compare":
        if len(args.tickers) < 2:
            ap.error("--mode compare needs at least two tickers")
        reps = [build_report(t) for t in args.tickers]
        out = Path(args.out or f"compare_{'_'.join(r.ticker for r in reps)}.html").expanduser()
        out.write_text(render_compare(reps), encoding="utf-8")
        print(f"wrote {out}")
        return 0

    for tk in args.tickers:
        rep = build_report(tk)
        if args.mode == "quick":
            print(render_quick(rep))
            if tk != args.tickers[-1]:
                print("\n" + "=" * 70 + "\n")
            continue
        out = Path(args.out or f"{rep.ticker}_fundamentals.html").expanduser()
        out.write_text(render_html(rep), encoding="utf-8")
        print(f"wrote {out}" + (f"  ({len(rep.flags)} data-quality flag(s))" if rep.flags else ""))
        if args.out and len(args.tickers) > 1:
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
