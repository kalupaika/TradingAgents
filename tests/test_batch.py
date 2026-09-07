"""Unit tests for scripts/batch.py — the batch / ensemble runner.

The real pipeline (`run_one` -> TradingAgentsGraph.propagate) is never invoked;
tests inject a fake runner.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location("batch", _SCRIPTS / "batch.py")
batch = importlib.util.module_from_spec(_SPEC)
sys.modules["batch"] = batch
_SPEC.loader.exec_module(batch)

pytestmark = pytest.mark.unit


def _ok(rating, side, secs=10.0):
    return {"rating": rating, "side": side, "seconds": secs, "run_dir": None, "error": None}


def _fail(msg="RuntimeError: boom", secs=2.0):
    return {"rating": None, "side": None, "seconds": secs, "run_dir": None, "error": msg}


def test_tier_to_side():
    assert batch.tier_to_side("Buy") == "BUY"
    assert batch.tier_to_side("Overweight") == "BUY"
    assert batch.tier_to_side("Hold") == "HOLD"
    assert batch.tier_to_side("REVIEW") == "HOLD"
    assert batch.tier_to_side("Underweight") == "SELL"
    assert batch.tier_to_side(None) == "HOLD"


def test_load_watchlist(tmp_path):
    wl = tmp_path / "wl.txt"
    wl.write_text("NVDA\n# a comment\naapl, tsla\nNVDA\n\n0700.HK  msft # inline\n")
    assert batch.load_watchlist(str(wl)) == ["NVDA", "AAPL", "TSLA", "0700.HK", "MSFT"]


def test_resolve_profiles_default_and_named(tmp_path):
    assert batch.resolve_profiles(None, None) == {"env": {}}
    got = batch.resolve_profiles("qwen3-8b,qwen3-14b", None)
    assert set(got) == {"qwen3-8b", "qwen3-14b"}
    assert got["qwen3-8b"]["deep_think_llm"] == "qwen3:8b"

    pf = tmp_path / "p.json"
    pf.write_text(json.dumps({"cloud": {"llm_provider": "anthropic", "deep_think_llm": "claude-sonnet-5"}}))
    got = batch.resolve_profiles("cloud", str(pf))
    assert got["cloud"]["llm_provider"] == "anthropic"


def test_resolve_profiles_rejects_unknown():
    with pytest.raises(SystemExit):
        batch.resolve_profiles("nope", None)


def test_build_plan_shape():
    plan = batch.build_plan(["NVDA", "AAPL"], {"a": {}, "b": {}}, 3)
    assert len(plan) == 2 * 2 * 3
    assert plan[0] == {"ticker": "NVDA", "profile": "a", "config": {}, "rep": 1}


def test_execute_flushes_incrementally(tmp_path):
    out = tmp_path / "b.json"
    plan = batch.build_plan(["NVDA", "AAPL"], {"env": {}}, 1)
    seq = iter([_ok("Buy", "BUY"), _fail()])
    res = batch.execute(plan, "2026-09-05", tmp_path, out,
                        runner=lambda *a: next(seq), echo=lambda *a, **k: None)
    assert [r["ticker"] for r in res] == ["NVDA", "AAPL"]
    assert res[1]["error"]
    saved = json.loads(out.read_text())
    assert len(saved["runs"]) == 2  # flushed, survives a crash


def test_summarize_consensus_and_split():
    runs = [
        {"ticker": "NVDA", "profile": "a", "rep": 1, **_ok("Buy", "BUY")},
        {"ticker": "NVDA", "profile": "b", "rep": 1, **_ok("Overweight", "BUY")},
        {"ticker": "AAPL", "profile": "a", "rep": 1, **_ok("Hold", "HOLD")},
        {"ticker": "AAPL", "profile": "b", "rep": 1, **_ok("Sell", "SELL")},
        {"ticker": "TSLA", "profile": "a", "rep": 1, **_ok("Buy", "BUY")},
        {"ticker": "TSLA", "profile": "b", "rep": 1, **_fail()},
    ]
    s = batch.summarize(runs)
    assert s["ok"] == 5 and s["failed"] == 1
    assert s["profiles"] == ["a", "b"]
    assert s["tickers"]["NVDA"]["unanimous"] is True
    assert s["tickers"]["NVDA"]["consensus"] == "BUY"
    assert s["tickers"]["AAPL"]["split"] is True
    # only one successful run -> not called unanimous
    assert s["tickers"]["TSLA"]["unanimous"] is False
    assert s["tickers"]["TSLA"]["failures"] == ["b"]


def test_summarize_stability_across_repeats():
    runs = [
        {"ticker": "NVDA", "profile": "a", "rep": 1, **_ok("Buy", "BUY")},
        {"ticker": "NVDA", "profile": "a", "rep": 2, **_ok("Hold", "HOLD")},
        {"ticker": "NVDA", "profile": "a", "rep": 3, **_ok("Buy", "BUY")},
    ]
    st = batch.summarize(runs)["tickers"]["NVDA"]["stability"]["a"]
    assert st == {"modal": "BUY", "agree": 2, "of": 3}


def test_render_summary_multi_profile_orders_columns():
    runs = [
        {"ticker": "NVDA", "profile": "qwen3-8b", "rep": 1, **_ok("Buy", "BUY")},
        {"ticker": "NVDA", "profile": "qwen3-14b", "rep": 1, **_ok("Sell", "SELL")},
    ]
    out = batch.render_summary("2026-09-05", "1×2×1", batch.summarize(runs))
    assert out.index("qwen3-8b") < out.index("qwen3-14b")  # plan order, not alphabetical
    assert "✗ SELL" in out or "✗ BUY" in out  # split marked


def test_main_dry_run(capsys):
    rc = batch.main(["NVDA", "AAPL", "--profiles", "qwen3-8b,qwen3-14b", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "2 ticker(s) × 2 profile(s) × 1 = 4 runs" in out
    assert "NVDA     qwen3-8b" in out


def test_main_runs_with_fake_runner(tmp_path, monkeypatch, capsys):
    seq = iter([_ok("Buy", "BUY"), _ok("Hold", "HOLD")])
    monkeypatch.setattr(batch, "run_one", lambda *a, **k: next(seq))
    rc = batch.main(["NVDA", "--repeat", "2", "--results-dir", str(tmp_path)])
    assert rc == 0

    batch_files = list((tmp_path / "_batch").glob("*.json"))
    assert len(batch_files) == 1
    saved = json.loads(batch_files[0].read_text())
    assert saved["summary"]["ok"] == 2
    assert saved["summary"]["tickers"]["NVDA"]["stability"]["env"]["of"] == 2


def test_main_bad_date():
    with pytest.raises(SystemExit):
        batch.main(["NVDA", "--date", "09-05-2026"])


def test_main_no_tickers():
    with pytest.raises(SystemExit):
        batch.main([])
