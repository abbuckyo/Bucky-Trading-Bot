# -*- coding: utf-8 -*-
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.blend_sleeve import evaluate_signal_health, signal_health


def _generate_returns(n_days: int = 200, daily_mean: float = 0.0015, daily_vol: float = 0.004) -> pd.Series:
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    rets = np.random.normal(daily_mean, daily_vol, n_days)
    return pd.Series(rets, index=dates, name="test_returns")


class TestSignalHealth:
    def test_evaluate_signal_health_active(self):
        rets = _generate_returns(n_days=200, daily_mean=0.0015, daily_vol=0.004)
        health = evaluate_signal_health(rets)

        assert health["status"] == "ACTIVE"
        assert health["k"] == 1.0
        assert health["k_state"] == 1.0
        assert health["k_dd"] == 1.0
        assert health["n_warnings"] == 0
        assert health["s_sr"] == 0
        assert health["drawdown_breach"] is False

    def test_no_watch_on_sr60_alone(self):
        # Spec Requirement 1: Do NOT trigger WATCH on SR_60 alone (SE=2.05).
        # Construct series where SR_60 < 0.516 but SR_120 >= 0.670.
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=200, freq="B")
        rets = pd.Series(np.random.normal(0.0015, 0.004, 200), index=dates)

        # Flat noise on the last 60 days to depress SR_60 while keeping prior 60 strong
        noise60 = np.random.normal(0, 0.006, 60)
        noise60 -= noise60.mean()
        rets.iloc[-60:] = noise60 + 0.00015

        health = evaluate_signal_health(rets)
        assert health["sharpe_60d"] < 0.516
        assert health["sharpe_120d"] >= 0.670
        assert health["s_sr"] == 0
        assert health["status"] == "ACTIVE"
        assert health["k"] == 1.0

    def test_joint_sharpe_watch(self):
        # Require (SR_60 < 0.516 AND SR_120 < 0.670) for WATCH (s_SR=1, k=0.75)
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=200, freq="B")
        rets = pd.Series(np.random.normal(0.0015, 0.004, 200), index=dates)

        noise120 = np.random.normal(0, 0.006, 120)
        noise120 -= noise120.mean()
        rets.iloc[-120:] = noise120 + 0.00025

        health = evaluate_signal_health(rets)
        assert health["sharpe_60d"] < 0.516
        assert health["sharpe_120d"] < 0.670
        assert health["s_sr"] == 1
        assert health["status"] == "WATCH"
        assert health["n_warnings"] == 1
        assert health["k"] == 0.75

    def test_sharpe_s2_warning(self):
        # Require (SR_60 < 0.258 AND SR_120 < 0.412) for s_SR=2 (k=0.50)
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=200, freq="B")
        rets = pd.Series(np.random.normal(0.0015, 0.004, 200), index=dates)

        noise120 = np.random.normal(0, 0.006, 120)
        noise120 -= noise120.mean()
        rets.iloc[-120:] = noise120 + 0.00010

        health = evaluate_signal_health(rets)
        assert health["sharpe_60d"] < 0.258
        assert health["sharpe_120d"] < 0.412
        assert health["s_sr"] == 2
        assert health["status"] == "WATCH"
        assert health["n_warnings"] == 2
        assert health["k"] == 0.50

    def test_k_dd_clipping_and_drawdown_watch(self):
        # Test drawdown warning (D_warn = -0.1238, D_stop = -0.1485)
        rets = _generate_returns(n_days=250, daily_mean=0.002, daily_vol=0.004)
        # Induce a ~13.5% drawdown (between D_warn and D_stop)
        rets.iloc[-7:] = -0.02
        health = evaluate_signal_health(rets)

        assert health["current_drawdown"] <= -0.1238
        assert health["current_drawdown"] > -0.1485
        assert health["drawdown_breach"] is False
        assert 0.0 < health["k_dd"] < 1.0
        assert health["status"] == "WATCH"
        assert health["k"] == min(health["k_state"], health["k_dd"])

    def test_evaluate_signal_health_review_drawdown_breach(self):
        # Drawdown <= D_stop (-0.1485) -> unconditional de-risk k = 0.0
        rets = _generate_returns(n_days=150, daily_mean=0.001, daily_vol=0.005)
        rets.iloc[-5:] = -0.04  # ~-18.5% drop
        health = evaluate_signal_health(rets)

        assert health["current_drawdown"] <= -0.1485
        assert health["drawdown_breach"] is True
        assert health["status"] == "REVIEW"
        assert health["k"] == 0.0
        assert health["k_state"] == 0.0
        assert health["k_dd"] == 0.0

    def test_evaluate_signal_health_empty(self):
        empty_s = pd.Series([], dtype=float)
        health = evaluate_signal_health(empty_s)
        assert health["status"] == "WATCH"
        assert health["k"] == 0.5
        assert health["drawdown_breach"] is False

    def test_signal_health_export_custom_path(self, tmp_path: Path):
        rets = _generate_returns(n_days=150, daily_mean=0.001, daily_vol=0.004)
        out_file = tmp_path / "telemetry.json"

        res = signal_health(returns=rets, export_path=out_file)
        assert out_file.exists()

        with open(out_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["status"] == res["status"]
        assert data["k"] == res["k"]
        assert "signal_health" in data
        assert "k" in data["signal_health"]
        assert data["signal_health"]["k"] == res["k"]
        assert "timestamp" in data

    def test_docs_telemetry_file_exists_and_valid(self):
        telemetry_path = Path(__file__).resolve().parents[1] / "docs" / "telemetry.json"
        assert telemetry_path.exists(), "docs/telemetry.json must exist"

        with open(telemetry_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "status" in data
        assert data["status"] in {"ACTIVE", "WATCH", "REVIEW"}
        assert "signal_health" in data
        assert "k" in data["signal_health"]
        assert 0.0 <= data["signal_health"]["k"] <= 1.0
        assert "sharpe_60d" in data["signal_health"]
        assert "sharpe_120d" in data["signal_health"]
        assert "current_drawdown" in data["signal_health"]
        assert "drawdown_breach" in data["signal_health"]
