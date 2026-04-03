"""
tests/unit/test_mcp_server.py

Unit tests for mcp_server_strat_pred.py.
No network calls - all data is mocked.
"""

from unittest.mock import patch

import pandas as pd

from mcp_servers.mcp_server_strat_pred import (
    compute_indicators,
    get_market_data,
    risk_analysis,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


def make_ohlcv(n: int = 60, start_price: float = 150.0) -> dict:
    """Generate a fake OHLCV dataset with n rows."""
    prices = [start_price + i * 0.5 for i in range(n)]
    return {
        "Open": [p - 0.2 for p in prices],
        "High": [p + 1.0 for p in prices],
        "Low": [p - 1.0 for p in prices],
        "Close": prices,
        "Volume": [1_000_000] * n,
    }


def make_flat_ohlcv(n: int = 60, price: float = 100.0) -> dict:
    """Generate a constant-price dataset (for RSI edge cases)."""
    return {
        "Open": [price] * n,
        "High": [price] * n,
        "Low": [price] * n,
        "Close": [price] * n,
        "Volume": [500_000] * n,
    }


# Tests: compute_indicators


class TestComputeIndicators:
    def test_returns_expected_keys(self):
        """Verify that all indicator columns are present."""
        result = compute_indicators(make_ohlcv())
        for key in ["SMA20", "SMA50", "EMA20", "RSI", "MACD", "MACD_signal"]:
            assert key in result, f"Missing key: {key}"

    def test_rsi_bounds(self):
        """RSI should always stay between 0 and 100."""
        result = compute_indicators(make_ohlcv(n=100))
        rsi_values = result["RSI"]
        assert all(0 <= v <= 100 for v in rsi_values), "RSI out of bounds [0, 100]"

    def test_sma20_less_than_sma50_on_downtrend(self):
        """On a downtrend, SMA20 should be < SMA50."""
        prices = [200.0 - i * 1.0 for i in range(80)]
        ohlcv = {
            "Open": prices,
            "High": prices,
            "Low": prices,
            "Close": prices,
            "Volume": [1_000_000] * 80,
        }
        result = compute_indicators(ohlcv)
        last_sma20 = result["SMA20"][-1]
        last_sma50 = result["SMA50"][-1]
        assert last_sma20 < last_sma50, "SMA20 should be < SMA50 on a downtrend"

    def test_sma20_greater_than_sma50_on_uptrend(self):
        """On an uptrend, SMA20 should be > SMA50."""
        result = compute_indicators(make_ohlcv(n=80))
        last_sma20 = result["SMA20"][-1]
        last_sma50 = result["SMA50"][-1]
        assert last_sma20 > last_sma50, "SMA20 should be > SMA50 on an uptrend"

    def test_macd_is_numeric(self):
        """MACD and MACD_signal should be valid floats."""
        result = compute_indicators(make_ohlcv(n=80))
        for val in result["MACD"]:
            assert isinstance(val, float), f"MACD non-float: {val}"
        for val in result["MACD_signal"]:
            assert isinstance(val, float), f"MACD_signal non-float: {val}"

    def test_empty_input_returns_error(self):
        """An empty input should return a dict with an 'error' key."""
        result = compute_indicators({})
        assert "error" in result

    def test_dropna_removes_incomplete_rows(self):
        """With 60 rows, NaNs should be removed after indicator computation."""
        result = compute_indicators(make_ohlcv(n=60))
        df = pd.DataFrame(result)
        assert df.isnull().sum().sum() == 0, "NaN values remain after dropna"

    def test_minimum_rows_required(self):
        """With fewer than 50 rows, result should be empty after dropping NaNs."""
        result = compute_indicators(make_ohlcv(n=30))
        # With only 30 rows, SMA50 is NaN everywhere, so everything is dropped.
        assert len(result.get("Close", [])) == 0


# Tests: risk_analysis


class TestRiskAnalysis:
    def test_returns_expected_keys(self):
        """Verify that the three risk metrics are present."""
        result = risk_analysis(make_ohlcv(n=100))
        for key in ["sharpe_ratio", "max_drawdown", "volatility"]:
            assert key in result, f"Missing key: {key}"

    def test_values_are_floats(self):
        """All metrics should be floats."""
        result = risk_analysis(make_ohlcv(n=100))
        for key, val in result.items():
            assert isinstance(val, float), f"{key} is not a float: {val}"

    def test_max_drawdown_non_negative(self):
        """Max drawdown should always be >= 0."""
        result = risk_analysis(make_ohlcv(n=100))
        assert result["max_drawdown"] >= 0

    def test_volatility_positive_on_varying_prices(self):
        """Volatility should be > 0 when prices vary."""
        result = risk_analysis(make_ohlcv(n=100))
        assert result["volatility"] > 0

    def test_sharpe_positive_on_uptrend(self):
        """On an uptrend, Sharpe ratio should be positive."""
        result = risk_analysis(make_ohlcv(n=100, start_price=100.0))
        assert result["sharpe_ratio"] > 0

    def test_rounded_to_4_decimals(self):
        """Values should be rounded to 4 decimals."""
        result = risk_analysis(make_ohlcv(n=100))
        for key, val in result.items():
            assert val == round(val, 4), f"{key} is not rounded to 4 decimals"


# Tests: get_market_data (mocked)


class TestGetMarketData:
    @patch("mcp_servers.mcp_server_strat_pred.yf.download")
    def test_returns_dict_with_ohlcv_keys(self, mock_download):
        """Verify that get_market_data returns a dict with OHLCV keys."""
        mock_df = pd.DataFrame(
            {
                "Open": [150.0, 151.0],
                "High": [152.0, 153.0],
                "Low": [149.0, 150.0],
                "Close": [151.0, 152.0],
                "Volume": [1_000_000, 1_100_000],
            },
            index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        )
        mock_download.return_value = mock_df

        result = get_market_data("AAPL")

        for key in ["Open", "High", "Low", "Close", "Volume"]:
            assert key in result, f"Missing key: {key}"

    @patch("mcp_servers.mcp_server_strat_pred.yf.download")
    def test_index_converted_to_string(self, mock_download):
        """Datetime index should be converted to string for JSON serialization."""
        mock_df = pd.DataFrame(
            {
                "Open": [150.0],
                "High": [151.0],
                "Low": [149.0],
                "Close": [150.5],
                "Volume": [1_000_000],
            },
            index=pd.to_datetime(["2024-01-01"]),
        )
        mock_download.return_value = mock_df

        result = get_market_data("AAPL")
        # Result should be serializable (no Timestamp index objects).
        import json

        json.dumps(result)  # Should not raise an exception.

    @patch("mcp_servers.mcp_server_strat_pred.yf.download")
    def test_empty_ticker_returns_empty(self, mock_download):
        """An invalid ticker should return an empty DataFrame -> empty dict."""
        mock_download.return_value = pd.DataFrame()
        result = get_market_data("INVALID_TICKER_XYZ")
        assert result == {} or all(len(v) == 0 for v in result.values())
