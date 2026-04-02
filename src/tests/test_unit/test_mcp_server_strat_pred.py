"""
tests/unit/test_mcp_server.py

Tests unitaires pour mcp_server_strat_pred.py.
Aucun appel réseau — toutes les données sont mockées.
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
    """Génère un faux dataset OHLCV avec n lignes."""
    prices = [start_price + i * 0.5 for i in range(n)]
    return {
        "Open": [p - 0.2 for p in prices],
        "High": [p + 1.0 for p in prices],
        "Low": [p - 1.0 for p in prices],
        "Close": prices,
        "Volume": [1_000_000] * n,
    }


def make_flat_ohlcv(n: int = 60, price: float = 100.0) -> dict:
    """Génère un dataset avec prix constants (pour tester les cas limites RSI)."""
    return {
        "Open": [price] * n,
        "High": [price] * n,
        "Low": [price] * n,
        "Close": [price] * n,
        "Volume": [500_000] * n,
    }


# ── Tests : compute_indicators ────────────────────────────────────────────────


class TestComputeIndicators:
    def test_returns_expected_keys(self):
        """Vérifie que toutes les colonnes d'indicateurs sont présentes."""
        result = compute_indicators(make_ohlcv())
        for key in ["SMA20", "SMA50", "EMA20", "RSI", "MACD", "MACD_signal"]:
            assert key in result, f"Clé manquante : {key}"

    def test_rsi_bounds(self):
        """Le RSI doit toujours être compris entre 0 et 100."""
        result = compute_indicators(make_ohlcv(n=100))
        rsi_values = result["RSI"]
        assert all(0 <= v <= 100 for v in rsi_values), "RSI hors bornes [0, 100]"

    def test_sma20_less_than_sma50_on_downtrend(self):
        """Sur une tendance baissière, SMA20 doit être < SMA50."""
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
        assert last_sma20 < last_sma50, (
            "SMA20 devrait être < SMA50 en tendance baissière"
        )

    def test_sma20_greater_than_sma50_on_uptrend(self):
        """Sur une tendance haussière, SMA20 doit être > SMA50."""
        result = compute_indicators(make_ohlcv(n=80))
        last_sma20 = result["SMA20"][-1]
        last_sma50 = result["SMA50"][-1]
        assert last_sma20 > last_sma50, (
            "SMA20 devrait être > SMA50 en tendance haussière"
        )

    def test_macd_is_numeric(self):
        """MACD et MACD_signal doivent être des flottants valides."""
        result = compute_indicators(make_ohlcv(n=80))
        for val in result["MACD"]:
            assert isinstance(val, float), f"MACD non-float : {val}"
        for val in result["MACD_signal"]:
            assert isinstance(val, float), f"MACD_signal non-float : {val}"

    def test_empty_input_returns_error(self):
        """Un input vide doit retourner un dict avec une clé 'error'."""
        result = compute_indicators({})
        assert "error" in result

    def test_dropna_removes_incomplete_rows(self):
        """Avec 60 lignes, les NaN (SMA50 besoin de 50 pts) doivent être supprimés."""
        result = compute_indicators(make_ohlcv(n=60))
        df = pd.DataFrame(result)
        assert df.isnull().sum().sum() == 0, "Des NaN subsistent après dropna"

    def test_minimum_rows_required(self):
        """Avec moins de 50 lignes, le résultat devrait être vide (tout NaN droppé)."""
        result = compute_indicators(make_ohlcv(n=30))
        # Avec seulement 30 lignes, SMA50 = NaN partout → tout est droppé
        assert len(result.get("Close", [])) == 0


# ── Tests : risk_analysis ─────────────────────────────────────────────────────


class TestRiskAnalysis:
    def test_returns_expected_keys(self):
        """Vérifie que les 3 métriques de risque sont présentes."""
        result = risk_analysis(make_ohlcv(n=100))
        for key in ["sharpe_ratio", "max_drawdown", "volatility"]:
            assert key in result, f"Clé manquante : {key}"

    def test_values_are_floats(self):
        """Toutes les métriques doivent être des flottants."""
        result = risk_analysis(make_ohlcv(n=100))
        for key, val in result.items():
            assert isinstance(val, float), f"{key} n'est pas un float : {val}"

    def test_max_drawdown_non_negative(self):
        """Le max drawdown doit toujours être >= 0."""
        result = risk_analysis(make_ohlcv(n=100))
        assert result["max_drawdown"] >= 0

    def test_volatility_positive_on_varying_prices(self):
        """La volatilité doit être > 0 si les prix varient."""
        result = risk_analysis(make_ohlcv(n=100))
        assert result["volatility"] > 0

    def test_sharpe_positive_on_uptrend(self):
        """Sur une tendance haussière, le Sharpe ratio doit être positif."""
        result = risk_analysis(make_ohlcv(n=100, start_price=100.0))
        assert result["sharpe_ratio"] > 0

    def test_rounded_to_4_decimals(self):
        """Les valeurs doivent être arrondies à 4 décimales."""
        result = risk_analysis(make_ohlcv(n=100))
        for key, val in result.items():
            assert val == round(val, 4), f"{key} non arrondi à 4 décimales"


# ── Tests : get_market_data (mocké) ──────────────────────────────────────────


class TestGetMarketData:
    @patch("mcp_servers.mcp_server_strat_pred.yf.download")
    def test_returns_dict_with_ohlcv_keys(self, mock_download):
        """Vérifie que get_market_data retourne bien un dict avec les clés OHLCV."""
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
            assert key in result, f"Clé manquante : {key}"

    @patch("mcp_servers.mcp_server_strat_pred.yf.download")
    def test_index_converted_to_string(self, mock_download):
        """L'index datetime doit être converti en string pour la sérialisation JSON."""
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
        # Le résultat doit être sérialisable (pas d'index Timestamp)
        import json

        json.dumps(result)  # Ne doit pas lever d'exception

    @patch("mcp_servers.mcp_server_strat_pred.yf.download")
    def test_empty_ticker_returns_empty(self, mock_download):
        """Un ticker invalide renvoie un DataFrame vide → dict vide."""
        mock_download.return_value = pd.DataFrame()
        result = get_market_data("INVALID_TICKER_XYZ")
        assert result == {} or all(len(v) == 0 for v in result.values())
