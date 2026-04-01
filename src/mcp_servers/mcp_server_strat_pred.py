"""
mcp_server_strat_pred.py - general-purpose MCP tools for financial data

Provides tools for an AI agent to fetch market data, compute technical
indicators, and analyze risk metrics.

Run standalone:
    python mcp_server_strat_pred.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
from fastmcp import FastMCP

# Create the MCP server
mcp = FastMCP("Strategie Prediction Server")


# Tool : Fetch market data
@mcp.tool()
def get_market_data(
    ticker: str,
    interval: str = "1d",
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
) -> dict:
    """
    Fetch historical OHLCV data using yfinance.

    Args:
        ticker: Stock symbol, e.g., 'AAPL' for apple
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        interval : The intervals in which the data is returned , e.g.,"1wk", "1m" , "1d"

    Returns:
        the  market data as a dictionary of lists
        (keys are :
                Open : opening price
                High : highest price
                Low : lowest price
                Close : closing price
                Adj Close : adjusted price
                Volume : exchanged volume
    """
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    data.dropna(inplace=True)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Convertir l'index datetime en string pour la sérialisation JSON
    data.index = data.index.astype(str)

    return data.to_dict(orient="list")


# Tool : Compute technical indicators
@mcp.tool()
def compute_indicators(ohlcv: dict) -> dict:
    """
    Compute SMA, EMA, RSI, MACD from OHLCV data.
    SMA  : moyenne des prix sur les n derniers jours
    EMA  : aussi une moyenne sur les prix mais donne plus de poids aux récents
    RSI  : mesure la force du mouvement du prix ( ratio entre les hausses moyennes et baisses moyennes)
    MACD : compare deux EMA

    Args:
        ohlcv: dict with keys ['Open', 'High', 'Low', 'Close', 'Volume']

    Returns:
        Updated dict with added indicators
    """

    if not ohlcv:
        return {"error": "ohlcv data missing. Call get_market_data first."}

    df = pd.DataFrame(ohlcv)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # SMA and EMA
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # SMA signal
    df["SMA_Signal"] = np.where(df["SMA20"] > df["SMA50"], "Bullish", "Bearish")

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df.dropna(inplace=True)
    return df.to_dict(orient="list")


# Tool : Risk analysis
@mcp.tool()
def risk_analysis(ohlcv: dict) -> dict:
    """
    Compute risk metrics: Sharpe ratio, max drawdown, annualized volatility.

    Args:
        ohlcv: dict with 'Close' prices

    Returns:
        Dictionary with risk metrics
    """
    df = pd.DataFrame(ohlcv)
    df["returns"] = df["Close"].pct_change()

    sharpe = (df["returns"].mean() / df["returns"].std()) * np.sqrt(252)
    equity_curve = (1 + df["returns"]).cumprod()
    max_dd = (equity_curve.cummax() - equity_curve).max()
    volatility = df["returns"].std() * np.sqrt(252)

    return {
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "volatility": round(volatility, 4),
    }


@mcp.tool()
def analyze_stock(
    ticker: str,
    interval: str = "1d",
    start_date: str = "2024-01-01",
    end_date: str = "2026-03-17",
) -> dict:
    """
    Fetch market data AND compute indicators for a given ticker.
    Returns OHLCV + SMA, EMA, RSI, MACD ready for strategy generation.

    Args:
        ticker: Stock symbol, e.g. 'AAPL'
        interval: '1d', '1wk', '1m'
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD
    """
    # Étape 1 : fetch
    ohlcv = get_market_data(ticker, interval, start_date, end_date)

    # Étape 2 : indicators
    result = compute_indicators(ohlcv)

    # Étape 3 : ne retourner que les 50 dernières lignes (suffisant pour l'agent)
    df = pd.DataFrame(result).tail(50)
    return df.to_dict(orient="list")


# Run MCP server
if __name__ == "__main__":
    mcp.run(show_banner=False)
