"""
mcp_server_stock.py — MCP server exposing stock market tools

Exposes two tools via the Model Context Protocol (MCP):
  - get_stock_price: fetches real-time stock data via yfinance (no API key needed)
  - calculate_growth: compound interest calculation

Run standalone to test:
    python mcp_server_stock.py

The agent (agent_stock_mcp.py) launches this as a subprocess
and communicates with it over stdio using the MCP protocol.
"""

import yfinance as yf
from fastmcp import FastMCP

mcp = FastMCP("Stock Agent Server")


@mcp.tool()
def get_stock_price(ticker: str) -> dict:
    """
    Get the current stock price and basic info for a given ticker symbol.

    Args:
        ticker: Stock ticker symbol, e.g. 'AAPL', 'MSFT', 'MC.PA' (LVMH), 'AIR.PA' (Airbus)

    Returns:
        A dict with company name, current price, currency, and 52-week range.
    """
    stock = yf.Ticker(ticker)
    info = stock.info

    price = info.get("currentPrice") or info.get("regularMarketPrice")
    if price is None:
        return {
            "error": f"Could not retrieve price for ticker '{ticker}'. Check the symbol."
        }

    return {
        "ticker": ticker.upper(),
        "company": info.get("longName", ticker),
        "price": round(price, 2),
        "currency": info.get("currency", "USD"),
        "52w_low": info.get("fiftyTwoWeekLow"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
    }


@mcp.tool()
def calculate_growth(principal: float, annual_rate_percent: float, years: int) -> dict:
    """
    Calculate the future value of an investment using compound interest.

    Formula: A = P * (1 + r)^n
      where P = principal, r = annual rate (as decimal), n = years

    Args:
        principal: Initial investment amount (e.g. 1000.0)
        annual_rate_percent: Annual growth rate in percent (e.g. 7.5 means 7.5%)
        years: Number of years to grow

    Returns:
        A dict with the final amount, total gain, and a year-by-year breakdown.
    """
    rate = annual_rate_percent / 100
    final_amount = principal * (1 + rate) ** years
    gain = final_amount - principal

    # Year-by-year breakdown (useful context for the agent)
    breakdown = [
        {"year": y, "value": round(principal * (1 + rate) ** y, 2)}
        for y in range(1, min(years + 1, 11))  # cap at 10 rows to keep output readable
    ]

    return {
        "principal": principal,
        "annual_rate_percent": annual_rate_percent,
        "years": years,
        "final_amount": round(final_amount, 2),
        "total_gain": round(gain, 2),
        "gain_percent": round((gain / principal) * 100, 2),
        "yearly_breakdown": breakdown,
    }


if __name__ == "__main__":
    mcp.run(show_banner=False)
