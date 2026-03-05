import pytest 
from mcp_servers.mcp_server_stock import get_stock_price, calculate_growth

def test_calculate_growth_basic():
    result = calculate_growth(
        principal=1000,
        annual_rate_percent=8,
        years=10
    )

    assert result["final_amount"] == 2158.92
    assert result["total_gain"] == 1158.92
    assert result["gain_percent"] == 115.89
    assert len(result["yearly_breakdown"]) == 10
    
    
def test_get_stock_price_real():
    result = get_stock_price("AAPL")

    assert "ticker" in result
    assert result["ticker"] == "AAPL"
    assert "price" in result
    # print(result["price"])
    assert result["price"] is not None
    
if __name__== "__main__":
    test_calculate_growth_basic()
    test_get_stock_price_real()
    




