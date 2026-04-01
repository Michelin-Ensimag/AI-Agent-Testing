import pytest
from agents.agents_demo.agent_stock_mcp import create_mcp_client


@pytest.mark.asyncio
async def test_mcp_tools_exist():
    """
    Vérifie que les outils MCP 'get_stock_price' et 'calculate_growth'
    sont bien exposés par le serveur MCP.
    """
    client, tools = await create_mcp_client()
    tool_names = [t.name for t in tools]

    assert "get_stock_price" in tool_names
    assert "calculate_growth" in tool_names
