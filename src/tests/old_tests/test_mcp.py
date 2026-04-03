import pytest

from agents.agents_demo.agent_stock_mcp import create_mcp_client


@pytest.mark.asyncio
async def test_mcp_tools_exist():
    """
    Verify that MCP tools 'get_stock_price' and 'calculate_growth'
    are exposed by the MCP server.
    """
    client, tools = await create_mcp_client()
    tool_names = [t.name for t in tools]

    assert "get_stock_price" in tool_names
    assert "calculate_growth" in tool_names
