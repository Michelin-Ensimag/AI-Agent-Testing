import pytest

from agents.agents_demo.agent_stock_mcp import run_cli


@pytest.mark.asyncio
async def test_cli_runs(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "Hello")
    await run_cli()
