import asyncio
from agents.agents_demo.agent_stock_mcp import run_agent_logic


class testLLM:
    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        class Response:
            tool_calls = []
            content = "Final test answer"

        return Response()


def test_agent_logic_simple():
    result = asyncio.run(run_agent_logic("Hello", testLLM(), tools=[]))
    assert result == "Final test answer"
