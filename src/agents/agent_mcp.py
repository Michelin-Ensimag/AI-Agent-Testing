"""
agent_mcp.py — LangChain v1.0 agent with MCP tools (stock + utils)

Uses create_agent from langchain.agents (the unified v1.0 API).
MCP sessions are opened once with AsyncExitStack so the stdio
servers stay alive for the whole session — no restart per tool call.

Requires:
  - mcp_servers/mcp_server_stock.py  (get_stock_price, calculate_growth)
  - mcp_servers/mcp_server_utils.py  (wikipedia_search, get_weather, convert_units)
  - Copilot proxy at http://localhost:4141 (bunx @jeffreycao/copilot-api@latest start)
"""

import asyncio
from contextlib import AsyncExitStack
from pathlib import Path
import httpx
import openai
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from copilot_proxy_utils import wrap_mcp_tool

# Set to False to pass raw MCP content blocks through (e.g. when not using the Copilot proxy).
FLATTEN_MCP_OUTPUT = True

# Set to True to print tool calls and responses during CLI runs.
VERBOSE = True

YELLOW = "\033[33m"
GREEN  = "\033[32m"
RESET  = "\033[0m"


SYSTEM_PROMPT = (
    "You are a data-retrieval assistant with tools for stock prices, weather, Wikipedia, and unit conversion.\n\n"
    "Tool call rules:\n"
    "- If multiple tool calls are independent (no result depends on another), issue them all in the same turn.\n"
    "- If a call depends on the result of a previous one (e.g. you need a city name before fetching weather), chain them sequentially.\n"
    "- Never narrate your process. No 'Let me check' or 'I will look that up'.\n"
    "- Only produce a text response once you have all the data needed to fully answer the question."
)

_MCP_DIR = Path(__file__).parent.parent / "mcp_servers"

MCP_SERVERS = {
    "stock": {
        "command": "python",
        "args": [str(_MCP_DIR / "mcp_server_stock.py")],
        "transport": "stdio",
    },
    "utils": {
        "command": "python",
        "args": [str(_MCP_DIR / "mcp_server_utils.py")],
        "transport": "stdio",
    },
}


def create_llm():
    """Return a ChatOpenAI pointed at the Copilot proxy."""
    return ChatOpenAI(
        base_url="http://localhost:4141/v1",
        api_key="dummy-key",
        model="gpt-5.2", # Change model here
        temperature=0
    )


async def open_mcp_sessions(stack: AsyncExitStack):
    """Open persistent sessions for all MCP servers and return their tools."""
    client = MultiServerMCPClient(MCP_SERVERS)
    tools = []
    for server_name in MCP_SERVERS:
        session = await stack.enter_async_context(client.session(server_name))
        server_tools = await load_mcp_tools(session)
        if FLATTEN_MCP_OUTPUT:
            server_tools = [wrap_mcp_tool(t) for t in server_tools]
        tools.extend(server_tools)
    return tools


async def run_agent(question: str, llm=None, tools=None):
    """
    Run the agent on a single question.

    If `llm` and `tools` are provided (e.g. from tests), reuse them directly.
    Otherwise opens MCP sessions for this call.
    """
    if llm is None:
        llm = create_llm()

    if tools is not None:
        agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)
        result = await agent.ainvoke({"messages": [("user", question)]})
        return result["messages"][-1].content

    async with AsyncExitStack() as stack:
        mcp_tools = await open_mcp_sessions(stack)
        agent = create_agent(model=llm, tools=mcp_tools, system_prompt=SYSTEM_PROMPT)
        result = await agent.ainvoke({"messages": [("user", question)]})
        return result["messages"][-1].content


def _print_tool_trace(messages):
    print("\n── Tool usage ──")
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  {YELLOW}→{RESET} {tc['name']}({tc['args']})")
        elif isinstance(msg, ToolMessage):
            preview = msg.content if msg.content else "(empty)"
            print(f"  {GREEN}←{RESET} {preview}")
    print()


async def run_cli():
    """Interactive CLI — MCP servers stay alive for the whole session."""
    llm = create_llm()

    async with AsyncExitStack() as stack:
        tools = await open_mcp_sessions(stack)

        print("\n── MCP tools ──")
        for t in tools:
            first_line = (t.description or "").split("\n")[0].strip()
            print(f"  • {t.name}: {first_line}")

        print("\n── Examples ──")
        print("  1. Convert 212 Fahrenheit to Celsius.")
        print("  2. What is the current price of LVMH (MC.PA)?")
        print("  3. What is the weather in the city where the Eiffel Tower is located?")
        print("  4. Get the price of AAPL, then calculate how much $5000 grows over 10 years at that % as annual rate.")
        print("  5. Find where Airbus is headquartered, get the weather of their hq, get their stock price, calculate how much €1000 grows over 10 years at 8%, and convert 500 km to miles.")
        print()

        agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)

        try:
            question = input("Your question: ")
        except EOFError:
            print("\nNo input.")
            return

        print("\n── Running... ──")

        try:
            result = await agent.ainvoke({"messages": [("user", question)]})
            if VERBOSE:
                _print_tool_trace(result["messages"])
            answer = result["messages"][-1].content
            print("── Answer ──")
            print(answer)
        except (httpx.ConnectError, openai.APIConnectionError):
            print("\nCan't reach Copilot proxy (http://localhost:4141).")
            print("Start it with: bunx @jeffreycao/copilot-api@latest start")


if __name__ == "__main__":
    asyncio.run(run_cli())
