"""
agent_mcp.py — LangChain v1.0 agent with MCP tools (stock + utils)

Uses create_agent from langchain.agents (the unified v1.0 API).
MCP sessions are opened once with AsyncExitStack so the stdio
servers stay alive for the whole session — no restart per tool call.

Requires:
  - mcp_server_stock.py  (get_stock_price, calculate_growth)
  - mcp_server_utils.py  (wikipedia_search, get_weather, convert_units)
  - Copilot proxy at http://localhost:4141 (bunx @jeffreycao/copilot-api@latest start)
"""

import asyncio
from contextlib import AsyncExitStack
import httpx
import openai
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.tools import StructuredTool

YELLOW = "\033[33m"
GREEN  = "\033[32m"
RESET  = "\033[0m"


# MCP tools return [{"type":"text","text":"..."}] instead of plain strings.
# The copilot proxy can't handle that, so we unwrap it here.
# Set to False to pass raw MCP output through.
FLATTEN_MCP_OUTPUT = True


def _flatten(result):
    if isinstance(result, list):
        return "\n".join(
            b["text"] if isinstance(b, dict) and "text" in b else str(b)
            for b in result
        )
    return str(result)


def _wrap_tool(tool):
    orig = tool.coroutine or tool.func

    if tool.coroutine:
        async def _flat(**kw):
            return _flatten(await orig(**kw))
    else:
        def _flat(**kw):
            return _flatten(orig(**kw))

    return StructuredTool.from_function(
        func=_flat if not tool.coroutine else None,
        coroutine=_flat if tool.coroutine else None,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
    )


SYSTEM_PROMPT = (
    "You are a data-retrieval assistant with tools for stock prices, weather, Wikipedia, and unit conversion.\n\n"
    "Tool call rules:\n"
    "- If multiple tool calls are independent (no result depends on another), issue them all in the same turn.\n"
    "- If a call depends on the result of a previous one (e.g. you need a city name before fetching weather), chain them sequentially.\n"
    "- Never narrate your process. No 'Let me check' or 'I will look that up'.\n"
    "- Only produce a text response once you have all the data needed to fully answer the question."
)

MCP_SERVERS = {
    "stock": {
        "command": "python",
        "args": ["mcp_server_stock.py"],
        "transport": "stdio",
    },
    "utils": {
        "command": "python",
        "args": ["mcp_server_utils.py"],
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
            server_tools = [_wrap_tool(t) for t in server_tools]
        tools.extend(server_tools)
    return tools


async def run_agent(question: str, llm=None, tools=None, verbose=False):
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
        return _extract_answer(result, verbose=verbose)

    async with AsyncExitStack() as stack:
        mcp_tools = await open_mcp_sessions(stack)
        agent = create_agent(model=llm, tools=mcp_tools, system_prompt=SYSTEM_PROMPT)
        result = await agent.ainvoke({"messages": [("user", question)]})
        return _extract_answer(result, verbose=verbose)


def _extract_answer(result, verbose=False):
    """Pull the final text out of the agent result. Optionally print tool usage."""
    messages = result.get("messages", [])

    if verbose:
        print("\n── Tool usage ──")
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"  {YELLOW}→{RESET} {tc['name']}({tc['args']})")
            elif isinstance(msg, ToolMessage):
                preview = msg.content if msg.content else "(empty)"
                print(f"  {GREEN}←{RESET} {preview}")
        print()

    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            return msg.content
    return "No answer produced."


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
            answer = _extract_answer(result, verbose=True)
            print("── Answer ──")
            print(answer)
        except (httpx.ConnectError, openai.APIConnectionError):
            print("\nCan't reach Copilot proxy (http://localhost:4141).")
            print("Start it with: bunx @jeffreycao/copilot-api@latest start")


if __name__ == "__main__":
    asyncio.run(run_cli())
