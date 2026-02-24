"""
agent_stock_mcp.py — LangChain agent with MCP tools + Copilot Proxy

This file implements a simple ReAct-style agent loop:
  1. The user asks a stock-related question
  2. The LLM decides which MCP tool(s) to call (get_stock_price, calculate_growth)
  3. Tool results are fed back to the LLM
  4. The LLM produces a final answer

Requires:
  - mcp_server_stock.py running as a subprocess (launched automatically via stdio)
  - Copilot proxy at http://localhost:4141  (bunx @jeffreycao/copilot-api@latest start)
"""

import asyncio
import httpx
import openai
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage


# Configuration
SYSTEM_MSG = (
    "You are a helpful stock market assistant. "
    "You have tools that return real data. "
    "When a tool returns data, USE it to answer the user. "
    "At the end of your answer, NEVER suggest a relevant follow-up question or suggestion."
)

MAX_ITERATIONS = 10


# LLM 
def create_llm():
    """Create and return configured ChatOpenAI instance."""
    return ChatOpenAI(
        base_url="http://localhost:4141/v1",
        api_key="dummy-key",
        model="gpt-4.1",
    )


# MCP 
async def create_mcp_client():
    """Start MCP server and retrieve available tools."""
    client = MultiServerMCPClient(
        {
            "stock": {
                "command": "python",
                "args": ["mcp_server_stock.py"],
                "transport": "stdio",
            }
        }
    )

    tools = await client.get_tools()
    return client, tools


# Agent Logic 
async def run_agent_logic(question: str, llm, tools, max_iterations=MAX_ITERATIONS):

    tools_by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(content=SYSTEM_MSG),
        HumanMessage(content=question),
    ]

    for _ in range(max_iterations):
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)

        # If no tool calls → final answer
        if not response.tool_calls:
            return response.content

        # Execute tool calls
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            tool = tools_by_name[tool_name]
            result = await tool.ainvoke(tool_args)

            # Convert MCP content blocks to string
            if isinstance(result, list):
                result = "\n".join(
                    block.get("text", str(block))
                    if isinstance(block, dict)
                    else str(block)
                    for block in result
                )

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                )
            )

    return "Max iterations reached."


# ─────────────────────────────────────────────
# CLI Interface
# ─────────────────────────────────────────────

async def run_cli():
    """Interactive CLI entrypoint."""

    client, tools = await create_mcp_client()
    llm = create_llm()

    print("\n── Available MCP tools ──")
    for tool in tools:
        print(f"  • {tool.name}: {tool.description[:65]}...")

    print("\n── Agent ready. Examples: ──")
    print("  1. What is the price of Apple (AAPL)?")
    print("  2. Compare the price of LVMH (MC.PA) and Microsoft (MSFT).")
    print("  3. What is Apple (AAPL)? If I invest that amount for 10 years at 8%, how much will I have?")
    print()

    try:
        question = input("Your question: ")
    except EOFError:
        print("\nNo input detected.")
        return

    print("\n── Running... ──\n")

    try:
        answer = await run_agent_logic(question, llm, tools)

        print("\n── Final answer ──\n")
        print(answer)

    except (httpx.ConnectError, openai.APIConnectionError):
        print("\nCould not connect to Copilot proxy (http://localhost:4141).")
        print("Start it with: bunx @jeffreycao/copilot-api@latest start")


if __name__ == "__main__":
    asyncio.run(run_cli())