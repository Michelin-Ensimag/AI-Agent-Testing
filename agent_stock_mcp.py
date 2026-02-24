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


# ── LLM ──
llm = ChatOpenAI(
    base_url="http://localhost:4141/v1",
    api_key="dummy-key",
    model="gpt-4.1",
)

SYSTEM_MSG = (
    "You are a helpful stock market assistant. "
    "You have tools that return real data. "
    "When a tool returns data, USE it to answer the user. "
    "At the end of your answer, NEVER suggest a relevant follow-up question or suggestion."
)

MAX_ITERATIONS = 10


async def run_agent():
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
    tools_by_name = {t.name: t for t in tools}

    print("\n── Available MCP tools ──")
    for tool in tools:
        print(f"  • {tool.name}: {tool.description[:65]}...")

    # Bind tools to the LLM so it knows about them
    llm_with_tools = llm.bind_tools(tools)

    print("\n── Agent ready. Examples: ──")
    print("  1. What is the price of Apple (AAPL)?")
    print("  2. Compare the price of LVMH (MC.PA) and Microsoft (MSFT).")
    print("  3. What is the price of Apple (AAPL)? If I invest that amount for 10 years at 8%, how much will I have?")
    print("  4. What is the price of Nvidia stock? Compare LVMH (MC.PA) and Microsoft (MSFT). What is Apple (AAPL)? If I invest that amount for 10 years at 8%, how much will I have?")
    print()

    try:
        question = input("Your question: ")
    except EOFError:
        print("\n❌ No input (run in an interactive terminal).")
        return

    print("\n── Running... ──\n")

    # Build the conversation
    messages = [
        SystemMessage(content=SYSTEM_MSG),
        HumanMessage(content=question),
    ]

    try:
        for i in range(MAX_ITERATIONS):
            # Ask the LLM
            response = await llm_with_tools.ainvoke(messages)
            messages.append(response)

            # If no tool calls, we have the final answer
            if not response.tool_calls:
                break

            # Execute each tool call
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                print(f"  [Calling tool: {tool_name}({tool_args})]")

                tool = tools_by_name[tool_name]
                result = await tool.ainvoke(tool_args)

                # MCP returns content blocks like [{'type':'text','text':'...','id':'...'}]
                # Convert to a plain string so the LLM can read it
                if isinstance(result, list):
                    parts = [
                        block["text"] if isinstance(block, dict) and "text" in block
                        else str(block)
                        for block in result
                    ]
                    result = "\n".join(parts)

                print(f"  [Result: {str(result)[:200]}]")

                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                ))

        print(f"\n── Final answer ──\n{response.content}")

    except (httpx.ConnectError, openai.APIConnectionError):
        print("\n❌ Could not connect to the Copilot proxy (http://localhost:4141).")
        print("   Start the proxy first: bunx @jeffreycao/copilot-api@latest start")


if __name__ == "__main__":
    asyncio.run(run_agent())
