"""
agent_strategy_mcp.py — LangChain agent with MCP tools for trading strategy

This agent:
 1. Receives a question about a stock
 2. Uses MCP tools to retrieve market data
 3. Computes technical indicators
 4. Analyzes the indicators
 5. Generates a trading strategy

Requires:
  - mcp_server_stock.py running as subprocess
"""

import asyncio
from pathlib import Path
import httpx
import openai

from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

date = "2024-01-03"


SYSTEM_MSG = (
    "We are the "+date+
    "You are a quantitative trading assistant. "
    "You have access to tools that retrieve market data and compute indicators. "
    "Use the tools to analyze the stock and generate a trading strategy.\n\n"
    
    "To analyze a stock:\n"
    "1. Call analyze_stock(ticker) — this fetches data AND computes all indicators.\n"
    "2. Optionally call risk_analysis with the raw OHLCV if needed.\n"
    "3. Use the results to generate a trading strategy.\n\n"

    "When analyzing indicators:\n"
    " SMA TREND:"
    " If SMA20 is numerically GREATER than SMA50 (SMA20 > SMA50), it is BULLISH."
    " If SMA20 is numerically LOWER than SMA50 (SMA20 < SMA50), it is BEARISH."
    "- RSI > 70 means overbought\n"
    "- RSI < 30 means oversold\n"
    "- MACD crossing above signal indicates buy momentum\n"
    "- MACD crossing below signal indicates sell momentum\n\n"
    
    "Your final answer must contain:\n"
    "1. Trend analysis\n"
    "2. Indicator interpretation\n"
    "3. Trading decision (BUY / SELL / HOLD)\n"
    "4. Suggested stop loss\n"
    "5. Suggested take profit\n"
)


MAX_ITERATIONS = 10


# LLM Creation 
def create_llm():
    return ChatOpenAI(
        base_url="http://localhost:4141/v1",
        api_key="dummy-key",
        model="gpt-5.2",
    )


# MCP Client
async def create_mcp_client():

    server_path = Path(__file__).parent.parent / "mcp_servers" / "mcp_server_strat_pred.py"
    client = MultiServerMCPClient(
        {
            "stock": {
                "command": "python",
                "args": [str(server_path)],
                "transport": "stdio",
            }
        }
    )

    tools = await client.get_tools()
    return client, tools



async def run_agent_logic(question, llm, tools, max_iterations=MAX_ITERATIONS):
    
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




async def run_cli():

    client, tools = await create_mcp_client()
    llm = create_llm()

    print("\n── Available MCP tools ──")

    for tool in tools:
        print(f"  • {tool.name}: {tool.description[:65]}...")

    print("\n── Agent ready ──")
    print("Example questions:\n")
    print("  1. Generate a trading strategy for Apple (AAPL)")
    print("  2. Analyze Tesla (TSLA) and suggest a trading strategy")
    print("  3. Based on indicators, should I buy or sell Microsoft (MSFT)?")
    print()

    try:
        question = input("Your question: ")
    except EOFError:
        print("\nNo input detected.")
        return

    print("\n── Running agent... ──\n")

    try:

        answer = await run_agent_logic(question, llm, tools)

        print("\n── Final strategy ──\n")
        print(answer)

    except (httpx.ConnectError, openai.APIConnectionError):

        print("\nCould not connect to Copilot proxy (http://localhost:4141).")
        print("Start it with:")
        print("bunx @jeffreycao/copilot-api@latest start")


if __name__ == "__main__":
    asyncio.run(run_cli())