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
    "You are an expert quantitative trading evaluator.\n"
    "Your role is NOT to generate a strategy, but to EVALUATE the logic and quality of a given strategy using only information available up to "+date+"\n\n"

    " CRITICAL DATA HANDLING RULES !!:\n"
    "- The strategy agent is equipped with live data tools. TREAT ALL NUMBERS mentioned (RSI, SMA, MACD prices) as ABSOLUTE FACTS retrieved from tools.\n"
    "- DO NOT fail or penalize a strategy because you don't see the raw data in your initial prompt. Source verification is NOT your priority.\n"
    "- NEVER mark a strategy as 'ungrounded' if it cites specific technical values. Assume these are the correct inputs for this specific trade.\n\n"

    "Your task is to critically assess the strategy using financial reasoning based ONLY on those values.\n\n"

    "Evaluation criteria:\n"
    "- LOGICAL CONSISTENCY: Does the decision (BUY/SELL/HOLD) actually match the stated indicators? (e.g., If RSI is 20, is a BUY justified?)\n"
    "- INDICATOR INTERPRETATION: Are technical levels (Overbought/Oversold/Crossovers) correctly understood?\n"
    "- RISK MANAGEMENT: Are stop loss and take profit levels realistic relative to the current price and volatility mentioned?\n"
    "- COHERENCE: Is the reasoning free of internal contradictions (like saying it's a bull trend while SMA20 < SMA50)?\n\n"
    "If the Input is a simple user question without numeric data, IGNORE the "
    "'consistency check versus Input'. Instead, judge the decision based ONLY on the numbers"
    " the agent claims to have found. Assume the agent's numbers are the ground truth for this evaluation."


    "Tool Usage:\n"
    "- Use 'get_market_data' ONLY if you suspect a gross mathematical impossibility (e.g., a stock price that doesn't exist) or if you need broader context to judge the risk.\n\n"

    "Output format:\n"
    "1. Summary of the evaluated strategy\n"
    "2. Strengths (Focus on logic)\n"
    "3. Weaknesses (Focus on flaws in reasoning or risk)\n"
    "4. Final judgment: GOOD / AVERAGE / BAD\n"
    "5. Short justification (Focus ONLY on the quality of the decision, ignore the data source issue)\n\n"

  

    "Be strict, objective, and critical. A good strategy must be logically bulletproof."
)
    

MAX_ITERATIONS = 10


# LLM Creation 
def create_llm():
    return ChatOpenAI(
        base_url="http://localhost:4141/v1",
        api_key="dummy-key",
        model="gpt-4.1",
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

    allowed_tools = ["get_market_data"]
    filtered_tools = [ t for t in tools if t.name in allowed_tools ]

    tools_by_name = {t.name: t for t in filtered_tools}
    llm_with_tools = llm.bind_tools(filtered_tools)
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