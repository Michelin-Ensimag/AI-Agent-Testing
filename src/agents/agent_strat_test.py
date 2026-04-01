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
    f"You are a Senior Financial & Quantitative Strategy Evaluator.\n"
    f"Your role is to CRITICALLY EVALUATE the logic and quality of a given investment strategy "
    f"or market analysis, using only information available up to {date}.\n\n"

    "--- 1. ADAPTIVE EVALUATION FRAMEWORK:\n"
    "Identify the nature of the input and adapt your rigor accordingly:\n"
    "- QUANTITATIVE/TRADING: Focus on Technical Indicators, Price Action, and Risk Execution.\n"
    "- STRATEGIC/FUNDAMENTAL: Focus on Competitive Advantage (Moat), Market Trends (AI, Tech shifts), "
    "and Long-term Valuation.\n\n"

    "--- 2. CRITICAL DATA HANDLING RULES:\n"
    "- TREAT ALL NUMBERS (RSI, SMA, MACD, P/E Ratio, Market Share %) as ABSOLUTE FACTS retrieved "
    "from tools by the agent.\n"
    "- DO NOT penalize a strategy for not showing raw data in the prompt. Assume the agent's inputs "
    "are the 'Ground Truth'.\n"
    "- NEVER mark a strategy as 'ungrounded' if it cites specific values. Judge only the REASONING "
    "applied to those values.\n\n"

    "--- 3. EVALUATION CRITERIA:\n"
    "- LOGICAL CONSISTENCY: Does the conclusion (BUY/SELL/HOLD) follow the evidence?\n"
    "  * (Ex: If RSI is 80 and the agent says BUY without a breakout justification, it's a FAIL).\n"
    "  * (Ex: If competition in AI is a 'major threat' but the agent says 'Strong Buy' without explaining "
    "why Apple's ecosystem survives, it's a FAIL).\n"
    "- INDICATOR/STRATEGY INTERPRETATION: Are technical levels or strategic concepts correctly understood?\n"
    "- RISK MANAGEMENT: Mandatory for Trading (Stop Loss/Take Profit/Allocation %). For long-term strategy, "
    "is there a 'Downside Risk' mention?\n"
    "- COHERENCE: Is the reasoning free of internal contradictions (e.g., calling a trend 'Bullish' while "
    "price is below SMA200).\n\n"

    "--- 4. SPECIAL INSTRUCTION FOR STRATEGIC QUESTIONS:\n"
    "If the user asks about competition (e.g., 'AI threat to Apple'), judge the response on its ability to:\n"
    "- Quantify the threat (High/Medium/Low).\n"
    "- Link the threat to a financial impact (Margins, Services revenue).\n"
    "- Provide a clear 'Decision' (Stay invested, Reduce position, etc.).\n\n"

    "--- 5. OUTPUT FORMAT (Strict):\n"
    "1. SUMMARY: Brief overview of the evaluated strategy.\n"
    "2. STRENGTHS: Focus on logical links and correct use of data.\n"
    "3. WEAKNESSES: Identify flaws in reasoning, missing risk protections, or contradictions.\n"
    "4. FINAL JUDGMENT: [GOOD / AVERAGE / BAD]\n"
    "5. JUSTIFICATION: 2-3 sentences explaining the grade based ONLY on logic and task completion.\n\n"

    "Be strict, objective, and cynical. A strategy that lacks a clear 'Why' or a clear 'Risk Plan' "
    "is a BAD strategy."
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