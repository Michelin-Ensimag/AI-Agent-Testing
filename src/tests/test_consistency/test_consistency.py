"""
test_agent_consistency.py - Consistency / determinism tests for the trading agent

Verifies that the agent returns the same decision (BUY/SELL/HOLD) and coherent
price levels when given the same input multiple times.

Run:
    deepeval test run src/tests/test_agent_consistency.py
"""

import asyncio
import re
from collections import Counter
from pathlib import Path

from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.metrics.base_metric import BaseMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI

# Configuration
DATE = "2024-01-03"
N_RUNS = 3
MAX_ITERATIONS = 10
MCP_SERVER_PATH = (
    Path(__file__).parent.parent.parent / "mcp_servers" / "mcp_server_strat_pred.py"
)

CONSISTENCY_SCENARIOS = [
    ("aapl_minimal_en", "Analyze Apple (AAPL) and give me a trading strategy.", "EN"),
    ("tsla_minimal_en", "Generate a trading strategy for Tesla (TSLA).", "EN"),
    (
        "msft_detailed_fr",
        "Analyse Microsoft (MSFT) en détail. Utilise les indicateurs RSI, MACD "
        "et les moyennes mobiles. Dis-moi si je dois acheter, vendre ou conserver "
        "l'action avec un stop loss précis.",
        "FR",
    ),
]


# System prompts
STRATEGY_SYSTEM_EN = (
    f"We are the {DATE}. "
    "You are a quantitative trading assistant. "
    "You have access to tools that retrieve market data and compute indicators. "
    "Use the tools to analyze the stock and generate a trading strategy.\n\n"
    "To analyze a stock:\n"
    "1. Call analyze_stock(ticker) — this fetches data AND computes all indicators.\n"
    "2. Optionally call risk_analysis with the raw OHLCV if needed.\n"
    "3. Use the results to generate a trading strategy.\n\n"
    "When analyzing indicators:\n"
    "- SMA TREND: If SMA20 > SMA50, it is BULLISH. If SMA20 < SMA50, it is BEARISH.\n"
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

STRATEGY_SYSTEM_FR = (
    f"Nous sommes le {DATE}. "
    "Tu es un assistant de trading quantitatif. "
    "Tu as accès à des outils pour récupérer des données de marché et calculer des indicateurs. "
    "Utilise ces outils pour analyser l'action et générer une stratégie de trading.\n\n"
    "Pour analyser une action :\n"
    "1. Appelle analyze_stock(ticker) — cela récupère les données ET calcule tous les indicateurs.\n"
    "2. Appelle optionnellement risk_analysis avec les données OHLCV brutes si nécessaire.\n"
    "3. Utilise les résultats pour générer une stratégie de trading.\n\n"
    "Lors de l'analyse des indicateurs :\n"
    "- TENDANCE SMA : Si SMA20 > SMA50, c'est HAUSSIER. Si SMA20 < SMA50, c'est BAISSIER.\n"
    "- RSI > 70 signifie suracheté\n"
    "- RSI < 30 signifie survendu\n"
    "- MACD croisant à la hausse le signal indique une dynamique d'achat\n"
    "- MACD croisant à la baisse le signal indique une dynamique de vente\n\n"
    "Ta réponse finale doit contenir :\n"
    "1. Analyse de tendance\n"
    "2. Interprétation des indicateurs\n"
    "3. Décision de trading (ACHETER / VENDRE / CONSERVER)\n"
    "4. Stop loss suggéré\n"
    "5. Take profit suggéré\n"
)

EVALUATOR_SYSTEM = (
    "You are an expert quantitative trading evaluator.\n"
    f"Your role is NOT to generate a strategy, but to EVALUATE the logic and quality "
    f"of a given strategy using only information available up to {DATE}\n\n"
    "CRITICAL DATA HANDLING RULES:\n"
    "- TREAT ALL NUMBERS mentioned (RSI, SMA, MACD prices) as ABSOLUTE FACTS retrieved from tools.\n"
    "- DO NOT fail or penalize a strategy because you don't see the raw data in your initial prompt.\n"
    "- NEVER mark a strategy as 'ungrounded' if it cites specific technical values.\n\n"
    "Evaluation criteria:\n"
    "- LOGICAL CONSISTENCY: Does the decision (BUY/SELL/HOLD) actually match the stated indicators?\n"
    "- INDICATOR INTERPRETATION: Are technical levels correctly understood?\n"
    "- RISK MANAGEMENT: Are stop loss and take profit levels realistic?\n"
    "- COHERENCE: Is the reasoning free of internal contradictions?\n\n"
    "If the Input is a simple user question without numeric data, judge the decision based ONLY on the numbers "
    "the agent claims to have found. Assume the agent's numbers are the ground truth for this evaluation.\n\n"
    "Tool Usage:\n"
    "- Use 'get_market_data' ONLY if you suspect a gross mathematical impossibility.\n\n"
    "Output format:\n"
    "1. Summary of the evaluated strategy\n"
    "2. Strengths (Focus on logic)\n"
    "3. Weaknesses (Focus on flaws in reasoning or risk)\n"
    "4. Final judgment: GOOD / AVERAGE / BAD\n"
    "5. Short justification\n\n"
    "Be strict, objective, and critical. A good strategy must be logically bulletproof."
)

SYSTEM_PROMPTS = {"EN": STRATEGY_SYSTEM_EN, "FR": STRATEGY_SYSTEM_FR}


# LLM
def create_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url="http://localhost:4141/v1",
        api_key="dummy-key",
        model="gpt-4.1",
    )


# Agent loop
async def run_agent(
    question: str,
    system_prompt: str,
    allowed_tool_names: list | None = None,
    max_iterations: int = MAX_ITERATIONS,
) -> str:
    llm = create_llm()
    client = MultiServerMCPClient(
        {
            "stock": {
                "command": "python",
                "args": [str(MCP_SERVER_PATH)],
                "transport": "stdio",
            }
        }
    )
    async with client.session("stock") as session:
        tools = await load_mcp_tools(session)
        if allowed_tool_names:
            tools = [t for t in tools if t.name in allowed_tool_names]
        tools_by_name = {t.name: t for t in tools}
        llm_with_tools = llm.bind_tools(tools)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
        ]

        for _ in range(max_iterations):
            response = await llm_with_tools.ainvoke(messages)
            messages.append(response)
            if not response.tool_calls:
                return response.content
            for tc in response.tool_calls:
                tool = tools_by_name.get(tc["name"])
                if tool is None:
                    continue
                result = await tool.ainvoke(tc["args"])
                if isinstance(result, list):
                    result = "\n".join(
                        b.get("text", str(b)) if isinstance(b, dict) else str(b)
                        for b in result
                    )
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    return "Max iterations reached."


# Proxy models
class ProxyLLM(DeepEvalBaseLLM):
    def __init__(self, system_prompt: str = STRATEGY_SYSTEM_EN):
        self.system_prompt = system_prompt

    def load_model(self):
        return None

    def generate(self, prompt: str) -> str:
        return asyncio.run(self.a_generate(prompt))

    async def a_generate(self, prompt: str) -> str:
        return await run_agent(prompt, self.system_prompt)

    def get_model_name(self) -> str:
        return "proxy-gpt-4.1"


class ProxyTestLLM(DeepEvalBaseLLM):
    def load_model(self):
        return None

    def generate(self, prompt: str) -> str:
        return asyncio.run(self.a_generate(prompt))

    async def a_generate(self, prompt: str) -> str:
        return await run_agent(
            prompt, EVALUATOR_SYSTEM, allowed_tool_names=["get_market_data"]
        )

    def get_model_name(self) -> str:
        return "proxy-test-gpt-4.1"


proxy_test_model = ProxyTestLLM()


# Helper
_DECISION_PATTERN = re.compile(
    r"\b(BUY|SELL|HOLD|ACHETER|VENDRE|CONSERVER)\b", re.IGNORECASE
)


def extract_decision(text: str) -> str | None:
    """Return the last BUY/SELL/HOLD decision found in the text, normalised."""
    _FR_TO_EN = {"ACHETER": "BUY", "VENDRE": "SELL", "CONSERVER": "HOLD"}
    matches = _DECISION_PATTERN.findall(text)
    if not matches:
        return None
    last = matches[-1].upper()
    return _FR_TO_EN.get(last, last)


# Custom metric : decision agreement rate
class DecisionConsistencyMetric(BaseMetric):
    """
    Measure agreement rate for BUY/SELL/HOLD decision across N runs.

    Expected test_case fields:
        - input          : user question (str)
        - actual_output  : majority decision (str, e.g. "BUY")
        - context        : list of N raw outputs (list[str])

    Score = (count of majority decision) / N_RUNS
    Passes if score >= threshold.
    """

    def __init__(self, threshold: float = 0.67, n_runs: int = N_RUNS):
        self.threshold = threshold
        self.n_runs = n_runs
        self.name = "Decision Consistency"

    def measure(self, test_case: LLMTestCase) -> float:
        raw_outputs: list[str] = test_case.context or []
        decisions = [extract_decision(o) for o in raw_outputs]
        decisions = [d for d in decisions if d is not None]

        if not decisions:
            self.score = 0.0
            self.reason = "No BUY/SELL/HOLD decision found in any run."
            self.success = False
            return self.score

        counter = Counter(decisions)
        majority_decision, majority_count = counter.most_common(1)[0]
        self.score = majority_count / self.n_runs
        self.reason = (
            f"Decision distribution over {self.n_runs} runs: {dict(counter)}. "
            f"Majority = '{majority_decision}' ({majority_count}/{self.n_runs} = {self.score:.0%})."
        )
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success


# GEval metric : cross-run coherence
# Evaluate whether N outputs are logically coherent with each other.
cross_run_coherence_metric = GEval(
    name="Cross-run coherence",
    criteria=(
        "You are given multiple trading strategy outputs generated by the same agent "
        "for the exact same question. Assess whether the outputs are logically consistent "
        "with each other.\n\n"
        "Focus on:\n"
        "- Do all runs reach the same trading decision (BUY / SELL / HOLD)?\n"
        "- Are the indicator interpretations (RSI, SMA, MACD) aligned across runs?\n"
        "- Are stop loss and take profit levels in a similar price range across runs?\n"
        "- Are contradictions between runs penalized heavily?\n\n"
        "Score 1.0 if all runs are fully consistent.\n"
        "Score 0.5 if minor wording differs but the decision and levels agree.\n"
        "Score 0.0 if the decision changes across runs (e.g., BUY vs SELL).\n\n"
        "The 'actual output' contains all N run outputs concatenated. "
        "The 'input' is the original user question."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.67,
    model=proxy_test_model,
)


# Test case builder
def build_consistency_test_cases(
    scenario_name: str,
    question: str,
    system_prompt: str,
    n_runs: int = N_RUNS,
) -> tuple[LLMTestCase, LLMTestCase]:
    """
    Run the agent N times for the same question.
    Returns two LLMTestCases:
      - one for DecisionConsistencyMetric (context = raw outputs)
      - one for cross_run_coherence_metric (actual_output = concatenated runs)
    """
    proxy = ProxyLLM(system_prompt)
    outputs = [proxy.generate(question) for _ in range(n_runs)]

    separator = "\n\n" + "─" * 60 + "\n\n"
    concatenated = separator.join(
        f"[Run {i + 1}]\n{out}" for i, out in enumerate(outputs)
    )

    # For DecisionConsistencyMetric: context carries raw outputs.
    decision_tc = LLMTestCase(
        name=f"{scenario_name}_decision_consistency",
        input=question,
        actual_output=Counter(
            d for d in (extract_decision(o) for o in outputs) if d
        ).most_common(1)[0][0]
        if any(extract_decision(o) for o in outputs)
        else "UNKNOWN",
        context=outputs,
    )

    # For GEval cross-run coherence: actual_output is the full concatenated text
    coherence_tc = LLMTestCase(
        name=f"{scenario_name}_cross_run_coherence",
        input=question,
        actual_output=concatenated,
    )

    return decision_tc, coherence_tc


# Tests
def test_consistency():
    """
    Run each scenario N_RUNS times and evaluate:
        1. DecisionConsistencyMetric - agreement rate on BUY/SELL/HOLD
        2. Cross-run coherence (GEval) - global logical coherence between runs
    """
    decision_metric = DecisionConsistencyMetric(threshold=0.67, n_runs=N_RUNS)

    decision_cases: list[LLMTestCase] = []
    coherence_cases: list[LLMTestCase] = []

    for scenario_name, question, prompt_key in CONSISTENCY_SCENARIOS:
        system_prompt = SYSTEM_PROMPTS[prompt_key]
        decision_tc, coherence_tc = build_consistency_test_cases(
            scenario_name, question, system_prompt, n_runs=N_RUNS
        )
        decision_cases.append(decision_tc)
        coherence_cases.append(coherence_tc)

    # Evaluation 1: decision agreement rate (pure Python metric, fast).
    evaluate(
        test_cases=decision_cases,
        metrics=[decision_metric],
    )

    # Evaluation 2: cross-run logical coherence (LLM judge).
    evaluate(
        test_cases=coherence_cases,
        metrics=[cross_run_coherence_metric],
    )
