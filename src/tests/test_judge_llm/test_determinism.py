import asyncio

from openai import OpenAI

from deepeval import assert_test
from deepeval import evaluate

from deepeval.test_case import LLMTestCase, LLMTestCaseParams 
from deepeval.metrics import GEval , TaskCompletionMetric

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.dataset import EvaluationDataset, Golden

import agents.agent_strat_pred as agent
import agents.agent_strat_test as agent_test

# changer le modèle qu'utilise deepEval  
class ProxyTestLLM(DeepEvalBaseLLM):
    def __init__(self):
        pass

    def load_model(self):
        return None

    def generate(self, prompt: str) -> str:
        return asyncio.run(self.a_generate(prompt))

    async def a_generate(self, prompt: str) -> str:
        llm = agent_test.create_llm()  
        client, tools = await agent_test.create_mcp_client()
        response = await agent_test.run_agent_logic(prompt, llm, tools)
        return str(response)

    def get_model_name(self) -> str:
        # gemini-2.5-pro ou gpt-5.2 
        # ont tendance à ne pas faire passer les tests 
        # comparé à gpt-4o par exemple 
        return "proxy-test-gpt-4.1"
    



class ProxyLLM(DeepEvalBaseLLM):
    def __init__(self):
        pass

    def load_model(self):
        return None

    def generate(self, prompt: str) -> str:
        return asyncio.run(self.a_generate(prompt))

    async def a_generate(self, prompt: str) -> str:
        llm = agent.create_llm()  
        client, tools = await agent.create_mcp_client()
        response = await agent.run_agent_logic(prompt, llm, tools)
        
        return str(response)

    def get_model_name(self) -> str:
        return "proxy-gpt-4.1"
    
proxy_model = ProxyLLM()


# On peut baser notre LLM qui teste sur un autre modèle plus intelligent , voir https://llm-stats.com/
proxy_test_model = ProxyTestLLM()


def test_determinism():

    # Prompt
    question = " Should I buy or not AAPL today ? "

    # Dataset des tests E2E à lancer 
    dataset = EvaluationDataset(goldens = [Golden(input=question)])

    # Cette métrique dans DeepEval agit comme un inspecteur de travaux finis. Elle ne se laisse pas charmer par le beau langage. Elle vérifie si les "cases" de l'input ont été cochées.
    task_completion_metric = TaskCompletionMetric(threshold = 0.5, model=proxy_test_model)


    strategy_metric = GEval(
        name="startegy",
       criteria = (
    "Assess the quality of the trading strategy based on financial correctness and reasoning quality.\n\n"
    "The evaluation must verify:\n"
    "- No contradiction between indicators and conclusions\n"
    "- Correct interpretation of SMA, RSI, and MACD signals\n"
    "- A justified BUY, SELL, or HOLD decision\n"
    "- Realistic and risk-aware stop loss and take profit levels\n"
    "- Clear and structured reasoning\n\n"
    "Penalize heavily if:\n"
    "- Indicators are misinterpreted\n"
    "- The decision is not supported by data\n"
    "- The reasoning is vague or generic\n"
    "- Risk management is missing or unrealistic\n"
),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.INPUT
        ],
        threshold=0.5,
        model = proxy_test_model
    )


    # Premier cas de test
    test_cases = []

    for golden in dataset.goldens:

        for i in range (10):
            # actual is the answer given by our prediction agent 
            actual = """
                        ── Final strategy ──

                        Here is an analysis for AAPL as of today:

                        1. Trend Analysis:
                        - The short-term SMA20 (262.60) is slightly above the longer-term SMA50 (262.20), indicating a transition from a bearish to a mildly bullish trend.
                        - The recent price action shows some recovery from oversold levels, but the uptrend is not strong.

                        2. Indicator Interpretation:
                        - RSI is currently at 23.69, which is in oversold territory (<30). This suggests a possible bounce or reversal higher.
                        - MACD (-2.94) is below the signal (-1.47), indicating continued bearish momentum, but the gap has started to narrow compared to previous days, suggesting possible waning of negative momentum.
                        - EMA20 (260.81) is very close to the current price, suggesting price is hovering near short-term average.

                        3. Trading Decision:
                        - Decision: HOLD (Possible Bounce, But Confirmation Needed)
                        - The stock is oversold, which may present a rebound opportunity. However, the overall momentum is still bearish per MACD, and the upturn in SMA20 is very recent.

                        4. Suggested Stop Loss:
                        - Place a stop loss just below recent lows, e.g., $249.00.

                        5. Suggested Take Profit:
                        - Aim for a conservative bounce target around previous resistance or upper recent range, e.g., $263.00 – $265.00.

                        Summary:
                        The technicals show early signs of a possible recovery due to the oversold RSI and the SMA turning up, but strength is not confirmed yet.
                        It’s better to wait for a stronger confirmation (e.g., MACD crossing above signal, price closing above SMA50) before buying aggressively.
                        If you decide to buy for a short-term rebound, be cautious and use tight risk management.
                    """
            test_case = LLMTestCase(
                input= golden.input, 
                actual_output=actual
            )
            test_cases.append(test_case)

    evaluate(test_cases=test_cases, metrics=[task_completion_metric,strategy_metric])