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
            actual = proxy_model.generate(golden.input)
            test_case = LLMTestCase(
                input= golden.input, 
                actual_output=actual
            )
            test_cases.append(test_case)

    evaluate(test_cases=test_cases, metrics=[task_completion_metric,strategy_metric])