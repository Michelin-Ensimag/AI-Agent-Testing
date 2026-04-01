import asyncio


from deepeval import evaluate

from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval, TaskCompletionMetric

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


def test_consistency():

    # Prompt
    question = "Should I buy or not AAPL ?"

    # Dataset des tests E2E à lancer
    dataset = EvaluationDataset(
        goldens=[
            Golden(input=question),
            Golden(
                input="Si j'entre sur AAPL à $180,"
                "propose-moi un plan de sortie incluant"
                " un Stop Loss à 5% et deux paliers de Take Profit"
                " basés sur les résistances historiques. "
            ),
            Golden(
                input="Compte tenu d'un portefeuille conservateur,"
                " quel est le risque de concentration si j'achète du AAPL maintenant ?"
            ),
            Golden(
                input="L'avance de la concurrence sur l'IA menace-t-elle la position dominante d'Apple à long terme ?"
            ),
        ]
    )
    #                                        Golden(input="Le RSI d'AAPL approche de 75. Stratégiquement, est-ce un moment d'achat ou faut-il attendre une correction ?")

    dataset = EvaluationDataset(goldens=[Golden(input=question)])

    # Cette métrique dans DeepEval agit comme un inspecteur de travaux finis. Elle ne se laisse pas charmer par le beau langage. Elle vérifie si les "cases" de l'input ont été cochées.
    task_completion_metric = TaskCompletionMetric(threshold=0.5, model=proxy_test_model)

    strategy_metric = GEval(
        name="Startegy",
        criteria=(
            "Evaluate if the financial decision (BUY/SELL/HOLD) is logically derived from the data provided.\n"
            "1. For Technical analysis: Check if RSI/SMA/MACD signals are interpreted correctly.\n"
            "2. For Strategic analysis (e.g., AI competition): Check if the competitive threat is quantified and linked to financial impact.\n"
            "3. Logic: Ensure zero internal contradictions.\n"
            "4. Actionability: Ensure clear stop-loss/take-profit for trades OR clear risk-mitigation for long-term views."
        ),
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.INPUT],
        threshold=0.5,
        model=proxy_test_model,
    )

    # Premier cas de test
    test_cases = []

    for golden in dataset.goldens:
        # Pour tester la consistence du juge, il faut mettre le meme output pour chaque golden
        actual = proxy_model.generate(golden.input)

        for i in range(10):
            test_case = LLMTestCase(input=golden.input, actual_output=actual)
            test_cases.append(test_case)

    evaluate(test_cases=test_cases, metrics=[task_completion_metric, strategy_metric])
