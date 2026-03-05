import asyncio

from openai import OpenAI

from deepeval import assert_test
from deepeval import evaluate

from deepeval.test_case import LLMTestCase, LLMTestCaseParams 
from deepeval.metrics import GEval , TaskCompletionMetric

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.dataset import EvaluationDataset, Golden

import agents.agent_stock_mcp as agent

# changer le modèle qu'utilise deepEval  
class ProxyTestLLM(DeepEvalBaseLLM):
    def __init__(self):
        self.client = OpenAI(
            base_url="http://localhost:4141/v1",
            api_key="dummy-key" 
        )

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4.1", # Choose the test model
            messages=[{"role": "user", "content": prompt}]
        )

        return str(response.choices[0].message.content)

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
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


def test_coherence():

    # Prompt
    question = " C'est quoi le prix d'Apple ? "

    # Dataset des tests E2E lancés
    dataset = EvaluationDataset(goldens = [Golden(input=question)])

    # Cette métrique dans DeepEval agit comme un inspecteur de travaux finis. Elle ne se laisse pas charmer par le beau langage. Elle vérifie si les "cases" de l'input ont été cochées.
    task_completion_metric = TaskCompletionMetric(threshold = 0.5, model=proxy_test_model)


    # Jugement qualitatif de l'output  
    coherence_metric = GEval(
        name="Cohérence",
        criteria=" Vérifie si la réponse est polie et cohérente.", 
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

        actual = proxy_model.generate(golden.input)
        test_case = LLMTestCase(
            input= golden.input, 
            actual_output=actual
        )
        test_cases.append(test_case)

    evaluate(test_cases=test_cases, metrics=[task_completion_metric,coherence_metric])