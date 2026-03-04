import asyncio

from deepeval import assert_test
from deepeval import evaluate

from deepeval.test_case import LLMTestCase, LLMTestCaseParams 
from deepeval.metrics import GEval 
from openai import OpenAI
from deepeval.models.base_model import DeepEvalBaseLLM

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


# on peut baser notre LLM qui teste sur un  autre modèle plus intelligent , voir https://llm-stats.com/
proxy_test_model = ProxyTestLLM()


def test_correctness():

    question = " C'est quoi Apple ?"

    actual = proxy_model.generate(question)

    correctness_metric = GEval(
        name="Coherence",
        criteria=" Check if the actual output replies to the input , without additional informations. If there is additional informations the score must be 0. ", 
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.INPUT
        ],
        threshold=0.5,
        model = proxy_test_model
    )

    test_case_1 = LLMTestCase(
        input= question, 
        actual_output=actual
    )

    evaluate(test_cases=[test_case_1], metrics=[correctness_metric])