import asyncio

from deepeval import assert_test
from deepeval import evaluate

from deepeval.test_case import LLMTestCase, LLMTestCaseParams 
from deepeval.metrics import GEval 
from openai import OpenAI
from deepeval.models.base_model import DeepEvalBaseLLM

import agent_stock_mcp as agent

# changer le modèle qu'utilise deepEval 
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


def test_correctness():

    actual = proxy_model.generate("Apple ?")

    correctness_metric = GEval(
        name="Correctness",
        criteria="Check if the answer is coherent and concise",
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT
        ],
        threshold=0.5
    )

    test_case = LLMTestCase(
        input="What is Apple ?",
        actual_output=actual
    )

    evaluate(test_cases=[test_case], metrics=[correctness_metric])