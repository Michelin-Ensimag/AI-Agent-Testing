import asyncio

import allure
from deepeval import evaluate
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import GEval, TaskCompletionMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

import agents.agent_strat_pred as agent
import agents.agent_strat_test as agent_test

MARKDOWN_ATTACHMENT_TYPE = getattr(allure.attachment_type, "MARKDOWN", "text/markdown")


# Change the model used by DeepEval.
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
        # Models like gemini-2.5-pro or gpt-5.2
        # tend to fail these tests more often
        # compared to gpt-4o, for example.
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


# The judge LLM can also be based on another stronger model: https://llm-stats.com/
proxy_test_model = ProxyTestLLM()


def _fmt_score(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "N/A"


def _fmt_reason(value: str | None) -> str:
    if not value:
        return "-"
    return value.replace("|", "\\|").replace("\n", "<br>")


def _fmt_status(metric) -> str:
    try:
        return "PASS" if metric.is_successful() else "FAIL"
    except Exception:
        return "N/A"


def test_consistency():

    # Prompt
    question = "Should I buy or not AAPL ?"

    # E2E test dataset
    dataset = EvaluationDataset(
        goldens=[
            Golden(input=question),
            Golden(
                input="If I enter AAPL at $180,"
                " propose an exit plan including"
                " a 5% stop-loss and two take-profit levels"
                " based on historical resistance zones."
            ),
            Golden(
                input="Given a conservative portfolio,"
                " what concentration risk do I face if I buy AAPL now?"
            ),
            Golden(
                input="Does AI competition threaten Apple's long-term dominant position?"
            ),
        ]
    )
    #                                        Golden(input="AAPL RSI is approaching 75. Strategically, buy now or wait for a correction?")

    dataset = EvaluationDataset(goldens=[Golden(input=question)])

    # This DeepEval metric checks whether the requested task was actually completed.
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

    # First test case
    test_cases = []
    agent_output = ""

    for golden in dataset.goldens:
        # To evaluate judge consistency, keep the same output for each golden.
        actual = proxy_model.generate(golden.input)
        agent_output = actual

        for i in range(10):
            test_case = LLMTestCase(input=golden.input, actual_output=actual)
            test_cases.append(test_case)

    try:
        evaluate(
            test_cases=test_cases, metrics=[task_completion_metric, strategy_metric]
        )
    finally:
        markdown_table = (
            "| Metric | Score | Passed | Reason |\n"
            "|--------|-------|--------|--------|\n"
            f"| {task_completion_metric.__class__.__name__} | {_fmt_score(getattr(task_completion_metric, 'score', None))} | "
            f"{_fmt_status(task_completion_metric)} | "
            f"{_fmt_reason(getattr(task_completion_metric, 'reason', None))} |\n"
            f"| {strategy_metric.name} | {_fmt_score(getattr(strategy_metric, 'score', None))} | "
            f"{_fmt_status(strategy_metric)} | "
            f"{_fmt_reason(getattr(strategy_metric, 'reason', None))} |"
        )

        allure.attach(
            question,
            name="1. User Input",
            attachment_type=allure.attachment_type.TEXT,
        )
        allure.attach(
            agent_output or "No output captured.",
            name="2. Agent Output",
            attachment_type=allure.attachment_type.TEXT,
        )
        allure.attach(
            markdown_table,
            name="3. DeepEval Metrics Summary",
            attachment_type=MARKDOWN_ATTACHMENT_TYPE,
        )
