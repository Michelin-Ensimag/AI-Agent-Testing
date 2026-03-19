"""
tests/unit/test_agent_logic.py

Tests unitaires pour la logique des agents (agent_strat_pred, agent_strat_test).
Aucun appel LLM réel — tout est mocké avec unittest.mock.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, ToolMessage



def make_fake_tool(name: str, return_value: str = "fake_result"):
    """Crée un faux outil MCP."""
    tool = MagicMock()
    tool.name = name
    tool.description = f"Fake tool: {name}"
    tool.ainvoke = AsyncMock(return_value=return_value)
    return tool


def make_ai_message(content: str = "", tool_calls: list = None):
    """Crée un faux AIMessage avec ou sans tool_calls."""
    msg = MagicMock(spec=AIMessage)
    msg.content = content
    msg.tool_calls = tool_calls or []
    return msg




class TestAgentStratPred:

    @pytest.mark.asyncio
    async def test_returns_content_when_no_tool_calls(self):
        """L'agent doit retourner le contenu direct si le LLM ne fait pas de tool call."""
        from agents.agent_strat_pred import run_agent_logic

        fake_llm = MagicMock()
        fake_llm.bind_tools = MagicMock(return_value=fake_llm)
        fake_llm.ainvoke = AsyncMock(
            return_value=make_ai_message(content="BUY AAPL — RSI oversold at 28.")
        )
        tools = [make_fake_tool("analyze_stock")]

        result = await run_agent_logic("Should I buy AAPL?", fake_llm, tools)

        assert result == "BUY AAPL — RSI oversold at 28."

    @pytest.mark.asyncio
    async def test_calls_tool_then_returns_final_answer(self):
        """L'agent doit appeler un outil, puis retourner la réponse finale."""
        from agents.agent_strat_pred import run_agent_logic

        # Premier appel : LLM demande un tool call
        tool_call = {"name": "analyze_stock", "args": {"ticker": "AAPL"}, "id": "call_001"}
        first_response = make_ai_message(tool_calls=[tool_call])

        # Deuxième appel : LLM donne la réponse finale
        final_response = make_ai_message(content="HOLD — MACD bearish crossover detected.")

        fake_llm = MagicMock()
        fake_llm.bind_tools = MagicMock(return_value=fake_llm)
        fake_llm.ainvoke = AsyncMock(side_effect=[first_response, final_response])

        tools = [make_fake_tool("analyze_stock", return_value="{'Close': [150, 151]}")]

        result = await run_agent_logic("Analyze AAPL", fake_llm, tools)

        assert result == "HOLD — MACD bearish crossover detected."
        # Vérifie que le tool a bien été appelé
        tools[0].ainvoke.assert_called_once_with({"ticker": "AAPL"})

    @pytest.mark.asyncio
    async def test_max_iterations_returns_fallback_message(self):
        """Si la limite d'itérations est atteinte, retourner le message d'erreur."""
        from agents.agent_strat_pred import run_agent_logic

        # LLM demande toujours un tool call → boucle infinie
        tool_call = {"name": "analyze_stock", "args": {"ticker": "TSLA"}, "id": "call_loop"}
        looping_response = make_ai_message(tool_calls=[tool_call])

        fake_llm = MagicMock()
        fake_llm.bind_tools = MagicMock(return_value=fake_llm)
        fake_llm.ainvoke = AsyncMock(return_value=looping_response)

        tools = [make_fake_tool("analyze_stock")]

        result = await run_agent_logic("Loop forever", fake_llm, tools, max_iterations=3)

        assert result == "Max iterations reached."

    @pytest.mark.asyncio
    async def test_tool_result_list_is_joined_as_string(self):
        """Si le tool retourne une liste de blocs, ils doivent être concaténés en string."""
        from agents.agent_strat_pred import run_agent_logic

        tool_call = {"name": "analyze_stock", "args": {"ticker": "MSFT"}, "id": "call_002"}
        first_response = make_ai_message(tool_calls=[tool_call])
        final_response = make_ai_message(content="SELL signal confirmed.")

        fake_llm = MagicMock()
        fake_llm.bind_tools = MagicMock(return_value=fake_llm)
        fake_llm.ainvoke = AsyncMock(side_effect=[first_response, final_response])

        # Tool retourne une liste de dicts (format MCP bloc)
        list_result = [{"text": "SMA20=150"}, {"text": "RSI=72"}]
        tools = [make_fake_tool("analyze_stock", return_value=list_result)]

        result = await run_agent_logic("Analyze MSFT", fake_llm, tools)

        assert result == "SELL signal confirmed."




class TestAgentStratTest:

    @pytest.mark.asyncio
    async def test_evaluator_only_uses_allowed_tools(self):
        """L'agent évaluateur ne doit utiliser que get_market_data, pas analyze_stock."""
        from agents.agent_strat_test import run_agent_logic

        final_response = make_ai_message(content="GOOD — strategy is logically consistent.")

        fake_llm = MagicMock()
        fake_llm.bind_tools = MagicMock(return_value=fake_llm)
        fake_llm.ainvoke = AsyncMock(return_value=final_response)

        # On donne les deux tools, mais seul get_market_data est autorisé
        tools = [
            make_fake_tool("analyze_stock"),
            make_fake_tool("get_market_data"),
        ]

        await run_agent_logic("Evaluate this strategy: BUY AAPL RSI=28", fake_llm, tools)

        # bind_tools ne doit avoir reçu que get_market_data
        bound_tools = fake_llm.bind_tools.call_args[0][0]
        tool_names = [t.name for t in bound_tools]
        assert "get_market_data" in tool_names
        assert "analyze_stock" not in tool_names

    @pytest.mark.asyncio
    async def test_evaluator_returns_judgment(self):
        """L'évaluateur doit retourner une réponse contenant un jugement."""
        from agents.agent_strat_test import run_agent_logic

        judgment = "GOOD — RSI oversold at 25 correctly leads to BUY decision."
        final_response = make_ai_message(content=judgment)

        fake_llm = MagicMock()
        fake_llm.bind_tools = MagicMock(return_value=fake_llm)
        fake_llm.ainvoke = AsyncMock(return_value=final_response)

        tools = [make_fake_tool("get_market_data")]

        result = await run_agent_logic("Is BUY with RSI=25 correct?", fake_llm, tools)

        assert result == judgment




class TestCreateLLM:

    def test_create_llm_returns_chat_openai(self):
        """create_llm doit retourner une instance ChatOpenAI."""
        from langchain_openai import ChatOpenAI
        from agents.agent_strat_pred import create_llm

        llm = create_llm()
        assert isinstance(llm, ChatOpenAI)

    def test_create_llm_uses_correct_model(self):
        """Le modèle configuré doit être gpt-5.2."""
        from agents.agent_strat_pred import create_llm

        llm = create_llm()
        assert llm.model_name == "gpt-5.2"