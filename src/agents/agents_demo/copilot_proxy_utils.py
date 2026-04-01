"""
copilot_proxy_utils.py — Shared MCP tool helpers

Why flattening is needed:
  MCP tools return content block lists: [{"type": "text", "text": "..."}]
  The Copilot proxy (localhost:4141) can't handle that format — it calls
  json.dumps() on the whole array, producing a heavily escaped string that
  confuses the LLM (hallucinations, dropped data). Flattening extracts the
  plain text before the proxy ever sees it, so the LLM gets clean input.

  Toggle flattening per-agent via FLATTEN_MCP_OUTPUT in each agent file.
"""

from langchain_core.tools import StructuredTool


def _flatten_mcp_result(result):
    if isinstance(result, list):
        return "\n".join(
            b["text"] if isinstance(b, dict) and "text" in b else str(b) for b in result
        )
    return str(result)


def wrap_mcp_tool(tool):
    """Wrap an MCP tool so its output is a plain string instead of a content block list."""
    orig = tool.coroutine or tool.func

    if tool.coroutine:

        async def _flat(**kw):
            return _flatten_mcp_result(await orig(**kw))
    else:

        def _flat(**kw):
            return _flatten_mcp_result(orig(**kw))

    return StructuredTool.from_function(
        func=_flat if not tool.coroutine else None,
        coroutine=_flat if tool.coroutine else None,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
    )
