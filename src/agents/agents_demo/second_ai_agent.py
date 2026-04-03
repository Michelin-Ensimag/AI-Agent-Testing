import os
from typing import TypedDict

import requests
from dotenv import load_dotenv
from langgraph.graph import StateGraph


# Agent state definition
class AgentState(TypedDict):
    topic: str
    information: str
    summary: str


load_dotenv()

# NewsData.io API key
NEWS_API_KEY = os.getenv("NEWSDATA_API_KEY")
if not NEWS_API_KEY:
    raise ValueError(
        "Missing NEWSDATA_API_KEY! Did you forget to set up your .env file?"
    )

# Local Copilot proxy URL
PROXY_URL = "http://localhost:4141/v1/chat/completions"
REQUEST_TIMEOUT = 30


# Call the Copilot proxy and return the generated text.
def generate_proxy_text(prompt: str) -> str:
    """
    Send a prompt to the Copilot proxy and return the text response.
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        r = requests.post(
            PROXY_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT
        )
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {e}"


# Search for news information on a topic
def search_information(state: AgentState):
    url = "https://newsdata.io/api/1/news"
    params = {
        "apikey": NEWS_API_KEY,
        "q": state["topic"],
        "language": "fr",
        "country": "fr",
    }
    r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    if r.status_code != 200:
        print("Error while requesting the NewsData API:", r.status_code)
        exit(1)
    found_articles = r.json().get("results", [])
    titles = "\n".join(" - " + article["title"] for article in found_articles)
    return {"topic": state["topic"], "information": titles, "summary": ""}


# Summarize retrieved information with the Copilot proxy
def summarize_information(state: AgentState):
    prompt = (
        "Summarize the following articles while preserving key information "
        f"and staying concise:\n{state['information']}"
    )
    summary = generate_proxy_text(prompt)
    return {
        "topic": state["topic"],
        "information": state["information"],
        "summary": summary,
    }


# Build the agent workflow
workflow = StateGraph(state_schema=AgentState)

# Add nodes to the workflow
workflow.add_node("search", search_information)
workflow.add_node("summarize", summarize_information)

# Define workflow transitions
workflow.set_entry_point("search")
workflow.add_edge("search", "summarize")
workflow.set_finish_point("summarize")

graph = workflow.compile()

# User input
topic = input("Which topic should I analyze? ")
result = graph.invoke({"topic": topic})

# Display final summary
print("Summary of the articles found:", result["summary"])
