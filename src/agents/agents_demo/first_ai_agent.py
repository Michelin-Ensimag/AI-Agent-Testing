import os
from typing import TypedDict

import requests
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph


class AgentState(TypedDict):
    topic: str
    information: str
    summary: str


load_dotenv()

# API key for NewsData.io used to fetch near real-time news articles.
NEWS_API_KEY = os.getenv("NEWSDATA_API_KEY")
if not NEWS_API_KEY:
    raise ValueError(
        "Missing NEWSDATA_API_KEY! Did you forget to set up your .env file?"
    )

# LLM initialization
LLM = OllamaLLM(model="gemma3:1b")
REQUEST_TIMEOUT = 30


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

    # Handle NewsData API request errors.
    if r.status_code != 200:
        print("Error while requesting the NewsData API:", r.status_code)
        exit(1)

    found_articles = r.json().get("results", [])

    titles = "\n".join(" - " + article["title"] for article in found_articles)

    return {"topic": state["topic"], "information": titles, "summary": ""}


# Summarize the retrieved information
def summarize_information(state: AgentState):

    prompt = (
        "Summarize the following articles while preserving key information "
        f"and staying concise:\n{state['information']}"
    )
    summary = LLM.invoke(prompt)

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


topic = input("Which topic should I analyze? ")
result = graph.invoke({"topic": topic})

print("Summary of the articles found:", result["summary"])
