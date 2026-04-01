from typing import TypedDict

from langgraph.graph import StateGraph
from langchain_ollama import OllamaLLM

import requests

import os
from dotenv import load_dotenv


class AgentState(TypedDict):
    sujet: str
    infos: str
    summary: str


load_dotenv()

# Clé d'API pour NewsData.io qui permet de rechercher des articles d'actualité sur un sujet donné en quasi-temps réel. Vous pouvez vous inscrire sur le site de NewsData.io pour obtenir votre propre clé d'API gratuite.
NEWS_API_KEY = os.getenv("NEWSDATA_API_KEY")
if not NEWS_API_KEY:
    raise ValueError(
        "Missing NEWSDATA_API_KEY! Did you forget to set up your .env file?"
    )

# Initialisation du LLM
LLM = OllamaLLM(model="gemma3:1b")


# Fonction recherche d'informations sur un sujet donné
def rechercher_infos(state):

    url = "https://newsdata.io/api/1/news"

    params = {
        "apikey": NEWS_API_KEY,
        "q": state["sujet"],
        "language": "fr",
        "country": "fr",
    }

    r = requests.get(url, params=params)

    # Gestion des erreurs de la requête à l'API NewsData
    if r.status_code != 200:
        print("Erreur lors de la requête à l'API NewsData : ", r.status_code)
        exit(1)

    articlesTrouvés = r.json().get("results", [])

    titres = "\n".join(" - " + article["title"] for article in articlesTrouvés)

    return {"sujet": state["sujet"], "infos": titres, "summary": ""}


# Fonction de résumé des informations trouvées


def resumer_infos(state):

    prompt = f"Résume moi les articles suivants : {state['infos']} en préservant les informations clés et en restant concis."
    summary = LLM.invoke(prompt)

    return {"sujet": state["sujet"], "infos": state["infos"], "summary": summary}


# Création du workflow de l'agent
workflow = StateGraph(state_schema=AgentState)

# Ajout des noeuds au workflow
workflow.add_node("rechercher", rechercher_infos)
workflow.add_node("résumer", resumer_infos)

# Définition des transitions entre les noeuds
workflow.set_entry_point("rechercher")
workflow.add_edge("rechercher", "résumer")
workflow.set_finish_point("résumer")

graph = workflow.compile()


sujet = input("Quel sujet voulez vous que je traite ?")
resultat = graph.invoke({"sujet": sujet})

print("Voici le résumé des articles trouvés sur le sujet ", resultat["summary"])
