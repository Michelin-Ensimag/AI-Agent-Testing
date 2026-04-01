from typing import TypedDict
from langgraph.graph import StateGraph
import requests

import os
from dotenv import load_dotenv

# Définition de l'état de l'agent
class AgentState(TypedDict):
    sujet: str
    infos: str
    summary: str

load_dotenv()

# Clé d'API pour NewsData.io
NEWS_API_KEY = os.getenv("NEWSDATA_API_KEY")
if not NEWS_API_KEY:
    raise ValueError("Missing NEWSDATA_API_KEY! Did you forget to set up your .env file?")

# URL du proxy Copilot local
PROXY_URL = "http://localhost:4141/v1/chat/completions"

# Fonction pour appeler le proxy Copilot
def generer_texte_proxy(prompt: str) -> str:
    """
    Envoie le prompt au proxy Copilot et retourne la réponse texte.
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        r = requests.post(PROXY_URL, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Erreur : {e}"

# Fonction recherche d'informations sur un sujet donné
def rechercher_infos(state):
    url = "https://newsdata.io/api/1/news"
    params = {
        "apikey": NEWS_API_KEY,
        "q": state["sujet"],
        "language": "fr",
        "country": "fr"
    }
    r = requests.get(url, params=params)
    if r.status_code != 200:
        print("Erreur lors de la requête à l'API NewsData : ", r.status_code)
        exit(1)
    articlesTrouvés = r.json().get("results", [])
    titres = "\n".join(" - " + article['title'] for article in articlesTrouvés)
    return {"sujet": state["sujet"], "infos": titres, "summary": ""}

# Fonction de résumé des informations trouvées via le proxy Copilot
def resumer_infos(state):
    prompt = f"Résume moi les articles suivants : {state['infos']} en préservant les informations clés et en restant concis."
    summary = generer_texte_proxy(prompt)
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

# Entrée utilisateur
sujet = input("Quel sujet voulez-vous que je traite ? ")
resultat = graph.invoke({"sujet": sujet})

# Affichage du résumé final
print("Voici le résumé des articles trouvés sur le sujet :", resultat["summary"])
