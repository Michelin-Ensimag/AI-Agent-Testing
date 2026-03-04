import requests
import json

# URL de ton proxy local
PROXY_URL = "http://localhost:4141/v1/chat/completions"

def traduire_fr_en(texte)  :
    """
    Cette fonction permet la traduction d'un texte du français vers l'anglais en utilisant le proxy Copilot.
    """
    headers = {"Content-Type": "application/json"}
    payload = {"model": "gpt-4o-mini","messages": [{"role": "user", "content": f"Traduis ce texte en anglais : {texte}"}]}
    try:
        response = requests.post(PROXY_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        data = response.json()
        # On récupère le contenu de la première réponse
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Erreur : {e}"

if __name__ == "__main__":
    print("Traduction français  en anglais. Tapez 'q' pour quitter.")
    while True:
        texte = input("Rentrez le texte en français que vous voulez traduire : ")
        if texte.lower() == "q":
            break
        traduction = traduire_fr_en(texte)
        print(f"Voici la traduction en anglais :  {traduction}\n")
