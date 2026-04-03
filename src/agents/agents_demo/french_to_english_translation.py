import json

import requests

# Local proxy URL
PROXY_URL = "http://localhost:4141/v1/chat/completions"
REQUEST_TIMEOUT = 30


def translate_french_to_english(text: str) -> str:
    """
    Translate text from French to English using the Copilot proxy.
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": f"Translate this text into English: {text}"}
        ],
    }
    try:
        response = requests.post(
            PROXY_URL,
            headers=headers,
            data=json.dumps(payload),
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        # Return the content of the first response choice.
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    print("French to English translation. Type 'q' to quit.")
    while True:
        text = input("Enter French text to translate: ")
        if text.lower() == "q":
            break
        translation = translate_french_to_english(text)
        print(f"English translation: {translation}\n")
