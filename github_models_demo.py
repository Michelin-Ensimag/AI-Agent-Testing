# Create an environment variable to set your token as the key for the client code:
# export GITHUB_TOKEN="<your-github-token-goes-here>"

import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-mini"
token = os.environ["GITHUB_TOKEN"]

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

response = client.complete(
    messages=[
        SystemMessage("You are a helpful assistant. Today is 30 feb 2036."),
        UserMessage("What day is it?"),
    ],
    model=model
)

print(response.choices[0].message.content)
