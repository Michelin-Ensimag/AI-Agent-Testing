"""
Setup instructions for using this script:

1. Install Bun: https://bun.com/docs/installation
2. Login to your GitHub account:
   bunx @jeffreycao/copilot-api@latest auth
3. Launch the server:
   bunx @jeffreycao/copilot-api@latest start
"""

from openai import OpenAI

# We point the base_url to your local Bun server
# The api_key can be anything (like "dummy") because the proxy handles the real auth
client = OpenAI(
    base_url="http://localhost:4141/v1",
    api_key="dummy-key" 
)

response = client.chat.completions.create(
    model="gpt-4.1", # Choose the model
    messages=[{"role": "user", "content": "No idea."}]
)

print(response.choices[0].message.content)