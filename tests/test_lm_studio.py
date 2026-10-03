
from openai import OpenAI

# Connect to the LM Studio local server
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",  # Placeholder; LM Studio may not require a real key
)

# Check which models are available
models = client.models.list()

print("Available models:")
for model in models.data:
    print(f"- {model.id}")

# Use the first available model
model_id = "qwen3.5-9b-mtp"

# Send a simple prompt
response = client.chat.completions.create(
    model=model_id,
    messages=[
        {
            "role": "user",
            "content": "Say hello in English in one short sentence.",
        }
    ],
    temperature=0,
)

print("\nModel response:")
print(response.choices[0].message.content)