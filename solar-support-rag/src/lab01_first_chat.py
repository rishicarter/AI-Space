from settings import build_chat_model

model = build_chat_model(temperature=0, need_reasoning=True)

response = model.invoke(
    "In One sentence, explain the purpose of a solar support assistant"
)

print(f"Response type: {type(response).__name__}")
print(f"Content: {response.content}")
print(f"Finish reason: {response.response_metadata.get('finish_reason')}")
print(f"response metadata: {response.response_metadata.get('token_usage')}")
