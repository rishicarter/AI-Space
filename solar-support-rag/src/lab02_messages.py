from langchain_core.messages import HumanMessage, SystemMessage
from settings import build_chat_model

# model = build_chat_model(temperature=0)
# messages = [
#     SystemMessage(content="You are the Solar Support Assistant. " \
#     "Use plain language and ask one clarifying question when important details are missing."),
#     HumanMessage(content="My solar controller shows E14. What should I do?")
# ]

# response = model.invoke(messages)
# print(f"Content: {response.content}")

# messages.append(response)
# messages.append(
#     HumanMessage(content="It started after a storm. Does that matter?")
# )

# follow_up = model.invoke(messages)
# print(follow_up.content)

# print(f"response metadata: {response.response_metadata.get('token_usage')}")
# print(f"follow_up metadata: {follow_up.response_metadata.get('token_usage')}")


prompt = "Suggest one short title for a solar support knowledge base."
precise_model = build_chat_model(temperature=0)
varied_model = build_chat_model(temperature=0.7)
print("Temperature 0:", precise_model.invoke(prompt).content)
print("Temperature 0.7:", varied_model.invoke(prompt).content)