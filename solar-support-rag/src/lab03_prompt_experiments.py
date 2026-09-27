from settings import build_chat_model

model = build_chat_model(temperature=0, need_reasoning=False)

customer_msg = (
    "The controller shows E14 after last night's storm." \
    "The red light returned after I restarted it."
)

weak_prompt = f"Help with this message: {customer_msg}"

improved_prompt = f'''
ROLE: you are a first-line solar support assistant.
Task: Summarize the issue and ask for missing information.
Context: No company knowledge base is connected yet.
Contraints:
- Do not invent the meaning of an error code.
- Do not give electrical repair instructions.
- Use at most 80 words.
Output:
Summary: <one sentence>
Next question: <one question>

<customer_message>
{customer_msg}
</customer_message>
'''

weak_res = model.invoke(weak_prompt).content
imporoved_res = model.invoke(improved_prompt).content

print("WEAK PROMPT\n", weak_res)
print("\nIMPROVED PROMPT\n", imporoved_res)