import time

import requests
import streamlit as st


st.title("Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input(
    "Send a Message"
):
    st.session_state.messages.append({
        "role": "user", "content": prompt
    })

    with st.chat_message("user"):
        st.text(prompt)

    response = requests.request("GET",
                                "http://localhost:8000/generate/text/",
                                params={"prompt": prompt})
    response.raise_for_status()

    def stream_data():
        for word in response.text.split(" "):
            yield word + " "
            time.sleep(0.2)

    with st.chat_message("assistant"):
        st.write_stream(stream_data)
        st.session_state.messages.append({
            "role": "assistant", "content": response.text
        })
