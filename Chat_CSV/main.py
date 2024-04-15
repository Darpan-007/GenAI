import streamlit as st
import time
from langchain_openai import OpenAI
from langchain_experimental.agents.agent_toolkits import (
    create_csv_agent,
    create_pandas_dataframe_agent,
)
from dotenv import load_dotenv
import pandas as pd

load_dotenv()


if "agent" not in st.session_state:
    st.session_state.agent = None


def response_generator(query):
    response = st.session_state.agent.invoke(query)["output"]

    for word in response.split():
        yield word + " "
        time.sleep(0.05)


def main():

    st.set_page_config(page_title="Chat with CSV", page_icon="📈")
    st.header("Ask your CSV 📈")

    with st.sidebar:
        file = st.file_uploader("Choose a CSV File", type=["csv"])

    if file is not None:
        # create an agent
        st.session_state.agent = create_csv_agent(
            OpenAI(temperature=0),
            file,
            verbose=True,
        )

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Accept user input
        if query := st.chat_input("Ask a question...."):
            # Display user message in chat message container
            with st.chat_message("user"):
                st.write(query)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": query})

            with st.chat_message("assistant"):
                response = st.write_stream(response_generator(query))
                # Add assistant response to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )


if __name__ == "__main__":
    main()
