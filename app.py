import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# Page config
st.set_page_config(page_title="LangChain Chatbot", page_icon="🤖")

# Title & description
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🤖 AI Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Ask anything and get intelligent answers</p>", unsafe_allow_html=True)

st.divider()

# Sidebar
st.sidebar.title("⚙️ Settings")
st.sidebar.write("This chatbot uses a free HuggingFace model")

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box
user_input = st.text_input("💬 Enter your question")

if user_input:

    # Load model
    pipe = pipeline("text-generation", model="gpt2")
    llm = HuggingFacePipeline(pipeline=pipe)

    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{question}")
    ])

    parser = StrOutputParser()
    chain = prompt | llm | parser

    response = chain.invoke({"question": user_input})

    # Save chat
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

# Display chat history
for role, message in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"🧑 **You:** {message}")
    else:
        st.markdown(f"🤖 **Bot:** {message}")

st.divider()

st.markdown("<p style='text-align:center; font-size:12px;'>Built with LangChain + HuggingFace + Streamlit</p>", unsafe_allow_html=True)
