import streamlit as st
from langchain_ollama import OllamaLLM, ChatOllama


@st.cache_resource
def get_llms():

    llm = OllamaLLM(
        model="llama3",
        temperature=0.7,
    )

    llm_translator = OllamaLLM(
        model="llama3",
        temperature=0,
    )

    llm_vision = ChatOllama(
        model="llava",
        temperature=0.2,
    )

    return {
        "llm": llm,
        "llm_vision": llm_vision,
        "llm_translator": llm_translator,
    }