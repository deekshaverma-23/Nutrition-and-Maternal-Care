import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import OllamaEmbeddings

DB_PATH = "db/"

@st.cache_resource
def get_rag_chain(llms):

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
    )

    retriever = db.as_retriever(search_kwargs={"k": 3})

    system_prompt = """
You are a nutrition & maternal-care assistant for India.

Rules:
- Answer in the SAME language as the question.
- Use simple practical advice.
- Personalize for pregnancy / infant / child.
- If unsure, say you don't know.

Context:
{context}
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    document_chain = create_stuff_documents_chain(
        llms["llm"],
        prompt,
    )

    return create_retrieval_chain(
        retriever,
        document_chain,
    )
