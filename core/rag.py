import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import OllamaEmbeddings

DB_PATH = "db/"

def get_rag_chain(_llms):

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
    )

    retriever = db.as_retriever(search_kwargs={"k": 3})

    system_prompt = """
    You are a trusted nutrition and maternal-care assistant for families in India.

    Your goals:
    - Give medically responsible, practical advice for pregnancy, infants, and young children.
    - Follow WHO / Indian public-health style guidance where relevant.
    - Be calm, warm, and reassuring in tone.

    Language rules:
    - ALWAYS reply in the same language as the user's question.
    - Use natural, fluent, native phrasing (not literal word-for-word translation).
    - Prefer commonly used Indian terms for food and health.
    - Avoid robotic or machine-translated style.

    Personalization rules:
    - Adapt advice based on:
    • pregnant woman stage
    • infant age (months)
    • child age (years)
    • diet type
    • medical conditions or allergies
    - Mention special cases clearly (e.g., lactose intolerance, anemia, diabetes).

    Safety rules:
    - Do NOT give diagnoses.
    - Encourage consulting a doctor for serious symptoms.
    - If information is uncertain, clearly say so.

    Style:
    - Short paragraphs or bullet points.
    - Simple everyday language.
    - No unnecessary introductions.
    - No meta commentary.
    - No mentioning that you are translating.

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
        _llms["llm"],
        prompt,
    )

    return create_retrieval_chain(
        retriever,
        document_chain,
    )