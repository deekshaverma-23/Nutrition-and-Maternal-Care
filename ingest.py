import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

DATA_PATH = "data/"
DB_PATH = "db/"


def create_vector_db():
    print("--- Starting the ingestion process ---")

    documents = []
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(DATA_PATH, filename)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())

    print(f"Loaded {len(documents)} pages from PDF files.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )

    chunks = splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")

    embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
    )


    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH,
    )

    db.persist()

    print("--- Vector database created successfully! ---")
    print(f"--- Stored in '{DB_PATH}' ---")


if __name__ == "__main__":
    create_vector_db()
