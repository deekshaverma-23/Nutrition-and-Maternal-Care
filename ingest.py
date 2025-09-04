import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

DATA_PATH = "data/"
DB_PATH = "db/"

def create_vector_db():
    """
    Creates a Chroma vector database from the documents in the DATA_PATH.
    """
    print("--- Starting the ingestion process ---")

    documents = []
    for filename in os.listdir(DATA_PATH):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(DATA_PATH, filename)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    print(f"Loaded {len(documents)} pages from PDF files.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    
    print("--- Vector database created successfully! ---")
    print(f"--- All data is stored in the '{DB_PATH}' directory. ---")


if __name__ == '__main__':
    create_vector_db()