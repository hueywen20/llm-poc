import streamlit as st
import os

# ------------------------------
# LangChain imports (updated)
# ------------------------------
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM

# ------------------------------
# Streamlit config
# ------------------------------
st.set_page_config(page_title="AI Knowledge Assistant", layout="wide")
st.title("📚 AI Knowledge Assistant (Local PoC)")

# ------------------------------
# Load and index PDF documents
# ------------------------------
@st.cache_resource
def load_vector_db():
    docs = []

    # Load all PDFs in "docs" folder
    for file in os.listdir("docs"):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(f"docs/{file}")
            docs.extend(loader.load())

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    # Generate embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Create FAISS vector DB
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db

vector_db = load_vector_db()

# ------------------------------
# Initialize Ollama LLM
# ------------------------------
llm = OllamaLLM(model="mistral")  # Replace "mistral" with your preferred model

# ------------------------------
# User query input
# ------------------------------
query = st.text_input("Ask a question based on your documents:")

if query:
    # Retrieve top 3 similar document chunks
    docs = vector_db.similarity_search(query, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    # Construct prompt for the LLM
    prompt = f"""
You are an assistant answering questions based only on the context below.

Context:
{context}

Question:
{query}

Answer clearly and concisely.
"""

    # Generate answer using OllamaLLM
    try:
        result = llm.generate([prompt])  # ✅ pass list of strings
        answer = result.generations[0][0].text  # ✅ access text
        st.subheader("Answer")
        st.write(answer)
    except Exception as e:
        st.error(f"Error generating answer: {e}")
