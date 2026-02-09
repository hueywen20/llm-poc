import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

st.set_page_config(page_title="AI Knowledge Assistant", layout="wide")
st.title("📚 Free AI Knowledge Assistant (HuggingFace)")

# -----------------------------
# Load PDF and build vector DB
# -----------------------------
@st.cache_resource
def load_vector_db():
    docs = []
    for file in os.listdir("docs"):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(f"docs/{file}")
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db

vector_db = load_vector_db()

# -----------------------------
# Load HuggingFace model
# -----------------------------
@st.cache_resource
def load_hf_model():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
    return HuggingFacePipeline(pipeline=pipe)

llm = load_hf_model()

# -----------------------------
# User query
# -----------------------------
query = st.text_input("Ask a question based on your documents:")

if query:
    docs = vector_db.similarity_search(query, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
Answer the question based only on the context below.

Context:
{context}

Question:
{query}

Answer clearly and concisely.
"""

    # Generate answer
    try:
        result = llm.generate([prompt])  # ✅ pass list of strings
        answer = result.generations[0][0].text  # ✅ extract text
        st.subheader("Answer")
        st.write(answer)
    except Exception as e:
        st.error(f"Error generating answer: {e}")
