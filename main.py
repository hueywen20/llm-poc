import os
import streamlit as st

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFacePipeline
)
from langchain_community.vectorstores import FAISS

# -----------------------------------
# Streamlit config
# -----------------------------------
st.set_page_config(
    page_title="AI Knowledge Assistant",
    layout="wide"
)

st.title("📚 Free AI Knowledge Assistant")

# -----------------------------------
# Load vector database
# -----------------------------------
@st.cache_resource
def load_vector_db():

    docs = []

    if not os.path.exists("docs"):
        os.makedirs("docs")

    for file in os.listdir("docs"):

        if file.endswith(".pdf"):

            loader = PyPDFLoader(
                os.path.join("docs", file)
            )

            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_db

vector_db = load_vector_db()

# -----------------------------------
# Load HuggingFace model
# -----------------------------------
@st.cache_resource
def load_hf_model():

    model_name = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name
    )

    pipe = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256
    )

    llm = HuggingFacePipeline(
        pipeline=pipe
    )

    return llm

llm = load_hf_model()

# -----------------------------------
# User input
# -----------------------------------
query = st.text_input(
    "Ask a question based on your documents:"
)

if query:

    docs = vector_db.similarity_search(
        query,
        k=3
    )

    context = "\n\n".join([
        d.page_content for d in docs
    ])

    prompt = f"""
Answer the question based ONLY on the context below.

Context:
{context}

Question:
{query}

Answer clearly and concisely.
"""

    try:

        answer = llm.invoke(prompt)

        st.subheader("Answer")
        st.write(answer)

    except Exception as e:

        st.error(
            f"Error generating answer: {str(e)}"
        )
