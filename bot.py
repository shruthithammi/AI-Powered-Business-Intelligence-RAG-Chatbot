"""
AI-Powered Business Intelligence Chatbot
RAG-Based Financial Report Analyzer
Optimized for 8GB RAM (phi3 via Ollama)

HOW TO RUN:
  pip install streamlit openai langchain langchain-community
  pip install langchain-text-splitters pypdf faiss-cpu sentence-transformers
  streamlit run bot.py
"""

import streamlit as st
import os
import tempfile
import warnings
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ─────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="AI Business Intelligence Chatbot",
    page_icon="📊",
    layout="wide"
)

# ─────────────────────────────────────────
# Session State
# ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "uploaded_files_names" not in st.session_state:
    st.session_state.uploaded_files_names = []

# ─────────────────────────────────────────
# Default Config (Ollama local)
# ─────────────────────────────────────────
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY  = "ollama"
DEFAULT_MODEL    = "phi3"

# ─────────────────────────────────────────
# Embeddings — cached so model loads only once
# ─────────────────────────────────────────
@st.cache_resource
def init_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
    )

# ─────────────────────────────────────────
# LLM Call
# ─────────────────────────────────────────
def get_completion(prompt, base_url, api_key, model):
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=60.0
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error calling LLM: {str(e)}"

# ─────────────────────────────────────────
# Process PDF Files
# ─────────────────────────────────────────
def process_uploaded_files(uploaded_files):
    with st.spinner("📚 Processing documents..."):
        try:
            docs = []
            with tempfile.TemporaryDirectory() as temp_dir:
                for uploaded_file in uploaded_files:
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    loader = PyPDFLoader(temp_path)
                    docs.extend(loader.load())

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            splits = splitter.split_documents(docs)
            embeddings = init_embeddings()
            vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
            return vectorstore, len(docs), len(splits)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            return None, 0, 0

# ─────────────────────────────────────────
# Build Prompt (FIXED — full context, k=3)
# ─────────────────────────────────────────
def build_prompt(query, docs):
    # FIXED: use full content of all 3 chunks (old code used only 150 chars!)
    context = "\n\n".join([doc.page_content for doc in docs]) if docs else "No context found."

    prompt = f"""You are a financial analyst assistant.
Answer the question accurately using ONLY the context provided below.
If the answer contains a number or financial figure, state it exactly as it appears in the context.
If the information is not in the context, say: "This information is not available in the uploaded document."
Be concise and precise.

CONTEXT FROM PDF:
{context}

QUESTION: {query}

ANSWER:"""
    return prompt

# ─────────────────────────────────────────
# RAG Pipeline
# ─────────────────────────────────────────
def get_rag_response(query, vectorstore, base_url, api_key, model):
    if vectorstore is None:
        return "⚠️ Please upload a PDF document first."
    try:
        # FIXED: k=3 (old code used k=1 — missed important chunks)
        docs = vectorstore.similarity_search(query, k=3)
        prompt = build_prompt(query, docs)
        return get_completion(prompt, base_url, api_key, model)
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────
st.title("📊 AI-Powered Business Intelligence Chatbot")
st.success("📄 Answers strictly from your uploaded PDF — no hallucination")

with st.sidebar:
    st.header("⚙️ Settings")
    base_url   = st.text_input("Base URL", value=DEFAULT_BASE_URL)
    api_key    = st.text_input("API Key",  value=DEFAULT_API_KEY, type="password")
    model_name = st.text_input("Model",    value=DEFAULT_MODEL)

    st.divider()
    st.header("📁 Upload PDF")
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        names = [f.name for f in uploaded_files]
        if names != st.session_state.uploaded_files_names:
            st.session_state.uploaded_files_names = names
            vectorstore, pages, chunks = process_uploaded_files(uploaded_files)
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.success(f"✅ {pages} pages loaded")
                st.info(f"🔢 {chunks} chunks indexed")
                st.session_state.messages = []

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("1. Upload PDF\n2. Split into chunks\n3. Store in FAISS\n4. Question → find top 3 chunks\n5. LLM reads chunks → Answer")

if not st.session_state.vectorstore:
    st.info("👈 Upload a PDF from the sidebar to begin.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about your PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🔎 Searching PDF and generating answer..."):
                response = get_rag_response(
                    prompt, st.session_state.vectorstore,
                    base_url, api_key, model_name
                )
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("---")
