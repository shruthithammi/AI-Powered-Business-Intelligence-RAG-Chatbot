"""
AI-Powered Business Intelligence Chatbot
PDF + Web Search Combined Mode
Uses DuckDuckGo (free, no API key needed)

HOW TO RUN:
  pip install streamlit openai langchain langchain-community
  pip install langchain-text-splitters pypdf faiss-cpu sentence-transformers
  pip install duckduckgo-search
  streamlit run bot_websearch.py
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
    page_title="AI Chatbot — PDF + Web Search",
    page_icon="🌐",
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
# Default Config
# ─────────────────────────────────────────
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY  = "ollama"
DEFAULT_MODEL    = "phi3"

# ─────────────────────────────────────────
# Embeddings
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
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ LLM Error: {str(e)}"

# ─────────────────────────────────────────
# Web Search — DuckDuckGo (free, no key)
# ─────────────────────────────────────────
def web_search(query, max_results=3):
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title":   r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url":     r.get("href", "")
                })
        return results
    except Exception:
        return []   # silent fallback — PDF will still answer

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
# Build Prompt (strict: actual vs forecast)
# ─────────────────────────────────────────
def build_combined_prompt(query, pdf_docs, web_results):
    if pdf_docs:
        pdf_context = "\n\n".join([doc.page_content for doc in pdf_docs])
    else:
        pdf_context = "No PDF uploaded."

    if web_results:
        web_context = "\n\n".join([
            f"[Web: {r['title']}]\n{r['snippet']}"
            for r in web_results if r.get("snippet")
        ])
    else:
        web_context = "No web results available."

    prompt = f"""You are a precise financial analyst assistant.
You have two sources: a PDF document and web search results.

CRITICAL RULES:
1. Always prefer the PDF context for exact figures.
2. Report ACTUAL reported figures — NOT future forecasts or guidance.
3. Example: "capital expenditures 2024" = the ACTUAL reported 2024 figure ($39.23B),
   NOT the 2025 forecast ($60-65B).
4. State numbers exactly as they appear in the PDF.
5. If PDF has no answer, use web results.
6. End your answer with [Source: PDF], [Source: Web], or [Source: PDF + Web].

PDF CONTEXT:
{pdf_context}

WEB SEARCH RESULTS:
{web_context}

QUESTION: {query}

ANSWER:"""
    return prompt

# ─────────────────────────────────────────
# Full Hybrid RAG Response
# ─────────────────────────────────────────
def get_hybrid_response(query, vectorstore, base_url, api_key, model, use_web, use_pdf):
    pdf_docs    = []
    web_results = []

    if use_pdf and vectorstore is not None:
        pdf_docs = vectorstore.similarity_search(query, k=3)

    if use_web:
        with st.spinner("🌐 Searching the web..."):
            web_results = web_search(query, max_results=3)

    prompt   = build_combined_prompt(query, pdf_docs, web_results)
    response = get_completion(prompt, base_url, api_key, model)
    return response, web_results

# ─────────────────────────────────────────
# UI — Title
# ─────────────────────────────────────────
st.title("🌐 AI Chatbot — PDF + Web Search")
st.success("Answers from your PDF document AND live web search combined")

# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    base_url   = st.text_input("Base URL", value=DEFAULT_BASE_URL)
    api_key    = st.text_input("API Key",  value=DEFAULT_API_KEY, type="password")
    model_name = st.text_input("Model",    value=DEFAULT_MODEL)

    st.divider()
    st.header("🔧 Search Mode")
    use_pdf = st.checkbox("📄 Use PDF",        value=True)
    use_web = st.checkbox("🌐 Use Web Search", value=True)

    if not use_pdf and not use_web:
        st.warning("⚠️ Enable at least one source!")

    st.divider()
    st.header("📁 Upload PDF (Optional)")
    uploaded_files = st.file_uploader(
        "Upload PDF files", type=["pdf"], accept_multiple_files=True
    )

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
    st.markdown("📄 **PDF only** — strict answers from document\n\n🌐 **Web only** — live DuckDuckGo search\n\n🔀 **Both ON** — PDF first, web fills the gaps")

# ─────────────────────────────────────────
# FIX: Status Bar — use if/else blocks
# NOT one-liner ternary (caused the rendering bug)
# ─────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    if st.session_state.vectorstore:
        st.success("📄 PDF: Loaded ✅")
    else:
        st.warning("📄 PDF: Not uploaded")

with col2:
    if use_web:
        st.success("🌐 Web: ON ✅")
    else:
        st.info("🌐 Web: OFF")

with col3:
    if use_pdf and use_web:
        st.success("🔀 Mode: PDF + Web")
    elif use_pdf:
        st.info("📄 Mode: PDF Only")
    elif use_web:
        st.info("🌐 Mode: Web Only")
    else:
        st.error("❌ No source selected")

st.markdown("---")

# ─────────────────────────────────────────
# Chat History
# ─────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─────────────────────────────────────────
# Chat Input
# ─────────────────────────────────────────
if prompt := st.chat_input("Ask anything — searches PDF and/or web..."):
    if not use_pdf and not use_web:
        st.error("⚠️ Please enable at least one source in the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🔎 Gathering information..."):
                response, web_results = get_hybrid_response(
                    prompt,
                    st.session_state.vectorstore,
                    base_url, api_key, model_name,
                    use_web, use_pdf
                )
            st.markdown(response)

            if use_web and web_results:
                with st.expander("🌐 Web Sources Used"):
                    for r in web_results:
                        if r.get("url"):
                            st.markdown(f"- [{r['title']}]({r['url']})")
            elif use_web and not web_results:
                st.caption("⚠️ Web search unavailable — answered from PDF only")

        st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("---")

