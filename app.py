import os
import shutil
import streamlit as st
from src.config import CHROMA_DIR, DATA_DIR, REPO_DOCS_DIR, IS_STREAMLIT_CLOUD

st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="brain",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.chain import ask_question
from src.ingest import run_ingestion


def seed_sample_docs():
    """On Streamlit Cloud cold start, copy sample PDFs from repo into writable /tmp."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    repo_pdfs = list(REPO_DOCS_DIR.glob("*.pdf")) if REPO_DOCS_DIR.exists() else []
    local_pdfs = list(DATA_DIR.glob("*.pdf"))
    if not local_pdfs and repo_pdfs:
        for pdf in repo_pdfs:
            shutil.copy2(pdf, DATA_DIR / pdf.name)
        st.toast(f"Copied {len(repo_pdfs)} sample PDF(s) for first-time setup")


if IS_STREAMLIT_CLOUD:
    seed_sample_docs()

if not CHROMA_DIR.exists():
    pdfs = list(DATA_DIR.glob("*.pdf"))
    if pdfs:
        with st.spinner(f"First run — ingesting {len(pdfs)} document(s)... this takes ~30 seconds"):
            run_ingestion()
        st.toast("Knowledge base ready!")

GREETINGS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "howdy", "sup", "yo"}

with st.sidebar:
    st.header("1. Enter API Key")
    user_api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get a free key at https://console.groq.com/keys"
    )
    if user_api_key:
        st.success("API key set")
    else:
        st.info("Get your free API key at [console.groq.com/keys](https://console.groq.com/keys)")

    st.divider()
    st.header("2. Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDFs to build your knowledge base"
    )

    if uploaded_files:
        if st.button("Build Knowledge Base", type="primary", use_container_width=True):
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            for f in DATA_DIR.glob("*.pdf"):
                f.unlink()
            if CHROMA_DIR.exists():
                shutil.rmtree(CHROMA_DIR)

            for uploaded_file in uploaded_files:
                dest = DATA_DIR / uploaded_file.name
                with open(dest, "wb") as out:
                    out.write(uploaded_file.getbuffer())

            with st.spinner(f"Ingesting {len(uploaded_files)} document(s)..."):
                run_ingestion()
            st.success(f"Done! {len(uploaded_files)} document(s) ingested. Start asking questions.")
            st.rerun()

    if CHROMA_DIR.exists():
        existing_pdfs = list(DATA_DIR.glob("*.pdf"))
        if existing_pdfs:
            st.divider()
            st.caption(f"Knowledge base: {len(existing_pdfs)} document(s) loaded")
            for pdf in existing_pdfs:
                st.caption(f"  - {pdf.name}")

if not user_api_key:
    st.info("Enter your Groq API key in the sidebar to get started.")
    st.stop()

if not CHROMA_DIR.exists():
    st.warning("Upload PDF documents in the sidebar to build your knowledge base.")
    st.stop()

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }

    .bg-layer {
        position: fixed; inset: 0; pointer-events: none; z-index: 0; overflow: hidden;
    }
    .orb {
        position: absolute; border-radius: 50%; filter: blur(100px);
    }
    .orb-1 {
        width: 500px; height: 500px;
        background: radial-gradient(circle, rgba(102,126,234,0.25), transparent 70%);
        top: -5%; left: -5%;
        animation: drift1 12s ease-in-out infinite alternate;
    }
    .orb-2 {
        width: 450px; height: 450px;
        background: radial-gradient(circle, rgba(118,75,162,0.20), transparent 70%);
        bottom: -10%; right: -5%;
        animation: drift2 15s ease-in-out infinite alternate;
    }
    .orb-3 {
        width: 350px; height: 350px;
        background: radial-gradient(circle, rgba(99,102,241,0.15), transparent 70%);
        top: 40%; left: 50%;
        animation: drift3 18s ease-in-out infinite alternate;
    }

    @keyframes drift1 { 0% { transform: translate(0, 0) scale(1); } 100% { transform: translate(80px, 60px) scale(1.1); } }
    @keyframes drift2 { 0% { transform: translate(0, 0) scale(1); } 100% { transform: translate(-60px, -80px) scale(1.05); } }
    @keyframes drift3 { 0% { transform: translate(0, 0) scale(1); } 100% { transform: translate(-40px, 50px) scale(0.95); } }

    .grid-overlay {
        position: absolute; inset: 0;
        background-image:
            linear-gradient(rgba(102,126,234,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(102,126,234,0.03) 1px, transparent 1px);
        background-size: 60px 60px;
        animation: gridPulse 8s ease-in-out infinite alternate;
    }
    @keyframes gridPulse { 0% { opacity: 0.3; } 100% { opacity: 0.7; } }

    .hero { text-align: center; padding: 2rem 0 0.5rem 0; position: relative; z-index: 1; }
    .hero-label {
        display: inline-block; font-size: 0.7rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.15em; color: #667eea;
        background: rgba(102,126,234,0.1); border: 1px solid rgba(102,126,234,0.2);
        padding: 0.3rem 1rem; border-radius: 20px; margin-bottom: 1rem;
    }
    .hero h1 { font-size: 3rem; font-weight: 800; line-height: 1.15; margin-bottom: 0.6rem; }
    .hero h1 .gradient-text {
        background: linear-gradient(135deg, #667eea 0%, #a855f7 50%, #667eea 100%);
        background-size: 200% 200%;
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        animation: gradientShift 4s ease-in-out infinite;
    }
    @keyframes gradientShift { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
    .hero h1 .white-text { color: #e6e6e6; -webkit-text-fill-color: #e6e6e6; }
    .hero .subtitle { color: #8892b0; font-size: 1.05rem; font-weight: 300; max-width: 500px; margin: 0 auto 1rem auto; line-height: 1.6; }
    .badge-row { display: flex; justify-content: center; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.5rem; }
    .badge {
        background: rgba(102,126,234,0.08); border: 1px solid rgba(102,126,234,0.2);
        color: #b4bfee; padding: 0.3rem 0.9rem; border-radius: 20px;
        font-size: 0.72rem; font-weight: 500;
    }

    div[data-testid="stChatMessage"] {
        background: rgba(255,255,255,0.03) !important; border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 16px !important; padding: 1rem 1.5rem !important; margin-bottom: 0.6rem !important;
        backdrop-filter: blur(8px);
    }
    div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] li,
    div[data-testid="stChatMessage"] code { color: #e6e6e6 !important; line-height: 1.8 !important; }
    div[data-testid="stChatMessage"] h1 { font-size: 1.2rem !important; font-weight: 700 !important; margin: 0.8rem 0 0.4rem 0 !important; }
    div[data-testid="stChatMessage"] h2 { font-size: 1.05rem !important; font-weight: 600 !important; margin: 0.6rem 0 0.3rem 0 !important; }
    div[data-testid="stChatMessage"] h3 { font-size: 0.95rem !important; font-weight: 600 !important; margin: 0.5rem 0 0.2rem 0 !important; }

    .source-card {
        background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 0.6rem;
        transition: all 0.3s ease;
    }
    .source-card:hover { background: rgba(255,255,255,0.06); border-color: rgba(102,126,234,0.2); transform: translateX(4px); }
    .source-header { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.4rem; }
    .source-file { color: #667eea; font-weight: 600; font-size: 0.85rem; }
    .source-page { color: #b4bfee; font-size: 0.7rem; background: rgba(102,126,234,0.12); padding: 0.15rem 0.5rem; border-radius: 10px; font-weight: 500; }
    .source-preview { color: #a0a8c0; font-size: 0.8rem; line-height: 1.6; }

    .footer { text-align: center; padding: 2rem 0 5rem 0; position: relative; z-index: 1; }
    .footer-inner {
        display: inline-flex; align-items: center; gap: 0.6rem;
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
        border-radius: 20px; padding: 0.5rem 1.2rem; font-size: 0.72rem; color: #5a6380;
    }
    .footer-inner a { color: #667eea; text-decoration: none; font-weight: 500; }
    .footer-dot { width: 6px; height: 6px; background: #22c55e; border-radius: 50%; display: inline-block; animation: pulse 2s ease-in-out infinite; }
    @keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.5; transform: scale(0.8); } }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="bg-layer">
    <div class="grid-overlay"></div>
    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>
    <div class="orb orb-3"></div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <span class="hero-label">Powered by RAG + Groq</span>
    <h1>
        <span class="white-text">Ask your</span><br>
        <span class="gradient-text">Knowledge Base</span>
    </h1>
    <p class="subtitle">
        Upload any PDF documents and ask questions — get AI-powered answers with source citations
    </p>
    <div class="badge-row">
        <span class="badge">LangChain</span>
        <span class="badge">ChromaDB</span>
        <span class="badge">Groq</span>
        <span class="badge">RAGAS Evaluated</span>
    </div>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0 1rem 0;">
        <p style="color: #8892b0; font-size: 0.95rem; margin-bottom: 0.8rem;">
            A sample Python tutorial is pre-loaded. Try asking:
        </p>
    </div>
    """, unsafe_allow_html=True)

    suggestions = [
        "What are Python decorators and how do they work?",
        "Explain list comprehensions with examples",
        "How does error handling work in Python?",
    ]
    cols = st.columns(len(suggestions))
    for col, suggestion in zip(cols, suggestions):
        if col.button(suggestion, use_container_width=True):
            st.session_state.pending_question = suggestion
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"View Sources ({len(msg['sources'])})"):
                for src in msg["sources"]:
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-header">
                            <span class="source-file">{src['file']}</span>
                            <span class="source-page">Page {src['page']}</span>
                        </div>
                        <div class="source-preview">{src['preview']}</div>
                    </div>
                    """, unsafe_allow_html=True)

question = st.chat_input("Try: 'What are Python decorators?' or ask anything from your documents...")

if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if question.strip().lower().rstrip("!.,") in GREETINGS:
            greeting = (
                "Welcome! I can answer questions from the documents in the knowledge base. "
                "Just ask me anything and I'll find the answer with source citations."
            )
            st.markdown(greeting)
            st.session_state.messages.append({"role": "assistant", "content": greeting, "sources": []})
        else:
            with st.spinner("Searching knowledge base..."):
                answer, docs = ask_question(question, api_key=user_api_key)

            if answer:
                st.markdown(answer)
                sources = []
                for doc in docs:
                    sources.append({
                        "file": os.path.basename(doc.metadata["source"]),
                        "page": doc.metadata.get("page", "?"),
                        "preview": doc.page_content[:250].replace("\n", " ") + "...",
                    })
                with st.expander(f"View Sources ({len(sources)})"):
                    for src in sources:
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-header">
                                <span class="source-file">{src['file']}</span>
                                <span class="source-page">Page {src['page']}</span>
                            </div>
                            <div class="source-preview">{src['preview']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
            else:
                no_info = "No relevant documents found. Try rephrasing your question."
                st.warning(no_info)
                st.session_state.messages.append({"role": "assistant", "content": no_info, "sources": []})

st.markdown("""
<div class="footer">
    <div class="footer-inner">
        <span class="footer-dot"></span>
        Built with LangChain, ChromaDB & Groq
        &nbsp;&middot;&nbsp;
        Evaluated with <a href="https://docs.ragas.io" target="_blank">RAGAS</a>
        &nbsp;&middot;&nbsp;
        Traced with <a href="https://smith.langchain.com" target="_blank">LangSmith</a>
    </div>
</div>
""", unsafe_allow_html=True)
