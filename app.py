import streamlit as st
from rag_pipeline import build_rag_pipeline

# Try Streamlit Cloud secrets first, then .env file
try:
    if "GROQ_API_KEY" not in os.environ:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except:
    # Fallback to .env file (local development)
    from dotenv import load_dotenv
    load_dotenv()
    
# ── Page Configuration ─────────────────────────────────────────────────
st.set_page_config(
    page_title="RideEasy Customer Support",
    page_icon="🏍️",
    layout="centered"
)

# ── Header ─────────────────────────────────────────────────────────────
st.title("🏍️ RideEasy Customer Support")
st.caption(
    "AI Assistant for sales & marketing team — "
    "pricing info, rental requirements, motorcycle recommendations, and services"
)

# ── Load RAG Pipeline ──────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    return build_rag_pipeline()

if "pipeline_loaded" not in st.session_state:
    with st.status("Loading AI system...", expanded=True) as status:
        st.write("Reading motorcycle rental catalog...")
        st.write("Building vector store...")
        st.write("Initializing language model...")
        chain, num_chunks = load_pipeline()
        st.session_state.chain = chain
        st.session_state.num_chunks = num_chunks
        st.session_state.pipeline_loaded = True
        status.update(
            label=f"System ready! {num_chunks} document chunks indexed.",
            state="complete"
        )

chain = st.session_state.chain

# ── Initialize Chat History ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Display Example Questions ──────────────────────────────────────────
if not st.session_state.messages:
    st.info(
        "**Example questions you can ask:**\n\n"
        "- How much is Honda Beat rental per day?\n"
        "- What are the requirements for renting a motorcycle?\n"
        "- Recommend a comfortable motorcycle for touring\n"
        "- Compare NMAX and PCX for long distance travel\n"
        "- What is the penalty for late return?\n"
        "- Do you provide airport pick-up and delivery?\n"
        "- Which motorcycle is the most fuel efficient?\n"
        "- Is there a discount for monthly rental?"
    )

# ── Display Chat History ───────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── User Input ─────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask anything about motorcycle rental..."):

    # Save and display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate answer from RAG chain
    with st.chat_message("assistant"):
        with st.spinner("Searching catalog..."):
            result = chain.invoke({"query": user_input})
            answer = result["result"]
            source_docs = result["source_documents"]

        st.markdown(answer)

    # Save answer to history
    st.session_state.messages.append({"role": "assistant", "content": answer})


# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📋 About")
    st.markdown(
        "This application uses **RAG** "
        "_(Retrieval-Augmented Generation)_ technology to answer "
        "questions based on the official motorcycle rental catalog.\n\n"
        "Answers are based **only** on the catalog document, "
        "not the AI's general knowledge."
    )

    st.divider()

    st.subheader("🏍️ Motorcycle Categories")
    st.markdown(
        "**Economy Matic:**\n"
        "- Honda Beat (Rp 75,000/day)\n"
        "- Honda Scoopy (Rp 75,000/day)\n"
        "- Yamaha Mio (Rp 75,000/day)\n"
        "- Yamaha Fazzio (Rp 80,000/day)\n\n"
        "**Premium Matic:**\n"
        "- Honda PCX 160 (Rp 125,000/day)\n"
        "- Yamaha NMAX (Rp 125,000/day)\n"
        "- Yamaha Aerox (Rp 130,000/day)\n\n"
        "**Economy Manual:**\n"
        "- Honda Supra X (Rp 60,000/day)\n"
        "- Honda Revo (Rp 60,000/day)\n"
        "- Yamaha Jupiter (Rp 65,000/day)\n\n"
        "**Premium Trail:**\n"
        "- Honda CRF 150L (Rp 150,000/day)\n"
        "- Kawasaki KLX 150 (Rp 150,000/day)\n\n"
        "**Electric:**\n"
        "- Gesits (Rp 85,000/day)\n"
        "- Volta 401 (Rp 100,000/day)"
    )

    st.divider()

    st.subheader("⚙️ System Architecture")
    st.markdown(
        "```\n"
        "Rental Catalog (MD)\n"
        "       ↓\n"
        "  Document Loader\n"
        "       ↓\n"
        "  Text Splitter\n"
        "       ↓\n"
        "HuggingFace Embeddings\n"
        "       ↓\n"
        "  FAISS Vector Store\n"
        "       ↓\n"
        "    Retriever\n"
        "       ↓\n"
        " Groq LLM (Llama 3.3)\n"
        "       ↓\n"
        "  Final Answer\n"
        "```"
    )

    st.divider()

    if st.button("🔄 Reset Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()