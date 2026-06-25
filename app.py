# ============================================================
# SMARTPHONE ADVISOR
# Chatbot Perbandingan Produk untuk Tim Sales & Marketing
# Powered by LangChain + Groq + FAISS
# ============================================================
#
# CARA MENJALANKAN:
#   streamlit run app.py
#
# ============================================================

import streamlit as st
from rag_pipeline import build_rag_pipeline

# ── Konfigurasi Halaman ────────────────────────────────────────────────
st.set_page_config(
    page_title="Smartphone Advisor",
    page_icon="📱",
    layout="centered"
)

# ── Header ─────────────────────────────────────────────────────────────
st.title("📱 Smartphone Advisor")
st.caption(
    "Asisten AI untuk tim sales & marketing — "
    "rekomendasi dan perbandingan produk smartphone berdasarkan katalog resmi"
)

# ── Load RAG Pipeline ──────────────────────────────────────────────────
# Menggunakan st.cache_resource agar pipeline hanya dibangun sekali.
# Tanpa ini, pipeline akan dibangun ulang setiap ada interaksi pengguna.
@st.cache_resource(show_spinner=False)
def load_pipeline():
    return build_rag_pipeline()

# Tampilkan proses loading kepada pengguna
if "pipeline_loaded" not in st.session_state:
    with st.status("Memuat sistem AI...", expanded=True) as status:
        st.write("Membaca katalog produk...")
        st.write("Membangun vector store...")
        st.write("Menginisialisasi model bahasa...")
        chain, num_chunks = load_pipeline()
        st.session_state.chain = chain
        st.session_state.num_chunks = num_chunks
        st.session_state.pipeline_loaded = True
        status.update(
            label=f"Sistem siap! {num_chunks} potongan dokumen berhasil diindeks.",
            state="complete"
        )

chain = st.session_state.chain

# ── Inisialisasi Riwayat Chat ──────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Tampilkan Contoh Pertanyaan (hanya saat belum ada chat) ───────────
if not st.session_state.messages:
    st.info(
        "**Contoh pertanyaan yang bisa Anda ajukan:**\n\n"
        "- Rekomendasikan smartphone untuk fotografi dengan budget 5 juta\n"
        "- Bandingkan Samsung Galaxy S24 dengan iPhone 15\n"
        "- HP mana yang cocok untuk konten kreator video?\n"
        "- Smartphone mana yang pengisian baterainya paling cepat?\n"
        "- Produk apa yang paling cocok untuk pengguna aktif outdoor?\n"
        "- Apa perbedaan iPhone 15 dan iPhone 15 Pro Max?"
    )

# ── Tampilkan Riwayat Chat ─────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Input Pengguna ─────────────────────────────────────────────────────
if user_input := st.chat_input("Tanyakan sesuatu tentang produk smartphone..."):

    # Simpan dan tampilkan pesan pengguna
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate jawaban dari RAG chain
    with st.chat_message("assistant"):
        with st.spinner("Mencari informasi di katalog..."):
            result = chain.invoke({"query": user_input})
            answer = result["result"]
            source_docs = result["source_documents"]

        st.markdown(answer)

        # Tampilkan referensi dokumen sumber (bisa di-collapse)
        with st.expander("Lihat referensi dari katalog"):
            for i, doc in enumerate(source_docs, 1):
                st.markdown(f"**Referensi {i}:**")
                st.text(doc.page_content[:300] + "...")
                st.divider()

    # Simpan jawaban ke riwayat
    st.session_state.messages.append({"role": "assistant", "content": answer})


# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Tentang Aplikasi")
    st.markdown(
        "Aplikasi ini menggunakan teknologi **RAG** "
        "_(Retrieval-Augmented Generation)_ untuk menjawab "
        "pertanyaan berdasarkan katalog produk resmi.\n\n"
        "Jawaban didasarkan **hanya** pada dokumen katalog, "
        "bukan pengetahuan umum AI."
    )

    st.divider()

    st.subheader("📱 Produk Tersedia")
    st.markdown(
        "1. Xiaomi Redmi Note 13 Pro+ 5G\n"
        "2. Samsung Galaxy A55 5G\n"
        "3. OPPO Reno 12 Pro\n"
        "4. Samsung Galaxy S24\n"
        "5. Apple iPhone 15\n"
        "6. Apple iPhone 15 Pro Max"
    )

    st.divider()

    st.subheader("⚙️ Arsitektur Sistem")
    st.markdown(
        "```\n"
        "Katalog Produk (TXT)\n"
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
        "  Jawaban Final\n"
        "```"
    )

    st.divider()

    if st.button("🔄 Reset Percakapan", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
