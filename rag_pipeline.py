import os
from dotenv import load_dotenv       

from langchain_community.document_loaders import TextLoader             
from langchain_text_splitters import RecursiveCharacterTextSplitter    
from langchain_huggingface import HuggingFaceEmbeddings                
from langchain_community.vectorstores import FAISS                      
from langchain_groq import ChatGroq               
from langchain.chains import RetrievalQA                               
from langchain.prompts import PromptTemplate                           

# Try Streamlit Cloud secrets first, then .env file
try:
    if "GROQ_API_KEY" not in os.environ:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except:
    # Fallback to .env file (local development)
    from dotenv import load_dotenv
    load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────

# Location of motorcycle rental catalog file
DATA_PATH = "data/Documents_Sales.txt"

# Location of system prompt file
SYSTEM_PROMPT_PATH = "system_prompt.txt"

# Embedding model: converts text into numerical vectors
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# LLM model that will answer questions (Google Gemini)
LLM_MODEL = "llama-3.3-70b-versatile"

# Temperature: controls randomness of LLM output
LLM_TEMPERATURE = 0.001

# Size of each text chunk (in characters)
CHUNK_SIZE = 800

# Overlap between chunks so context is not cut off
CHUNK_OVERLAP = 100

# How many chunks to retrieve for each question
TOP_K_RESULTS = 4


# ── Load System Prompt from File ───────────────────────────────────────

def load_system_prompt(path: str) -> str:
    """
    Reads the system_prompt.txt file and returns it as a string.

    System prompt is stored in a separate file so that:
    - Easy to modify without touching Python code
    - Safer: system instructions separate from program logic
    - Cleaner: Python code focuses on logic, not long text

    The file uses XML-style delimiters for:
    - Structural clarity (each section has opening and closing tags)
    - Security: LLMs are trained to respect XML tags as structural boundaries
    - Readability: anyone opening the file immediately understands which part is what
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


SYSTEM_PROMPT_TEMPLATE = load_system_prompt(SYSTEM_PROMPT_PATH)


# ── Build Pipeline Function ────────────────────────────────────────────

def build_rag_pipeline():
    """
    Builds a complete RAG pipeline from scratch.

    Returns:
    - chain: RetrievalQA object ready to accept questions
    - num_chunks: number of text chunks successfully indexed
    """

    # ------------------------------------------------------------------
    # STEP 1: LOAD — Read the motorcycle rental catalog file
    # ------------------------------------------------------------------
    # TextLoader reads a plain text file and converts it into Document
    loader = TextLoader(DATA_PATH, encoding="utf-8")
    documents = loader.load()

    # ------------------------------------------------------------------
    # STEP 2: CHUNK — Split the document into smaller pieces
    # ------------------------------------------------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n---\n", "\n\n", "\n", " "]
    )
    chunks = splitter.split_documents(documents)

    # ------------------------------------------------------------------
    # STEP 3: EMBED — Convert text into numerical vectors
    # ------------------------------------------------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # ------------------------------------------------------------------
    # STEP 4: STORE — Save vectors to FAISS
    # ------------------------------------------------------------------
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # ------------------------------------------------------------------
    # STEP 5: RETRIEVER — Set up the search mechanism
    # ------------------------------------------------------------------
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS}
    )

    # ------------------------------------------------------------------
    # STEP 6: LLM — Initialize Google Gemini via API
    # ------------------------------------------------------------------
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=os.getenv("GROQ_API_KEY")
    )

    # ------------------------------------------------------------------
    # STEP 7: PROMPT — Instruction template for the LLM
    # ------------------------------------------------------------------
    prompt = PromptTemplate(
        template=SYSTEM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    # ------------------------------------------------------------------
    # STEP 8: CHAIN — Combine all components
    # ------------------------------------------------------------------
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return chain, len(chunks)