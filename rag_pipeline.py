from dotenv import load_dotenv       
import os

from langchain_community.document_loaders import TextLoader             
from langchain_text_splitters import RecursiveCharacterTextSplitter    
from langchain_huggingface import HuggingFaceEmbeddings                
from langchain_community.vectorstores import FAISS                      
from langchain_groq import ChatGroq               
from langchain.chains import RetrievalQA                               
from langchain.prompts import PromptTemplate                           

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────
# Location of motorcycle rental catalog file
DATA_PATH = "data/Documents_Sales.txt"

# Location of system prompt file
SYSTEM_PROMPT_PATH = "system_prompt.txt"

# Embedding model: converts text into numerical vectors
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# LLM model that will answer questions
LLM_MODEL = "llama-3.3-70b-versatile"

# Temperature: controls randomness of LLM output
LLM_TEMPERATURE = 0.1

# Size of each text chunk (in characters)
CHUNK_SIZE = 800

# Overlap between chunks so context is not cut off
CHUNK_OVERLAP = 100

# How many chunks to retrieve for each question
TOP_K_RESULTS = 4


# ── Load System Prompt from File ───────────────────────────────────────

def load_system_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
    
SYSTEM_PROMPT_TEMPLATE = load_system_prompt(SYSTEM_PROMPT_PATH)


# ── Build Pipeline Function ────────────────────────────────────────────

def build_rag_pipeline():
    # TextLoader reads a plain text file and converts it into Document
    loader = TextLoader(DATA_PATH, encoding="utf-8")
    documents = loader.load()

    # CHUNK — Split the document into smaller pieces
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,chunk_overlap=CHUNK_OVERLAP,separators=["\n---\n", "\n\n", "\n", " "])
    chunks = splitter.split_documents(documents)

    # EMBED — Convert text into numerical vectors
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,model_kwargs={"device": "cpu"},encode_kwargs={"normalize_embeddings": True})

    # STORE — Save vectors to FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # RETRIEVER — Set up the search mechanism
    retriever = vectorstore.as_retriever(
        search_type="similarity",search_kwargs={"k": TOP_K_RESULTS})

    # LLM — Initialize Google Gemini via API
    llm = ChatGroq(model=LLM_MODEL,temperature=LLM_TEMPERATURE,api_key=os.getenv("GROQ_API_KEY"))

    # PROMPT — Instruction template for the LLM
    prompt = PromptTemplate(template=SYSTEM_PROMPT_TEMPLATE,input_variables=["context", "question"])

    # CHAIN — Combine all components
    chain = RetrievalQA.from_chain_type(
        llm=llm,chain_type="stuff",retriever=retriever,return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return chain, len(chunks)