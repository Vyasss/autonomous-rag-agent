import os
import shutil
import time
import logging
import platform
import subprocess
import string
from typing import List, Dict, Any

# FastAPI & Pydantic
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

# Environment
from dotenv import load_dotenv

# Vector DB & Embeddings
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain_pinecone import PineconeVectorStore

# Google Gemini & LangChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.document_loaders import PDFPlumberLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# --- 1. CONFIGURATION & LOGGING ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    raise RuntimeError("Missing API Keys in .env")

# --- 2. PINECONE & EMBEDDINGS SETUP ---
# We setup this FIRST so the tools can access it
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "personal-assistant-index"
EMBED_DIM = 384 

existing_indexes = pc.list_indexes().names()
if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
pinecone_index = pc.Index(INDEX_NAME)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
class LocalEmbeddingWrapper:
    def embed_documents(self, texts: List[str]):
        return embed_model.encode(texts, convert_to_numpy=True).tolist()
    def embed_query(self, text: str):
        return embed_model.encode(text, convert_to_numpy=True).tolist()
embeddings = LocalEmbeddingWrapper()

vectorstore = PineconeVectorStore(
    index=pinecone_index,
    embedding=embeddings,
    text_key="text" 
)

# --- 3. DEFINE TOOLS (THE AI'S HANDS) ---

@tool
def search_uploaded_documents(query: str) -> str:
    """
    Searches for information inside the PDF/TXT files the user has UPLOADED to the chat sidebar.
    ALWAYS use this tool FIRST if the user asks for a summary, specific details, or about a person.
    """
    try:
        # Retrieve top 10 relevant chunks
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        docs = retriever.invoke(query)
        
        if not docs:
            return "I searched the uploaded documents but found no relevant information."
        
        # Combine the content into a single string for the LLM to read
        result_text = "Found the following in uploaded documents:\n"
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            result_text += f"--- Source: {source} ---\n{doc.page_content}\n\n"
            
        return result_text
    except Exception as e:
        return f"Error searching uploaded docs: {e}"

@tool
def list_files_in_directory(directory_path: str = ".") -> str:
    """Lists all files in a specific directory path."""
    try:
        if not os.path.exists(directory_path):
            return "Directory does not exist."
        files = os.listdir(directory_path)
        return f"Files in {directory_path}: {', '.join(files)}"
    except Exception as e:
        return f"Error reading directory: {e}"

@tool
def open_file_on_pc(file_path: str) -> str:
    """Opens a file on the computer using the default application."""
    try:
        if platform.system() == 'Windows':
            os.startfile(file_path)
        elif platform.system() == 'Darwin':
            subprocess.call(('open', file_path))
        else:
            subprocess.call(('xdg-open', file_path))
        return f"Successfully opened {file_path}"
    except Exception as e:
        return f"Error opening file: {e}"

@tool
def find_file(filename: str, search_path: List[str] = None) -> str:
    """
    Searches for a file across ALL available drives (C:, D:, etc.).
    Automatically skips system folders.
    """
    results = []
    if search_path is None:
        search_path = [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]

    print(f"🔎 DEBUG: Scanning drives: {search_path}")

    for path in search_path:
        for root, dirs, files in os.walk(path):
            ignore_dirs = {
                'Windows', 'Program Files', 'Program Files (x86)', 
                'System Volume Information', '$Recycle.Bin', 'AppData',
                'node_modules', '.git', 'venv'
            }
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            for file in files:
                if filename.lower() in file.lower():
                    found_path = os.path.join(root, file)
                    results.append(found_path)
                    if len(results) >= 5: break
        if len(results) >= 5: break

    if not results: return "File not found on local drives."
    if len(results) == 1: return f"Found exactly one file: {results[0]}"
    
    output = "I found multiple files. Which one do you want?\n"
    for i, res in enumerate(results):
        output += f"{i+1}. {res}\n"
    return output

@tool
def read_file_content(file_path: str) -> str:
    """Reads the content of a local file (PDF or TXT) found on the computer."""
    try:
        content = ""
        if file_path.lower().endswith(".pdf"):
            loader = PDFPlumberLoader(file_path)
            documents = loader.load()
            for doc in documents: content += doc.page_content + "\n"
        elif file_path.lower().endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f: content = f.read()
        else: return "Error: Only PDF or TXT supported."
        return f"File Content:\n{content[:20000]}"
    except Exception as e: return f"Error reading file: {e}"

# UPDATE: Added 'search_uploaded_documents' to the list!
tools = [search_uploaded_documents, list_files_in_directory, open_file_on_pc, find_file, read_file_content]

# --- 4. LLM & AGENT SETUP ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=GOOGLE_API_KEY,
    temperature=0, 
)

# UPDATED PROMPT: Tells the AI to check uploads first
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are an advanced Autonomous AI Assistant.
        You have two sources of information:
        1. UPLOADED DOCUMENTS (via 'search_uploaded_documents' tool).
        2. LOCAL FILES on the computer (via 'find_file' and 'read_file_content' tools).

        STRATEGY:
        - If the user asks for a summary or info about a specific person/topic, CHECK UPLOADED DOCUMENTS FIRST.
        - If not found there, ask the user if you should search their local computer drives.
        - If the user explicitly asks to "open" or "find" a file on the PC, use the local file tools immediately.
        """),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 5. MEMORY SETUP ---
memory_store: Dict[str, Any] = {}

def get_memory(session_id: str = "default"):
    if session_id not in memory_store:
        memory_store[session_id] = ConversationBufferWindowMemory(
            memory_key="chat_history", return_messages=True, k=5
        )
    return memory_store[session_id]

# --- 6. API ENDPOINTS ---
app = FastAPI(title="Unified AI Agent")

class ChatRequest(BaseModel):
    message: str

def process_file_ingestion(file_path: str, source_name: str):
    try:
        if file_path.endswith(".pdf"):
            loader = PDFPlumberLoader(file_path)
            documents = loader.load()
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path)
            documents = loader.load()
        else: raise ValueError("Unsupported format")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        for chunk in chunks: chunk.metadata["source"] = source_name
        vectorstore.add_documents(chunks)
        return len(chunks)
    except Exception as e: raise e

@app.post("/agent_chat")
def agent_chat(req: ChatRequest):
    try:
        memory = get_memory()
        response = agent_executor.invoke(
            {"input": req.message, "chat_history": memory.chat_memory.messages}
        )
        output = response["output"]
        memory.chat_memory.add_user_message(req.message)
        memory.chat_memory.add_ai_message(output)
        return {"reply": output}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
def upload_document(file: UploadFile = File(...)):
    allowed_types = ["application/pdf", "text/plain"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Invalid file type.")
    temp_filename = f"temp_{int(time.time())}_{file.filename}"
    try:
        with open(temp_filename, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
        num_chunks = process_file_ingestion(temp_filename, file.filename)
        return {"filename": file.filename, "status": "Success", "chunks_added": num_chunks}
    except Exception as e: raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        if os.path.exists(temp_filename): os.remove(temp_filename)

@app.get("/")
def home():
    return {"status": "Unified Agent Ready"}