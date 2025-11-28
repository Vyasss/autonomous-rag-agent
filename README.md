# Autonomous RAG Agent with Local OS Integration

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-Agent-orange.svg)
![Gemini](https://img.shields.io/badge/Model-Gemini%202.5%20Flash-purple.svg)

## 📋 Project Overview!!

This is not just a chatbot; it is an **Autonomous Agent**.

This application bridges the gap between **Static Knowledge Retrieval (RAG)** and **Dynamic Operating System Control**. It uses an event-driven agentic architecture to intelligently decide whether to answer a user's question using uploaded documents (Vector DB) or by autonomously searching, reading, and opening files on the user's local machine (C:/D: drives).

Built with **FastAPI** for the backend and **Streamlit** for the frontend, it leverages **Google's Gemini 2.5 Flash** for high-speed reasoning and **Pinecone** for semantic memory.

## ✨ Key Features

* **🧠 Hybrid Information Retrieval:**
    * **RAG Pipeline:** Ingests PDF/TXT files, chunks them using Recursive Character Splitting, and retrieves context via Pinecone.
    * **Local OS Access:** Can autonomously scan hard drives (C:, D:) to find files based on fuzzy string matching.
* **⚡ Latency-Optimized Search:** Implemented a smart directory walker that dynamically filters out system heavyweights (e.g., `node_modules`, `Windows`, `.git`), reducing search times from minutes to seconds.
* **🛠️ Tool-Calling Agent:** The LLM is not hard-coded; it autonomously selects tools (`open_file`, `read_file`, `search_uploaded_docs`) based on user intent.
* **💸 Cost-Efficient Architecture:** Uses **Local Embeddings** (`SentenceTransformer/all-MiniLM-L6-v2`) to handle vectorization locally on the CPU, eliminating embedding API costs.
* **💾 Conversational Memory:** Maintains a sliding window of conversation history (`k=5`) to support context-aware follow-up questions.

## 🏗️ Tech Stack

* **Orchestration:** LangChain (Agents & Toolkits)
* **LLM:** Google Gemini 2.5 Flash (via Google AI Studio)
* **Vector Database:** Pinecone (Serverless)
* **Backend:** FastAPI (Async/Sync hybrid handling)
* **Frontend:** Streamlit
* **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (Local)
* **File Handling:** `pdfplumber`, `subprocess`, `os` module

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/autonomous-rag-agent.git](https://github.com/yourusername/autonomous-rag-agent.git)
cd autonomous-rag-agent

2. Create Virtual Environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

4. Configure API Keys
GOOGLE_API_KEY=your_gemini_key_here
PINECONE_API_KEY=your_pinecone_key_here

🏃‍♂️ Usage
This application follows a Microservices pattern with separate Backend and Frontend services.

Step 1: Start the Backend (FastAPI)
This handles the Agent logic, File I/O, and LLM reasoning.

Bash

uvicorn main:app --reload
Server will start at http://127.0.0.1:8000

Step 2: Start the Frontend (Streamlit)
This provides the chat interface and file upload capability.

Bash

streamlit run frontend.py
UI will open at http://localhost:8501

🤖 Capabilities & Commands
Once running, you can interact with the Agent using natural language:

📄 RAG (Uploaded Documents)
Action: Upload a PDF in the sidebar.

Prompt: "Give me a summary of the uploaded resume."

Prompt: "What skills does the candidate have?"

🖥️ Local OS Control
Find Files: "Find the file 'project_notes' on my D drive." (Scans drives automatically).

Read Local Files: "Find 'requirements.txt' and tell me what libraries are listed."

Open Files: "Open 'frontend.py' for me." (Actually opens the app on your PC).

📂 Project Structure
├── main.py              # FastAPI Backend & Agent Logic
├── frontend.py          # Streamlit User Interface
├── requirements.txt     # Python Dependencies
├── .env                 # API Credentials (Excluded from Git)
└── README.md            # Documentation
🛡️ Architecture Diagram
Code snippet

graph TD
    User[User] -->|Chat/Commands| Frontend[Streamlit UI]
    Frontend -->|HTTP Requests| Backend[FastAPI Server]
    Backend -->|Invokes| Agent[LangChain Agent]
    
    Agent -->|Decision| Tool1[Search Uploaded Docs]
    Agent -->|Decision| Tool2[Find Local File]
    Agent -->|Decision| Tool3[Open/Read File]
    
    Tool1 -->|Query| Pinecone[Pinecone Vector DB]
    Tool2 -->|Scan| OS[Local File System C:/D:]
    
    Agent -->|Reasoning| Gemini[Gemini 2.5 Flash]
🔮 Future Improvements
Dockerization: Containerizing the backend for cloud deployment.

Voice Interface: Adding Speech-to-Text (STT) for voice commands.

Gmail Integration: Adding OAuth2 tools to send/read emails autonomously.

Created by [Chetan Vyas]


Since I wrote `pip install -r requirements.txt` in the README, you need to create that file so others (and recruiters) know what libraries to install.

Create a file named `requirements.txt` and paste this inside:

```text
fastapi
uvicorn
python-dotenv
pinecone-client
langchain-pinecone
sentence-transformers
langchain-google-genai
langchain-community
langchain-core
langchain
pypdf
pdfplumber
streamlit
requests
