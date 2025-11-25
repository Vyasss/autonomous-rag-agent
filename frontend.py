import streamlit as st
import requests

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8000"  # Address of your running FastAPI backend

st.set_page_config(
    page_title="Personal AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# --- HEADER ---
st.title("🤖 Chetan Personal AI Assistant")


# --- SIDEBAR: File Upload ---
with st.sidebar:
    st.header("📂 Document Ingestion")
    uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Uploading and Chunking..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    response = requests.post(f"{API_URL}/upload", files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"✅ Processed {data['chunks_added']} chunks from '{data['filename']}'!")
                    else:
                        st.error(f"❌ Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

    st.divider()
    st.markdown("### 🔧 System Status")
    try:
        if requests.get(f"{API_URL}/").status_code == 200:
            st.write("🟢 Backend: Online")
        else:
            st.write("🔴 Backend: Offline")
    except:
        st.write("🔴 Backend: Offline")

# --- MAIN CHAT INTERFACE ---

# 1. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.write(f"- {source}")

# 3. Handle User Input
if prompt := st.chat_input("Ask a question or tell me to open a file..."):
    # A. Display User Message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # B. Get AI Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            # --- CHANGE 1: Endpoint is now /agent_chat ---
            # --- CHANGE 2: Input key is now "message" ---
            payload = {"message": prompt}
            response = requests.post(f"{API_URL}/agent_chat", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                
                # --- CHANGE 3: Response key is now "reply" ---
                answer = data.get("reply", "No answer found.")
                
                # Note: Agents don't always return structured sources like RAG.
                # We leave this empty for now to prevent errors.
                sources = [] 
                
                # Update UI
                message_placeholder.markdown(answer)
                
                # Save to History
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer
                })
            else:
                message_placeholder.markdown(f"❌ Error: {response.text}")
                
        except Exception as e:
            message_placeholder.markdown(f"❌ Connection Error. Is the backend running? \n\nDetails: {e}")