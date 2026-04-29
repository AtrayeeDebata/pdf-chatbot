import os
import streamlit as st
import tempfile
from dotenv import load_dotenv

import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("📄 PDF Chatbot")
st.caption("Upload a PDF and ask questions about it!")

# ── API Key ───────────────────────────────────────────────────────────────────
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found! Add it to Streamlit Secrets.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# ── Session state ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ── PDF Upload ────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file and st.session_state.vectorstore is None:
    with st.spinner("Processing PDF... please wait ⏳"):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(pages)

        # Free HuggingFace embeddings - no API key needed!
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)

    st.success(f"✅ PDF processed! ({len(pages)} pages, {len(chunks)} chunks)")

# ── Chat Interface ────────────────────────────────────────────────────────────
if st.session_state.vectorstore:

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_question = st.chat_input("Ask a question about your PDF...")

    if user_question:
        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                docs = st.session_state.vectorstore.similarity_search(user_question, k=3)
                context = "\n\n".join([doc.page_content for doc in docs])

                history_text = ""
                for msg in st.session_state.chat_history[-4:]:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    history_text += f"{role}: {msg['content']}\n"

                model = genai.GenerativeModel("gemini-2.0-flash")
                prompt = f"""You are a helpful assistant. Answer the question based on the context below.

Context from PDF:
{context}

Previous conversation:
{history_text}

Question: {user_question}

Answer:"""
                response = model.generate_content(prompt)
                answer = response.text

            st.write(answer)

            pages_cited = set()
            for doc in docs:
                page_num = doc.metadata.get("page", "?")
                pages_cited.add(page_num + 1 if isinstance(page_num, int) else page_num)
            st.caption(f"📌 Sources: Page(s) {', '.join(str(p) for p in sorted(pages_cited))}")

        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

else:
    st.info("👆 Please upload a PDF to get started!")
