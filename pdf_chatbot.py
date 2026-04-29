import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import tempfile

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("📄 PDF Chatbot")
st.caption("Upload a PDF and ask questions about it!")

# ── API Key (use .env or Colab Secrets) ──────────────────────────────────────
# In Colab: use userdata.get('GOOGLE_API_KEY')
# In VS Code: use python-dotenv
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found! Add it to your .env file.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ── Session state setup ───────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# ── PDF Upload ────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file and st.session_state.qa_chain is None:
    with st.spinner("Processing PDF... please wait ⏳"):

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Load PDF
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(pages)

        # Create embeddings and vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Setup memory for chat history
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        # Create conversational chain
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=True
        )

    st.success(f"✅ PDF processed! ({len(pages)} pages, {len(chunks)} chunks)")

# ── Chat Interface ────────────────────────────────────────────────────────────
if st.session_state.qa_chain:

    # Display previous chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User input
    user_question = st.chat_input("Ask a question about your PDF...")

    if user_question:
        # Show user message
        with st.chat_message("user"):
            st.write(user_question)

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.qa_chain.invoke({"question": user_question})
                answer = result["answer"]
                source_docs = result.get("source_documents", [])

            st.write(answer)

            # Show source pages
            if source_docs:
                pages_cited = set()
                for doc in source_docs:
                    page_num = doc.metadata.get("page", "?")
                    pages_cited.add(page_num + 1 if isinstance(page_num, int) else page_num)
                st.caption(f"📌 Sources: Page(s) {', '.join(str(p) for p in sorted(pages_cited))}")

        # Save to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

else:
    st.info("👆 Please upload a PDF to get started!")
