# 📄 PDF Chatbot using RAG (LangChain + Groq + Streamlit)

A RAG (Retrieval-Augmented Generation) based chatbot that answers questions from any PDF document — deployed live on Streamlit Cloud!

🚀 **Live Demo:** https://pdf-chatbot-n5rwiqquk4z7uaprpmcplo.streamlit.app/

## ✨ Features
- Upload any PDF and ask questions instantly
- Remembers previous questions (chat history)
- Shows source page numbers for every answer
- Clean web UI — no setup needed for users

## 🛠️ Tech Stack
- **Python**
- **LangChain** — document loading and text splitting
- **Groq API** (Llama 3.3 70B) — LLM for answering questions
- **HuggingFace Embeddings** (all-MiniLM-L6-v2) — vector embeddings
- **FAISS** — vector store for similarity search
- **Streamlit** — web interface and deployment

## 🧠 How It Works (RAG Pipeline)
1. PDF is loaded and split into chunks
2. Chunks are converted to vector embeddings using HuggingFace
3. Embeddings are stored in FAISS vector store
4. User question is matched to relevant chunks via similarity search
5. Relevant chunks + question are sent to Groq LLM for answering

## 📦 Installation (Local)
```bash
pip install streamlit langchain-community langchain-text-splitters faiss-cpu pypdf python-dotenv sentence-transformers groq
```

## ▶️ How to Run Locally
1. Clone this repository
2. Create a `.env` file and add your Groq API key:
```
GROQ_API_KEY=your_key_here
```
3. Run the app:
```bash
streamlit run pdf_chatbot.py
```

## 📁 Project Structure
```
pdf-chatbot/
├── pdf_chatbot.py       # Streamlit app (deployment)
├── pdf_chatbot.ipynb    # Jupyter notebook (experimentation)
├── requirements.txt     # Dependencies
└── README.md
```
