# 📄 PDF Chatbot using LangChain + Google Gemini

A RAG (Retrieval-Augmented Generation) based chatbot that answers questions from any PDF document.

## 🚀 Features
- Upload any PDF and ask questions
- Remembers previous questions (chat history)
- Shows source page numbers for every answer

## 🛠️ Tech Stack
- Python
- LangChain
- Google Gemini API (gemini-2.5-flash)
- FAISS Vector Store
- Google Colab

## 📦 Installation
```bash
pip install langchain langchain-google-genai langchain-community langchain-text-splitters pypdf faiss-cpu google-generativeai
```

## ▶️ How to Run
1. Open `pdf_chatbot.ipynb` in Google Colab
2. Add your Gemini API key in Colab Secrets as `GOOGLE_API_KEY`
3. Run all cells in order
4. Upload your PDF and start chatting!
