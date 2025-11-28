# 🤖 AI FAQ Agent

This is an AI-powered FAQ Agent that answers employee questions based on an internal FAQ document using Retrieval-Augmented Generation (RAG).

## 🧩 Features
- Answers questions based only on `faq.txt`
- Uses FAISS vector database for semantic search
- HuggingFace Sentence Transformer embeddings
- LangChain Retrieval-QA chain
- CLI and optional Streamlit UI

---

## 📂 Project Structure

```text
my_agent/
├── app.py
├── agent.py
├── build_db.py
├── faq.txt
├── requirements.txt
├── .env.example
└── README.md
