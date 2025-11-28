import os

# Try to import streamlit (for cloud deployment); ignore if not available
try:
    import streamlit as st
except ImportError:
    st = None

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _load_api_key_and_model():
    """
    Load API key & model name.

    1) Try Streamlit secrets (for Streamlit Cloud).
    2) If that fails for ANY reason, fall back to .env (local run).
    """
    # Try Streamlit secrets (for Streamlit Cloud)
    if st is not None:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]  # may raise if not configured
            model = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
            return api_key, model
        except Exception:
            # No secrets configured or running locally without secrets.toml
            pass

    # Fall back to .env (local dev)
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return api_key, model


OPENAI_API_KEY, OPENAI_MODEL = _load_api_key_and_model()

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found. "
        "Set it in .env for local run OR in Streamlit Secrets on cloud."
    )

INDEX_DIR = "faiss_index"
FAQ_FILE = "faq.txt"


def _build_vectorstore_from_faq() -> FAISS:
    """Build FAISS index from faq.txt if it doesn't exist."""
    if not os.path.exists(FAQ_FILE):
        raise FileNotFoundError(
            f"{FAQ_FILE} not found. Make sure the file is in the project root."
        )

    with open(FAQ_FILE, "r", encoding="utf-8") as f:
        faq_text = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "?", "!"],
    )

    docs = text_splitter.create_documents([faq_text])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

    # Save index to disk
    os.makedirs(INDEX_DIR, exist_ok=True)
    vectorstore.save_local(INDEX_DIR)

    return vectorstore


def load_vectorstore() -> FAISS:
    """
    Load the FAISS index from disk.
    If it doesn't exist, automatically build it from faq.txt.
    """
    if not os.path.exists(INDEX_DIR):
        # First run: build index from FAQ
        return _build_vectorstore_from_faq()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


def get_qa_chain() -> RetrievalQA:
    """Create the RetrievalQA chain using the vectorstore and LLM."""
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL,
        temperature=0.2,
    )

    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )
    return qa_chain


def answer_question(question: str) -> str:
    """Given a question string, return the model's answer."""
    qa_chain = get_qa_chain()
    result = qa_chain({"query": question})
    return result["result"]



