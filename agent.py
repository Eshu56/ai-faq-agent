import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

    os.makedirs(INDEX_DIR, exist_ok=True)
    vectorstore.save_local(INDEX_DIR)

    return vectorstore


def load_vectorstore() -> FAISS:
    """Load FAISS index or build it from faq.txt if missing."""
    if not os.path.exists(INDEX_DIR):
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


def answer_question(question: str) -> str:
    """Retrieve top FAQ answer (no OpenAI / no API key needed)."""
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(question)
    if not docs:
        return "Sorry, I couldn't find an answer for that in the FAQ."
    return docs[0].page_content.strip()


