import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

INDEX_DIR = "faiss_index"
FAQ_FILE = "faq.txt"


def _build_vectorstore_from_faq() -> FAISS:
    """Build FAISS index from faq.txt using per-Q&A pairs."""
    if not os.path.exists(FAQ_FILE):
        raise FileNotFoundError(f"{FAQ_FILE} not found in project folder")

    with open(FAQ_FILE, "r", encoding="utf-8") as f:
        faq_text = f.read()

    if not faq_text.strip():
        raise ValueError("faq.txt is empty. Please add some Q&A content.")

    blocks = faq_text.strip().split("\n\n")

    texts = []
    metadatas = []

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue

        question = lines[0]
        answer = " ".join(lines[1:])

        texts.append(answer)
        metadatas.append({"question": question})

    if not texts:
        raise ValueError("No valid Q&A pairs found in faq.txt")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

    os.makedirs(INDEX_DIR, exist_ok=True)
    vectorstore.save_local(INDEX_DIR)

    return vectorstore


def load_vectorstore() -> FAISS:
    """Load FAISS index or build it if missing."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if not os.path.exists(INDEX_DIR):
        return _build_vectorstore_from_faq()

    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


def answer_question(question: str) -> str:
    """
    Retrieve the most relevant ANSWER from faq.txt for the given question.
    No OpenAI / no API key.
    """
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # New LangChain API: use invoke() instead of get_relevant_documents()
    docs = retriever.invoke(question)

    if not docs:
        return "Sorry, I couldn't find an answer for that in the FAQ."

    best = docs[0]
    answer = best.page_content.strip()
    # If you ever want to see the question too:
    # question_text = best.metadata.get("question", "")
    # return f"{question_text}\n{answer}"

    return answer


