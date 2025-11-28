import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

INDEX_DIR = "faiss_index"
FAQ_FILE = "faq.txt"


def parse_faq_file():
    """Parse faq.txt into (question, answer) pairs.

    Assumes format like:
    Q1?
    A1...

    Q2?
    A2...

    Blank lines are allowed but not required.
    """
    if not os.path.exists(FAQ_FILE):
        raise FileNotFoundError(f"{FAQ_FILE} not found in project folder")

    with open(FAQ_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    lines = [ln for ln in lines if ln]

    qa_pairs = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].strip()
        if line.endswith("?"):
            question = line
            i += 1
            answer_lines = []
            while i < n and not lines[i].strip().endswith("?"):
                answer_lines.append(lines[i].strip())
                i += 1
            answer = " ".join(answer_lines).strip()
            if answer:
                qa_pairs.append((question, answer))
        else:
            i += 1

    if not qa_pairs:
        raise ValueError("No valid Q&A pairs found in faq.txt")

    return qa_pairs


def _build_vectorstore_from_faq() -> FAISS:
    """Build FAISS index from faq.txt using per Q&A pair."""
    qa_pairs = parse_faq_file()

    texts = []
    metadatas = []

    for q, a in qa_pairs:
        texts.append(a)
        metadatas.append({"question": q})

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
    Retrieve the most relevant single ANSWER from faq.txt.
    No OpenAI / no API key needed.
    """
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Modern LangChain retriever API: use invoke()
    docs = retriever.invoke(question)

    if not docs:
        return "Sorry, I couldn't find an answer for that in the FAQ."

    best = docs[0]
    answer = best.page_content.strip()
    # If you want, you can also include question:
    # q = best.metadata.get("question", "")
    # return f"{q}\n{answer}"

    return answer

