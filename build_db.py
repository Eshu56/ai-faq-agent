import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

FAQ_FILE = "faq.txt"
INDEX_DIR = "faiss_index"


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

    # Remove completely empty lines
    lines = [ln for ln in lines if ln]

    qa_pairs = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].strip()
        # Treat any line ending with '?' as a question
        if line.endswith("?"):
            question = line
            i += 1
            answer_lines = []
            # Collect all following lines until next question or end of file
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


def main():
    print(f"📄 Parsing FAQ from {FAQ_FILE} ...")
    qa_pairs = parse_faq_file()
    print(f"✅ Found {len(qa_pairs)} Q&A pairs.")

    # Build texts and metadata
    texts = []
    metadatas = []

    for q, a in qa_pairs:
        texts.append(a)  # we embed the ANSWER text
        metadatas.append({"question": q})

    print("🧠 Creating embeddings (HuggingFace sentence-transformers)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("🧱 Building FAISS index...")
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

    os.makedirs(INDEX_DIR, exist_ok=True)
    vectorstore.save_local(INDEX_DIR)

    print(f"🎉 Done! Vector database saved in folder: {INDEX_DIR}")


if __name__ == "__main__":
    main()

