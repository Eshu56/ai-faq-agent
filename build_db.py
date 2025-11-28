import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

FAQ_FILE = "faq.txt"
INDEX_DIR = "faiss_index"


def main():
    if not os.path.exists(FAQ_FILE):
        raise FileNotFoundError(f"{FAQ_FILE} not found in project folder")

    # 1. Read raw FAQ text
    with open(FAQ_FILE, "r", encoding="utf-8") as f:
        faq_text = f.read()

    if not faq_text.strip():
        raise ValueError("faq.txt is empty. Please add some Q&A content.")

    # 2. Split into Q&A blocks separated by blank lines
    blocks = faq_text.strip().split("\n\n")

    texts = []       # what will be embedded
    metadatas = []   # store question separately if needed

    for block in blocks:
        # remove extra empty lines
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue  # skip incomplete pairs

        question = lines[0]
        answer = " ".join(lines[1:])

        # We embed only the ANSWER text (clean & short)
        texts.append(answer)
        metadatas.append({"question": question})

    if not texts:
        raise ValueError("No valid Q&A pairs found in faq.txt")

    print(f"✅ Found {len(texts)} Q&A pairs to index.")

    # 3. Embeddings (FREE – no OpenAI)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Build FAISS index from texts
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

    # 5. Save FAISS index
    os.makedirs(INDEX_DIR, exist_ok=True)
    vectorstore.save_local(INDEX_DIR)

    print(f"🎉 Vector database built and saved in folder: {INDEX_DIR}")


if __name__ == "__main__":
    main()
