import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load environment variables (still useful later if needed)
load_dotenv()

FAQ_FILE = "faq.txt"

if not os.path.exists(FAQ_FILE):
    raise FileNotFoundError(f"{FAQ_FILE} not found in project folder")

# 2. Read FAQ text
with open(FAQ_FILE, "r", encoding="utf-8") as f:
    faq_text = f.read()

# 3. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", "?", "!"],
)

docs = text_splitter.create_documents([faq_text])

# 4. Use FREE HuggingFace embeddings (no OpenAI, no API key)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 5. Build FAISS vector store
vectorstore = FAISS.from_documents(docs, embeddings)

# 6. Save to disk
INDEX_DIR = "faiss_index"
os.makedirs(INDEX_DIR, exist_ok=True)
vectorstore.save_local(INDEX_DIR)

print(f"✅ Vector database built and saved in folder: {INDEX_DIR}")
