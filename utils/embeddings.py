import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pdf_loader import load_and_split_pdf  # ✅ Fixed import

FAISS_DIR = os.path.join("utils", "faiss_index")

def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if not os.path.exists(os.path.join(FAISS_DIR, "index.faiss")):
        print("🆕 FAISS index not found. Creating a new one...")
        documents = load_and_split_pdf(r"F:\Rag_chat_bot\data\Human-Resources-Policy-Manual-RHA-Updated-February2022.pdf")
        vectordb = FAISS.from_documents(documents, embedding_model)
        vectordb.save_local(FAISS_DIR)
        print("✅ FAISS index created and saved.")
    else:
        print("📦 Loading FAISS vectorstore from disk...")
        vectordb = FAISS.load_local(
            FAISS_DIR,
            embedding_model,
            allow_dangerous_deserialization=True
        )

    return vectordb

if __name__ == "__main__":
    load_vectorstore()
