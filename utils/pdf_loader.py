from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    return chunks



if __name__ == "__main__":
    path = r"F:\Rag_chat_bot\data\Human-Resources-Policy-Manual-RHA-Updated-February2022.pdf"  # Update this path to your actual PDF
    docs = load_and_split_pdf(path)
    print(f"Loaded and split {len(docs)} chunks from PDF.")
