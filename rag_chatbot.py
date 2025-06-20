import sys
import os

# Add 'utils' folder to the path for custom module imports
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from embeddings import load_vectorstore  # from utils/embeddings.py

def build_qa_chain():
    print("🔍 Loading vectorstore...")
    vectordb = load_vectorstore()
    retriever = vectordb.as_retriever()

    prompt_template = PromptTemplate(
        template=(
            "You are an HR assistant. Answer the question using ONLY the HR policy document provided.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n"
            "Answer:"
        ),
        input_variables=["context", "question"]
    )

    print("🤖 Loading Hugging Face model (flan-t5-base)...")
    llm = HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-base",
        task="text2text-generation",
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )

    print("🔗 Building RetrievalQA chain...")
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template}
    )

if __name__ == "__main__":
    print("🚀 Starting QA chain build...")
    try:
        qa = build_qa_chain()
        print("✅ QA chain is ready. You can now call `qa.run('Your question')` to ask questions.")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
