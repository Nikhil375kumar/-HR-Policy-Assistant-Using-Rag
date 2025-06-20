import streamlit as st
from rag_chatbot import build_qa_chain

# Set Streamlit page configuration
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")

# Title and intro
st.title("🧠 HR Policy Assistant")
st.markdown("Ask questions about the HR policy PDF.")

# Load QA chain (cached to avoid reloading every time)
@st.cache_resource
def load_qa():
    return build_qa_chain()

qa = load_qa()

# Input box for user question
user_question = st.text_input("❓ Enter your question:")

# If user submits question
if st.button("Get Answer") and user_question.strip():
    with st.spinner("Thinking..."):
        answer = qa.run(user_question)
        st.success("📝 Answer:")
        st.write(answer)
