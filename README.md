# -HR-Policy-Assistant-Using-Rag

Here’s a short and concise version of what you can add to your `README.md`:

---

# RAG Chatbot for HR Policy

This project is a Retrieval-Augmented Generation (RAG) chatbot that answers HR-related questions using content from an uploaded HR policy PDF.

## Features

* Loads and splits PDF documents
* Generates and saves vector embeddings using FAISS
* Uses `google/flan-t5-base` for local, open-access inference
* Answers user queries using document context

## How to Run

1. Clone the repo and place your PDF in the `data` folder
2. Install requirements: `pip install -r requirements.txt`
3. Run `embeddings.py` to generate vectorstore
4. Run `rag_chatbot.py` to start the QA pipeline

## Folder Structure

* `utils/embeddings.py` – Loads PDF, generates and saves FAISS index
* `utils/pdf_loader.py` – Loads and splits PDF
* `rag_chatbot.py` – Main QA chain

## Model Used

* `google/flan-t5-base` via HuggingFace

---

Let me know if you also want the `requirements.txt` content.
