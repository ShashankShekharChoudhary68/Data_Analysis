from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
import os
import streamlit as st
import logging

logging.basicConfig(level=logging.DEBUG)


def load_and_process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(pages)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(texts, embeddings)


def query_syllabus(db, query):
    try:
        retriever = db.as_retriever(search_kwargs={"k": 3})
        qa = RetrievalQA.from_chain_type(
            llm=Ollama(model="mistral", num_ctx=2048, temperature=0.3, timeout=120),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        result = qa({"query": query})
        return result["result"], [doc.metadata["page"]+1 for doc in result["source_documents"]]
    except Exception as e:
        return f"Error: {str(e)}", []


# Streamlit UI
def main():
    st.title("📚 Course Syllabus Q&A Bot")

    uploaded_file = st.file_uploader("Upload Syllabus PDF", type="pdf")

    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        db = load_and_process_pdf("temp.pdf")

        query = st.text_input("Ask a question (e.g., 'What topics are covered in Week 3?')")

        if query:
            answer, pages = query_syllabus(db, query)
            st.write("**Answer:**", answer)
            st.write("**Relevant Pages:**", ", ".join(map(str, pages)))


if __name__ == "__main__":
    main()