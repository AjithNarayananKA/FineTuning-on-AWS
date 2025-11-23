# ui.py
import streamlit as st
from rag_app_backend import generate_answer

st.set_page_config(page_title="RAG on Fine-tuned Model", layout="centered")

st.title("RAG System using Fine-Tuned TinyLlama")

question = st.text_input("Enter your question:")

if st.button("Generate Answer"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Running RAG pipeline..."):
            response = generate_answer(question)

        st.subheader("Retrieved Context")
        st.info(response["context"])

        st.subheader("LLM Answer")
        st.success(response)