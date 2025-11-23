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
        
        # Required keys check
        if not isinstance(response, dict):
            st.error("Backend returned invalid response type.")
            st.code(str(response))
            st.stop()

        # Required keys validation
        if "context" not in response or "answer" not in response:
            st.error("Backend missing keys: expected 'context' and 'answer'")
            st.code(str(response))
            st.stop()

        # Display context
        st.subheader("Retrieved Context")
        st.info(response["context"])

        # Display LLM Answer
        st.subheader("LLM Answer")
        st.success(response["answer"])



# def call_finetuned_llm(prompt: str):
#     payload = {"inputs": prompt}
#     resp = requests.post(API_URL, json=payload)

#     try:
#         data = resp.json()
#         return data.get("result", "No response found.")
#     except Exception as e:
#         return f"Error decoding LLM response: {str(e)}"

# def call_finetuned_llm(prompt: str):
#     payload = {"inputs": prompt}
#     resp = requests.post(API_URL, json=payload)

#     try:
#         data = resp.json()

#         result = data
#         # SageMaker returns: {"result": [ { "generated_text": "..."} ]}
#         #result = data.get("result", [])

#         if isinstance(result, list) and len(result) > 0:
#             first = result[0]
#             if isinstance(first, dict):
#                 return first.get("generated_text", "")
#             return str(first)

#         # fallback
#         return data
#         #return str(result)

#     except Exception as e:
#         return f"Error decoding LLM response: {str(e)}"