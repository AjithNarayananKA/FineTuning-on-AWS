import streamlit as st
import requests
import json
import os
API_URL = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY")

st.set_page_config(page_title="Inference App", layout="wide")
st.title("Inference Application")

prompt=st.text_area("Enter your prompt:", height=200)

if st.button("Generate Response"):
    if not prompt.strip():
        st.warning("Please enter a prompt")
    else:
        with st.spinner("Calling LLM..."):
            try:
                payload = {"inputs": prompt}
                
                headers = {
                "x-api-key": API_KEY,
                "Content-Type": "application/json"
               }

                response = requests.post(API_URL, json=payload, headers=headers)

                if response.status_code == 200:
                    data = response.json()
                    st.success("Response Received")
                    st.write("### Model Output:")
                    st.info(data["result"])
                else:
                    st.error(f"Error: {response.status_code}")
                    st.write(response.text)

            except Exception as e:
                st.error("Something went wrong!")
                st.exception(e)