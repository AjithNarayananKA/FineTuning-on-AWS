import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv
load_dotenv()

API_URL = os.getenv("API_URL") 
print(API_URL)# Replace with your actual API Gateway endpoint
API_KEY = os.getenv("API_KEY")  # Replace with your actual API key if needed

st.set_page_config(page_title="Inference App", layout="wide")
st.title("Inference Application")

prompt = st.text_area("Enter your prompt:", height=200)

if st.button("Generate Response"):
    # 1. Guard: empty / whitespace-only prompt
    if not prompt or not prompt.strip():
        st.warning("Please enter a prompt before generating a response.")
        st.stop()  # do *not* call the API

    # 2. Call API Gateway
    with st.spinner("Calling LLM..."):
        try:
            payload = {"inputs": prompt}

            # Add a timeout so the UI doesn’t hang forever
            response = requests.post(API_URL, json=payload, timeout=60)

            # Try to decode JSON (API Gateway normally returns JSON)
            try:
                raw_json = response.json()
            except Exception:
                raw_json = {"raw_text": response.text}

            # Always show raw JSON for debugging
            with st.expander("RAW JSON FROM API:", expanded=False):
                st.code(json.dumps(raw_json, indent=2, ensure_ascii=False))

            # 3. Handle non-200 HTTP status first
            if response.status_code != 200:
                msg = raw_json.get("message", f"HTTP {response.status_code}")
                st.error(f"Error from API: {msg}")
                st.stop()

            # 4. Lambda proxy success shape: {"statusCode": 200, "body": "...json string..."}
            if "body" in raw_json:
                body_str = raw_json.get("body", "{}")
                try:
                    inner = json.loads(body_str)  # {"result": [ {...}, ... ]}
                except json.JSONDecodeError:
                    st.error("Could not parse Lambda body JSON.")
                    st.code(body_str)
                    st.stop()
            else:
                # Fallback: Lambda already returned the inner JSON
                inner = raw_json

            # 5. Extract `result` → may be list, dict, or something else
            result = inner.get("result", [])

            if isinstance(result, list) and result:
                generated_texts = []
                for item in result:
                    if isinstance(item, dict):
                        generated_texts.append(item.get("generated_text", str(item)))
                    else:
                        generated_texts.append(str(item))
                generated = "\n\n".join(generated_texts)
            elif isinstance(result, dict):
                generated = result.get("generated_text", str(result))
            else:
                generated = str(result)

            # 6. Show nicely in the UI
            st.success("Response Received")
            st.subheader("LLM Answer")
            st.markdown(generated)

        except requests.Timeout:
            st.error("Request to API Gateway timed out. Try again or shorten the prompt.")
        except Exception as e:
            st.error("Something went wrong while calling the API.")
            st.exception(e)