import streamlit as st
import requests

# Constants
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

# Function to query the Mistral API
def query_mistral_api(prompt):
    headers = {
        "Authorization": f"Bearer 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"
    }
    payload = {
        "inputs": prompt
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Streamlit application
st.title("Mistral API Query Interface")
st.write("Enter a prompt to get a response from the Mistral model hosted on Hugging Face.")

prompt = st.text_area("Prompt:", "")

if st.button("Submit"):
    if prompt:
        with st.spinner("Querying the Mistral API..."):
            result = query_mistral_api(prompt)
            st.write("Response from Mistral:")
            st.write(result)
    else:
        st.warning("Please enter a prompt to query the Mistral API.")

