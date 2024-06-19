import streamlit as st
import requests
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

# Load environment variables from .env file
load_dotenv()

# Access the API token from environment variable
hf_token = os.getenv('HUGGINGFACE_TOKEN')    

# Constants
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

# Function to query the Mistral API
def query_mistral_api(prompt):
    headers = {
        "Authorization": f"Bearer {hf_token}"
    }
    payload = {
        "inputs": prompt
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    response_json = response.json()

    if response_json and 'generated_text' in response_json[0]:
        generated_text = response_json[0]['generated_text']
        
        # Find and return only the part after "Response:"
        response_start = generated_text.find("Response:")
        if response_start != -1:
            return generated_text[response_start + len("Response:"):].strip()
        else:
            return "No 'Response:' found in the generated text."
    else:
        return "Unexpected response structure."

# Streamlit application
st.title("Mistral API Query Interface")
st.write("Enter a prompt to get a response from the Mistral model hosted on Hugging Face.")

prompt = st.text_area("Prompt:", "")


template = """[INST] give only concise answers

Riddle: {riddle_space}
[/INST]"""

if st.button("Submit"):
    if prompt:
        var1 = template.format(riddle_space = prompt)
        with st.spinner("Querying the Mistral API..."):
            result = query_mistral_api(var1)
            st.write("Response from Mistral:")
            st.write(result)
    else:
        st.warning("Please enter a prompt to query the Mistral API.")

