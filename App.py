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


template = """[INST] You are a science prodigy currently competing in a National Science competition. You are now in the fifth round, where you must first reason through the clues of the given riddle and then provide a short answer. Remember, your answer should consist of just the term the riddle is pointing to, and nothing else. Adding additional text will result in point deductions.
Here's an example to guide you:
Riddle: You might think I am a rather unstable character because I never stay at one place. However, my motion obeys strict rules and I always return to where I started and even if I have to leave that spot again I do it in strict accordance with time. I can be named in electrical and mechanical contexts; in all cases I obey the same mathematical rules. In order to fully analyze me, you would think about a stiffness or force constant, restoring force, and angular frequency.
Answer: oscillator

Read the riddle below and provide the three possible correct answers as a json with keys: answer1, answer2, answer3

NOTE: You are allowed to include an answer multiple times if your reasoning shows that it is likely the correct answer. Do not provide any explanations.

Riddle: {riddle_space}
[/INST]"""

if st.button("Submit"):
    if prompt:
        prompt = template.format(riddle_space = prompt)
        with st.spinner("Querying the Mistral API..."):
            result = query_mistral_api(prompt)
            st.write("Response from Mistral:")
            st.write(result)
    else:
        st.warning("Please enter a prompt to query the Mistral API.")

