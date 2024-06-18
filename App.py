

from dataclasses import dataclass
import streamlit as st
import time
import requests
import re

# Replace <random-id> with your actual ngrok ID
base_url = "https://3aac-34-125-236-111.ngrok-free.app"


@dataclass
class Message:
    actor: str
    payload: str

# Constants for actors and session state keys
USER = "user"
ASSISTANT = "ai"
MESSAGES = "messages"

# Initialize session state
def initialize_session_state():
    if MESSAGES not in st.session_state:
        st.session_state[MESSAGES] = [Message(actor=ASSISTANT, payload="Hi! How can I help you?")]

# Generate a response from the assistant
def generate_response(user_input: str) -> str:
    try:
        response = requests.get(f"{base_url}/gen_llm/{user_input}")
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

# Display message function
def display_message(message: str, message_type: str):
    placeholder = st.empty()
    if message_type == "user-message":
        message_content = f"""
            <div class="user-message">
                {message}
            </div>
        """
    placeholder.markdown(message_content, unsafe_allow_html=True)
    time.sleep(1)

# Page functions
def chat_page():
    # Initialize the session state
    initialize_session_state()

    # Custom CSS for user messages
    st.markdown("""
        <style>
        .user-message {
            width: fit-content;
            background-color: #c8e6c9;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            max-width: 70%;
            text-align: right;
            margin-left: auto;
            align-self: flex-end;
        }
        </style>
        """, unsafe_allow_html=True)

    # Display previous messages
    for msg in st.session_state[MESSAGES]:
        if msg.actor == ASSISTANT:
            with st.chat_message(name=msg.actor, avatar="bot_image.png"):
                st.write(msg.payload)
        else:
            display_message(msg.payload, "user-message")

    # Get the user input
    prompt: str = st.chat_input("Begin your prompt with ""/"" if you want to retrive NSMQ questions")

    # If the user has entered a prompt, process it
    if prompt:
        # Add the user's message to the session state
        st.session_state[MESSAGES].append(Message(actor=USER, payload=prompt))
        
        display_message(prompt, "user-message")
        
        # Generate and display the assistant's response
        response = generate_response(prompt)
        st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=response))
        
        with st.chat_message(name=ASSISTANT, avatar="bot_image.png"):
            st.write(response)




# Sidebar with navigation
st.sidebar.title("Navigation")

chat_page()

