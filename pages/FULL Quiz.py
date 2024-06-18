

from dataclasses import dataclass
import streamlit as st
import time

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
        st.session_state[MESSAGES] = [Message(actor=ASSISTANT, payload="")]
    if 'hid_text' not in st.session_state:
        st.session_state['hid_text'] = True

# Generate a response from the assistant
def generate_response(user_input: str) -> str:
    # Here you can implement the logic for generating a response from the assistant.
    # For simplicity, we return a fixed response.
    return "I'm here to assist you with any questions you have."

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

# Display initial text if hid_text is False
def display_initial_text():
    if st.session_state['hid_text']:
        st.title("👩‍🔬 Welcome Back! 👨‍🔬")
        st.write("### 🌟 Description 🌟")
        st.write("🧪 You will be asked **a series of questions**.")
        st.write("🔄 At least one from each round of the NSMQ**.")
        st.write("✍️ Provide **concise answers** which will be evaluated at the end.")
        st.write("📈 Remember, every attempt **contributes towards your general progress** on this app.")

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

    # Display initial text if applicable
    display_initial_text()

    # Display previous messages
    for msg in st.session_state[MESSAGES]:
        if msg.actor == ASSISTANT:
            with st.chat_message(name=msg.actor, avatar="bot_image.png"):
                st.write(msg.payload)
        else:
            display_message(msg.payload, "user-message")

    # Get the user input
    import streamlit as st

    # Create two columns
    col1, col2 = st.columns([3, 1])

    # Add text input to the first column
    with col1:
        prompt = st.text_input("Enter a prompt here")

    # Add button to the second column
    with col2:
        button = st.button("Submit")
    prompt: str = st.chat_input("Enter a prompt here")

    # If the user has entered a prompt, process it
    if prompt:
        st.session_state['hid_text'] = False
        # Add the user's message to the session state
        st.session_state[MESSAGES].append(Message(actor=USER, payload=prompt))
        
        display_message(prompt, "user-message")
        
        # Generate and display the assistant's response
        response = generate_response(prompt)
        st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=response))
        
        with st.chat_message(name=ASSISTANT, avatar="bot_image.png"):
            st.write(response)


def circular_countdown(target_time):
    """
    Creates a circular countdown timer in Streamlit.

    Args:
        target_time: The target time in seconds for the countdown.
    """
    start_time = time.time()
    remaining_time = target_time

    countdown_placeholder = st.empty()
    
    while remaining_time > 0:
        # Update remaining time
        elapsed_time = time.time() - start_time
        remaining_time = target_time - elapsed_time

        # Calculate progress for circle animation
        progress = (target_time - remaining_time) / target_time

        # Create layout for countdown
        with countdown_placeholder.container():
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Time Remaining: {int(remaining_time)} seconds")
            with col2:
                # Add circle element using progress for animation
                st.write(f"""
                <svg width="100" height="100">
                    <circle cx="50" cy="50" r="40" stroke="#0000ff" stroke-width="8" fill="none" />
                    <circle cx="50" cy="50" r="40" stroke="#ffffff" stroke-width="8" stroke-dasharray="251.2 251.2"
                            stroke-dashoffset="{251.2 - (progress * 251.2)}" />
                    <text x="50%" y="50%" text-anchor="middle" dy=".3em" font-size="20" fill="#ff0080">{int(remaining_time)}</text>
                </svg>
                """, unsafe_allow_html=True)

        # Update progress and sleep for smoother animation
        time.sleep(0.1)

    countdown_placeholder.empty()

# User input for countdown time
countdown_time = 5

# Start countdown on button click
if st.button("Start Countdown"):
    circular_countdown(countdown_time)



# Sidebar with navigation
st.sidebar.title("Navigation")

chat_page()

