# import streamlit as st
# import time

# def circular_countdown(target_time):
#     """
#     Creates a circular countdown timer in Streamlit.

#     Args:
#         target_time: The target time in seconds for the countdown.
#     """
#     start_time = time.time()
#     remaining_time = target_time

#     countdown_placeholder = st.empty()
    
#     while remaining_time > 0:
#         # Update remaining time
#         elapsed_time = time.time() - start_time
#         remaining_time = target_time - elapsed_time

#         # Calculate progress for circle animation
#         progress = (target_time - remaining_time) / target_time

#         # Create layout for countdown
#         with countdown_placeholder.container():
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.write(f"Time Remaining: {int(remaining_time)} seconds")
#             with col2:
#                 # Add circle element using progress for animation
#                 st.write(f"""
#                 <svg width="100" height="100">
#                     <circle cx="50" cy="50" r="40" stroke="#0000ff" stroke-width="8" fill="none" />
#                     <circle cx="50" cy="50" r="40" stroke="#ffffff" stroke-width="8" stroke-dasharray="251.2 251.2"
#                             stroke-dashoffset="{251.2 - (progress * 251.2)}" />
#                     <text x="50%" y="50%" text-anchor="middle" dy=".3em" font-size="20" fill="#ff0080">{int(remaining_time)}</text>
#                 </svg>
#                 """, unsafe_allow_html=True)

#         # Update progress and sleep for smoother animation
#         time.sleep(0.1)

#     countdown_placeholder.empty()

# # User input for countdown time
# countdown_time = 5

# # Start countdown on button click
# if st.button("Start Countdown"):
#     circular_countdown(countdown_time)




import streamlit as st
from dataclasses import dataclass
import time

# Set the page layout
st.set_page_config(layout="wide")

# Check if a button has been clicked and update session state
if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

# Function to go back to button screen
def go_back():
    st.session_state.button_clicked = False
    st.experimental_rerun()

# Display buttons only if none has been clicked
if not st.session_state.button_clicked:
    # Custom CSS to position buttons at each corner of the screen with specified dimensions
    st.markdown("""
        <style>
            .corner-button {
                position: fixed;
                padding: 10px 20px;
                font-size: 16px;
                z-index: 1000;
            }
            #btn1 { top: 10px; left: 10px; width: 150px; height: 75px; }
            #btn2 { top: 10px; right: 10px; width: 200px; height: 100px; }
            #btn3 { bottom: 10px; left: 10px; width: 180px; height: 90px; }
            #btn4 { bottom: 10px; right: 10px; width: 160px; height: 80px; }
        </style>
    """, unsafe_allow_html=True)

    # Create buttons and assign ids for CSS positioning
    btn1 = st.button("self paced quiz button", key="btn1", help="Self Paced Quiz")
    btn2 = st.button("learn button", key="btn2", help="Learn")
    btn3 = st.button("pop quiz button", key="btn3", help="Pop Quiz")
    btn4 = st.button("full quiz button", key="btn4", help="Full Quiz")

    # Check which button was clicked and update session state
    if btn1 or btn2 or btn3 or btn4:
        st.session_state.button_clicked = True
        st.experimental_rerun()

    # Apply CSS classes to the buttons
    st.markdown("""
        <script>
            document.getElementsByClassName('stButton')[0].classList.add('corner-button');
            document.getElementsByClassName('stButton')[0].id = 'btn1';
            document.getElementsByClassName('stButton')[1].classList.add('corner-button');
            document.getElementsByClassName('stButton')[1].id = 'btn2';
            document.getElementsByClassName('stButton')[2].classList.add('corner-button');
            document.getElementsByClassName('stButton')[2].id = 'btn3';
            document.getElementsByClassName('stButton')[3].classList.add('corner-button');
            document.getElementsByClassName('stButton')[3].id = 'btn4';
        </script>
    """, unsafe_allow_html=True)

else:
    # Define a Message dataclass to store the messages
    @dataclass
    class Message:
        actor: str
        payload: str

    # Constants for the actors and session state key
    USER = "user"
    ASSISTANT = "ai"
    MESSAGES = "messages"

    # Initialize session state if not already present
    if MESSAGES not in st.session_state:
        st.session_state[MESSAGES] = [Message(actor=ASSISTANT, payload="Hi! How can I help you?")]

    # Custom CSS for message styling and sidebar styling
    st.markdown("""
        <style>
            .user-message, .assistant-message {
                padding: 10px;
                max-width: 100%;
                height: auto;
                border-radius: 10px;
                margin: 5px 0;
                width: fit-content;
                opacity: 0;
                animation: fadeIn 1s forwards;
                word-wrap: break-word;
            }
            .user-message {
                background-color: #d3d3d3;
                text-align: right;
                margin-left: auto;
            }
            .assistant-message {
                background-color: #add8e6;
            }
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            .sidebar-content {
                position: fixed;
                top: 10px;
                right: 10px;
                z-index: 1000;
            }
        </style>
    """, unsafe_allow_html=True)

    # Display the back arrow button in the sidebar
    with st.sidebar:
        if st.button("← Go Back", key="back_arrow", help="Go back to button screen"):
            go_back()

    # Function to display messages with a delay
    def display_message(message, message_type):
        placeholder = st.empty()
        placeholder.markdown(f'<div class="{message_type}">{message}</div>', unsafe_allow_html=True)
        time.sleep(1)

    # Display existing messages with animation
    for msg in st.session_state[MESSAGES]:
        if msg.actor == USER:
            display_message(msg.payload, "user-message")
        else:
            display_message(msg.payload, "assistant-message")

    # Input prompt from user
    prompt = st.chat_input("Enter a prompt here")

    if prompt:
        # Add user message to session state and display it
        st.session_state[MESSAGES].append(Message(actor=USER, payload=prompt))
        display_message(prompt, "user-message")

        # Generate a placeholder response
        with st.spinner('Generating response...'):
            response = f"I received your message: {prompt}"

        # Add the assistant's response to session state and display it
        st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=response))
        display_message(response, "assistant-message")
