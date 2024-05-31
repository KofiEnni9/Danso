import streamlit as st

# Set the page layout
st.set_page_config(layout="centered")

# Custom CSS for styling the container and buttons
st.markdown(
    """
    <style>
    .blue-container {
        background-color: #cce7ff;
        padding: 50px;
        border-radius: 10px;
        width: 80%;
        margin: auto;
    }
    .button {
        width: 200px;
        height: 50px;
        font-size: 20px;,
        margin: 10px;
        border-radius: 5px;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Create a container for the buttons
st.markdown('<div class="blue-container">', unsafe_allow_html=True)

# Define the columns layout
col1, col2 = st.columns(2)

# First row of buttons
with col1:
    st.markdown('<button class="button">self paced quiz button</button>', unsafe_allow_html=True)
with col2:
    st.markdown('<button class="button">learn button</button>', unsafe_allow_html=True)

# Second row of buttons
with col1:
    st.markdown('<button class="button">pop quiz button</button>', unsafe_allow_html=True)
with col2:
    st.markdown('<button class="button">full quiz button</button>', unsafe_allow_html=True)

# Close the container div
st.markdown('</div>', unsafe_allow_html=True)
