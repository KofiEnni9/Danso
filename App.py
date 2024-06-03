import streamlit as st

# Set the page layout
st.set_page_config(layout="centered")

# Create a container for the buttons
container = st.container()

# Define the columns layout
col1, col2 = container.columns([1, 1])

# First row of buttons
with col1:
    st.button("self paced quiz button")
with col2:
    st.button("learn button")

# Second row of buttons
with col1:
    st.button("pop quiz button")
with col2:
    st.button("full quiz button")

tab1, tab2 = st.tabs(["Tab 1", "Tab2"])
tab1.write("this is tab 1")
tab2.write("this is tab 2")