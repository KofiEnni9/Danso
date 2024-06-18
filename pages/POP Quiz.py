import streamlit as st

def main():
    # Initialize an empty list in session state
    if "my_list" not in st.session_state:
        st.session_state.my_list = []

    # Title and description
    st.title("👩‍🔬 Welcome Back! 👨‍🔬")

    st.write("### 🌟 Description 🌟")
    st.write("🧪 You will be asked **one question at a time**.")
    st.write("✍️ Provide **concise answers** which will be evaluated.")
    st.write("🔄 You have **two attempts** per question.")
    st.write("📈 Remember, every attempt **contributes towards your general progress** on this app.")

if __name__ == "__main__":
    main()
