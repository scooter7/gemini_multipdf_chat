import streamlit as st

st.set_page_config(
    page_title="Carnegie Business Development Suite",
    page_icon="👋",
)

st.write("# Welcome to Carnegie's AI-Powered Business Development Suite! 👋")

st.sidebar.success("Select one of the apps")

# Function to clear chat history
def clear_chat_history():
    if "messages" in st.session_state:
        del st.session_state.messages

# Function to handle app navigation
def navigate_to_page(page):
    clear_chat_history()
    st.session_state.page = page

# Set up navigation
if "page" not in st.session_state:
    st.session_state.page = "Welcome"

# Sidebar for navigation
with st.sidebar:
    if st.button("Go to App 1"):
        navigate_to_page("App1")
    if st.button("Go to App 2"):
        navigate_to_page("App2")
    if st.button("Go to App 3"):
        navigate_to_page("App3")

# Main content area
if st.session_state.page == "Welcome":
    st.write("Welcome to the Carnegie Business Development Suite! Please select an app from the sidebar.")
elif st.session_state.page == "App1":
    import app1  # Assuming app1.py exists and defines a function main()
    app1.main()
elif st.session_state.page == "App2":
    import app2  # Assuming app2.py exists and defines a function main()
    app2.main()
elif st.session_state.page == "App3":
    import app3  # Assuming app3.py exists and defines a function main()
    app3.main()
