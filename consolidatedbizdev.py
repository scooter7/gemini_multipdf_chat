import streamlit as st
from streamlit_option_menu import option_menu

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

# Function to handle page refresh
def refresh_page():
    clear_chat_history()
    st.experimental_rerun()

# Set up navigation with option menu
selected = option_menu(
    menu_title=None,
    options=["Welcome", "App1", "App2", "App3"],
    icons=["house", "file-earmark", "file-earmark", "file-earmark"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    on_change=refresh_page
)

# Main content area
if selected == "Welcome":
    st.write("Welcome to the Carnegie Business Development Suite! Please select an app from the sidebar.")
elif selected == "App1":
    import app1  # Assuming app1.py exists and defines a function main()
    app1.main()
elif selected == "App2":
    import app2  # Assuming app2.py exists and defines a function main()
    app2.main()
elif selected == "App3":
    import app3  # Assuming app3.py exists and defines a function main()
    app3.main()
