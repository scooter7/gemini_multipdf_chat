import streamlit as st

# Set the page configuration
st.set_page_config(
    page_title="Carnegie Business Development Suite",
    page_icon="👋",
)

# Main welcome message
st.write("# Welcome to Carnegie's AI-Powered Business Development Suite! 👋")

# Sidebar message
st.sidebar.success("Select one of the apps from the sidebar")

# Main content area message
st.write("Please use the sidebar to navigate to the different apps.")
