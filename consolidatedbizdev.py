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

# Ensure vector store creation/loading is correct
def load_or_create_vector_store(chunks, metadata):
    if os.path.exists("faiss_index"):
        try:
            embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
            vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            st.write("Vector store loaded successfully")
            return vector_store
        except Exception as e:
            st.error(f"Failed to load FAISS index: {e}")
    st.write("Creating new vector store")
    return get_vector_store(chunks, metadata)

# Detailed logging for similarity search
def user_input(user_question, max_retries=5, delay=2):
    vector_store = load_or_create_vector_store([], [])
    if not vector_store:
        st.error("Failed to load or create the vector store.")
        return {"output_text": ["Failed to load or create the vector store."], "citations": []}

    try:
        if not vector_store:
            raise ValueError("Vector store is not initialized.")
        docs = vector_store.similarity_search(user_question)
        st.write(f"Found {len(docs)} documents matching the query.")
    except Exception as e:
        st.error(f"Failed to perform similarity search: {e}")
        return {"output_text": [f"Failed to perform similarity search: {e}"], "citations": []}

    response_text = ""
    citations = []

    for doc in docs:
        response_text += doc.page_content + "\n\n"
        citations.append(doc.metadata['source'])

    return {"output_text": [response_text], "citations": citations}

# Set up navigation with option menu
selected = option_menu(
    menu_title=None,
    options=["Welcome", "App1", "App2", "App3"],
    icons=["house", "file-earmark", "file-earmark", "file-earmark"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    key="tab_selection"
)

# Main content area
if selected == "Welcome":
    st.write("Welcome to the Carnegie Business Development Suite! Please select an app from the sidebar.")
elif selected == "App1":
    clear_chat_history()
    from pages import app1  # Assuming app1.py exists and defines a function main()
    app1.main()
elif selected == "App2":
    clear_chat_history()
    from pages import app2  # Assuming app2.py exists and defines a function main()
    app2.main()
elif selected == "App3":
    clear_chat_history()
    from pages import qna  # Assuming qna.py exists and defines a function main()
    qna.main()
