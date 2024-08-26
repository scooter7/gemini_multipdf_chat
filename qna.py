import os
import requests
from io import BytesIO
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.schema import Document
import time

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Function to download and read a specific PDF from GitHub
def get_specific_pdf_text(pdf_url):
    try:
        response = requests.get(pdf_url, timeout=10)  # Add timeout
        response.raise_for_status()  # Raise an error for bad responses
        pdf_reader = PdfReader(BytesIO(response.content))
        text = []
        for page_num, page in enumerate(pdf_reader.pages):
            if page_num >= 5:  # Limit to the first 5 pages for testing
                break
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
            else:
                st.warning(f"Failed to extract text from page {page_num + 1} in the PDF.")
        return text
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading PDF: {e}")
        return []

# Split text into chunks
def get_text_chunks(text, chunk_size=2000, chunk_overlap=500):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(" ".join(text))

# Get embeddings for each chunk
def get_vector_store(chunks):
    if not chunks:
        st.error("No text chunks available for embedding")
        return None
    try:
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        documents = [Document(page_content=chunk) for chunk in chunks]
        vector_store = FAISS.from_documents(documents, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return vector_store
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        return None

def load_or_create_vector_store(chunks):
    if os.path.exists("faiss_index"):
        try:
            embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
            vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            return vector_store
        except Exception as e:
            st.error(f"Failed to load FAISS index: {e}")
    return get_vector_store(chunks)

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Find and engage with past proposal questions and answers."}]

def user_input(user_question, vector_store):
    try:
        docs = vector_store.similarity_search(user_question)
        if not docs:
            return {"output_text": ["No relevant documents found."], "citations": []}
    except Exception as e:
        st.error(f"Failed to perform similarity search: {e}")
        return {"output_text": [f"Failed to perform similarity search: {e}"]}

    response_text = ""
    citations = []

    for doc in docs:
        response_text += doc.page_content + "\n\n"
        citations.append(doc.metadata.get('source', 'Unknown Source'))

    return {"output_text": [response_text], "citations": citations}

def chunk_query(query, chunk_size=200):
    # Split the query into chunks
    return [query[i:i+chunk_size] for i in range(0, len(query), chunk_size)]

def modify_response_language(original_response, citations):
    # You can enhance this function to further refine the style using GPT or similar tools
    response = original_response
    if citations:
        response += "\n\nSources:\n" + "\n".join(f"- [{citation}](https://github.com/scooter7/gemini_multipdf_chat/blob/main/qna/{citation.split(' - ')[0]})" for citation in citations)
    return response

def main():
    st.set_page_config(page_title="Past Proposal Q&A")

    with st.spinner("Downloading and processing PDFs..."):
        # Load facts from Carnegie RFP FAQ PDF
        rfp_faq_url = "https://github.com/scooter7/gemini_multipdf_chat/raw/main/qna/Carnegie%20RFP%20standard%20content_FAQ.pdf"
        rfp_faq_text = get_specific_pdf_text(rfp_faq_url)
        if rfp_faq_text:
            fact_chunks = get_text_chunks(rfp_faq_text)
            vector_store = load_or_create_vector_store(fact_chunks)
            if vector_store:
                st.success("PDF processing complete")
            else:
                st.error("Failed to create vector store")
                return
        else:
            st.error("No text extracted from the RFP FAQ PDF")
            return

        # Load writing style from Carnegie web copy PDF
        web_copy_url = "https://github.com/scooter7/gemini_multipdf_chat/raw/main/qna/carnegiewebcopy.pdf"
        web_copy_text = get_specific_pdf_text(web_copy_url)
        if not web_copy_text:
            st.error("Failed to load the writing style from the web copy PDF")
            return

    # Main content area for displaying chat messages
    st.title("Past Proposal Q&A")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Chat input
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Find and engage with past proposal questions and answers."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Split the prompt into smaller chunks and process each one
        query_chunks = chunk_query(prompt)
        full_response = ''
        all_citations = []

        for chunk in query_chunks:
            response = user_input(chunk, vector_store)
            for item in response['output_text']:
                full_response += item
            all_citations.extend(response['citations'])

        # Use the web copy to refine the style of the response
        modified_response = modify_response_language(full_response, all_citations)

        with st.chat_message("assistant"):
            st.write(modified_response)
            st.session_state.messages.append({"role": "assistant", "content": modified_response})

if __name__ == "__main__":
    main()
