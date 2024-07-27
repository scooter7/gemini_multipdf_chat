import streamlit as st
import os
import time
import requests
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.schema import Document

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@st.cache(allow_output_mutation=True)
def download_pdfs_from_github():
    api_url = "https://api.github.com/repos/scooter7/gemini_multipdf_chat/contents/docs"
    headers = {"Accept": "application/vnd.github.v3+json"}
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        files = response.json()
        pdf_files = [file['download_url'] for file in files if file['name'].endswith('.pdf')]
        pdf_docs = []
        for url in pdf_files:
            response = requests.get(url)
            if response.status_code == 200:
                file_name = url.split('/')[-1]
                with open(file_name, 'wb') as f:
                    f.write(response.content)
                pdf_docs.append(file_name)
        return pdf_docs
    else:
        st.error(f"Failed to fetch list of PDF files from GitHub: {response.status_code}")
        return []

@st.cache(allow_output_mutation=True)
def load_pdf_text(pdf_docs):
    text = []
    source_metadata = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
                source_metadata.append({'source': f"{pdf} - Page {page_num + 1}", 'url': f"https://github.com/scooter7/gemini_multipdf_chat/blob/main/docs/{pdf}"})
            else:
                st.warning(f"Failed to extract text from page in {pdf}")
    return text, source_metadata

@st.cache(allow_output_mutation=True)
def create_vector_store(text, metadata):
    splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    chunks = []
    chunk_metadata = []
    for i, page_text in enumerate(text):
        page_chunks = splitter.split_text(page_text)
        chunks.extend(page_chunks)
        chunk_metadata.extend([metadata[i]] * len(page_chunks))
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    documents = [Document(page_content=chunk, metadata=chunk_metadata[i]) for i, chunk in enumerate(chunks)]
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

@st.cache(allow_output_mutation=True)
def load_vector_store():
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return vector_store

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Find and engage with past Carnegie proposals."}]

def user_input(user_question, max_retries=5, delay=2):
    vector_store = load_vector_store()
    if not vector_store:
        st.error("Failed to load or create the vector store.")
        return {"output_text": ["Failed to load or create the vector store."]}

    try:
        docs = vector_store.similarity_search(user_question)
    except Exception as e:
        st.error(f"Failed to perform similarity search: {e}")
        return {"output_text": [f"Failed to perform similarity search: {e}"]}

    response_text = ""
    citations = []

    for doc in docs:
        response_text += doc.page_content + "\n\n"
        citations.append(doc.metadata['source'])

    return {"output_text": [response_text], "citations": citations}

def chunk_query(query, chunk_size=200):
    return [query[i:i+chunk_size] for i in range(0, len(query), chunk_size)]

def modify_response_language(original_response, citations):
    response = original_response
    if citations:
        response += "\n\nSources:\n" + "\n".join(f"- [{citation}](https://github.com/scooter7/gemini_multipdf_chat/blob/main/docs/{citation.split(' - ')[0]})" for citation in citations)
    return response

def main():
    st.set_page_config(page_title="Leverage Existing Proposal Content")

    with st.spinner("Downloading and processing PDFs..."):
        pdf_docs = download_pdfs_from_github()
        if pdf_docs:
            raw_text, source_metadata = load_pdf_text(pdf_docs)
            if raw_text:
                create_vector_store(raw_text, source_metadata)
                st.success("PDF processing complete")
            else:
                st.error("No text extracted from PDFs")
        else:
            st.error("No PDFs downloaded")

    st.title("Find and engage with past proposal content")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Find and engage with past Carnegie proposals."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        query_chunks = chunk_query(prompt)
        full_response = ''
        all_citations = []

        for chunk in query_chunks:
            response = user_input(chunk)
            for item in response['output_text']:
                full_response += item
            all_citations.extend(response['citations'])

        modified_response = modify_response_language(full_response, all_citations)

        with st.chat_message("assistant"):
            st.write(modified_response)
            st.session_state.messages.append({"role": "assistant", "content": modified_response})

if __name__ == "__main__":
    main()
