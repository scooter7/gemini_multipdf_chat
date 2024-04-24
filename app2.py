import requests
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")

def fetch_pdfs_from_github(repo_url):
    contents_url = f"https://api.github.com/repos/{repo_url}/contents/docs"
    response = requests.get(contents_url)
    files = response.json()
    pdf_files = [f for f in files if f['name'].endswith('.pdf')]
    pdf_texts = []
    for file in pdf_files:
        download_url = file['download_url']
        pdf_response = requests.get(download_url)
        pdf_bytes = pdf_response.content
        pdf_reader = PdfReader(bytes(pdf_bytes))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        pdf_texts.append(text)
    return pdf_texts

def get_text_chunks(texts):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    return chunks

@st.cache_data
def process_pdf_texts(texts):
    text_chunks = get_text_chunks(texts)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def main():
    st.set_page_config(page_title="Learn about Carnegie")
    repo_url = 'scooter7/gemini_multipdf_chat'
    if 'vector_store' not in st.session_state or not os.path.exists("faiss_index"):
        pdf_texts = fetch_pdfs_from_github(repo_url)
        st.session_state['vector_store'] = process_pdf_texts(pdf_texts)

    st.title("Learn about Carnegie")
    st.write("Chat with Carnegie AI!")
    if "messages" not in st.session_state:
        st.session_state['messages'] = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})

if __name__ == "__main__":
    main()
