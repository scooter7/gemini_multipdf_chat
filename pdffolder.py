import os
import requests
from PyPDF2 import PdfReader
from io import BytesIO
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text_from_github(repo_url):
    response = requests.get(repo_url)
    file_names = response.json()  # Assuming the repo URL directly points to a JSON listing the files
    text = ""
    for file_info in file_names:
        if file_info['name'].endswith('.pdf'):
            pdf_url = file_info['download_url']  # URL to the raw PDF file
            pdf_response = requests.get(pdf_url)
            pdf_reader = PdfReader(BytesIO(pdf_response.content))
            for page in pdf_reader.pages:
                text += page.extract_text() if page.extract_text() else ""
    return text

@st.cache
def process_pdf_folder(github_repo_url):
    raw_text = get_pdf_text_from_github(github_repo_url)
    splitter = RecursiveCharacterTextSplitter(chunk_size=15000, chunk_overlap=5000)
    text_chunks = splitter.split_text(raw_text)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def main():
    st.set_page_config(page_title="Learn about Carnegie")
    github_repo_url = 'https://api.github.com/repos/scooter7/gemini_multipdf_chat/contents/docs'  # Adjust the API URL as necessary
    if 'vector_store' not in st.session_state or not os.path.exists("faiss_index"):
        st.session_state['vector_store'] = process_pdf_folder(github_repo_url)

    st.title("Learn about Carnegie")
    st.write("Chat with Carnegie AI!")
    if "messages" not in st.session_state:
        st.session_state['messages'] = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = user_input(prompt, st.session_state['vector_store'])
        if response is not None:
            full_response = ''.join(response['output_text'])
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            with st.chat_message("assistant"):
                st.markdown(full_response)

if __name__ == "__main__":
    main()
