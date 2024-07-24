import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import requests

# Configure Google API key from Streamlit secrets
genai.configure(api_key=st.secrets["google_api_key"])
google_api_key = st.secrets["google_api_key"]

def fetch_pdfs_from_github(repo_url):
    response = requests.get(repo_url)
    pdf_urls = []
    if response.status_code == 200:
        content = response.json()
        for file_info in content:
            if file_info['name'].endswith('.pdf'):
                pdf_urls.append(file_info['download_url'])
    return pdf_urls

def extract_text_from_pdf(url):
    response = requests.get(url)
    with open("temp.pdf", "wb") as f:
        f.write(response.content)
    pdf_reader = PdfReader("temp.pdf")
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    os.remove("temp.pdf")
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks

@st.cache_resource
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", client=genai, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "upload an RFP and ask about scope, due dates, anything you'd like..."}
    ]

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response

def main():
    st.set_page_config(page_title="RFP Summarization Bot")

    repo_url = 'https://api.github.com/repos/scooter7/gemini_multipdf_chat/contents/docs'
    pdf_urls = fetch_pdfs_from_github(repo_url)
    pdf_texts = {url: extract_text_from_pdf(url) for url in pdf_urls}

    st.title("PDF Query App")

    query = st.text_input("Enter your query")
    if query:
        st.write(f"Results for query: {query}")
        for url, text in pdf_texts.items():
            if query.lower() in text.lower():
                st.write(f"**PDF URL:** {url}")
                st.write(text[:500])

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "upload an RFP and ask about scope, due dates, anything you'd like..."}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response['output_text']:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)

if __name__ == "__main__":
    main()
