import io
import os
import requests
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def fetch_pdfs_from_github(repo_url):
    contents_url = f"https://api.github.com/repos/{repo_url}/contents/docs"
    response = requests.get(contents_url)
    files = response.json()
    texts = []
    for file in files:
        if file['name'].endswith('.pdf'):
            download_url = file['download_url']
            pdf_response = requests.get(download_url)
            pdf_file = io.BytesIO(pdf_response.content)
            pdf_reader = PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            texts.append(text)
    return texts

def get_text_chunks(texts):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    all_chunks = []
    for text in texts:
        chunks = splitter.split_text(text)
        all_chunks.extend(chunks)
    return all_chunks

@st.cache_data
def process_pdfs_from_github(repo_url):
    texts = fetch_pdfs_from_github(repo_url)
    text_chunks = get_text_chunks(texts)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, 'answer is not available in the context', don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", client=genai, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

def user_input(user_question, vector_store):
    new_db = FAISS.load_local("faiss_index", vector_store.embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response

def main():
    st.set_page_config(page_title="Learn about Carnegie")
    repo_url = 'scooter7/gemini_multipdf_chat'
    if 'vector_store' not in st.session_state or not os.path.exists("faiss_index"):
        st.session_state['vector_store'] = process_pdfs_from_github(repo_url)

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
