import os
import time
import requests
from io import BytesIO
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Function to get list of PDFs from GitHub repository
def get_pdfs_from_github():
    api_url = "https://api.github.com/repos/scooter7/gemini_multipdf_chat/contents/docs"
    headers = {"Accept": "application/vnd.github.v3+json"}
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        files = response.json()
        pdf_files = [file['download_url'] for file in files if file['name'].endswith('.pdf')]
        return pdf_files
    else:
        st.error(f"Failed to fetch list of PDF files from GitHub: {response.status_code}")
        return []

# Function to download PDF files from the GitHub repository
def download_pdfs_from_github():
    pdf_urls = get_pdfs_from_github()
    pdf_docs = []
    for url in pdf_urls:
        response = requests.get(url)
        if response.status_code == 200:
            file_name = url.split('/')[-1]
            with open(file_name, 'wb') as f:
                f.write(response.content)
            pdf_docs.append(file_name)
        else:
            st.error(f"Failed to download {url}")
    return pdf_docs

# Read all pdf files and return text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
            else:
                st.warning(f"Failed to extract text from page in {pdf}")
    return text

# Split text into chunks
def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return chunks  # list of strings

# Get embeddings for each chunk
def get_vector_store(chunks):
    if not chunks:
        st.error("No text chunks available for embedding")
        return None
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def load_or_create_vector_store(chunks):
    if os.path.exists("faiss_index"):
        try:
            embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
            vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            return vector_store
        except Exception as e:
            st.error(f"Failed to load FAISS index: {e}")
    return get_vector_store(chunks)

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatOpenAI(api_key=OPENAI_API_KEY)
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Query the RFP repository and ask about scope, due dates, anything you'd like..."}]

def user_input(user_question, max_retries=5, delay=2):
    vector_store = load_or_create_vector_store([])
    if not vector_store:
        st.error("Failed to load or create the vector store.")
        return {"output_text": ["Failed to load or create the vector store."]}

    try:
        docs = vector_store.similarity_search(user_question)
    except Exception as e:
        st.error(f"Failed to perform similarity search: {e}")
        return {"output_text": [f"Failed to perform similarity search: {e}"]}

    chain = get_conversational_chain()
    response_text = ""

    for doc in docs:
        context = doc.page_content
        context_chunks = get_text_chunks(context)

        for chunk in context_chunks:
            for attempt in range(max_retries):
                try:
                    response = chain(
                        {"input_documents": [doc], "context": chunk, "question": user_question}, return_only_outputs=True, )
                    response_text += response['output_text'][0] + " "
                    break
                except Exception as e:
                    if 'Resource has been exhausted' in str(e):
                        st.warning(f"API quota has been exhausted. Retrying in {delay} seconds...")
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                    else:
                        st.error(f"Failed to generate response: {e}")
                        return {"output_text": [f"Failed to generate response: {e}"]}

    return {"output_text": [response_text]}

def chunk_query(query, chunk_size=200):
    # Split the query into chunks
    return [query[i:i+chunk_size] for i in range(0, len(query), chunk_size)]

def main():
    st.set_page_config(
        page_title="RFP Summarization Bot",
    )

    # Automatically download and process PDFs from GitHub
    with st.spinner("Downloading and processing PDFs..."):
        pdf_docs = download_pdfs_from_github()
        if pdf_docs:
            raw_text = get_pdf_text(pdf_docs)
            if raw_text:
                text_chunks = get_text_chunks(raw_text)
                if text_chunks:
                    vector_store = load_or_create_vector_store(text_chunks)
                    if vector_store:
                        st.success("PDF processing complete")
                    else:
                        st.error("Failed to create vector store")
                else:
                    st.error("No text chunks created")
            else:
                st.error("No text extracted from PDFs")
        else:
            st.error("No PDFs downloaded")

    # Main content area for displaying chat messages
    st.title("Summarize and ask questions about RFPs")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Chat input
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Query the RFP repository and ask about scope, due dates, anything you'd like..."}]

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

        for chunk in query_chunks:
            response = user_input(chunk)
            for item in response['output_text']:
                full_response += item

        with st.chat_message("assistant"):
            st.write(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
