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
import openai  # Import the official OpenAI library

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Initialize the OpenAI client
openai.api_key = OPENAI_API_KEY

# Function to get list of PDFs from GitHub repository
def get_pdfs_from_github(folder_url):
    api_url = folder_url.replace("github.com", "api.github.com/repos").replace("tree/main", "contents")
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {GITHUB_TOKEN}"
    }
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        files = response.json()
        pdf_files = [file['download_url'] for file in files if file['name'].endswith('.pdf')]
        return pdf_files
    else:
        st.error(f"Failed to fetch list of PDF files from GitHub: {response.status_code}")
        return []

# Function to download PDF files from the GitHub repository
def download_pdfs_from_github(folder_url):
    pdf_urls = get_pdfs_from_github(folder_url)
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
    text = []
    source_metadata = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
                source_metadata.append({'source': f"{pdf} - Page {page_num + 1}", 'url': f"https://github.com/scooter7/gemini_multipdf_chat/blob/main/qna/{pdf}"})
            else:
                st.warning(f"Failed to extract text from page in {pdf}")
    return text, source_metadata

# Split text into chunks
def get_text_chunks(text, metadata, chunk_size=2000, chunk_overlap=500):
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    chunk_metadata = []
    for i, page_text in enumerate(text):
        page_chunks = splitter.split_text(page_text)
        chunks.extend(page_chunks)
        chunk_metadata.extend([metadata[i]] * len(page_chunks))  # Assign correct metadata to each chunk
    return chunks, chunk_metadata

# Get embeddings for each chunk
def get_vector_store(chunks, metadata):
    if not chunks:
        st.error("No text chunks available for embedding")
        return None
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    documents = [Document(page_content=chunk, metadata=metadata[i]) for i, chunk in enumerate(chunks)]
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def load_or_create_vector_store(chunks, metadata):
    if os.path.exists("faiss_index"):
        try:
            embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
            vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            return vector_store
        except Exception as e:
            st.error(f"Failed to load FAISS index: {e}")
    return get_vector_store(chunks, metadata)

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Find and engage with past proposal questions and answers."}]

def rephrase_with_style(text, writing_style):
    # Construct a message for GPT-4o-mini
    messages = [
        {"role": "system", "content": "You are a highly skilled assistant who helps rewrite content in a specific tone and style."},
        {"role": "user", "content": f"Original Content: {text}\n\nWriting Style: {writing_style}\n\nPlease rewrite the content above using the provided writing style."}
    ]
    
    # Generate the response using GPT-4o-mini via OpenAI API
    completion = openai.ChatCompletion.create(
        messages=messages,
        model="gpt-4o-mini",
        max_tokens=1500,
        temperature=0.7
    )
    
    return completion.choices[0].message['content'].strip()

def user_input(user_question, writing_style, max_retries=5, delay=2):
    vector_store = load_or_create_vector_store([], [])
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

    # Modify the response style based on the writing style document
    response_text = rephrase_with_style(response_text, writing_style)

    return {"output_text": [response_text], "citations": citations}

def chunk_query(query, chunk_size=200):
    # Split the query into chunks
    return [query[i:i+chunk_size] for i in range(0, len(query), chunk_size)]

def modify_response_language(original_response, citations):
    response = original_response
    if citations:
        response += "\n\nSources:\n" + "\n".join(f"- [{citation}](https://github.com/scooter7/gemini_multipdf_chat/blob/main/qna/{citation.split(' - ')[0]})" for citation in citations)
    return response

def main():
    st.set_page_config(
        page_title="Past Proposal Q&A",
    )

    writing_style = None  # Initialize the variable here

    # Automatically download and process PDFs from GitHub
    with st.spinner("Downloading and processing PDFs..."):
        qna_folder_url = "https://github.com/scooter7/gemini_multipdf_chat/tree/main/qna"
        website_folder_url = "https://github.com/scooter7/gemini_multipdf_chat/tree/main/Website"

        # Process QnA PDFs
        qna_pdf_docs = download_pdfs_from_github(qna_folder_url)
        if qna_pdf_docs:
            raw_text, source_metadata = get_pdf_text(qna_pdf_docs)
            if raw_text:
                text_chunks, chunk_metadata = get_text_chunks(raw_text, source_metadata)
                if text_chunks:
                    vector_store = load_or_create_vector_store(text_chunks, chunk_metadata)
                    if vector_store:
                        st.success("QnA PDF processing complete")
                    else:
                        st.error("Failed to create vector store")
                else:
                    st.error("No text chunks created")
            else:
                st.error("No text extracted from QnA PDFs")
        else:
            st.error("No QnA PDFs downloaded")

        # Process Writing Style PDF
        website_pdf_docs = download_pdfs_from_github(website_folder_url)
        if website_pdf_docs:
            website_text, _ = get_pdf_text(website_pdf_docs)
            if website_text:
                writing_style = " ".join(website_text)  # Assign the writing style here
                st.success("Writing Style PDF processing complete")
            else:
                st.error("No text extracted from Writing Style PDF")
        else:
            st.error("No Writing Style PDF downloaded")

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
            if writing_style:
                response = user_input(chunk, writing_style)
                for item in response['output_text']:
                    full_response += item
                all_citations.extend(response['citations'])
            else:
                st.error("Writing style not loaded, unable to generate response.")

        modified_response = modify_response_language(full_response, all_citations)

        with st.chat_message("assistant"):
            st.write(modified_response)
            st.session_state.messages.append({"role": "assistant", "content": modified_response})

if __name__ == "__main__":
    main()
