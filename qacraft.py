import os
import requests
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.schema import Document
import re

# Load environment variables
load_dotenv()

# Load the OpenAI API key from Streamlit secrets
OPENAI_API_KEY = st.secrets["openai_api_key"]["api_key"]

# Debugging print statement to check the type of OPENAI_API_KEY
st.write(f"API Key Type: {type(OPENAI_API_KEY)}")

if not isinstance(OPENAI_API_KEY, str):
    st.error("The API key is not a string. Please check the secrets configuration.")

# Function to get list of PDFs from GitHub repository
def get_pdfs_from_github(repo, folder):
    api_url = f"https://api.github.com/repos/{repo}/contents/{folder}"
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
def download_pdfs_from_github(repo, folder):
    pdf_urls = get_pdfs_from_github(repo, folder)
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
        if page_chunks:  # Ensure that the chunking is working
            chunks.extend(page_chunks)
            chunk_metadata.extend([metadata[i]] * len(page_chunks))  # Assign correct metadata to each chunk
        else:
            st.warning(f"No chunks created for page {i+1} in {metadata[i]['source']}")
    return chunks, chunk_metadata

# Get embeddings for each chunk
def get_vector_store(chunks, metadata, index_name):
    if not chunks:
        st.error("No text chunks available for embedding")
        return None
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    documents = [Document(page_content=chunk, metadata=metadata[i]) for i, chunk in enumerate(chunks)]
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    vector_store.save_local(index_name)
    return vector_store

def load_or_create_vector_store(chunks, metadata, index_name):
    if os.path.exists(index_name):
        try:
            embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
            vector_store = FAISS.load_local(index_name, embeddings, allow_dangerous_deserialization=True)
            return vector_store
        except Exception as e:
            st.error(f"Failed to load FAISS index: {e}")
    return get_vector_store(chunks, metadata, index_name)

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Find and engage with past proposal questions and answers."}]

def blend_styles(factual_text, style_text):
    # Tone adjustments - making the language more professional, authoritative, and empowering
    tone_mapping = {
        r'\bwe\b': 'our team',
        r'\byou\b': 'your organization',
        r'\bdo not\b': "don't",
        r'\bcannot\b': "can't",
        r'\bwill\b': 'shall',
        r'\bare\b': 'are likely to',
        r'\bshould\b': 'might consider',
        r'\bis\b': 'is generally regarded as',
        r'\bwe\b': 'we’ll work together to',
        r'\byour\b': 'your tailored solution',
    }
    
    # Apply tone adjustments
    for pattern, replacement in tone_mapping.items():
        factual_text = re.sub(pattern, replacement, factual_text, flags=re.IGNORECASE)

    # Adding collaborative language and strategic word choices
    factual_text = re.sub(r'\bsolution\b', 'customized solution', factual_text, flags=re.IGNORECASE)
    factual_text = re.sub(r'\bdata\b', 'intelligent data', factual_text, flags=re.IGNORECASE)
    factual_text = re.sub(r'\bmarket\b', 'key market', factual_text, flags=re.IGNORECASE)

    # Example - Breaking down long sentences to match a more concise style, but also combining where necessary for complexity
    sentences = re.split(r'(?<=[.!?]) +', factual_text)
    complex_factual_text = ""
    for sentence in sentences:
        if len(sentence.split()) > 20:  # Example threshold for long sentences
            parts = re.split(r',|\band\b|\bor\b', sentence)
            complex_factual_text += ' '.join(parts) + ". "
        else:
            complex_factual_text += sentence + " "

    # Adding stylistic elements from the style text if present
    stylistic_elements = {
        'introduction': re.search(r'(introduction.*?)(\n|$)', style_text, re.IGNORECASE),
        'conclusion': re.search(r'(conclusion.*?)(\n|$)', style_text, re.IGNORECASE)
    }

    if stylistic_elements['introduction']:
        introduction = stylistic_elements['introduction'].group(1)
        complex_factual_text = f"{introduction}\n\n{complex_factual_text}"

    if stylistic_elements['conclusion']:
        conclusion = stylistic_elements['conclusion'].group(1)
        complex_factual_text += f"\n\n{conclusion}"

    return complex_factual_text.strip()

def user_input(user_question, max_retries=5, delay=2):
    factual_store = load_or_create_vector_store([], [], "faiss_index_factual")
    style_store = load_or_create_vector_store([], [], "faiss_index_style")
    
    if not factual_store or not style_store:
        st.error("Failed to load or create the vector store.")
        return {"output_text": ["Failed to load or create the vector store."]}

    try:
        factual_docs = factual_store.similarity_search(user_question)
        style_docs = style_store.similarity_search(user_question)
    except Exception as e:
        st.error(f"Failed to perform similarity search: {e}")
        return {"output_text": [f"Failed to perform similarity search: {e}"]}

    response_text = ""
    citations = []

    for doc in factual_docs:
        response_text += doc.page_content + "\n\n"
        citations.append(doc.metadata['source'])

    # Blend in the style elements
    for style_doc in style_docs:
        response_text = blend_styles(response_text, style_doc.page_content)

    return {"output_text": [response_text], "citations": citations}

def chunk_query(query, chunk_size=200):
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

    with st.spinner("Downloading and processing PDFs..."):
        factual_docs = download_pdfs_from_github("scooter7/gemini_multipdf_chat", "qna")
        style_docs = download_pdfs_from_github("scooter7/gemini_multipdf_chat", "Website")
        
        if factual_docs and style_docs:
            raw_factual_text, factual_metadata = get_pdf_text(factual_docs)
            raw_style_text, style_metadata = get_pdf_text(style_docs)

            if raw_factual_text and raw_style_text:
                factual_chunks, factual_chunk_metadata = get_text_chunks(raw_factual_text, factual_metadata)
                style_chunks, style_chunk_metadata = get_text_chunks(raw_style_text, style_metadata)

                if factual_chunks and style_chunks:
                    factual_store = load_or_create_vector_store(factual_chunks, factual_chunk_metadata, "faiss_index_factual")
                    style_store = load_or_create_vector_store(style_chunks, style_chunk_metadata, "faiss_index_style")
                    
                    if factual_store and style_store:
                        st.success("PDF processing complete")
                    else:
                        st.error("Failed to create vector store")
                else:
                    st.error("No text chunks created")
            else:
                st.error("No text extracted from PDFs")
        else:
            st.error("No PDFs downloaded")

    st.title("Past Proposal Q&A")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

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
