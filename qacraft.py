import os
import requests
from PyPDF2 import PdfReader
import streamlit as st
from dotenv import load_dotenv
import re
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai_api_key = st.secrets.get("google_gen_ai", {}).get("api_key", None)

# Check API Key
if not genai_api_key:
    st.error("Google Gemini API key is missing.")
else:
    # Initialize Google Gemini with API Key
    genai.configure(api_key=genai_api_key)

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

# Function to blend styles from the FAQ and website documents
def blend_styles(factual_text, style_text):
    # Tone adjustments - making the language more professional, authoritative, and concise
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
    
    # Avoid repetition and irrelevant content by keeping it concise and focused
    factual_text = re.sub(r'(\b\w+\b)( \1\b)+', r'\1', factual_text)  # Remove repetitive words
    factual_text = re.sub(r'\s+', ' ', factual_text).strip()  # Remove excessive spaces
    
    # Use style text to adjust tone, but ensure content remains on-topic
    if style_text:
        style_sentences = re.split(r'(?<=[.!?]) +', style_text)
        for sentence in style_sentences:
            if len(sentence.split()) > 3:  # Avoid very short or incomplete sentences
                factual_text += f" {sentence.strip()}"

    # Final cleanup to ensure response is clean and makes sense
    final_response = re.sub(r'\b(\w+)[’\']ve\b', r'\1 have', factual_text)  # Correct contractions
    final_response = re.sub(r'\b(\w+)[’\']re\b', r'\1 are', final_response)
    
    return final_response.strip()

# Function to generate a response using Google Gemini
def user_input(user_question, faq_text, style_text):
    # Blend factual content with stylistic elements
    blended_text = blend_styles(faq_text, style_text)

    # Generate a response using Google Gemini
    model = genai.GenerativeModel("gemini-pro")
    chat = model.start_chat(history=[])
    response = chat.send_message(blended_text)

    if hasattr(response, 'text'):
        return response.text
    else:
        st.error(f"Error interacting with Gemini: {getattr(response, 'finish_reason', 'Unknown error')}")
        return ""

def main():
    st.set_page_config(page_title="Past Proposal Q&A")

    with st.spinner("Downloading and processing PDFs..."):
        factual_docs = download_pdfs_from_github("scooter7/gemini_multipdf_chat", "qna")
        style_docs = download_pdfs_from_github("scooter7/gemini_multipdf_chat", "Website")
        
        if factual_docs and style_docs:
            raw_factual_text, factual_metadata = get_pdf_text(factual_docs)
            raw_style_text, style_metadata = get_pdf_text(style_docs)

            if raw_factual_text and raw_style_text:
                faq_text = " ".join(raw_factual_text)  # Combine all the factual text into one string
                style_text = " ".join(raw_style_text)  # Combine all the style text into one string
                st.success("PDF processing complete")
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

        # Generate the response using Google Gemini and display it
        final_response = user_input(prompt, faq_text, style_text)

        with st.chat_message("assistant"):
            st.write(final_response)
            st.session_state.messages.append({"role": "assistant", "content": final_response})

if __name__ == "__main__":
    main()
