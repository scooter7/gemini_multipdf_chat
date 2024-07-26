import streamlit as st
import os
import time
import requests
from io import BytesIO
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import base64
import google.generativeai as genai
from langchain.vectorstores import FAISS as LangchainFAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)


# Shared Functions
def get_pdf_text(pdf_docs):
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


# App 1 Functions
def get_pdfs_from_github_1():
    api_url = "https://api.github.com/repos/scooter7/gemini_multipdf_chat/contents/qna"
    headers = {"Accept": "application/vnd.github.v3+json"}
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        files = response.json()
        pdf_files = [file['download_url'] for file in files if file['name'].endswith('.pdf')]
        return pdf_files
    else:
        st.error(f"Failed to fetch list of PDF files from GitHub: {response.status_code}")
        return []


def download_pdfs_from_github_1():
    pdf_urls = get_pdfs_from_github_1()
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


def clear_chat_history_1():
    st.session_state.messages = [
        {"role": "assistant", "content": "Find and engage with past proposal questions and answers."}]


def user_input_1(user_question, max_retries=5, delay=2):
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

    return {"output_text": [response_text], "citations": citations}


def chunk_query_1(query, chunk_size=200):
    return [query[i:i + chunk_size] for i in range(0, len(query), chunk_size)]


def modify_response_language_1(original_response, citations):
    response = original_response
    if citations:
        response += "\n\nSources:\n" + "\n".join(f"- [{citation}](https://github.com/scooter7/gemini_multipdf_chat/blob/main/qna/{citation.split(' - ')[0]})" for citation in citations)
    return response


def app1():
    with st.spinner("Downloading and processing PDFs..."):
        pdf_docs = download_pdfs_from_github_1()
        if pdf_docs:
            raw_text, source_metadata = get_pdf_text(pdf_docs)
            if raw_text:
                text_chunks, chunk_metadata = get_text_chunks(raw_text, source_metadata)
                if text_chunks:
                    vector_store = load_or_create_vector_store(text_chunks, chunk_metadata)
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

    st.title("Past Proposal Q&A")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history_1)

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

        query_chunks = chunk_query_1(prompt)
        full_response = ''
        all_citations = []

        for chunk in query_chunks:
            response = user_input_1(chunk)
            for item in response['output_text']:
                full_response += item
            all_citations.extend(response['citations'])

        modified_response = modify_response_language_1(full_response, all_citations)

        with st.chat_message("assistant"):
            st.write(modified_response)
            st.session_state.messages.append({"role": "assistant", "content": modified_response})


# App 2 Functions
def get_pdf_text_2(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks_2(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks


def get_vector_store_2(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore
    vector_store = LangchainFAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain_2():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client=genai,
                                   temperature=0.3,
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain


def clear_chat_history_2():
    st.session_state.messages = [
        {"role": "assistant", "content": "upload an RFP and ask about scope, due dates, anything you'd like..."}]


def user_input_2(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore

    new_db = LangchainFAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain_2()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True, )

    return response


def app2():
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text_2(pdf_docs)
                text_chunks = get_text_chunks_2(raw_text)
                get_vector_store_2(text_chunks)
                st.success("Done")

    st.title("Summarize and ask questions about RFPs")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history_2)

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "upload an RFP and ask about scope, due dates, anything you'd like..."}]

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
                response = user_input_2(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response['output_text']:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)


# App 3 Functions
def get_pdfs_from_github_3():
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


def download_pdfs_from_github_3():
    pdf_urls = get_pdfs_from_github_3()
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


def get_conversational_chain_3():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = "gpt-4o-mini"
    return model


def clear_chat_history_3():
    st.session_state.messages = [
        {"role": "assistant", "content": "Find and engage with past Carnegie proposals."}]


def user_input_3(user_question, max_retries=5, delay=2):
    vector_store = load_or_create_vector_store([], [])
    if not vector_store:
        st.error("Failed to load or create the vector store.")
        return {"output_text": ["Failed to load or create the vector store."]}

    try:
        docs = vector_store.similarity_search(user_question)
    except Exception as e:
        st.error(f"Failed to perform similarity search: {e}")
        return {"output_text": [f"Failed to perform similarity search: {e}"]}

    model = get_conversational_chain_3()
    response_text = ""
    citations = []

    for doc in docs:
        context = doc.page_content
        context_chunks = get_text_chunks([context], [doc.metadata])

        for chunk, meta in zip(context_chunks[0], context_chunks[1]):
            for attempt in range(max_retries):
                try:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You specialize in leveraging the PDF content in the repository to find and discuss information from these documents."},
                            {"role": "user", "content": f"Context: {chunk}\n\nQuestion: {user_question}"}
                        ]
                    )
                    response_text += completion.choices[0].message.content + " "
                    citations.append(meta['source'])
                    break
                except Exception as e:
                    if 'Resource has been exhausted' in str(e):
                        st.warning(f"API quota has been exhausted. Retrying in {delay} seconds...")
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                    else:
                        st.error(f"Failed to generate response: {e}")
                        return {"output_text": [f"Failed to generate response: {e}"]}

    return {"output_text": [response_text], "citations": citations}


def chunk_query_3(query, chunk_size=200):
    return [query[i:i + chunk_size] for i in range(0, len(query), chunk_size)]


def modify_response_language_3(original_response, citations):
    response = original_response.replace(" they ", " we ")
    response = response.replace("They ", "We ")
    response = response.replace(" their ", " our ")
    response.replace("Their ", "Our ")
    response = response.replace(" them ", " us ")
    response = response.replace("Them ", "Us ")
    if citations:
        response += "\n\nSources:\n" + "\n".join(f"- [{citation}](https://github.com/scooter7/gemini_multipdf_chat/blob/main/docs/{citation.split(' - ')[0]})" for citation in citations)
    return response


def app3():
    with st.spinner("Downloading and processing PDFs..."):
        pdf_docs = download_pdfs_from_github_3()
        if pdf_docs:
            raw_text, source_metadata = get_pdf_text(pdf_docs)
            if raw_text:
                text_chunks, chunk_metadata = get_text_chunks(raw_text, source_metadata)
                if text_chunks:
                    vector_store = load_or_create_vector_store(text_chunks, chunk_metadata)
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

    st.title("Find and engage with past proposal content")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history_3)

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

        query_chunks = chunk_query_3(prompt)
        full_response = ''
        all_citations = []

        for chunk in query_chunks:
            response = user_input_3(chunk)
            for item in response['output_text']:
                full_response += item
            all_citations.extend(response['citations'])

        modified_response = modify_response_language_3(full_response, all_citations)

        with st.chat_message("assistant"):
            st.write(modified_response)
            st.session_state.messages.append({"role": "assistant", "content": modified_response})


# Main App
def main():
    st.set_page_config(page_title="Combined Application", layout="wide")
    st.title("Combined Application")
    tab1, tab2, tab3 = st.tabs(["Tab 1: Past Proposal Q&A", "Tab 2: RFP Summarization Bot", "Tab 3: Leverage Existing Proposal Content"])

    with tab1:
        app1()

    with tab2:
        app2()

    with tab3:
        app3()


if __name__ == "__main__":
    main()
