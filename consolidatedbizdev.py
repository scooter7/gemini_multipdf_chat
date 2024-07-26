import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Multi-App Streamlit Application")

# Define each app as a function
def app1():
    import os
    from PyPDF2 import PdfReader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    import google.generativeai as genai
    from langchain.vectorstores import FAISS
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.chains.question_answering import load_qa_chain
    from langchain.prompts import PromptTemplate
    from dotenv import load_dotenv

    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    def get_pdf_text(pdf_docs):
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def get_text_chunks(text):
        splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = splitter.split_text(text)
        return chunks

    def get_vector_store(chunks):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")

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
        st.session_state.messages = [{"role": "assistant", "content": "Upload an RFP and ask about scope, due dates, anything you'd like..."}]

    def user_input(user_question):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response

    st.title("RFP Summarization Bot")
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Upload an RFP and ask about scope, due dates, anything you'd like..."}]

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
                full_response = ''.join(response['output_text'])
                st.write(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

def app2():
    import os
    import requests
    from PyPDF2 import PdfReader
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.embeddings import OpenAIEmbeddings
    import streamlit as st
    from langchain_community.vectorstores import FAISS
    from dotenv import load_dotenv
    from langchain.schema import Document

    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    def get_pdfs_from_github():
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

    def get_text_chunks(text, metadata, chunk_size=2000, chunk_overlap=500):
        splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = []
        chunk_metadata = []
        for i, page_text in enumerate(text):
            page_chunks = splitter.split_text(page_text)
            chunks.extend(page_chunks)
            chunk_metadata.extend([metadata[i]] * len(page_chunks))
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

    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "Find and engage with past proposal questions and answers."}]

    def user_input(user_question, max_retries=5, delay=2):
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

    def chunk_query(query, chunk_size=200):
        return [query[i:i+chunk_size] for i in range(0, len(query), chunk_size)]

    def modify_response_language(original_response, citations):
        response = original_response
        if citations:
            response += "\n\nSources:\n" + "\n".join(f"- [{citation}](https://github.com/scooter7/gemini_multipdf_chat/blob/main/qna/{citation.split(' - ')[0]})" for citation in citations)
        return response

    st.title("Pastr Proposal Q&A")
    with st.spinner("Downloading and processing PDFs..."):
        pdf_docs = download_pdfs_from_github()
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

    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Find and engage with past proposal questions and answers."}]

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

def app3():
    import os
    import requests
    from PyPDF2 import PdfReader
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.embeddings import OpenAIEmbeddings
    import streamlit as st
    from langchain_community.vectorstores import FAISS
    from openai import OpenAI
    from dotenv import load_dotenv
    from langchain.schema import Document

    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    client = OpenAI(api_key=OPENAI_API_KEY)

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
        splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = []
        chunk_metadata = []
        for i, page_text in enumerate(text):
            page_chunks = splitter.split_text(page_text)
            chunks.extend(page_chunks)
            chunk_metadata.extend([metadata[i]] * len(page_chunks))
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

    def get_conversational_chain():
        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        model = "gpt-4o-mini"
        return model

    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "Find and engage with past Carnegie proposals."}]

    def user_input(user_question, max_retries=5, delay=2):
        vector_store = load_or_create_vector_store([], [])
        if not vector_store:
            st.error("Failed to load or create the vector store.")
            return {"output_text": ["Failed to load or create the vector store."]}

        try:
            docs = vector_store.similarity_search(user_question)
        except Exception as e:
            st.error(f"Failed to perform similarity search: {e}")
            return {"output_text": [f"Failed to perform similarity search: {e}"]}

        model = get_conversational_chain()
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

    def chunk_query(query, chunk_size=200):
        return [query[i:i+chunk_size] for i in range(0, len(query), chunk_size)]

    def modify_response_language(original_response, citations):
    response = original_response.replace(" they ", " we ")
    response = response.replace("They ", "We ")
    response = response.replace(" their ", " our ")
    response = response.replace("Their ", "Our ")
    response = response.replace(" them ", " us ")
    response = response.replace("Them ", "Us ")
    if citations:
        response += "\n\nSources:\n" + "\n".join(f"- [{citation}](https://github.com/scooter7/gemini_multipdf_chat/blob/main/docs/{citation.split(' - ')[0]})" for citation in citations)
    return response

    st.title("Leverage Existing Proposal Content")
    with st.spinner("Downloading and processing PDFs..."):
        pdf_docs = download_pdfs_from_github()
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

    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Find and engage with past Carnegie proposals."}]

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

# Main function to run the Streamlit app
def main():
    st.title("Multi-App Streamlit Application")

    # Define the tab menu
    selected = option_menu(
        menu_title=None,
        options=["App1", "App2", "App3"],
        icons=["house", "gear", "list-task"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )

    # Display the selected tab
    if selected == "App1":
        app1()
    elif selected == "App2":
        app2()
    elif selected == "App3":
        app3()

if __name__ == "__main__":
    main()
