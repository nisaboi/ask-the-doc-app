import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, WebBaseLoader
import tempfile
import pysqlite3
import sys
import requests
import validators
import os

# Fix the sqlite3 module issue
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Page configuration
st.set_page_config(page_title='🦜🔗 Secure Doc and Code App', layout="wide")

# Initialize session state variables
if 'document_list' not in st.session_state:
    st.session_state['document_list'] = []
if 'query_history' not in st.session_state:
    st.session_state['query_history'] = []

# Secure API Key Handling
openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
together_api_key = st.secrets.get("TOGETHER_API_KEY", "")

def add_to_sidebar(doc_name):
    """Update the sidebar with the new document."""
    if doc_name not in st.session_state['document_list']:
        st.session_state['document_list'].append(doc_name)

def load_document(file=None, url=None):
    """Load a document securely from a file or validated URL."""
    documents = []
    try:
        if file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()
            os.remove(temp_file_path)  # Cleanup
            add_to_sidebar(file.name)
        
        elif url is not None:
            if not validators.url(url):  # Validate URL
                st.error("Invalid URL! Please enter a valid and safe URL.")
                return []
            
            loader = WebBaseLoader(url)
            documents = loader.load()
            add_to_sidebar(url)

    except Exception as e:
        st.error(f"Failed to load document: {str(e)}")
    
    return documents

def generate_response(documents, query_text):
    """Generate a response securely."""
    try:
        if not openai_api_key:
            return "Error: Missing OpenAI API Key."

        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        db = Chroma.from_documents(texts, embeddings, persist_directory="chromadb_storage")
        db.persist()
        
        retriever = db.as_retriever(search_kwargs={"k": 3})
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(openai_api_key=openai_api_key, max_tokens=150),
            chain_type='stuff',
            retriever=retriever
        )
        return qa.run(query_text)
    except Exception as e:
        return f"Error: {str(e)}"

def generate_code(prompt):
    """Securely generate code with Together.ai"""
    try:
        if not together_api_key:
            return "Error: Missing Together.ai API Key."

        url = "https://api.together.ai/code"
        headers = {"Authorization": f"Bearer {together_api_key}"}
        data = {"prompt": prompt, "model": "code-llama"}
        
        response = requests.post(url, headers=headers, json=data, timeout=10)
        response.raise_for_status()  # Raise error for non-200 responses
        
        return response.json().get('code', "No code returned")
    except requests.RequestException as e:
        return f"Error: {str(e)}"

# Sidebar
with st.sidebar:
    st.image("https://lwfiles.mycourse.app/65a58160c1646a4dce257fac-public/a82c64f84b9bb42db4e72d0d673a50d0.png", use_column_width=True)
    
    st.write("**Loaded Documents**")
    for doc in st.session_state['document_list']:
        st.write(f"- {doc}")

# Tabs
tabs = st.tabs(["Document Q&A", "Code Generation"])

# Tab 1: Document Q&A
with tabs[0]:
    st.title("Secure Document Q&A")
    uploaded_file = st.file_uploader('Upload a PDF document', type='pdf')
    uploaded_url = st.text_input('Enter a website URL (optional)')
    documents = []

    if uploaded_file:
        documents = load_document(file=uploaded_file)
    elif uploaded_url:
        documents = load_document(url=uploaded_url)

    query_text = st.text_input('Enter your question:', placeholder='Ask about the loaded documents.', disabled=not documents)
    
    if query_text and documents:
        with st.spinner('Generating response...'):
            response = generate_response(documents, query_text)
            st.session_state['query_history'].append((query_text, response))
            st.write("**Response:**", response)

# Tab 2: Code Generation
with tabs[1]:
    st.title("Secure Code Generation")
    code_prompt = st.text_area("Enter your coding prompt:")
    
    if code_prompt:
        with st.spinner("Generating code..."):
            generated_code = generate_code(code_prompt)
            st.code(generated_code, language="python")

    
    if code_prompt:
        with st.spinner("Generating code..."):
            generated_code = generate_code(code_prompt)
            st.code(generated_code, language="python")
