from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document
import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
# Load environment variables
load_dotenv()

def load_documents(file_path: str):
    if file_path.endswith('.pdf'):
       loader= PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path)
    elif file_path.endswith('.csv'):
        loader = CSVLoader(file_path)
    elif file_path.endswith('.xlsx'):
        loader = UnstructuredExcelLoader(file_path)
    else:
        raise ValueError("Unsupported file format")
    documents = loader.load()
    if not documents:
        raise ValueError(f"No content found in {file_path}")
    total_content = sum(len(doc.page_content.strip()) for doc in documents)
    if total_content == 0:
        raise ValueError(f"No content found in {file_path}")   
    return documents

def split_documents(documents):
    if not documents:
        raise ValueError("No documents provided to split")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)
    return split_docs

def combine_all_documents(documents_list):
    """Combine all documents into one large document before splitting"""
    combined_text = ""
    combined_metadata = {"source": "multiple_documents"}
    
    for doc in documents_list:
        combined_text += doc.page_content + "\n\n"
        if hasattr(doc, 'metadata'):
            combined_metadata.update(doc.metadata)
    
    # Create one large document
    combined_document = Document(
        page_content=combined_text,
        metadata=combined_metadata
    )
    return [combined_document]

def vector_store_documents(split_docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(split_docs, embeddings)
    return vector_store

def query_vector_store(vector_store, query):
    # Get API key from environment variable
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    
    results = vector_store.similarity_search(query=query, k=2)
    docs_page_content = " ".join([doc.page_content for doc in results])
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=google_api_key
    )
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are an intelligent document assistant. Use the following context retrieved from the document(s) to answer the user's question accurately and comprehensively.

        Context: {context}

        Question: {question}

        Instructions:
        - Answer based solely on the information provided in the context
        - If the context doesn't contain enough information to answer the question, say "I cannot find sufficient information in the provided context to answer this question"
        - Quote relevant parts of the context when appropriate
        - Be precise and factual
        - If multiple documents are referenced in the context, clearly distinguish between them

        Answer:"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(context=docs_page_content, question=query)
    response = response.replace("\n", " ")
    return response