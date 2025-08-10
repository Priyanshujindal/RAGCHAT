# Core LangChain components
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document
import os
from typing import List
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

def load_documents(file_path: str) -> List[Document]:
    """
    Load documents from a file with support for multiple formats.
    
    Args:
        file_path: Path to the file to load (PDF, DOCX, TXT, CSV, XLSX)
        
    Returns:
        List of Document objects with metadata
        
    Raises:
        ValueError: If file format is unsupported or no content is found
    """
    file_name = os.path.basename(file_path)
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        # Select appropriate loader based on file extension
        if file_ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_ext == '.docx':
            loader = Docx2txtLoader(file_path)
        elif file_ext == '.txt':
            loader = TextLoader(file_path, autodetect_encoding=True)
        elif file_ext == '.csv':
            loader = CSVLoader(file_path)
        elif file_ext in ('.xlsx', '.xls'):
            loader = UnstructuredExcelLoader(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
            
        # Load and validate documents
        documents = loader.load()
        
        if not documents:
            raise ValueError(f"No content found in {file_name}")
            
        # Add metadata to documents
        for i, doc in enumerate(documents):
            if not hasattr(doc, 'metadata') or not doc.metadata:
                doc.metadata = {}
            doc.metadata.update({
                'source_file': file_name,
                'page': i + 1,
                'total_pages': len(documents)
            })
        
        return documents
        
    except Exception as e:
        raise ValueError(f"Error processing {file_name}: {str(e)}")

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks for processing.
    
    Args:
        documents: List of Document objects to split
        
    Returns:
        List of split Document objects
        
    Raises:
        ValueError: If no documents are provided or if documents are invalid
    """
    if not documents:
        raise ValueError("No documents provided to split")
    
    if not all(isinstance(doc, Document) for doc in documents):
        raise ValueError("All items must be Document objects")
    
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True
        )
        
        split_docs = text_splitter.split_documents(documents)
        return split_docs
        
    except Exception as e:
        raise ValueError(f"Failed to split documents: {str(e)}")

def combine_all_documents(documents_list: List[Document]) -> List[Document]:
    """
    Combine multiple documents into a single document with combined metadata.
    
    Args:
        documents_list: List of Document objects to combine
        
    Returns:
        List containing a single combined Document
        
    Raises:
        ValueError: If no valid documents are provided
    """
    if not documents_list:
        raise ValueError("No documents provided to combine")
    
    try:
        combined_text = ""
        combined_metadata = {
            "source": "combined_documents",
            "total_original_documents": len(documents_list),
            "content_pages": 0
        }
        
        # Track sources for better metadata
        sources = set()
        
        for doc in documents_list:
            if not doc.page_content.strip():
                continue
                
            combined_text += doc.page_content.strip() + "\n\n"
            
            # Preserve source information
            if hasattr(doc, 'metadata'):
                if 'source_file' in doc.metadata:
                    sources.add(doc.metadata['source_file'])
                
                # Update page count if available
                if 'page' in doc.metadata and 'total_pages' in doc.metadata:
                    combined_metadata['content_pages'] = max(
                        combined_metadata.get('content_pages', 0),
                        doc.metadata['total_pages']
                    )
        
        if not combined_text.strip():
            raise ValueError("No valid content found in any of the documents")
        
        # Add source information to metadata
        if sources:
            combined_metadata['sources'] = list(sources)
        
        return [Document(
            page_content=combined_text.strip(),
            metadata=combined_metadata
        )]
        
    except Exception as e:
        raise ValueError(f"Failed to combine documents: {str(e)}")

def vector_store_documents(split_docs: List[Document]):
    """
    Create a vector store from document chunks.
    
    Args:
        split_docs: List of Document objects to index
        
    Returns:
        FAISS vector store containing the document embeddings
        
    Raises:
        ValueError: If no valid documents are provided
        RuntimeError: If vector store creation fails
    """
    if not split_docs:
        raise ValueError("No document chunks provided for vector store creation")
    
    try:
        # Use a pre-trained sentence transformer model for embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU is available
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Create and return FAISS vector store
        vector_store = FAISS.from_documents(documents=split_docs, embedding=embeddings)
        return vector_store
        
    except Exception as e:
        raise RuntimeError(f"Failed to create vector store: {str(e)}")

def query_vector_store(vector_store, query: str) -> str:
    """
    Query the vector store and generate a response using Google's Gemini model.
    
    Args:
        vector_store: The FAISS vector store containing document embeddings
        query: The user's question or query
        
    Returns:
        str: The generated response based on the document context
        
    Raises:
        ValueError: If the API key is missing or if the query is empty
        RuntimeError: If there's an error during the query or response generation
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
        
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    
    try:
        # Search for relevant document chunks
        results = vector_store.similarity_search(query=query, k=3)  # Increased to 3 for better context
        if not results:
            return "I couldn't find any relevant information in the documents to answer your question."
            
        # Combine the content of the top results
        docs_page_content = "\n\n".join(
            f"[Document {i+1}]\n{doc.page_content}" 
            for i, doc in enumerate(results)
        )
        
        # Initialize the language model
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=google_api_key,
            temperature=0.3,  # Lower temperature for more focused answers
            max_output_tokens=1000
        )
        
        # Create a more detailed prompt template
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are an intelligent document assistant. Use the following context retrieved from the document(s) 
            to answer the user's question accurately and concisely.

            DOCUMENT CONTEXT:
            {context}

            USER QUESTION: {question}

            INSTRUCTIONS:
            1. Answer based SOLELY on the information provided in the context
            2. Be accurate, concise, and to the point
            3. If the context doesn't contain enough information to answer the question, 
               say "I cannot find sufficient information in the provided documents to answer this question."
            4. If the question is unclear or too broad, ask for clarification
            5. If referring to specific parts of the document, mention the document number in square brackets (e.g., [Document 1])
            6. If the answer requires combining information from multiple documents, clearly indicate this

            ANSWER:
            """
        )
        
        # Create and run the LLM chain
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(context=docs_page_content, question=query)
        
        # Clean up the response
        response = response.strip()
        response = " ".join(response.split())  # Normalize whitespace
        
        return response
        
    except Exception as e:
        return f"I encountered an error while processing your request: {str(e)}"