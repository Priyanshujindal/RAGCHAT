import streamlit as st
from implementation import load_documents, split_documents, vector_store_documents, query_vector_store,combine_all_documents
import os

st.title("Document Query Application")
st.write("Upload your document and ask questions about its content.")

# File uploader with multiple file support
uploaded_files = st.file_uploader(
    "Upload Document(s)",
    type=["pdf", "docx", "txt", "csv", "xlsx", "xls"],
    accept_multiple_files=True,
    help="You can upload multiple files at once. Supported formats: PDF, DOCX, TXT, CSV, XLSX"
)

user_query = st.text_input("Enter your question about the document(s):", 
                         placeholder="What would you like to know about these documents?")

def process_documents(uploaded_files, user_query):
    """Process uploaded documents and generate a response to the user's query."""
    temp_file_paths = []
    all_documents = []
    
    try:
        # First, save all uploaded files
        for uploaded_file in uploaded_files:
            file_path = f"temp_{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            temp_file_paths.append(file_path)
        
        st.info(f"Processing {len(temp_file_paths)} uploaded file{'s' if len(temp_file_paths) > 1 else ''}...")
        
        # Process each file
        for file_path in temp_file_paths:
            try:
                documents = load_documents(file_path)
                all_documents.extend(documents)
            except Exception as e:
                st.warning(f"Skipping file {os.path.basename(file_path)} due to error: {e}")
        
        if not all_documents:
            st.error("No valid content found in the uploaded files.")
            return
            
        with st.spinner("Analyzing document content..."):
            # Count total documents/chunks processed
            total_chunks = len(all_documents)
            st.info(f"Processed {len(temp_file_paths)} file{'s' if len(temp_file_paths) > 1 else ''} with {total_chunks} text chunk{'s' if total_chunks != 1 else ''} total")
            
            combined_docs = combine_all_documents(all_documents)
            split_docs = split_documents(combined_docs)
            
            with st.spinner("Creating search index..."):
                vector_store = vector_store_documents(split_docs)
            
            with st.spinner("Generating answer..."):
                answer = query_vector_store(vector_store, user_query)
            
            st.markdown("### Answer:")
            st.write(answer)
            
    except Exception as e:
        st.error(f"An error occurred while processing your request: {str(e)}")
        
    finally:
        # Always clean up temporary files
        for path in temp_file_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                print(f"Warning: Could not remove temporary file {path}: {e}")

# Main application logic
if st.button('Submit', type="primary"):
    if not uploaded_files:
        st.warning("Please upload at least one document.")
    elif not user_query.strip():
        st.warning("Please enter a question about the document(s).")
    else:
        process_documents(uploaded_files, user_query)
        

        


            