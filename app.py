import streamlit as st
from implementation import load_documents, split_documents, vector_store_documents, query_vector_store,combine_all_documents
import os

st.title("Document Query Application")
st.write("Upload your document and ask questions about its content.")
uploaded_files=st.file_uploader(
    "Upload Document",
    type=["pdf", "docx", "txt", "csv", "xlsx"],
    accept_multiple_files=True
)
user_query=st.text_input("Enter your question about the document(s):")
if st.button('Submit'):
    if uploaded_files and user_query:
        temp_file_paths = []
        for uploaded_file in uploaded_files:
            file_path= f"temp_{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            temp_file_paths.append(file_path)
        all_documents=[]
        for file_path in temp_file_paths:
            try:
                documents=load_documents(file_path)
                all_documents.extend(documents)
            except Exception as e:
                st.warning(f"Skipping file {file_path} due to error: {e}")
        if not all_documents:
                st.error("No valid documents loaded.")
        else:
             with st.spinner("Combining documents..."):
                    combined_docs =combine_all_documents(all_documents)
                    st.info(f"Combined {len(all_documents)} documents into one cohesive text")
             with st.spinner("Splitting combined document..."):
                    split_docs = split_documents(combined_docs)
                    st.info(f"Created {len(split_docs)} chunks from combined document")
             with st.spinner("Creating vector store..."):
                    vector_store = vector_store_documents(split_docs)
             with st.spinner("Generating answer..."):
                    answer = query_vector_store(vector_store,user_query)
             st.markdown("### Answer:")
             st.write(answer)
            # Clean up temporary files
             for path in temp_file_paths:
                try:
                    os.remove(path)
                except Exception:
                        pass
        

        


            