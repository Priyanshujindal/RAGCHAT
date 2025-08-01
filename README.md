# RAG Chat - Document Q&A Application

A powerful document question-answering application built with Streamlit, LangChain, and Google Gemini AI. Upload documents and ask questions to get intelligent answers based on the document content.

## 🚀 Features

- **Multi-format Document Support**: Upload PDF (text-based only), DOCX, TXT, CSV, and XLSX files
- **Text Extraction**: Extracts text from supported document types
- **Advanced RAG Pipeline**: Uses FAISS vector store with HuggingFace embeddings
- **Google Gemini Integration**: Powered by Google's latest AI model
- **Streamlit Web Interface**: Clean, user-friendly web application
- **Multi-file Processing**: Upload and process multiple documents simultaneously
- **Real-time Processing**: See live progress updates during document processing

> **System Requirements:**  
> For full document processing capabilities, install these system dependencies:
> ```bash
> # Linux
> sudo apt install libmagic-dev poppler-utils
> 
> # macOS
> brew install libmagic poppler
> ```

## 📋 Prerequisites

- Python 3.8 or higher
- Google API Key for Gemini AI
- System dependencies (see above)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd RAG_LangChain