# RAG Bot - FastAPI + Streamlit Migration

A modern RAG (Retrieval-Augmented Generation) chatbot application migrated from Django to FastAPI backend and Streamlit frontend.

## Features

- 📄 **Document Upload**: Support for PDF, DOC, DOCX, and TXT files
- 🤖 **AI-Powered Q&A**: Ask questions about uploaded documents using RAG
- 🔒 **Secure Processing**: Documents processed securely with no data retention
- ⚡ **Fast & Efficient**: Optimized vector store with persistent storage
- 🎨 **Modern UI**: Beautiful, responsive Streamlit interface

## Project Structure

```
mysite/
├── backend/
│   └── main.py              # FastAPI backend application
├── frontend/
│   └── app.py               # Streamlit frontend application
├── config.py                # Configuration settings
├── vectorstore_manager.py   # Vector store management
├── rag_query.py            # RAG query processing
├── embedding.py            # Embedding model wrapper
├── requirements.txt        # Python dependencies
├── temp_uploads/           # Temporary file uploads
└── chroma_vectorstore/      # Persistent vector store data
```

## Installation

1. **Create and activate virtual environment** (Python 3.11):
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables** (optional):
Create a `.env` file or set environment variables:
```bash
API_BASE_URL=http://localhost:8000
RAG_API_KEY=your_api_key_here
PERSIST_FOLDER=./chroma_vectorstore
EMBEDDING_CACHE_FOLDER=./huggingface_models
```

## Running the Application

### Option 1: Run Backend and Frontend Separately

1. **Start FastAPI backend**:
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

2. **Start Streamlit frontend** (in a new terminal):
```bash
cd frontend
streamlit run app.py --server.port 8501
```

### Option 2: Run from Project Root

1. **Start FastAPI backend**:
```bash
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

2. **Start Streamlit frontend** (in a new terminal):
```bash
streamlit run frontend/app.py --server.port 8501
```

## Usage

1. Open your browser and navigate to `http://localhost:8501`
2. Upload a document (PDF, DOC, DOCX, or TXT)
3. Wait for the document to be processed
4. Ask questions about the document
5. Get AI-powered answers based on the document content

## API Endpoints

### FastAPI Backend (`http://localhost:8000`)

- `GET /` - Health check
- `POST /upload` - Upload and process a document
- `POST /query` - Query a document using RAG
- `POST /upload-and-query` - Combined upload and query endpoint

### API Documentation

Once the backend is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Configuration

Edit `config.py` to customize:
- API endpoints and ports
- Vector store persistence location
- Embedding model and cache folder
- RAG API credentials
- Chunking parameters

## Python Version

This project is compatible with **Python 3.11**.

## Dependencies

Key dependencies:
- FastAPI - Modern web framework for building APIs
- Streamlit - Frontend framework for data apps
- LangChain - Framework for LLM applications
- ChromaDB - Vector database for embeddings
- Sentence Transformers - Embedding models
- Unstructured - Document processing

See `requirements.txt` for the complete list.

## Migration Notes

This project has been migrated from Django to FastAPI + Streamlit:
- ✅ All functionality preserved
- ✅ Data handling maintained
- ✅ Python 3.11 compatibility ensured
- ✅ Modern UI with Streamlit
- ✅ RESTful API with FastAPI

## Troubleshooting

1. **API not connecting**: Ensure the FastAPI backend is running on port 8000
2. **File upload errors**: Check that `temp_uploads/` directory exists and is writable
3. **Vector store errors**: Ensure `chroma_vectorstore/` directory exists
4. **Embedding model errors**: The model will download automatically on first use

## License

This project is for educational and development purposes.

