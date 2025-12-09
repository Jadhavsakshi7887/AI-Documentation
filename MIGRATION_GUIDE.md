# Migration Guide: Django to FastAPI + Streamlit

This guide explains the migration from Django to FastAPI + Streamlit and how to use the new architecture.

## What Changed

### Backend Migration (Django → FastAPI)

**Before (Django):**
- Django views handling HTTP requests
- Django templates for HTML rendering
- Django URL routing
- Session-based file storage

**After (FastAPI):**
- FastAPI REST API endpoints
- JSON-based communication
- Async file handling
- Stateless API design

### Frontend Migration (Django Templates → Streamlit)

**Before (Django):**
- HTML templates with Django template language
- Server-side rendering
- Form submissions via POST requests

**After (Streamlit):**
- Python-based UI components
- Reactive interface
- Modern, responsive design
- Real-time updates

## Key Differences

### 1. File Upload Handling

**Django:**
```python
uploaded_file = request.FILES.get("document")
file_path = os.path.join(temp_dir, uploaded_file.name)
```

**FastAPI:**
```python
file: UploadFile = File(...)
file_path = TEMP_UPLOADS_DIR / file.filename
```

### 2. Query Processing

**Django:**
- Synchronous request handling
- Session-based state management

**FastAPI:**
- Async/await support
- Stateless API calls
- JSON request/response

### 3. UI Rendering

**Django:**
- HTML templates with Jinja2
- Server-side rendering

**Streamlit:**
- Python widgets and components
- Client-side interactivity
- Automatic UI updates

## Migration Checklist

- [x] FastAPI backend created with all endpoints
- [x] Streamlit frontend with modern UI
- [x] File upload functionality preserved
- [x] RAG query functionality preserved
- [x] Vector store management maintained
- [x] Configuration system implemented
- [x] Error handling improved
- [x] Documentation created

## Running the Migrated Application

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start backend:**
   ```bash
   python -m uvicorn backend.main:app --reload
   ```

3. **Start frontend:**
   ```bash
   streamlit run frontend/app.py
   ```

4. **Or use the convenience scripts:**
   - Windows: `start.bat`
   - Linux/Mac: `bash start.sh`

## API Endpoints

### POST /upload
Upload and process a document file.

**Request:**
- `file`: Multipart file upload

**Response:**
```json
{
  "success": true,
  "message": "File uploaded and processed successfully",
  "file_name": "document.pdf",
  "file_path": "/path/to/file"
}
```

### POST /query
Query a document using RAG.

**Request:**
```json
{
  "question": "What is the main topic?",
  "file_path": "/path/to/file"
}
```

**Response:**
```json
{
  "answer": "The main topic is...",
  "success": true
}
```

### POST /upload-and-query
Combined upload and query in one request.

**Request:**
- `file`: Multipart file upload
- `question`: Form data string

**Response:**
```json
{
  "success": true,
  "file_name": "document.pdf",
  "answer": "The answer is..."
}
```

## Configuration

All configuration is centralized in `config.py`:

- API settings (host, port, base URL)
- File paths (uploads, vector stores)
- Model settings (embedding, RAG)
- Chunking parameters

## Benefits of Migration

1. **Performance**: FastAPI is faster than Django for API endpoints
2. **Modern UI**: Streamlit provides a modern, interactive interface
3. **Separation of Concerns**: Clear separation between backend and frontend
4. **Scalability**: Easier to scale backend and frontend independently
5. **Developer Experience**: Better tooling and debugging
6. **API Documentation**: Automatic OpenAPI/Swagger docs with FastAPI

## Troubleshooting

### Backend not starting
- Check Python version (3.11 required)
- Verify all dependencies installed
- Check port 8000 is available

### Frontend not connecting
- Ensure backend is running
- Check API_BASE_URL in frontend/app.py
- Verify CORS settings in backend

### File upload errors
- Check temp_uploads directory exists
- Verify file permissions
- Check file size limits

### Vector store errors
- Ensure chroma_vectorstore directory exists
- Check disk space
- Verify embedding model can download

## Next Steps

1. Test all functionality
2. Customize UI in `frontend/app.py`
3. Add authentication if needed
4. Deploy backend and frontend separately
5. Set up environment variables for production

