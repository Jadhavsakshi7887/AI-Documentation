@echo off
echo ========================================
echo   RAG Bot - FastAPI + Streamlit
echo ========================================
echo.
echo Starting both backend and frontend...
echo.
echo Backend will run on: http://localhost:8000
echo Frontend will run on: http://localhost:8501
echo.
echo Press Ctrl+C to stop both servers
echo.

start "FastAPI Backend" cmd /k "python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000"
timeout /t 3 /nobreak >nul
start "Streamlit Frontend" cmd /k "streamlit run frontend/app.py --server.port 8501"

echo.
echo Both servers are starting...
echo Check the opened windows for status.
pause

