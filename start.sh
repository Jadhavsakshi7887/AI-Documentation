#!/bin/bash

echo "========================================"
echo "  RAG Bot - FastAPI + Streamlit"
echo "========================================"
echo ""
echo "Starting both backend and frontend..."
echo ""
echo "Backend will run on: http://localhost:8000"
echo "Frontend will run on: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Start backend in background
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait a bit for backend to start
sleep 3

# Start frontend
streamlit run frontend/app.py --server.port 8501 &
FRONTEND_PID=$!

echo ""
echo "Both servers are starting..."
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for user interrupt
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT TERM
wait

