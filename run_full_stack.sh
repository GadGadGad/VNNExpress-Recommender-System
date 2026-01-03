#!/bin/bash

# Kill ports if running
fuser -k 8000/tcp 2>/dev/null
fuser -k 3000/tcp 2>/dev/null

echo "🚀 Starting News RecSys Premium..."

# Add current directory to PYTHONPATH to ensure backend imports work
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Start Backend (Run from root to avoid 'app' package conflict)
echo "Starting FastAPI Backend..."
# We use backend.app.main:app so python looks for 'backend' folder first
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Start Frontend
echo "Starting Next.js Frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo "✅ App is running!"
echo "👉 Frontend: http://localhost:3000"
echo "👉 Backend Info: http://localhost:8000/api/v1"

# Wait for process to finish
wait $BACKEND_PID $FRONTEND_PID
