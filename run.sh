#!/bin/bash

echo "========================================"
echo "HSN RAG System - Run Script"
echo "========================================"
echo

# Check if virtual environment exists
if [ ! -f "hsn_env/bin/activate" ]; then
    echo "ERROR: Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source hsn_env/bin/activate
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment"
    exit 1
fi
echo "SUCCESS: Virtual environment activated"
echo

# Check if required files exist
if [ ! -f "output/vectors/hsn_vector_config.json" ]; then
    echo "ERROR: Vector store not found. Please run setup.sh first."
    exit 1
fi

if [ ! -f "output/models/hsn_knowledge_graph.pkl" ]; then
    echo "ERROR: Knowledge graph not found. Please run setup.sh first."
    exit 1
fi

# Run Streamlit app
echo "Starting HSN RAG System web interface..."
echo
echo "The web interface will open in your default browser."
echo "You can also access it at: http://localhost:8501"
echo
echo "Press Ctrl+C to stop the server."
echo

streamlit run app.py

# Deactivate environment when done
deactivate

echo
echo "HSN RAG System stopped."