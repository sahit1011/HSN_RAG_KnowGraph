#!/bin/bash

echo "========================================"
echo "HSN RAG System Setup Script"
echo "========================================"
echo

# Create virtual environment
echo "1. Creating virtual environment..."
python3 -m venv hsn_env
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment"
    exit 1
fi
echo "SUCCESS: Virtual environment created"
echo

# Activate virtual environment
echo "2. Activating virtual environment..."
source hsn_env/bin/activate
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment"
    exit 1
fi
echo "SUCCESS: Virtual environment activated"
echo

# Upgrade pip
echo "3. Upgrading pip..."
python -m pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo "WARNING: Failed to upgrade pip, continuing..."
fi
echo

# Install requirements
echo "4. Installing requirements..."
if command -v uv &> /dev/null; then
    echo "INFO: Using uv package manager for faster installation..."
    uv pip install -r ../requirements.txt
    if [ $? -ne 0 ]; then
        echo "WARNING: uv failed, falling back to pip..."
        pip install -r ../requirements.txt
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to install requirements with both uv and pip"
            exit 1
        fi
    fi
else
    echo "INFO: uv not found, using pip..."
    pip install -r ../requirements.txt
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install requirements"
        exit 1
    fi
fi
echo "SUCCESS: Requirements installed"
echo

# Check if data extraction is needed
echo "5. Checking data extraction..."
if [ -f "../output/data/extraction_complete.csv" ]; then
    echo "INFO: Data extraction already completed, skipping..."
else
    echo "INFO: Running data extraction..."
    python ../src/data_processing/test_extraction.py
    if [ $? -ne 0 ]; then
        echo "ERROR: Data extraction failed"
        exit 1
    fi
    echo "SUCCESS: Data extraction completed"
fi
echo

# Check if data enhancement is needed
echo "6. Checking data enhancement..."
if [ -f "../output/data/sample_enhanced_data.csv" ]; then
    echo "INFO: Data enhancement already completed, skipping..."
else
    echo "INFO: Running data enhancement..."
    python ../src/data_processing/test_enhancement.py
    if [ $? -ne 0 ]; then
        echo "ERROR: Data enhancement failed"
        exit 1
    fi
    echo "SUCCESS: Data enhancement completed"
fi
echo

# Check if vector store needs to be built
echo "7. Checking vector store..."
if [ -f "../output/vectors/hsn_vector_config.json" ]; then
    echo "INFO: Vector store already exists, skipping..."
else
    echo "INFO: Building vector store..."
    python ../src/rag_system/vector_store.py
    if [ $? -ne 0 ]; then
        echo "ERROR: Vector store creation failed"
        exit 1
    fi
    echo "SUCCESS: Vector store created"
fi
echo

# Check if knowledge graph needs to be built
echo "8. Checking knowledge graph..."
if [ -f "../output/models/hsn_knowledge_graph.pkl" ]; then
    echo "INFO: Knowledge graph already exists, skipping..."
else
    echo "INFO: Building knowledge graph..."
    python ../src/knowledge_graph/graph_implementation.py
    if [ $? -ne 0 ]; then
        echo "ERROR: Knowledge graph creation failed"
        exit 1
    fi
    echo "SUCCESS: Knowledge graph created"
fi
echo

# Test system initialization
echo "9. Testing system initialization..."
python -c "
from src.rag_system.hsn_rag_system import HSN_RAG_System
system = HSN_RAG_System()
if system.initialize_system():
    print('SUCCESS: System initialization test passed')
    metrics = system.get_system_metrics()
    print(f'  Total queries: {metrics[\"total_queries\"]}')
    print(f'  Success rate: {metrics.get(\"success_rate\", 0):.1%}')
else:
    print('ERROR: System initialization test failed')
    exit(1)
"
if [ $? -ne 0 ]; then
    echo "ERROR: System initialization test failed"
    exit 1
fi
echo

echo "========================================"
echo "SETUP COMPLETE!"
echo "========================================"
echo
echo "Your HSN RAG System is now ready!"
echo
echo "To run the system:"
echo "1. Activate the environment: source hsn_env/bin/activate"
echo "2. Run the web app: streamlit run ../app.py"
echo "   OR run CLI: python ../main.py"
echo
echo "For inference and testing, use run.sh"
echo