"""
HSN RAG System Configuration
Centralized configuration for the modular HSN classification system
"""

import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, will use system environment variables

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Source directory
SRC_DIR = PROJECT_ROOT / "src"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "output"
DATA_DIR = OUTPUT_DIR / "data"
MODELS_DIR = OUTPUT_DIR / "models"
VECTORS_DIR = OUTPUT_DIR / "vectors"
LOGS_DIR = OUTPUT_DIR / "logs"
REPORTS_DIR = OUTPUT_DIR / "reports"

# File paths
ENHANCED_DATA_PATH = DATA_DIR / "sample_enhanced_data.csv"
SCHEMA_FILE = DATA_DIR / "graph_schema.json"
GRAPH_FILE = MODELS_DIR / "hsn_knowledge_graph.pkl"
VECTOR_CONFIG = VECTORS_DIR / "hsn_vector_config.json"

# Model configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2

# System configuration
MAX_QUERY_LENGTH = 500
DEFAULT_TOP_K = 5
SIMILARITY_THRESHOLD = 0.3
MAX_RESPONSE_TIME = 5.0  # seconds

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Test configuration
TEST_QUERIES = [
    "What is the HSN code for natural rubber latex?",
    "HSN code for prevulcanised rubber",
    "Rubber products classification",
    "Tell me about HSN 40011010",
    "Similar products to natural rubber latex"
]

# Create directories if they don't exist
def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        OUTPUT_DIR, DATA_DIR, MODELS_DIR, VECTORS_DIR,
        LOGS_DIR, REPORTS_DIR, SRC_DIR
    ]

    for directory in directories:
        directory.mkdir(exist_ok=True, parents=True)

# Environment variables (optional)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
STREAMLIT_SERVER_PORT = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
STREAMLIT_SERVER_ADDRESS = os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost")

# LLM Configuration
LLM_CONFIG = {
    "provider": "openrouter",
    "model": "deepseek/deepseek-r1-0528:free",  # DeepSeek R1 free model
    "temperature": 0.3,
    "max_tokens": 1000,
    "api_base": "https://openrouter.ai/api/v1"
}

# RAG Mode Configuration
RAG_MODES = {
    "rule_based": {
        "name": "Rule-based RAG",
        "description": "Traditional rule-based approach using predefined schemas and similarity search",
        "uses_llm": False
    },
    "llm_enhanced": {
        "name": "LLM-Enhanced RAG",
        "description": "Combines rule-based retrieval with LLM-powered graph generation and response",
        "uses_llm": True
    },
    "llm_only": {
        "name": "LLM-Only RAG",
        "description": "Fully LLM-driven approach for graph generation and response generation",
        "uses_llm": True
    }
}

# Performance settings
BATCH_SIZE = 32
MAX_WORKERS = 4
CACHE_SIZE = 1000

# Export policy mappings (example)
EXPORT_POLICIES = {
    "free": "No restrictions",
    "restricted": "Export restricted",
    "prohibited": "Export prohibited",
    "licensed": "Requires export license"
}

# System metadata
SYSTEM_INFO = {
    "name": "HSN RAG Classification System",
    "version": "1.0.0",
    "description": "Intelligent HSN code classification using RAG and Knowledge Graphs",
    "author": "HSN RAG Team",
    "modules": [
        "data_processing",
        "knowledge_graph",
        "rag_system",
        "utils"
    ]
}