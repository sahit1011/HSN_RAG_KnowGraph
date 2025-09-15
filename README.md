# HSN RAG Classification System

A modular, intelligent HSN (Harmonized System Nomenclature) code classification system using Retrieval-Augmented Generation (RAG) and Knowledge Graphs.

## 🚀 Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Intelligent Classification**: Combines vector search, knowledge graphs, and natural language processing
- **Web Interface**: Streamlit-based chat interface for easy testing
- **Command Line Interface**: Full CLI support for batch processing and automation
- **Unified Setup Scripts**: Cross-platform setup scripts with fast dependency installation
- **Comprehensive Testing**: Built-in test suite with performance monitoring
- **Scalable Design**: Production-ready architecture with proper error handling
- **Fast Installation**: UV package manager support for 10-100x faster dependency installation

## 📁 Project Structure

```
hsn_rag/
├── src/                          # Source code
│   ├── data_processing/          # Data extraction and enhancement
│   │   ├── test_extraction.py
│   │   ├── test_enhancement.py
│   │   └── __init__.py
│   ├── knowledge_graph/          # Graph design and implementation
│   │   ├── graph_design.py
│   │   ├── graph_implementation.py
│   │   ├── graph_visualization.py
│   │   └── __init__.py
│   ├── rag_system/               # RAG components
│   │   ├── vector_store.py
│   │   ├── query_processor.py
│   │   ├── disambiguation_engine.py
│   │   ├── hsn_rag_system.py
│   │   └── __init__.py
│   ├── utils/                    # Shared utilities
│   │   ├── llm_client.py
│   │   └── __init__.py
│   └── __init__.py
├── output/                       # Generated data and models (created during setup)
│   ├── data/                     # Processed data files
│   ├── models/                   # Trained models
│   ├── vectors/                  # Vector embeddings
│   ├── logs/                     # System logs
│   └── reports/                  # Performance reports
├── setup.bat                     # Windows setup script
├── setup.sh                      # Linux/macOS setup script
├── run.bat                       # Windows run script
├── run.sh                        # Linux/macOS run script
├── main.py                       # CLI entry point
├── app.py                        # Streamlit web app
├── config.py                     # Configuration settings
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
├── plan.md                       # Project documentation
├── docs/                         # Additional documentation
└── README.md                     # This file
```

## 🛠️ Quick Start

### For Recruiters/Reviewers (Recommended)

1. **Clone the repository**:
    ```bash
    git clone https://github.com/sahit1011/HSN_RAG_KnowGraph.git
    cd HSN_RAG_KnowGraph
    ```

2. **Set up environment variables** (optional but recommended):
    ```bash
    cp .env.example .env
    # Edit .env with your API keys and preferences
    ```

3. **Run the automated setup**:
    ```bash
    # Windows
    setup.bat

    # Linux/macOS
    chmod +x setup.sh
    ./setup.sh
    ```

3. **Launch the application**:
    ```bash
    # Windows
    run.bat

    # Linux/macOS
    chmod +x run.sh
    ./run.sh
    ```

That's it! The system will automatically:
- Create a virtual environment
- Install all dependencies (using UV for speed)
- Process the data and build models
- Launch the Streamlit web interface

### Manual Installation (Advanced Users)

If you prefer manual setup:

1. **Clone and navigate**:
    ```bash
    git clone https://github.com/sahit1011/HSN_RAG_KnowGraph.git
    cd HSN_RAG_KnowGraph
    ```

2. **Create virtual environment**:
    ```bash
    python -m venv hsn_env
    source hsn_env/bin/activate  # Linux/macOS
    # or
    hsn_env\Scripts\activate     # Windows
    ```

3. **Install dependencies**:
    ```bash
    # Fast installation with UV (recommended)
    uv pip install -r requirements.txt

    # Or fallback to pip
    pip install -r requirements.txt
    ```

4. **Set up environment variables** (optional):
    ```bash
    export OPENROUTER_API_KEY="your-openrouter-api-key-here"
    # Or optionally use OpenAI
    export OPENAI_API_KEY="your-openai-api-key-here"
    ```

## 🚀 Usage

### Web Interface (Recommended)

After running the setup scripts, launch the web application:

```bash
# Windows
run.bat

# Linux/macOS
./run.sh
```

Navigate to `http://localhost:8501` to access the interactive interface.

### Manual Web Interface Launch

If you prefer manual launch:

```bash
# Activate virtual environment first
source hsn_env/bin/activate  # Linux/macOS
# or
hsn_env\Scripts\activate     # Windows

# Then run Streamlit
streamlit run app.py
```

### Command Line Interface

#### Single Query
```bash
python main.py --query "What is the HSN code for natural rubber latex?"
```

#### Batch Processing
```bash
# Create a file with queries (one per line)
echo -e "What is the HSN code for natural rubber latex?\nHSN code for prevulcanised rubber" > queries.txt

# Process batch
python main.py --batch queries.txt
```

#### System Testing
```bash
# Run comprehensive tests
python main.py --test

# Show system metrics
python main.py --metrics
```

#### Interactive Mode
```bash
python main.py
```

### Python API

```python
from src.rag_system.hsn_rag_system import HSN_RAG_System

# Initialize system
system = HSN_RAG_System()
system.initialize_system()

# Classify a product
result = system.classify_product("natural rubber latex")
print(f"HSN Code: {result.hsn_code}")
print(f"Confidence: {result.confidence:.1%}")

# Batch processing
queries = ["query1", "query2", "query3"]
results = system.batch_classify(queries)
```

## ⚙️ Setup Scripts Overview

The automated setup scripts handle the complete system initialization:

### What `setup.bat`/`setup.sh` Does:

1. **Environment Setup**:
   - Creates Python virtual environment (`hsn_env/`)
   - Activates the environment
   - Upgrades pip

2. **Dependency Installation**:
   - Uses UV package manager for fast installation (10-100x faster than pip)
   - Falls back to pip if UV is not available
   - Installs all required packages from `requirements.txt`

3. **Data Processing Pipeline**:
    - **Data Extraction**: Extracts HSN data from PDF documents
    - **Data Enhancement**: Processes and structures the data
    - **Graph Schema Design**: Creates graph nodes, relationships, and schema files
    - **Vector Store Creation**: Builds FAISS vector embeddings
    - **Knowledge Graph Creation**: Constructs NetworkX knowledge graph from schema

4. **System Validation**:
   - Tests system initialization
   - Verifies all components are working
   - Provides setup summary

### What `run.bat`/`run.sh` Does:

1. **Environment Activation**: Activates the virtual environment
2. **System Validation**: Checks if all required files exist
3. **Web App Launch**: Starts Streamlit application
4. **Browser Access**: Opens `http://localhost:8501`

### Generated Files Structure:

```
output/
├── data/
│   ├── extraction_complete.csv      # Raw extracted data
│   ├── sample_enhanced_data.csv     # Processed data
│   ├── graph_nodes.json             # Graph nodes data (JSON)
│   ├── graph_relationships.json     # Graph relationships data (JSON)
│   ├── graph_schema.json            # Complete graph schema definition
│   ├── graph_nodes.csv              # Graph nodes data (CSV)
│   ├── graph_relationships.csv      # Graph relationships data (CSV)
│   └── graph_schema_documentation.md # Graph documentation
├── vectors/
│   ├── hsn_faiss_index.idx         # FAISS index
│   ├── hsn_embeddings.npy          # Vector embeddings
│   ├── hsn_vector_config.json      # Vector store config
│   └── hsn_vector_data.pkl         # Vector metadata
├── models/
│   └── hsn_knowledge_graph.pkl     # Knowledge graph (NetworkX)
├── logs/
│   └── hsn_rag_system.log          # System logs
└── reports/
    ├── performance_report_*.json   # Performance metrics
    └── system_test_*.json         # Test results
```

## 📊 System Architecture

### Core Components

1. **Data Processing Module** (`src/data_processing/`)
   - PDF extraction from Trade Notice documents
   - Data cleaning and enhancement
   - Hierarchical structure creation

2. **Knowledge Graph Module** (`src/knowledge_graph/`)
   - Graph schema design
   - NetworkX-based implementation
   - Interactive visualizations

3. **RAG System Module** (`src/rag_system/`)
   - Vector embeddings with FAISS
   - Query processing and analysis
   - Intelligent disambiguation
   - Main system integration

4. **Utilities** (`src/utils/`)
   - Shared helper functions
   - Common configurations

### Data Flow

```
User Query → Query Processor → Vector Search + Graph Search → Disambiguation → Response
```

## 🔧 Configuration

Edit `config.py` to customize:

- File paths and directories
- Model parameters
- Performance settings
- API keys and endpoints

## 🧪 Testing

### Built-in Tests

The system includes comprehensive test cases:

```bash
# Run all tests
python main.py --test

# Test specific queries
python -c "
from src.rag_system.hsn_rag_system import HSN_RAG_System
system = HSN_RAG_System()
system.initialize_system()
result = system.run_system_tests()
print(f'Tests: {result[\"passed_tests\"]}/{result[\"total_tests\"]} passed')
"
```

### Test Cases

1. **Direct Product Query**: "What is the HSN code for natural rubber latex?"
2. **Specific Product Type**: "HSN code for prevulcanised rubber"
3. **Broad Category**: "Rubber products classification"
4. **Direct HSN Lookup**: "Tell me about HSN 40011010"
5. **Similar Products**: "Similar products to natural rubber latex"

## 📈 Performance Monitoring

### Metrics Tracked

- Query success rate
- Average response time
- Confidence scores
- System uptime
- Component load times

### Viewing Metrics

```bash
# Via CLI
python main.py --metrics

# Via web interface
# Access the sidebar in the Streamlit app
```

## 🔍 API Reference

### HSN_RAG_System

#### Methods

- `initialize_system() → bool`: Initialize all system components
- `classify_product(query: str, enable_disambiguation: bool = True) → HSNClassificationResult`: Classify a single product
- `batch_classify(queries: List[str]) → List[HSNClassificationResult]`: Process multiple queries
- `get_exact_hsn_info(hsn_code: str) → Optional[Dict]`: Get detailed HSN code information
- `get_similar_products(hsn_code: str, limit: int = 5) → List[Dict]`: Find similar products
- `get_system_metrics() → Dict`: Get comprehensive system metrics
- `run_system_tests() → Dict`: Run test suite

### HSNClassificationResult

```python
@dataclass
class HSNClassificationResult:
    query: str
    hsn_code: Optional[str]
    description: Optional[str]
    confidence: float
    processing_time: float
    sources_used: List[str]
    disambiguation_needed: bool
    query_type: str
    export_policy: Optional[str]
    hierarchical_path: Optional[str]
    suggestions: List[str]
    metadata: Dict[str, Any]
```

## 🚀 Deployment

### Local Development (Automated)

1. Clone repository: `git clone https://github.com/sahit1011/HSN_RAG_KnowGraph.git`
2. Run setup: `./setup.sh` (Linux/macOS) or `setup.bat` (Windows)
3. Launch app: `./run.sh` (Linux/macOS) or `run.bat` (Windows)

### Local Development (Manual)

1. Install dependencies: `uv pip install -r requirements.txt` (or `pip install -r requirements.txt`)
2. Run tests: `python main.py --test`
3. Start web app: `streamlit run app.py`

### Production Deployment

1. **Containerization**:
    ```dockerfile
    FROM python:3.9-slim

    # Install UV for fast dependency installation
    RUN pip install uv

    COPY . /app
    WORKDIR /app

    # Use UV for faster installation
    RUN uv pip install --system -r requirements.txt

    # Run setup script to build models
    RUN chmod +x setup.sh && ./setup.sh

    CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
    ```

2. **Environment Variables**:
    ```bash
    export STREAMLIT_SERVER_PORT=8501
    export STREAMLIT_SERVER_ADDRESS=0.0.0.0
    export OPENROUTER_API_KEY="your-openrouter-api-key"
    # Or optionally use OpenAI
    export OPENAI_API_KEY="your-openai-api-key"
    ```

3. **Build and Run**:
    ```bash
    docker build -t hsn-rag-system .
    docker run -p 8501:8501 hsn-rag-system
    ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make changes and add tests
4. Run tests: `python main.py --test`
5. Commit changes: `git commit -am 'Add new feature'`
6. Push to branch: `git push origin feature/new-feature`
7. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions or issues:

1. Check the [Issues](https://github.com/sahit1011/HSN_RAG_KnowGraph/issues) page
2. Review the documentation in `plan.md`
3. Check the setup scripts for common issues
4. Contact the development team

### Repository
- **GitHub**: https://github.com/sahit1011/HSN_RAG_KnowGraph
- **Documentation**: See `plan.md` and `docs/` directory

## 🆕 Recent Updates

### v2.0.0 - Automated Setup & GitHub Integration
- ✅ **Unified Setup Scripts**: Cross-platform setup scripts (`setup.bat`/`setup.sh`)
- ✅ **Fast Installation**: UV package manager support for 10-100x faster dependency installation
- ✅ **Automated Run Scripts**: Simple launch scripts (`run.bat`/`run.sh`)
- ✅ **Codebase Cleanup**: Removed unnecessary test/demo files
- ✅ **GitHub Integration**: Repository published at https://github.com/sahit1011/HSN_RAG_KnowGraph
- ✅ **Optimized .gitignore**: Excludes large data files while preserving directory structure
- ✅ **Enhanced Documentation**: Updated README with automated setup instructions

### Key Improvements:
- **One-click Setup**: Recruiters can clone and run with just 2 commands
- **Cross-platform Support**: Works on Windows, Linux, and macOS
- **Smart File Management**: Only processes data when needed
- **Production Ready**: Complete end-to-end pipeline automation

## 🎯 Roadmap

- [ ] Enhanced disambiguation algorithms
- [ ] Multi-language support
- [ ] Real-time model updates
- [ ] Advanced analytics dashboard
- [ ] API endpoint for external integrations
- [ ] Mobile application companion
- [x] Automated setup scripts (Completed v2.0.0)
- [x] GitHub repository setup (Completed v2.0.0)

---

**Built with ❤️ for accurate HSN code classification**