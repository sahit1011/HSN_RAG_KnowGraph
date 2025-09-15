# HSN RAG Classification System

A modular, intelligent HSN (Harmonized System Nomenclature) code classification system using Retrieval-Augmented Generation (RAG) and Knowledge Graphs.

## 🚀 Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Intelligent Classification**: Combines vector search, knowledge graphs, and natural language processing
- **Web Interface**: Streamlit-based chat interface for easy testing
- **Command Line Interface**: Full CLI support for batch processing and automation
- **Comprehensive Testing**: Built-in test suite with performance monitoring
- **Scalable Design**: Production-ready architecture with proper error handling

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
│   │   └── __init__.py
│   └── __init__.py
├── output/                       # Generated data and models
│   ├── data/                     # Processed data files
│   ├── models/                   # Trained models
│   ├── vectors/                  # Vector embeddings
│   ├── logs/                     # System logs
│   └── reports/                  # Performance reports
├── main.py                       # CLI entry point
├── app.py                        # Streamlit web app
├── config.py                     # Configuration settings
├── requirements.txt              # Python dependencies
├── plan.md                       # Project documentation
└── README.md                     # This file
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd hsn_rag
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment** (optional):
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## 🚀 Usage

### Web Interface (Recommended)

Start the Streamlit web application:

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` to access the interactive interface.

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

### Local Development

1. Install dependencies: `pip install -r requirements.txt`
2. Run tests: `python main.py --test`
3. Start web app: `streamlit run app.py`

### Production Deployment

1. **Containerization**:
   ```dockerfile
   FROM python:3.9-slim
   COPY . /app
   WORKDIR /app
   RUN pip install -r requirements.txt
   CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Environment Variables**:
   ```bash
   export STREAMLIT_SERVER_PORT=8501
   export STREAMLIT_SERVER_ADDRESS=0.0.0.0
   export OPENAI_API_KEY="your-production-key"
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

1. Check the [Issues](https://github.com/your-repo/issues) page
2. Review the documentation in `plan.md`
3. Contact the development team

## 🎯 Roadmap

- [ ] Enhanced disambiguation algorithms
- [ ] Multi-language support
- [ ] Real-time model updates
- [ ] Advanced analytics dashboard
- [ ] API endpoint for external integrations
- [ ] Mobile application companion

---

**Built with ❤️ for accurate HSN code classification**