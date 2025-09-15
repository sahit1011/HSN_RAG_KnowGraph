# HSN RAG System - Complete Query Processing Workflow

## Overview
This document explains how the HSN RAG (Retrieval-Augmented Generation) system processes user queries through three different modes: Rule-Based, LLM-Enhanced, and LLM-Only RAG.

**Example Query:** "What is the HSN code for natural rubber latex?"

---

## 1. System Architecture Overview

### Core Components
- **Streamlit App** (`app.py`) - User interface
- **HSN_RAG_System** (`src/rag_system/hsn_rag_system.py`) - Main orchestration
- **Query Processor** (`src/rag_system/query_processor.py`) - Query analysis and processing
- **Vector Store** (`src/rag_system/vector_store.py`) - Semantic search
- **Knowledge Graph** (`src/knowledge_graph/graph_implementation.py`) - Hierarchical relationships
- **LLM Client** (`src/utils/llm_client.py`) - AI enhancement (optional)

---

## 2. Common Initial Steps (All Modes)

### Step 1: User Input Processing
**File:** `app.py` (lines 202-221)
```python
# User submits query through Streamlit interface
query = st.text_input("Enter your product query:", value=default_query)
if submit_button and query.strip():
    result = st.session_state.system.classify_product(
        query.strip(),
        rag_mode=selected_rag_mode  # "rule_based", "llm_enhanced", or "llm_only"
    )
```

### Step 2: Main Classification Router
**File:** `src/rag_system/hsn_rag_system.py` (lines 257-291)
```python
def classify_product(self, query: str, rag_mode: str = "rule_based"):
    # Route to appropriate processing method
    if rag_mode == "rule_based":
        return self._classify_rule_based(query, enable_disambiguation, start_time)
    elif rag_mode == "llm_enhanced":
        return self._classify_llm_enhanced(query, enable_disambiguation, start_time)
    elif rag_mode == "llm_only":
        return self._classify_llm_only(query, start_time)
```

---

## 3. Rule-Based RAG Pipeline

### Step 3A: Rule-Based Processing
**File:** `src/rag_system/hsn_rag_system.py` (lines 313-326)
```python
def _classify_rule_based(self, query: str, enable_disambiguation: bool, start_time: float):
    # Process query through traditional RAG pipeline
    query_response = self.query_processor.process_query(query)
    # Continue with rule-based logic...
```

### Step 4A: Query Analysis
**File:** `src/rag_system/query_processor.py` (lines 136-203)
```python
def analyze_query(self, query: str) -> QueryAnalysis:
    # Analyze query type and extract entities
    query_lower = query.lower().strip()

    # Check for product-to-code patterns
    if self.query_patterns['product_query_pattern']['pattern'].search(query):
        analysis.query_type = QueryType.PRODUCT_TO_CODE
        analysis.intent = QueryIntent.FIND_CODE
        analysis.confidence = 0.8

        # Extract product name
        product_match = re.search(r'what is the hsn code for (.+)', query, re.IGNORECASE)
        if product_match:
            analysis.entities.append(product_match.group(1).strip())  # "natural rubber latex"

    # Extract keywords for search
    analysis.keywords = self._extract_keywords(query)
    return analysis
```

**Result:** Query classified as `PRODUCT_TO_CODE` with entity "natural rubber latex"

### Step 5A: Information Retrieval
**File:** `src/rag_system/query_processor.py` (lines 217-290)
```python
def retrieve_information(self, analysis: QueryAnalysis) -> RetrievalResult:
    # Vector search for semantic similarity
    search_query = ' '.join(analysis.entities + analysis.keywords)
    vector_results = self.vector_store.search_similar(search_query, top_k=5)

    # Graph search for hierarchical context
    if vector_results:
        for result in vector_results[:3]:
            hsn_code = result.get('hsn_code')
            if hsn_code:
                # Get hierarchical ancestors
                ancestors = self.knowledge_graph.query_hierarchical_ancestors(f"hsn_{hsn_code}")
                graph_results.extend(ancestors.results)

    return RetrievalResult(vector_results, graph_results, hybrid_score, retrieval_time)
```

### Step 6A: Vector Store Search
**File:** `src/rag_system/vector_store.py` (lines 203-244)
```python
def search_similar(self, query: str, top_k: int = 5):
    # Ensure model is loaded (lazy loading)
    if not self._ensure_model_loaded():
        return []

    # Encode query and search
    query_embedding = self.embedding_model.encode([query])[0]
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    query_embedding = query_embedding.reshape(1, -1).astype('float32')

    # Search FAISS index
    scores, indices = self.index.search(query_embedding, top_k)

    # Format results
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(self.documents):
            doc = self.documents[idx].copy()
            doc['similarity_score'] = float(score)
            results.append(doc)

    return results
```

### Step 7A: Response Generation
**File:** `src/rag_system/query_processor.py` (lines 328-352)
```python
def _generate_answer_text(self, analysis: QueryAnalysis, retrieval: RetrievalResult):
    if analysis.query_type == QueryType.PRODUCT_TO_CODE:
        if retrieval.vector_results:
            result = retrieval.vector_results[0]
            confidence_pct = int(result.get('similarity_score', 0) * 100)
            answer = f"The HSN code for '{analysis.entities[0]}' is {result['hsn_code']}.\n"
            answer += f"Description: {result['description']}\n"
            answer += f"Confidence: {confidence_pct}%\n"
            return answer
```

---

## 4. LLM-Enhanced RAG Pipeline

### Step 3B: LLM-Enhanced Processing
**File:** `src/rag_system/hsn_rag_system.py` (lines 429-474)
```python
def _classify_llm_enhanced(self, query: str, enable_disambiguation: bool, start_time: float):
    # First get rule-based results
    rule_based_result = self._classify_rule_based(query, enable_disambiguation, start_time)

    # If rule-based failed, return as-is
    if not rule_based_result.hsn_code:
        rule_based_result.metadata['rag_mode'] = 'llm_enhanced'
        return rule_based_result

    # Enhance with LLM if available
    if hasattr(self.knowledge_graph, 'llm_client') and self.knowledge_graph.llm_client:
        try:
            # Get retrieved documents for context
            retrieved_docs = []
            graph_context = []

            # Get additional context from knowledge graph
            if rule_based_result.hsn_code:
                similar_results = self.knowledge_graph.query_similar_products(f"hsn_{rule_based_result.hsn_code}")
                graph_context = similar_results.results if hasattr(similar_results, 'results') else []

            # Generate enhanced response
            enhanced_response = self.knowledge_graph.llm_client.generate_response(
                query=query,
                retrieved_docs=retrieved_docs,
                graph_context=graph_context,
                hsn_result={
                    'hsn_code': rule_based_result.hsn_code,
                    'description': rule_based_result.description
                }
            )

            # Update result with enhanced information
            rule_based_result.metadata['llm_enhanced'] = True
            rule_based_result.metadata['enhanced_description'] = enhanced_response
            rule_based_result.metadata['rag_mode'] = 'llm_enhanced'

            # Boost confidence slightly
            rule_based_result.confidence = min(1.0, rule_based_result.confidence * 1.1)

        except Exception as e:
            logger.warning(f"LLM enhancement failed: {str(e)}")
            rule_based_result.metadata['llm_enhancement_error'] = str(e)

    return rule_based_result
```

### Step 4B: LLM Enhancement
**File:** `src/utils/llm_client.py` (generate_response method)
```python
def generate_response(self, query: str, retrieved_docs: List, graph_context: List, hsn_result: Dict):
    # Use LLM to enhance the response with additional insights
    # Combine rule-based results with AI-generated context
    enhanced_description = self.llm.generate([
        {"role": "system", "content": "Enhance this HSN classification with additional context..."},
        {"role": "user", "content": f"Query: {query}\nHSN Result: {hsn_result}\nGraph Context: {graph_context}"}
    ])

    return enhanced_description
```

---

## 5. LLM-Only RAG Pipeline

### Step 3C: LLM-Only Processing
**File:** `src/rag_system/hsn_rag_system.py` (lines 477-526)
```python
def _classify_llm_only(self, query: str, start_time: float) -> HSNClassificationResult:
    # Use LLM to directly generate classification
    if hasattr(self.knowledge_graph, 'llm_client') and self.knowledge_graph.llm_client:
        try:
            # Generate response using LLM with minimal context
            llm_response = self.knowledge_graph.llm_client.generate_response(
                query=query,
                retrieved_docs=[],  # No pre-retrieved docs
                graph_context=[],   # No graph context
                hsn_result=None     # No base HSN result
            )

            processing_time = time.time() - start_time

            # Try to extract HSN code from LLM response
            import re
            hsn_match = re.search(r'\b\d{4,10}\b', llm_response)
            hsn_code = hsn_match.group(0) if hsn_match else None

            # Create result
            result = HSNClassificationResult(
                query=query,
                hsn_code=hsn_code,
                description=llm_response[:200] if not hsn_code else f"LLM-generated classification: {llm_response[:200]}",
                confidence=0.7,  # Lower confidence for LLM-only
                processing_time=processing_time,
                sources_used=['llm'],
                disambiguation_needed=False,
                query_type='llm_generated',
                export_policy=None,
                hierarchical_path=None,
                suggestions=["This is an LLM-generated classification", "Verify with official sources"],
                metadata={
                    'rag_mode': 'llm_only',
                    'llm_response': llm_response,
                    'confidence_note': 'Lower confidence due to LLM-only approach'
                }
            )

            self.metrics.successful_queries += 1
            return result

        except Exception as e:
            logger.error(f"LLM-only classification failed: {str(e)}")

    # Fallback to rule-based if LLM fails
    logger.warning("LLM-only failed, falling back to rule-based")
    return self._classify_rule_based(query, True, start_time)
```

### Step 4C: Pure LLM Generation
**File:** `src/utils/llm_client.py`
```python
def generate_response(self, query: str, retrieved_docs: List, graph_context: List, hsn_result: Dict):
    # Generate classification purely from LLM knowledge
    response = self.llm.generate([
        {"role": "system", "content": "You are an expert in HSN codes. Classify this product and provide the most appropriate HSN code with explanation."},
        {"role": "user", "content": f"Product: {query}\n\nProvide the HSN code and detailed explanation."}
    ])

    return response
```

---

## 6. Final Result Processing (All Modes)

### Step 8: Result Formatting and Return
**File:** `src/rag_system/hsn_rag_system.py` (lines 350-426)
```python
# Format final result with metadata
result = HSNClassificationResult(
    query=query,
    hsn_code=final_hsn_code,
    description=final_description,
    confidence=calculated_confidence,
    processing_time=time.time() - start_time,
    sources_used=sources_list,
    disambiguation_needed=disambiguation_flag,
    query_type=query_type,
    export_policy=export_policy,
    hierarchical_path=hierarchical_path,
    suggestions=suggestions,
    metadata={
        'rag_mode': rag_mode,
        'llm_enhanced': is_llm_enhanced,
        'similarity_score': similarity_score,
        'total_candidates': total_candidates,
        'graph_context_used': graph_used,
        'processing_details': processing_details
    }
)

logger.info(f"Query processed in {result.processing_time:.2f}s")
return result
```

### Step 9: Streamlit Display
**File:** `app.py` (lines 222-224)
```python
# Display result in Streamlit interface
st.subheader("Classification Result")
display_result(result)
```

---

## 7. Mode Comparison Summary

| Aspect | Rule-Based RAG | LLM-Enhanced RAG | LLM-Only RAG |
|--------|----------------|------------------|--------------|
| **Primary Method** | Vector + Graph Search | Rule-Based + LLM Enhancement | Pure LLM Generation |
| **Confidence** | High (0.8-0.95) | Very High (0.85-1.0) | Medium (0.6-0.8) |
| **Speed** | Fastest | Medium | Slowest |
| **Accuracy** | High | Highest | Variable |
| **Context Usage** | Full (Vector + Graph) | Full + LLM | Minimal |
| **Fallback** | None needed | Rule-Based | Rule-Based |
| **Best For** | Standard queries | Complex/ambiguous queries | When no training data |

---

## 8. Error Handling and Fallbacks

### System-Level Error Handling
**File:** `src/rag_system/hsn_rag_system.py` (lines 292-311)
```python
except Exception as e:
    logger.error(f"Error processing query '{query}': {str(e)}")
    self.metrics.error_count += 1

    # Return error result
    return HSNClassificationResult(
        query=query,
        hsn_code=None,
        description=None,
        confidence=0.0,
        processing_time=time.time() - start_time,
        sources_used=[],
        disambiguation_needed=False,
        query_type='error',
        export_policy=None,
        hierarchical_path=None,
        suggestions=["Please try again", "Contact support if issue persists"],
        metadata={'error': str(e)}
    )
```

### LLM Fallback Mechanisms
- **LLM-Enhanced**: Falls back to rule-based if LLM fails
- **LLM-Only**: Falls back to rule-based if LLM fails
- **Rule-Based**: No fallback (primary method)

---

## 9. Performance Monitoring

### Metrics Collection
**File:** `src/rag_system/hsn_rag_system.py` (SystemMetrics class)
```python
@dataclass
class SystemMetrics:
    total_queries: int = 0
    successful_queries: int = 0
    average_response_time: float = 0.0
    average_confidence: float = 0.0
    disambiguation_rate: float = 0.0
    component_load_times: Dict[str, float] = None
    error_count: int = 0
    uptime_start: datetime = None
```

### Real-time Performance Tracking
- Query processing time
- Success/failure rates
- Component load times
- Confidence scores
- Error tracking

---

## 10. Configuration and Modes

### RAG Mode Configuration
**File:** `src/rag_system/hsn_rag_system.py` (lines 29-36)
```python
RAG_MODES = {
    "rule_based": {"name": "Rule-based RAG", "uses_llm": False},
    "llm_enhanced": {"name": "LLM-Enhanced RAG", "uses_llm": True},
    "llm_only": {"name": "LLM-Only RAG", "uses_llm": True}
}
```

### Dynamic Mode Selection
**File:** `app.py` (lines 144-155)
```python
# RAG Mode Selection in sidebar
rag_mode_options = {mode: info['name'] for mode, info in RAG_MODES.items()}
selected_rag_mode = st.selectbox(
    "Choose RAG processing mode:",
    options=list(rag_mode_options.keys()),
    format_func=lambda x: rag_mode_options[x],
    help="Select the RAG approach for processing your queries"
)
```

This comprehensive workflow ensures robust, flexible, and intelligent HSN code classification across all processing modes.