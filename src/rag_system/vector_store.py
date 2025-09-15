#!/usr/bin/env python3
"""
HSN Vector Store Implementation - Phase 3.1
Creates and manages vector embeddings for RAG system using FAISS and Sentence Transformers
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
from pathlib import Path
import pickle
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics.pairwise import cosine_similarity
import time
from tqdm import tqdm

# File paths
OUTPUT_DIR = Path("output")
DATA_DIR = OUTPUT_DIR / "data"
MODELS_DIR = OUTPUT_DIR / "models"
VECTOR_DIR = OUTPUT_DIR / "vectors"
ENHANCED_DATA_PATH = DATA_DIR / "extraction_complete.csv"

VECTOR_DIR.mkdir(exist_ok=True)

class HSNVectorStore:
    """
    Vector store for HSN classification documents using FAISS and Sentence Transformers
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector store

        Args:
            model_name: Sentence Transformer model name
        """
        self.model_name = model_name
        self.embedding_model = None
        self.index = None
        self.documents = []
        self.embeddings = None
        self.metadata = {}
        self.model_loaded = False
        self.fallback_mode = False  # Use keyword search if embeddings fail

    def load_embedding_model(self) -> bool:
        """
        Load the sentence transformer model

        Returns:
            True if loaded successfully
        """
        if self.model_loaded and self.embedding_model is not None:
            return True

        try:
            # Suppress excessive logging from transformers and urllib3
            import logging
            logging.getLogger("transformers").setLevel(logging.WARNING)
            logging.getLogger("urllib3").setLevel(logging.WARNING)
            logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

            print(f"Loading embedding model: {self.model_name}")
            self.embedding_model = SentenceTransformer(self.model_name)
            self.model_loaded = True
            print(f"SUCCESS: Model loaded with dimension {self.embedding_model.get_sentence_embedding_dimension()}")
            return True
        except Exception as e:
            print(f"ERROR: Failed to load embedding model: {str(e)}")
            print("INFO: Switching to fallback mode (keyword search only)")
            self.fallback_mode = True
            return False

    def _ensure_model_loaded(self) -> bool:
        """
        Ensure the embedding model is loaded (lazy loading)

        Returns:
            True if model is available
        """
        if not self.model_loaded:
            print("Loading embedding model on demand...")
            return self.load_embedding_model()
        return True

    def load_documents(self, data_path: Path = ENHANCED_DATA_PATH) -> bool:
        """
        Load documents from enhanced data CSV

        Args:
            data_path: Path to the enhanced data CSV

        Returns:
            True if loaded successfully
        """
        try:
            df = pd.read_csv(data_path)
            print(f"Loaded {len(df)} documents from {data_path}")

            # Extract document content and metadata
            for idx, row in df.iterrows():
                # Get HSN code
                hsn_code = row.get('hsn_code') or row.get('full_hsn_code') or row.get('subheading') or row.get('heading') or row.get('chapter', '')

                # Create document content if not present
                document_content = row.get('document_content', '')
                if not document_content:
                    # Create document content from available fields
                    doc_parts = []

                    if hsn_code:
                        doc_parts.append(f"HSN Code: {hsn_code}")

                    # Add hierarchical context
                    if row.get('chapter_title'):
                        doc_parts.append(f"Chapter: {row['chapter_title']}")
                    if row.get('heading_title'):
                        doc_parts.append(f"Heading: {row['heading_title']}")
                    if row.get('subheading_title'):
                        doc_parts.append(f"Subheading: {row['subheading_title']}")

                    # Product description
                    if row.get('description'):
                        doc_parts.append(f"Description: {row['description']}")

                    # Export policy
                    if row.get('export_policy') and not pd.isna(row['export_policy']):
                        doc_parts.append(f"Export Policy: {row['export_policy']}")

                    # Complete context
                    if row.get('complete_context'):
                        doc_parts.append(f"Full Context: {row['complete_context']}")

                    document_content = '\n'.join(doc_parts)

                # Create searchable keywords if not present
                search_keywords = row.get('search_keywords', '')
                if not search_keywords:
                    keywords = []
                    if hsn_code:
                        keywords.append(str(hsn_code))
                    if row.get('description') and not pd.isna(row['description']):
                        desc_words = str(row['description']).lower().split()
                        keywords.extend([word for word in desc_words if len(word) > 3])
                    search_keywords = ', '.join(keywords[:10]) if keywords else ''

                doc = {
                    'id': f"doc_{idx}",
                    'content': document_content,
                    'hsn_code': hsn_code,
                    'code_level': row.get('code_level', ''),
                    'description': row.get('description', ''),
                    'export_policy': row.get('export_policy', ''),
                    'search_keywords': search_keywords,
                    'hierarchy_level': row.get('hierarchy_level', ''),
                    'document_length': len(document_content),
                    'chapter_title': row.get('chapter_title', ''),
                    'heading_title': row.get('heading_title', ''),
                    'subheading_title': row.get('subheading_title', ''),
                    'complete_context': row.get('complete_context', '')
                }
                self.documents.append(doc)

            print(f"SUCCESS: Processed {len(self.documents)} documents")
            return True

        except Exception as e:
            print(f"ERROR: Failed to load documents: {str(e)}")
            return False

    def create_embeddings(self, batch_size: int = 32) -> bool:
        """
        Create vector embeddings for all documents

        Args:
            batch_size: Batch size for embedding generation

        Returns:
            True if embeddings created successfully
        """
        if not self.embedding_model:
            print("ERROR: Embedding model not loaded")
            return False

        try:
            print(f"Creating embeddings for {len(self.documents)} documents...")

            # Extract texts for embedding
            texts = [doc['content'] for doc in self.documents]

            # Create embeddings in batches
            embeddings = []
            with tqdm(total=len(texts), desc="Creating embeddings", unit="docs") as pbar:
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    batch_embeddings = self.embedding_model.encode(batch_texts, show_progress_bar=False)
                    embeddings.extend(batch_embeddings)
                    pbar.update(len(batch_texts))

            self.embeddings = np.array(embeddings)
            print(f"SUCCESS: Created embeddings with shape {self.embeddings.shape}")

            # Store embedding metadata
            self.metadata['embedding_dimension'] = self.embeddings.shape[1]
            self.metadata['num_documents'] = len(self.documents)
            self.metadata['model_name'] = self.model_name
            self.metadata['created_at'] = pd.Timestamp.now().isoformat()

            return True

        except Exception as e:
            print(f"ERROR: Failed to create embeddings: {str(e)}")
            return False

    def build_faiss_index(self) -> bool:
        """
        Build FAISS index for efficient similarity search

        Returns:
            True if index built successfully
        """
        if self.embeddings is None:
            print("ERROR: No embeddings available")
            return False

        try:
            print("Building FAISS index...")

            dimension = self.embeddings.shape[1]

            # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
            self.index = faiss.IndexFlatIP(dimension)

            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            normalized_embeddings = self.embeddings / norms

            # Add vectors to index
            self.index.add(normalized_embeddings.astype('float32'))

            print(f"SUCCESS: FAISS index built with {self.index.ntotal} vectors")
            return True

        except Exception as e:
            print(f"ERROR: Failed to build FAISS index: {str(e)}")
            return False

    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity or keyword fallback

        Args:
            query: Search query text
            top_k: Number of top results to return

        Returns:
            List of similar documents with scores
        """
        # If in fallback mode, use keyword search
        if self.fallback_mode:
            print("INFO: Using keyword search (fallback mode)")
            return self.search_by_keywords(query.split(), top_k=top_k)

        # Ensure model is loaded (lazy loading)
        if not self._ensure_model_loaded():
            print("INFO: Model loading failed, using keyword search")
            return self.search_by_keywords(query.split(), top_k=top_k)

        if not self.index or self.embeddings is None:
            print("ERROR: Vector store not properly initialized")
            return []

        try:
            # Encode query
            query_embedding = self.embedding_model.encode([query])[0]
            query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize
            query_embedding = query_embedding.reshape(1, -1).astype('float32')

            # Search
            scores, indices = self.index.search(query_embedding, top_k)

            # Format results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):  # Valid index
                    doc = self.documents[idx].copy()
                    doc['similarity_score'] = float(score)
                    doc['rank'] = len(results) + 1
                    results.append(doc)

            return results

        except Exception as e:
            print(f"ERROR: Vector search failed: {str(e)}, falling back to keyword search")
            return self.search_by_keywords(query.split(), top_k=top_k)

    def search_by_hsn_code(self, hsn_code: str) -> Optional[Dict[str, Any]]:
        """
        Search for exact HSN code match

        Args:
            hsn_code: HSN code to search for

        Returns:
            Document if found, None otherwise
        """
        # Clean the search code
        search_code = str(hsn_code).strip()
        print(f"DEBUG: Searching for HSN code: '{search_code}' in {len(self.documents)} documents")

        found_candidates = []

        for doc in self.documents:
            doc_code = doc.get('hsn_code')
            if doc_code is not None:
                # Skip NaN values
                if isinstance(doc_code, float) and str(doc_code) == 'nan':
                    continue

                # Clean the document code
                doc_code_str = str(doc_code).strip().rstrip('.0')

                # Exact string match
                if doc_code_str == search_code:
                    print(f"DEBUG: Found exact match for HSN code: {search_code}")
                    return doc

                # Try numeric comparison for partial matches
                try:
                    doc_numeric = float(doc_code) if doc_code != '' else None
                    search_numeric = float(search_code)

                    if doc_numeric is not None and doc_numeric == search_numeric:
                        print(f"DEBUG: Found numeric match for HSN code: {search_code}")
                        return doc
                except (ValueError, TypeError):
                    pass

        print(f"DEBUG: No exact match found for HSN code: {search_code}")
        return None

    def search_by_keywords(self, keywords: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search documents containing specific keywords

        Args:
            keywords: List of keywords to search for
            top_k: Number of results to return

        Returns:
            List of matching documents with similarity_score for compatibility
        """
        results = []
        keywords_lower = [kw.lower() for kw in keywords]

        for doc in self.documents:
            doc_keywords = doc.get('search_keywords', '').lower()
            content = doc.get('content', '').lower()

            # Check if all keywords are present
            keyword_matches = sum(1 for kw in keywords_lower
                                if kw in doc_keywords or kw in content)

            if keyword_matches > 0:
                doc_copy = doc.copy()
                doc_copy['keyword_matches'] = keyword_matches
                doc_copy['match_score'] = keyword_matches / len(keywords)
                # Add similarity_score for compatibility with vector search results
                doc_copy['similarity_score'] = doc_copy['match_score']
                doc_copy['rank'] = len(results) + 1
                results.append(doc_copy)

        # Sort by match score and return top_k
        results.sort(key=lambda x: x['match_score'], reverse=True)
        return results[:top_k]

    def hybrid_search(self, query: str, keywords: Optional[List[str]] = None,
                     top_k: int = 5, similarity_weight: float = 0.7) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector similarity and keyword matching

        Args:
            query: Semantic search query
            keywords: Optional keyword list for boosting
            top_k: Number of results to return
            similarity_weight: Weight for similarity score (0-1)

        Returns:
            Ranked list of documents
        """
        # Get vector similarity results
        vector_results = self.search_similar(query, top_k=top_k*2)  # Get more candidates

        if not keywords:
            return vector_results[:top_k]

        # Get keyword matching results
        keyword_results = self.search_by_keywords(keywords, top_k=top_k*2)

        # Create combined scoring
        combined_results = {}

        # Add vector results
        for result in vector_results:
            doc_id = result['id']
            combined_results[doc_id] = {
                'doc': result,
                'similarity_score': result['similarity_score'] * similarity_weight,
                'keyword_score': 0.0
            }

        # Add keyword boost
        for result in keyword_results:
            doc_id = result['id']
            keyword_score = result['match_score'] * (1 - similarity_weight)

            if doc_id in combined_results:
                combined_results[doc_id]['keyword_score'] = keyword_score
            else:
                combined_results[doc_id] = {
                    'doc': result,
                    'similarity_score': 0.0,
                    'keyword_score': keyword_score
                }

        # Calculate final scores and rank
        final_results = []
        for doc_id, scores in combined_results.items():
            final_score = scores['similarity_score'] + scores['keyword_score']
            doc = scores['doc'].copy()
            doc['final_score'] = final_score
            doc['similarity_contribution'] = scores['similarity_score']
            doc['keyword_contribution'] = scores['keyword_score']
            final_results.append(doc)

        # Sort by final score
        final_results.sort(key=lambda x: x['final_score'], reverse=True)
        return final_results[:top_k]

    def save_vector_store(self, base_path: Path = VECTOR_DIR) -> Dict[str, str]:
        """
        Save the vector store components

        Args:
            base_path: Base directory to save files

        Returns:
            Dictionary of saved file paths
        """
        saved_files = {}

        try:
            # Save FAISS index
            index_path = base_path / "hsn_faiss_index.idx"
            faiss.write_index(self.index, str(index_path))
            saved_files['faiss_index'] = str(index_path)

            # Save embeddings
            embeddings_path = base_path / "hsn_embeddings.npy"
            np.save(embeddings_path, self.embeddings)
            saved_files['embeddings'] = str(embeddings_path)

            # Save documents and metadata
            data_path = base_path / "hsn_vector_data.pkl"
            with open(data_path, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'metadata': self.metadata
                }, f)
            saved_files['documents'] = str(data_path)

            # Save configuration
            config_path = base_path / "hsn_vector_config.json"
            config = {
                'model_name': self.model_name,
                'embedding_dimension': self.metadata.get('embedding_dimension', 0),
                'num_documents': len(self.documents),
                'created_at': self.metadata.get('created_at', ''),
                'saved_files': saved_files
            }
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            saved_files['config'] = str(config_path)

            print(f"SUCCESS: Vector store saved to {len(saved_files)} files")
            return saved_files

        except Exception as e:
            print(f"ERROR: Failed to save vector store: {str(e)}")
            return {}

    def load_vector_store(self, base_path: Path = VECTOR_DIR) -> bool:
        """
        Load the vector store from saved files

        Args:
            base_path: Base directory containing saved files

        Returns:
            True if loaded successfully
        """
        try:
            config_path = base_path / "hsn_vector_config.json"
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Load FAISS index
            index_path = base_path / "hsn_faiss_index.idx"
            self.index = faiss.read_index(str(index_path))

            # Load embeddings
            embeddings_path = base_path / "hsn_embeddings.npy"
            self.embeddings = np.load(embeddings_path)

            # Load documents and metadata
            data_path = base_path / "hsn_vector_data.pkl"
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.metadata = data['metadata']

            # Load model name but don't load the actual model yet (lazy loading)
            self.model_name = config['model_name']
            # self.load_embedding_model()  # Removed for lazy loading

            print(f"SUCCESS: Vector store loaded with {len(self.documents)} documents")
            return True

        except Exception as e:
            print(f"ERROR: Failed to load vector store: {str(e)}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the vector store

        Returns:
            Dictionary with statistics
        """
        stats = {
            'documents': {
                'total': len(self.documents),
                'by_level': {},
                'avg_length': 0
            },
            'embeddings': {
                'dimension': self.metadata.get('embedding_dimension', 0),
                'model': self.model_name
            },
            'index': {
                'type': 'FAISS IndexFlatIP',
                'vectors': self.index.ntotal if self.index else 0
            }
        }

        # Document statistics
        total_length = 0
        level_counts = {}

        for doc in self.documents:
            level = doc.get('code_level', 'unknown')
            level_counts[level] = level_counts.get(level, 0) + 1
            total_length += len(doc.get('content', ''))

        stats['documents']['by_level'] = level_counts
        stats['documents']['avg_length'] = total_length / len(self.documents) if self.documents else 0

        return stats

def main():
    """Main execution function for vector store setup."""
    print("Starting HSN Vector Store Setup (Phase 3.1)")
    print("=" * 60)

    # Initialize vector store
    vector_store = HSNVectorStore()

    try:
        # Load embedding model
        print("\n1. Loading embedding model...")
        if not vector_store.load_embedding_model():
            raise Exception("Failed to load embedding model")

        # Load documents
        print("\n2. Loading documents...")
        if not vector_store.load_documents():
            raise Exception("Failed to load documents")

        # Create embeddings
        print("\n3. Creating embeddings...")
        if not vector_store.create_embeddings():
            raise Exception("Failed to create embeddings")

        # Build FAISS index
        print("\n4. Building FAISS index...")
        if not vector_store.build_faiss_index():
            raise Exception("Failed to build FAISS index")

        # Test basic functionality
        print("\n5. Testing vector search...")
        test_query = "natural rubber latex"
        results = vector_store.search_similar(test_query, top_k=3)
        print(f"Query: '{test_query}'")
        print(f"Found {len(results)} similar documents")
        if results:
            print(f"Top result: {results[0]['hsn_code']} - {results[0]['description'][:50]}...")

        # Test hybrid search
        print("\n6. Testing hybrid search...")
        hybrid_results = vector_store.hybrid_search(
            query="synthetic rubber",
            keywords=["styrene", "butadiene"],
            top_k=3
        )
        print(f"Hybrid search found {len(hybrid_results)} results")

        # Save vector store
        print("\n7. Saving vector store...")
        saved_files = vector_store.save_vector_store()
        print(f"Saved {len(saved_files)} files to {VECTOR_DIR}")

        # Display statistics
        print("\n8. Vector store statistics...")
        stats = vector_store.get_statistics()
        print(f"Documents: {stats['documents']['total']}")
        print(f"Embeddings: {stats['embeddings']['dimension']}D")
        print(f"Model: {stats['embeddings']['model']}")
        print(f"Index: {stats['index']['vectors']} vectors")

        # Summary
        print("\n" + "=" * 60)
        print("PHASE 3.1 VECTOR STORE SETUP COMPLETE")
        print("=" * 60)
        print(f"SUCCESS: Vector store created with {len(vector_store.documents)} documents")
        print(f"SUCCESS: Embeddings generated with {vector_store.metadata.get('embedding_dimension', 0)} dimensions")
        print(f"SUCCESS: FAISS index built for efficient similarity search")
        print(f"SUCCESS: Hybrid search functionality implemented")
        print(f"Files saved to: {VECTOR_DIR}")
        print("Ready to proceed to Phase 3.2: Query Processing Engine")
        print("=" * 60)

    except Exception as e:
        print(f"ERROR: Error in vector store setup: {str(e)}")
        raise

if __name__ == "__main__":
    main()