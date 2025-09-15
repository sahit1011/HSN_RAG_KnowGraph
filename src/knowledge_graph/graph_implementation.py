#!/usr/bin/env python3
"""
HSN Knowledge Graph Implementation - Phase 2.2
Builds and manages the NetworkX-based knowledge graph from the designed schema
"""

import json
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import pickle
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from tqdm import tqdm

# Import LLM components
try:
    from ..utils.llm_client import LangChainLLMClient
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logging.warning("LLM components not available. LLM features will be disabled.")

# File paths
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"
DATA_DIR = OUTPUT_DIR / "data"
MODELS_DIR = OUTPUT_DIR / "models"
SCHEMA_FILE = DATA_DIR / "graph_schema.json"
GRAPH_FILE = MODELS_DIR / "hsn_knowledge_graph.pkl"

class GraphQueryType(Enum):
    HIERARCHICAL_DESCENDANTS = "hierarchical_descendants"
    HIERARCHICAL_ANCESTORS = "hierarchical_ancestors"
    SIMILAR_PRODUCTS = "similar_products"
    SAME_EXPORT_POLICY = "same_export_policy"
    CHAPTER_CONTENTS = "chapter_contents"
    CODE_LOOKUP = "code_lookup"

@dataclass
class GraphQueryResult:
    """Result of a graph query"""
    query_type: GraphQueryType
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    execution_time: float

class HSNKnowledgeGraph:
    """
    NetworkX-based implementation of the HSN Knowledge Graph
    """

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.schema = None
        self.node_types = {}
        self.relationship_types = {}
        self.metadata = {}
        self.llm_client = None

        # Initialize LLM client if available (don't fail if it doesn't work)
        if LLM_AVAILABLE:
            try:
                self.llm_client = LangChainLLMClient()
                logging.info("SUCCESS: LLM client initialized for knowledge graph")
            except Exception as e:
                logging.warning(f"Failed to initialize LLM client: {e}")
                logging.warning("Continuing without LLM client")
                self.llm_client = None

    def load_schema(self, schema_file: Path) -> bool:
        """
        Load graph schema from JSON file

        Args:
            schema_file: Path to the schema JSON file

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            print(f"DEBUG: Attempting to load schema from: {schema_file}")
            print(f"DEBUG: Schema file exists: {schema_file.exists()}")
            print(f"DEBUG: Schema file absolute path: {schema_file.absolute()}")

            with open(schema_file, 'r', encoding='utf-8') as f:
                schema_data = json.load(f)

            self.schema = schema_data
            self.node_types = schema_data.get('node_types', {})
            self.relationship_types = schema_data.get('relationship_types', {})
            self.metadata = schema_data.get('metadata', {})

            print(f"SUCCESS: Loaded schema with {len(schema_data.get('nodes', {}))} nodes and {len(schema_data.get('relationships', []))} relationships")
            return True

        except FileNotFoundError:
            print(f"ERROR: Schema file not found: {schema_file}")
            return False
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON in schema file: {e}")
            return False
        except Exception as e:
            print(f"ERROR: Failed to load schema: {str(e)}")
            print(f"ERROR: Exception type: {type(e)}")
            import traceback
            traceback.print_exc()
            return False

    def build_graph(self) -> bool:
        """
        Build the NetworkX graph from the loaded schema

        Returns:
            True if built successfully, False otherwise
        """
        if not self.schema:
            print("ERROR: No schema loaded. Call load_schema() first.")
            return False

        try:
            # Clear existing graph
            self.graph.clear()

            # Add nodes
            nodes_data = self.schema.get('nodes', {})
            for node_id, node_props in tqdm(nodes_data.items(), desc="Adding nodes", unit="node"):
                # Convert node properties to graph node attributes
                node_attrs = dict(node_props)
                node_attrs['node_type'] = node_id.split('_')[0]  # Extract type from ID

                self.graph.add_node(node_id, **node_attrs)

            # Add relationships
            relationships_data = self.schema.get('relationships', [])
            for rel_data in tqdm(relationships_data, desc="Adding relationships", unit="rel"):
                source = rel_data['source_id']
                target = rel_data['target_id']
                rel_type = rel_data['type']
                properties = rel_data.get('properties', {})

                # Add edge with relationship type as key
                self.graph.add_edge(source, target, key=rel_type, **properties)

            # Add graph-level metadata
            self.graph.graph['schema_metadata'] = self.metadata
            self.graph.graph['node_types'] = self.node_types
            self.graph.graph['relationship_types'] = self.relationship_types
            self.graph.graph['build_info'] = {
                'library': 'networkx',
                'version': nx.__version__,
                'build_time': pd.Timestamp.now().isoformat()
            }

            print(f"SUCCESS: Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            return True

        except Exception as e:
            print(f"ERROR: Failed to build graph: {str(e)}")
            return False

    def validate_graph(self) -> Dict[str, Any]:
        """
        Validate the built graph for consistency and completeness

        Returns:
            Validation results dictionary
        """
        validation = {
            'node_count': self.graph.number_of_nodes(),
            'edge_count': self.graph.number_of_edges(),
            'is_directed': self.graph.is_directed(),
            'is_multigraph': self.graph.is_multigraph(),
            'connected_components': nx.number_weakly_connected_components(self.graph),
            'node_types_present': {},
            'relationship_types_present': {},
            'orphaned_nodes': [],
            'issues': []
        }

        # Check node types
        node_types_found = {}
        for node_id, node_data in self.graph.nodes(data=True):
            node_type = node_data.get('node_type', 'unknown')
            node_types_found[node_type] = node_types_found.get(node_type, 0) + 1

        validation['node_types_present'] = node_types_found

        # Check relationship types
        rel_types_found = {}
        for source, target, key, data in self.graph.edges(keys=True, data=True):
            rel_types_found[key] = rel_types_found.get(key, 0) + 1

        validation['relationship_types_present'] = rel_types_found

        # Find orphaned nodes (nodes with no edges)
        orphaned = []
        for node_id in self.graph.nodes():
            if self.graph.degree(node_id) == 0:
                orphaned.append(node_id)

        validation['orphaned_nodes'] = orphaned

        # Check for common issues
        if validation['node_count'] == 0:
            validation['issues'].append("Graph has no nodes")

        if validation['edge_count'] == 0:
            validation['issues'].append("Graph has no edges")

        if len(orphaned) > 0:
            validation['issues'].append(f"Found {len(orphaned)} orphaned nodes")

        # Calculate validation score
        issues_penalty = len(validation['issues'])
        orphaned_penalty = min(len(orphaned) / validation['node_count'], 1.0) if validation['node_count'] > 0 else 1.0

        validation['validation_score'] = max(0.0, 1.0 - issues_penalty * 0.2 - orphaned_penalty * 0.3)

        return validation

    def query_hierarchical_descendants(self, node_id: str, max_depth: int = 3) -> GraphQueryResult:
        """
        Find all hierarchical descendants of a node

        Args:
            node_id: Starting node ID
            max_depth: Maximum depth to traverse

        Returns:
            GraphQueryResult with descendant information
        """
        import time
        start_time = time.time()

        if node_id not in self.graph:
            return GraphQueryResult(
                query_type=GraphQueryType.HIERARCHICAL_DESCENDANTS,
                results=[],
                metadata={'error': f'Node {node_id} not found'},
                execution_time=time.time() - start_time
            )

        descendants = []
        visited = set()

        def traverse_descendants(current_node, depth=0):
            if depth >= max_depth or current_node in visited:
                return

            visited.add(current_node)

            # Find all IS_PARENT_OF relationships from current node
            for successor in self.graph.successors(current_node):
                edge_data = self.graph.get_edge_data(current_node, successor)
                for rel_type, rel_props in edge_data.items():
                    if rel_type == 'IS_PARENT_OF' or str(rel_type).endswith('IS_PARENT_OF'):
                        node_data = dict(self.graph.nodes[successor])
                        descendants.append({
                            'node_id': successor,
                            'node_type': node_data.get('node_type'),
                            'code': node_data.get('code'),
                            'description': node_data.get('description'),
                            'depth': depth + 1,
                            'relationship': rel_props
                        })
                        traverse_descendants(successor, depth + 1)

        traverse_descendants(node_id)

        return GraphQueryResult(
            query_type=GraphQueryType.HIERARCHICAL_DESCENDANTS,
            results=descendants,
            metadata={
                'start_node': node_id,
                'max_depth': max_depth,
                'total_descendants': len(descendants)
            },
            execution_time=time.time() - start_time
        )

    def query_similar_products(self, node_id: str, min_similarity: float = 0.3) -> GraphQueryResult:
        """
        Find products similar to the given node based on heading/chapter

        Args:
            node_id: Node to find similar products for
            min_similarity: Minimum similarity threshold

        Returns:
            GraphQueryResult with similar products
        """
        import time
        start_time = time.time()

        if node_id not in self.graph:
            return GraphQueryResult(
                query_type=GraphQueryType.SIMILAR_PRODUCTS,
                results=[],
                metadata={'error': f'Node {node_id} not found'},
                execution_time=time.time() - start_time
            )

        similar_products = []

        # For HSN nodes, find other HSN codes in the same heading
        if node_id.startswith('hsn_'):
            node_data = dict(self.graph.nodes[node_id])
            heading_code = node_data.get('heading')

            if heading_code:
                # Find all HSN codes in the same heading
                for other_node_id, other_data in self.graph.nodes(data=True):
                    if (other_node_id.startswith('hsn_') and
                        other_node_id != node_id and
                        other_data.get('heading') == heading_code):

                        similar_products.append({
                            'node_id': other_node_id,
                            'node_type': 'hsn',
                            'code': other_data.get('code'),
                            'description': other_data.get('description'),
                            'similarity_score': 0.8,  # Same heading = high similarity
                            'shared_keywords': [f'heading_{heading_code}'],
                            'relationship': 'same_heading'
                        })

                        # Limit to top 5 similar products
                        if len(similar_products) >= 5:
                            break

        return GraphQueryResult(
            query_type=GraphQueryType.SIMILAR_PRODUCTS,
            results=similar_products,
            metadata={
                'query_node': node_id,
                'min_similarity': min_similarity,
                'total_similar': len(similar_products)
            },
            execution_time=time.time() - start_time
        )

    def query_code_lookup(self, hsn_code: str) -> GraphQueryResult:
        """
        Look up information for a specific HSN code

        Args:
            hsn_code: HSN code to look up

        Returns:
            GraphQueryResult with code information
        """
        import time
        start_time = time.time()

        # Find node with matching code
        matching_node = None
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get('code') == hsn_code:
                matching_node = node_id
                break

        if not matching_node:
            return GraphQueryResult(
                query_type=GraphQueryType.CODE_LOOKUP,
                results=[],
                metadata={'error': f'HSN code {hsn_code} not found'},
                execution_time=time.time() - start_time
            )

        # Get node information
        node_data = dict(self.graph.nodes[matching_node])

        # Get hierarchical context
        ancestors = self.query_hierarchical_ancestors(matching_node)
        descendants = self.query_hierarchical_descendants(matching_node, max_depth=1)

        result = {
            'node_id': matching_node,
            'code': hsn_code,
            'description': node_data.get('description'),
            'title': node_data.get('title'),
            'export_policy': node_data.get('export_policy'),
            'hierarchy_level': node_data.get('level'),
            'ancestors': ancestors.results,
            'descendants': descendants.results,
            'document_content': node_data.get('document_content'),
            'search_keywords': node_data.get('search_keywords')
        }

        return GraphQueryResult(
            query_type=GraphQueryType.CODE_LOOKUP,
            results=[result],
            metadata={
                'hsn_code': hsn_code,
                'found': True,
                'ancestors_count': len(ancestors.results),
                'descendants_count': len(descendants.results)
            },
            execution_time=time.time() - start_time
        )

    def query_hierarchical_ancestors(self, node_id: str) -> GraphQueryResult:
        """
        Find all hierarchical ancestors of a node

        Args:
            node_id: Starting node ID

        Returns:
            GraphQueryResult with ancestor information
        """
        import time
        start_time = time.time()

        if node_id not in self.graph:
            return GraphQueryResult(
                query_type=GraphQueryType.HIERARCHICAL_ANCESTORS,
                results=[],
                metadata={'error': f'Node {node_id} not found'},
                execution_time=time.time() - start_time
            )

        ancestors = []
        current_node = node_id

        # For HSN nodes, find their heading and chapter
        if node_id.startswith('hsn_'):
            node_data = dict(self.graph.nodes[current_node])

            # Find heading ancestor
            heading_code = node_data.get('heading')
            if heading_code:
                heading_id = f"heading_{heading_code}"
                if self.graph.has_node(heading_id):
                    heading_data = dict(self.graph.nodes[heading_id])
                    ancestors.append({
                        'node_id': heading_id,
                        'node_type': 'heading',
                        'code': heading_code,
                        'description': heading_data.get('description', f'Heading {heading_code}'),
                        'depth': 1,
                        'relationship': 'CONTAINS_HSN'
                    })

                    # Find chapter ancestor
                    chapter_code = node_data.get('chapter')
                    if chapter_code:
                        chapter_id = f"chapter_{chapter_code}"
                        if self.graph.has_node(chapter_id):
                            chapter_data = dict(self.graph.nodes[chapter_id])
                            ancestors.append({
                                'node_id': chapter_id,
                                'node_type': 'chapter',
                                'code': chapter_code,
                                'description': chapter_data.get('description', f'Chapter {chapter_code}'),
                                'depth': 2,
                                'relationship': 'CONTAINS_HEADING'
                            })

        return GraphQueryResult(
            query_type=GraphQueryType.HIERARCHICAL_ANCESTORS,
            results=ancestors,
            metadata={
                'start_node': node_id,
                'total_ancestors': len(ancestors)
            },
            execution_time=time.time() - start_time
        )

    def save_graph(self, filepath: Path) -> bool:
        """
        Save the graph to a pickle file

        Args:
            filepath: Path to save the graph

        Returns:
            True if saved successfully
        """
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.graph, f)
            print(f"SUCCESS: Graph saved to {filepath}")
            return True
        except Exception as e:
            print(f"ERROR: Failed to save graph: {str(e)}")
            return False

    def load_graph(self, filepath: Path) -> bool:
        """
        Load the graph from a pickle file

        Args:
            filepath: Path to load the graph from

        Returns:
            True if loaded successfully
        """
        try:
            print(f"DEBUG: Attempting to load graph from {filepath}")
            print(f"DEBUG: File exists: {filepath.exists()}")
            print(f"DEBUG: File size: {filepath.stat().st_size if filepath.exists() else 0} bytes")

            with open(filepath, 'rb') as f:
                self.graph = pickle.load(f)

            print(f"SUCCESS: Graph loaded from {filepath}")
            print(f"DEBUG: Loaded graph has {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            return True
        except FileNotFoundError:
            print(f"ERROR: Graph file not found: {filepath}")
            return False
        except pickle.UnpicklingError as e:
            print(f"ERROR: Failed to unpickle graph file: {e}")
            return False
        except Exception as e:
            print(f"ERROR: Failed to load graph: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the graph

        Returns:
            Dictionary with graph statistics
        """
        stats = {
            'nodes': {
                'total': self.graph.number_of_nodes(),
                'by_type': {}
            },
            'edges': {
                'total': self.graph.number_of_edges(),
                'by_type': {}
            },
            'connectivity': {
                'connected_components': nx.number_weakly_connected_components(self.graph),
                'average_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0
            },
            'hierarchy': {
                'max_depth': self._calculate_max_depth(),
                'leaf_nodes': len([n for n in self.graph.nodes() if self.graph.out_degree(n) == 0])
            }
        }

        # Count nodes by type
        for node_id, node_data in self.graph.nodes(data=True):
            node_type = node_data.get('node_type', 'unknown')
            stats['nodes']['by_type'][node_type] = stats['nodes']['by_type'].get(node_type, 0) + 1

        # Count edges by type
        for source, target, key, data in self.graph.edges(keys=True, data=True):
            stats['edges']['by_type'][key] = stats['edges']['by_type'].get(key, 0) + 1

        return stats

    def _calculate_max_depth(self) -> int:
        """Calculate the maximum depth of the hierarchy"""
        max_depth = 0

        # Find root nodes (chapters)
        root_nodes = [n for n in self.graph.nodes() if n.startswith('chapter_')]

        for root in root_nodes:
            depths = nx.shortest_path_length(self.graph, source=root)
            if depths:
                max_depth = max(max_depth, max(depths.values()))

        return max_depth

    def enhance_graph_with_llm(self, product_data: List[Dict[str, Any]]) -> bool:
        """
        Enhance the knowledge graph with LLM-generated relationships

        Args:
            product_data: List of product dictionaries to analyze

        Returns:
            True if enhancement successful
        """
        if not self.llm_client:
            logging.warning("LLM client not available for graph enhancement")
            return False

        try:
            logging.info(f"Enhancing graph with LLM for {len(product_data)} products")

            for product in product_data:
                # Generate relationships using LLM
                relationships = self.llm_client.generate_product_relationships(product)

                # Add LLM-generated relationships to graph
                self._add_llm_relationships_to_graph(product, relationships)

            logging.info("SUCCESS: Graph enhancement with LLM completed")
            return True

        except Exception as e:
            logging.error(f"Error enhancing graph with LLM: {str(e)}")
            return False

    def _add_llm_relationships_to_graph(self, product: Dict[str, Any], relationships: Dict[str, Any]):
        """Add LLM-generated relationships to the NetworkX graph"""
        product_id = f"hsn_{product.get('hsn_code', 'unknown')}"

        # Add categories as nodes and relationships
        for category in relationships.get('categories', []):
            category_id = f"category_{category.lower().replace(' ', '_')}"

            # Add category node if it doesn't exist
            if not self.graph.has_node(category_id):
                self.graph.add_node(category_id,
                                  node_type='category',
                                  name=category,
                                  description=f"Category: {category}")

            # Add relationship from product to category
            if not self.graph.has_edge(product_id, category_id):
                self.graph.add_edge(product_id, category_id,
                                  key='BELONGS_TO_CATEGORY',
                                  confidence=relationships.get('confidence_score', 0.5),
                                  source='llm_generated')

        # Add similar product relationships
        for similar_product in relationships.get('similar_products', []):
            similar_id = f"similar_{similar_product.lower().replace(' ', '_')}"

            if not self.graph.has_node(similar_id):
                self.graph.add_node(similar_id,
                                  node_type='similar_product',
                                  name=similar_product,
                                  description=f"Similar to: {similar_product}")

            if not self.graph.has_edge(product_id, similar_id):
                self.graph.add_edge(product_id, similar_id,
                                  key='HAS_SIMILAR_PRODUCT',
                                  confidence=relationships.get('confidence_score', 0.5),
                                  source='llm_generated')

    def query_with_llm_enhancement(self, query: str, base_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Enhance query results with LLM-generated insights

        Args:
            query: Original query string
            base_results: Base results from traditional graph query

        Returns:
            Enhanced results with LLM insights
        """
        if not self.llm_client:
            return {"enhanced_results": base_results, "llm_insights": None}

        try:
            # Use LLM to generate additional insights
            llm_response = self.llm_client.generate_response(
                query=query,
                retrieved_docs=[],
                graph_context=base_results,
                hsn_result=base_results[0] if base_results else None
            )

            return {
                "enhanced_results": base_results,
                "llm_insights": llm_response,
                "enhancement_type": "llm_enhanced"
            }

        except Exception as e:
            logging.error(f"Error in LLM enhancement: {str(e)}")
            return {"enhanced_results": base_results, "llm_insights": None, "error": str(e)}

    def generate_llm_schema_suggestions(self) -> Dict[str, Any]:
        """
        Use LLM to suggest improvements to the graph schema

        Returns:
            Dictionary with schema improvement suggestions
        """
        if not self.llm_client:
            return {"error": "LLM client not available"}

        try:
            # Get current schema info
            current_schema = {
                "node_types": list(self.node_types.keys()),
                "relationship_types": list(self.relationship_types.keys()),
                "total_nodes": self.graph.number_of_nodes(),
                "total_edges": self.graph.number_of_edges()
            }

            suggestions = self.llm_client.generate_graph_schema_suggestions(current_schema)
            return suggestions

        except Exception as e:
            logging.error(f"Error generating schema suggestions: {str(e)}")
            return {"error": str(e)}

def create_visualization(graph: HSNKnowledgeGraph, output_file: Path = None) -> plt.Figure:
    """
    Create a visualization of the knowledge graph

    Args:
        graph: HSNKnowledgeGraph instance
        output_file: Optional path to save the visualization

    Returns:
        Matplotlib figure object
    """
    plt.figure(figsize=(15, 10))

    # Create position layout
    pos = nx.spring_layout(graph.graph, k=2, iterations=50, seed=42)

    # Define colors for different node types
    node_colors = {
        'chapter': 'red',
        'heading': 'orange',
        'subheading': 'yellow',
        'hsn': 'green'
    }

    # Get node colors
    colors = []
    for node_id in graph.graph.nodes():
        node_type = graph.graph.nodes[node_id].get('node_type', 'unknown')
        colors.append(node_colors.get(node_type, 'gray'))

    # Draw nodes
    nx.draw_networkx_nodes(graph.graph, pos, node_color=colors, node_size=300, alpha=0.7)

    # Draw edges by type
    edge_types = ['IS_PARENT_OF', 'BELONGS_TO_CHAPTER', 'HAS_SIMILAR_DESCRIPTION', 'SHARES_EXPORT_POLICY']

    for i, edge_type in enumerate(edge_types):
        edges = [(u, v) for u, v, k, d in graph.graph.edges(keys=True, data=True) if k == edge_type]
        if edges:
            nx.draw_networkx_edges(graph.graph, pos, edgelist=edges, alpha=0.3,
                                 edge_color=plt.cm.tab10(i), width=1, label=edge_type)

    # Draw labels (only for important nodes to avoid clutter)
    labels = {}
    for node_id, node_data in graph.graph.nodes(data=True):
        if node_data.get('node_type') in ['chapter', 'heading']:
            code = node_data.get('code', '')
            labels[node_id] = f"{node_data.get('node_type')[:3]}\n{code}"

    nx.draw_networkx_labels(graph.graph, pos, labels, font_size=8, font_weight='bold')

    plt.title("HSN Knowledge Graph\nRed: Chapters, Orange: Headings, Yellow: Subheadings, Green: HSN Codes")
    plt.legend()
    plt.axis('off')

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"SUCCESS: Visualization saved to {output_file}")

    return plt.gcf()

def main():
    """Main execution function for graph implementation."""
    print("Starting HSN Knowledge Graph Implementation (Phase 2.2)")
    print("=" * 70)

    # Create output directories
    MODELS_DIR.mkdir(exist_ok=True)

    # Initialize graph
    kg = HSNKnowledgeGraph()

    try:
        # Load schema
        print("\n1. Loading graph schema...")
        if not kg.load_schema(SCHEMA_FILE):
            raise Exception("Failed to load schema")

        # Build graph
        print("\n2. Building NetworkX graph...")
        if not kg.build_graph():
            raise Exception("Failed to build graph")

        # Validate graph
        print("\n3. Validating graph...")
        validation = kg.validate_graph()
        print(f"Validation score: {validation['validation_score']:.2f}")
        print(f"Nodes: {validation['node_count']}, Edges: {validation['edge_count']}")

        # Get statistics
        print("\n4. Analyzing graph statistics...")
        stats = kg.get_graph_statistics()
        print(f"Graph Statistics:")
        print(f"  Nodes: {stats['nodes']['total']} ({stats['nodes']['by_type']})")
        print(f"  Edges: {stats['edges']['total']} ({stats['edges']['by_type']})")
        print(f"  Max hierarchy depth: {stats['hierarchy']['max_depth']}")
        print(f"  Leaf nodes: {stats['hierarchy']['leaf_nodes']}")

        # Test queries
        print("\n5. Testing graph queries...")

        # Test hierarchical query
        heading_descendants = kg.query_hierarchical_descendants('heading_4001', max_depth=2)
        print(f"Heading 4001 has {len(heading_descendants.results)} descendants")

        chapter_descendants = kg.query_hierarchical_descendants('chapter_40', max_depth=2)
        print(f"Chapter 40 has {len(chapter_descendants.results)} descendants")

        # Test code lookup
        code_info = kg.query_code_lookup('40011010')
        if code_info.results:
            print(f"Found HSN code 40011010: {code_info.results[0]['description'][:50]}...")

        # Test similarity query
        similar = kg.query_similar_products('hsn_40011010')
        print(f"HSN 40011010 has {len(similar.results)} similar products")

        # Save graph
        print("\n6. Saving graph...")
        kg.save_graph(GRAPH_FILE)

        # Create visualization
        print("\n7. Creating visualization...")
        vis_file = DATA_DIR / "hsn_graph_visualization.png"
        create_visualization(kg, vis_file)

        # Summary
        print("\n" + "=" * 70)
        print("PHASE 2.2 GRAPH IMPLEMENTATION COMPLETE")
        print("=" * 70)
        print(f"SUCCESS: NetworkX graph created with {kg.graph.number_of_nodes()} nodes")
        print(f"SUCCESS: {kg.graph.number_of_edges()} relationships established")
        print(f"SUCCESS: Graph validation score: {validation['validation_score']:.2f}")
        print(f"Graph saved to: {GRAPH_FILE}")
        print(f"Visualization saved to: {vis_file}")
        print("Ready to proceed to Phase 2.3: Graph Visualization")
        print("=" * 70)

    except Exception as e:
        print(f"ERROR: Error in graph implementation: {str(e)}")
        raise

if __name__ == "__main__":
    main()