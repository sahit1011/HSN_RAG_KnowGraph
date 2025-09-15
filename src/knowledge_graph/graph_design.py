#!/usr/bin/env python3
"""
HSN Knowledge Graph Design - Phase 2.1
Combines rule-based and LLM approaches for graph schema design
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import json
from pathlib import Path
import re
from dataclasses import dataclass, asdict
from enum import Enum
from tqdm import tqdm

# Import LLM client for enhanced relationship discovery
from ..utils.llm_client import LangChainLLMClient

# File paths
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"
DATA_DIR = OUTPUT_DIR / "data"
MODELS_DIR = OUTPUT_DIR / "models"
ENHANCED_DATA_PATH = DATA_DIR / "extraction_complete.csv"

class NodeType(Enum):
    CHAPTER = "chapter"
    HEADING = "heading"
    SUBHEADING = "subheading"
    HSN_CODE = "hsn_code"

class RelationshipType(Enum):
    BELONGS_TO_CHAPTER = "BELONGS_TO_CHAPTER"
    IS_PARENT_OF = "IS_PARENT_OF"
    IS_CHILD_OF = "IS_CHILD_OF"
    HAS_SIMILAR_DESCRIPTION = "HAS_SIMILAR_DESCRIPTION"
    SHARES_EXPORT_POLICY = "SHARES_EXPORT_POLICY"

@dataclass
class NodeProperties:
    """Properties for graph nodes"""
    id: str
    code: str
    level: str
    description: str
    title: str
    export_policy: str = ""
    policy_condition: str = ""
    document_content: str = ""
    search_keywords: str = ""
    parent_codes: str = ""
    child_codes: str = ""
    hierarchy_path: str = ""
    completeness_score: float = 0.0
    # Hierarchical properties for HSN nodes
    chapter: str = ""
    heading: str = ""
    subheading: str = ""

@dataclass
class Relationship:
    """Graph relationship definition"""
    source_id: str
    target_id: str
    type: RelationshipType
    properties: Dict[str, Any] = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}

@dataclass
class GraphSchema:
    """Complete graph schema definition"""
    nodes: Dict[str, NodeProperties]
    relationships: List[Relationship]
    node_types: Dict[NodeType, Dict[str, str]]
    relationship_types: Dict[RelationshipType, Dict[str, str]]
    metadata: Dict[str, Any]

def load_enhanced_data() -> pd.DataFrame:
    """Load the enhanced HSN data for graph construction."""
    if ENHANCED_DATA_PATH.exists():
        df = pd.read_csv(ENHANCED_DATA_PATH)
        print(f"SUCCESS: Loaded {len(df)} records from {ENHANCED_DATA_PATH}")
        return df
    else:
        raise FileNotFoundError(f"Enhanced data not found at {ENHANCED_DATA_PATH}")

def design_rule_based_schema(df: pd.DataFrame) -> GraphSchema:
    """
    Design graph schema using rule-based approach.

    Args:
        df: Enhanced HSN DataFrame

    Returns:
        GraphSchema: Complete graph schema definition
    """

    print("Designing rule-based graph schema...")

    nodes = {}
    relationships = []

    # Define node type schemas
    node_types = {
        NodeType.CHAPTER: {
            "description": "Top-level HSN chapter (2-digit codes)",
            "properties": ["id", "code", "description", "title", "export_policy", "child_codes", "hierarchy_path"]
        },
        NodeType.HEADING: {
            "description": "HSN heading level (4-digit codes)",
            "properties": ["id", "code", "description", "title", "export_policy", "parent_codes", "child_codes", "hierarchy_path"]
        },
        NodeType.SUBHEADING: {
            "description": "HSN subheading level (6-digit codes)",
            "properties": ["id", "code", "description", "title", "export_policy", "parent_codes", "child_codes", "hierarchy_path"]
        },
        NodeType.HSN_CODE: {
            "description": "Complete HSN codes (8-digit codes)",
            "properties": ["id", "code", "description", "title", "export_policy", "policy_condition", "document_content", "search_keywords", "parent_codes", "hierarchy_path", "completeness_score"]
        }
    }

    # Define relationship type schemas
    relationship_types = {
        RelationshipType.BELONGS_TO_CHAPTER: {
            "description": "Entity belongs to a specific chapter",
            "direction": "directed",
            "properties": ["chapter_code"]
        },
        RelationshipType.IS_PARENT_OF: {
            "description": "Hierarchical parent-child relationship",
            "direction": "directed",
            "properties": ["hierarchy_level", "code_prefix"]
        },
        RelationshipType.IS_CHILD_OF: {
            "description": "Reverse hierarchical relationship",
            "direction": "directed",
            "properties": ["hierarchy_level", "code_prefix"]
        },
        RelationshipType.HAS_SIMILAR_DESCRIPTION: {
            "description": "Semantic similarity between descriptions",
            "direction": "undirected",
            "properties": ["similarity_score", "shared_keywords"]
        },
        RelationshipType.SHARES_EXPORT_POLICY: {
            "description": "Entities with same export policy",
            "direction": "undirected",
            "properties": ["policy_type", "policy_details"]
        }
    }

    # Process each row to create nodes
    for idx, row in tqdm(df.iterrows(), desc="Creating nodes", total=len(df), unit="row"):
        node_id = None
        node_type = None

        # Determine node type and ID based on code level
        # Clean codes to remove .0 suffix from floats
        def clean_code(code):
            if pd.isna(code):
                return None
            code_str = str(code)
            if code_str.endswith('.0'):
                return code_str[:-2]
            return code_str

        node_code = None
        if row.get('code_level') == '2_digit':
            node_code = clean_code(row.get('chapter') or row.get('hsn_code'))
            node_id = f"chapter_{node_code}"
            node_type = NodeType.CHAPTER
        elif row.get('code_level') == '4_digit':
            node_code = clean_code(row.get('heading') or row.get('hsn_code'))
            node_id = f"heading_{node_code}"
            node_type = NodeType.HEADING
        elif row.get('code_level') == '6_digit':
            node_code = clean_code(row.get('subheading') or row.get('hsn_code'))
            node_id = f"subheading_{node_code}"
            node_type = NodeType.SUBHEADING
        elif row.get('code_level') == '8_digit':
            node_code = clean_code(row.get('full_hsn_code') or row.get('hsn_code'))
            node_id = f"hsn_{node_code}"
            node_type = NodeType.HSN_CODE

        if node_id and node_type:
            # Create node properties with corrected code
            properties = NodeProperties(
                id=node_id,
                code=node_code if node_code else str(row.get('full_hsn_code') or row.get('subheading') or row.get('heading') or row.get('chapter', '')),
                level=row.get('code_level', ''),
                description=str(row.get('description', '')),
                title=str(row.get('chapter_title') or row.get('heading_title') or row.get('subheading_title') or ''),
                export_policy=str(row.get('export_policy', '')),
                policy_condition=str(row.get('policy_condition', '')),
                document_content=str(row.get('document_content', '')),
                search_keywords=str(row.get('search_keywords', '')),
                parent_codes=str(row.get('parent_codes', '')),
                child_codes=str(row.get('child_codes', '')),
                hierarchy_path=str(row.get('full_hierarchy_path', '')),
                completeness_score=0.0
            )

            # Add hierarchical properties for HSN nodes to support graph queries
            if node_type == NodeType.HSN_CODE and node_code:
                code_str = str(node_code).strip()
                if len(code_str) >= 8:
                    # Extract hierarchical codes from HSN code
                    properties.chapter = code_str[:2]  # First 2 digits
                    properties.heading = code_str[:4]  # First 4 digits
                    properties.subheading = code_str[:6]  # First 6 digits
                elif len(code_str) >= 6:
                    properties.chapter = code_str[:2]
                    properties.heading = code_str[:4]
                    properties.subheading = code_str[:6]
                elif len(code_str) >= 4:
                    properties.chapter = code_str[:2]
                    properties.heading = code_str[:4]
                elif len(code_str) >= 2:
                    properties.chapter = code_str[:2]

            nodes[node_id] = properties

    # Create hierarchical relationships
    print("Creating hierarchical relationships...")

    # Group nodes by type for relationship creation
    chapters = {k: v for k, v in nodes.items() if k.startswith('chapter_')}
    headings = {k: v for k, v in nodes.items() if k.startswith('heading_')}
    subheadings = {k: v for k, v in nodes.items() if k.startswith('subheading_')}
    hsn_codes = {k: v for k, v in nodes.items() if k.startswith('hsn_')}

    # Chapter -> Heading relationships
    for heading_id, heading_props in tqdm(headings.items(), desc="Creating chapter-heading relationships", unit="heading"):
        # Extract chapter code from heading code (first 2 digits)
        heading_code = str(heading_props.code)
        if len(heading_code) >= 2:
            chapter_code = heading_code[:2]
            chapter_id = f"chapter_{chapter_code}"

            if chapter_id in chapters:
                # IS_PARENT_OF: Chapter -> Heading
                relationships.append(Relationship(
                    source_id=chapter_id,
                    target_id=heading_id,
                    type=RelationshipType.IS_PARENT_OF,
                    properties={
                        "hierarchy_level": "chapter_to_heading",
                        "code_prefix": chapter_code
                    }
                ))

                # IS_CHILD_OF: Heading -> Chapter
                relationships.append(Relationship(
                    source_id=heading_id,
                    target_id=chapter_id,
                    type=RelationshipType.IS_CHILD_OF,
                    properties={
                        "hierarchy_level": "heading_to_chapter",
                        "code_prefix": chapter_code
                    }
                ))

                # BELONGS_TO_CHAPTER: Heading -> Chapter
                relationships.append(Relationship(
                    source_id=heading_id,
                    target_id=chapter_id,
                    type=RelationshipType.BELONGS_TO_CHAPTER,
                    properties={"chapter_code": chapter_code}
                ))

    # Heading -> Subheading relationships
    for subheading_id, subheading_props in tqdm(subheadings.items(), desc="Creating heading-subheading relationships", unit="subheading"):
        # Extract heading code from subheading code (first 4 digits)
        subheading_code = str(subheading_props.code)
        if len(subheading_code) >= 4:
            heading_code = subheading_code[:4]
            heading_id = f"heading_{heading_code}"

            if heading_id in headings:
                # IS_PARENT_OF: Heading -> Subheading
                relationships.append(Relationship(
                    source_id=heading_id,
                    target_id=subheading_id,
                    type=RelationshipType.IS_PARENT_OF,
                    properties={
                        "hierarchy_level": "heading_to_subheading",
                        "code_prefix": heading_code
                    }
                ))

                # IS_CHILD_OF: Subheading -> Heading
                relationships.append(Relationship(
                    source_id=subheading_id,
                    target_id=heading_id,
                    type=RelationshipType.IS_CHILD_OF,
                    properties={
                        "hierarchy_level": "subheading_to_heading",
                        "code_prefix": heading_code
                    }
                ))

                # BELONGS_TO_CHAPTER: Subheading -> Chapter
                chapter_code = subheading_code[:2]
                chapter_id = f"chapter_{chapter_code}"
                if chapter_id in chapters:
                    relationships.append(Relationship(
                        source_id=subheading_id,
                        target_id=chapter_id,
                        type=RelationshipType.BELONGS_TO_CHAPTER,
                        properties={"chapter_code": chapter_code}
                    ))

    # Subheading -> HSN Code relationships
    for hsn_id, hsn_props in tqdm(hsn_codes.items(), desc="Creating subheading-HSN relationships", unit="hsn"):
        # Extract subheading code from HSN code (first 6 digits)
        hsn_code = str(hsn_props.code)
        if len(hsn_code) >= 6:
            subheading_code = hsn_code[:6]
            subheading_id = f"subheading_{subheading_code}"

            if subheading_id in subheadings:
                # IS_PARENT_OF: Subheading -> HSN Code
                relationships.append(Relationship(
                    source_id=subheading_id,
                    target_id=hsn_id,
                    type=RelationshipType.IS_PARENT_OF,
                    properties={
                        "hierarchy_level": "subheading_to_hsn",
                        "code_prefix": subheading_code
                    }
                ))

                # IS_CHILD_OF: HSN Code -> Subheading
                relationships.append(Relationship(
                    source_id=hsn_id,
                    target_id=subheading_id,
                    type=RelationshipType.IS_CHILD_OF,
                    properties={
                        "hierarchy_level": "hsn_to_subheading",
                        "code_prefix": subheading_code
                    }
                ))

                # BELONGS_TO_CHAPTER: HSN Code -> Chapter
                chapter_code = hsn_code[:2]
                chapter_id = f"chapter_{chapter_code}"
                if chapter_id in chapters:
                    relationships.append(Relationship(
                        source_id=hsn_id,
                        target_id=chapter_id,
                        type=RelationshipType.BELONGS_TO_CHAPTER,
                        properties={"chapter_code": chapter_code}
                    ))

    # Create metadata
    metadata = {
        "total_nodes": len(nodes),
        "total_relationships": len(relationships),
        "node_counts": {
            "chapters": len(chapters),
            "headings": len(headings),
            "subheadings": len(subheadings),
            "hsn_codes": len(hsn_codes)
        },
        "relationship_counts": {
            rel_type.value: sum(1 for r in relationships if r.type == rel_type)
            for rel_type in RelationshipType
        },
        "design_method": "rule_based",
        "created_at": pd.Timestamp.now().isoformat()
    }

    schema = GraphSchema(
        nodes=nodes,
        relationships=relationships,
        node_types=node_types,
        relationship_types=relationship_types,
        metadata=metadata
    )

    print(f"SUCCESS: Rule-based schema created: {len(nodes)} nodes, {len(relationships)} relationships")
    return schema

def enhance_schema_with_llm(schema: GraphSchema, df: pd.DataFrame) -> GraphSchema:
    """
    Enhance the rule-based schema with LLM-assisted relationship discovery.

    Args:
        schema: Base rule-based schema
        df: Enhanced HSN DataFrame

    Returns:
        Enhanced GraphSchema with LLM-discovered relationships
    """

    print("Enhancing schema with LLM-assisted relationship discovery...")

    enhanced_relationships = schema.relationships.copy()

    try:
        # Initialize LLM client
        llm_client = LangChainLLMClient()
        print("SUCCESS: LLM client initialized for relationship discovery")

        # Get HSN nodes for LLM analysis (focus on detailed product codes)
        hsn_nodes = {node_id: props for node_id, props in schema.nodes.items()
                    if node_id.startswith('hsn_') and props.description.strip()}

        print(f"Analyzing {len(hsn_nodes)} HSN nodes with LLM...")

        # Process nodes in batches to avoid overwhelming the LLM
        batch_size = 5
        processed_count = 0

        for i in tqdm(range(0, len(hsn_nodes), batch_size), desc="LLM relationship discovery"):
            batch_nodes = list(hsn_nodes.items())[i:i+batch_size]

            for node_id1, props1 in batch_nodes:
                # Use LLM to analyze product relationships
                product_data = {
                    'description': props1.description,
                    'hsn_code': props1.code,
                    'category': props1.title or 'HSN Product'
                }

                try:
                    # Get LLM-generated relationships for this product
                    llm_relationships = llm_client.generate_product_relationships(product_data)

                    # Extract similar products and create relationships
                    similar_products = llm_relationships.get('similar_products', [])
                    for similar_product in similar_products[:3]:  # Limit to top 3
                        # Find matching nodes in the schema
                        matching_nodes = find_nodes_by_description(schema, similar_product)

                        for matching_node_id in matching_nodes[:2]:  # Limit connections per product
                            if matching_node_id != node_id1:  # Don't connect to self
                                enhanced_relationships.append(Relationship(
                                    source_id=node_id1,
                                    target_id=matching_node_id,
                                    type=RelationshipType.HAS_SIMILAR_DESCRIPTION,
                                    properties={
                                        "similarity_score": 0.8,  # High confidence from LLM
                                        "llm_discovered": True,
                                        "similar_product": similar_product,
                                        "relationship_reason": "LLM-identified semantic similarity"
                                    }
                                ))

                    # Extract categories and create cross-domain relationships
                    categories = llm_relationships.get('categories', [])
                    for category in categories[:2]:  # Limit categories
                        # Find nodes in the same category
                        category_nodes = find_nodes_by_category(schema, category)

                        for category_node_id in category_nodes[:3]:  # Limit connections
                            if category_node_id != node_id1:
                                # Create a new relationship type for category similarity
                                enhanced_relationships.append(Relationship(
                                    source_id=node_id1,
                                    target_id=category_node_id,
                                    type=RelationshipType.HAS_SIMILAR_DESCRIPTION,  # Could be extended to new type
                                    properties={
                                        "similarity_score": 0.7,
                                        "llm_discovered": True,
                                        "shared_category": category,
                                        "relationship_reason": f"LLM-identified category similarity: {category}"
                                    }
                                ))

                except Exception as e:
                    print(f"WARNING: LLM analysis failed for {node_id1}: {str(e)}")
                    continue

            processed_count += len(batch_nodes)
            print(f"Processed {processed_count}/{len(hsn_nodes)} nodes with LLM")

        # Cross-domain relationship discovery
        print("Discovering cross-domain relationships...")
        cross_domain_relationships = discover_cross_domain_relationships_llm(llm_client, schema, hsn_nodes)
        enhanced_relationships.extend(cross_domain_relationships)

        # Semantic similarity analysis between different levels
        print("Analyzing semantic similarities across hierarchy levels...")
        semantic_relationships = discover_semantic_relationships_llm(llm_client, schema)
        enhanced_relationships.extend(semantic_relationships)

        # Keep the original export policy relationships (they're still valuable)
        policy_groups = {}
        for node_id, node_props in schema.nodes.items():
            policy = node_props.export_policy.strip()
            if policy:
                if policy not in policy_groups:
                    policy_groups[policy] = []
                policy_groups[policy].append(node_id)

        for policy, node_ids in policy_groups.items():
            if len(node_ids) > 1:
                for i, node_id1 in enumerate(node_ids):
                    for node_id2 in node_ids[i+1:]:
                        enhanced_relationships.append(Relationship(
                            source_id=node_id1,
                            target_id=node_id2,
                            type=RelationshipType.SHARES_EXPORT_POLICY,
                            properties={
                                "policy_type": policy,
                                "policy_details": f"Shared export policy: {policy}"
                            }
                        ))

        # Update metadata with LLM enhancement details
        schema.metadata["llm_enhanced"] = True
        schema.metadata["llm_model_used"] = "openrouter_llm"
        schema.metadata["total_relationships_llm"] = len(enhanced_relationships)
        schema.metadata["new_relationships"] = len(enhanced_relationships) - len(schema.relationships)
        schema.metadata["llm_processed_nodes"] = processed_count
        schema.metadata["llm_discovery_timestamp"] = pd.Timestamp.now().isoformat()

        schema.relationships = enhanced_relationships

        print(f"SUCCESS: Schema enhanced with {len(enhanced_relationships) - len(schema.relationships)} additional LLM-discovered relationships")

    except Exception as e:
        print(f"ERROR: LLM enhancement failed: {str(e)}")
        print("Falling back to basic enhancement...")

        # Fallback to basic keyword similarity if LLM fails
        enhanced_relationships = fallback_similarity_detection(schema)
        schema.metadata["llm_enhanced"] = False
        schema.metadata["llm_error"] = str(e)
        schema.metadata["fallback_used"] = True
        schema.relationships = enhanced_relationships

    return schema

def find_nodes_by_description(schema: GraphSchema, description: str, max_results: int = 5) -> List[str]:
    """
    Find nodes that match a given description using fuzzy matching.

    Args:
        schema: Graph schema
        description: Description to search for
        max_results: Maximum number of results to return

    Returns:
        List of matching node IDs
    """
    matching_nodes = []
    search_terms = set(description.lower().split())

    for node_id, node_props in schema.nodes.items():
        node_desc = node_props.description.lower()
        node_title = node_props.title.lower()

        # Check if search terms appear in description or title
        desc_matches = sum(1 for term in search_terms if term in node_desc)
        title_matches = sum(1 for term in search_terms if term in node_title)

        # Calculate match score
        total_terms = len(search_terms)
        match_score = (desc_matches + title_matches) / total_terms if total_terms > 0 else 0

        if match_score > 0.3:  # 30% term match threshold
            matching_nodes.append((node_id, match_score))

    # Sort by match score and return top results
    matching_nodes.sort(key=lambda x: x[1], reverse=True)
    return [node_id for node_id, score in matching_nodes[:max_results]]

def find_nodes_by_category(schema: GraphSchema, category: str, max_results: int = 5) -> List[str]:
    """
    Find nodes that belong to a specific category.

    Args:
        schema: Graph schema
        category: Category to search for
        max_results: Maximum number of results to return

    Returns:
        List of matching node IDs
    """
    matching_nodes = []
    category_lower = category.lower()

    for node_id, node_props in schema.nodes.items():
        # Check if category appears in title, description, or search keywords
        title_match = category_lower in node_props.title.lower() if node_props.title else False
        desc_match = category_lower in node_props.description.lower() if node_props.description else False
        keyword_match = any(category_lower in keyword.lower() for keyword in
                          (node_props.search_keywords.split(', ') if node_props.search_keywords else []))

        if title_match or desc_match or keyword_match:
            matching_nodes.append(node_id)

        if len(matching_nodes) >= max_results:
            break

    return matching_nodes[:max_results]

def discover_cross_domain_relationships_llm(llm_client: LangChainLLMClient, schema: GraphSchema, hsn_nodes: Dict[str, NodeProperties]) -> List[Relationship]:
    """
    Discover cross-domain relationships using LLM analysis.

    Args:
        llm_client: Initialized LLM client
        schema: Graph schema
        hsn_nodes: HSN nodes to analyze

    Returns:
        List of discovered relationships
    """
    relationships = []

    try:
        # Sample a few nodes from different chapters for cross-domain analysis
        chapter_groups = {}
        for node_id, props in hsn_nodes.items():
            chapter = props.chapter
            if chapter not in chapter_groups:
                chapter_groups[chapter] = []
            chapter_groups[chapter].append((node_id, props))

        # Get 2 nodes from each chapter for cross-domain analysis
        cross_domain_candidates = []
        for chapter, nodes in chapter_groups.items():
            cross_domain_candidates.extend(nodes[:2])

        print(f"Analyzing {len(cross_domain_candidates)} nodes for cross-domain relationships...")

        # Analyze cross-domain relationships in small batches
        for i in range(0, len(cross_domain_candidates), 3):
            batch = cross_domain_candidates[i:i+3]

            for j, (node_id1, props1) in enumerate(batch):
                for node_id2, props2 in batch[j+1:]:
                    # Skip if same chapter
                    if props1.chapter == props2.chapter:
                        continue

                    # Use LLM to check for cross-domain relationships
                    prompt = f"""
                    Analyze if these two products from different HSN chapters might be related:

                    Product 1: {props1.description} (HSN: {props1.code}, Chapter: {props1.chapter})
                    Product 2: {props2.description} (HSN: {props2.code}, Chapter: {props2.chapter})

                    Are these products related in terms of:
                    1. Manufacturing process similarity?
                    2. Material composition similarity?
                    3. End-use application similarity?
                    4. Supply chain relationship?

                    If they are related, explain how. If not, say "no relationship".

                    Keep response under 100 words.
                    """

                    try:
                        response = llm_client.llm.invoke(prompt).content.strip()

                        if "no relationship" not in response.lower() and len(response) > 20:
                            relationships.append(Relationship(
                                source_id=node_id1,
                                target_id=node_id2,
                                type=RelationshipType.HAS_SIMILAR_DESCRIPTION,
                                properties={
                                    "similarity_score": 0.6,
                                    "llm_discovered": True,
                                    "cross_domain": True,
                                    "relationship_reason": response[:200],
                                    "chapter_difference": f"{props1.chapter} -> {props2.chapter}"
                                }
                            ))
                    except Exception as e:
                        print(f"WARNING: Cross-domain analysis failed for {node_id1}-{node_id2}: {str(e)}")
                        continue

    except Exception as e:
        print(f"ERROR: Cross-domain relationship discovery failed: {str(e)}")

    return relationships

def discover_semantic_relationships_llm(llm_client: LangChainLLMClient, schema: GraphSchema) -> List[Relationship]:
    """
    Discover semantic relationships across different hierarchy levels using LLM.

    Args:
        llm_client: Initialized LLM client
        schema: Graph schema

    Returns:
        List of discovered semantic relationships
    """
    relationships = []

    try:
        # Get chapter and heading nodes for semantic analysis
        chapters = {nid: props for nid, props in schema.nodes.items() if nid.startswith('chapter_')}
        headings = {nid: props for nid, props in schema.nodes.items() if nid.startswith('heading_')}

        print(f"Analyzing semantic relationships between {len(chapters)} chapters and {len(headings)} headings...")

        # Analyze semantic relationships between chapters and their headings
        for chapter_id, chapter_props in chapters.items():
            chapter_code = chapter_props.code

            # Find headings that belong to this chapter
            chapter_headings = []
            for heading_id, heading_props in headings.items():
                if str(heading_props.code).startswith(str(chapter_code)):
                    chapter_headings.append((heading_id, heading_props))

            if len(chapter_headings) > 1:
                # Use LLM to analyze semantic coherence within the chapter
                heading_descriptions = [f"- {props.description}" for _, props in chapter_headings[:5]]

                prompt = f"""
                Analyze the semantic coherence of products within HSN Chapter {chapter_code}:

                Chapter Description: {chapter_props.description}

                Products in this chapter:
                {chr(10).join(heading_descriptions)}

                Are all these products semantically related? Do they share common themes, materials, or applications?

                Provide a brief analysis (under 150 words) of the chapter's semantic coherence.
                """

                try:
                    response = llm_client.llm.invoke(prompt).content.strip()

                    # Create semantic relationships based on LLM analysis
                    for i, (heading_id1, _) in enumerate(chapter_headings):
                        for heading_id2, _ in chapter_headings[i+1:]:
                            relationships.append(Relationship(
                                source_id=heading_id1,
                                target_id=heading_id2,
                                type=RelationshipType.HAS_SIMILAR_DESCRIPTION,
                                properties={
                                    "similarity_score": 0.9,  # High confidence for same chapter
                                    "llm_discovered": True,
                                    "semantic_analysis": response[:300],
                                    "relationship_type": "chapter_coherence",
                                    "shared_chapter": chapter_code
                                }
                            ))

                except Exception as e:
                    print(f"WARNING: Semantic analysis failed for chapter {chapter_code}: {str(e)}")
                    continue

    except Exception as e:
        print(f"ERROR: Semantic relationship discovery failed: {str(e)}")

    return relationships

def fallback_similarity_detection(schema: GraphSchema) -> List[Relationship]:
    """
    Fallback similarity detection using basic keyword matching when LLM fails.

    Args:
        schema: Graph schema

    Returns:
        List of relationships discovered through basic similarity
    """
    print("Using fallback keyword-based similarity detection...")

    enhanced_relationships = schema.relationships.copy()

    # Basic keyword similarity (original implementation)
    nodes_by_level = {}
    for node_id, node_props in schema.nodes.items():
        level = node_props.level
        if level not in nodes_by_level:
            nodes_by_level[level] = []
        nodes_by_level[level].append((node_id, node_props))

    for level, nodes_list in nodes_by_level.items():
        if len(nodes_list) < 2:
            continue

        for i, (node_id1, props1) in enumerate(nodes_list):
            keywords1 = set(props1.search_keywords.lower().split(', ')) if props1.search_keywords else set()

            for j, (node_id2, props2) in enumerate(nodes_list[i+1:], i+1):
                keywords2 = set(props2.search_keywords.lower().split(', ')) if props2.search_keywords else set()

                if keywords1 and keywords2:
                    intersection = keywords1.intersection(keywords2)
                    union = keywords1.union(keywords2)
                    similarity = len(intersection) / len(union) if union else 0

                    if similarity > 0.3:
                        enhanced_relationships.append(Relationship(
                            source_id=node_id1,
                            target_id=node_id2,
                            type=RelationshipType.HAS_SIMILAR_DESCRIPTION,
                            properties={
                                "similarity_score": similarity,
                                "shared_keywords": list(intersection)[:5],
                                "fallback_method": True
                            }
                        ))

    return enhanced_relationships

def validate_graph_schema(schema: GraphSchema, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate the graph schema against the original data.

    Args:
        schema: Graph schema to validate
        df: Original enhanced DataFrame

    Returns:
        Validation results dictionary
    """

    print("Validating graph schema...")

    validation_results = {
        "node_coverage": {},
        "relationship_consistency": {},
        "data_integrity": {},
        "schema_completeness": {},
        "issues": []
    }

    # Check node coverage
    total_records = len(df)
    nodes_created = len(schema.nodes)

    validation_results["node_coverage"] = {
        "total_records": total_records,
        "nodes_created": nodes_created,
        "coverage_rate": nodes_created / total_records if total_records > 0 else 0
    }

    # Check relationship consistency
    rel_counts = {}
    for rel in schema.relationships:
        rel_type = rel.type.value
        rel_counts[rel_type] = rel_counts.get(rel_type, 0) + 1

    validation_results["relationship_consistency"] = {
        "total_relationships": len(schema.relationships),
        "relationship_types": rel_counts,
        "avg_relationships_per_node": len(schema.relationships) / len(schema.nodes) if schema.nodes else 0
    }

    # Check data integrity - ensure all referenced nodes exist
    node_ids = set(schema.nodes.keys())
    missing_sources = set()
    missing_targets = set()

    for rel in schema.relationships:
        if rel.source_id not in node_ids:
            missing_sources.add(rel.source_id)
        if rel.target_id not in node_ids:
            missing_targets.add(rel.target_id)

    validation_results["data_integrity"] = {
        "missing_source_nodes": len(missing_sources),
        "missing_target_nodes": len(missing_targets),
        "integrity_score": 1.0 - (len(missing_sources) + len(missing_targets)) / (2 * len(schema.relationships)) if schema.relationships else 1.0
    }

    if missing_sources or missing_targets:
        validation_results["issues"].extend([
            f"Missing source nodes: {list(missing_sources)[:5]}..." if len(missing_sources) > 5 else f"Missing source nodes: {list(missing_sources)}",
            f"Missing target nodes: {list(missing_targets)[:5]}..." if len(missing_targets) > 5 else f"Missing target nodes: {list(missing_targets)}"
        ])

    # Check schema completeness
    expected_node_types = {nt.value for nt in NodeType}
    actual_node_types = set()

    for node_id in schema.nodes.keys():
        if node_id.startswith('chapter_'):
            actual_node_types.add('chapter')
        elif node_id.startswith('heading_'):
            actual_node_types.add('heading')
        elif node_id.startswith('subheading_'):
            actual_node_types.add('subheading')
        elif node_id.startswith('hsn_'):
            actual_node_types.add('hsn_code')

    validation_results["schema_completeness"] = {
        "expected_node_types": list(expected_node_types),
        "actual_node_types": list(actual_node_types),
        "completeness_score": len(actual_node_types) / len(expected_node_types) if expected_node_types else 1.0
    }

    # Overall validation score
    coverage_score = validation_results["node_coverage"]["coverage_rate"]
    integrity_score = validation_results["data_integrity"]["integrity_score"]
    completeness_score = validation_results["schema_completeness"]["completeness_score"]

    validation_results["overall_score"] = (coverage_score + integrity_score + completeness_score) / 3

    print(f"SUCCESS: Schema validation complete. Overall score: {validation_results['overall_score']:.2f}")
    return validation_results

def export_graph_schema(schema: GraphSchema, validation_results: Dict, output_dir: Path) -> Dict[str, str]:
    """
    Export the graph schema and documentation.

    Args:
        schema: Graph schema to export
        validation_results: Validation results
        output_dir: Output directory

    Returns:
        Dictionary of exported file paths
    """

    print("Exporting graph schema and documentation...")

    exported_files = {}

    # Export nodes as JSON
    nodes_file = output_dir / "graph_nodes.json"
    nodes_data = {node_id: asdict(props) for node_id, props in schema.nodes.items()}
    with open(nodes_file, 'w') as f:
        json.dump(nodes_data, f, indent=2, default=str)
    exported_files["nodes_json"] = str(nodes_file)

    # Export relationships as JSON
    relationships_file = output_dir / "graph_relationships.json"
    relationships_data = [asdict(rel) for rel in schema.relationships]
    with open(relationships_file, 'w') as f:
        json.dump(relationships_data, f, indent=2, default=str)
    exported_files["relationships_json"] = str(relationships_file)

    # Export complete schema
    schema_file = output_dir / "graph_schema.json"
    schema_data = {
        "nodes": nodes_data,
        "relationships": relationships_data,
        "node_types": {nt.value: desc for nt, desc in schema.node_types.items()},
        "relationship_types": {rt.value: desc for rt, desc in schema.relationship_types.items()},
        "metadata": schema.metadata,
        "validation": validation_results
    }
    with open(schema_file, 'w') as f:
        json.dump(schema_data, f, indent=2, default=str)
    exported_files["schema_json"] = str(schema_file)

    # Export schema documentation
    doc_file = output_dir / "graph_schema_documentation.md"
    with open(doc_file, 'w') as f:
        f.write("# HSN Knowledge Graph Schema Documentation\n\n")
        f.write("## Overview\n\n")
        f.write(f"This document describes the knowledge graph schema for HSN (Harmonized System Nomenclature) codes, designed using a hybrid rule-based and LLM-assisted approach.\n\n")
        f.write(f"**Schema Statistics:**\n")
        f.write(f"- Total Nodes: {schema.metadata['total_nodes']}\n")
        f.write(f"- Total Relationships: {schema.metadata['total_relationships_llm'] if 'total_relationships_llm' in schema.metadata else schema.metadata['total_relationships']}\n")
        f.write(f"- Node Types: {len(schema.node_types)}\n")
        f.write(f"- Relationship Types: {len(schema.relationship_types)}\n\n")

        f.write("## Node Types\n\n")
        for node_type, desc in schema.node_types.items():
            f.write(f"### {node_type.value.upper()}\n")
            f.write(f"**Description:** {desc['description']}\n")
            f.write(f"**Properties:** {', '.join(desc['properties'])}\n")
            f.write(f"**Count:** {schema.metadata['node_counts'].get(node_type.value + 's', 0)}\n\n")

        f.write("## Relationship Types\n\n")
        for rel_type, desc in schema.relationship_types.items():
            f.write(f"### {rel_type.value}\n")
            f.write(f"**Description:** {desc['description']}\n")
            f.write(f"**Direction:** {desc['direction']}\n")
            f.write(f"**Properties:** {', '.join(desc['properties'])}\n")
            f.write(f"**Count:** {schema.metadata.get('relationship_counts', {}).get(rel_type.value, 0)}\n\n")

        f.write("## Validation Results\n\n")
        f.write(f"**Overall Score:** {validation_results['overall_score']:.2f}/1.0\n\n")
        f.write("### Coverage Metrics\n")
        for key, value in validation_results['node_coverage'].items():
            f.write(f"- {key}: {value}\n")
        f.write("\n### Relationship Metrics\n")
        for key, value in validation_results['relationship_consistency'].items():
            if isinstance(value, dict):
                f.write(f"- {key}:\n")
                for sub_key, sub_value in value.items():
                    f.write(f"  - {sub_key}: {sub_value}\n")
            else:
                f.write(f"- {key}: {value}\n")
        f.write("\n### Data Integrity\n")
        for key, value in validation_results['data_integrity'].items():
            f.write(f"- {key}: {value}\n")

        if validation_results['issues']:
            f.write("\n### Issues Found\n")
            for issue in validation_results['issues']:
                f.write(f"- {issue}\n")

    exported_files["documentation_md"] = str(doc_file)

    # Export for graph visualization (CSV format)
    nodes_csv = output_dir / "graph_nodes.csv"
    nodes_df = pd.DataFrame([
        {
            "id": node_id,
            "type": node_id.split('_')[0],
            "code": props.code,
            "level": props.level,
            "description": props.description[:100] + "..." if len(props.description) > 100 else props.description,
            "title": props.title[:100] + "..." if len(props.title) > 100 else props.title
        }
        for node_id, props in schema.nodes.items()
    ])
    nodes_df.to_csv(nodes_csv, index=False)
    exported_files["nodes_csv"] = str(nodes_csv)

    relationships_csv = output_dir / "graph_relationships.csv"
    relationships_df = pd.DataFrame([
        {
            "source": rel.source_id,
            "target": rel.target_id,
            "type": rel.type.value,
            "properties": str(rel.properties)
        }
        for rel in schema.relationships
    ])
    relationships_df.to_csv(relationships_csv, index=False)
    exported_files["relationships_csv"] = str(relationships_csv)

    print(f"SUCCESS: Exported {len(exported_files)} schema files")
    return exported_files

def main():
    """Main execution function for graph design."""
    print("Starting HSN Knowledge Graph Design (Phase 2.1)")
    print("=" * 60)

    try:
        # Load enhanced data
        print("\n1. Loading enhanced HSN data...")
        df = load_enhanced_data()

        # Design rule-based schema
        print("\n2. Designing rule-based graph schema...")
        schema = design_rule_based_schema(df)

        # Enhance with LLM-assisted relationships
        print("\n3. Enhancing schema with LLM-assisted relationships...")
        enhanced_schema = enhance_schema_with_llm(schema, df)

        # Validate schema
        print("\n4. Validating graph schema...")
        validation_results = validate_graph_schema(enhanced_schema, df)

        # Export schema and documentation
        print("\n5. Exporting graph schema and documentation...")
        exported_files = export_graph_schema(enhanced_schema, validation_results, DATA_DIR)

        # Summary
        print("\n" + "=" * 60)
        print("PHASE 2.1 GRAPH DESIGN COMPLETE")
        print("=" * 60)
        print(f"SUCCESS: Created knowledge graph with {enhanced_schema.metadata['total_nodes']} nodes")
        print(f"SUCCESS: Established {len(enhanced_schema.relationships)} relationships")
        print(f"SUCCESS: Validation score: {validation_results['overall_score']:.2f}/1.0")
        print(f"Files exported to: {DATA_DIR}")
        print("Ready to proceed to Phase 2.2: Graph Implementation")
        print("=" * 60)

        # Display sample results
        print("\nSample Results:")
        print(f"Node types: {enhanced_schema.metadata['node_counts']}")
        print(f"Relationship types: {enhanced_schema.metadata['relationship_counts']}")

    except Exception as e:
        print(f"ERROR: Error in graph design: {str(e)}")
        raise

if __name__ == "__main__":
    main()