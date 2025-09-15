#!/usr/bin/env python3
"""
Test script for HSN data enhancement (Phase 1.2)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import re
import json
from pathlib import Path

# File paths
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"
DATA_DIR = OUTPUT_DIR / "data"
EXTRACTED_DATA_PATH = DATA_DIR / "extraction_sample.csv"

def load_extracted_data():
    """Load the extracted data from Phase 1.1."""
    if EXTRACTED_DATA_PATH.exists():
        hsn_data = pd.read_csv(EXTRACTED_DATA_PATH)
        print(f"SUCCESS: Loaded {len(hsn_data)} records from {EXTRACTED_DATA_PATH}")
        print(f"Columns: {list(hsn_data.columns)}")
        return hsn_data
    else:
        print(f"ERROR: Extracted data not found at {EXTRACTED_DATA_PATH}")
        return None

def build_hierarchy_mappings(df: pd.DataFrame) -> Tuple[Dict, Dict, Dict]:
    """Build mappings for chapters, headings, and subheadings."""
    chapter_descriptions = {}
    heading_descriptions = {}
    subheading_descriptions = {}

    # Helper function to clean HSN code (convert float to int string)
    def clean_hsn_code(code):
        if pd.isna(code):
            return None
        try:
            return str(int(float(code)))
        except (ValueError, TypeError):
            return str(code).replace('.0', '')

    # Build chapter descriptions (2-digit level)
    chapter_rows = df[df['code_level'] == '2_digit']
    for _, row in chapter_rows.iterrows():
        chapter = clean_hsn_code(row['chapter'])
        if chapter and chapter not in chapter_descriptions:
            chapter_descriptions[chapter] = {
                'description': row['description'],
                'export_policy': row['export_policy'],
                'chapter_title': f"Chapter {chapter}: {row['description']}",
                'level': 'chapter'
            }

    # Build heading descriptions (4-digit level)
    heading_rows = df[df['code_level'] == '4_digit']
    for _, row in heading_rows.iterrows():
        heading = clean_hsn_code(row['heading'])
        if heading and heading not in heading_descriptions:
            heading_descriptions[heading] = {
                'description': row['description'],
                'export_policy': row['export_policy'],
                'chapter': heading[:2],
                'heading_title': f"Heading {heading}: {row['description']}",
                'level': 'heading'
            }

    # Build subheading descriptions (6-digit level)
    subheading_rows = df[df['code_level'] == '6_digit']
    for _, row in subheading_rows.iterrows():
        subheading = clean_hsn_code(row['subheading'])
        if subheading and subheading not in subheading_descriptions:
            subheading_descriptions[subheading] = {
                'description': row['description'],
                'export_policy': row['export_policy'],
                'chapter': subheading[:2],
                'heading': subheading[:4],
                'subheading_title': f"Subheading {subheading}: {row['description']}",
                'level': 'subheading'
            }

    print(f"Built hierarchy mappings:")
    print(f"  Chapters: {len(chapter_descriptions)}")
    print(f"  Headings: {len(heading_descriptions)}")
    print(f"  Subheadings: {len(subheading_descriptions)}")

    return chapter_descriptions, heading_descriptions, subheading_descriptions

def enrich_with_hierarchy_context(df: pd.DataFrame,
                                  chapter_descriptions: Dict,
                                  heading_descriptions: Dict,
                                  subheading_descriptions: Dict) -> pd.DataFrame:
    """Enrich each row with complete hierarchical context."""
    enriched_df = df.copy()

    # Helper function to clean HSN code
    def clean_hsn_code(code):
        if pd.isna(code):
            return None
        try:
            return str(int(float(code)))
        except (ValueError, TypeError):
            return str(code).replace('.0', '')

    def get_hierarchy_context(row):
        """Get complete hierarchical context for a row."""
        context = {
            'chapter_description': '',
            'chapter_title': '',
            'heading_description': '',
            'heading_title': '',
            'subheading_description': '',
            'subheading_title': '',
            'full_hierarchy_path': '',
            'complete_context': '',
            'parent_codes': '',
            'child_codes': ''
        }

        # Clean codes for lookup
        chapter = clean_hsn_code(row.get('chapter', ''))
        heading = clean_hsn_code(row.get('heading', ''))
        subheading = clean_hsn_code(row.get('subheading', ''))
        full_hsn_code = clean_hsn_code(row.get('full_hsn_code', ''))

        # Get chapter context
        if chapter and chapter in chapter_descriptions:
            context['chapter_description'] = chapter_descriptions[chapter]['description']
            context['chapter_title'] = chapter_descriptions[chapter]['chapter_title']

        # Get heading context
        if heading and heading in heading_descriptions:
            context['heading_description'] = heading_descriptions[heading]['description']
            context['heading_title'] = heading_descriptions[heading]['heading_title']

        # Get subheading context
        if subheading and subheading in subheading_descriptions:
            context['subheading_description'] = subheading_descriptions[subheading]['description']
            context['subheading_title'] = subheading_descriptions[subheading]['subheading_title']

        # Build hierarchy path
        hierarchy_parts = []
        if context['chapter_title']:
            hierarchy_parts.append(context['chapter_title'])
        if context['heading_title']:
            hierarchy_parts.append(context['heading_title'])
        if context['subheading_title']:
            hierarchy_parts.append(context['subheading_title'])

        context['full_hierarchy_path'] = ' → '.join(hierarchy_parts)

        # Build complete context
        complete_parts = []
        if context['chapter_description']:
            complete_parts.append(f"Chapter: {context['chapter_description']}")
        if context['heading_description']:
            complete_parts.append(f"Heading: {context['heading_description']}")
        if context['subheading_description']:
            complete_parts.append(f"Subheading: {context['subheading_description']}")
        if row.get('description', ''):
            complete_parts.append(f"Product: {row['description']}")

        context['complete_context'] = ' | '.join(complete_parts)

        # Build parent-child relationships
        parents = []
        children = []

        if full_hsn_code:
            # This is an 8-digit code, find parents
            parents.extend([full_hsn_code[:6], full_hsn_code[:4], full_hsn_code[:2]])
        elif subheading:
            # This is a 6-digit code, find parents and children
            parents.extend([subheading[:4], subheading[:2]])
            # Find 8-digit children
            children_8 = []
            for _, child_row in df.iterrows():
                child_full = clean_hsn_code(child_row.get('full_hsn_code', ''))
                if child_full and child_full.startswith(subheading):
                    children_8.append(child_full)
            children.extend(children_8)
        elif heading:
            # This is a 4-digit code, find parents and children
            parents.append(heading[:2])
            # Find 6-digit and 8-digit children
            children_6 = []
            children_8 = []
            for _, child_row in df.iterrows():
                child_sub = clean_hsn_code(child_row.get('subheading', ''))
                child_full = clean_hsn_code(child_row.get('full_hsn_code', ''))
                if child_sub and child_sub.startswith(heading):
                    children_6.append(child_sub)
                if child_full and child_full.startswith(heading):
                    children_8.append(child_full)
            children.extend(children_6 + children_8)
        elif chapter:
            # This is a 2-digit code, find children
            children_4 = []
            children_6 = []
            children_8 = []
            for _, child_row in df.iterrows():
                child_head = clean_hsn_code(child_row.get('heading', ''))
                child_sub = clean_hsn_code(child_row.get('subheading', ''))
                child_full = clean_hsn_code(child_row.get('full_hsn_code', ''))
                if child_head and child_head.startswith(chapter):
                    children_4.append(child_head)
                if child_sub and child_sub.startswith(chapter):
                    children_6.append(child_sub)
                if child_full and child_full.startswith(chapter):
                    children_8.append(child_full)
            children.extend(children_4 + children_6 + children_8)

        context['parent_codes'] = ','.join(set(parents))
        context['child_codes'] = ','.join(set(children))

        return pd.Series(context)

    # Apply enrichment
    print("Enriching data with hierarchical context...")
    hierarchy_context = enriched_df.apply(get_hierarchy_context, axis=1)

    # Merge with original data
    enriched_df = pd.concat([enriched_df, hierarchy_context], axis=1)

    print(f"SUCCESS: Enriched {len(enriched_df)} records with hierarchical context")

    return enriched_df

def create_structured_documents(df: pd.DataFrame) -> pd.DataFrame:
    """Create structured documents optimized for vectorization and retrieval."""
    documents_df = df.copy()

    # Helper function to clean HSN code
    def clean_hsn_code(code):
        if pd.isna(code):
            return None
        try:
            return str(int(float(code)))
        except (ValueError, TypeError):
            return str(code).replace('.0', '')

    def create_document_content(row):
        """Create comprehensive document content for each HSN code."""
        doc_parts = []

        # Basic information - prioritize full_hsn_code, then subheading, heading, chapter
        hsn_code = None
        if not pd.isna(row.get('full_hsn_code')):
            hsn_code = clean_hsn_code(row['full_hsn_code'])
        elif not pd.isna(row.get('subheading')):
            hsn_code = clean_hsn_code(row['subheading'])
        elif not pd.isna(row.get('heading')):
            hsn_code = clean_hsn_code(row['heading'])
        elif not pd.isna(row.get('chapter')):
            hsn_code = clean_hsn_code(row['chapter'])

        if hsn_code:
            doc_parts.append(f"HSN Code: {hsn_code}")

        # Hierarchical context
        if row.get('chapter_title'):
            doc_parts.append(f"Chapter: {row['chapter_title']}")
        if row.get('heading_title'):
            doc_parts.append(f"Heading: {row['heading_title']}")
        if row.get('subheading_title'):
            doc_parts.append(f"Subheading: {row['subheading_title']}")

        # Product description
        if row.get('description'):
            doc_parts.append(f"Description: {row['description']}")

        # Export policy and conditions
        if row.get('export_policy') and not pd.isna(row['export_policy']):
            doc_parts.append(f"Export Policy: {row['export_policy']}")
        if row.get('policy_condition') and not pd.isna(row['policy_condition']):
            doc_parts.append(f"Policy Conditions: {row['policy_condition']}")

        # Complete context
        if row.get('complete_context'):
            doc_parts.append(f"Full Context: {row['complete_context']}")

        # Join all parts
        document_content = '\n'.join(doc_parts)

        # Create searchable keywords
        keywords = []
        if hsn_code:
            keywords.append(hsn_code)
        if row.get('description') and not pd.isna(row['description']):
            # Extract key terms from description
            desc_words = str(row['description']).lower().split()
            keywords.extend([word for word in desc_words if len(word) > 3])

        # Add hierarchical keywords
        if row.get('chapter_title'):
            keywords.extend(str(row['chapter_title']).lower().split()[:3])

        return pd.Series({
            'document_content': document_content,
            'search_keywords': ', '.join(keywords[:10]) if keywords else '',
            'document_length': len(document_content),
            'hierarchy_level': row.get('code_level', 'unknown'),
            'document_type': 'hsn_classification'
        })

    # Apply document creation
    print("Creating structured documents for vectorization...")
    document_data = documents_df.apply(create_document_content, axis=1)

    # Merge with original data
    documents_df = pd.concat([documents_df, document_data], axis=1)

    print(f"SUCCESS: Created {len(documents_df)} structured documents")
    print(f"Average document length: {documents_df['document_length'].mean():.0f} characters")

    return documents_df

def validate_enhanced_data(df: pd.DataFrame) -> Dict:
    """Validate the enhanced HSN data quality."""
    validation = {
        'total_records': len(df),
        'hierarchy_completeness': {},
        'document_quality': {},
        'enhancement_metrics': {},
        'data_quality_score': 0.0
    }

    # Check document quality
    if 'document_length' in df.columns:
        validation['document_quality']['avg_document_length'] = df['document_length'].mean()
        validation['document_quality']['min_document_length'] = df['document_length'].min()
        validation['document_quality']['max_document_length'] = df['document_length'].max()

    # Check enhancement metrics
    enhancement_score = 0
    if 'chapter_title' in df.columns:
        chapter_titles = df['chapter_title'].notna().sum()
        validation['enhancement_metrics']['chapters_with_titles'] = chapter_titles
        enhancement_score += chapter_titles / len(df)

    if 'heading_title' in df.columns:
        heading_titles = df['heading_title'].notna().sum()
        validation['enhancement_metrics']['headings_with_titles'] = heading_titles
        enhancement_score += heading_titles / len(df)

    if 'subheading_title' in df.columns:
        subheading_titles = df['subheading_title'].notna().sum()
        validation['enhancement_metrics']['subheadings_with_titles'] = subheading_titles
        enhancement_score += subheading_titles / len(df)

    if 'document_content' in df.columns:
        documents_created = df['document_content'].notna().sum()
        validation['enhancement_metrics']['documents_created'] = documents_created
        enhancement_score += documents_created / len(df)

    validation['data_quality_score'] = enhancement_score / 4  # Average of 4 metrics

    return validation

def main():
    """Main execution function."""
    print("Starting HSN Data Enhancement Test (Phase 1.2)")
    print("=" * 60)

    # Step 1: Load extracted data
    print("\n1. Loading extracted data...")
    hsn_data = load_extracted_data()
    if hsn_data is None:
        return

    # Step 2: Build hierarchy mappings
    print("\n2. Building hierarchical mappings...")
    chapter_descriptions, heading_descriptions, subheading_descriptions = build_hierarchy_mappings(hsn_data)

    # Step 3: Enrich with hierarchical context
    print("\n3. Enriching data with hierarchical context...")
    enriched_hsn_data = enrich_with_hierarchy_context(
        hsn_data,
        chapter_descriptions,
        heading_descriptions,
        subheading_descriptions
    )

    # Step 4: Create structured documents
    print("\n4. Creating structured documents...")
    structured_documents = create_structured_documents(enriched_hsn_data)

    # Step 5: Validate enhanced data
    print("\n5. Validating enhanced data...")
    enhancement_validation = validate_enhanced_data(structured_documents)

    # Display results
    print("\n" + "=" * 60)
    print("PHASE 1.2 ENHANCEMENT RESULTS")
    print("=" * 60)
    print(f"Total enhanced records: {len(structured_documents)}")
    print(f"Data Quality Score: {enhancement_validation['data_quality_score']:.2f}/1.0")

    for category, metrics in enhancement_validation.items():
        if category not in ['total_records', 'data_quality_score']:
            print(f"\n{category.upper()}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.2f}")
                else:
                    print(f"  {metric}: {value}")

    # Show sample enhanced data
    print("\nSample Enhanced Data:")
    sample_cols = ['chapter', 'heading', 'subheading', 'full_hsn_code', 'description', 'chapter_title']
    available_cols = [col for col in sample_cols if col in structured_documents.columns]
    print(structured_documents[available_cols].head(5))

    # Show sample documents
    print("\nSample Structured Documents:")
    doc_sample = structured_documents[['full_hsn_code', 'document_content']].dropna(subset=['full_hsn_code']).head(2)
    for idx, row in doc_sample.iterrows():
        print(f"\n--- Document {idx + 1} ---")
        print(f"HSN Code: {row['full_hsn_code']}")
        print(f"Content preview: {row['document_content'][:150]}...")

    # Export sample enhanced data
    sample_enhanced_path = DATA_DIR / "sample_enhanced_data.csv"
    structured_documents.head(20).to_csv(sample_enhanced_path, index=False)
    print(f"\nSample enhanced data exported to: {sample_enhanced_path}")

    print("\n" + "=" * 60)
    print("PHASE 1.2 DATA ENHANCEMENT COMPLETE")
    print("=" * 60)
    print("SUCCESS: Successfully enhanced HSN data with hierarchical context")
    print("SUCCESS: Created structured documents for RAG system")
    print("SUCCESS: Built complete hierarchy mappings")
    print("READY: Ready to proceed to Phase 2: Knowledge Graph Construction")
    print("=" * 60)

if __name__ == "__main__":
    main()