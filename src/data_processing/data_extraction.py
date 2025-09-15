#!/usr/bin/env python3
"""
Test script for HSN data extraction from PDF
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import re
from pathlib import Path
import pdfplumber
from pdfplumber.page import Page
import PyPDF2
from tqdm import tqdm

# File paths
PDF_PATH = Path(__file__).parent.parent.parent / "Trade_Notice_First_50_Pages.pdf"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"
DATA_DIR = OUTPUT_DIR / "data"

# Create directories
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

def analyze_pdf_structure(pdf_path: str) -> Dict:
    """Analyze the PDF structure to understand layout and content organization."""
    with pdfplumber.open(pdf_path) as pdf:
        print(f"PDF Analysis:")
        print(f"Total pages: {len(pdf.pages)}")

        # Analyze first few pages
        metadata = pdf.metadata
        print(f"PDF Metadata: {metadata}")

        # Check page structure
        first_page = pdf.pages[0]
        print(f"First page dimensions: {first_page.width} x {first_page.height}")

        # Extract text from first page
        first_page_text = first_page.extract_text()
        print(f"\nFirst page preview (first 500 chars):\n{first_page_text[:500]}...")

        return {
            'total_pages': len(pdf.pages),
            'metadata': metadata,
            'first_page_text': first_page_text
        }

def extract_tables_from_pdf(pdf_path: str, start_page: int = 0, end_page: Optional[int] = None) -> List[pd.DataFrame]:
    """Extract all tables from the PDF with proper handling of merged cells and formatting."""
    all_tables = []

    with pdfplumber.open(pdf_path) as pdf:
        pages_to_process = pdf.pages[start_page:end_page]

        for i, page in enumerate(tqdm(pages_to_process, desc="Processing pages", unit="page")):
            page_num = start_page + i

            # Extract tables from current page
            tables = page.extract_tables()

            if tables:
                for j, table in enumerate(tables):
                    if table and len(table) > 1:  # Skip empty tables or headers only
                        df = pd.DataFrame(table[1:], columns=table[0] if table[0] else None)
                        df['page_number'] = page_num + 1
                        df['table_index'] = j
                        all_tables.append(df)
            else:
                tqdm.write(f"No tables found on page {page_num + 1}")

    return all_tables

def clean_table_data(tables: List[pd.DataFrame]) -> pd.DataFrame:
    """Clean and standardize the extracted table data."""
    if not tables:
        return pd.DataFrame()

    # Process each table individually to handle different structures
    processed_tables = []

    for i, df in enumerate(tables):
        print(f"Processing table {i}, shape: {df.shape}")

        # Skip if table is too small
        if df.shape[0] < 2:
            continue

        # Clean column names
        df.columns = [str(col).strip() if col is not None else f'col_{j}'
                     for j, col in enumerate(df.columns)]

        # Check if this is a header table (contains chapter info)
        first_cell = str(df.iloc[0, 0]).strip() if df.shape[0] > 0 else ""
        if "SCHEDULE-II" in first_cell or "CHAPTERS" in first_cell:
            print(f"Table {i}: Skipping header table")
            continue

        # Try to identify the actual data columns
        # Look for patterns in the first few rows
        sample_rows = df.head(3)

        # Find columns that look like HSN codes (8 digits) or chapter numbers (2 digits)
        hsn_col = None
        chapter_col = None
        description_col = None
        policy_col = None

        for col_idx, col_name in enumerate(df.columns):
            if col_name in ['page_number', 'table_index', 'source_table']:
                continue

            # Check first few values in this column
            sample_values = df.iloc[:5, col_idx].astype(str).str.strip()

            # Look for HSN codes (8+ digits)
            if any(re.match(r'^\d{8,}', val) for val in sample_values if val):
                hsn_col = col_idx
                print(f"Table {i}: Found HSN column at index {col_idx}")

            # Look for chapter numbers (2 digits)
            elif any(re.match(r'^\d{2}$', val) for val in sample_values if val):
                chapter_col = col_idx
                print(f"Table {i}: Found chapter column at index {col_idx}")

            # Look for policy keywords
            elif any('Free' in val or 'Prohibited' in val or 'Restricted' in val for val in sample_values if val):
                policy_col = col_idx
                print(f"Table {i}: Found policy column at index {col_idx}")

        # If we can't identify columns automatically, use positional mapping
        if hsn_col is None:
            # Assume standard layout: [Chapter, HSN, Description, Policy, ...]
            if df.shape[1] >= 4:
                chapter_col = 0
                hsn_col = 1
                description_col = 2
                policy_col = 3

        # Create standardized dataframe
        std_df = pd.DataFrame()

        # Map columns to standard names
        if chapter_col is not None and chapter_col < df.shape[1]:
            std_df['chapter_number'] = df.iloc[:, chapter_col].astype(str).str.strip()

        if hsn_col is not None and hsn_col < df.shape[1]:
            std_df['hsn_code'] = df.iloc[:, hsn_col].astype(str).str.strip()

        if description_col is not None and description_col < df.shape[1]:
            std_df['description'] = df.iloc[:, description_col].astype(str).str.strip()
        elif df.shape[1] > 2:
            std_df['description'] = df.iloc[:, 2].astype(str).str.strip()

        if policy_col is not None and policy_col < df.shape[1]:
            std_df['export_policy'] = df.iloc[:, policy_col].astype(str).str.strip()
        elif df.shape[1] > 3:
            std_df['export_policy'] = df.iloc[:, 3].astype(str).str.strip()

        # Add remaining columns if available
        remaining_cols = ['policy_condition', 'notification_no', 'notification_date']
        col_idx = 4
        for col_name in remaining_cols:
            if col_idx < df.shape[1]:
                std_df[col_name] = df.iloc[:, col_idx].astype(str).str.strip()
            else:
                std_df[col_name] = ''
            col_idx += 1

        # Add metadata
        std_df['source_table'] = i
        std_df['page_number'] = df['page_number'].iloc[0] if 'page_number' in df.columns else 0

        # Remove rows that are clearly headers or empty
        std_df = std_df[~std_df['chapter_number'].str.contains('CHAPTER|SCHEDULE', case=False, na=False)]
        std_df = std_df[~std_df['hsn_code'].str.contains('ITC|HS|Code', case=False, na=False)]

        print(f"Table {i}: Standardized to {std_df.shape}")
        processed_tables.append(std_df)

    # Combine all processed tables
    if processed_tables:
        combined_df = pd.concat(processed_tables, ignore_index=True)
    else:
        return pd.DataFrame()

    print(f"Combined DataFrame shape: {combined_df.shape}")

    # Remove completely empty rows
    combined_df = combined_df.dropna(how='all')

    # Fill missing values with empty strings
    combined_df = combined_df.fillna('')

    # Clean string columns
    for col in combined_df.columns:
        if combined_df[col].dtype == 'object':
            combined_df[col] = combined_df[col].astype(str).str.strip()

    print(f"After cleaning - DataFrame shape: {combined_df.shape}")
    print(f"Columns: {list(combined_df.columns)}")

    return combined_df

def parse_hsn_data(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """Parse and structure the HSN data according to the expected format."""

    # Map columns to standard names
    column_mapping = {
        'S. No.': 'serial_no',
        'Chapter Number': 'chapter_number',
        'ITC(HS) Code': 'hsn_code',
        'Description': 'description',
        'Export Policy': 'export_policy',
        'Policy Condition': 'policy_condition',
        'Notification No': 'notification_no',
        'Notification Date': 'notification_date'
    }

    # Rename columns if they exist
    df_renamed = cleaned_df.copy()
    for old_col, new_col in column_mapping.items():
        if old_col in df_renamed.columns:
            df_renamed = df_renamed.rename(columns={old_col: new_col})

    # Extract HSN code components
    def extract_hsn_components(hsn_code: str) -> Dict:
        """Extract hierarchical components from HSN code."""
        hsn_code = str(hsn_code).strip()

        # Handle different HSN code formats
        if len(hsn_code) >= 8:
            return {
                'chapter': hsn_code[:2],
                'heading': hsn_code[:4],
                'subheading': hsn_code[:6],
                'full_code': hsn_code[:8],
                'code_type': '8_digit'
            }
        elif len(hsn_code) == 6:
            return {
                'chapter': hsn_code[:2],
                'heading': hsn_code[:4],
                'subheading': hsn_code,
                'full_code': '',
                'code_type': '6_digit'
            }
        elif len(hsn_code) == 4:
            return {
                'chapter': hsn_code[:2],
                'heading': hsn_code,
                'subheading': '',
                'full_code': '',
                'code_type': '4_digit'
            }
        elif len(hsn_code) == 2:
            return {
                'chapter': hsn_code,
                'heading': '',
                'subheading': '',
                'full_code': '',
                'code_type': '2_digit'
            }
        else:
            return {
                'chapter': '',
                'heading': '',
                'subheading': '',
                'full_code': '',
                'code_type': 'unknown'
            }

    # Apply HSN component extraction
    if 'hsn_code' in df_renamed.columns:
        hsn_components = df_renamed['hsn_code'].apply(extract_hsn_components)
        df_renamed['chapter'] = hsn_components.apply(lambda x: x['chapter'])
        df_renamed['heading'] = hsn_components.apply(lambda x: x['heading'])
        df_renamed['subheading'] = hsn_components.apply(lambda x: x['subheading'])
        df_renamed['full_hsn_code'] = hsn_components.apply(lambda x: x['full_code'])
        df_renamed['code_level'] = hsn_components.apply(lambda x: x['code_type'])

    # Filter for relevant chapters (40-98)
    if 'chapter' in df_renamed.columns:
        df_renamed['chapter_num'] = pd.to_numeric(df_renamed['chapter'], errors='coerce')
        df_filtered = df_renamed[
            (df_renamed['chapter_num'] >= 40) &
            (df_renamed['chapter_num'] <= 98)
        ].copy()
    else:
        df_filtered = df_renamed.copy()

    print(f"Parsed HSN data shape: {df_filtered.shape}")
    print(f"Chapters found: {sorted(df_filtered['chapter_num'].unique())}")

    return df_filtered

def validate_hsn_data(df: pd.DataFrame) -> Dict:
    """Perform comprehensive validation on the extracted HSN data."""
    validation_results = {
        'total_records': len(df),
        'missing_values': {},
        'data_quality': {},
        'chapter_distribution': {},
        'issues': []
    }

    # Check for missing values
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            validation_results['missing_values'][col] = missing_count

    # Validate HSN codes
    if 'full_hsn_code' in df.columns:
        valid_codes = df['full_hsn_code'].str.match(r'^\d{8}$', na=False).sum()
        validation_results['data_quality']['valid_8digit_codes'] = valid_codes
        validation_results['data_quality']['invalid_codes'] = len(df) - valid_codes

    # Chapter distribution
    if 'chapter_num' in df.columns:
        chapter_counts = df['chapter_num'].value_counts().sort_index()
        validation_results['chapter_distribution'] = chapter_counts.to_dict()

    # Check for data consistency
    if 'chapter' in df.columns and 'heading' in df.columns:
        inconsistent = df[df['heading'].str[:2] != df['chapter']].shape[0]
        if inconsistent > 0:
            validation_results['issues'].append(f"{inconsistent} records have inconsistent chapter-heading relationships")

    # Check for duplicate HSN codes
    if 'full_hsn_code' in df.columns:
        duplicates = df[df['full_hsn_code'].duplicated() & df['full_hsn_code'].notna()].shape[0]
        if duplicates > 0:
            validation_results['issues'].append(f"{duplicates} duplicate HSN codes found")

    return validation_results

def main():
    """Main execution function."""
    print("Starting HSN Data Extraction Test")
    print("=" * 50)

    # Step 1: Analyze PDF structure
    print("\n1. Analyzing PDF structure...")
    pdf_analysis = analyze_pdf_structure(PDF_PATH)

    # Step 2: Extract tables
    print("\n2. Extracting tables from PDF...")
    raw_tables = extract_tables_from_pdf(PDF_PATH, start_page=0, end_page=None)  # Process ALL pages

    if not raw_tables:
        print("No tables found in the PDF. Checking raw text extraction...")
        # Fallback: try raw text extraction
        with pdfplumber.open(PDF_PATH) as pdf:
            for page in pdf.pages[:10]:  # Check first 10 pages
                text = page.extract_text()
                print(f"Page {page.page_number} text length: {len(text)}")
                print(f"First 200 chars: {text[:200]}")
        return

    # Step 3: Clean data
    print("\n3. Cleaning extracted data...")
    cleaned_data = clean_table_data(raw_tables)

    # Step 4: Parse HSN data
    print("\n4. Parsing HSN data structure...")
    structured_hsn_data = parse_hsn_data(cleaned_data)

    # Step 5: Validate data
    print("\n5. Validating extracted data...")
    validation_report = validate_hsn_data(structured_hsn_data)

    # Display results
    print("\n" + "=" * 60)
    print("EXTRACTION RESULTS")
    print("=" * 60)
    print(f"Total tables extracted: {len(raw_tables)}")
    print(f"Total records: {validation_report['total_records']}")
    print(f"Valid 8-digit codes: {validation_report['data_quality'].get('valid_8digit_codes', 0)}")
    print(f"Chapters found: {list(validation_report['chapter_distribution'].keys())}")

    if validation_report['issues']:
        print("\nIssues found:")
        for issue in validation_report['issues']:
            print(f"- {issue}")
    else:
        print("\nNo major issues detected!")

    # Show sample data
    print("\nSample extracted data:")
    print(structured_hsn_data.head(10))

    # Export complete data
    complete_path = DATA_DIR / "extraction_complete.csv"
    structured_hsn_data.to_csv(complete_path, index=False)
    print(f"\nComplete data exported to: {complete_path}")

    # Export sample for reference
    sample_path = DATA_DIR / "extraction_sample.csv"
    structured_hsn_data.head(50).to_csv(sample_path, index=False)
    print(f"Sample data exported to: {sample_path}")

if __name__ == "__main__":
    main()