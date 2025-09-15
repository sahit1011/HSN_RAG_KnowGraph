# HSN Knowledge Graph Schema Documentation

## Overview

This document describes the knowledge graph schema for HSN (Harmonized System Nomenclature) codes, designed using a hybrid rule-based and LLM-assisted approach.

**Schema Statistics:**
- Total Nodes: 2079
- Total Relationships: 1352266
- Node Types: 4
- Relationship Types: 5

## Node Types

### CHAPTER
**Description:** Top-level HSN chapter (2-digit codes)
**Properties:** id, code, description, title, export_policy, child_codes, hierarchy_path
**Count:** 15

### HEADING
**Description:** HSN heading level (4-digit codes)
**Properties:** id, code, description, title, export_policy, parent_codes, child_codes, hierarchy_path
**Count:** 135

### SUBHEADING
**Description:** HSN subheading level (6-digit codes)
**Properties:** id, code, description, title, export_policy, parent_codes, child_codes, hierarchy_path
**Count:** 297

### HSN_CODE
**Description:** Complete HSN codes (8-digit codes)
**Properties:** id, code, description, title, export_policy, policy_condition, document_content, search_keywords, parent_codes, hierarchy_path, completeness_score
**Count:** 1632

## Relationship Types

### BELONGS_TO_CHAPTER
**Description:** Entity belongs to a specific chapter
**Direction:** directed
**Properties:** chapter_code
**Count:** 1632

### IS_PARENT_OF
**Description:** Hierarchical parent-child relationship
**Direction:** directed
**Properties:** hierarchy_level, code_prefix
**Count:** 1632

### IS_CHILD_OF
**Description:** Reverse hierarchical relationship
**Direction:** directed
**Properties:** hierarchy_level, code_prefix
**Count:** 1632

### HAS_SIMILAR_DESCRIPTION
**Description:** Semantic similarity between descriptions
**Direction:** undirected
**Properties:** similarity_score, shared_keywords
**Count:** 0

### SHARES_EXPORT_POLICY
**Description:** Entities with same export policy
**Direction:** undirected
**Properties:** policy_type, policy_details
**Count:** 0

## Validation Results

**Overall Score:** 1.00/1.0

### Coverage Metrics
- total_records: 2079
- nodes_created: 2079
- coverage_rate: 1.0

### Relationship Metrics
- total_relationships: 1352266
- relationship_types:
  - IS_PARENT_OF: 1632
  - IS_CHILD_OF: 1632
  - BELONGS_TO_CHAPTER: 1632
  - SHARES_EXPORT_POLICY: 1347370
- avg_relationships_per_node: 650.4405964405964

### Data Integrity
- missing_source_nodes: 0
- missing_target_nodes: 0
- integrity_score: 1.0
