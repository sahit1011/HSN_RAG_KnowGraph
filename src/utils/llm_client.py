#!/usr/bin/env python3
"""
LangChain-based LLM Integration for HSN RAG System
Uses LangChain with OpenRouter for knowledge graph generation and response generation
"""

import os
from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import BaseMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
import logging

from config import OPENROUTER_API_KEY, LLM_CONFIG

logger = logging.getLogger(__name__)

class ProductRelationshipSchema(BaseModel):
    """Schema for LLM-generated product relationships"""
    categories: List[str] = Field(description="Related product categories")
    similar_products: List[str] = Field(description="Similar product examples")
    characteristics: List[str] = Field(description="Key product characteristics")
    export_policy: str = Field(description="Export policy description")
    hierarchy_path: List[str] = Field(description="Hierarchical classification path")
    confidence_score: float = Field(description="Confidence score between 0 and 1")

class GraphSchemaSuggestionSchema(BaseModel):
    """Schema for graph schema improvement suggestions"""
    missing_relationships: List[str] = Field(description="Missing relationship types")
    additional_properties: List[str] = Field(description="Additional node properties")
    new_node_types: List[str] = Field(description="New node types to add")
    optimization_suggestions: List[str] = Field(description="Schema optimization suggestions")

class LangChainLLMClient:
    """
    LangChain-based client for OpenRouter API
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or OPENROUTER_API_KEY

        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable.")

        # Initialize LangChain OpenAI client with OpenRouter
        self.llm = ChatOpenAI(
            model=LLM_CONFIG["model"],
            openai_api_key=self.api_key,
            openai_api_base=LLM_CONFIG["api_base"],
            temperature=LLM_CONFIG["temperature"],
            max_tokens=LLM_CONFIG["max_tokens"],
            model_kwargs={
                "extra_headers": {
                    "HTTP-Referer": "https://github.com/krishraghavan/hsn-rag",
                    "X-Title": "HSN RAG System"
                }
            }
        )

        logger.info(f"SUCCESS: LangChain LLM client initialized with model: {LLM_CONFIG['model']}")

    def generate_product_relationships(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate relationships and properties for a product using LLM

        Args:
            product_data: Dictionary containing product information

        Returns:
            Dictionary with generated relationships and properties
        """
        prompt_template = """
        Analyze the following product information and generate knowledge graph relationships for HSN classification:

        Product Description: {description}
        HSN Code: {hsn_code}
        Category: {category}

        Based on this information, please provide:
        1. Related product categories (3-5 categories most relevant to this product)
        2. Similar products (3-5 specific product examples)
        3. Key characteristics/features of this product
        4. Export policy implications
        5. Hierarchical classification path in HSN system

        Be specific and accurate in your analysis. Consider the product's material composition, manufacturing process, and end-use applications.

        {format_instructions}
        """

        parser = JsonOutputParser(pydantic_object=ProductRelationshipSchema)

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["description", "hsn_code", "category"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)

        try:
            result = chain.run(
                description=product_data.get('description', ''),
                hsn_code=product_data.get('hsn_code', ''),
                category=product_data.get('category', '')
            )

            parsed_result = parser.parse(result)
            return parsed_result.dict()

        except Exception as e:
            logger.error(f"Error generating product relationships: {str(e)}")
            return {
                "categories": ["rubber_products"],
                "similar_products": ["natural_rubber", "synthetic_rubber"],
                "characteristics": ["elastic", "durable"],
                "export_policy": "free",
                "hierarchy_path": ["chapter_40", "heading_4001"],
                "confidence_score": 0.5,
                "error": str(e)
            }

    def generate_response(self,
                         query: str,
                         retrieved_docs: List[Dict[str, Any]],
                         graph_context: List[Dict[str, Any]],
                         hsn_result: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a comprehensive response using retrieved information

        Args:
            query: Original user query
            retrieved_docs: Documents retrieved from vector search
            graph_context: Context from knowledge graph
            hsn_result: HSN classification result

        Returns:
            Generated response string
        """

        # Prepare context with detailed logging
        context_parts = []

        # Log vector search results separately
        logger.info("=" * 60)
        logger.info("VECTOR SEARCH RETRIEVAL RESULTS:")
        logger.info("=" * 60)
        if retrieved_docs:
            for i, doc in enumerate(retrieved_docs):
                logger.info(f"Vector Doc {i+1}: HSN={doc.get('hsn_code', 'N/A')} | Desc={doc.get('description', '')[:100]}...")
        else:
            logger.info("No vector search results retrieved")
        logger.info("=" * 60)

        # Log graph context results separately
        logger.info("GRAPH CONTEXT RETRIEVAL RESULTS:")
        logger.info("=" * 60)
        if graph_context:
            for i, ctx in enumerate(graph_context):
                logger.info(f"Graph Context {i+1}: Type={ctx.get('node_type', 'N/A')} | Code={ctx.get('code', 'N/A')} | Desc={ctx.get('description', '')[:100]}...")
        else:
            logger.info("No graph context results retrieved")
        logger.info("=" * 60)

        if hsn_result:
            context_parts.append(f"HSN Code: {hsn_result.get('hsn_code', 'N/A')}")
            context_parts.append(f"Description: {hsn_result.get('description', 'N/A')}")

        if retrieved_docs:
            context_parts.append("Retrieved Information:")
            for i, doc in enumerate(retrieved_docs[:3]):
                context_parts.append(f"{i+1}. {doc.get('description', '')[:200]}...")

        if graph_context:
            context_parts.append("Related Products:")
            for ctx in graph_context[:3]:
                context_parts.append(f"- {ctx.get('description', '')[:100]}...")

        context = "\n".join(context_parts)

        # Log the final combined context
        logger.info("FINAL LLM CONTEXT (Combined):")
        logger.info("=" * 60)
        logger.info(context)
        logger.info("=" * 60)

        prompt_template = """
        You are an expert HSN (Harmonized System of Nomenclature) classification assistant. Based on the following context, provide a comprehensive and helpful response to the user's query.

        User Query: {query}

        Context Information:
        {context}

        Please provide a response that:
        1. Directly answers the user's query
        2. Includes the relevant HSN code if found
        3. Explains the classification reasoning
        4. Provides additional relevant information
        5. Suggests similar products or alternatives if appropriate
        6. Is conversational and easy to understand

        If you're uncertain about any information, clearly state that. Keep the response focused and helpful.
        """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["query", "context"]
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)

        try:
            # Log the exact prompt being sent to LLM for debugging
            full_prompt = prompt.format(query=query, context=context)
            logger.info("=" * 80)
            logger.info("LLM REQUEST - FULL PROMPT BEING SENT:")
            logger.info("=" * 80)
            logger.info(full_prompt)
            logger.info("=" * 80)
            logger.info(f"Context details: retrieved_docs={len(retrieved_docs)}, graph_context={len(graph_context)}, hsn_result_present={hsn_result is not None}")
            logger.info("=" * 80)

            response = chain.run(query=query, context=context)
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return self._generate_fallback_response(query, hsn_result)

    def _generate_fallback_response(self, query: str, hsn_result: Optional[Dict[str, Any]]) -> str:
        """Generate a basic fallback response"""
        if hsn_result and hsn_result.get('hsn_code'):
            return f"Based on your query '{query}', I found HSN code {hsn_result['hsn_code']}: {hsn_result.get('description', 'No description available')}."
        else:
            return f"I'm sorry, I couldn't find specific HSN classification information for your query '{query}'. Please try rephrasing your question or providing more details about the product."

    def generate_graph_schema_suggestions(self, existing_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to suggest improvements to the knowledge graph schema

        Args:
            existing_schema: Current graph schema

        Returns:
            Suggested improvements and additions
        """
        prompt_template = """
        Analyze this knowledge graph schema for HSN classification and suggest improvements:

        Current Schema: {schema}

        As an expert in knowledge graph design for product classification systems, suggest:
        1. Missing relationship types that would be useful
        2. Additional node properties for better classification
        3. New node types that would enhance the graph
        4. Schema optimization opportunities

        Focus on relationships and properties that would improve HSN code classification accuracy and product similarity detection.

        {format_instructions}
        """

        parser = JsonOutputParser(pydantic_object=GraphSchemaSuggestionSchema)

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["schema"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)

        try:
            result = chain.run(schema=str(existing_schema))
            parsed_result = parser.parse(result)
            return parsed_result.dict()

        except Exception as e:
            logger.error(f"Error generating schema suggestions: {str(e)}")
            return {
                "missing_relationships": ["HAS_MATERIAL_COMPOSITION"],
                "additional_properties": ["manufacturing_process", "end_use"],
                "new_node_types": ["material_type"],
                "optimization_suggestions": ["Add material composition relationships"],
                "error": str(e)
            }

def main():
    """Test the LangChain LLM client"""
    try:
        client = LangChainLLMClient()
        print("SUCCESS: LangChain LLM client initialized successfully")

        # Test basic response generation
        test_query = "What is HSN code for natural rubber latex?"
        test_result = {"hsn_code": "40011010", "description": "Natural rubber latex"}

        response = client.generate_response(test_query, [], [], test_result)
        print(f"SUCCESS: LLM Response: {response[:200]}...")

        # Test product relationship generation
        product_data = {
            "description": "Natural rubber latex",
            "hsn_code": "40011010",
            "category": "Rubber products"
        }

        relationships = client.generate_product_relationships(product_data)
        print(f"SUCCESS: Generated relationships: {relationships}")

    except Exception as e:
        print(f"ERROR: Failed to initialize LangChain LLM client: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()