#!/usr/bin/env python3
"""
HSN RAG System - Main Entry Point
Modular HSN Code Classification System using RAG and Knowledge Graphs
"""

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.rag_system.hsn_rag_system import HSN_RAG_System, HSNClassificationResult
from typing import List, Optional
import argparse

def main():
    """Main entry point for the HSN RAG System"""
    print("HSN RAG System - Modular Implementation")
    print("=" * 50)

    parser = argparse.ArgumentParser(description="HSN Code Classification System")
    parser.add_argument("--query", "-q", type=str, help="Single query to process")
    parser.add_argument("--batch", "-b", type=str, help="File with queries to process in batch")
    parser.add_argument("--test", action="store_true", help="Run system tests")
    parser.add_argument("--metrics", action="store_true", help="Show system metrics")

    args = parser.parse_args()

    # Initialize system
    system = HSN_RAG_System()

    try:
        print("\nInitializing HSN RAG System...")
        if not system.initialize_system():
            print("ERROR: Failed to initialize system")
            return 1

        if args.test:
            # Run system tests
            print("\nRunning system tests...")
            test_results = system.run_system_tests()
            print(f"Test Status: {test_results['overall_status'].upper()}")
            print(f"Passed: {test_results['passed_tests']}/{test_results['total_tests']}")

        elif args.query:
            # Process single query - use rule_based mode to show retrieved context
            print(f"\nProcessing query: '{args.query}'")
            result = system.classify_product(args.query, rag_mode="rule_based")

            print("\nResult:")
            print("-" * 30)
            print(f"HSN Code: {result.hsn_code or 'Not found'}")
            print(f"Description: {result.description or 'N/A'}")
            print(".1%")
            print(".2f")
            print(f"Query Type: {result.query_type}")

            # Display retrieved context if available (for rule-based mode)
            if result.metadata and 'retrieved_context' in result.metadata:
                ctx = result.metadata['retrieved_context']
                if ctx.get('vector_results'):
                    print(f"\nRetrieved Context - Vector Search ({len(ctx['vector_results'])} results):")
                    for i, item in enumerate(ctx['vector_results'][:3], 1):
                        similarity_pct = int(item.get('similarity_score', 0) * 100)
                        print(f"  {i}. HSN {item.get('hsn_code', 'N/A')}: {item.get('description', '')[:60]}... ({similarity_pct}%)")

                if ctx.get('graph_results'):
                    print(f"Retrieved Context - Graph Search ({len(ctx['graph_results'])} results):")
                    for i, item in enumerate(ctx['graph_results'][:3], 1):
                        print(f"  {i}. {item.get('node_type', 'N/A').upper()}: {item.get('code', 'N/A')} - {item.get('description', '')[:60]}...")

            # Display disambiguation options if available
            if result.metadata and 'disambiguation_options' in result.metadata:
                options = result.metadata['disambiguation_options']
                if options:
                    print(f"\nMultiple Options Found ({len(options)}):")
                    for i, option in enumerate(options, 1):
                        print(f"  {i}. HSN {option['hsn_code']}: {option['description'][:60]}...")
                        print(".1%")

            if result.suggestions:
                print("\nSuggestions:")
                for suggestion in result.suggestions:
                    print(f"  • {suggestion}")

        elif args.batch:
            # Process batch queries
            if not os.path.exists(args.batch):
                print(f"ERROR: Batch file '{args.batch}' not found")
                return 1

            print(f"\nProcessing batch queries from: {args.batch}")
            with open(args.batch, 'r') as f:
                queries = [line.strip() for line in f if line.strip()]

            results = system.batch_classify(queries, rag_mode="llm_enhanced")

            print(f"\nProcessed {len(results)} queries:")
            print("-" * 50)
            for i, (query, result) in enumerate(zip(queries, results), 1):
                status = "SUCCESS" if result.hsn_code else "FAILED"
                print("2d")

        elif args.metrics:
            # Show system metrics
            metrics = system.get_system_metrics()
            print("\nSystem Metrics:")
            print("-" * 30)
            print(f"Status: {metrics['status']}")
            print(f"Total Queries: {metrics['total_queries']}")
            print(".1%")
            print(".2f")
            print(".1%")
            print(f"Errors: {metrics['error_count']}")

        else:
            # Interactive mode
            print("\nEntering interactive mode. Type 'quit' to exit.")
            print("Example queries:")
            print("  • What is the HSN code for natural rubber latex?")
            print("  • HSN code for prevulcanised rubber")
            print("  • Tell me about HSN 40011010")
            print()

            while True:
                try:
                    query = input("Enter your query: ").strip()
                    if query.lower() in ['quit', 'exit', 'q']:
                        break

                    if query:
                        result = system.classify_product(query, rag_mode="llm_enhanced")

                        print("\nResult:")
                        print("-" * 30)
                        print(f"HSN Code: {result.hsn_code or 'Not found'}")
                        print(f"Description: {result.description or 'N/A'}")
                        print(".1%")
                        print(".2f")
                        print(f"Query Type: {result.query_type}")

                        if result.suggestions:
                            print("\nSuggestions:")
                            for suggestion in result.suggestions[:2]:
                                print(f"  • {suggestion}")
                        print()

                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
                except Exception as e:
                    print(f"Error processing query: {e}")

        # Show final metrics if requested or if we processed queries
        if args.metrics or args.query or args.batch:
            print("\nFinal System Status:")
            metrics = system.get_system_metrics()
            print(f"  Status: {metrics['status']}")
            print(f"  Total Queries: {metrics['total_queries']}")
            print(".1%")

    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    print("\nDone!")
    return 0

if __name__ == "__main__":
    sys.exit(main())