#!/usr/bin/env python3
"""
HSN RAG System - Streamlit Web Application
Interactive web interface for testing HSN code classification queries
"""

import streamlit as st
import sys
from pathlib import Path
import time
from datetime import datetime

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.rag_system.hsn_rag_system import HSN_RAG_System
from config import RAG_MODES

# Page configuration
st.set_page_config(
    page_title="HSN Code Classification System",
    page_icon="search",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'system' not in st.session_state:
    st.session_state.system = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

def initialize_system():
    """Initialize the HSN RAG system"""
    if not st.session_state.initialized:
        with st.spinner("Initializing HSN RAG System..."):
            try:
                st.session_state.system = HSN_RAG_System()
                if st.session_state.system.initialize_system():
                    st.session_state.initialized = True
                    st.success("SUCCESS: System initialized successfully!")
                    return True
                else:
                    st.error("ERROR: Failed to initialize system")
                    return False
            except Exception as e:
                st.error(f"ERROR: Error initializing system: {e}")
                return False
    return True

def display_result(result):
    """Display classification result in a nice format"""
    # Check if this is LLM-enhanced response without specific HSN code
    if result.metadata.get('llm_enhanced') and not result.hsn_code and result.metadata.get('enhanced_description'):
        st.info("**AI-Generated Classification Response**")
        st.write(result.metadata['enhanced_description'])

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confidence", ".1%")
            st.metric("Processing Time", ".2f")

        with col2:
            st.metric("Query Type", result.query_type.title())
            st.metric("Processing Mode", "LLM-Enhanced RAG")

        # Show retrieved context
        has_retrieved_context = result.metadata.get('retrieved_context') is not None
        if has_retrieved_context:
            ctx = result.metadata['retrieved_context']
            with st.expander("🔍 Retrieved Results (Vector + Graph)", expanded=True):
                # Vector search results
                if ctx.get('vector_results'):
                    st.markdown("**📄 Top Vector Search Results:**")
                    for i, item in enumerate(ctx['vector_results'][:5], 1):
                        similarity_pct = int(item.get('similarity_score', 0) * 100)
                        st.markdown(f"""
                        **{i}. HSN {item.get('hsn_code', 'N/A')}**
                        • **Description**: {item.get('description', '')[:100]}...
                        • **Similarity**: {similarity_pct}%
                        """)

                # Graph context results
                if ctx.get('graph_results'):
                    st.markdown("**🕸️ Top Graph Context Results:**")
                    for i, item in enumerate(ctx['graph_results'][:3], 1):
                        node_type = item.get('node_type', 'N/A').upper()
                        code = item.get('code', 'N/A')
                        desc = item.get('description', '')[:100]
                        st.markdown(f"""
                        **{i}. {node_type}: {code}**
                        • **Description**: {desc}...
                        """)

                # Show summary metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Vector Results Found", len(ctx.get('vector_results', [])))
                with col2:
                    st.metric("Graph Results Found", len(ctx.get('graph_results', [])))

        if result.suggestions:
            with st.expander("💡 Suggestions"):
                for suggestion in result.suggestions:
                    st.write(f"• {suggestion}")

        if result.metadata.get('llm_enhancement_error'):
            st.warning(f"WARNING: LLM enhancement encountered an issue: {result.metadata['llm_enhancement_error']}")

    elif result.hsn_code:
        st.success(f"**HSN Code Found:** {result.hsn_code}")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confidence", ".1%")
            st.metric("Processing Time", ".2f")

        with col2:
            st.metric("Query Type", result.query_type.title())
            if result.sources_used:
                st.metric("Sources Used", ", ".join(result.sources_used))

        if result.description:
            st.info(f"**Description:** {result.description}")

        if result.export_policy:
            st.write(f"**Export Policy:** {result.export_policy}")

        if result.hierarchical_path:
            st.write(f"**Hierarchical Path:** {result.hierarchical_path}")

        # Display retrieved context for rule-based mode
        rag_mode = result.metadata.get('rag_mode', 'rule_based')
        has_retrieved_context = result.metadata.get('retrieved_context') is not None

        if rag_mode == 'rule_based' and has_retrieved_context:
            ctx = result.metadata['retrieved_context']
            with st.expander("📋 Retrieved Context (Vector + Graph)", expanded=True):
                # Vector search results
                if ctx.get('vector_results'):
                    st.subheader("📄 Vector Search Results (Top 5)")
                    for i, item in enumerate(ctx['vector_results'][:5], 1):
                        similarity_pct = int(item.get('similarity_score', 0) * 100)
                        st.markdown(f"""
                        **{i}. HSN {item.get('hsn_code', 'N/A')}**
                        - **Description**: {item.get('description', '')[:80]}...
                        - **Similarity**: {similarity_pct}%
                        """)

                # Graph context results
                if ctx.get('graph_results'):
                    st.subheader("🕸️ Graph Context Results (Top 3)")
                    for i, item in enumerate(ctx['graph_results'][:3], 1):
                        node_type = item.get('node_type', 'N/A').upper()
                        code = item.get('code', 'N/A')
                        desc = item.get('description', '')[:80]
                        st.markdown(f"""
                        **{i}. {node_type}: {code}**
                        - **Description**: {desc}...
                        """)

                # Show counts
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Vector Results", len(ctx.get('vector_results', [])))
                with col2:
                    st.metric("Graph Results", len(ctx.get('graph_results', [])))

        if result.suggestions:
            with st.expander("Suggestions"):
                for suggestion in result.suggestions:
                    st.write(f"• {suggestion}")

        # Show RAG mode information
        rag_mode = result.metadata.get('rag_mode', 'rule_based')
        if rag_mode in RAG_MODES:
            st.info(f"**Processing Mode**: {RAG_MODES[rag_mode]['name']}")

        # Show LLM enhancement information if available
        if result.metadata.get('llm_enhanced'):
            st.success("**LLM Enhanced**: Response includes AI-generated insights")
            if result.metadata.get('enhanced_description'):
                with st.expander("AI-Generated Insights"):
                    st.write(result.metadata['enhanced_description'])

        if result.metadata.get('llm_enhancement_error'):
            st.warning(f"WARNING: LLM enhancement encountered an issue: {result.metadata['llm_enhancement_error']}")

    else:
        # Check if this is a disambiguation case with multiple options
        disambiguation_options = result.metadata.get('disambiguation_options', [])
        if disambiguation_options:
            st.warning("**Multiple HSN Code Options Found**")
            st.info("Your query matches several possible classifications. Here are the top matching results:")

            # Show retrieved context prominently for disambiguation
            rag_mode = result.metadata.get('rag_mode', 'rule_based')
            has_retrieved_context = result.metadata.get('retrieved_context') is not None

            if rag_mode == 'rule_based' and has_retrieved_context:
                ctx = result.metadata['retrieved_context']
                st.subheader("🔍 Retrieved Results (Vector + Graph)")

                # Vector search results
                if ctx.get('vector_results'):
                    st.markdown("**📄 Top Vector Search Results:**")
                    for i, item in enumerate(ctx['vector_results'][:5], 1):
                        similarity_pct = int(item.get('similarity_score', 0) * 100)
                        st.markdown(f"""
                        **{i}. HSN {item.get('hsn_code', 'N/A')}**
                        • **Description**: {item.get('description', '')[:100]}...
                        • **Similarity**: {similarity_pct}%
                        """)

                # Graph context results
                if ctx.get('graph_results'):
                    st.markdown("**🕸️ Top Graph Context Results:**")
                    for i, item in enumerate(ctx['graph_results'][:3], 1):
                        node_type = item.get('node_type', 'N/A').upper()
                        code = item.get('code', 'N/A')
                        desc = item.get('description', '')[:100]
                        st.markdown(f"""
                        **{i}. {node_type}: {code}**
                        • **Description**: {desc}...
                        """)

                # Show summary metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Vector Results Found", len(ctx.get('vector_results', [])))
                with col2:
                    st.metric("Graph Results Found", len(ctx.get('graph_results', [])))

            # Show disambiguation options
            st.subheader("🎯 Suggested HSN Code Options:")
            for i, option in enumerate(disambiguation_options, 1):
                with st.expander(f"Option {i}: HSN {option['hsn_code']} (Confidence: {option['confidence']:.1%})", expanded=(i<=3)):
                    st.markdown(f"""
                    **HSN Code:** {option['hsn_code']}
                    **Description:** {option['description']}
                    **Similarity Score:** {option['similarity_score']:.1%}
                    """)

            if result.suggestions:
                with st.expander("💡 Additional Suggestions"):
                    for suggestion in result.suggestions:
                        st.write(f"• {suggestion}")

        else:
            st.error("ERROR: No HSN code found for this query")

            # Show retrieved context prominently for rule-based mode when no HSN code found
            rag_mode = result.metadata.get('rag_mode', 'rule_based')
            has_retrieved_context = result.metadata.get('retrieved_context') is not None

            if rag_mode == 'rule_based' and has_retrieved_context:
                ctx = result.metadata['retrieved_context']
                st.subheader("🔍 Retrieved Results (Vector + Graph)")

                # Vector search results
                if ctx.get('vector_results'):
                    st.markdown("**📄 Top Vector Search Results:**")
                    for i, item in enumerate(ctx['vector_results'][:5], 1):
                        similarity_pct = int(item.get('similarity_score', 0) * 100)
                        st.markdown(f"""
                        **{i}. HSN {item.get('hsn_code', 'N/A')}**
                        • **Description**: {item.get('description', '')[:100]}...
                        • **Similarity**: {similarity_pct}%
                        """)

                # Graph context results
                if ctx.get('graph_results'):
                    st.markdown("**🕸️ Top Graph Context Results:**")
                    for i, item in enumerate(ctx['graph_results'][:3], 1):
                        node_type = item.get('node_type', 'N/A').upper()
                        code = item.get('code', 'N/A')
                        desc = item.get('description', '')[:100]
                        st.markdown(f"""
                        **{i}. {node_type}: {code}**
                        • **Description**: {desc}...
                        """)

                # Show summary metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Vector Results Found", len(ctx.get('vector_results', [])))
                with col2:
                    st.metric("Graph Results Found", len(ctx.get('graph_results', [])))

            if result.suggestions:
                st.warning("**Suggestions:**")
                for suggestion in result.suggestions:
                    st.write(f"• {suggestion}")

def main():
    """Main Streamlit application"""
    st.title("HSN Code Classification System")
    st.markdown("*Intelligent HSN code classification using RAG and Knowledge Graphs*")

    # Sidebar
    with st.sidebar:
        st.header("System Status")

        if st.button("Initialize System", type="primary"):
            initialize_system()

        if st.session_state.initialized:
            st.success("System Ready")

            # System metrics
            try:
                metrics = st.session_state.system.get_system_metrics()
                st.subheader("Metrics")
                st.metric("Total Queries", metrics['total_queries'])
                st.metric("Success Rate", ".1%")
                st.metric("Avg Response Time", ".2f")
            except:
                pass

            # Quick actions
            st.subheader("Quick Actions")
            if st.button("Run System Tests"):
                with st.spinner("Running tests..."):
                    test_results = st.session_state.system.run_system_tests()
                    if test_results['overall_status'] == 'passed':
                        st.success(f"Tests Passed: {test_results['passed_tests']}/{test_results['total_tests']}")
                    else:
                        st.error(f"Tests Failed: {test_results['passed_tests']}/{test_results['total_tests']}")

        else:
            st.warning("System not initialized")
            st.info("Click 'Initialize System' to start")

        # RAG Mode Selection
        st.subheader("RAG Mode Selection")
        rag_mode_options = {mode: info['name'] for mode, info in RAG_MODES.items()}
        selected_rag_mode = st.selectbox(
            "Choose RAG processing mode:",
            options=list(rag_mode_options.keys()),
            index=list(rag_mode_options.keys()).index('rule_based'),  # Default to rule_based
            format_func=lambda x: rag_mode_options[x],
            help="Select the RAG approach for processing your queries"
        )

        # Show mode description
        if selected_rag_mode in RAG_MODES:
            st.info(f"**{RAG_MODES[selected_rag_mode]['name']}**: {RAG_MODES[selected_rag_mode]['description']}")

        # Example queries
        st.subheader("Example Queries")
        examples = [
            "What is the HSN code for natural rubber latex?",
            "HSN code for prevulcanised rubber",
            "Rubber products classification",
            "Tell me about HSN 40011010",
            "Similar products to natural rubber latex"
        ]

        for example in examples:
            if st.button(example, key=f"example_{hash(example)}"):
                st.session_state.current_query = example

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Query Interface")

        # Query input
        if 'current_query' in st.session_state:
            default_query = st.session_state.current_query
            del st.session_state.current_query
        else:
            default_query = ""

        query = st.text_input(
            "Enter your product query:",
            value=default_query,
            placeholder="e.g., What is the HSN code for natural rubber latex?",
            key="query_input"
        )

        col_a, col_b, col_c = st.columns([1, 1, 2])
        with col_a:
            submit_button = st.button("Classify", type="primary", use_container_width=True)
        with col_b:
            clear_button = st.button("Clear History", use_container_width=True)

        if clear_button:
            st.session_state.chat_history = []
            st.rerun()

        # Process query
        if submit_button and query.strip():
            if not st.session_state.initialized:
                st.error("ERROR: Please initialize the system first")
            else:
                with st.spinner("Processing query..."):
                    start_time = time.time()
                    result = st.session_state.system.classify_product(
                        query.strip(),
                        rag_mode=selected_rag_mode
                    )
                    processing_time = time.time() - start_time

                    # Add to chat history
                    st.session_state.chat_history.append({
                        'timestamp': datetime.now(),
                        'query': query.strip(),
                        'result': result,
                        'processing_time': processing_time
                    })

                    # Display result
                    st.subheader("Classification Result")
                    display_result(result)

    with col2:
        st.subheader("Recent Activity")

        if st.session_state.chat_history:
            for i, entry in enumerate(reversed(st.session_state.chat_history[-5:])):
                with st.expander(f"Query {len(st.session_state.chat_history)-i}: {entry['query'][:30]}...", expanded=(i==0)):
                    if entry['result'].hsn_code:
                        st.success(f"SUCCESS: {entry['result'].hsn_code}")
                    else:
                        st.error("No result")

                    st.caption(".2f")
                    st.caption(f"Confidence: {entry['result'].confidence:.1%}")
        else:
            st.info("No queries processed yet")

    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit • HSN RAG System v1.0.0*")

if __name__ == "__main__":
    main()