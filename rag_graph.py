"""
LangGraph RAG System - Main Graph Implementation
Builds the state graph with nodes and conditional edges.
"""

import os
import sys
from typing import Literal

# Set encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8')

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from graph_state import RAGState
from graph_nodes import (
    analyze_query_node,
    retrieve_documents_node,
    grade_documents_node,
    rewrite_query_node,
    generate_answer_node,
    direct_answer_node,
    clarify_query_node
)
from graph_config import MEMORY_CONFIG


def route_after_analysis(state: RAGState) -> Literal["retrieve", "direct_answer", "clarify"]:
    """
    Conditional edge after query analysis.
    Routes to retrieval, direct answer, or clarification based on query type.
    """
    decision = state.get("route_decision", "retrieve")
    if decision == "direct_answer":
        return "direct_answer"
    elif decision == "clarify":
        return "clarify"
    return "retrieve"


def route_after_grading(state: RAGState) -> Literal["generate", "rewrite", "clarify"]:
    """
    Conditional edge after document grading.
    Routes to generation, query rewriting, or clarification based on relevance.
    """
    decision = state.get("route_decision", "generate")
    if decision == "rewrite":
        return "rewrite"
    elif decision == "clarify":
        return "clarify"
    return "generate"


def build_rag_graph():
    """
    Build and compile the RAG state graph.
    
    Graph structure:
    1. analyze_query -> [retrieve OR direct_answer]
    2. retrieve -> grade_documents
    3. grade_documents -> [generate OR rewrite]
    4. rewrite -> retrieve (loop back)
    5. generate -> END
    6. direct_answer -> END
    
    Returns:
        Compiled graph ready for execution
    """
    # Create graph
    workflow = StateGraph(RAGState)
    
    # Add nodes
    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("retrieve", retrieve_documents_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("generate", generate_answer_node)
    workflow.add_node("direct_answer", direct_answer_node)
    workflow.add_node("clarify", clarify_query_node)
    
    # Set entry point
    workflow.set_entry_point("analyze_query")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "analyze_query",
        route_after_analysis,
        {
            "retrieve": "retrieve",
            "direct_answer": "direct_answer",
            "clarify": "clarify"
        }
    )
    
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate": "generate",
            "rewrite": "rewrite_query",
            "clarify": "clarify"
        }
    )
    
    # Add normal edges
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("rewrite_query", "retrieve")  # Loop back
    workflow.add_edge("generate", END)
    workflow.add_edge("direct_answer", END)
    workflow.add_edge("clarify", END)  # Ask for clarification and end
    
    # Compile with memory (checkpointing)
    if MEMORY_CONFIG["enable_checkpointing"]:
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)
    else:
        app = workflow.compile()
    
    return app


def invoke_rag_graph(question: str, thread_id: str = "default", verbose: bool = False):
    """
    Invoke the RAG graph with a question.
    
    Args:
        question: User's question
        thread_id: Conversation thread ID for memory
        verbose: Print debug information
    
    Returns:
        Final state after graph execution
    """
    app = build_rag_graph()
    
    # Create initial state
    from graph_state import create_initial_state
    initial_state = create_initial_state(question)
    
    # Configure execution
    config = {"configurable": {"thread_id": thread_id}}
    
    # Execute graph
    if verbose:
        print("\n--- Graph Execution Started ---")
    
    final_state = None
    for state in app.stream(initial_state, config):
        if verbose:
            print(f"\nNode: {list(state.keys())[0]}")
            print(f"State update: {state}")
        final_state = state
    
    if verbose:
        print("\n--- Graph Execution Completed ---")
    
    # Extract the actual state from the last node output
    if final_state:
        # The stream returns dict with node name as key
        # Get the last value which contains the updated state
        last_node_name = list(final_state.keys())[0]
        return final_state[last_node_name]
    
    return None


# For visualization (optional)
def visualize_graph(output_path: str = "graph_visualization.png"):
    """
    Generate a visualization of the graph structure.
    Requires graphviz to be installed.
    
    Args:
        output_path: Path to save the visualization
    """
    try:
        app = build_rag_graph()
        
        # Get the graph as mermaid or PNG
        from IPython.display import Image, display
        
        # Try to get PNG
        try:
            png_data = app.get_graph().draw_mermaid_png()
            with open(output_path, "wb") as f:
                f.write(png_data)
            print(f"Graph visualization saved to: {output_path}")
        except Exception as e:
            # Fallback to mermaid text
            mermaid = app.get_graph().draw_mermaid()
            print("Graph structure (Mermaid):")
            print(mermaid)
            
    except Exception as e:
        print(f"Could not generate visualization: {e}")
        print("Install graphviz for PNG output: pip install pygraphviz")


if __name__ == "__main__":
    # Test the graph
    print("Testing RAG Graph...")
    
    # Test question
    test_question = "Doanh nghiệp là gì?"
    
    result = invoke_rag_graph(test_question, verbose=True)
    
    if result:
        print("\n" + "="*50)
        print("FINAL ANSWER:")
        print("="*50)
        print(result.get("generation", "No answer generated"))
        print("\n" + "="*50)
        print("DEBUG INFO:")
        print(result.get("debug_info", {}))
