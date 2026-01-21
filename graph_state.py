"""
State schema and node implementations for LangGraph RAG system.
"""

import os
import sys
from typing import List, TypedDict, Annotated
from operator import add

# Set encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8')

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


class RAGState(TypedDict):
    """
    State schema for the RAG graph.
    This state is passed between all nodes and updated throughout execution.
    """
    # User input
    question: str
    
    # Conversation history
    messages: Annotated[List[BaseMessage], add]
    
    # Retrieved documents
    context: List[Document]
    
    # Generated answer
    generation: str
    
    # Relevance metrics
    relevance_score: float
    relevant_docs_count: int
    
    # Query rewriting
    needs_rewrite: bool
    rewritten_question: str
    iteration_count: int
    
    # Routing decisions
    route_decision: str  # "retrieve", "direct_answer", "generate", "rewrite"
    
    # Debug information
    debug_info: dict


def create_initial_state(question: str, messages: List[BaseMessage] = None) -> RAGState:
    """
    Create initial state for a new query.
    
    Args:
        question: User's question
        messages: Existing conversation history (optional)
    
    Returns:
        Initial RAGState
    """
    return RAGState(
        question=question,
        messages=messages or [],
        context=[],
        generation="",
        relevance_score=0.0,
        relevant_docs_count=0,
        needs_rewrite=False,
        rewritten_question="",
        iteration_count=0,
        route_decision="",
        debug_info={}
    )
