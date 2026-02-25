"""
Configuration file for LangGraph RAG system.
Contains all configurable parameters for graph behavior.
"""

import os

# Project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Retrieval Configuration
RETRIEVAL_CONFIG = {
    "k": 3,                          # Number of documents to retrieve
    "score_threshold": 0.2,          # Minimum similarity score
    "max_iterations": 2,             # Maximum query rewrite iterations
}

# Document Grading Configuration
GRADING_CONFIG = {
    "relevance_threshold": 0.6,      # Minimum average relevance score
    "min_relevant_docs": 2,          # Minimum number of relevant documents required
    "ambiguity_threshold": 0.5,      # If relevance < this, consider query ambiguous
    "enable_llm_grading": False,     # Use LLM to grade each document
}

# Generation Configuration
GENERATION_CONFIG = {
    "temperature": 0.3,              # LLM temperature for answer generation
    "max_tokens": 1000,              # Maximum tokens in generated answer
}

# Conversation Memory Configuration
MEMORY_CONFIG = {
    "max_messages": 20,              # Maximum messages to keep in history
    "enable_checkpointing": True,    # Enable conversation persistence
}

# Vector Store Configuration
VECTORSTORE_CONFIG = {
    "persist_directory": os.path.join(PROJECT_ROOT, "db", "chroma_db"),
    "collection_metadata": {"hnsw:space": "cosine"}
}

# Data Configuration
DATA_CONFIG = {
    "raw_docs_path": os.path.join(PROJECT_ROOT, "data", "raw"),
}

# LLM Configuration
LLM_CONFIG = {
    "model": "openai-gpt-oss-120b",
    "temperature": 0.3,
    "base_url": "https://ai.megallm.io/v1"
}

# Embedding Configuration
EMBEDDING_CONFIG = {
    "model": "dangvantuan/vietnamese-document-embedding", # Vietnamese model
}

# Debug Configuration
DEBUG_CONFIG = {
    "verbose": False,                # Print detailed execution logs
    "save_graph_visualization": True # Save graph execution as PNG
}
