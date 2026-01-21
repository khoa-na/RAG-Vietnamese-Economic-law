# Legal RAG System - Vietnamese Economic Law

A Retrieval-Augmented Generation (RAG) system designed to answer questions about Vietnamese Economic Law using advanced embedding models and Large Language Models. The system specializes in processing legal documents including Enterprise Law, Commercial Law, Investment Law, Securities Law, and other Vietnamese economic regulations.

## Overview

This system combines semantic search capabilities with natural language generation to provide accurate, citation-backed answers to legal questions. It uses a specialized chunking strategy optimized for Vietnamese legal document structure and leverages state-of-the-art embedding models for semantic understanding.

## Key Features

- **Semantic Chunking**: Intelligent document splitting strategy that respects Vietnamese legal document hierarchy (Chapter → Section → Article → Clause → Point) to preserve legal context
- **Advanced Embeddings**: Utilizes `qwen3-embedding:4b` via Ollama for high-quality semantic search with 4-billion parameter model
- **Vector Database**: ChromaDB implementation with cosine similarity for efficient and accurate retrieval
- **LLM Integration**: Connects to MegaLLM (OpenAI-compatible API) for natural language answer generation
- **Source Citation**: Automatically cites specific legal articles and clauses in responses
- **Debug Tools**: Built-in utilities to inspect retrieved chunks and verify retrieval accuracy
- **Resource Management**: VRAM management utilities to optimize memory usage

## System Architecture

The system consists of four main components:

1. **Data Acquisition**: Web crawler for fetching legal documents from official sources
2. **Ingestion Pipeline**: Document processing, chunking, and vector embedding generation
3. **RAG Chatbot**: Interactive question-answering interface with retrieval and generation
4. **Utilities**: Resource management and debugging tools

## Prerequisites

- **Python**: Version 3.8 or higher
- **Ollama**: Required for running the embedding model locally
- **MegaLLM API Key**: Or any OpenAI-compatible API key for the language model
- **System Requirements**: Minimum 8GB RAM recommended, GPU optional but beneficial for faster embeddings

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd RAG-Vietnamese-Economic-law
```

### 2. Install Python Dependencies

```bash
pip install langchain-community langchain-chroma langchain-huggingface langchain-openai langchain-ollama python-dotenv chardet sentence-transformers chromadb
```

### 3. Setup Ollama for Embeddings

Download and install Ollama from [ollama.com](https://ollama.com/).

Pull the required embedding model:

```bash
ollama pull qwen3-embedding:4b
```

Ensure Ollama is running in the background:

```bash
ollama serve
```

### 4. Configure Environment Variables

Create a `.env` file in the project root directory:

```env
megallm_api_key=your_api_key_here
```

Replace `your_api_key_here` with your actual MegaLLM API key.

### 5. Install Playwright (Required for Web Crawler)

```bash
playwright install
```

This installs the necessary browser binaries for web scraping.

## Usage Guide

### Step 1: Data Acquisition (Optional)

If you want to fetch the latest legal documents from official sources:

```bash
python crawler.py
```

This will download the following Vietnamese economic laws:
- Enterprise Law 2020
- Investment Law 2020
- Commercial Law 2025
- Securities Law 2019
- State Budget Law 2025
- Labor Law 2019
- Intellectual Property Law 2025
- Competition Law 2018
- Tax Administration Law 2019

Documents are saved to the `docs/` directory as `.txt` files.

### Step 2: Build the Knowledge Base

Process documents from the `docs/` folder and create the vector database:

```bash
python ingestion_pipeline.py
```

**Important Notes**:
- Ensure Ollama is running before executing this command
- If switching embedding models, delete the `db/` folder first to avoid conflicts
- This process may take several minutes depending on document size
- The system uses a chunk size of 8000 characters with 800 character overlap, optimized for the embedding model's context window

### Step 3: Run the Interactive Chatbot

Start the question-answering system:

```bash
python rag_chatbot.py
```

**Example Interaction**:

```
Bạn hỏi: Mua bán hàng hóa là gì?
AI trả lời: Mua bán hàng hóa là hoạt động thương mại... (Trích dẫn Điều 3, Luật Thương mại)
```

To exit the chatbot, type `exit`, `quit`, or `thoát`.

### Step 4: Debug Mode (Optional)

To inspect which document chunks are being retrieved before the AI generates an answer:

```bash
python 3_rag_debug.py
```

This is useful for:
- Verifying retrieval accuracy
- Understanding which legal articles are being referenced
- Debugging unexpected answers
- Tuning retrieval parameters

### Step 5: Resource Management

Unload the embedding model from VRAM when finished:

```bash
python unload_models.py
```

This frees up GPU/system memory for other tasks.

## Project Structure

```
RAG-Vietnamese-Economic-law/
├── crawler.py              # Web crawler for legal documents from phapluat.gov.vn
├── ingestion_pipeline.py   # Document processing, chunking, and embedding pipeline
├── rag_chatbot.py          # Interactive RAG chatbot interface
├── 3_rag_debug.py          # Debug tool with verbose retrieval output
├── unload_models.py        # Utility to free VRAM by unloading models
├── docs/                   # Directory for legal text files (.txt format)
├── db/                     # ChromaDB vector database storage
├── .env                    # Environment variables (API keys)
├── .gitignore              # Git ignore configuration
└── README.md               # This file
```

## Configuration

### Chunking Parameters

The system uses optimized chunking settings in `ingestion_pipeline.py`:

- **Chunk Size**: 8000 characters (matches embedding model's optimal context window)
- **Chunk Overlap**: 800 characters (10% overlap to preserve context across boundaries)
- **Separators**: Hierarchical splitting based on Vietnamese legal structure

### Retrieval Parameters

Configured in `rag_chatbot.py`:

- **Search Type**: Similarity with score threshold
- **Top K**: 5 most relevant chunks
- **Score Threshold**: 0.2 (minimum similarity score)
- **Distance Metric**: Cosine similarity

### LLM Configuration

- **Model**: openai-gpt-oss-120b (via MegaLLM)
- **Temperature**: 0.3 (balanced between accuracy and creativity)
- **Base URL**: https://ai.megallm.io/v1

## Technical Details

### Embedding Model

The system uses `qwen3-embedding:4b`, a 4-billion parameter embedding model that provides:
- High-quality semantic representations
- Support for Vietnamese language
- Efficient local inference via Ollama
- 8192 token context window

### Document Processing Pipeline

1. **Loading**: Reads all `.txt` files from `docs/` directory with automatic encoding detection
2. **Chunking**: Splits documents using recursive character splitting with legal-specific separators
3. **Embedding**: Generates vector embeddings for each chunk using Ollama
4. **Storage**: Persists embeddings in ChromaDB with cosine similarity indexing

### RAG Chain

The retrieval-augmented generation chain follows this flow:

1. User submits a question
2. Question is embedded using the same embedding model
3. Vector similarity search retrieves top K relevant chunks
4. Retrieved chunks are formatted as context
5. Context + question are sent to LLM with specialized legal prompt
6. LLM generates answer with citations

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError` when running scripts
- **Solution**: Ensure all dependencies are installed via pip

**Issue**: Ollama connection errors
- **Solution**: Verify Ollama is running with `ollama serve`

**Issue**: Empty retrieval results
- **Solution**: Check that documents exist in `docs/` and vector database is built

**Issue**: Out of memory errors
- **Solution**: Reduce chunk size or use `unload_models.py` to free VRAM

**Issue**: Encoding errors on Windows
- **Solution**: Scripts include UTF-8 encoding configuration for Windows console

## Performance Optimization

- **VRAM Usage**: The embedding model requires approximately 4-6GB VRAM
- **Processing Time**: Initial ingestion takes 5-15 minutes for 9 legal documents
- **Query Speed**: Average response time is 2-5 seconds per query
- **Database Size**: Expect 100-500MB for the vector database depending on document corpus

## Future Enhancements

Potential improvements for the system:

- Support for additional legal document formats (PDF, DOCX)
- Multi-turn conversation with context retention
- Advanced citation formatting with clickable references
- Web interface for easier access
- Batch query processing
- Fine-tuned embedding model for Vietnamese legal text
- Hybrid search combining semantic and keyword matching

## Contributing

Contributions are welcome. Please ensure:

- Code follows existing style and structure
- New features include appropriate documentation
- Legal document sources are properly cited
- Changes are tested with sample queries

## License

Please refer to the repository license file for usage terms.

## Acknowledgments

- Legal documents sourced from [phapluat.gov.vn](https://phapluat.gov.vn)
- Embedding model: Qwen3-Embedding by Alibaba Cloud
- LLM service: MegaLLM
- Vector database: ChromaDB
- Framework: LangChain

## Contact

For questions, issues, or suggestions, please open an issue in the repository.