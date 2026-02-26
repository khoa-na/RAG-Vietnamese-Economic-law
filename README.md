# Legal RAG System - Vietnamese Economic Law

A Retrieval-Augmented Generation (RAG) system designed to answer questions about Vietnamese Economic Law using advanced embedding models and Large Language Models. The system specializes in processing legal documents including Enterprise Law, Commercial Law, Investment Law, Securities Law, and other Vietnamese economic regulations.

## Overview

This system combines semantic search capabilities with natural language generation to provide accurate, citation-backed answers to legal questions. It uses a specialized chunking strategy optimized for Vietnamese legal document structure and leverages state-of-the-art embedding models for semantic understanding.

## Key Features

- **Semantic Chunking**: Intelligent document splitting strategy that respects Vietnamese legal document hierarchy (Chapter → Section → Article → Clause → Point) to preserve legal context
- **Advanced Embeddings**: Utilizes `dangvantuan/vietnamese-document-embedding` for high-quality semantic search, specifically optimized for Vietnamese legal text.
- **Vector Database**: LanceDB -- disk-native vector store with minimal RAM usage and cosine similarity search.
- **Hybrid Search**: Combines dense vector search with BM25 full-text search (via Tantivy) using Reciprocal Rank Fusion for superior retrieval accuracy.
- **LLM Integration**: Connects to MegaLLM (OpenAI-compatible API) for natural language answer generation.
- **Source Citation**: Automatically cites specific legal articles and clauses in responses.
- **Debug Tools**: Built-in utilities to inspect retrieved chunks and verify retrieval accuracy.
- **Resource Management**: Automatic VRAM cleanup for models.

## System Architecture

The system consists of four main components:

1. **Data Acquisition**: Web crawler for fetching legal documents from official sources
2. **Ingestion Pipeline**: Document processing, chunking, and vector embedding generation
3. **RAG Chatbot**: Interactive question-answering interface with retrieval and generation
4. **Utilities**: Resource management and debugging tools

## Prerequisites

- **Python**: Version 3.8 or higher
- **MegaLLM API Key**: Or any OpenAI-compatible API key for the language model
- **System Requirements**: Minimum 8GB RAM recommended, CUDA-compatible GPU recommended for faster local embeddings (HuggingFace models).

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd RAG-Vietnamese-Economic-law
```

### 2. Install Python Dependencies

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
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
python scripts/crawler.py
```

This will download Vietnamese economic laws to `data/raw/`.

### Step 2: Build the Knowledge Base

Process documents from the `data/raw/` folder and create the vector database:

```bash
python scripts/ingest.py
```

**Important Notes**:
- This uses the `dangvantuan/vietnamese-document-embedding` model (HuggingFace).
- If switching embedding models, delete the `db/` folder first to avoid conflicts.
- The system uses a chunk size of 8000 characters with 800 character overlap.

### Step 3: Run the Interactive Chatbot

Start the question-answering system:

```bash
python main.py
```

**Example Interaction**:

```
Bạn hỏi: Mua bán hàng hóa là gì?
AI trả lời: Mua bán hàng hóa là hoạt động thương mại... (Trích dẫn Điều 3, Luật Thương mại)
```

To exit the chatbot, type `exit`, `quit`, or `thoát`.

### Step 4: Benchmark (Optional)

Run performance benchmarks on the system:

```bash
python scripts/benchmark.py
```

## Project Structure

```
RAG-Vietnamese-Economic-law/
├── main.py                     # Entry point - Interactive RAG chatbot
├── src/                        # Core RAG package
│   ├── __init__.py
│   ├── config.py               # All configuration parameters
│   ├── state.py                # RAGState schema (TypedDict)
│   ├── nodes.py                # Graph node implementations (7 nodes)
│   ├── graph.py                # LangGraph builder and routing
│   └── utils.py                # VRAM cleanup utilities
├── scripts/                    # Utility scripts
│   ├── crawler.py              # Web crawler for legal documents
│   ├── ingest.py               # Document ingestion pipeline
│   └── benchmark.py            # Performance benchmark
├── data/
│   └── raw/                    # Legal text files (.txt format)
├── db/
│   └── lancedb/               # LanceDB vector database storage
├── .env                        # Environment variables (API keys)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Configuration

### Chunking Parameters

The system uses optimized chunking settings in `scripts/ingest.py`:

- **Chunk Size**: 8000 characters (matches embedding model's optimal context window)
- **Chunk Overlap**: 800 characters (10% overlap to preserve context across boundaries)
- **Separators**: Hierarchical splitting based on Vietnamese legal structure

### Retrieval Parameters

Configured in `src/config.py`:

- **Top K**: 3 most relevant chunks (optimized for speed)
- **Score Threshold**: 0.2
- **Document Grading**: Auto-grading by LLM is disabled by default for performance.

### LLM Configuration

- **Model**: openai-gpt-oss-120b (via MegaLLM)
- **Temperature**: 0.3
- **Base URL**: https://ai.megallm.io/v1

## Technical Details

### Embedding Model

The system uses `dangvantuan/vietnamese-document-embedding`, a specialized model for Vietnamese:
- High-quality semantic representations for Vietnamese text
- 4096 token context window
- Runs locally via HuggingFace/Transformers

### Document Processing Pipeline

1. **Loading**: Reads `.txt` files with auto encoding detection
2. **Chunking**: Splits documents using recursive character splitting
3. **Embedding**: Generates vector embeddings for each chunk
4. **Storage**: Persists in LanceDB with cosine similarity indexing

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


**Issue**: Empty retrieval results
- **Solution**: Check that documents exist in `docs/` and vector database is built

**Issue**: Out of memory errors
- **Solution**: Reduce chunk size or use `unload_models.py` to free VRAM

**Issue**: Encoding errors on Windows
- **Solution**: Scripts include UTF-8 encoding configuration for Windows console

## Performance Optimization

- **VRAM Usage**: Optimized to unload models when not in use.
- **Latency**: Reduced retrieval count (k=3) and disabled extra grading steps for faster response times.

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
- Embedding model: `dangvantuan/vietnamese-document-embedding` by Dang Van Tuan
- LLM service: MegaLLM
- Vector database: LanceDB
- Framework: LangChain

## Contact

For questions, issues, or suggestions, please open an issue in the repository.