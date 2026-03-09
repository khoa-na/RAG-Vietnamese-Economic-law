# Legal RAG System - Vietnamese Economic Law

A Retrieval-Augmented Generation (RAG) system designed to answer questions about Vietnamese Economic Law using advanced embedding models and Large Language Models. The system specializes in processing legal documents including Enterprise Law, Commercial Law, Investment Law, Securities Law, and other Vietnamese economic regulations.

## Overview

This system combines semantic search capabilities with natural language generation to provide accurate, citation-backed answers to legal questions. It uses a legal-aware chunking strategy optimized for Vietnamese legal document hierarchy and leverages state-of-the-art embedding models for semantic understanding.

## Key Features

- **Legal-Aware Chunking**: Hierarchical document splitting that prioritizes `Điều -> Khoản -> Điểm` boundaries before falling back to character-based splitting.
- **Data Preprocessing**: Regex-based cleaning of raw legal text to fix hard-wraps, remove separator lines, and join broken paragraphs.
- **Metadata Enrichment**: Automatic extraction of law name, chapter (Chương), section (Mục), article (Điều), clause range (Khoản), and point range (Điểm) metadata for each chunk.
- **Advanced Embeddings**: Uses `dangvantuan/vietnamese-document-embedding` for semantic search on Vietnamese legal text.
- **Vector Database**: LanceDB for disk-native vector storage with cosine similarity search and full-text search support.
- **Hybrid Search**: Combines dense vector retrieval with BM25 full-text search using Reciprocal Rank Fusion (RRF).
- **LLM Integration**: Connects to MegaLLM (OpenAI-compatible API) for natural language answer generation.
- **Source Citation**: Answers are grounded in retrieved legal chunks and include article-level citations.
- **Debug Tools**: CLI debug mode and benchmark utilities for retrieval inspection and latency measurement.
- **Resource Management**: Includes VRAM cleanup utilities for local embedding workflows.

## System Architecture

The system consists of five main components:

1. **Data Acquisition**: Web crawler for fetching legal documents from official sources.
2. **Data Processing**: Preprocessing raw text to repair formatting before chunking.
3. **Ingestion Pipeline**: Legal-aware chunking, metadata enrichment, and vector embedding generation.
4. **RAG Chatbot**: Interactive question-answering interface with retrieval, routing, and generation.
5. **Utilities**: Benchmarking, visualization, and resource management tools.

## Prerequisites

- **Python**: Version 3.8 or higher
- **MegaLLM API Key**: Or any OpenAI-compatible API key for the language model
- **System Requirements**: Minimum 8GB RAM recommended, CUDA-compatible GPU recommended for faster local embeddings

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd RAG-Vietnamese-Economic-law
```

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

### 4. Install Playwright (Optional, Required for Web Crawler)

```bash
playwright install
```

This installs the browser binaries used by the crawler.

## Usage Guide

### Step 1: Data Acquisition (Optional)

If you want to fetch the legal documents from official sources:

```bash
python scripts/crawler.py
```

This downloads Vietnamese economic law documents into `data/raw/`.

### Step 2: Build the Knowledge Base

Process documents from `data/raw/` and create the vector database:

```bash
python scripts/ingest.py
```

**Important Notes**:
- This uses the `dangvantuan/vietnamese-document-embedding` model locally via HuggingFace.
- The ingestion pipeline now uses legal-aware chunking with a default target size of `4000` characters.
- `chunk_overlap=200` is only used by the fallback character splitter when an individual legal unit is still too long.
- If you change the embedding model, chunking schema, or chunk metadata fields, delete `db/lancedb/` and rebuild to avoid LanceDB schema conflicts.

### Step 3: Run the Interactive Chatbot

Start the question-answering system:

```bash
python main.py
```

**Example Interaction**:

```text
Bạn hỏi: Mua bán hàng hóa là gì?
AI trả lời: Mua bán hàng hóa là hoạt động thương mại... (Trích dẫn Điều 3, Luật Thương mại)
```

To exit the chatbot, type `exit`, `quit`, or `thoát`.

### Step 4: Benchmark (Optional)

Run a simple latency benchmark on the graph pipeline:

```bash
python scripts/benchmark.py
```

## Project Structure

```text
RAG-Vietnamese-Economic-law/
├── main.py                     # Entry point - Interactive RAG chatbot
├── src/                        # Core RAG package
│   ├── __init__.py
│   ├── config.py               # Configuration parameters
│   ├── state.py                # RAGState schema (TypedDict)
│   ├── nodes.py                # Graph node implementations
│   ├── graph.py                # LangGraph builder and routing
│   ├── preprocess.py           # Vietnamese legal text cleaning
│   ├── metadata.py             # Law name and structure metadata extraction
│   └── utils.py                # VRAM cleanup utilities
├── scripts/                    # Utility scripts
│   ├── crawler.py              # Web crawler for legal documents
│   ├── ingest.py               # Legal-aware ingestion pipeline
│   └── benchmark.py            # Performance benchmark
├── data/
│   └── raw/                    # Legal text files (.txt format)
├── db/
│   └── lancedb/                # LanceDB vector database storage
├── .env                        # Environment variables (API keys)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Configuration

### Chunking Parameters

The default chunking settings live in `scripts/ingest.py`:

- **Target Chunk Size**: `4000` characters
- **Fallback Overlap**: `200` characters
- **Hierarchy**: `Điều -> Khoản -> Điểm -> fallback char split`
- **Merge Rule**: legal units are merged only within the same `Điều`

### Retrieval Parameters

Configured in `src/config.py`:

- **Top K**: 3 most relevant chunks
- **Score Threshold**: 0.2
- **Max Rewrite Iterations**: 2
- **Document Grading**: LLM grading is disabled by default for performance

### LLM Configuration

- **Model**: `openai-gpt-oss-120b` (via MegaLLM)
- **Temperature**: `0.3`
- **Base URL**: `https://ai.megallm.io/v1`

## Technical Details

### Embedding Model

The system uses `dangvantuan/vietnamese-document-embedding`, a specialized model for Vietnamese:

- High-quality semantic representations for Vietnamese text
- Suitable for local embedding generation through HuggingFace
- Shared between ingestion and retrieval

### Document Processing Pipeline

1. **Loading**: Reads `.txt` files with auto encoding detection.
2. **Preprocessing**: Cleans raw legal text by fixing hard-wraps, removing separator lines, and normalizing whitespace.
3. **Chunking**: Splits documents with a legal-aware hierarchy: `Điều -> Khoản -> Điểm`, then falls back to character splitting only when a single legal unit is too large.
4. **Metadata Enrichment**: Extracts law name plus `Chương`, `Mục`, `Điều`, `Khoản`, and `Điểm` ranges for each chunk.
5. **Embedding**: Generates vector embeddings for each chunk.
6. **Storage**: Persists chunks in LanceDB and creates FTS indexes for hybrid search.

### RAG Graph Flow

The LangGraph pipeline follows this flow:

1. User submits a question.
2. The system analyzes whether retrieval is needed or whether the query is too vague.
3. Hybrid retrieval fetches relevant chunks from LanceDB using dense search and BM25.
4. Retrieved chunks are optionally graded for relevance.
5. The system either generates an answer, rewrites the query, or asks the user for clarification.
6. The final answer is generated with the retrieved legal context.

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError` when running scripts
- **Solution**: Ensure all dependencies are installed with `pip install -r requirements.txt`.

**Issue**: LanceDB error like `Field '...' not found in target schema`
- **Solution**: Delete `db/lancedb/` and rebuild with `python scripts/ingest.py`. This usually happens after changing chunk metadata fields or chunking logic.

**Issue**: Empty retrieval results
- **Solution**: Check that documents exist in `data/raw/` and that the vector database has been rebuilt.

**Issue**: Out-of-memory errors during ingestion
- **Solution**: Reduce the target chunk size, ingest on CPU, or free GPU memory before embedding.

**Issue**: Encoding errors on Windows
- **Solution**: The scripts configure UTF-8 console output, but your terminal environment still needs to support UTF-8.

## Performance Optimization

- **VRAM Usage**: Models are loaded lazily and GPU cache is cleared after ingestion.
- **Latency**: Retrieval uses `k=3` by default and LLM grading is disabled by default to reduce response time.
- **Retrieval Quality**: Legal-aware chunking improves citation precision compared with large mixed-article chunks.

## Future Enhancements

Potential improvements for the system:

- Support for additional legal document formats (PDF, DOCX)
- Metadata-based filtering during retrieval (for example by law name)
- Better answer formatting at `Điều/Khoản` granularity
- Persistent conversation memory beyond in-memory checkpointing
- Web UI (Gradio or Streamlit)
- Formal evaluation set with retrieval and answer-quality metrics
- Fine-tuned embedding model for Vietnamese legal text

## Contributing

Contributions are welcome. Please ensure:

- Code follows the existing style and structure
- New features include appropriate documentation
- Legal document sources are properly cited
- Changes are tested with sample queries or ingestion runs

## License

Please refer to the repository license file for usage terms.

## Acknowledgments

- Legal documents sourced from [phapluat.gov.vn](https://phapluat.gov.vn)
- Embedding model: `dangvantuan/vietnamese-document-embedding` by Dang Van Tuan
- LLM service: MegaLLM
- Vector database: LanceDB
- Frameworks: LangChain and LangGraph

## Contact

For questions, issues, or suggestions, please open an issue in the repository.
