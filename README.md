# Legal RAG System - Vietnamese Economic Law

A Retrieval-Augmented Generation (RAG) system designed to answer questions about Vietnamese Economic Law (Enterprise Law, Commercial Law, etc.) using a specialized embedding model and a Large Language Model (MegaLLM).

## Features

- **Semantic Chunking**: Optimized splitting strategy for legal documents (Article -> Clause -> Point) to preserve context.
- **Advanced Embeddings**: Uses **`qwen3-embedding:4b`** via **Ollama** for state-of-the-art semantic search.
- **Vector Search**: ChromaDB implementation for efficient retrieval.
- **LLM Integration**: Connects to MegaLLM (OpenAI-compatible) for natural language answers.
- **Debug Tools**: Built-in tools to inspect retrieved chunks and verify source citations.

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com/) (Required for embeddings)
- [MegaLLM API Key](https://megallm.io) (or any OpenAI-compatible API key)

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd RAG-Vietnamese-Economic-law
    ```

2.  **Install dependencies**:
    ```bash
    pip install langchain-community langchain-chroma langchain-huggingface langchain-openai langchain-ollama python-dotenv chardet sentence-transformers chromadb
    ```

3.  **Setup Ollama (Embeddings)**:
    - Download and install [Ollama](https://ollama.com/).
    - Pull the embedding model:
      ```bash
      ollama pull qwen3-embedding:4b
      ```
    - Ensure Ollama is running (`ollama serve`).

4.  **Configure Environment**:
    Create a `.env` file in the root directory:
    ```env
    megallm_api_key=your_api_key_here
    ```

5.  **Install Playwright** (for Crawler):
    ```bash
    playwright install
    ```

## Usage

### 1. Ingestion (Build Knowledge Base)
Process documents from the `docs/` folder and build the Vector Store.
*Note: If you are switching models, delete the old `db/` folder first.*
```bash
python ingestion_pipeline.py
```
*Tip: Ensure Ollama is running in the background.*

### 2. Run RAG Chatbot
Start the interactive Q&A system.
```bash
python rag_chatbot.py
```
**Example Interaction:**
> **Bạn hỏi**: Mua bán hàng hóa là gì?
> **AI trả lời**: Mua bán hàng hóa là hoạt động thương mại... (Trích dẫn Điều 3, Luật Thương mại)

### 3. Debug Mode
If you want to see exactly which documents are being retrieved before the AI answers:
```bash
python 3_rag_debug.py
```

### 4. Data Acquisition (Crawler)
Fetch legal documents from `phapluat.gov.vn`.
```bash
python 0_crawler.py
```

### 5. Manage Resources (VRAM)
Unload the embedding model from memory (VRAM) when finished to free up resources.
```bash
python unload_models.py
```

## Project Structure

- `0_crawler.py`: Fetch legal text from `phapluat.gov.vn`.
- `ingestion_pipeline.py`: Data loading and embedding (Ollama).
- `rag_chatbot.py`: Full RAG application.
- `3_rag_debug.py`: Debug tool with verbose retrieval output.
- `docs/`: Place your legal text files (`.txt`) here.
- `docs/`: Place your legal text files (`.txt`) here.
- `db/`: Stores the ChromaDB vector database.
- `unload_models.py`: Utility to unload models from VRAM.