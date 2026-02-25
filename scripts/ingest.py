import os
import sys

# Project root resolution
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Set encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8')

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from src.utils import unload_model
from src.config import VECTORSTORE_CONFIG, EMBEDDING_CONFIG, DATA_CONFIG

load_dotenv()


def load_documents(docs_path=None):
    """Load all text files from the data/raw directory"""
    if docs_path is None:
        docs_path = DATA_CONFIG["raw_docs_path"]
    
    print(f"Loading documents from {docs_path}...")
    
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your legal documents.")
    
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True}
    )
    
    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your legal documents.")
    
    print(f"Found {len(documents)} documents in total.")
    
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}:")
        print(f"  Source: {os.path.basename(doc.metadata['source'])}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")

    return documents


def split_documents(documents, chunk_size=8000, chunk_overlap=800):
    """Split documents into smaller chunks with overlap, optimized for Vietnamese legal text"""
    print("Splitting documents into chunks...")
    
    separators = [
        r"\nChương [IVXLCDM]+\b",
        r"\nChương \d+\b",
        r"\nMục \d+\b",
        r"\nĐiều \d+\.",
        r"\n\d+\. ",
        r"\n[a-zđ]\) ",
        "\n\n",
        "\n",
        " ",
        ""
    ]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=separators,
        is_separator_regex=True
    )
    
    chunks = text_splitter.split_documents(documents)
    
    if chunks:
        for i, chunk in enumerate(chunks[:2]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content preview: {chunk.page_content[:200]}...")
            print("-" * 50)
        
        if len(chunks) > 2:
            print(f"\n... and {len(chunks) - 2} more chunks")
    
    return chunks


def create_vector_store(chunks, persist_directory=None):
    """Create and persist ChromaDB vector store"""
    if persist_directory is None:
        persist_directory = VECTORSTORE_CONFIG["persist_directory"]
    
    print("Creating embeddings and storing in ChromaDB...")
    
    print("Initializing HuggingFace embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_CONFIG["model"],
        model_kwargs={"trust_remote_code": True}
    )
    
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata=VECTORSTORE_CONFIG["collection_metadata"]
    )
    print("--- Finished creating vector store ---")
    
    print(f"Vector store created and saved to {persist_directory}")

    # Free VRAM
    print("\n--- Cleaning up resources ---")
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return vectorstore


def main():
    documents = load_documents()
    chunks = split_documents(documents)
    vectorstore = create_vector_store(chunks)


if __name__ == "__main__":
    main()