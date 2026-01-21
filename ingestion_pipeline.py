import os
import sys

# Set encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8')
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from datetime import datetime
from unload_models import unload_model
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")
    
    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")
    
    # Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True}
    )
    
    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")
    
   
    print(f"Found {len(documents)} documents in total.")
    
    for i, doc in enumerate(documents[:2]):  # Show first 2 documents
        print(f"\nDocument {i+1}:")
        print(f"  Source: {os.path.basename(doc.metadata['source'])}") # Clean up display
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        # print(f"  metadata: {doc.metadata}") # Reduce noise

    return documents

def split_documents(documents, chunk_size=8000, chunk_overlap=800):
    """Split documents into smaller chunks with overlap, optimized for Vietnamese legal text"""
    print("Splitting documents into chunks...")
    
    # Define separators for Vietnamese Legal Text
    # 1. Split by Articles (Điều X.)
    # 2. Split by Clauses (1. ) - carefully to avoid confusion with dates or other numbers
    # 3. Split by Points (a) )
    separators = [
        r"\nChương [IVXLCDM]+\b", # Chapter (Roman numerals)
        r"\nChương \d+\b",        # Chapter (Digits)
        r"\nMục \d+\b",           # Section
        r"\nĐiều \d+\.",          # Article
        r"\n\d+\. ",              # Clause
        r"\n[a-zđ]\) ",           # Point
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
    
        for i, chunk in enumerate(chunks[:2]): # Reduced preview to 2 for conciseness
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            # print(f"Content:")
            print(f"Content preview: {chunk.page_content[:200]}...") # Preview instead of full content
            print("-" * 50)
        
        if len(chunks) > 2:
            print(f"\n... and {len(chunks) - 2} more chunks")
    
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")
        
    # embedding_model = HuggingFaceEmbeddings(
    #     model_name="dangvantuan/vietnamese-document-embedding",
    #     model_kwargs={"trust_remote_code": True}
    # )
    
    # Define embedding model based on provider (now using hardcoded choice for this script or could load from config)
    # Ideally should import from graph_config but keeping independent for now or matching logic
    
    # We will use the model specified in the plan: dangvantuan/vietnamese-document-embedding
    print("Initializing HuggingFace embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="dangvantuan/vietnamese-document-embedding",
        model_kwargs={"trust_remote_code": True}
    )
    
    # Create ChromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")
    
    print(f"Vector store created and saved to {persist_directory}")

    # Unload model to free VRAM
    print("\n--- Cleaning up resources ---")
    # Unload model to free VRAM (Handled by OS/Torch for HF, or explicit cleanup)
    print("\n--- Cleaning up resources ---")
    # unload_model("qwen3-embedding:4b") # Not needed for HF local model in this script context immediately, 
    # but good practice would be torch.cuda.empty_cache() if running in loop.
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return vectorstore

def main():
    #1 load documents
    documents = load_documents(docs_path="docs")
    #2 split documents
    chunks = split_documents(documents)
    #3 create vector store
    vectorstore = create_vector_store(chunks)

if __name__ == "__main__":
    main()