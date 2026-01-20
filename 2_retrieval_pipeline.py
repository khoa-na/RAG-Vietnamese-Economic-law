import sys
import os

# Set encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8')

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"

def main():
    # Load embeddings and vector store
    print("Loading vector store...")
    # embedding_model = HuggingFaceEmbeddings(
    #         model_name="dangvantuan/vietnamese-document-embedding",
    #         model_kwargs={"trust_remote_code": True}
    #     )

    from langchain_ollama import OllamaEmbeddings
    embedding_model = OllamaEmbeddings(
        model="qwen3-embedding:4b"
    )

    db = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}  
    )

    # Search for relevant documents
    query = "mua bán hàng hóa là gì"
    print(f"\nUser Query: {query}")
    print("Searching...")

    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 5,
            "score_threshold": 0.2
        }
    )

    relevant_docs = retriever.invoke(query)

    # Display results
    print(f"\n--- Found {len(relevant_docs)} relevant documents ---")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"\nResult {i}:")
        print(f"Source: {os.path.basename(doc.metadata.get('source', 'Unknown'))}")
        # print(f"Score: {doc.metadata.get('score', 'N/A')}") # Note: score not directly available in simple invoke
        print(f"Content preview:\n{doc.page_content[:500]}...") # Show first 500 chars
        print("-" * 50)

if __name__ == "__main__":
    main()


