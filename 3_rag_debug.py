import os
import sys

# Set encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Configuration
PERSISTENCE_DIRECTORY = "db/chroma_db"
MEGALLM_API_KEY = os.getenv("megallm_api_key")
MEGALLM_BASE_URL = "https://ai.megallm.io/v1"

if not MEGALLM_API_KEY:
    raise ValueError("MegaLLM API Key not found in .env file. Please check 'megallm_api_key'.")

def main():
    print("--- Initializing Legal RAG System (DEBUG MODE) ---")
    
    # 1. Load Vector Store
    print("Loading vector store...")
    # embedding_model = HuggingFaceEmbeddings(
    #     model_name="dangvantuan/vietnamese-document-embedding",
    #     model_kwargs={"trust_remote_code": True}
    # )
    
    embedding_model = OllamaEmbeddings(
        model="qwen3-embedding:4b"
    )
    
    vectorstore = Chroma(
        persist_directory=PERSISTENCE_DIRECTORY,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 5, 
            "score_threshold": 0.2
        } # Fetch 5 chunks for inspection
    )

    # 2. Initialize LLM (MegaLLM)
    llm = ChatOpenAI(
        api_key=MEGALLM_API_KEY,
        base_url=MEGALLM_BASE_URL,
        model="gpt-4o",
        temperature=0.3 
    )

    # 3. Define Prompt
    system_prompt = """Bạn là một trợ lý luật sư AI chuyên về Luật Kinh tế Việt Nam.
Dựa vào ngữ cảnh pháp lý sau đây để trả lời câu hỏi của người dùng.
Nếu không có thông tin trong ngữ cảnh, hãy nói "Tôi không tìm thấy thông tin trong tài liệu được cung cấp".
Đừng tự bịa ra luật. Luôn trích dẫn điều khoản cụ thể nếu có.

Ngữ cảnh:
{context}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    def format_docs(docs):
        # Join content for the LLM
        return "\n\n".join(doc.page_content for doc in docs)

    # 4. Build Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 5. Interactive Loop
    print("\n--- Hệ thống sẵn sàng. Nhập 'exit' để thoát. ---")
    
    while True:
        query = input("\n[DEBUG] Bạn hỏi: ")
        if query.lower() in ["exit", "quit", "thoát"]:
            break
        
        if not query.strip():
            continue
            
        print("\n[1/3] Đang tìm kiếm tài liệu (Retrieving)...")
        retrieved_docs = retriever.invoke(query)
        
        print("\n" + "="*50)
        print(f" TÌM THẤY {len(retrieved_docs)} CHUNK LIÊN QUAN")
        print("="*50)
        
        for i, doc in enumerate(retrieved_docs, 1):
            source = os.path.basename(doc.metadata.get('source', 'Unknown'))
            print(f"\n[Chunk {i}] Nguồn: {source}")
            print(f"Nội dung ({len(doc.page_content)} chars):")
            print("-" * 20)
            print(doc.page_content.strip()[:500] + ("..." if len(doc.page_content) > 500 else "")) # Show partial content to keep it readable
            print("-" * 20)
            
        print("\n[2/3] Đang gửi cho AI xử lý...")
        try:
            # We already displayed chunks, so we can just invoke the chain or invoke llm manually if we want to be explicit,
            # but invoking the chain recalculates retrieval. To be efficient and consistent, we could pass the docs directly,
            # but standard chain invocation is robust enough here.
            
            response = rag_chain.invoke(query)
            print(f"\n[3/3] AI TRẢ LỜI:\n{response}")
            print("-" * 50)
        except Exception as e:
            print(f"Lỗi: {e}")

if __name__ == "__main__":
    main()
