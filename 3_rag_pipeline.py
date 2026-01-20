import os
import sys

# Set encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
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
    print("--- Initializing Legal RAG System ---")
    
    # 1. Load Vector Store
    print("Loading vector store...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="dangvantuan/vietnamese-document-embedding",
        model_kwargs={"trust_remote_code": True}
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
        }
    )

    # 2. Initialize LLM (MegaLLM)
    print("Initializing LLM (MegaLLM)...")
    llm = ChatOpenAI(
        api_key=MEGALLM_API_KEY,
        base_url=MEGALLM_BASE_URL,
        model="openai-gpt-oss-120b", 
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
    
    # Test query first
    # test_query = "mua bán hàng hóa là gì"
    # print(f"\n[Test] Hỏi: {test_query}")
    # print("[Test] Đang suy nghĩ...")
    # response = rag_chain.invoke(test_query)
    # print(f"[Test] Trả lời:\n{response}\n")
    # print("-" * 50)

    while True:
        query = input("\nBạn hỏi: ")
        if query.lower() in ["exit", "quit", "thoát"]:
            break
        
        if not query.strip():
            continue
            
        print("Đang suy nghĩ...")
        try:
            response = rag_chain.invoke(query)
            print(f"\nAI trả lời:\n{response}")
            print("-" * 50)
        except Exception as e:
            print(f"Lỗi: {e}")

if __name__ == "__main__":
    main()
