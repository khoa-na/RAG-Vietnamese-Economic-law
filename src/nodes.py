"""
Node implementations for LangGraph RAG system.
Each node is a function that takes state and returns updated state.
"""

import os
import sys
from typing import Dict, Any

# Set encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
from langchain_community.vectorstores import LanceDB
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from src.state import RAGState
from src.config import (
    RETRIEVAL_CONFIG, 
    GRADING_CONFIG, 
    GENERATION_CONFIG,
    VECTORSTORE_CONFIG,
    LLM_CONFIG,
    EMBEDDING_CONFIG
)

# Load environment variables
load_dotenv()

# Initialize components (lazy loading)
_vectorstore = None
_llm = None
_embedding_model = None
_fts_initialized = False


def get_vectorstore():
    """Lazy load vector store with hybrid search support."""
    global _vectorstore
    if _vectorstore is None:
        from lancedb.rerankers import LinearCombinationReranker
        
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_CONFIG["model"],
            model_kwargs={"trust_remote_code": True}
        )
        reranker = LinearCombinationReranker(weight=0.7)  # 0.7 dense + 0.3 BM25
        _vectorstore = LanceDB(
            uri=VECTORSTORE_CONFIG["uri"],
            embedding=embedding_model,
            table_name=VECTORSTORE_CONFIG["table_name"],
            reranker=reranker
        )
    return _vectorstore


def get_llm():
    """Lazy load LLM."""
    global _llm
    if _llm is None:
        api_key = os.getenv("megallm_api_key")
        if not api_key:
            raise ValueError("MegaLLM API Key not found in .env file")
        _llm = ChatOpenAI(
            api_key=api_key,
            base_url=LLM_CONFIG["base_url"],
            model=LLM_CONFIG["model"],
            temperature=LLM_CONFIG["temperature"]
        )
    return _llm


# ============================================================================
# NODE 1: Analyze Query
# ============================================================================

def analyze_query_node(state: RAGState) -> Dict[str, Any]:
    """
    Analyze the user's question to determine if retrieval is needed.
    
    Simple questions like greetings don't need retrieval.
    Vague questions need clarification.
    Legal questions need retrieval from the knowledge base.
    """
    question = state["question"].lower().strip()
    
    # Simple heuristics for direct answers
    greetings = ["xin chào", "hello", "hi", "chào", "cảm ơn", "thank"]
    
    # Check if it's a greeting
    if any(greeting in question for greeting in greetings) and len(question.split()) < 5:
        return {
            "route_decision": "direct_answer",
            "debug_info": {"analysis": "Greeting detected, no retrieval needed"}
        }
    
    # Detect overly vague questions (too short and generic)
    vague_keywords = ["thủ tục", "điều kiện", "quy định", "luật", "pháp luật", "hồ sơ"]
    word_count = len(question.split())
    
    # If question is just 1-2 words and contains vague keywords → clarify
    if word_count <= 2 and any(keyword in question for keyword in vague_keywords):
        return {
            "route_decision": "clarify",
            "debug_info": {"analysis": "Vague question detected (too short), needs clarification"}
        }
    
    # If question ends with "?" but has no specific context
    if question.endswith("?") and word_count <= 3:
        # Check if it's just a generic term
        if any(keyword == question.replace("?", "").strip() for keyword in vague_keywords):
            return {
                "route_decision": "clarify",
                "debug_info": {"analysis": "Generic question detected, needs clarification"}
            }
    
    # Most legal questions need retrieval
    return {
        "route_decision": "retrieve",
        "debug_info": {"analysis": "Legal question detected, retrieval needed"}
    }


# ============================================================================
# NODE 2: Retrieve Documents
# ============================================================================

def retrieve_documents_node(state: RAGState) -> Dict[str, Any]:
    """
    Retrieve relevant documents from the vector store.
    Uses hybrid search (BM25 + Dense) for better retrieval.
    """
    vectorstore = get_vectorstore()
    
    # Use rewritten question if available, otherwise use original
    query = state.get("rewritten_question") or state["question"]
    
    # Use LanceDB native hybrid search API (bypasses langchain wrapper bugs)
    from langchain_core.documents import Document
    import json
    
    tbl = vectorstore.get_table()
    
    # Create FTS index if not exists (idempotent with replace=True)
    global _fts_initialized
    if not _fts_initialized:
        tbl.create_fts_index("text", replace=True)
        _fts_initialized = True
    
    # Embed query for vector search
    query_vector = vectorstore._embedding.embed_query(query)
    
    # Hybrid search: vector + FTS combined with reranker
    from lancedb.rerankers import LinearCombinationReranker
    reranker = LinearCombinationReranker(weight=0.7)
    
    hybrid_results = (
        tbl.search(query=query_vector, vector_column_name="vector")
        .limit(RETRIEVAL_CONFIG["k"] * 2)  # Get more candidates for reranking
        .to_list()
    )
    
    fts_results = (
        tbl.search(query=query, query_type="fts")
        .limit(RETRIEVAL_CONFIG["k"] * 2)
        .to_list()
    )
    
    # Merge results by RRF (Reciprocal Rank Fusion)
    scored = {}
    for rank, r in enumerate(hybrid_results):
        scored[r["id"]] = {"data": r, "rrf": 1.0 / (rank + 60)}
    for rank, r in enumerate(fts_results):
        key = r["id"]
        rrf_score = 1.0 / (rank + 60)
        if key in scored:
            scored[key]["rrf"] += rrf_score  # Boost docs found in both
        else:
            scored[key] = {"data": r, "rrf": rrf_score}
    
    # Sort by combined RRF score and take top-K
    ranked = sorted(scored.values(), key=lambda x: x["rrf"], reverse=True)
    top_k = ranked[:RETRIEVAL_CONFIG["k"]]
    
    # Convert to LangChain Documents
    results = []
    for item in top_k:
        r = item["data"]
        metadata = json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"]
        results.append(Document(page_content=r["text"], metadata=metadata))
    
    return {
        "context": results,
        "debug_info": {
            "retrieved_count": len(results),
            "query_used": query,
            "search_type": "hybrid (BM25 + Dense + RRF)",
            "vector_hits": len(hybrid_results),
            "fts_hits": len(fts_results)
        }
    }


# ============================================================================
# NODE 3: Grade Documents
# ============================================================================

def grade_documents_node(state: RAGState) -> Dict[str, Any]:
    """
    Grade the relevance of retrieved documents to the question using LLM.
    Determines if documents are good enough or if query needs rewriting/clarification.
    """
    question = state["question"]
    documents = state["context"]
    
    # If no documents retrieved, need to rewrite
    if not documents:
        return {
            "relevance_score": 0.0,
            "relevant_docs_count": 0,
            "needs_rewrite": True,
            "route_decision": "clarify",  # Ask for clarification
            "debug_info": {"grading": "No documents retrieved, needs clarification"}
        }
    
    # Use LLM to grade each document if enabled
    if GRADING_CONFIG.get("enable_llm_grading", False):
        llm = get_llm()
        
        # Grading prompt
        grading_prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn là chuyên gia đánh giá độ liên quan của tài liệu pháp lý.
Nhiệm vụ: Đánh giá xem tài liệu có liên quan đến câu hỏi của người dùng không.

Trả lời CHÍNH XÁC một trong hai từ:
- "relevant" nếu tài liệu có thông tin liên quan đến câu hỏi
- "irrelevant" nếu tài liệu không liên quan

Không giải thích, chỉ trả lời một từ."""),
            ("human", """Câu hỏi: {question}

Tài liệu:
{document}

Đánh giá:""")
        ])
        
        chain = grading_prompt | llm | StrOutputParser()
        
        # Grade documents in PARALLEL using batch
        relevant_count = 0
        graded_docs = []
        
        # Prepare batch inputs
        batch_inputs = [
            {
                "question": question,
                "document": doc.page_content[:500]  # First 500 chars
            }
            for doc in documents
        ]
        
        try:
            # Batch invoke for parallel processing
            grades = chain.batch(batch_inputs)
            
            # Process results
            for doc, grade in zip(documents, grades):
                is_relevant = "relevant" in grade.strip().lower()
                if is_relevant:
                    relevant_count += 1
                    graded_docs.append(doc)
        except Exception as e:
            # If batch grading fails, fall back to assuming all relevant
            print(f"Grading error: {e}")
            relevant_count = len(documents)
            graded_docs = documents
        
        # Calculate relevance score
        avg_score = relevant_count / len(documents) if documents else 0.0
        
        # Update context with only relevant docs
        updated_context = graded_docs if graded_docs else documents
        
    else:
        # Simple heuristic grading (fallback)
        relevant_count = len(documents)
        avg_score = 0.8
        updated_context = documents
    
    # Detect ambiguous query
    is_ambiguous = avg_score < GRADING_CONFIG.get("ambiguity_threshold", 0.5)
    is_first_iteration = state["iteration_count"] == 0
    
    # Decision logic
    if is_ambiguous and is_first_iteration:
        # Very ambiguous on first try → ask for clarification
        route = "clarify"
        needs_rewrite = False
    elif relevant_count < GRADING_CONFIG["min_relevant_docs"] and state["iteration_count"] < RETRIEVAL_CONFIG["max_iterations"]:
        # Not enough relevant docs → rewrite query
        route = "rewrite"
        needs_rewrite = True
    else:
        # Good enough → generate answer
        route = "generate"
        needs_rewrite = False
    
    return {
        "context": updated_context,
        "relevance_score": avg_score,
        "relevant_docs_count": relevant_count,
        "needs_rewrite": needs_rewrite,
        "route_decision": route,
        "debug_info": {
            "grading": f"Found {relevant_count}/{len(documents)} relevant documents",
            "avg_score": avg_score,
            "is_ambiguous": is_ambiguous
        }
    }


# ============================================================================
# NODE 4: Rewrite Query
# ============================================================================

def rewrite_query_node(state: RAGState) -> Dict[str, Any]:
    """
    Rewrite the user's question to improve retrieval results.
    """
    llm = get_llm()
    
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", """Bạn là chuyên gia về luật kinh tế Việt Nam. 
Nhiệm vụ của bạn là viết lại câu hỏi của người dùng để tìm kiếm tài liệu pháp lý chính xác hơn.

Hãy:
- Làm rõ các thuật ngữ pháp lý
- Thêm ngữ cảnh liên quan
- Sử dụng từ khóa chính xác trong luật

Chỉ trả về câu hỏi đã được viết lại, không giải thích."""),
        ("human", "Câu hỏi gốc: {question}\n\nViết lại câu hỏi:")
    ])
    
    chain = rewrite_prompt | llm | StrOutputParser()
    rewritten = chain.invoke({"question": state["question"]})
    
    return {
        "rewritten_question": rewritten.strip(),
        "iteration_count": state["iteration_count"] + 1,
        "debug_info": {
            "rewrite": f"Iteration {state['iteration_count'] + 1}",
            "rewritten_query": rewritten.strip()
        }
    }


# ============================================================================
# NODE 5: Generate Answer
# ============================================================================

def generate_answer_node(state: RAGState) -> Dict[str, Any]:
    """
    Generate final answer based on retrieved context.
    """
    llm = get_llm()
    question = state["question"]
    documents = state["context"]
    
    # Format context
    context_text = "\n\n".join([doc.page_content for doc in documents])
    
    # Create prompt
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
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context_text, "question": question})
    
    # Update messages
    new_messages = [
        HumanMessage(content=question),
        AIMessage(content=answer)
    ]
    
    return {
        "generation": answer,
        "messages": new_messages,
        "route_decision": "end",
        "debug_info": {
            "generation": "Answer generated successfully",
            "context_length": len(context_text)
        }
    }


# ============================================================================
# NODE 6: Direct Answer (No Retrieval)
# ============================================================================

def direct_answer_node(state: RAGState) -> Dict[str, Any]:
    """
    Provide direct answer without retrieval for simple queries.
    """
    question = state["question"]
    
    # Simple responses for greetings
    if "xin chào" in question.lower() or "hello" in question.lower():
        answer = "Xin chào! Tôi là trợ lý AI chuyên về Luật Kinh tế Việt Nam. Tôi có thể giúp bạn tìm hiểu về các quy định pháp luật. Bạn cần hỏi gì?"
    elif "cảm ơn" in question.lower():
        answer = "Rất vui được giúp đỡ! Nếu bạn có thêm câu hỏi về luật, đừng ngại hỏi nhé."
    else:
        answer = "Tôi là trợ lý AI về Luật Kinh tế Việt Nam. Vui lòng đặt câu hỏi cụ thể về pháp luật để tôi có thể hỗ trợ bạn tốt hơn."
    
    # Update messages
    new_messages = [
        HumanMessage(content=question),
        AIMessage(content=answer)
    ]
    
    return {
        "generation": answer,
        "messages": new_messages,
        "route_decision": "end",
        "debug_info": {"direct_answer": "Simple response provided"}
    }


# ============================================================================
# NODE 7: Clarify Query (Ask for More Details)
# ============================================================================

def clarify_query_node(state: RAGState) -> Dict[str, Any]:
    """
    Ask user for clarification when query is too ambiguous.
    Provides suggestions based on common legal topics.
    """
    question = state["question"]
    
    # Generate clarification message
    clarification = f"""Câu hỏi "{question}" còn chung chung. Bạn có thể làm rõ hơn không?

Ví dụ, bạn muốn hỏi về:
1. Thủ tục thành lập doanh nghiệp
2. Thủ tục đăng ký thuế
3. Thủ tục cấp giấy phép kinh doanh
4. Thủ tục đầu tư
5. Thủ tục khác (vui lòng nêu rõ)

Vui lòng đặt lại câu hỏi cụ thể hơn để tôi có thể hỗ trợ tốt hơn."""
    
    # Update messages
    new_messages = [
        HumanMessage(content=question),
        AIMessage(content=clarification)
    ]
    
    return {
        "generation": clarification,
        "messages": new_messages,
        "route_decision": "end",
        "debug_info": {
            "clarification": "Asked user for more specific query",
            "reason": "Ambiguous query detected"
        }
    }

