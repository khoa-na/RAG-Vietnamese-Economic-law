"""
LangGraph-based RAG Chatbot for Vietnamese Economic Law
Interactive chatbot with conversation memory and advanced routing.
"""

import os
import sys
import uuid

# Set encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
from rag_graph import build_rag_graph, visualize_graph
from graph_state import create_initial_state
from unload_models import unload_model

# Load environment variables
load_dotenv()

# Configuration
MEGALLM_API_KEY = os.getenv("megallm_api_key")

if not MEGALLM_API_KEY:
    raise ValueError("MegaLLM API Key not found in .env file. Please check 'megallm_api_key'.")


def print_help():
    """Print available commands."""
    print("\n" + "="*60)
    print("LỆNH HỖ TRỢ:")
    print("="*60)
    print("  exit/quit/thoát  - Thoát chương trình")
    print("  reset            - Bắt đầu cuộc hội thoại mới")
    print("  debug            - Bật/tắt chế độ debug")
    print("  help             - Hiển thị trợ giúp này")
    print("  visualize        - Tạo hình ảnh minh họa graph")
    print("="*60 + "\n")


def main():
    print("="*60)
    print("  HỆ THỐNG RAG LUẬT KINH TẾ VIỆT NAM (LangGraph)")
    print("="*60)
    print("\nĐang khởi tạo hệ thống...")
    
    # Build graph
    app = build_rag_graph()
    print("Graph đã được xây dựng thành công!")
    
    # Initialize conversation
    thread_id = str(uuid.uuid4())  # Unique thread for this session
    debug_mode = False
    
    print("\nHệ thống sẵn sàng. Nhập 'help' để xem các lệnh hỗ trợ.")
    print_help()
    
    while True:
        try:
            # Get user input
            question = input("\nBạn hỏi: ").strip()
            
            if not question:
                continue
            
            # Handle commands
            if question.lower() in ["exit", "quit", "thoát"]:
                print("\n--- Đang thoát hệ thống ---")
                print("Giải phóng tài nguyên...")
                print("Giải phóng tài nguyên...")
                unload_model("dangvantuan/vietnamese-document-embedding")
                print("Hẹn gặp lại!")
                break
            
            elif question.lower() == "reset":
                thread_id = str(uuid.uuid4())
                print("[OK] Đã bắt đầu cuộc hội thoại mới!")
                continue
            
            elif question.lower() == "debug":
                debug_mode = not debug_mode
                status = "BẬT" if debug_mode else "TẮT"
                print(f"[OK] Chế độ debug: {status}")
                continue
            
            elif question.lower() == "help":
                print_help()
                continue
            
            elif question.lower() == "visualize":
                print("Đang tạo hình ảnh minh họa graph...")
                visualize_graph()
                continue
            
            # Process question
            print("Đang xử lý...")
            
            # Create initial state
            initial_state = create_initial_state(question)
            
            # Configure execution
            config = {"configurable": {"thread_id": thread_id}}
            
            # Execute graph
            if debug_mode:
                print("\n" + "-"*60)
                print("DEBUG: Graph Execution")
                print("-"*60)
            
            final_state = None
            for state_update in app.stream(initial_state, config):
                if debug_mode:
                    node_name = list(state_update.keys())[0]
                    node_state = state_update[node_name]
                    print(f"\n[Node] {node_name}")
                    
                    # Print relevant debug info
                    if "route_decision" in node_state:
                        print(f"   Route: {node_state['route_decision']}")
                    if "debug_info" in node_state:
                        print(f"   Info: {node_state['debug_info']}")
                
                final_state = state_update
            
            if debug_mode:
                print("-"*60 + "\n")
            
            # Extract answer from final state
            if final_state:
                last_node_name = list(final_state.keys())[0]
                result = final_state[last_node_name]
                answer = result.get("generation", "Không có câu trả lời.")
                
                # Print answer
                print("\nAI trả lời:")
                print("-"*60)
                print(answer)
                print("-"*60)
                
                # Show stats if debug mode
                if debug_mode:
                    print(f"\n[Thống kê]")
                    print(f"   - Số tài liệu truy xuất: {len(result.get('context', []))}")
                    print(f"   - Điểm liên quan: {result.get('relevance_score', 0):.2f}")
                    print(f"   - Số lần viết lại: {result.get('iteration_count', 0)}")
            else:
                print("[Lỗi] Không nhận được kết quả từ graph.")
        
        except KeyboardInterrupt:
            print("\n\n--- Đã dừng bởi người dùng ---")
            unload_model("dangvantuan/vietnamese-document-embedding")
            break
        
        except Exception as e:
            print(f"\n[Lỗi] {e}")
            if debug_mode:
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
