
import time
import sys
import os
from dotenv import load_dotenv

# Project root resolution
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Load env variables including API key
load_dotenv()

from src.graph import build_rag_graph
from src.state import create_initial_state

def benchmark():
    print("--- Starting Benchmark ---")
    start_time = time.time()
    
    print("Building graph...")
    app = build_rag_graph()
    build_time = time.time()
    print(f"Graph build time: {build_time - start_time:.2f}s")
    
    question = "Doanh nghiệp tư nhân là gì?"
    print(f"\nProcessing Question: {question}")
    
    initial_state = create_initial_state(question)
    config = {"configurable": {"thread_id": "benchmark_test"}}
    
    last_node_end = build_time
    
    print("\n--- Node Execution Times ---")
    for output in app.stream(initial_state, config):
        current_time = time.time()
        node_name = list(output.keys())[0]
        duration = current_time - last_node_end
        print(f"Node '{node_name}': {duration:.2f}s")
        last_node_end = current_time
        
        # Print extra info if available
        state = output[node_name]
        if "debug_info" in state:
             print(f"  Debug: {state['debug_info']}")

    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f}s")

if __name__ == "__main__":
    benchmark()
