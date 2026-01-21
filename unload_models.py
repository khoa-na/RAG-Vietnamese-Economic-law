import requests
import sys

# Set encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8')

def unload_model(model_name):
    print(f"Đang giải phóng model: {model_name}...")
    url = "http://localhost:11434/api/generate"
    
    # Sending keep_alive: 0 instantly unloads the model
    data = {
        "model": model_name,
        "keep_alive": 0
    }
    
    try:
        if "qwen" in model_name or "llama" in model_name or ":" in model_name:
            # Assume Ollama model if it has a tag or typical name
            response = requests.post(url, json=data)
            if response.status_code == 200:
                print(f"Đã giải phóng thành công: {model_name}")
            else:
                # It's possible the model wasn't running or endpoint involves chat
                 print(f"Phản hồi từ server: {response.text}")
        else:
             # Assume HuggingFace/Torch model
             import torch
             import gc
             if torch.cuda.is_available():
                 torch.cuda.empty_cache()
                 gc.collect()
                 print(f"Đã giải phóng VRAM cho model: {model_name}")
             else:
                 print("Không tìm thấy GPU để giải phóng.")
                 
    except Exception as e:
        print(f"Lỗi khi giải phóng model: {e}")

if __name__ == "__main__":
    # List of models used in this project
    models_to_unload = [
        "dangvantuan/vietnamese-document-embedding", 
        "qwen3-embedding:4b",
        # Add other models here if you use local LLMs (e.g., "llama3")
    ]
    
    for model in models_to_unload:
        unload_model(model)
        
    print("\nĐã gửi lệnh giải phóng VRAM cho các model.")
