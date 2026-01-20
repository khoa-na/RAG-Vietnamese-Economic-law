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
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print(f"Đã giải phóng thành công: {model_name}")
        else:
            # It's possible the model wasn't running or endpoint involves chat
             print(f"Phản hồi từ server: {response.text}")
             
    except Exception as e:
        print(f"Lỗi kết nối đến Ollama: {e}")

if __name__ == "__main__":
    # List of models used in this project
    models_to_unload = [
        "qwen3-embedding:4b",
        # Add other models here if you use local LLMs (e.g., "llama3")
    ]
    
    for model in models_to_unload:
        unload_model(model)
        
    print("\nĐã gửi lệnh giải phóng VRAM cho các model.")
