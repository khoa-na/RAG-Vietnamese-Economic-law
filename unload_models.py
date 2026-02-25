import sys

# Set encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8')

def unload_model(model_name):
    """Unload a HuggingFace/PyTorch model and free VRAM."""
    print(f"Đang giải phóng model: {model_name}...")
    try:
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
    models_to_unload = [
        "dangvantuan/vietnamese-document-embedding",
    ]
    
    for model in models_to_unload:
        unload_model(model)
        
    print("\nĐã gửi lệnh giải phóng VRAM cho các model.")

