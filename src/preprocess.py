import re
import sys

# Set encoding for Windows console (important for Vietnamese printing)
sys.stdout.reconfigure(encoding='utf-8')

def clean_legal_text(text: str) -> str:
    """
    Tiền xử lý văn bản pháp luật tiếng Việt để dọn dẹp các lỗi định dạng 
    khi copy/phân tích từ file raw (như lỗi đứt dòng, thừa ký tự rác...).
    """
    
    # 1. Xoá các ký tự phân cách rác như: --------, =====, _________ (nguyên dòng)
    text = re.sub(r'^[=\-_]{3,}$', '', text, flags=re.MULTILINE)
    
    # 2. Chuẩn hóa khoảng trắng đầu cuối mỗi dòng
    lines = [line.strip() for line in text.splitlines()]
    text = '\n'.join(lines)
    
    # 3. Gom các dòng bị đứt (hard wrap). 
    # Logic: Nếu có 1 dấu \n duy nhất -> đổi thành khoảng trắng.
    # Nếu có từ 2 dấu \n trở lên -> giữ lại 1 dấu \n (đoạn mới).
    # Regex lookbehind/lookahead để không thay thế \n\n
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    
    # 4. Gom nhiều dòng trống thành tối đa 2 dòng (1 khoảng trắng giữa các đoạn)
    text = re.sub(r'\n{2,}', '\n\n', text)
    
    # 5. Dọn dẹp khoảng trắng liên tiếp (tab/nhiều space)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # 6. Xoá khoảng trắng sau newline
    text = re.sub(r'\n ', '\n', text)
    
    return text.strip()

# Test thử nghiệm nhỏ
if __name__ == "__main__":
    test_text = """
Điều 4. Giải thích từ ngữ

1.
Chứng khoán
là tài sản, bao gồm các loại sau đây:

a)
Cổ phiếu, trái phiếu, chứng chỉ quỹ;

b)
Cơ quan, tổ chức, cá nhân khác có liên quan đến hoạt
động về chứng khoán.

Qu
ỹ
đóng là quỹ đại chúng.

---------------
================

Tài sản cơ sở của chứng khoán phái sinh
(sau đây gọi là tài sản cơ sở) là chứng khoán, chỉ số.
    """
    
    cleaned = clean_legal_text(test_text)
    print("TRƯỚC:")
    print(test_text)
    print("\nSAU:")
    print(cleaned)
