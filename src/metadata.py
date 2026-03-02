"""
Metadata extraction for Vietnamese legal document chunks.
Extracts law name from filename and structural metadata (Chương, Mục, Điều) from text.
"""

import os
import re
import sys

# Set encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8')


# ============================================================================
# Law Name Mapping
# ============================================================================

LAW_NAME_MAP = {
    "Luat_Doanh_Nghiep_2020.txt": "Luật Doanh Nghiệp 2020",
    "Luat_Dau_Tu_2020.txt": "Luật Đầu Tư 2020",
    "Luat_Thuong_Mai_2025.txt": "Luật Thương Mại 2025",
    "Luat_Chung_Khoan_2019.txt": "Luật Chứng Khoán 2019",
    "Luat_Ngan_Sach_Nha_Nuoc_2025.txt": "Luật Ngân Sách Nhà Nước 2025",
    "Luat_Lao_Dong_2019.txt": "Luật Lao Động 2019",
    "Luat_So_Huu_Tri_Tue_2025.txt": "Luật Sở Hữu Trí Tuệ 2025",
    "Luat_Canh_Tranh_2018.txt": "Luật Cạnh Tranh 2018",
    "Luat_Quan_Ly_Thue_2019.txt": "Luật Quản Lý Thuế 2019",
}


def extract_law_name(filename: str) -> str:
    """
    Extract law name from filename.
    
    Uses a predefined mapping for known files, falls back to parsing the filename.
    
    Args:
        filename: Basename of the file (e.g., 'Luat_Doanh_Nghiep_2020.txt')
    
    Returns:
        Vietnamese law name (e.g., 'Luật Doanh Nghiệp 2020')
    """
    basename = os.path.basename(filename)
    
    # Check predefined map first
    if basename in LAW_NAME_MAP:
        return LAW_NAME_MAP[basename]
    
    # Fallback: parse from filename
    name = os.path.splitext(basename)[0]  # Remove .txt
    name = name.replace("_", " ")         # Underscores to spaces
    # Replace 'Luat' with 'Luật' if at start
    if name.startswith("Luat "):
        name = "Luật " + name[5:]
    return name


# ============================================================================
# Structure Metadata Extraction
# ============================================================================

# Regex patterns for Vietnamese legal document structure
_CHUONG_PATTERN = re.compile(
    r'Chương\s+([IVXLCDM]+|\d+)\.?\s*(.*?)(?:\n|$)',
    re.IGNORECASE
)

_MUC_PATTERN = re.compile(
    r'Mục\s+(\d+)\.?\s*(.*?)(?:\n|$)',
    re.IGNORECASE
)

_DIEU_PATTERN = re.compile(
    r'Điều\s+(\d+)\.?\s*(.*?)(?:\n|$)'
)


def extract_structure_metadata(text: str) -> dict:
    """
    Extract structural metadata from a chunk of legal text.
    
    Finds the LAST occurrence of Chương/Mục and the FIRST Điều in the chunk,
    which best represents the primary content of the chunk.
    
    Args:
        text: The text content of a document chunk
    
    Returns:
        Dict with keys: chuong, chuong_ten, muc, muc_ten, dieu, dieu_ten
    """
    result = {
        "chuong": "",
        "chuong_ten": "",
        "muc": "",
        "muc_ten": "",
        "dieu": "",
        "dieu_ten": "",
    }
    
    # Find all Chương matches, take the last one (most relevant to chunk content)
    chuong_matches = list(_CHUONG_PATTERN.finditer(text))
    if chuong_matches:
        match = chuong_matches[-1]
        result["chuong"] = f"Chương {match.group(1)}"
        result["chuong_ten"] = match.group(2).strip().rstrip('.')
    
    # Find all Mục matches, take the last one
    muc_matches = list(_MUC_PATTERN.finditer(text))
    if muc_matches:
        match = muc_matches[-1]
        result["muc"] = f"Mục {match.group(1)}"
        result["muc_ten"] = match.group(2).strip().rstrip('.')
    
    # Find all Điều matches, take the first one (primary article in this chunk)
    dieu_match = _DIEU_PATTERN.search(text)
    if dieu_match:
        result["dieu"] = f"Điều {dieu_match.group(1)}"
        result["dieu_ten"] = dieu_match.group(2).strip().rstrip('.')
    
    return result


def enrich_chunk_metadata(chunk, filename: str = None) -> None:
    """
    Enrich a LangChain Document chunk with law name and structure metadata.
    Modifies the chunk's metadata in-place.
    
    Args:
        chunk: A LangChain Document object
        filename: Optional filename override; defaults to chunk.metadata['source']
    """
    source = filename or os.path.basename(chunk.metadata.get("source", ""))
    
    # Add law name
    chunk.metadata["law_name"] = extract_law_name(source)
    
    # Add structural metadata
    structure = extract_structure_metadata(chunk.page_content)
    chunk.metadata.update(structure)


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    # Test law name extraction
    print("=== Law Name Extraction ===")
    test_files = [
        "Luat_Doanh_Nghiep_2020.txt",
        "Luat_Canh_Tranh_2018.txt",
        "Unknown_Law_2025.txt",
    ]
    for f in test_files:
        print(f"  {f} → {extract_law_name(f)}")
    
    # Test structure extraction
    print("\n=== Structure Extraction ===")
    test_texts = [
        "Chương II. THÀNH LẬP DOANH NGHIỆP\nĐiều 17. Quyền thành lập, góp vốn\n1. Tổ chức, cá nhân có quyền...",
        "Mục 1. QUY ĐỊNH CHUNG\nĐiều 54. Nguyên tắc tố tụng cạnh tranh\n1. Hoạt động tố tụng...",
        "Điều 4. Giải thích từ ngữ\nTrong Luật này, các từ ngữ dưới đây...",
        "Một đoạn text không có cấu trúc pháp lý",
    ]
    for text in test_texts:
        meta = extract_structure_metadata(text)
        print(f"\n  Text: {text[:60]}...")
        for k, v in meta.items():
            if v:
                print(f"    {k}: {v}")
