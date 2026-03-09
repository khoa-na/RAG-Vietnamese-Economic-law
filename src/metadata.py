"""
Metadata extraction for Vietnamese legal document chunks.
Extracts law name from filename and structural metadata (Chương, Mục, Điều, Khoản, Điểm) from text.
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
    """
    basename = os.path.basename(filename)

    if basename in LAW_NAME_MAP:
        return LAW_NAME_MAP[basename]

    name = os.path.splitext(basename)[0]
    name = name.replace("_", " ")
    if name.startswith("Luat "):
        name = "Luật " + name[5:]
    return name


# ============================================================================
# Structure Metadata Extraction
# ============================================================================

_CHUONG_PATTERN = re.compile(
    r"Chương\s+([IVXLCDM]+|\d+)\.?\s*(.*?)(?:\n|$)",
    re.IGNORECASE
)

_MUC_PATTERN = re.compile(
    r"Mục\s+(\d+)\.?\s*(.*?)(?:\n|$)",
    re.IGNORECASE
)

_DIEU_PATTERN = re.compile(
    r"Điều\s+(\d+)\.?\s*(.*?)(?:\n|$)"
)

_KHOAN_PATTERN = re.compile(
    r"(?m)^(\d+)\.\s+"
)

_DIEM_PATTERN = re.compile(
    r"(?m)^([a-zđ])\)\s+",
    re.IGNORECASE
)


def extract_structure_metadata(text: str) -> dict:
    """
    Extract structural metadata from a chunk of legal text.

    Returns the most relevant chapter/section/article and the clause/point range
    covered by the chunk.
    """
    result = {
        "chuong": "",
        "chuong_ten": "",
        "muc": "",
        "muc_ten": "",
        "dieu": "",
        "dieu_ten": "",
        "khoan_start": "",
        "khoan_end": "",
        "diem_start": "",
        "diem_end": "",
    }

    chuong_matches = list(_CHUONG_PATTERN.finditer(text))
    if chuong_matches:
        match = chuong_matches[-1]
        result["chuong"] = f"Chương {match.group(1)}"
        result["chuong_ten"] = match.group(2).strip().rstrip('.')

    muc_matches = list(_MUC_PATTERN.finditer(text))
    if muc_matches:
        match = muc_matches[-1]
        result["muc"] = f"Mục {match.group(1)}"
        result["muc_ten"] = match.group(2).strip().rstrip('.')

    dieu_match = _DIEU_PATTERN.search(text)
    if dieu_match:
        result["dieu"] = f"Điều {dieu_match.group(1)}"
        result["dieu_ten"] = dieu_match.group(2).strip().rstrip('.')

    khoan_matches = list(_KHOAN_PATTERN.finditer(text))
    if khoan_matches:
        result["khoan_start"] = khoan_matches[0].group(1)
        result["khoan_end"] = khoan_matches[-1].group(1)

    diem_matches = list(_DIEM_PATTERN.finditer(text))
    if diem_matches:
        result["diem_start"] = diem_matches[0].group(1)
        result["diem_end"] = diem_matches[-1].group(1)

    return result


def enrich_chunk_metadata(chunk, filename: str = None) -> None:
    """
    Enrich a LangChain Document chunk with law name and structure metadata.
    Preserves metadata that was already set during chunk construction.
    """
    source = filename or os.path.basename(chunk.metadata.get("source", ""))

    chunk.metadata["law_name"] = extract_law_name(source)

    structure = extract_structure_metadata(chunk.page_content)
    for key, value in structure.items():
        if value and not chunk.metadata.get(key):
            chunk.metadata[key] = value


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=== Law Name Extraction ===")
    test_files = [
        "Luat_Doanh_Nghiep_2020.txt",
        "Luat_Canh_Tranh_2018.txt",
        "Unknown_Law_2025.txt",
    ]
    for f in test_files:
        print(f"  {f} -> {extract_law_name(f)}")

    print("\n=== Structure Extraction ===")
    test_texts = [
        "Chương II. THÀNH LẬP DOANH NGHIỆP\nĐiều 17. Quyền thành lập, góp vốn\n1. Tổ chức, cá nhân có quyền...",
        "Mục 1. QUY ĐỊNH CHUNG\nĐiều 54. Nguyên tắc tố tụng cạnh tranh\n1. Hoạt động tố tụng...",
        "Điều 4. Giải thích từ ngữ\n1. Chứng khoán là tài sản.\na) Cổ phiếu.\nb) Trái phiếu.",
        "Một đoạn text không có cấu trúc pháp lý",
    ]
    for text in test_texts:
        meta = extract_structure_metadata(text)
        print(f"\n  Text: {text[:60]}...")
        for k, v in meta.items():
            if v:
                print(f"    {k}: {v}")
