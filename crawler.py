import asyncio
import os
import sys
import re
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

# Set encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8')

# Configuration
OUTPUT_DIR = "docs"

# List of laws to crawl (URL, Filename)
LAWS_TO_CRAWL = [
    ("https://phapluat.gov.vn/legal-documents/a07ae05c-02d7-4e00-4a92-da0cee037ace?tabName=noidung", "Luat_Doanh_Nghiep_2020.txt"),
    ("https://phapluat.gov.vn/legal-documents/9e563f10-02d7-a500-321d-e490cb59efcd?tabName=noidung", "Luat_Dau_Tu_2020.txt"),
    ("https://phapluat.gov.vn/legal-documents/bc2337e7-0644-b800-e7f8-989ac535f247?tabName=noidung", "Luat_Thuong_Mai_2025.txt"),
    ("https://phapluat.gov.vn/legal-documents/6ac7b55c-02bb-6300-71c8-8ee8503a16a7?tabName=noidung", "Luat_Chung_Khoan_2019.txt"),
    ("https://phapluat.gov.vn/legal-documents/75a8b3a9-0630-4900-c20e-67b4f2580f17?tabName=noidung", "Luat_Ngan_Sach_Nha_Nuoc_2025.txt"),
    ("https://phapluat.gov.vn/legal-documents/475f00ba-02bb-9700-a961-493e70c0abd0?tabName=noidung", "Luat_Lao_Dong_2019.txt"),
    ("https://phapluat.gov.vn/legal-documents/921d2342-0646-8c00-aede-daaa903ab24b?tabName=noidung", "Luat_So_Huu_Tri_Tue_2025.txt"),
    ("https://phapluat.gov.vn/legal-documents/77b35daf-0283-9a00-5495-1162a15ed27c?tabName=noidung", "Luat_Canh_Tranh_2018.txt"),
    ("https://phapluat.gov.vn/legal-documents/799f3750-02ab-a600-58b4-638118dd7253?tabName=noidung", "Luat_Quan_Ly_Thue_2019.txt")

]

def clean_text(text):
    """
    Cleans the raw text:
    - Removes empty lines (more than 2 consecutive).
    - Removes leading/trailing whitespace from lines.
    - Preserves paragraph structure.
    """
    lines = text.splitlines()
    cleaned_lines = []
    
    for line in lines:
        stripped = line.strip()
        if stripped:
            cleaned_lines.append(stripped)
        else:
            # Keep empty lines but limit to 1 consecutive
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
    
    return "\n".join(cleaned_lines)

async def crawl_single_law(page, url, filename):
    print(f"\nDang xu ly: {filename}...")
    try:
        await page.goto(url)
        
        # Wait for dynamic content
        selector = "div.document-content.prose"
        try:
            await page.wait_for_selector(selector, timeout=15000)
        except:
            print(f"Warning: Timeout waiting for selector on {filename}. Page format might differ.")
            return

        # Get content
        content_html = await page.inner_html(selector)
        soup = BeautifulSoup(content_html, "html.parser")
        raw_text = soup.get_text(separator="\n")
        
        # Clean text
        final_text = clean_text(raw_text)
        
        # Save
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            
        output_path = os.path.join(OUTPUT_DIR, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_text)
            
        print(f"OK: {filename} ({len(final_text)} ky tu)")
        
    except Exception as e:
        print(f"Error {filename}: {e}")

async def main_crawler():
    print(f"--- Bat dau cao {len(LAWS_TO_CRAWL)} van ban luat ---")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        for url, filename in LAWS_TO_CRAWL:
            await crawl_single_law(page, url, filename)
            
        await browser.close()
    print("\n--- Hoan tat ---")

if __name__ == "__main__":
    asyncio.run(main_crawler())
