import os
import re
import sys

# Project root resolution
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Set encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import LanceDB
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import DATA_CONFIG, EMBEDDING_CONFIG, VECTORSTORE_CONFIG
from src.metadata import enrich_chunk_metadata
from src.preprocess import clean_legal_text

load_dotenv()

ARTICLE_PATTERN = re.compile(r"(?m)^Điều\s+(\d+)\.\s*(.*)$")
CLAUSE_PATTERN = re.compile(r"(?m)^(\d+)\.\s+")
POINT_PATTERN = re.compile(r"(?m)^([a-zđ])\)\s+", re.IGNORECASE)
CHUONG_PATTERN = re.compile(r"(?m)^Chương\s+([IVXLCDM]+|\d+)\.?\s*(.*)$", re.IGNORECASE)
MUC_PATTERN = re.compile(r"(?m)^Mục\s+(\d+)\.?\s*(.*)$", re.IGNORECASE)

FALLBACK_SEPARATORS = [
    r"\n[a-zđ]\) ",
    "\n\n",
    "\n",
    " ",
    "",
]

CHUNK_METADATA_DEFAULTS = {
    "chunk_strategy": "legal_hierarchical_v1",
    "chunk_target_size": 0,
    "chunk_level": "",
    "dieu": "",
    "dieu_ten": "",
    "chuong": "",
    "chuong_ten": "",
    "muc": "",
    "muc_ten": "",
    "khoan_start": "",
    "khoan_end": "",
    "diem_start": "",
    "diem_end": "",
    "legal_unit_count": 0,
    "fragment_index": 0,
    "fragment_count": 0,
}


def load_documents(docs_path=None):
    """Load all text files from the data/raw directory."""
    if docs_path is None:
        docs_path = DATA_CONFIG["raw_docs_path"]

    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your legal documents.")

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True}
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your legal documents.")

    print(f"Found {len(documents)} documents in total.")

    print("Preprocessing documents to clean up formatting issues...")
    for doc in documents:
        original_len = len(doc.page_content)
        doc.page_content = clean_legal_text(doc.page_content)
        doc.metadata["preprocessed"] = True
        doc.metadata["original_length"] = original_len
        doc.metadata["cleaned_length"] = len(doc.page_content)

    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}:")
        print(f"  Source: {os.path.basename(doc.metadata['source'])}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")

    return documents


def _compose_text(*parts):
    clean_parts = [part.strip() for part in parts if part and part.strip()]
    return "\n\n".join(clean_parts).strip()


def _compose_list_body(prefix, segments):
    clean_segments = [segment.strip() for segment in segments if segment and segment.strip()]
    if prefix and clean_segments:
        return prefix.strip() + "\n" + "\n".join(clean_segments)
    if prefix:
        return prefix.strip()
    return "\n".join(clean_segments).strip()


def _build_fallback_splitter(chunk_size, chunk_overlap):
    safe_chunk_size = max(800, chunk_size)
    safe_overlap = min(chunk_overlap, max(0, safe_chunk_size // 5))
    return RecursiveCharacterTextSplitter(
        chunk_size=safe_chunk_size,
        chunk_overlap=safe_overlap,
        separators=FALLBACK_SEPARATORS,
        is_separator_regex=True
    )


def _last_match_before(pattern, text, pos):
    matches = list(pattern.finditer(text[:pos]))
    return matches[-1] if matches else None


def _extract_context(full_text, article_start):
    context = {}

    chuong_match = _last_match_before(CHUONG_PATTERN, full_text, article_start)
    if chuong_match:
        context["chuong"] = f"Chương {chuong_match.group(1)}"
        context["chuong_ten"] = chuong_match.group(2).strip().rstrip(".")

    muc_match = _last_match_before(MUC_PATTERN, full_text, article_start)
    if muc_match:
        context["muc"] = f"Mục {muc_match.group(1)}"
        context["muc_ten"] = muc_match.group(2).strip().rstrip(".")

    return context


def _compose_context_prefix(context):
    lines = []
    if context.get("chuong"):
        line = context["chuong"]
        if context.get("chuong_ten"):
            line = f"{line}. {context['chuong_ten']}"
        lines.append(line)
    if context.get("muc"):
        line = context["muc"]
        if context.get("muc_ten"):
            line = f"{line}. {context['muc_ten']}"
        lines.append(line)
    return _compose_text(*lines)


def _make_document(source_doc, text, extra_metadata):
    metadata = dict(source_doc.metadata)
    metadata.update(CHUNK_METADATA_DEFAULTS)
    metadata.update(extra_metadata)
    return Document(page_content=text.strip(), metadata=metadata)


def _extract_article_parts(article_text):
    clause_matches = list(CLAUSE_PATTERN.finditer(article_text))
    if not clause_matches:
        return article_text.strip(), []

    header_text = article_text[:clause_matches[0].start()].strip()
    clauses = []
    for idx, match in enumerate(clause_matches):
        start = match.start()
        end = clause_matches[idx + 1].start() if idx + 1 < len(clause_matches) else len(article_text)
        clauses.append({
            "number": match.group(1),
            "text": article_text[start:end].strip(),
        })
    return header_text, clauses


def _extract_point_parts(clause_text):
    point_matches = list(POINT_PATTERN.finditer(clause_text))
    if not point_matches:
        return clause_text.strip(), []

    clause_intro = clause_text[:point_matches[0].start()].strip()
    points = []
    for idx, match in enumerate(point_matches):
        start = match.start()
        end = point_matches[idx + 1].start() if idx + 1 < len(point_matches) else len(clause_text)
        points.append({
            "label": match.group(1),
            "text": clause_text[start:end].strip(),
        })
    return clause_intro, points


def _fallback_units(text, chunk_size, chunk_overlap, prefix_text, metadata):
    available_size = max(800, chunk_size - len(prefix_text) - 40)
    splitter = _build_fallback_splitter(available_size, chunk_overlap)
    parts = splitter.split_text(text.strip())
    units = []
    for index, part in enumerate(parts, start=1):
        unit_metadata = dict(metadata)
        unit_metadata["chunk_level"] = f"{metadata.get('chunk_level', 'legal_unit')}_fragment"
        unit_metadata["fragment_index"] = index
        unit_metadata["fragment_count"] = len(parts)
        units.append({
            "text": part.strip(),
            "metadata": unit_metadata,
        })
    return units


def _point_units(clause_intro, points, chunk_size, chunk_overlap, prefix_text, clause_number):
    units = []
    current_points = []
    current_labels = []

    def flush():
        if not current_points:
            return
        units.append({
            "text": _compose_list_body(clause_intro, current_points),
            "metadata": {
                "chunk_level": "point_group",
                "khoan_start": clause_number,
                "khoan_end": clause_number,
                "diem_start": current_labels[0],
                "diem_end": current_labels[-1],
            }
        })

    for point in points:
        candidate_points = current_points + [point["text"]]
        candidate_body = _compose_list_body(clause_intro, candidate_points)
        candidate_text = _compose_text(prefix_text, candidate_body)

        if current_points and len(candidate_text) > chunk_size:
            flush()
            current_points = [point["text"]]
            current_labels = [point["label"]]
            continue

        point_text = _compose_text(prefix_text, _compose_list_body(clause_intro, [point["text"]]))
        if not current_points and len(point_text) > chunk_size:
            units.extend(
                _fallback_units(
                    _compose_list_body(clause_intro, [point["text"]]),
                    chunk_size,
                    chunk_overlap,
                    prefix_text,
                    {
                        "chunk_level": "point_group",
                        "khoan_start": clause_number,
                        "khoan_end": clause_number,
                        "diem_start": point["label"],
                        "diem_end": point["label"],
                    }
                )
            )
            continue

        current_points.append(point["text"])
        current_labels.append(point["label"])

    flush()
    return units


def _expand_clause_units(clause, chunk_size, chunk_overlap, prefix_text):
    clause_metadata = {
        "chunk_level": "clause",
        "khoan_start": clause["number"],
        "khoan_end": clause["number"],
        "diem_start": "",
        "diem_end": "",
    }
    clause_text = clause["text"].strip()

    if len(_compose_text(prefix_text, clause_text)) <= chunk_size:
        return [{
            "text": clause_text,
            "metadata": clause_metadata,
        }]

    clause_intro, points = _extract_point_parts(clause_text)
    if points:
        return _point_units(clause_intro, points, chunk_size, chunk_overlap, prefix_text, clause["number"])

    return _fallback_units(clause_text, chunk_size, chunk_overlap, prefix_text, clause_metadata)


def _first_non_empty(units, key):
    for unit in units:
        value = unit["metadata"].get(key)
        if value:
            return value
    return ""


def _last_non_empty(units, key):
    for unit in reversed(units):
        value = unit["metadata"].get(key)
        if value:
            return value
    return ""


def _derive_chunk_level(units):
    if len(units) == 1:
        return units[0]["metadata"].get("chunk_level", "legal_unit")

    clause_numbers = {unit["metadata"].get("khoan_start", "") for unit in units if unit["metadata"].get("khoan_start")}
    if len(clause_numbers) <= 1:
        return "point_group"
    return "clause_group"


def _build_chunk_from_units(source_doc, prefix_text, units, base_metadata):
    body = "\n\n".join(unit["text"] for unit in units if unit["text"].strip())
    chunk_text = _compose_text(prefix_text, body)

    chunk_metadata = dict(base_metadata)
    chunk_metadata["chunk_level"] = _derive_chunk_level(units)
    chunk_metadata["khoan_start"] = _first_non_empty(units, "khoan_start")
    chunk_metadata["khoan_end"] = _last_non_empty(units, "khoan_end")
    chunk_metadata["diem_start"] = _first_non_empty(units, "diem_start")
    chunk_metadata["diem_end"] = _last_non_empty(units, "diem_end")
    chunk_metadata["legal_unit_count"] = len(units)

    if len(units) == 1:
        for key in ("fragment_index", "fragment_count"):
            if key in units[0]["metadata"]:
                chunk_metadata[key] = units[0]["metadata"][key]

    return _make_document(source_doc, chunk_text, chunk_metadata)


def _article_base_metadata(context, article_number, article_title, chunk_size):
    metadata = {
        "chunk_strategy": "legal_hierarchical_v1",
        "chunk_target_size": chunk_size,
        "dieu": f"Điều {article_number}",
        "dieu_ten": article_title.strip().rstrip("."),
        "khoan_start": "",
        "khoan_end": "",
        "diem_start": "",
        "diem_end": "",
    }
    metadata.update(context)
    return metadata


def _split_preamble(document, text, chunk_size, chunk_overlap):
    if not text.strip():
        return []

    splitter = _build_fallback_splitter(chunk_size, chunk_overlap)
    parts = splitter.split_text(text.strip())
    chunks = []
    for index, part in enumerate(parts, start=1):
        chunks.append(
            _make_document(
                document,
                part,
                {
                    "chunk_strategy": "legal_hierarchical_v1",
                    "chunk_level": "preamble",
                    "fragment_index": index,
                    "fragment_count": len(parts),
                }
            )
        )
    return chunks


def _split_article(document, article_text, context, chunk_size, chunk_overlap):
    article_match = ARTICLE_PATTERN.search(article_text)
    if not article_match:
        return _split_preamble(document, article_text, chunk_size, chunk_overlap)

    article_number = article_match.group(1)
    article_title = article_match.group(2)
    base_metadata = _article_base_metadata(context, article_number, article_title, chunk_size)
    context_prefix = _compose_context_prefix(context)

    header_text, clauses = _extract_article_parts(article_text)
    if not clauses:
        article_heading, _, article_body = article_text.partition("\n")
        prefix_text = _compose_text(context_prefix, article_heading)
        article_body = article_body.strip()

        if not article_body:
            return [
                _make_document(
                    document,
                    prefix_text,
                    {**base_metadata, "chunk_level": "article", "legal_unit_count": 1}
                )
            ]

        if len(_compose_text(prefix_text, article_body)) <= chunk_size:
            return [
                _make_document(
                    document,
                    _compose_text(prefix_text, article_body),
                    {**base_metadata, "chunk_level": "article", "legal_unit_count": 1}
                )
            ]

        units = _fallback_units(
            article_body,
            chunk_size,
            chunk_overlap,
            prefix_text,
            {
                "chunk_level": "article",
                "khoan_start": "",
                "khoan_end": "",
                "diem_start": "",
                "diem_end": "",
            }
        )
        return [_build_chunk_from_units(document, prefix_text, [unit], base_metadata) for unit in units]

    prefix_text = _compose_text(context_prefix, header_text)
    clause_units = []
    for clause in clauses:
        clause_units.extend(_expand_clause_units(clause, chunk_size, chunk_overlap, prefix_text))

    chunks = []
    current_units = []
    for unit in clause_units:
        candidate_units = current_units + [unit]
        candidate_body = "\n\n".join(item["text"] for item in candidate_units)
        candidate_text = _compose_text(prefix_text, candidate_body)

        if current_units and len(candidate_text) > chunk_size:
            chunks.append(_build_chunk_from_units(document, prefix_text, current_units, base_metadata))
            current_units = [unit]
            continue

        current_units = candidate_units

    if current_units:
        chunks.append(_build_chunk_from_units(document, prefix_text, current_units, base_metadata))

    return chunks


def split_documents(documents, chunk_size=4000, chunk_overlap=200):
    """
    Split documents using legal-aware hierarchy:
    Điều -> Khoản -> Điểm -> fallback character splitting.

    `chunk_overlap` is only used when an individual legal unit is still too long
    and must be split by the fallback character splitter.
    """
    print("Splitting documents into legal-aware chunks...")

    chunks = []
    for document in documents:
        text = document.page_content
        article_matches = list(ARTICLE_PATTERN.finditer(text))

        if not article_matches:
            chunks.extend(_split_preamble(document, text, chunk_size, chunk_overlap))
            continue

        preamble = text[:article_matches[0].start()]
        chunks.extend(_split_preamble(document, preamble, chunk_size, chunk_overlap))

        for idx, article_match in enumerate(article_matches):
            start = article_match.start()
            end = article_matches[idx + 1].start() if idx + 1 < len(article_matches) else len(text)
            article_text = text[start:end].strip()
            context = _extract_context(text, start)
            chunks.extend(_split_article(document, article_text, context, chunk_size, chunk_overlap))

    if chunks:
        for i, chunk in enumerate(chunks[:2]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Level: {chunk.metadata.get('chunk_level', 'N/A')}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content preview: {chunk.page_content[:200]}...")
            print("-" * 50)

        if len(chunks) > 2:
            print(f"\n... and {len(chunks) - 2} more chunks")

    return chunks


def create_vector_store(chunks, uri=None, table_name=None):
    """Create and persist LanceDB vector store with progress bar."""
    if uri is None:
        uri = VECTORSTORE_CONFIG["uri"]
    if table_name is None:
        table_name = VECTORSTORE_CONFIG["table_name"]

    print("Creating embeddings and storing in LanceDB...")

    print("Initializing HuggingFace embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_CONFIG["model"],
        model_kwargs={"trust_remote_code": True}
    )

    import lancedb as ldb
    db = ldb.connect(uri)
    existing = db.list_tables()
    if table_name in existing:
        db.drop_table(table_name)
        print(f"Dropped existing table '{table_name}'")

    from tqdm import tqdm
    total = len(chunks)
    print(f"Embedding {total} chunks...")
    pbar = tqdm(total=total, desc="Embedding", unit="chunk")

    class ProgressEmbeddings:
        """Wrapper that adds progress tracking to an embedding model."""

        def __init__(self, model, progress_bar):
            self._model = model
            self._pbar = progress_bar

        def embed_documents(self, texts):
            result = self._model.embed_documents(texts)
            self._pbar.update(len(texts))
            return result

        def embed_query(self, text):
            return self._model.embed_query(text)

    progress_embedding = ProgressEmbeddings(embedding_model, pbar)

    vectorstore = LanceDB.from_documents(
        documents=chunks,
        embedding=progress_embedding,
        uri=uri,
        table_name=table_name
    )
    pbar.close()

    tbl = db.open_table(table_name)
    row_count = tbl.count_rows()
    print(f"\nVector store created at {uri} (table: {table_name})")
    print(f"   Total chunks: {total} | Rows in DB: {row_count}")

    print("\n--- Cleaning up resources ---")
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return vectorstore


def enrich_metadata(chunks):
    """Enrich each chunk with law name and structure metadata."""
    print("Enriching chunks with metadata (law name, Chương, Mục, Điều, Khoản)...")
    for chunk in chunks:
        enrich_chunk_metadata(chunk)

    if chunks:
        sample = chunks[0]
        print(f"\n--- Sample Metadata ---")
        print(f"  law_name:     {sample.metadata.get('law_name', 'N/A')}")
        print(f"  chuong:       {sample.metadata.get('chuong', 'N/A')}")
        print(f"  muc:          {sample.metadata.get('muc', 'N/A')}")
        print(f"  dieu:         {sample.metadata.get('dieu', 'N/A')}")
        print(f"  khoan_start:  {sample.metadata.get('khoan_start', 'N/A')}")
        print(f"  khoan_end:    {sample.metadata.get('khoan_end', 'N/A')}")
        print(f"  chunk_level:  {sample.metadata.get('chunk_level', 'N/A')}")
        print("-" * 50)

    print(f"Enriched {len(chunks)} chunks with metadata.")
    return chunks


def main():
    documents = load_documents()
    chunks = split_documents(documents)
    chunks = enrich_metadata(chunks)
    create_vector_store(chunks)


if __name__ == "__main__":
    main()

