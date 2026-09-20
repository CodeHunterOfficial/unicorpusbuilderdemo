# documents/universal_doc_parser.py
"""
Universal document parser (v2, 2026-09).

Extracts plain text from PDF, DOCX, PPTX, XLSX, EPUB, FB2, HTML, MOBI,
images (via OCR) and nested archives. Metadata and rubric labels are
derived from file names and directory structure; the language of each
document is detected automatically.

Version 2 changes
-----------------
* PDF OCR fallback for scanned documents (pypdfium2 + PaddleOCR)
* Core-property metadata (title / author / created) for DOCX, PPTX, XLSX
* Optional parallel processing with ThreadPoolExecutor
* Retry with backoff for OCR failures
* Heuristic garbage-text detection
* Optional quality metrics per document
* Boilerplate pattern filtering
* --dry-run and --resume CLI modes
* Progress bar displays the current file name
* Thread-safe OCR engine cache
* File-hash logging for reproducibility

Installation
------------
    pip install PyMuPDF paddlepaddle paddleocr fast-langdetect trafilatura safezip
    pip install python-docx python-pptx openpyxl ebooklib rarfile py7zr
    pip install beautifulsoup4 Pillow tqdm PyYAML

Optional (PDF OCR for scanned documents):
    pip install pypdfium2
    # Also install the poppler utilities for your platform.

Usage
-----
    python universal_doc_parser.py <folder> [-c doc_config.yaml] [-o out.jsonl]
    python universal_doc_parser.py <folder> --dry-run
    python universal_doc_parser.py <folder> --resume
    python universal_doc_parser.py <folder> --workers 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger_setup import get_file_logger

logger = get_file_logger("doc_parser", "logs/doc_parser.log")


# ============================================================
# Optional dependency registry
# ============================================================

LIBRARIES: Dict[str, bool] = {}


def check_library(name: str, import_path: str) -> bool:
    """Register availability of an optional dependency."""
    try:
        __import__(import_path)
        LIBRARIES[name] = True
        return True
    except ImportError:
        LIBRARIES[name] = False
        return False


# Core format support
check_library("pdf",         "pymupdf")
check_library("docx",        "docx")
check_library("pptx",        "pptx")
check_library("xlsx",        "openpyxl")
check_library("epub",        "ebooklib")
check_library("bs4",         "bs4")
check_library("pillow",      "PIL")
check_library("rar",         "rarfile")
check_library("7z",          "py7zr")

# Extended support
check_library("paddleocr",   "paddleocr")
check_library("langdetect",  "fast_langdetect")
check_library("trafilatura", "trafilatura")
check_library("safezip",     "safezip")
check_library("docling",     "docling")
check_library("pypdfium2",   "pypdfium2")

# Conditional imports
if LIBRARIES["pdf"]:
    import pymupdf as fitz  # PyMuPDF: modern import path (fitz is deprecated)

if LIBRARIES["docx"]:
    from docx import Document
    from docx.table import Table as DocxTable

if LIBRARIES["pptx"]:
    from pptx import Presentation

if LIBRARIES["xlsx"]:
    import openpyxl

if LIBRARIES["epub"]:
    import ebooklib
    from ebooklib import epub

if LIBRARIES["bs4"]:
    from bs4 import BeautifulSoup

if LIBRARIES["pillow"]:
    from PIL import Image

if LIBRARIES["rar"]:
    import rarfile

if LIBRARIES["7z"]:
    import py7zr

if LIBRARIES["langdetect"]:
    from fast_langdetect import detect as ft_detect


# ============================================================
# OCR engine cache (thread-safe)
# ============================================================

_OCR_INSTANCES: Dict[str, Any] = {}
_OCR_LOCK = threading.Lock()


def get_ocr_engine(lang_code: str = 'en', use_gpu: bool = False):
    """
    Return a cached PaddleOCR engine for the given language.

    On Windows with PaddlePaddle 3.x, oneDNN instructions crash
    with 'ConvertPirAttribute2RuntimeAttribute not supported'.
    Passing enable_mkldnn=False to the constructor avoids this
    path without affecting recognition quality.
    """
    key = f"{lang_code}_{use_gpu}"
    with _OCR_LOCK:
        if key not in _OCR_INSTANCES:
            from paddleocr import PaddleOCR
            _OCR_INSTANCES[key] = PaddleOCR(
                lang=lang_code,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device='gpu' if use_gpu else 'cpu',
                enable_mkldnn=False,   # ← key fix for Windows
            )
            logger.info(f"Initialized PaddleOCR: lang={lang_code}, gpu={use_gpu}, mkldnn=False")
        return _OCR_INSTANCES[key]


# ============================================================
# Generic helpers
# ============================================================

def sha256_hex(text: str, trunc: Optional[int] = 32) -> str:
    """Return a hex digest of the input string, optionally truncated."""
    h = hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()
    return h[:trunc] if trunc else h


def file_content_hash(file_path: str, sample_bytes: int = 65536) -> str:
    """
    Return a short hash of the first `sample_bytes` of a file.

    Used to log a stable identifier for the document without
    reading the whole file into memory.
    """
    try:
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read(sample_bytes)).hexdigest()[:16]
    except OSError:
        return "0000000000000000"


def clean_text(text: str, config: dict) -> str:
    """
    Normalize whitespace and strip boilerplate patterns.

    The `boilerplate_patterns` list from the config is applied as
    multiline regular expressions before whitespace normalization.
    """
    if not text:
        return ""

    for pattern in config.get("boilerplate_patterns", []) or []:
        try:
            text = re.sub(pattern, "", text, flags=re.MULTILINE)
        except re.error as exc:
            logger.warning(f"Invalid boilerplate pattern {pattern!r}: {exc}")

    if config.get("strip_html", True):
        text = re.sub(r"<[^>]+>", "", text)

    if config.get("normalize_whitespace", True):
        text = re.sub(r"\s+", " ", text)

    return text.strip()


def is_garbage_text(text: str) -> bool:
    """
    Heuristic detection of non-linguistic output.

    Returns True when the extracted string looks like OCR noise or
    PDF extraction artifacts rather than natural language. Used to
    flag suspicious documents in the output.
    """
    if not text or len(text) < 80:
        return False

    # Fraction of alphanumeric or whitespace characters
    alnum = sum(c.isalnum() or c.isspace() for c in text)
    if alnum / len(text) < 0.5:
        return True

    # Repetition of unique words
    words = text.lower().split()
    if len(words) >= 200:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.05:
            return True

    return False


def load_config(config_path: str) -> dict:
    """Load a YAML configuration file. Missing files yield empty config."""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
            logger.info(f"Configuration loaded from {config_path}")
            return config
    logger.warning(f"Configuration file not found: {config_path}")
    return {}


def build_extension_map(config: dict) -> Dict[str, str]:
    """
    Flatten the `supported_extensions` section into a {ext: type} map.

    Example input:
        supported_extensions:
          documents: {extensions: [.pdf, .docx], type: pdf}
    Result: {".pdf": "pdf", ".docx": "pdf"}
    """
    ext_map: Dict[str, str] = {}
    for category, data in config.get("supported_extensions", {}).items():
        for ext in data.get("extensions", []):
            ext_map[ext.lower()] = data.get("type", category)
    return ext_map


def parse_filename_metadata(filename: str, config: dict) -> Dict[str, Optional[str]]:
    """
    Derive author and title from the file name.

    The configuration defines a list of separators (default: '_') and
    whether the author appears before or after the separator.
    """
    name = os.path.splitext(filename)[0]
    separators = config.get("filename_parsing", {}).get("separators", ["_"])
    author_first = config.get("filename_parsing", {}).get("author_first", True)

    for sep in separators:
        if sep in name:
            parts = name.split(sep, 1)
            if author_first:
                return {"author": parts[0].strip(), "title": parts[1].strip()}
            return {"author": parts[1].strip(), "title": parts[0].strip()}

    return {"author": None, "title": name}


def parse_rubric_from_path(file_path: str, root_dir: str, config: dict) -> Optional[str]:
    """
    Extract a rubric label from the directory structure.

    `depth` selects which directory level to treat as the rubric.
    `ignore_root` skips files located directly in the root directory.
    """
    depth = config.get("rubric_parsing", {}).get("depth", 0)
    ignore_root = config.get("rubric_parsing", {}).get("ignore_root", True)

    try:
        rel_path = os.path.relpath(os.path.dirname(file_path), root_dir)
        if rel_path == "." and ignore_root:
            return None
        parts = [p for p in rel_path.split(os.sep) if p]
        if len(parts) > depth:
            return parts[depth]
        return None
    except Exception:
        return None


# ============================================================
# Language detection
# ============================================================

def detect_language(text: str, config: dict) -> str:
    """
    Detect the ISO language code of `text`.

    Primary detector: fast-langdetect (FastText lid.176). When the
    detector is unavailable or fails, a script-based heuristic is
    used as a fallback.
    """
    if not config.get("language", {}).get("auto_detect", True):
        return config.get("language", {}).get("default", "unknown")

    if not text or len(text.strip()) < 20:
        return config.get("language", {}).get("default", "unknown")

    if LIBRARIES.get("langdetect"):
        try:
            # fast-langdetect requires input without newlines
            snippet = text[:500].replace("\n", " ").strip()
            result = ft_detect(snippet, model="auto")
            if isinstance(result, list) and result:
                lang = result[0].get("lang")
                if lang:
                    return lang
        except Exception as exc:
            logger.warning(f"fast-langdetect failed: {exc}")

    return _legacy_detect_language(text, config)


def _legacy_detect_language(text: str, config: dict) -> str:
    """Script-based fallback used only when fast-langdetect is absent."""
    cyrillic = len(re.findall(r"[а-яёәөүҗңһӣғқҳ]", text, re.I))
    latin = len(re.findall(r"[a-z]", text, re.I))
    arabic = len(re.findall(r"[\u0600-\u06FF]", text))

    if arabic > cyrillic and arabic > latin:
        return "arabic"
    if cyrillic > latin:
        return "cyrillic"
    if latin > 0:
        return "latin"
    return config.get("language", {}).get("default", "unknown")


# ============================================================
# Text extraction: basic formats
# ============================================================

def extract_text_txt(file_path: str, config: dict) -> Tuple[str, dict]:
    """Read a plain-text file trying the configured encodings in order."""
    encodings = config.get("text_reading", {}).get("encodings", ["utf-8", "cp1251"])
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.read(), {"encoding": enc, "extractor": "text"}
        except (UnicodeDecodeError, UnicodeError):
            continue

    logger.warning(f"Could not decode text file: {file_path}")
    return "", {"error": "encoding detection failed"}


def extract_text_pdf(file_path: str, config: dict) -> Tuple[str, dict]:
    """
    Extract text from a PDF using PyMuPDF.

    When the extraction yields too little content (a common symptom
    of scanned documents), and the OCR fallback is enabled in the
    configuration, `extract_text_pdf_ocr` is invoked.
    """
    if not LIBRARIES.get("pdf"):
        return "", {"error": "PyMuPDF not installed"}

    try:
        doc = fitz.open(file_path)
        pages: List[str] = []
        for page_num, page in enumerate(doc, 1):
            text = page.get_text("text")
            if text and text.strip():
                pages.append(f"--- Page {page_num} ---\n{text}")

        page_count = len(doc)
        metadata = doc.metadata or {}
        doc.close()

        stats = {
            "extractor": "pymupdf",
            "pages": page_count,
            "pdf_title": metadata.get("title"),
            "pdf_author": metadata.get("author"),
        }

        joined = "\n".join(pages)
        min_chars = config.get("pdf", {}).get("ocr_fallback_below_chars", 200)

        if (
            len(joined) < min_chars
            and page_count > 0
            and config.get("pdf", {}).get("use_ocr_fallback", True)
        ):
            logger.info(f"Low text yield, trying OCR fallback: {file_path}")
            ocr_text, ocr_stats = extract_text_pdf_ocr(file_path, config)
            if ocr_text and len(ocr_text) > len(joined):
                return ocr_text, ocr_stats

        return joined, stats

    except Exception as exc:
        logger.error(f"PyMuPDF failed for {file_path}: {exc}")

        if config.get("pdf", {}).get("use_docling_fallback") and LIBRARIES.get("docling"):
            logger.info(f"Trying Docling fallback for {file_path}")
            text, stats = extract_text_docling(file_path, config)
            if text:
                return text, stats

        return "", {"error": str(exc)}


def extract_text_pdf_ocr(file_path: str, config: dict) -> Tuple[str, dict]:
    """
    Render PDF pages to images and run PaddleOCR on each page.

    Uses pypdfium2 (Chrome's PDF engine) for rendering, so no
    external binaries such as poppler are required. This path is
    much slower than direct text extraction and is intended only
    for scanned documents without a text layer.
    """
    if not LIBRARIES.get("pypdfium2"):
        return "", {"error": "pypdfium2 not installed"}

    try:
        import pypdfium2 as pdfium
    except ImportError:
        return "", {"error": "pypdfium2 import failed"}

    pdf_cfg = config.get("pdf", {})
    ocr_cfg = config.get("ocr", {})

    dpi = pdf_cfg.get("ocr_dpi", 200)
    lang_code = ocr_cfg.get("paddle_lang", "ru")
    use_gpu = ocr_cfg.get("use_gpu", False)
    min_conf = ocr_cfg.get("min_confidence", 0.5)

    try:
        pdf = pdfium.PdfDocument(file_path)
    except Exception as exc:
        logger.error(f"pypdfium2 could not open {file_path}: {exc}")
        return "", {"error": f"pypdfium2 open: {exc}"}

    ocr = get_ocr_engine(lang_code, use_gpu)
    collected: List[str] = []
    total_scores: List[float] = []

    # PDF native resolution is 72 DPI; scale factor maps to target DPI
    scale = dpi / 72.0

    for page_index in range(len(pdf)):
        try:
            page = pdf[page_index]
            pil_image = page.render(scale=scale).to_pil()
            tmp_path = os.path.join(
                tempfile.gettempdir(),
                f"_pdf_ocr_{os.getpid()}_{page_index}.png",
            )
            try:
                pil_image.save(tmp_path)
                result = ocr.predict(tmp_path)
                for page_result in result:
                    texts = page_result.get("rec_texts", [])
                    scores = page_result.get("rec_scores", [])
                    for text, score in zip(texts, scores):
                        if score >= min_conf and text and text.strip():
                            collected.append(text.strip())
                            total_scores.append(float(score))
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        except Exception as exc:
            logger.warning(f"Page {page_index} OCR failed in {file_path}: {exc}")
            continue

    try:
        pdf.close()
    except Exception:
        pass

    avg_conf = round(sum(total_scores) / len(total_scores), 3) if total_scores else 0.0

    return "\n".join(collected), {
        "extractor": "paddleocr_pdf",
        "pages": len(pdf),
        "dpi": dpi,
        "lang": lang_code,
        "blocks": len(collected),
        "avg_confidence": avg_conf,
    }

def extract_text_docx(file_path: str, config: dict) -> Tuple[str, dict]:
    """Extract paragraphs, tables and core properties from a DOCX file."""
    if not LIBRARIES.get("docx"):
        return "", {"error": "python-docx not installed"}

    try:
        doc = Document(file_path)
        parts: List[str] = []

        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                parts.append(paragraph.text)

        for table in doc.tables:
            _extract_docx_table(table, parts)

        core = doc.core_properties
        stats = {
            "extractor": "python-docx",
            "paragraphs": len(parts),
            "doc_title": core.title,
            "doc_author": core.author,
            "doc_created": core.created.isoformat() if core.created else None,
            "doc_modified": core.modified.isoformat() if core.modified else None,
        }
        return "\n".join(parts), stats

    except Exception as exc:
        logger.error(f"DOCX extraction failed for {file_path}: {exc}")
        return "", {"error": str(exc)}


def _extract_docx_table(table: DocxTable, parts: List[str], depth: int = 0) -> None:
    """Append table rows (and nested tables) to `parts` in reading order."""
    indent = "  " * depth
    for row in table.rows:
        row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
        if row_text:
            parts.append(f"{indent}[TABLE] {row_text}")
        for cell in row.cells:
            for nested in cell.tables:
                _extract_docx_table(nested, parts, depth + 1)


def extract_text_pptx(file_path: str, config: dict) -> Tuple[str, dict]:
    """Extract slide text, tables and core properties from a PPTX file."""
    if not LIBRARIES.get("pptx"):
        return "", {"error": "python-pptx not installed"}

    try:
        prs = Presentation(file_path)
        slides: List[str] = []

        for slide_num, slide in enumerate(prs.slides, 1):
            slide_lines: List[str] = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_lines.append(shape.text)
                if shape.has_table:
                    for row in shape.table.rows:
                        row_text = " | ".join(
                            cell.text.strip() for cell in row.cells if cell.text.strip()
                        )
                        if row_text:
                            slide_lines.append(f"[TABLE] {row_text}")

            if slide_lines:
                slides.append(f"=== Slide {slide_num} ===\n" + "\n".join(slide_lines))

        core = prs.core_properties
        stats = {
            "extractor": "python-pptx",
            "slides": len(prs.slides),
            "doc_title": core.title,
            "doc_author": core.author,
            "doc_created": core.created.isoformat() if core.created else None,
        }
        return "\n\n".join(slides), stats

    except Exception as exc:
        logger.error(f"PPTX extraction failed for {file_path}: {exc}")
        return "", {"error": str(exc)}


def extract_text_xlsx(file_path: str, config: dict) -> Tuple[str, dict]:
    """Extract cell values and core properties from an XLSX workbook."""
    if not LIBRARIES.get("xlsx"):
        return "", {"error": "openpyxl not installed"}

    try:
        wb = openpyxl.load_workbook(file_path, data_only=True, read_only=True)
        sheets: List[str] = []

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows: List[str] = []
            for row in ws.iter_rows(values_only=True):
                values = [str(c) for c in row if c is not None]
                if values:
                    rows.append("\t".join(values))
            if rows:
                sheets.append(f"=== {sheet_name} ===\n" + "\n".join(rows))

        props = wb.properties
        stats = {
            "extractor": "openpyxl",
            "sheets": len(wb.sheetnames),
            "doc_title": getattr(props, "title", None),
            "doc_author": getattr(props, "creator", None),
        }
        wb.close()
        return "\n\n".join(sheets), stats

    except Exception as exc:
        logger.error(f"XLSX extraction failed for {file_path}: {exc}")
        return "", {"error": str(exc)}


# ============================================================
# Text extraction: images (OCR)
# ============================================================

def extract_text_image(file_path: str, config: dict, retries: int = 2) -> Tuple[str, dict]:
    """
    Run PaddleOCR on an image file.

    On transient failures the OCR call is retried up to `retries`
    additional times with linear backoff.
    """
    if not LIBRARIES.get("paddleocr"):
        return "", {"error": "PaddleOCR not installed"}

    ocr_cfg = config.get("ocr", {})
    min_w = ocr_cfg.get("min_width", 100)
    min_h = ocr_cfg.get("min_height", 100)
    min_conf = ocr_cfg.get("min_confidence", 0.5)
    lang_code = ocr_cfg.get("paddle_lang", "en")
    use_gpu = ocr_cfg.get("use_gpu", False)

    size_str = "unknown"
    if LIBRARIES.get("pillow"):
        try:
            with Image.open(file_path) as img:
                if img.width < min_w or img.height < min_h:
                    return "", {"error": "image too small"}
                size_str = f"{img.width}x{img.height}"
        except Exception as exc:
            logger.warning(f"Pillow could not read {file_path}: {exc}")

    ocr = get_ocr_engine(lang_code, use_gpu)

    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            result = ocr.predict(file_path)
            break
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
            else:
                logger.error(f"PaddleOCR failed for {file_path}: {exc}")
                return "", {"error": f"OCR failed after {retries + 1} attempts: {exc}"}

    texts: List[str] = []
    scores: List[float] = []
    for page_result in result:
        rec_texts = page_result.get("rec_texts", [])
        rec_scores = page_result.get("rec_scores", [])
        for text, score in zip(rec_texts, rec_scores):
            if score >= min_conf and text and text.strip():
                texts.append(text.strip())
                scores.append(float(score))

    avg_conf = round(sum(scores) / len(scores), 3) if scores else 0.0

    return "\n".join(texts), {
        "extractor": "paddleocr",
        "lang": lang_code,
        "size": size_str,
        "blocks": len(texts),
        "avg_confidence": avg_conf,
    }


# ============================================================
# Text extraction: ebooks
# ============================================================

def extract_text_epub(file_path: str, config: dict) -> Tuple[str, dict]:
    """Extract readable content from an EPUB file."""
    if not LIBRARIES.get("epub") or not LIBRARIES.get("bs4"):
        return "", {"error": "ebooklib/bs4 not installed"}

    try:
        book = epub.read_epub(file_path)
        chapters: List[str] = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), "html.parser")
                text = soup.get_text()
                if text.strip():
                    chapters.append(text)
        return "\n\n".join(chapters), {
            "extractor": "ebooklib",
            "chapters": len(chapters),
        }
    except Exception as exc:
        logger.error(f"EPUB extraction failed for {file_path}: {exc}")
        return "", {"error": str(exc)}


def extract_text_fb2(file_path: str, config: dict) -> Tuple[str, dict]:
    """Extract paragraph text from an FB2 book."""
    if not LIBRARIES.get("bs4"):
        return "", {"error": "BeautifulSoup not installed"}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        soup = BeautifulSoup(content, "xml")
        bodies = soup.find_all("body")
        sections = [
            p.get_text(strip=True)
            for body in bodies
            for p in body.find_all("p")
            if p.get_text(strip=True)
        ]
        return "\n\n".join(sections), {
            "extractor": "fb2",
            "sections": len(sections),
        }
    except Exception as exc:
        logger.error(f"FB2 extraction failed for {file_path}: {exc}")
        return "", {"error": str(exc)}


def extract_text_mobi(file_path: str, config: dict) -> Tuple[str, dict]:
    """Best-effort extraction from a MOBI/AZW file."""
    if LIBRARIES.get("epub"):
        try:
            return extract_text_epub(file_path, config)
        except Exception:
            pass

    try:
        with open(file_path, "rb") as f:
            content = f.read()
        try:
            text = content.decode("utf-8")
            return clean_text(text, {}), {"extractor": "mobi_text"}
        except UnicodeDecodeError:
            text = "".join(chr(b) for b in content if 32 <= b < 127 or b in (10, 13))
            return text, {"extractor": "mobi_binary"}
    except Exception as exc:
        logger.error(f"MOBI extraction failed for {file_path}: {exc}")
        return "", {"error": str(exc)}


# ============================================================
# Text extraction: HTML (Trafilatura with fallback)
# ============================================================

def extract_text_html(file_path: str, config: dict) -> Tuple[str, dict]:
    """
    Extract body text from an HTML file.

    Primary extractor: Trafilatura, which removes navigation and
    boilerplate. Fallback: BeautifulSoup-based selector.
    """
    if LIBRARIES.get("trafilatura"):
        try:
            import trafilatura
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            text = trafilatura.extract(
                content,
                output_format="txt",
                include_comments=False,
                include_tables=True,
            )
            if text and text.strip():
                return text, {"extractor": "trafilatura"}
        except Exception as exc:
            logger.warning(f"Trafilatura failed for {file_path}: {exc}")

    return _legacy_extract_text_html(file_path, config)


def _legacy_extract_text_html(file_path: str, config: dict) -> Tuple[str, dict]:
    """Minimal HTML text extraction used when Trafilatura is unavailable."""
    if not LIBRARIES.get("bs4"):
        return "", {"error": "BeautifulSoup not installed"}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        soup = BeautifulSoup(content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        lines = [line.strip() for line in soup.get_text(separator="\n").splitlines() if line.strip()]
        return "\n".join(lines), {
            "extractor": "bs4",
            "html_title": soup.title.string if soup.title else None,
        }
    except Exception as exc:
        logger.error(f"HTML extraction failed for {file_path}: {exc}")
        return "", {"error": str(exc)}


# ============================================================
# Optional Docling extractor
# ============================================================

def extract_text_docling(file_path: str, config: dict) -> Tuple[str, dict]:
    """
    Use Docling for layout-aware extraction (tables, headings, reading order).

    Returns Markdown. Requires ~3-5 GB of dependencies; kept optional.
    """
    if not LIBRARIES.get("docling"):
        return "", {"error": "Docling not installed"}

    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(file_path)
        return result.document.export_to_markdown(), {
            "extractor": "docling",
            "format": "markdown",
        }
    except Exception as exc:
        logger.error(f"Docling failed for {file_path}: {exc}")
        return "", {"error": str(exc)}


# ============================================================
# Archives: safe extraction
# ============================================================

def extract_files_from_archive(
    archive_path: str,
    config: dict,
    max_depth: int = 3,
) -> Tuple[List[str], str]:
    """
    Extract archive contents into a fresh temporary directory.

    Protections:
      * ZipSlip (path traversal) — safezip or manual verification
      * ZIP bombs — limits on file count and total size
      * tarfile filter="data" on Python 3.12+

    Returns a tuple (extracted_files, temp_dir). The caller is
    responsible for deleting `temp_dir`.
    """
    extracted_files: List[str] = []
    ext = os.path.splitext(archive_path)[1].lower()
    temp_dir = tempfile.mkdtemp(prefix="_extracted_")

    proc_cfg = config.get("processing", {})
    max_total = proc_cfg.get("max_archive_total_mb", 500) * 1024 * 1024
    max_files = proc_cfg.get("max_archive_files", 1000)

    try:
        if ext == ".zip":
            _safe_extract_zip(archive_path, temp_dir, max_total, max_files)
        elif ext == ".rar" and LIBRARIES.get("rar"):
            with rarfile.RarFile(archive_path, "r") as rf:
                rf.extractall(temp_dir)
        elif ext == ".7z" and LIBRARIES.get("7z"):
            with py7zr.SevenZipFile(archive_path, "r") as szf:
                szf.extractall(temp_dir)
        elif ext in (".tar", ".gz", ".bz2"):
            import tarfile
            try:
                with tarfile.open(archive_path, "r:*") as tf:
                    tf.extractall(temp_dir, filter="data")
            except TypeError:
                # Python < 3.12: no filter argument available
                with tarfile.open(archive_path, "r:*") as tf:
                    tf.extractall(temp_dir)
        else:
            logger.warning(f"Unsupported archive format: {ext}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return [], ""

        for root, _dirs, files in os.walk(temp_dir):
            for f in files:
                extracted_files.append(os.path.join(root, f))

        if max_depth > 1:
            nested: List[str] = []
            for f in extracted_files[:]:
                if os.path.splitext(f)[1].lower() in (".zip", ".rar", ".7z", ".tar", ".gz", ".bz2"):
                    nested.append(f)
                    extracted_files.remove(f)
            for nf in nested:
                try:
                    nested_files, _ = extract_files_from_archive(nf, config, max_depth - 1)
                    extracted_files.extend(nested_files)
                except Exception as exc:
                    logger.error(f"Nested archive extraction failed {nf}: {exc}")

        return extracted_files, temp_dir

    except Exception as exc:
        logger.error(f"Archive extraction failed for {archive_path}: {exc}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return [], ""


def _safe_extract_zip(
    archive_path: str,
    temp_dir: str,
    max_total: int,
    max_files: int,
) -> None:
    """Safely extract a ZIP archive using safezip when available."""
    if LIBRARIES.get("safezip"):
        try:
            from safezip import SafeZipFile
            with SafeZipFile(
                archive_path,
                max_file_size=100 * 1024 * 1024,
                max_total_size=max_total,
                max_files=max_files,
            ) as zf:
                zf.extractall(temp_dir)
            return
        except Exception as exc:
            logger.warning(f"safezip failed, using fallback: {exc}")

    import zipfile
    with zipfile.ZipFile(archive_path, "r") as zf:
        real_temp = os.path.realpath(temp_dir)
        total_size = 0
        count = 0
        for member in zf.infolist():
            member_path = os.path.realpath(os.path.join(temp_dir, member.filename))
            if not member_path.startswith(real_temp):
                logger.warning(f"ZipSlip blocked: {member.filename}")
                continue
            if count >= max_files:
                logger.warning(f"ZIP file count limit reached: {max_files}")
                break
            total_size += member.file_size
            if total_size > max_total:
                logger.warning(f"ZIP total size limit reached: {max_total}")
                break
            zf.extract(member, temp_dir)
            count += 1


# ============================================================
# Extractor registry
# ============================================================

EXTRACTORS = {
    "text":    extract_text_txt,
    "pdf":     extract_text_pdf,
    "docx":    extract_text_docx,
    "pptx":    extract_text_pptx,
    "xlsx":    extract_text_xlsx,
    "image":   extract_text_image,
    "ebook":   extract_text_epub,
    "docling": extract_text_docling,
}

SPECIAL_HANDLERS = {
    ".html": extract_text_html,
    ".htm":  extract_text_html,
    ".fb2":  extract_text_fb2,
    ".mobi": extract_text_mobi,
    ".azw":  extract_text_mobi,
    ".azw3": extract_text_mobi,
}


# ============================================================
# Per-file processing (parallel-safe)
# ============================================================

def process_single_file(
    file_path: str,
    config: dict,
    ext_map: Dict[str, str],
    root_dir: str,
    max_size_mb: float,
    min_length: int,
    skip_empty: bool,
    collect_quality: bool,
) -> Optional[Dict[str, Any]]:
    """
    Extract one file and return a JSONL-ready record.

    Returns None when the file is skipped (size, emptiness, or an
    extraction error). All exceptions are caught and logged so that
    parallel workers never propagate them to the pool.
    """
    try:
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            return None

        ext = os.path.splitext(file_path)[1].lower()
        file_type = ext_map.get(ext, "text")
        filename = os.path.basename(file_path)

        if ext in SPECIAL_HANDLERS:
            text, ext_stats = SPECIAL_HANDLERS[ext](file_path, config)
        else:
            extractor = EXTRACTORS.get(file_type, extract_text_txt)
            text, ext_stats = extractor(file_path, config)

        text = clean_text(text, config.get("text_cleaning", {}))

        if skip_empty and (not text or len(text) < min_length):
            logger.info(f"Skipping (empty): {filename} [{len(text)} chars]")
            return None

        meta = parse_filename_metadata(filename, config)
        rubric = parse_rubric_from_path(file_path, root_dir, config)
        language = detect_language(text, config)

        item: Dict[str, Any] = {
            "url": file_path,
            "title": meta.get("title") or filename,
            "content": text,
            "excerpt": text[:260] + "..." if len(text) > 260 else text,
            "date": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
            "author": meta.get("author") or config.get("defaults", {}).get("author"),
            "category": rubric or config.get("defaults", {}).get("category"),
            "site": config.get("defaults", {}).get("site", "local_documents"),
            "hash": sha256_hex(text),
            "language": language or config.get("defaults", {}).get("language"),
            "source_type": config.get("defaults", {}).get("source_type", "document"),
            "file_name": filename,
            "file_type": file_type,
            "folder": rubric,
            "text_length": len(text),
            "file_size_kb": round(os.path.getsize(file_path) / 1024, 1),
            "extraction_stats": ext_stats,
            "scraped_at": datetime.now().isoformat(),
        }

        if collect_quality:
            words = text.split()
            item["quality"] = {
                "chars_per_page": _chars_per_page(len(text), ext_stats),
                "words_count": len(words),
                "unique_words_ratio": round(len(set(w.lower() for w in words)) / max(len(words), 1), 3),
                "has_garbage": is_garbage_text(text),
                "extraction_confidence": _confidence_level(ext_stats, text),
            }

        return item

    except Exception as exc:
        logger.error(f"Error processing {file_path}: {exc}")
        return None


def _chars_per_page(text_len: int, ext_stats: dict) -> Optional[float]:
    """Compute characters-per-page when the page count is available."""
    pages = (ext_stats or {}).get("pages")
    if isinstance(pages, int) and pages > 0:
        return round(text_len / pages, 1)
    return None


def _confidence_level(ext_stats: dict, text: str) -> str:
    """
    Assign a coarse confidence label to an extraction result.

    Uses OCR confidence when available and falls back to text length.
    """
    if not ext_stats:
        return "low"
    if "error" in ext_stats:
        return "low"
    avg = ext_stats.get("avg_confidence")
    if isinstance(avg, (int, float)):
        if avg >= 0.85:
            return "high"
        if avg >= 0.6:
            return "medium"
        return "low"
    return "high" if len(text) >= 500 else "medium"


# ============================================================
# Main parser
# ============================================================

def parse_documents(
    root_dir: str,
    config: dict,
    output_file: Optional[str] = None,
    dry_run: bool = False,
    resume: bool = False,
    workers: int = 1,
) -> Dict[str, Any]:
    """
    Parse every supported file under `root_dir`.

    Arguments:
        root_dir:    directory to scan
        config:      parsed YAML configuration
        output_file: destination JSONL path (auto-generated when None)
        dry_run:     list files without extracting
        resume:      skip files already present in `output_file`
        workers:     number of parallel workers (1 = sequential)
    """
    logger.info(f"Starting document parsing in '{root_dir}'")
    ext_map = build_extension_map(config)
    processing = config.get("processing", {})
    output_cfg = config.get("output", {})

    max_size = processing.get("max_file_size_mb", 100)
    min_length = processing.get("min_text_length", 100)
    skip_empty = processing.get("skip_empty", True)
    collect_quality = processing.get("collect_quality", True)

    # Resolve output path
    if not output_file:
        output_dir = output_cfg.get("directory", "output/documents")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_cfg.get("filename_template", "documents_{timestamp}.jsonl").format(
            timestamp=timestamp
        )
        output_file = os.path.join(output_dir, filename)
    else:
        output_path = Path(output_file)
        if output_path.suffix:
            os.makedirs(output_path.parent, exist_ok=True)
        else:
            os.makedirs(output_path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = output_cfg.get("filename_template", "documents_{timestamp}.jsonl").format(
                timestamp=timestamp
            )
            output_file = os.path.join(str(output_path), filename)

    logger.info(f"Output file: {output_file}")
    print(f"[OUTPUT] Saving to: {output_file}")

    # Collect files
    all_files: List[str] = []
    archive_files: List[str] = []
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            if f.startswith("."):
                continue
            ext = os.path.splitext(f)[1].lower()
            if ext in ext_map:
                file_path = os.path.join(root, f)
                if ext_map[ext] == "archive":
                    archive_files.append(file_path)
                else:
                    all_files.append(file_path)

    # Recursive archive extraction
    temp_dirs_to_clean: List[str] = []
    if processing.get("recursive_archives", True) and archive_files:
        max_depth = processing.get("max_archive_depth", 3)
        for archive_path in archive_files:
            try:
                extracted, temp_dir = extract_files_from_archive(archive_path, config, max_depth)
                all_files.extend(extracted)
                if temp_dir:
                    temp_dirs_to_clean.append(temp_dir)
                logger.info(f"Extracted {len(extracted)} files from archive: {archive_path}")
            except Exception as exc:
                logger.error(f"Failed to extract archive {archive_path}: {exc}")

    # Resume mode: skip files already in the output
    if resume and os.path.exists(output_file):
        processed_urls = set()
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            processed_urls.add(json.loads(line).get("url"))
                        except json.JSONDecodeError:
                            continue
            before = len(all_files)
            all_files = [fp for fp in all_files if fp not in processed_urls]
            logger.info(f"Resume mode: {before - len(all_files)} files skipped, {len(all_files)} to process")
        except OSError as exc:
            logger.warning(f"Resume: could not read existing output: {exc}")

    logger.info(f"Found {len(all_files)} supported files")
    print(f"[FILES] Files found: {len(all_files)}")

    if dry_run:
        print("\n[DRY-RUN] Files that would be processed:")
        for fp in all_files:
            print(f"  {fp}")
        return {"stats": {"total": len(all_files)}, "output_file": output_file, "results_count": 0}

    # Processing
    results: List[Dict[str, Any]] = []
    stats = {
        "total": len(all_files),
        "processed": 0,
        "skipped_empty": 0,
        "skipped_size": 0,
        "skipped_error": 0,
        "by_type": {},
        "by_extractor": {},
    }

    try:
        if workers > 1:
            logger.info(f"Running in parallel mode with {workers} workers")
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        process_single_file,
                        fp, config, ext_map, root_dir, max_size, min_length,
                        skip_empty, collect_quality,
                    ): fp
                    for fp in all_files
                }
                for future in tqdm(
                    as_completed(futures), total=len(futures),
                    desc="[PROCESS]", unit="file",
                ):
                    fp = futures[future]
                    try:
                        item = future.result()
                    except Exception as exc:
                        logger.error(f"Worker failed for {fp}: {exc}")
                        stats["skipped_error"] += 1
                        continue
                    if item is None:
                        stats["skipped_empty"] += 1
                        continue
                    results.append(item)
                    stats["processed"] += 1
                    _bump_stats(stats, item)
        else:
            pbar = tqdm(all_files, desc="[PROCESS]", unit="file")
            for fp in pbar:
                pbar.set_postfix_str(os.path.basename(fp)[:40])
                item = process_single_file(
                    fp, config, ext_map, root_dir, max_size, min_length,
                    skip_empty, collect_quality,
                )
                if item is None:
                    stats["skipped_empty"] += 1
                    continue
                results.append(item)
                stats["processed"] += 1
                _bump_stats(stats, item)

    finally:
        for td in temp_dirs_to_clean:
            shutil.rmtree(td, ignore_errors=True)

    # Persist results
    if results:
        os.makedirs(
            os.path.dirname(output_file) if os.path.dirname(output_file) else ".",
            exist_ok=True,
        )
        mode = "a" if resume else "w"
        with open(output_file, mode, encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        if output_cfg.get("save_all", True) and not resume:
            all_file = output_file.replace(".jsonl", "_all.jsonl")
            with open(all_file, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        _print_summary(stats, output_file)
        logger.info(
            f"Parsing completed. Processed: {stats['processed']}, "
            f"skipped empty: {stats['skipped_empty']}, errors: {stats['skipped_error']}"
        )
    else:
        logger.warning("No documents with text found after processing")
        print("[WARN] No documents with text found")

    return {
        "stats": stats,
        "output_file": output_file,
        "results_count": len(results),
    }


def _bump_stats(stats: dict, item: dict) -> None:
    """Update type and extractor counters for a successfully processed item."""
    ftype = item.get("file_type", "unknown")
    stats["by_type"][ftype] = stats["by_type"].get(ftype, 0) + 1

    extr = (item.get("extraction_stats") or {}).get("extractor", "unknown")
    stats["by_extractor"][extr] = stats["by_extractor"].get(extr, 0) + 1


def _print_summary(stats: dict, output_file: str) -> None:
    """Print a human-readable summary of the parsing run."""
    print(f"\n{'=' * 60}")
    print("[OK] Parsing completed!")
    print(f"  Processed:     {stats['processed']}")
    print(f"  Skipped empty: {stats['skipped_empty']}")
    print(f"  Skipped large: {stats['skipped_size']}")
    print(f"  Errors:        {stats['skipped_error']}")
    print(f"  Total files:   {stats['total']}")

    if stats["by_type"]:
        print("\n  By type:")
        for ftype, count in sorted(stats["by_type"].items()):
            print(f"    {ftype}: {count}")

    if stats["by_extractor"]:
        print("\n  By extractor:")
        for extr, count in sorted(stats["by_extractor"].items()):
            print(f"    {extr}: {count}")

    print(f"\n[SAVED] {output_file}")
    print(f"{'=' * 60}\n")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal document parser (v2, 2026)")
    parser.add_argument("folder", help="Folder with documents")
    parser.add_argument("-c", "--config", default="doc_config.yaml", help="Path to config")
    parser.add_argument("-o", "--output", default=None, help="Output JSONL file")
    parser.add_argument("--dry-run", action="store_true", help="List files without extracting")
    parser.add_argument("--resume", action="store_true", help="Skip files already in output")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (default: 1)")
    args = parser.parse_args()

    logger.info(
        f"CLI started: folder={args.folder}, config={args.config}, "
        f"output={args.output}, workers={args.workers}, resume={args.resume}"
    )

    # Dependency diagnostics
    print("[LIBS] Detected:")
    for name, ok in sorted(LIBRARIES.items()):
        mark = "OK " if ok else "-- "
        print(f"  {mark}{name}")

    config = load_config(args.config)
    parse_documents(
        root_dir=args.folder,
        config=config,
        output_file=args.output,
        dry_run=args.dry_run,
        resume=args.resume,
        workers=args.workers,
    )