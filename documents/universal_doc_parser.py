# documents/universal_doc_parser.py
"""
Универсальный парсер документов с поддержкой YAML-конфигурации.
Использование:
  python universal_doc_parser.py <папка> [-c doc_config.yaml] [-o output.jsonl]
"""

import os
import sys
import json
import hashlib
import re
import shutil
import tempfile
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from io import BytesIO, StringIO

import yaml
from tqdm import tqdm

# =====================================================
# Проверка библиотек
# =====================================================

LIBRARIES = {}

def check_library(name: str, import_path: str) -> bool:
    try:
        __import__(import_path)
        LIBRARIES[name] = True
        return True
    except ImportError:
        LIBRARIES[name] = False
        return False

check_library('pdf', 'PyPDF2')
check_library('docx', 'docx')
check_library('ocr', 'pytesseract')
check_library('pptx', 'pptx')
check_library('xlsx', 'openpyxl')
check_library('epub', 'ebooklib')
check_library('rar', 'rarfile')
check_library('7z', 'py7zr')
check_library('bs4', 'bs4')
check_library('pillow', 'PIL')

# Импортируем после проверки
if LIBRARIES['pdf']:
    from PyPDF2 import PdfReader
if LIBRARIES['docx']:
    from docx import Document
if LIBRARIES['ocr']:
    import pytesseract
    from PIL import Image
if LIBRARIES['pptx']:
    from pptx import Presentation
if LIBRARIES['xlsx']:
    import openpyxl
if LIBRARIES['epub']:
    from ebooklib import epub
    import ebooklib
if LIBRARIES['bs4']:
    from bs4 import BeautifulSoup
if LIBRARIES['rar']:
    import rarfile
if LIBRARIES['7z']:
    import py7zr
if LIBRARIES['pillow']:
    from PIL import Image


# =====================================================
# Утилиты
# =====================================================

def sha256_hex(text: str, trunc: int = 32) -> str:
    h = hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()
    return h[:trunc] if trunc else h


def clean_text(text: str, config: dict) -> str:
    if not text:
        return ""
    if config.get('strip_html', True):
        text = re.sub(r'<[^>]+>', '', text)
    if config.get('normalize_whitespace', True):
        text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_config(config_path: str) -> dict:
    """Загружает конфиг из YAML."""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}


def build_extension_map(config: dict) -> Dict[str, str]:
    """Строит карту расширений из конфига."""
    ext_map = {}
    for category, data in config.get('supported_extensions', {}).items():
        for ext in data.get('extensions', []):
            ext_map[ext.lower()] = data.get('type', category)
    return ext_map


def parse_filename_metadata(filename: str, config: dict) -> Dict[str, Optional[str]]:
    """Извлекает автора и название по правилам из конфига."""
    name = os.path.splitext(filename)[0]
    separators = config.get('filename_parsing', {}).get('separators', ['_'])
    
    for sep in separators:
        if sep in name:
            parts = name.split(sep, 1)
            if config.get('filename_parsing', {}).get('author_first', True):
                return {"author": parts[0].strip(), "title": parts[1].strip()}
            else:
                return {"author": parts[1].strip(), "title": parts[0].strip()}
    
    return {"author": None, "title": name}


def parse_rubric_from_path(file_path: str, root_dir: str, config: dict) -> Optional[str]:
    """Извлекает рубрику из структуры папок."""
    depth = config.get('rubric_parsing', {}).get('depth', 0)
    ignore_root = config.get('rubric_parsing', {}).get('ignore_root', True)
    
    try:
        rel_path = os.path.relpath(os.path.dirname(file_path), root_dir)
        if rel_path == '.' and ignore_root:
            return None
        parts = [p for p in rel_path.split(os.sep) if p]
        if len(parts) > depth:
            return parts[depth]
        return None
    except:
        return None


def detect_language(text: str, config: dict) -> str:
    if not config.get('language', {}).get('auto_detect', True):
        return config.get('language', {}).get('default', 'unknown')
    
    if not text:
        return config.get('language', {}).get('default', 'unknown')
    
    cyrillic = len(re.findall(r'[а-яёәөүҗңһӣғқҳ]', text, re.I))
    latin = len(re.findall(r'[a-z]', text, re.I))
    arabic = len(re.findall(r'[\u0600-\u06FF]', text))
    
    if arabic > cyrillic and arabic > latin:
        return "arabic"
    elif cyrillic > latin:
        return "cyrillic"
    elif latin > 0:
        return "latin"
    return config.get('language', {}).get('default', 'unknown')


# =====================================================
# Экстракторы
# =====================================================

def extract_text_txt(file_path: str, config: dict) -> Tuple[str, dict]:
    encodings = config.get('text_reading', {}).get('encodings', ['utf-8'])
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                return f.read(), {'encoding': enc}
        except (UnicodeDecodeError, UnicodeError):
            continue
    return "", {'error': 'encoding detection failed'}


def extract_text_pdf(file_path: str, config: dict) -> Tuple[str, dict]:
    if not LIBRARIES.get('pdf'):
        return "", {'error': 'PyPDF2 not installed'}
    try:
        reader = PdfReader(file_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return '\n'.join(pages), {'pages': len(reader.pages)}
    except Exception as e:
        return "", {'error': str(e)}


def extract_text_docx(file_path: str, config: dict) -> Tuple[str, dict]:
    if not LIBRARIES.get('docx'):
        return "", {'error': 'python-docx not installed'}
    try:
        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                row_text = ' | '.join(cell.text for cell in row.cells if cell.text)
                if row_text.strip():
                    paragraphs.append(row_text)
        return '\n'.join(paragraphs), {'paragraphs': len(paragraphs)}
    except Exception as e:
        return "", {'error': str(e)}


def extract_text_pptx(file_path: str, config: dict) -> Tuple[str, dict]:
    if not LIBRARIES.get('pptx'):
        return "", {'error': 'python-pptx not installed'}
    try:
        prs = Presentation(file_path)
        slides = []
        for slide_num, slide in enumerate(prs.slides, 1):
            slide_text = [shape.text for shape in slide.shapes if hasattr(shape, 'text') and shape.text.strip()]
            if slide_text:
                slides.append(f"=== Слайд {slide_num} ===\n" + '\n'.join(slide_text))
        return '\n\n'.join(slides), {'slides': len(prs.slides)}
    except Exception as e:
        return "", {'error': str(e)}


def extract_text_xlsx(file_path: str, config: dict) -> Tuple[str, dict]:
    if not LIBRARIES.get('xlsx'):
        return "", {'error': 'openpyxl not installed'}
    try:
        wb = openpyxl.load_workbook(file_path, data_only=True)
        sheets = []
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows = []
            for row in ws.iter_rows(values_only=True):
                values = [str(c) for c in row if c is not None]
                if values:
                    rows.append('\t'.join(values))
            if rows:
                sheets.append(f"=== {sheet_name} ===\n" + '\n'.join(rows))
        return '\n\n'.join(sheets), {'sheets': len(wb.sheetnames)}
    except Exception as e:
        return "", {'error': str(e)}


def extract_text_image(file_path: str, config: dict) -> Tuple[str, dict]:
    if not LIBRARIES.get('ocr') or not LIBRARIES.get('pillow'):
        return "", {'error': 'pytesseract/Pillow not installed'}
    try:
        img = Image.open(file_path)
        if img.width < config.get('ocr', {}).get('min_width', 100) or \
           img.height < config.get('ocr', {}).get('min_height', 100):
            return "", {'error': 'image too small'}
        languages = config.get('ocr', {}).get('languages', 'rus+eng')
        text = pytesseract.image_to_string(img, lang=languages)
        return text, {'size': f"{img.width}x{img.height}"}
    except Exception as e:
        return "", {'error': str(e)}


def extract_text_epub(file_path: str, config: dict) -> Tuple[str, dict]:
    if not LIBRARIES.get('epub') or not LIBRARIES.get('bs4'):
        return "", {'error': 'ebooklib/bs4 not installed'}
    try:
        book = epub.read_epub(file_path)
        chapters = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text = soup.get_text()
                if text.strip():
                    chapters.append(text)
        return '\n\n'.join(chapters), {'chapters': len(chapters)}
    except Exception as e:
        return "", {'error': str(e)}


def extract_text_fb2(file_path: str, config: dict) -> Tuple[str, dict]:
    if not LIBRARIES.get('bs4'):
        return "", {'error': 'BeautifulSoup not installed'}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        soup = BeautifulSoup(content, 'xml')
        bodies = soup.find_all('body')
        sections = [p.get_text(strip=True) for body in bodies for p in body.find_all('p') if p.get_text(strip=True)]
        return '\n\n'.join(sections), {'sections': len(sections)}
    except Exception as e:
        return "", {'error': str(e)}


# =====================================================
# Главный парсер
# =====================================================

EXTRACTORS = {
    'text': extract_text_txt,
    'pdf': extract_text_pdf,
    'docx': extract_text_docx,
    'pptx': extract_text_pptx,
    'xlsx': extract_text_xlsx,
    'image': extract_text_image,
    'ebook': extract_text_epub,
}


def parse_documents(root_dir: str, config: dict, output_file: str = None):
    """Главная функция парсинга."""
    
    ext_map = build_extension_map(config)
    processing = config.get('processing', {})
    output_cfg = config.get('output', {})
    
    max_size = processing.get('max_file_size_mb', 100)
    min_length = processing.get('min_text_length', 100)
    
    # Выходной файл
    if not output_file:
        output_dir = output_cfg.get('directory', 'documents_jsonl')
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_cfg.get('filename_template', 'documents_{timestamp}.jsonl').format(timestamp=timestamp)
        output_file = os.path.join(output_dir, filename)
    
    # Сбор файлов
    all_files = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in ext_map and ext_map[ext] != 'archive':
                all_files.append(os.path.join(root, f))
    
    print(f"📂 Найдено файлов: {len(all_files)}")
    
    results = []
    stats = {'total': len(all_files), 'processed': 0, 'skipped_size': 0, 'skipped_empty': 0}
    
    for file_path in tqdm(all_files, desc="📄 Обработка", unit="файл"):
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > max_size:
            stats['skipped_size'] += 1
            continue
        
        ext = os.path.splitext(file_path)[1].lower()
        file_type = ext_map.get(ext, 'text')
        filename = os.path.basename(file_path)
        
        extractor = EXTRACTORS.get(file_type, extract_text_txt)
        text, ext_stats = extractor(file_path, config)
        text = clean_text(text, config.get('text_cleaning', {}))
        
        if not text or len(text) < min_length:
            stats['skipped_empty'] += 1
            continue
        
        meta = parse_filename_metadata(filename, config)
        rubric = parse_rubric_from_path(file_path, root_dir, config)
        language = detect_language(text, config)
        
        item = {
            "url": file_path,
            "title": meta.get("title") or filename,
            "content": text,
            "excerpt": text[:260] + "..." if len(text) > 260 else text,
            "date": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
            "author": meta.get("author") or config.get('defaults', {}).get('author'),
            "category": rubric or config.get('defaults', {}).get('category'),
            "site": config.get('defaults', {}).get('site', 'local_documents'),
            "hash": sha256_hex(text),
            "language": language or config.get('defaults', {}).get('language'),
            "source_type": config.get('defaults', {}).get('source_type', 'document'),
            "file_name": filename,
            "folder": rubric,
            "text_length": len(text),
            "file_size_kb": round(os.path.getsize(file_path) / 1024, 1),
            "extraction_stats": ext_stats,
            "scraped_at": datetime.now().isoformat(),
        }
        results.append(item)
        stats['processed'] += 1
    
    # Сохранение
    if results:
        with open(output_file, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n✅ Обработано: {stats['processed']} | Пустых: {stats['skipped_empty']} | Крупных: {stats['skipped_size']}")
        print(f"📁 Сохранено: {output_file}")
    else:
        print("❌ Не найдено документов с текстом")


# =====================================================
# CLI
# =====================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Универсальный парсер документов")
    parser.add_argument('folder', help='Папка с документами')
    parser.add_argument('-c', '--config', default='doc_config.yaml', help='Путь к конфигу')
    parser.add_argument('-o', '--output', default=None, help='Выходной JSONL файл')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    parse_documents(args.folder, config, args.output)