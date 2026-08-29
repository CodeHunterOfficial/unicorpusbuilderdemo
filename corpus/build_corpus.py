# corpus/build_corpus.py
"""
Multilingual corpus builder – unified JSONL from all sources.

Usage:
  python build_corpus.py --lang tg --output corpus_tg.jsonl --sources all
  python build_corpus.py --lang tt --sources news,wiki --max-items-per-site 100
  python build_corpus.py --lang tg --sources social --social-sources vk,rutube
  python main.py corpus --lang tg --sources news,social --social-sources vk
"""

import json
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.loader import load_modular_config
from pipeline.pipeline_extraction import run as run_extraction


OUTPUT_BASE = "output"
DEFAULT_CORPUS_DIR = os.path.join(OUTPUT_BASE, "corpus")


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    items = []
    if not os.path.isfile(file_path):
        return items
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return items


def unify_item(item: Dict[str, Any], source_type: str,
               language: Optional[str] = None) -> Dict[str, Any]:
    url = item.get('url') or item.get('video_url') or ''
    title = item.get('title') or ''
    content = item.get('content') or item.get('text') or ''
    author = item.get('author') or item.get('author_id') or item.get('from_id')

    if source_type == 'social_rutube' and not author:
        lines = content.split('\n')
        if lines and lines[0].strip():
            author = lines[0].strip()
        parts = [l for l in content.split('\n') if l.strip()]
        if len(parts) >= 3:
            content = '\n'.join(parts[2:]).strip()

    date = item.get('date') or item.get('published') or item.get('scraped_at')
    category = item.get('category')
    if isinstance(category, list):
        category = category[0] if category else None

    lang = language or item.get('language') or item.get('default_language')

    return {
        'url': url,
        'title': title.strip() if title else '',
        'content': content.strip(),
        'author': author,
        'date': date,
        'category': category,
        'source_type': source_type,
        'language': lang
    }


def collect_news_corpus(language: str, config_path: str,
                        max_items_per_site: int = 50) -> List[Dict[str, Any]]:
    config = load_modular_config(config_path)
    sites = config.get('sites', {})
    target = [k for k, v in sites.items() if v.get('default_language') == language]
    if not target:
        print(f"  No sites configured for language '{language}'")
        return []

    all_items = []
    for site_key in target:
        match = sites[site_key].get('match', [])
        if not match:
            continue
        start_url = sites[site_key].get('start_url') or f"https://{match[0]}"
        print(f"  Scraping {site_key} ({start_url})...")
        try:
            result = run_extraction(
                start_url=start_url,
                yaml_path=config_path,
                max_items=max_items_per_site,
                output_jsonl=None,
                output_json=None
            )
            articles = result.get('items', [])
            for art in articles:
                all_items.append(unify_item(art, source_type='news', language=language))
            print(f"    -> {len(articles)} articles extracted")
        except Exception as e:
            print(f"    error: {e}")
    return all_items


def collect_social_corpus(language: str,
                          platforms: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    
    social_dir = os.path.join(OUTPUT_BASE, "social")
    if not os.path.isdir(social_dir):
        print(f"  Directory '{social_dir}' not found – run social scrapers first.")
        return items

    if platforms is None:
        allow_vk = allow_rutube = allow_telegram = True
    else:
        pl = [p.lower() for p in platforms]
        allow_vk = 'vk' in pl
        allow_rutube = 'rutube' in pl
        allow_telegram = 'telegram' in pl

    print(f"  Searching in {social_dir}/ ...")

    if allow_vk:
        found = False
        for subdir in ["posts", "comments"]:
            d = os.path.join(social_dir, subdir)
            if os.path.isdir(d):
                for fname in os.listdir(d):
                    if fname.startswith("vk_") and fname.endswith(".jsonl"):
                        fp = os.path.join(d, fname)
                        for obj in load_jsonl(fp):
                            if subdir == "comments":
                                owner = obj.get('owner_id', 0)
                                post_id = obj.get('post_id', '')
                                obj['url'] = f"https://vk.com/wall{owner}_{post_id}"
                            items.append(unify_item(obj, source_type=f'social_vk_{subdir[:-1]}'))
                        found = True
        if not found:
            print("  No VK files found – run 'python main.py social vk' first.")

    if allow_rutube:
        found = False
        for fname in os.listdir(social_dir):
            if fname.startswith("rutube_") and fname.endswith(".jsonl"):
                fp = os.path.join(social_dir, fname)
                for obj in load_jsonl(fp):
                    items.append(unify_item(obj, source_type='social_rutube'))
                found = True
        if not found:
            print("  No Rutube files found – run 'python main.py social rutube' first.")

    if allow_telegram:
        found = False
        for fname in os.listdir(social_dir):
            if fname.startswith("telegram_") and fname.endswith(".jsonl"):
                fp = os.path.join(social_dir, fname)
                for obj in load_jsonl(fp):
                    items.append(unify_item(obj, source_type='social_telegram'))
                found = True
        if not found:
            print("  No Telegram files found – run scraper with a proxy if needed.")

    for it in items:
        if not it.get('language'):
            it['language'] = language

    return items


def collect_wiki_corpus(language: str) -> List[Dict[str, Any]]:
    items = []
    wiki_dir = os.path.join(OUTPUT_BASE, "wiki")
    if not os.path.isdir(wiki_dir):
        print(f"  Directory '{wiki_dir}' not found.")
        return items

    for root, dirs, files in os.walk(wiki_dir):
        for fname in files:
            if fname.endswith("_all.jsonl"):
                prefix = fname.split('-')[0].replace('wiki', '')
                if prefix and language and prefix.lower() != language:
                    continue
                fp = os.path.join(root, fname)
                for obj in load_jsonl(fp):
                    unified = unify_item(obj, source_type='wiki')
                    unified['content'] = obj.get('text', '')
                    unified['url'] = obj.get('url', '')
                    cats = obj.get('categories', [])
                    if isinstance(cats, list) and cats:
                        unified['category'] = cats[0]
                    if not unified['language']:
                        unified['language'] = language
                    items.append(unified)

    return items


def collect_documents_corpus(language: str) -> List[Dict[str, Any]]:
    items = []
    docs_dir = os.path.join(OUTPUT_BASE, "documents")
    if not os.path.isdir(docs_dir):
        print(f"  Directory '{docs_dir}' not found.")
        return items

    for fname in os.listdir(docs_dir):
        if fname.endswith(".jsonl"):
            fp = os.path.join(docs_dir, fname)
            for obj in load_jsonl(fp):
                unified = unify_item(obj, source_type='document')
                if not unified['language']:
                    unified['language'] = language or 'unknown'
                items.append(unified)
    return items


def build_corpus(language: str = 'tg',
                 output_file: str = None,
                 config_path: str = 'config/universal.yaml',
                 max_items_per_site: int = 50,
                 sources: str = 'all',
                 social_sources: Optional[str] = None) -> List[Dict[str, Any]]:

    if sources == 'all':
        active = {'news', 'social', 'wiki', 'docs'}
    else:
        active = set(s.strip().lower() for s in sources.split(',') if s.strip())

    social_platforms = None
    if social_sources and social_sources != 'all':
        social_platforms = [p.strip().lower() for p in social_sources.split(',') if p.strip()]

    all_items: List[Dict[str, Any]] = []
    print(f"=== Building corpus for language '{language}' ===\n")

    if 'news' in active:
        print("[News] Collecting...")
        news = collect_news_corpus(language, config_path, max_items_per_site)
        all_items.extend(news)
        print(f"  Total news: {len(news)}")
    else:
        print("[News] Skipped.")

    if 'social' in active:
        print("\n[Social] Collecting...")
        social = collect_social_corpus(language, platforms=social_platforms)
        all_items.extend(social)
        print(f"  Total social: {len(social)}")
    else:
        print("[Social] Skipped.")

    if 'wiki' in active:
        print("\n[Wiki] Collecting...")
        wiki = collect_wiki_corpus(language)
        all_items.extend(wiki)
        print(f"  Total wiki: {len(wiki)}")
    else:
        print("[Wiki] Skipped.")

    if 'docs' in active:
        print("\n[Docs] Collecting...")
        docs = collect_documents_corpus(language)
        all_items.extend(docs)
        print(f"  Total docs: {len(docs)}")
    else:
        print("[Docs] Skipped.")

    unique: Dict[str, Dict[str, Any]] = {}
    for it in all_items:
        key = it.get('url') or it.get('content')
        if key and key not in unique:
            unique[key] = it
    corpus = list(unique.values())

    if not output_file:
        os.makedirs(DEFAULT_CORPUS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(DEFAULT_CORPUS_DIR, f"corpus_{language}_{timestamp}.jsonl")
    else:
        output_path = Path(output_file)
        if output_path.suffix:
            os.makedirs(output_path.parent, exist_ok=True)
        else:
            os.makedirs(output_path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(str(output_path), f"corpus_{language}_{timestamp}.jsonl")

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in corpus:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\nCorpus saved: {output_file} ({len(corpus)} records)")
    return corpus


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build a unified text corpus from multiple sources")
    parser.add_argument('--lang', required=True, help='Language code (tg, tt, ru, ba, en)')
    parser.add_argument('--output', default=None, help='Output JSONL file (optional)')
    parser.add_argument('--config', default='config/universal.yaml', help='Path to the scraper configuration')
    parser.add_argument('--max-items-per-site', type=int, default=50, help='Max articles per news site')
    parser.add_argument('--sources', default='all', help='Sources: all, news, social, wiki, docs (comma separated)')
    parser.add_argument('--social-sources', default='all', help='Social platforms: all, vk, rutube, telegram (comma separated)')

    args = parser.parse_args()
    build_corpus(
        language=args.lang,
        output_file=args.output,
        config_path=args.config,
        max_items_per_site=args.max_items_per_site,
        sources=args.sources,
        social_sources=args.social_sources
    )