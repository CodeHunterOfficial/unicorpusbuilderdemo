# main.py — Universal entry point for the scraper pipeline
"""
TajikPersianNLP Scraper — single entry point for all operations.

Usage:
  # Full pipeline (discovery + extraction)
  python main.py pipeline https://khovar.tj/

  # Config check (test site configuration)
  python main.py config https://khovar.tj/

  # Social scrapers (Telegram, VK, Rutube)
  python main.py social vk --lang tt --max-posts 50
  python main.py social vk --domains club1135692 --lang ba --max-posts 10
  python main.py social telegram --lang tt --max-posts 100
  python main.py social telegram --channels vatantat --lang tt --max-posts 50
  python main.py social rutube --max-comments 200
  python main.py social rutube --video-id abc123 --max-comments 100
  python main.py social all --max-posts 100

  # Wiki dump parser
  python main.py wiki <dump_url> [output_dir] [--max-articles N] [--min-length N] [--transliterate]

  # Document processing (PDF, DOCX, HTML, TXT)
  python main.py docs <input_path> [output_dir]

  # Bulk pipeline — run pipeline on all sites of a language
  python main.py bulk tg
  python main.py bulk all 50

  # Corpus building
  python main.py corpus --lang tg --sources news,social
  python main.py corpus --lang tg --sources all --output corpus_tg.jsonl

  # Help
  python main.py help
"""
from __future__ import annotations
from pathlib import Path
import sys
import os
import json
import time
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.loader import load_modular_config, get_site_key, get_site_config
from pipeline.pipeline_core import PipelineEngine
from pipeline.pipeline_extraction import ExtractionEngine, run as run_extraction
from logger_setup import get_file_logger

logger = get_file_logger("main", "logs/main.log")

OUTPUT_BASE = "output"

SOCIAL_SCRIPTS = {
    "vk": "social/vk_scraper.py",
    "telegram": "social/telegram_scraper.py",
    "rutube": "social/rutube_scraper.py",
}


def _forward_social_args(args):
    """Forward extra arguments to social scrapers."""
    return list(args)


def cmd_config(*args):
    if not args:
        print("Usage: python main.py config <url> [yaml_path]")
        return
    url = args[0]
    yaml_path = args[1] if len(args) > 1 else "config/universal.yaml"
    logger.info(f"Config check for {url}")
    config = load_modular_config(yaml_path)
    site_key = get_site_key(url, config)
    site_cfg = get_site_config(url, "", config)
    lang = site_cfg.get("default_language", "unknown")
    print(f"\n{'='*60}")
    print(f"Site key:  {site_key}")
    print(f"Domain:    {site_cfg.get('_domain')}")
    print(f"Language:  {lang}")
    print(f"Author:    {site_cfg.get('_default_author') or 'auto-detect'}")
    print(f"Start URL: {site_cfg.get('start_url') or url}")
    print(f"SSL verify: {site_cfg.get('verify_ssl')}")
    print(f"{'='*60}")
    print(f"Rubric strategy: {site_cfg.get('rubric_strategy')}")
    print(f"Author strategy: {site_cfg.get('author_strategy')}")
    print(f"Category strategy: {site_cfg.get('category_strategy')}")
    print(f"Noise words:     {len(site_cfg.get('_noise_words', []))}")
    if lang and lang in config.get("languages", {}):
        pkg = config["languages"][lang]
        print(f"{'='*60}")
        print(f"Language package '{lang}':")
        print(f"  Noise words:    {len(pkg.get('noise_words', []))}")
        print(f"  Stopwords:      {len(pkg.get('stopwords', []))}")
        if pkg.get("date_locale"):
            print(f"  Date locale:    {len(pkg['date_locale'])} months")
        if pkg.get("author_regex"):
            print(f"  Author regex:   {len(pkg['author_regex'])} patterns")
        if pkg.get("category_url_patterns"):
            print(f"  Category URL:   {len(pkg['category_url_patterns'])} patterns")
    author_patterns = site_cfg.get("_author_regex_patterns", [])
    if author_patterns:
        print(f"{'='*60}")
        print(f"Resolved author patterns: {len(author_patterns)}")
        for i, pat in enumerate(author_patterns, 1):
            print(f"  {i}. {pat[:100]}...")
    print(f"{'='*60}\n")


def cmd_pipeline(*args):
    if not args:
        print("Usage: python main.py pipeline <url> [max_items]")
        return
    url = args[0]
    max_items = None
    for arg in args[1:]:
        if arg.isdigit():
            max_items = int(arg)
            break
    logger.info(f"Pipeline started for {url} (max_items={max_items})")
    result = run_extraction(start_url=url, yaml_path="config/universal.yaml", max_items=max_items)
    logger.info(f"Pipeline finished: {len(result.get('items', []))} articles")
    print(f"\n{'='*60}")
    print(f"Done: {len(result.get('items', []))} articles extracted")
    print(f"{'='*60}\n")


def cmd_bulk(*args):
    if not args:
        print("Usage: python main.py bulk <language> [max_items_per_site]")
        print("Example: python main.py bulk tg 50")
        return
    language = args[0]
    max_items = None
    for arg in args[1:]:
        if arg.isdigit():
            max_items = int(arg)
            break
    logger.info(f"Bulk pipeline started: language={language}, max_items={max_items}")
    config = load_modular_config("config/universal.yaml")
    sites = config.get("sites", {})
    target_sites = sites if language == "all" else {k: v for k, v in sites.items() if v.get("default_language") == language}
    if not target_sites:
        print(f"No sites found for language '{language}'")
        return
    total = len(target_sites)
    print(f"\n{'='*60}")
    print(f"Found {total} sites for language '{language}'")
    if max_items:
        print(f"Max items per site: {max_items}")
    print(f"{'='*60}\n")
    for idx, (site_key, site_cfg) in enumerate(target_sites.items()):
        match = site_cfg.get("match", [])
        if not match:
            continue
        start_url = site_cfg.get("start_url") or f"https://{match[0]}"
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{total}] Processing: {site_key} ({start_url})")
        print(f"{'='*60}")
        logger.info(f"Bulk [{idx+1}/{total}]: {site_key}")
        cmd = [sys.executable, "main.py", "pipeline", start_url]
        if max_items:
            cmd.append(str(max_items))
        subprocess.run(cmd)
        if idx < total - 1:
            print("Waiting 2 seconds before next site...")
            time.sleep(2)
    print(f"\n{'='*60}")
    print(f"Bulk pipeline finished: {total} sites processed")
    print(f"{'='*60}\n")
    logger.info(f"Bulk pipeline finished: {total} sites")


def cmd_corpus(*args):
    script = "corpus/build_corpus.py"
    if not args:
        print("Usage: python main.py corpus --lang <code> --sources <sources> [--output file]")
        return
    logger.info(f"Corpus building: {' '.join(args)}")
    subprocess.run([sys.executable, script] + list(args))
    logger.info("Corpus building finished")


def cmd_social(*args):
    if not args:
        print("Usage: python main.py social <platform> [scraper options...]")
        print("Platforms: vk, telegram, rutube, all")
        print("Options are forwarded directly to the scraper.")
        print("Examples:")
        print("  python main.py social vk --lang tt --max-posts 50")
        print("  python main.py social vk --domains club1135692 --lang ba --max-posts 10")
        print("  python main.py social telegram --lang tt --max-posts 100")
        print("  python main.py social rutube --max-comments 200")
        print("  python main.py social all --max-posts 100")
        return

    platform = args[0]
    extra_args = list(args[1:])

    if platform == "all":
        for p in SOCIAL_SCRIPTS:
            print(f"\n{'='*60}")
            print(f"Running {p} scraper...")
            print(f"{'='*60}")
            subprocess.run([sys.executable, SOCIAL_SCRIPTS[p]] + extra_args)
    elif platform in SOCIAL_SCRIPTS:
        print(f"\n{'='*60}")
        print(f"Running {platform} scraper...")
        print(f"{'='*60}")
        subprocess.run([sys.executable, SOCIAL_SCRIPTS[platform]] + extra_args)
    else:
        print(f"Unknown platform: {platform}")
        print("Available: vk, telegram, rutube, all")

    logger.info("Social scrapers finished")


def cmd_wiki(*args):
    script = "wiki/wiki_dump_parser.py"
    if not args:
        print("Usage: python main.py wiki <dump_url> [output_dir] [--max-articles N] [--min-length N] [--transliterate]")
        return
    logger.info(f"Running wiki parser: {' '.join(args)}")
    subprocess.run([sys.executable, script] + list(args))
    logger.info("Wiki parser finished")


def cmd_docs(*args):
    script = "documents/universal_doc_parser.py"
    if not args:
        print("Usage: python main.py docs <input_file_or_dir> [output_dir]")
        return
    input_path = args[0]
    output_dir = args[1] if len(args) > 1 else None
    logger.info(f"Running document parser: {input_path} -> {output_dir or 'auto'}")
    if output_dir:
        out_path = Path(output_dir)
        if out_path.suffix:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_path.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = os.path.join(OUTPUT_BASE, "documents")
        os.makedirs(output_dir, exist_ok=True)
    cmd = [sys.executable, script, input_path]
    default_config = "documents/doc_config.yaml"
    if os.path.exists(default_config):
        cmd.extend(['-c', default_config])
    if output_dir:
        cmd.extend(['-o', output_dir])
    subprocess.run(cmd)
    logger.info("Document parser finished")

def cmd_build_vk(*args):
    """Build unified VK dataset from collected posts and comments."""
    script = "social/scripts/build_vk_dataset.py"
    logger.info("Building VK unified dataset")
    print(f"\n{'='*60}")
    print("Building VK unified dataset from posts and comments...")
    if args:
        print(f"Arguments: {' '.join(args)}")
    print(f"{'='*60}")
    subprocess.run([sys.executable, script] + list(args))
    logger.info("VK dataset building finished")

COMMANDS = {
    "config": cmd_config,
    "pipeline": cmd_pipeline,
    "bulk": cmd_bulk,
    "corpus": cmd_corpus,
    "social": cmd_social,
    "wiki": cmd_wiki,
    "docs": cmd_docs,
    "build-vk": cmd_build_vk,
}

HELP = """
TajikPersianNLP Scraper — multilingual news + social + wiki + docs scraper

Commands:
  config   <url>              Check site configuration
  pipeline <url> [max_items]  Run full pipeline (discover + extract)
  bulk     <lang> [max_items] Run pipeline on ALL sites of a language
  corpus   --lang <lg> ...    Build unified corpus from collected data
  social   <platform> [...]   Run social scrapers (vk, telegram, rutube, all)
  wiki     <dump_url> ...     Run Wiki dump parser
  build-vk  [--lang <code>]   Build cleaned unified VK dataset from collected data
                              Optionally filter by language (tg, os, udm, ba, tt, ...)  
  docs     <input> [output]   Process documents (PDF, DOCX, HTML, TXT)

Output paths (all inside output/):
  News articles:   output/<site_key>_<date>_articles.jsonl
  Social data:     output/social/
  Wiki dumps:      output/wiki/<name>/
  Documents:       output/documents/

Examples:
  python main.py config https://khovar.tj/
  python main.py pipeline https://khovar.tj/
  python main.py pipeline https://khovar.tj/ 50
  python main.py bulk tg 50
  python main.py bulk all
  python main.py corpus --lang tg --sources news,social
  python main.py corpus --lang tg --sources all
  python main.py social vk --lang tt --max-posts 100
  python main.py social vk --domains club1135692 irta_tv --lang ba --max-posts 10
  python main.py social vk --token your_token --domains allahtan --lang tt --max-posts 5
  python main.py social telegram --lang tt --max-posts 200
  python main.py social telegram --channels vatantat --lang tt --max-posts 50
  python main.py social rutube --max-comments 150
  python main.py social rutube --video-id abc123 --max-comments 100
  python main.py social rutube --playlist-id playlist_id_1 --max-videos 10 --max-comments 50
  python main.py social all --lang tt --max-posts 100
  python main.py wiki "https://dumps.wikimedia.org/ttwiki/latest/ttwiki-latest-pages-articles.xml.bz2" wikipedia --max-articles 100
  python main.py docs D:/FinSoft output/documents
  python main.py build-vk
  python main.py build-vk --lang tg
  python main.py build-vk --lang os
  python main.py build-vk --lang udm
"""

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("help", "--help", "-h"):
        print(HELP)
        sys.exit(0)
    cmd = sys.argv[1]
    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd}")
        print(HELP)
        sys.exit(1)
    args = sys.argv[2:]
    logger.info(f"Command: {cmd}, Args: {args}")
    try:
        COMMANDS[cmd](*args)
    except TypeError as e:
        print(f"Wrong arguments for '{cmd}': {e}")
        print(f"Use 'python main.py help' for usage")
        logger.error(f"Wrong arguments for '{cmd}': {args}")
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Error in '{cmd}': {e}")
        sys.exit(1)