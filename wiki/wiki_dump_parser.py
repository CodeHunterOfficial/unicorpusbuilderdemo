# wiki/wiki_dump_parser.py
"""
Universal Wikimedia dump parser with logging, integrity checks, and optional prefix filter.
Usage:
    python wiki_dump_parser.py <dump_url> [output_dir] [--max-articles N] [--min-length N] [--transliterate] [--filter-prefix PREFIX]
"""

import os
import sys
import json
import re
import argparse
import requests
import bz2
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Dict, List, Optional
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wiki.tat_transliterator import transliterate
from logger_setup import get_file_logger

logger = get_file_logger("wiki_parser", "logs/wiki_parser.log")


def validate_bz2(filepath: str) -> bool:
    try:
        with bz2.open(filepath, 'rb') as f:
            f.read(1024 * 1024)
        return True
    except (EOFError, OSError, Exception):
        return False


def download_dump(url: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    filename = url.split('/')[-1]
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath):
        if validate_bz2(filepath):
            print(f"[OK] Dump already downloaded and valid: {filepath}")
            logger.info(f"Dump already exists and valid: {filepath}")
            return filepath
        else:
            print(f"[WARN] Existing file corrupted, re-downloading...")
            logger.warning(f"Corrupted file deleted: {filepath}")
            os.remove(filepath)

    print(f"[DOWNLOAD] {url}")
    logger.info(f"Downloading dump: {url}")
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))

    with tqdm(total=total_size, unit='B', unit_scale=True, desc="[DOWNLOAD]") as pbar:
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    if total_size > 0 and os.path.getsize(filepath) < total_size * 0.99:
        print(f"[ERROR] Download incomplete")
        logger.error(f"Download incomplete: {os.path.getsize(filepath)} < {total_size}")
        os.remove(filepath)
        raise RuntimeError("Download incomplete")

    if not validate_bz2(filepath):
        print(f"[ERROR] Downloaded file corrupted")
        logger.error("Downloaded file failed validation")
        os.remove(filepath)
        raise RuntimeError("Downloaded file corrupted")

    print(f"[OK] Downloaded: {filepath}")
    logger.info(f"Downloaded {total_size} bytes to {filepath}")
    return filepath


class WikiDumpParser:
    def __init__(self, dump_url: str, output_dir: str, max_articles: Optional[int] = None,
                 min_length: int = 100, do_transliterate: bool = False,
                 filter_prefix: Optional[str] = None):
        self.dump_url = dump_url
        if not os.path.isabs(output_dir):
            self.output_dir = os.path.join("output", "wiki", output_dir)
        else:
            self.output_dir = output_dir

        self.max_articles = max_articles
        self.min_length = min_length
        self.do_transliterate = do_transliterate
        self.filter_prefix = filter_prefix

        self.cyrillic_pattern = re.compile(r'[а-яёәөүҗңһӣғқҳҷӯ]', re.I)
        self.latin_pattern = re.compile(r'[a-z]', re.I)

        self.stats = {
            'total_pages': 0,
            'articles_found': 0,
            'cyrillic_articles': 0,
            'latin_articles': 0,
            'mixed_articles': 0,
            'other_articles': 0,
            'redirects_skipped': 0,
            'short_skipped': 0,
            'top_categories': {}
        }
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Initialized WikiDumpParser: url={dump_url}, output={self.output_dir}, "
                    f"filter_prefix={self.filter_prefix}")

    def detect_alphabet(self, text: str) -> Dict:
        if not text:
            return {'alphabet': 'empty', 'cyr_ratio': 0, 'lat_ratio': 0}
        text_lower = text.lower()
        cyr_cnt = len(self.cyrillic_pattern.findall(text_lower))
        lat_cnt = len(self.latin_pattern.findall(text_lower))
        total = cyr_cnt + lat_cnt
        if total == 0:
            return {'alphabet': 'other', 'cyr_ratio': 0, 'lat_ratio': 0}
        cyr_ratio = cyr_cnt / total
        lat_ratio = lat_cnt / total
        if cyr_ratio > 0.8:
            alphabet = 'cyrillic'
        elif lat_ratio > 0.8:
            alphabet = 'latin'
        elif cyr_ratio > 0.4 and lat_ratio > 0.4:
            alphabet = 'mixed'
        elif cyr_ratio > 0.3:
            alphabet = 'cyrillic'
        elif lat_ratio > 0.3:
            alphabet = 'latin'
        else:
            alphabet = 'other'
        return {'alphabet': alphabet, 'cyr_ratio': cyr_ratio, 'lat_ratio': lat_ratio}

    def extract_categories(self, text: str) -> List[str]:
        cats = []
        patterns = [
            r'\[Category:\s*([^|]+)',
            r'\[Category:\s*([^|]+)',
            r'\[Category:\s*([^|]+)',
            r'\[Category:\s*([^|]+)'
        ]
        for pat in patterns:
            for m in re.findall(pat, text, re.I | re.U):
                c = m.split('|')[0].strip()
                if c and len(c) < 100:
                    cats.append(re.sub(r'\s+', ' ', c))
        return list(set(cats))

    def clean_wikitext(self, text: str) -> str:
        steps = [
            (r'\{\{.*?\}\}', ''),
            (r'<ref[^>]*>.*?</ref>', ''),
            (r'<[^>]*>', ''),
            (r'\[\[[^]]*?\|([^]]*?)]]', r'\1'),
            (r'\[\[([^]]*?)]]', r'\1'),
            (r"'''?", ''),
            (r'==+.*?==+', ''),
            (r'\{\|[^}]*?\|\}', ''),
            (r'&[^;]+;', ' '),
            (r'\[https?://[^\]]*\]', ''),
        ]
        for pat, rep in steps:
            text = re.sub(pat, rep, text, flags=re.DOTALL)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        return text.strip()

    def parse(self):
        logger.info("Starting dump parsing...")
        dump_path = download_dump(self.dump_url, self.output_dir)
        print("[START] Parsing dump...")

        base = os.path.splitext(os.path.basename(dump_path))[0].replace('.xml', '')
        all_file = os.path.join(self.output_dir, f"{base}_all.jsonl")
        cyr_file = os.path.join(self.output_dir, f"{base}_cyrillic.jsonl")
        lat_file = os.path.join(self.output_dir, f"{base}_latin.jsonl")
        mix_file = os.path.join(self.output_dir, f"{base}_mixed.jsonl")

        handles = {
            'all': open(all_file, 'w', encoding='utf-8'),
            'cyrillic': open(cyr_file, 'w', encoding='utf-8'),
            'latin': open(lat_file, 'w', encoding='utf-8'),
            'mixed': open(mix_file, 'w', encoding='utf-8')
        }
        logger.info(f"Output files: {all_file}, {cyr_file}, {lat_file}, {mix_file}")

        all_cats = []

        try:
            with bz2.open(dump_path, 'rt', encoding='utf-8') as f:
                for event, elem in tqdm(ET.iterparse(f, events=('end',)), desc="Processing pages"):
                    if not elem.tag.endswith('page'):
                        continue
                    self.stats['total_pages'] += 1

                    ns = elem.find('.//{*}ns')
                    if ns is not None and ns.text != '0':
                        elem.clear()
                        continue

                    title = elem.find('.//{*}title')
                    title = title.text if title is not None else ''

                    # Prefix filter (for incubator)
                    if self.filter_prefix and not title.startswith(self.filter_prefix):
                        elem.clear()
                        continue

                    text_elem = elem.find('.//{*}text')
                    raw = text_elem.text if text_elem is not None else ''

                    if not raw or raw.startswith(('#REDIRECT', '#redirect', '#Redirect')):
                        self.stats['redirects_skipped'] += 1
                        elem.clear()
                        continue

                    clean = self.clean_wikitext(raw)
                    if len(clean) < self.min_length:
                        self.stats['short_skipped'] += 1
                        elem.clear()
                        continue

                    self.stats['articles_found'] += 1
                    alph = self.detect_alphabet(clean)
                    cats = self.extract_categories(raw)
                    all_cats.extend(cats)

                    original_text = clean
                    if self.do_transliterate and alph['alphabet'] == 'latin':
                        clean = transliterate(clean)
                        alph['alphabet'] = 'cyrillic'

                    item = {
                        'title': title,
                        'text': clean,
                        'original_text': original_text if self.do_transliterate else clean,
                        'categories': cats,
                        'alphabet': alph['alphabet'],
                        'cyr_ratio': alph['cyr_ratio'],
                        'lat_ratio': alph['lat_ratio'],
                        'text_length': len(clean)
                    }
                    domain = self._domain_from_url()
                    item['url'] = f"{domain}/wiki/{title.replace(' ', '_')}"

                    handles['all'].write(json.dumps(item, ensure_ascii=False) + '\n')
                    if alph['alphabet'] in handles:
                        handles[alph['alphabet']].write(json.dumps(item, ensure_ascii=False) + '\n')

                    self.stats[f'{alph["alphabet"]}_articles'] = \
                        self.stats.get(f'{alph["alphabet"]}_articles', 0) + 1

                    elem.clear()
                    if self.max_articles and self.stats['articles_found'] >= self.max_articles:
                        logger.info(f"Reached max articles limit: {self.max_articles}")
                        break

        except EOFError:
            print(f"\n[WARN] Dump ended unexpectedly after {self.stats['articles_found']} articles")
            print("[INFO] The dump file may be corrupted. Delete it and re-run:")
            print(f"  del {dump_path}")
            logger.warning(f"EOFError after {self.stats['articles_found']} articles")
        except Exception as e:
            print(f"\n[ERROR] Parsing failed: {e}")
            logger.error(f"Parsing error: {e}")
            raise
        finally:
            for h in handles.values():
                h.close()

        if all_cats:
            self.stats['top_categories'] = dict(Counter(all_cats).most_common(20))

        self._print_stats()
        self._save_stats()
        logger.info(f"Parsing completed. Articles: {self.stats['articles_found']}")

    def _domain_from_url(self) -> str:
        parts = self.dump_url.split('/')
        project = parts[3]
        if project.endswith('wiki'):
            return f"https://{project[:-4]}.wikipedia.org"
        elif project.endswith('wikibooks'):
            return f"https://{project[:-9]}.wikibooks.org"
        else:
            return f"https://{project}.org"

    def _print_stats(self):
        print("\n" + "=" * 60)
        print("[STATS] Statistics")
        print("=" * 60)
        print(f"Total pages: {self.stats['total_pages']}")
        print(f"Articles: {self.stats['articles_found']}")
        print(f"  cyrillic: {self.stats['cyrillic_articles']}")
        print(f"  latin: {self.stats['latin_articles']}")
        print(f"  mixed: {self.stats['mixed_articles']}")
        print(f"Redirects skipped: {self.stats['redirects_skipped']}")
        print(f"Short articles skipped: {self.stats['short_skipped']}")

    def _save_stats(self):
        path = os.path.join(self.output_dir, 'stats.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        logger.info(f"Stats saved to {path}")


def main():
    parser = argparse.ArgumentParser(description='Parse Wikimedia dump into JSONL')
    parser.add_argument('dump_url', help='URL of .xml.bz2 dump')
    parser.add_argument('output_dir', nargs='?', default='wiki_output',
                        help='Output directory (default: wiki_output)')
    parser.add_argument('--max-articles', type=int, default=None)
    parser.add_argument('--min-length', type=int, default=100)
    parser.add_argument('--transliterate', action='store_true',
                        help='Transliterate Latin articles to Cyrillic')
    parser.add_argument('--filter-prefix', default=None,
                        help='Only process pages with titles starting with this prefix')
    args = parser.parse_args()

    logger.info(f"CLI started: url={args.dump_url}, output={args.output_dir}, "
                f"max_articles={args.max_articles}, transliterate={args.transliterate}, "
                f"filter_prefix={args.filter_prefix}")

    p = WikiDumpParser(args.dump_url, args.output_dir,
                       max_articles=args.max_articles,
                       min_length=args.min_length,
                       do_transliterate=args.transliterate,
                       filter_prefix=args.filter_prefix)
    p.parse()


if __name__ == '__main__':
    main()