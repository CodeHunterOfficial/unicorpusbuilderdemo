# wiki_analyzers/wiki_dump_parser.py
"""
Universal Wikimedia dump parser.
Usage:
    python wiki_dump_parser.py <dump_url> [output_dir] [--max-articles N] [--min-length N] [--transliterate]
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

# подключаем локальный транслитератор
from wiki_analyzers.tat_transliterator import transliterate

# -------------------------- Парсер --------------------------

class WikiDumpParser:
    def __init__(self, dump_url: str, output_dir: str, max_articles: Optional[int] = None,
                 min_length: int = 100, do_transliterate: bool = False):
        self.dump_url = dump_url
        self.output_dir = output_dir
        self.max_articles = max_articles
        self.min_length = min_length
        self.do_transliterate = do_transliterate

        self.cyrillic_pattern = re.compile(r'[а-яәөүҗңһ]', re.I)
        self.latin_pattern = re.compile(r'[a-zəğıiöüşç]', re.I)

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
        os.makedirs(output_dir, exist_ok=True)

    # -------- вспомогательные методы (анализ текста) --------
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
        for pat in [r'\[Категория:\s*([^|]+)', r'\[Category:\s*([^|]+)', r'\[Төркем:\s*([^|]+)', r' \[Törkem:\s*([^|]+)']:
            for m in re.findall(pat, text, re.I | re.U):
                c = m.split('|')[0].strip()
                if c and len(c) < 100:
                    cats.append(re.sub(r'\s+', ' ', c))
        return list(set(cats))

    def clean_wikitext(self, text: str) -> str:
        # удаляем вики-разметку
        steps = [(r'', ''), (r'\{\{.*?\}\}', ''), (r']*>.*?', ''),
                 (r'<[^>]*>', ''), (r'\[\[[^]]*?\|([^]]*?)]]', r'\1'),
                 (r'\[\[([^]]*?)]]', r'\1'), (r"'''?", ''),
                 (r'==+.*?==+', ''), (r'\{\|[^}]*?\|\}', ''), (r'&[^;]+;', ' ')]
        for pat, rep in steps:
            text = re.sub(pat, rep, text, flags=re.DOTALL)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        return text.strip()

    # -------- скачивание и парсинг дампа --------
    def download_dump(self) -> str:
        fname = os.path.basename(self.dump_url.split('/')[-1])
        path = os.path.join(self.output_dir, fname)
        if os.path.exists(path):
            print("✅ Дамп уже скачан:", path)
            return path

        print(f"📥 Скачиваем {self.dump_url} ...")
        resp = requests.get(self.dump_url, stream=True)
        total = int(resp.headers.get('content-length', 0))
        with open(path, 'wb') as f, tqdm(desc="Скачивание", total=total, unit='B', unit_scale=True) as pbar:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
                pbar.update(len(chunk))
        return path

    def parse(self):
        dump_path = self.download_dump()
        print("🚀 Парсинг дампа...")

        # имена выходных файлов
        base = os.path.splitext(os.path.basename(dump_path))[0].replace('.xml', '')
        all_file = os.path.join(self.output_dir, f"{base}_all.jsonl")
        cyr_file = os.path.join(self.output_dir, f"{base}_cyrillic.jsonl")
        lat_file = os.path.join(self.output_dir, f"{base}_latin.jsonl")
        mix_file = os.path.join(self.output_dir, f"{base}_mixed.jsonl")

        handles = {'all': open(all_file, 'w', encoding='utf-8'),
                   'cyrillic': open(cyr_file, 'w', encoding='utf-8'),
                   'latin': open(lat_file, 'w', encoding='utf-8'),
                   'mixed': open(mix_file, 'w', encoding='utf-8')}

        all_cats = []

        with bz2.open(dump_path, 'rt', encoding='utf-8') as f:
            for event, elem in tqdm(ET.iterparse(f, events=('end',)), desc="Обработка страниц"):
                if not elem.tag.endswith('page'):
                    continue
                self.stats['total_pages'] += 1

                ns = elem.find('.//{*}ns')
                if ns is not None and ns.text != '0':
                    elem.clear()
                    continue

                title = elem.find('.//{*}title')
                title = title.text if title is not None else ''
                text_elem = elem.find('.//{*}text')
                raw = text_elem.text if text_elem is not None else ''

                if not raw or raw.startswith(('#REDIRECT', '#перенаправление')):
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

                # транслитерация (если нужна и текст латинский)
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
                # url
                domain = self._domain_from_url()
                item['url'] = f"{domain}/wiki/{title.replace(' ', '_')}"

                # сохраняем во все файлы
                for h in handles.values():
                    h.write(json.dumps(item, ensure_ascii=False) + '\n')
                if alph['alphabet'] in handles:
                    handles[alph['alphabet']].write(json.dumps(item, ensure_ascii=False) + '\n')

                self.stats[f'{alph["alphabet"]}_articles'] = \
                    self.stats.get(f'{alph["alphabet"]}_articles', 0) + 1

                elem.clear()
                if self.max_articles and self.stats['articles_found'] >= self.max_articles:
                    break

        for h in handles.values():
            h.close()

        if all_cats:
            self.stats['top_categories'] = dict(Counter(all_cats).most_common(20))

        self._print_stats()
        self._save_stats()

    def _domain_from_url(self) -> str:
        # из URL дампа вида https://dumps.wikimedia.org/ttwiki/... -> https://tt.wikipedia.org
        parts = self.dump_url.split('/')
        project = parts[3]  # ttwiki, ttwikibooks
        if project.endswith('wiki'):
            return f"https://{project[:-4]}.wikipedia.org"
        elif project.endswith('wikibooks'):
            return f"https://{project[:-9]}.wikibooks.org"
        else:
            return f"https://{project}.org"

    def _print_stats(self):
        print("\n" + "="*60)
        print("📊 Статистика")
        print("="*60)
        print(f"Страниц всего: {self.stats['total_pages']}")
        print(f"Статей: {self.stats['articles_found']}")
        print(f"  кириллица: {self.stats['cyrillic_articles']}")
        print(f"  латиница: {self.stats['latin_articles']}")
        print(f"  смешанные: {self.stats['mixed_articles']}")
        print(f"Пропущено перенаправлений: {self.stats['redirects_skipped']}")
        print(f"Пропущено коротких: {self.stats['short_skipped']}")

    def _save_stats(self):
        path = os.path.join(self.output_dir, 'stats.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)

# -------------------------- CLI --------------------------

def main():
    parser = argparse.ArgumentParser(description='Parse Wikimedia dump into JSONL')
    parser.add_argument('dump_url', help='URL of .xml.bz2 dump')
    parser.add_argument('output_dir', nargs='?', default='wiki_output',
                        help='Output directory (default: wiki_output)')
    parser.add_argument('--max-articles', type=int, default=None)
    parser.add_argument('--min-length', type=int, default=100)
    parser.add_argument('--transliterate', action='store_true',
                        help='Transliterate Latin articles to Cyrillic')
    args = parser.parse_args()

    p = WikiDumpParser(args.dump_url, args.output_dir,
                       max_articles=args.max_articles,
                       min_length=args.min_length,
                       do_transliterate=args.transliterate)
    p.parse()

if __name__ == '__main__':
    main()

#python wiki_analyzers/wiki_dump_parser.py  "https://dumps.wikimedia.org/ttwiki/latest/ttwiki-latest-pages-articles.xml.bz2"  wiki_output/wikipedia --max-articles 200
#python wiki_analyzers/wiki_dump_parser.py   "https://dumps.wikimedia.org/ttwikibooks/latest/ttwikibooks-latest-pages-articles.xml.bz2"   wiki_output/wikibooks --tat_transliterator