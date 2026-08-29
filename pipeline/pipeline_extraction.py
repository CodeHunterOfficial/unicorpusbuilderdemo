# pipeline/pipeline_extraction.py
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import os
import re
import time
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Set
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Tag
from tqdm import tqdm

# Imports from Part 1 (core + all utilities)
from pipeline.pipeline_core import (
    PipelineEngine,
    abs_url,
    clean_text,
    ensure_list,
    get_domain,
    get_path,
    is_noise_text,
    is_noise_url,
    is_valid_http_url,
    parse_datetime_value,
    same_domain,
    sha256_hex,
    extract_time_part,
    apply_date_locale,
    now_iso,
)

from logger_setup import get_file_logger

# Initialize file-only logger
logger = get_file_logger("pipeline_extraction", "logs/pipeline_extraction.log")

# =====================================================
# Extraction Engine
# =====================================================

class ExtractionEngine(PipelineEngine):
    """
    Inherits all Part1 infrastructure (config loading, HTTP, discovery...)
    and adds deep extraction of article fields.
    """

    # -------------------------------------------------
    # JSON-LD / author helpers
    # -------------------------------------------------

    def jsonld_to_author(self, data: Any) -> Optional[str]:
        try:
            if isinstance(data, dict):
                typ = data.get("@type")
                if isinstance(typ, list):
                    typ = " ".join(map(str, typ))
                if typ in ("NewsArticle", "Article", "ReportageNewsArticle", "BlogPosting", "WebPage"):
                    author = data.get("author")
                    if isinstance(author, dict) and author.get("name"):
                        return clean_text(author["name"])
                    if isinstance(author, list):
                        for a in author:
                            if isinstance(a, dict) and a.get("name"):
                                return clean_text(a["name"])
                            if isinstance(a, str) and a.strip():
                                return clean_text(a)

                if "@graph" in data and isinstance(data["@graph"], list):
                    for item in data["@graph"]:
                        res = self.jsonld_to_author(item)
                        if res:
                            return res

                for v in data.values():
                    if isinstance(v, (dict, list)):
                        res = self.jsonld_to_author(v)
                        if res:
                            return res

            elif isinstance(data, list):
                for item in data:
                    res = self.jsonld_to_author(item)
                    if res:
                        return res
        except Exception:
            pass
        return None

    def extract_author_regex(self, soup: BeautifulSoup, site_cfg: Dict[str, Any]) -> Optional[str]:
        global_text = clean_text(soup.get_text(" ", strip=True))
        patterns = site_cfg.get("author_regex_patterns") or [
            r"(?:Author|By|From\s+the\s+author|Author|Prepared(?:by)?|Text\s+author)\s*[:\-]?\s*([A-Za-zА-Яа-яЁёӨөҮүҚқҒғҲҳӢӣЪъІіЇї'’\-\.\s]{2,120})",
            r"(?:written\s+by|reported\s+by)\s*[:\-]?\s*([A-Za-zА-Яа-яЁёӨөҮүҚқҒғҲҳӢӣЪъІіЇї'’\-\.\s]{2,120})",
        ]
        for pat in patterns:
            m = re.search(pat, global_text, flags=re.IGNORECASE)
            if m:
                val = clean_text(m.group(1))
                if val:
                    return val
        return None

    def extract_author_from_soup(self, soup: BeautifulSoup, site_cfg: Dict[str, Any], url: str) -> Optional[str]:
        strategies = site_cfg.get("author_strategy") or self.global_cfg().get(
            "author_strategies_order"
        ) or [
            "meta_tag",
            "class_selector",
            "style_based",
            "json_ld_graph",
            "json_ld_simple",
            "regex_in_content",
            "priority_list",
            "default_fallback",
        ]

        def meta_tag() -> Optional[str]:
            for sel in site_cfg.get("author_selectors", []):
                # use safe select_one
                try:
                    el = self._safe_select_one(soup, sel)
                except Exception:
                    el = None
                if el:
                    if el.name == "meta" and el.get("content"):
                        val = clean_text(el.get("content"))
                    else:
                        val = clean_text(el.get_text(" ", strip=True))
                    if val:
                        return val
            return None

        def class_selector() -> Optional[str]:
            selectors = [
                ".author",
                ".article-author",
                ".byline",
                ".by",
                ".c-author",
                ".c-author__content",
                ".media-block__title--author",
                ".post-author",
                ".news-author",
            ]
            for sel in selectors:
                el = soup.select_one(sel)
                if el:
                    txt = clean_text(el.get_text(" ", strip=True))
                    if txt:
                        return txt
            return None

        def style_based() -> Optional[str]:
            for el in soup.find_all(style=True):
                style = (el.get("style") or "").lower()
                txt = clean_text(el.get_text(" ", strip=True))
                if not txt:
                    continue
                if "author" in style or "byline" in style or re.search(r"author|author|author", txt, re.I):
                    return txt
            return None

        def json_ld_simple() -> Optional[str]:
            for tag in soup.find_all("script", type="application/ld+json"):
                raw = tag.get_text(strip=True)
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except Exception:
                    continue
                if isinstance(data, dict):
                    author = data.get("author")
                    if isinstance(author, dict) and author.get("name"):
                        return clean_text(author["name"])
                    if isinstance(author, list):
                        for a in author:
                            if isinstance(a, dict) and a.get("name"):
                                return clean_text(a["name"])
                            if isinstance(a, str) and a.strip():
                                return clean_text(a)
            return None

        def json_ld_graph() -> Optional[str]:
            for tag in soup.find_all("script", type="application/ld+json"):
                raw = tag.get_text(strip=True)
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except Exception:
                    continue
                res = self.jsonld_to_author(data)
                if res:
                    return res
            return None

        def priority_list() -> Optional[str]:
            ordered = site_cfg.get("author_priority") or strategies
            return self.resolve_author_by_order(soup, site_cfg, url, ordered)

        def default_fallback() -> Optional[str]:
            return site_cfg.get("default_author")

        mapping = {
            "meta_tag": meta_tag,
            "class_selector": class_selector,
            "style_based": style_based,
            "json_ld_simple": json_ld_simple,
            "json_ld_graph": json_ld_graph,
            "regex_in_content": lambda: self.extract_author_regex(soup, site_cfg),
            "priority_list": priority_list,
            "default_fallback": default_fallback,
        }

        return self.resolve_author_by_order(soup, site_cfg, url, strategies, mapping)

    def resolve_author_by_order(
        self,
        soup: BeautifulSoup,
        site_cfg: Dict[str, Any],
        url: str,
        order: List[str],
        mapping: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        mapping = mapping or {}

        for key in order:
            fn = mapping.get(key)
            if not fn:
                continue
            try:
                val = fn()
                if val:
                    val = clean_text(val)
                    val = re.sub(r"^(Author|By|From the author|Author)\s*[:\-]?\s*", "", val, flags=re.I)
                    val = clean_text(val)
                    if val:
                        return val
            except Exception:
                continue

        return site_cfg.get("default_author")

    # -------------------------------------------------
    # Title
    # -------------------------------------------------

    def extract_title(self, soup: BeautifulSoup, site_cfg: Dict[str, Any]) -> Optional[str]:
        for sel in site_cfg.get("title_selectors", []):
            try:
                el = self._safe_select_one(soup, sel)
            except Exception:
                el = None
            if el:
                if el.name == "meta" and el.get("content"):
                    val = clean_text(el.get("content"))
                else:
                    val = clean_text(el.get_text(" ", strip=True))
                if val:
                    return val
        if soup.title and soup.title.string:
            return clean_text(soup.title.string)
        return None

    # -------------------------------------------------
    # Date (with locale_map support)
    # -------------------------------------------------
    def extract_date(
        self,
        soup: BeautifulSoup,
        site_cfg: Dict[str, Any],
        locale_map: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """
        Parse date, applying locale_map first.
        """
        for sel in site_cfg.get("date_selectors", []):
            try:
                el = self._safe_select_one(soup, sel)
            except Exception:
                el = None
            if el:
                val = el.get("content") or el.get("datetime") or el.get_text(" ", strip=True)
                if val:
                    parsed = parse_datetime_value(val, locale_map)
                    if parsed:
                        return parsed
        return None

    # -------------------------------------------------
    # Category (with url_path_parsing strategy)
    # -------------------------------------------------
    def extract_category(self, soup: BeautifulSoup, url: str, site_cfg: Dict[str, Any]) -> Optional[str]:
        strategies = site_cfg.get("category_strategy") or []
        
        if "url_path_parsing" in strategies:
            # Patterns from reusable_strategies (global)
            patterns = self.root_cfg.get("reusable_strategies", {}).get(
                "category_sources", {}).get("url_path_parsing", {}).get("patterns", [])
            
            # Additional language-specific patterns
            lang = site_cfg.get("default_language")
            if lang:
                lang_patterns = self.root_cfg.get("languages", {}).get(lang, {}).get("category_url_patterns", [])
                patterns = list(set(patterns + lang_patterns))
            
            for pattern in patterns:
                m = re.search(pattern, url)
                if m:
                    return m.group(1)

        for sel in site_cfg.get("category_selectors", []):
            try:
                el = self._safe_select_one(soup, sel)
            except Exception:
                el = None
            if el:
                if el.name == "meta" and el.get("content"):
                    val = clean_text(el.get("content"))
                else:
                    val = clean_text(el.get_text(" ", strip=True))
                if val:
                    return val

        path = get_path(url).lower()
        if "/news/rubric/list/" in path:
            return path.split("/news/rubric/list/")[-1].split("/")[0].split("?")[0]
        if "/news/" in path:
            return "news"
        if "/photo/" in path:
            return "photo"
        if "/video/" in path:
            return "video"
        return None

    # -------------------------------------------------
    # Language / Image
    # -------------------------------------------------

    def extract_language(self, soup: BeautifulSoup, site_cfg: Dict[str, Any]) -> Optional[str]:
        lang = site_cfg.get("default_language")
        if lang:
            return lang
        html_lang = soup.find("html")
        if html_lang and html_lang.get("lang"):
            return clean_text(html_lang.get("lang"))
        return None

    def extract_image_url(self, soup: BeautifulSoup, base_url: str, site_cfg: Dict[str, Any]) -> Optional[str]:
        for sel in site_cfg.get("image_selectors", []):
            try:
                node = self._safe_select_one(soup, sel)
            except Exception:
                node = None
            if node and node.get("content"):
                img = abs_url(base_url, node.get("content"))
                if img and not img.lower().endswith(".svg"):
                    return img

        for img in soup.find_all("img"):
            for attr in ["src", "data-src", "data-lazy-src", "data-original"]:
                if img.get(attr):
                    img_url = abs_url(base_url, img.get(attr))
                    if img_url and not img_url.lower().endswith(".svg"):
                        return img_url
        return None

    # -------------------------------------------------
    # Content cleanup / extraction
    # -------------------------------------------------

    def clean_html(self, soup: BeautifulSoup, extra_remove: Optional[List[str]] = None) -> BeautifulSoup:
        for sel in ["script", "style", "noscript", "iframe", "svg", "form", "button", "canvas"]:
            for node in soup.select(sel):
                try:
                    node.decompose()
                except Exception:
                    pass

        for sel in extra_remove or []:
            for node in soup.select(sel):
                try:
                    node.decompose()
                except Exception:
                    pass

        return soup

    def extract_content_text(self, container: BeautifulSoup, noise_words: Optional[List[str]] = None) -> str:
        parts: List[str] = []
        seen: Set[str] = set()
        for el in container.find_all(["h1", "h2", "h3", "h4", "p", "li", "blockquote"], recursive=True):
            if not isinstance(el, Tag):
                continue
            txt = clean_text(el.get_text(" ", strip=True))
            if not txt or is_noise_text(txt, noise_words=noise_words):
                continue
            key = sha256_hex(txt, trunc=None)
            if key in seen:
                continue
            seen.add(key)
            if el.name in ["h1", "h2", "h3", "h4"]:
                parts.append(f"\n{txt}\n")
            else:
                parts.append(txt)
        return re.sub(r"\n{3,}", "\n\n", "\n".join(parts).strip())

    def extract_full_visible_text(self, soup: BeautifulSoup, noise_words: Optional[List[str]] = None) -> str:
        parts: List[str] = []
        seen: Set[str] = set()
        for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "blockquote"]):
            if not isinstance(el, Tag):
                continue
            txt = clean_text(el.get_text(" ", strip=True))
            if not txt or is_noise_text(txt, noise_words=noise_words):
                continue
            key = sha256_hex(txt, trunc=None)
            if key in seen:
                continue
            seen.add(key)
            if el.name in ["h1", "h2", "h3", "h4"]:
                parts.append(f"\n{txt}\n")
            else:
                parts.append(txt)
        return re.sub(r"\n{3,}", "\n\n", "\n".join(parts).strip())

    def find_best_content_container(self, soup: BeautifulSoup, site_cfg: Dict[str, Any]) -> BeautifulSoup:
        for sel in site_cfg.get("content_selectors", []):
            try:
                node = self._safe_select_one(soup, sel)
            except Exception:
                node = None
            if node:
                return node

        candidates: List[tuple[float, Tag]] = []
        for tag in soup.find_all(["article", "main", "section", "div"]):
            if not isinstance(tag, Tag):
                continue
            text = clean_text(tag.get_text(" ", strip=True))
            if len(text) < 180:
                continue

            cls_id = " ".join(tag.get("class", [])) + " " + (tag.get("id") or "")
            cls_id = cls_id.lower()

            penalty = 0
            if any(x in cls_id for x in ["nav", "menu", "header", "footer", "sidebar", "banner", "social", "share", "cookie", "breadcrumb"]):
                penalty += 500

            link_count = len(tag.find_all("a", href=True))
            text_len = max(len(text), 1)
            link_density = min(1.0, link_count / max(10, text_len / 80))

            score = text_len - int(link_density * 1000) - penalty
            if tag.name == "article":
                score += 300
            if tag.name == "main":
                score += 200
            if tag.find("h1"):
                score += 100

            candidates.append((score, tag))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]
        return soup

    # -------------------------------------------------
    # Page meta and article fields
    # -------------------------------------------------

    def extract_page_meta(
        self,
        soup: BeautifulSoup,
        url: str,
        site_cfg: Dict[str, Any],
        locale_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Collect page metadata.
        locale_map is passed to extract_date for month name support.
        """
        return {
            "title": self.extract_title(soup, site_cfg),
            "date": self.extract_date(soup, site_cfg, locale_map=locale_map),
            "author": self.extract_author_from_soup(soup, site_cfg, url),
            "category": self.extract_category(soup, url, site_cfg),
            "image_url": self.extract_image_url(soup, url, site_cfg),
            "language": self.extract_language(soup, site_cfg),
        }

    def extract_article_fields(self, html: str, url: str) -> Dict[str, Any]:
        soup = BeautifulSoup(html, "html.parser")
        site_cfg = self.site_cfg(url)
        locale_map = self.get_date_locale_map(url)
        noise_words = site_cfg.get("_noise_words", [])

        soup = self.clean_html(soup, extra_remove=site_cfg.get("remove_selectors", []))
        page_meta = self.extract_page_meta(soup, url, site_cfg, locale_map=locale_map)

        container = self.find_best_content_container(soup, site_cfg)
        if container:
            container = self.clean_html(container, extra_remove=site_cfg.get("remove_selectors", []))
            content = self.extract_content_text(container, noise_words=noise_words)
        else:
            content = ""

        if not content:
            content = self.extract_full_visible_text(soup, noise_words=noise_words)

        excerpt_len = int(self.global_cfg().get("limits", {}).get("excerpt_len", 260))
        full_clean = clean_text(content)
        excerpt = full_clean[:excerpt_len].rstrip()
        if len(full_clean) > excerpt_len:
            excerpt += "..."

        h = sha256_hex((content or "") + "|" + (page_meta.get("title") or "") + "|" + (url or ""), trunc=32)

        return {
            "url": url,
            "title": page_meta.get("title"),
            "content": content,
            "excerpt": excerpt,
            "date": page_meta.get("date"),
            "category": page_meta.get("category"),
            "author": page_meta.get("author"),
            "time": extract_time_part(page_meta.get("date") or ""),
            "site": get_domain(url),
            "hash": h,
            "image_url": page_meta.get("image_url"),
            "language": page_meta.get("language"),
            "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "page_type": "article",
        }

    # -------------------------------------------------
    # AMP fallback / article fetching
    # -------------------------------------------------

    def _amp_url_from(self, url: str) -> Optional[str]:
        site_cfg = self.site_cfg(url)
        mode = site_cfg.get("amp_mode", "none")
        if mode == "none":
            return None
        parsed = urlparse(url)
        if mode == "prefix":
            amp_path = "/amp" + parsed.path if parsed.path.startswith("/") else "/amp/" + parsed.path
            return f"{parsed.scheme}://{parsed.netloc}{amp_path}"
        if mode == "suffix":
            amp_path = parsed.path.rstrip("/") + "/amp"
            return f"{parsed.scheme}://{parsed.netloc}{amp_path}"
        return None

    def fetch_article_html(self, url: str) -> Optional[str]:
        try:
            return self.fetch_html(url)
        except Exception:
            pass

        if self.use_amp_fallback():
            amp = self._amp_url_from(url)
            if amp:
                try:
                    return self.fetch_html(amp)
                except Exception:
                    pass
        return None

    def extract_item_from_url(self, url: str, fallback: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        if not self.can_fetch_robots(url):
            return fallback

        html = self.fetch_article_html(url)
        if not html:
            return fallback

        data = self.extract_article_fields(html, url)

        if fallback:
            for k in ["title", "content", "date", "author", "category", "time"]:
                if not data.get(k) and fallback.get(k):
                    data[k] = fallback.get(k)
            if fallback.get("source_page") and not data.get("source_page"):
                data["source_page"] = fallback.get("source_page")
            if fallback.get("page_type") and not data.get("page_type"):
                data["page_type"] = fallback.get("page_type")

        return data

    # -------------------------------------------------
    # JSONL / JSON save
    # -------------------------------------------------

    def save_items_json(self, items: List[Dict[str, Any]], output_json: str) -> Dict[str, Any]:
        payload = {"items": items}
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return payload

    # -------------------------------------------------
    # Full scraping pipeline
    # -------------------------------------------------

    def scrape_items(
        self,
        candidates: Dict[str, Dict[str, Any]],
        output_jsonl: str,
    ) -> List[Dict[str, Any]]:
        seen_hashes: Set[str] = set()
        seen_urls: Set[str] = set()
        items: List[Dict[str, Any]] = []
        write_lock = threading.Lock()

        try:
            open(output_jsonl, "w", encoding="utf-8").close()
        except Exception:
            pass

        max_threads = int(self.request_cfg().get("max_threads", 8))

        def worker(url: str) -> Optional[Dict[str, Any]]:
            fallback = candidates.get(url)
            return self.extract_item_from_url(url, fallback=fallback)

        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = {executor.submit(worker, u): u for u in candidates.keys()}
            for f in tqdm(as_completed(futures), total=len(futures), desc="[EXTRACT] Downloading articles"):
                try:
                    data = f.result()
                    if not data:
                        continue

                    url = data.get("url")
                    if not url:
                        continue

                    h = data.get("hash") or sha256_hex(
                        (data.get("title") or "") + "|" + (data.get("content") or "") + "|" + url,
                        trunc=32,
                    )

                    with write_lock:
                        if url in seen_urls or h in seen_hashes:
                            continue
                        seen_urls.add(url)
                        seen_hashes.add(h)

                        item = {
                            "title": data.get("title"),
                            "content": data.get("content"),
                            "url": data.get("url"),
                            "date": data.get("date"),
                            "author": data.get("author"),
                            "category": data.get("category"),
                            "time": data.get("time"),
                        }
                        items.append(item)

                        with open(output_jsonl, "a", encoding="utf-8") as out:
                            out.write(json.dumps(item, ensure_ascii=False) + "\n")

                except Exception:
                    continue

        return items

    def run_full_pipeline(
        self,
        start_url: str,
        output_jsonl: Optional[str] = None,
        output_json: Optional[str] = None,
        context_rubrics: Optional[Iterable[str]] = None,
        max_items_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        limits = self.limits_cfg()
        if max_items_override is not None:
            limits["max_items"] = max_items_override   # forced limit

        output_jsonl = output_jsonl or self.global_cfg().get("output_jsonl", "items_v6_5.jsonl")
        output_json = output_json or self.global_cfg().get("output_json", "items_v6_5.json")

        try:
            open(output_jsonl, "w", encoding="utf-8").close()
        except Exception:
            pass

        logger.info(f"Pipeline started: {start_url}")
        print(f"[START] URL: {start_url}")
        print(f"[PROFILE] Site key: {self.site_key(start_url)}")

        candidates = self.detect_page_candidates(start_url, context_rubrics=context_rubrics)
        logger.info(f"Candidates found: {len(candidates)}")
        print(f"[CANDIDATES] Found: {len(candidates)}")

        items = self.scrape_items(candidates, output_jsonl)
        logger.info(f"Unique items extracted: {len(items)}")
        print(f"[ITEMS] Unique: {len(items)}")

        payload = self.save_items_json(items, output_json)

        logger.info(f"Output JSONL: {output_jsonl}")
        logger.info(f"Output JSON: {output_json}")
        print(f"[OK] JSONL: {output_jsonl}")
        print(f"[OK] JSON:  {output_json}")
        return {
            **payload,
            "output_jsonl": output_jsonl,
            "output_json": output_json,
        }


# =====================================================
# Standalone runner
# =====================================================

def run(
    start_url: str,
    yaml_path: str = "config/universal.yaml",
    output_jsonl: Optional[str] = None,
    output_json: Optional[str] = None,
    context_rubrics: Optional[Iterable[str]] = None,
    max_items: Optional[int] = None,
) -> Dict[str, Any]:
    engine = ExtractionEngine(yaml_path=yaml_path)
    
    # Auto-generate output filenames from site_key + date
    if output_jsonl is None or output_json is None:
        site_key = engine.site_key(start_url)
        date_str = time.strftime("%Y-%m-%d")
        os.makedirs("output", exist_ok=True) 
        if output_jsonl is None:
            output_jsonl = f"output/{site_key}_{date_str}_articles.jsonl"
        if output_json is None:
            output_json = f"output/{site_key}_{date_str}_articles.json"
    
    return engine.run_full_pipeline(
        start_url=start_url,
        output_jsonl=output_jsonl,
        output_json=output_json,
        context_rubrics=context_rubrics,
        max_items_override=max_items,
    )


if __name__ == "__main__":
    import sys
    import json

    yaml_path = "config/universal.yaml"
    start = "https://khovar.tj/"
    out_jsonl = None   
    out_json = None    
    max_items = None

    if len(sys.argv) >= 2:
        yaml_path = sys.argv[1]
    if len(sys.argv) >= 3:
        start = sys.argv[2]
    if len(sys.argv) >= 4:
        out_jsonl = sys.argv[3]
    if len(sys.argv) >= 5:
        out_json = sys.argv[4]
    if len(sys.argv) >= 6:
        try:
            max_items = int(sys.argv[5])
        except ValueError:
            print("[WARN] Invalid max items format. Ignoring.")

    print(f"Config: {yaml_path}")
    print(f"URL: {start}")
    if max_items:
        print(f"Max items: {max_items}")

    result = run(
        start_url=start,
        yaml_path=yaml_path,
        output_jsonl=out_jsonl,
        output_json=out_json,
        max_items=max_items,
    )
    print(f"Output JSONL: {result.get('output_jsonl', 'N/A')}")
    print(f"Output JSON:  {result.get('output_json', 'N/A')}")