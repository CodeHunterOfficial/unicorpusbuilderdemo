# pipeline/pipeline_core.py
from __future__ import annotations
import threading
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
import hashlib
import json
import random
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse, urldefrag

import requests
from bs4 import BeautifulSoup, Tag
from dateutil import parser as dateparser
import warnings
from bs4 import XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

from config.loader import get_config, load_yaml_config, match_site
from logger_setup import get_file_logger

# Initialize file-only logger
logger = get_file_logger("pipeline_core", "logs/pipeline_core.log")

# =====================================================
# Utilities
# =====================================================

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def clean_text(text: Any) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text), flags=re.UNICODE).strip()


def abs_url(base: str, href: Optional[str]) -> Optional[str]:
    if not href:
        return None
    href = href.strip()
    if href.startswith(("javascript:", "mailto:", "tel:", "#")):
        return None
    return urldefrag(urljoin(base, href))[0]


def get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().strip()
    except Exception:
        return ""


def get_path(url: str) -> str:
    try:
        return urlparse(url).path or ""
    except Exception:
        return ""


def same_domain(url1: str, url2: str) -> bool:
    return get_domain(url1) == get_domain(url2)


def is_valid_http_url(url: Optional[str]) -> bool:
    return bool(url) and url.startswith(("http://", "https://"))


def sha256_hex(text: str, trunc: Optional[int] = 32) -> str:
    h = hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()
    return h[:trunc] if trunc else h


def apply_date_locale(date_str: str, locale_map: Optional[Dict[str, str]]) -> str:
    """Replace non-standard month names with English equivalents."""
    if not locale_map:
        return date_str
    for tg, en in locale_map.items():
        date_str = date_str.replace(tg, en)
    return date_str


def parse_datetime_value(
    value: Optional[str],
    locale_map: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    if not value:
        return None
    value = clean_text(value)
    if not value:
        return None
    # Apply locale mapping before parsing
    value = apply_date_locale(value, locale_map)
    try:
        dt = dateparser.parse(value, fuzzy=True)
        if dt:
            return dt.isoformat()
    except Exception:
        pass
    return value


def extract_time_part(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    m = re.search(r"\b(\d{1,2}:\d{2})\b", value)
    return m.group(1) if m else None


def is_noise_url(url: Optional[str], noise_parts: Optional[List[str]] = None) -> bool:
    if not url:
        return True
    low = url.lower()
    if noise_parts is None:
        noise_parts = [
            "/photo/", "/video/", "/special-projects/", "/widget/",
            "/tag/list/", "vkvideo", "youtube", "youtu.be",
            "instagram", "facebook", "telegram", "ok.ru",
            "/login", "/signup", "/signin", "/register",
        ]
    return any(p in low for p in noise_parts)


def is_noise_text(text: str, noise_words: Optional[List[str]] = None) -> bool:
    t = clean_text(text).lower()
    if len(t) < 3:
        return True
    if noise_words is None:
        noise_words = []        
    return any(x in t for x in noise_words)

def ensure_list(v: Any) -> List[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, tuple):
        return list(v)
    return [v]


def uniq_keep_order(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in items:
        if not x:
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# =====================================================
# Core Engine
# =====================================================

@dataclass
class PipelineEngine:
    yaml_path: str = "config/universal.yaml"
    root_cfg: Dict[str, Any] = field(default_factory=dict)
    session_pool: threading.local = field(default_factory=threading.local, init=False)

    def __post_init__(self) -> None:
        if not self.root_cfg:
            self.root_cfg = load_yaml_config(self.yaml_path)

    # -------------------------------------------------
    # Config helpers
    # -------------------------------------------------

    def global_cfg(self) -> Dict[str, Any]:
        return self.root_cfg.get("global", {}) if isinstance(self.root_cfg, dict) else {}

    def limits_cfg(self) -> Dict[str, Any]:
        g = self.global_cfg()
        return g.get("limits", {}) if isinstance(g, dict) else {}

    def request_cfg(self) -> Dict[str, Any]:
        g = self.global_cfg()
        return g.get("request", {}) if isinstance(g, dict) else {}

    def site_key(self, url: str) -> str:
        return match_site(url, self.root_cfg)

    def site_cfg(self, url: str) -> Dict[str, Any]:
        return get_config(url, self.root_cfg)

    def domain_only(self) -> bool:
        return bool(self.global_cfg().get("same_domain_only", True))

    def use_amp_fallback(self) -> bool:
        return bool(self.global_cfg().get("use_amp_fallback", True))

    # -------------------------------------------------
    # Date locale support
    # -------------------------------------------------

    def get_date_locale_map(self, url: str) -> Optional[Dict[str, str]]:
        site_cfg = self.site_cfg(url)
        locale_name = site_cfg.get("date_locale")
        if not locale_name:
            return None
        
        # New modular path: config.languages.<lang>.date_locale
        languages = self.root_cfg.get("languages", {})
        if isinstance(languages, dict) and locale_name in languages:
            lang_pkg = languages[locale_name]
            if isinstance(lang_pkg, dict):
                return lang_pkg.get("date_locale")
        
        # Fallback: old monolithic path reusable_strategies.date_locale
        reusable = self.root_cfg.get("reusable_strategies", {})
        date_locales = reusable.get("date_locale", {})
        if isinstance(date_locales, dict):
            result = date_locales.get(locale_name)
            if result:
                return result
        
        return None

    # -------------------------------------------------
    # HTTP Session
    # -------------------------------------------------

    def _build_headers_and_cookies(self, url: str) -> Tuple[Dict[str, str], Dict[str, str]]:
        g = self.global_cfg()
        site_cfg = self.site_cfg(url)
        contact = g.get("contact_email", "researcher@example.com")
        ua = f"UniversalJSONLScraper/6.5 (+{contact})"
        headers = {
            "User-Agent": ua,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8,tg;q=0.7,uz;q=0.7,ba;q=0.7",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "DNT": "1",
        }
        cookies = {}
        lang = site_cfg.get("cookie_lang")
        if lang:
            cookies["lang"] = str(lang)
        return headers, cookies

    def _session(self) -> requests.Session:
        sess = getattr(self.session_pool, "session", None)
        if sess is None:
            sess = requests.Session()
            self.session_pool.session = sess
        return sess

    def fetch_html(self, url: str) -> str:
        site_cfg = self.site_cfg(url)
        req = self.request_cfg()
        timeout = int(req.get("timeout", 25))
        retries = int(req.get("retries", 3))
        backoff_base = float(req.get("backoff_base", 1.6))
        # verify_ssl from site config (already resolved by loader)
        verify_ssl = site_cfg.get("verify_ssl", True)

        sess = self._session()
        headers, cookies = self._build_headers_and_cookies(url)
        last_exc: Optional[Exception] = None

        for attempt in range(retries):
            try:
                r = sess.get(
                    url,
                    timeout=timeout,
                    allow_redirects=True,
                    headers=headers,
                    cookies=cookies,
                    verify=verify_ssl,
                )
                r.raise_for_status()
                return r.text
            except Exception as e:
                last_exc = e
                delay = (backoff_base ** attempt) + random.uniform(0.15, 0.9)
                time.sleep(delay)

        if last_exc:
            raise last_exc
        raise RuntimeError(f"Failed to fetch URL: {url}")

    def fetch_api_json(self, url: str, site_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Execute GET request to API and return JSON."""
        api_cfg = site_cfg.get('api_source', {})
        timeout = api_cfg.get('timeout', self.request_cfg().get('timeout', 30))
        retries = api_cfg.get('retries', self.request_cfg().get('retries', 3))
        backoff_base = self.request_cfg().get('backoff_base', 1.6)
        
        sess = self._session()
        headers, cookies = self._build_headers_and_cookies(url)
        # Add API-specific headers if specified in config
        extra_headers = api_cfg.get('headers', {})
        headers.update(extra_headers)
        
        for attempt in range(retries):
            try:
                r = sess.get(url, headers=headers, cookies=cookies,
                            timeout=timeout, verify=site_cfg.get('verify_ssl', True))
                r.raise_for_status()
                return r.json()
            except Exception as e:
                if attempt == retries - 1:
                    raise
                time.sleep(backoff_base ** attempt + random.uniform(0.1, 0.5))
        return {}

    def can_fetch_robots(self, url: str) -> bool:
        if not bool(self.global_cfg().get("respect_robots", False)):
            return True
        try:
            import urllib.robotparser as robotparser

            parsed = urlparse(url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            rp = robotparser.RobotFileParser()
            rp.set_url(robots_url)
            rp.read()
            return rp.can_fetch("*", url)
        except Exception:
            return True

    # -------------------------------------------------
    # Safe selectors with :contains() support
    # -------------------------------------------------

    def _safe_select(self, soup, css_selector: str) -> List[Tag]:
        """Safe select with :contains() support for BeautifulSoup."""
        if ':contains(' in css_selector:
            # Extract text to search for
            match = re.search(r":contains\('([^']*)'\)", css_selector)
            if not match:
                return []
            text = match.group(1)
            # Remove :contains() and everything after, leaving clean CSS
            clean_selector = re.sub(r":contains\('[^']*'\)", '', css_selector).strip()
            if clean_selector:
                try:
                    elements = soup.select(clean_selector)
                except Exception:
                    elements = []
                # Filter by text
                return [el for el in elements if isinstance(el, Tag) and text in el.get_text()]
            else:
                # Find all tags with the needed text
                return [el for el in soup.find_all(True) if isinstance(el, Tag) and text in el.get_text()]
        else:
            try:
                return soup.select(css_selector)
            except Exception:
                return []

    def _safe_select_one(self, soup, css_selector: str) -> Optional[Tag]:
        results = self._safe_select(soup, css_selector)
        return results[0] if results else None

    # -------------------------------------------------
    # Page type detection
    # -------------------------------------------------

    def detect_page_type(self, soup: BeautifulSoup, url: str) -> str:
        site_cfg = self.site_cfg(url)
        low = url.lower()
        scores = defaultdict(float)

        # URL heuristics
        if any(x in low for x in ["/news/", "/article/", "/story/", "/post/", "/a/"]):
            scores["article"] += 0.25
        if any(x in low for x in ["/rubric/", "/category/", "/tag/", "/section/", "/archive/", "/page/"]):
            scores["listing"] += 0.25
        if get_path(url) in ("", "/"):
            scores["listing"] += 0.10

        # HTML heuristics
        if soup.select_one("article"):
            scores["article"] += 0.35
        if soup.select_one('meta[property="og:type"][content="article"]'):
            scores["article"] += 0.35
        if soup.select_one("h1"):
            scores["article"] += 0.10

        # Card counting
        cards = 0
        for sel in site_cfg.get("card_selectors", []):
            try:
                cards += len(self._safe_select(soup, sel))
            except Exception:
                pass
        if cards >= 3:
            scores["listing"] += 0.35

        links = len(soup.find_all("a", href=True))
        paras = len(soup.find_all("p"))
        if links > 30 and paras < 25:
            scores["listing"] += 0.35

        # Pagination signals
        if (
            soup.select_one("a[rel='next']")
            or soup.select_one("link[rel='next']")
            or soup.select_one(".pagination")
            or soup.select_one(".pager")
        ):
            scores["listing"] += 0.10

        best = "unknown"
        best_score = 0.0
        for k, v in scores.items():
            if v > best_score:
                best = k
                best_score = v

        if best == "article" and best_score < 0.35:
            best = "listing"
        return best

    # -------------------------------------------------
    # Pagination
    # -------------------------------------------------

    def get_next_page_url(self, soup: BeautifulSoup, current_url: str) -> Optional[str]:
        site_cfg = self.site_cfg(current_url)

        for sel in site_cfg.get("pagination_selectors", []):
            try:
                for el in self._safe_select(soup, sel):
                    href = el.get("href")
                    if href:
                        full = abs_url(current_url, href)
                        if full:
                            return full
            except Exception:
                pass

        # Additional text-based detection
        next_texts = ["Next", "→", ">"]
        for a in soup.find_all("a", href=True):
            txt = clean_text(a.get_text(" ", strip=True))
            if txt and any(nt.lower() in txt.lower() for nt in next_texts):
                full = abs_url(current_url, a.get("href"))
                if full:
                    return full

        # Query-parameter based pagination
        parsed = urlparse(current_url)
        qs = parse_qs(parsed.query)

        if "p" in qs:
            try:
                cur = int(qs["p"][0])
                qs["p"] = [str(cur + 1)]
                return urlunparse(parsed._replace(query=urlencode(qs, doseq=True)))
            except Exception:
                pass

        if "page" in qs:
            try:
                cur = int(qs["page"][0])
                qs["page"] = [str(cur + 1)]
                return urlunparse(parsed._replace(query=urlencode(qs, doseq=True)))
            except Exception:
                pass

        # Path-based pagination
        m = re.search(r"/page/(\d+)", parsed.path)
        if m:
            cur = int(m.group(1))
            new_path = re.sub(r"/page/\d+", f"/page/{cur + 1}", parsed.path)
            return urlunparse(parsed._replace(path=new_path))


        # If URL doesn't have p or page param yet, look for any link with ?p= on the page
        parsed = urlparse(current_url)
        qs = parse_qs(parsed.query)
        if "p" not in qs and "page" not in qs:
            for a in soup.find_all("a", href=True):
                href = a.get("href")
                match = re.search(r"[?&]p=(\d+)", href)
                if match:
                    next_num = int(match.group(1)) + 1
                    new_qs = qs.copy()
                    new_qs["p"] = [str(next_num)]
                    return urlunparse(parsed._replace(query=urlencode(new_qs, doseq=True)))

        return None

    # -------------------------------------------------
    # Rubric collection
    # -------------------------------------------------

    def _rubrics_menu_scraping(self, base_url: str, soup: BeautifulSoup, site_cfg: Dict[str, Any]) -> List[str]:
        out: List[str] = []
        seen: Set[str] = set()
        for sel in site_cfg.get("rubric_selectors", []):
            try:
                for a in self._safe_select(soup, sel):
                    href = a.get("href") if isinstance(a, Tag) else None
                    full = abs_url(base_url, href)
                    if not full:
                        continue
                    if full not in seen:
                        seen.add(full)
                        out.append(full)
            except Exception:
                pass
        return out

    def _rubrics_pattern_fallback(self, base_url: str, soup: BeautifulSoup, site_cfg: Dict[str, Any]) -> List[str]:
        out: List[str] = []
        seen: Set[str] = set()
        filters = ensure_list(site_cfg.get("article_url_filters", []))
        excludes = ensure_list(site_cfg.get("article_url_excludes", []))

        for a in soup.find_all("a", href=True):
            full = abs_url(base_url, a.get("href"))
            if not full:
                continue
            if is_noise_url(full):
                continue
            low = full.lower()
            if excludes and any(x.lower() in low for x in excludes):
                continue
            if filters and not any(x.lower() in low for x in filters):
                continue
            if full not in seen:
                seen.add(full)
                out.append(full)
        return out

    def _rubrics_archive_generation(self, base_url: str, site_cfg: Dict[str, Any]) -> List[str]:
        limits = self.limits_cfg()
        years_back = int(limits.get("archive_years_back", 8))
        parsed = urlparse(base_url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        current_year = time.localtime().tm_year
        out: List[str] = []

        patterns = site_cfg.get("archive_patterns") or [
            "{base_url}/{year}/",
            "{base_url}/{year}/{month:02d}/",
            "{base_url}/{year}/{month:02d}/page/1/",
        ]

        for year in range(current_year - years_back, current_year + 1):
            for month in range(1, 13):
                for pat in patterns:
                    try:
                        out.append(pat.format(base_url=base, year=year, month=month))
                    except Exception:
                        continue
        return out

    def _rubrics_nuxt_state(self, base_url: str, html: str, site_cfg: Dict[str, Any]) -> List[str]:
        regexes = site_cfg.get("nuxt_url_regexes") or [
            r'https?://[^"\'\s<>]+',
            r'/(?:news|rubric|category|tag|article|post|a)/[^"\'\s<>]*',
        ]
        out: List[str] = []
        for pat in regexes:
            try:
                for m in re.findall(pat, html, flags=re.I | re.M):
                    full = abs_url(base_url, m)
                    if full:
                        out.append(full)
            except Exception:
                pass

        for key in site_cfg.get("nuxt_json_keys", ["path", "href", "url"]):
            pat = rf'"{re.escape(key)}"\s*:\s*"([^"]+)"'
            try:
                for m in re.findall(pat, html, flags=re.I):
                    full = abs_url(base_url, m)
                    if full:
                        out.append(full)
            except Exception:
                pass

        return uniq_keep_order(out)

    def _rubrics_auto_detect_multi(self, base_url: str, soup: BeautifulSoup, html: str, site_cfg: Dict[str, Any]) -> List[str]:
        candidates: List[str] = []
        candidates.extend(self._rubrics_menu_scraping(base_url, soup, site_cfg))
        if len(candidates) < 3:
            candidates.extend(self._rubrics_pattern_fallback(base_url, soup, site_cfg))
        if len(candidates) < 3:
            candidates.extend(self._rubrics_nuxt_state(base_url, html, site_cfg))
        return uniq_keep_order(candidates)

    def _extract_api_items(self, payload: dict, site_cfg: dict) -> list:
        """Parameterized extraction of news list from API JSON response."""
        api_cfg = site_cfg['api_source']
        items_path = api_cfg.get('items_path', 'data')
        
        # Find the container with items
        container = payload
        for part in items_path.split('.'):
            if isinstance(container, dict):
                container = container.get(part, [])
            else:
                container = []
        if not isinstance(container, list):
            return []
        
        fm = api_cfg.get('field_map', {})
        nested = api_cfg.get('nested_extraction', {})
        url_template = fm.get('url_template')
        site_url = api_cfg.get('site_url', '')
        
        items = []
        for obj in container:
            item = {}
            item['id'] = self._get_nested_value(obj, fm.get('id'))
            item['title'] = self._get_nested_value(obj, fm.get('title'))
            
            # Date: try multiple fields
            date_val = None
            for date_field in ensure_list(fm.get('date', [])):
                date_val = self._get_nested_value(obj, date_field)
                if date_val:
                    break
            item['date'] = date_val
            
            item['content_html'] = self._get_nested_value(obj, fm.get('content_html'))
            
            # Generate article URL
            if url_template:
                item['url'] = url_template.format(site_url=site_url, id=item.get('id'))
            else:
                item['url'] = None
            
            # Nested lists (images, categories, cities)
            for key, rule in nested.items():
                path = rule.get('path')
                sub_key = rule.get('key')
                sub_container = self._get_nested_value(obj, path) or []
                values = []
                for sub in ensure_list(sub_container):
                    val = self._get_nested_value(sub, sub_key) if sub_key else sub
                    if val:
                        values.append(val)
                item[key] = values
            
            # Keep raw data
            item['raw'] = obj
            items.append(item)
        
        return items

    def collect_rubrics(self, base_url: str, context_rubrics: Optional[Iterable[str]] = None) -> List[str]:
        """
        Universal rubric (category/section) collection for any site from config.
        """               
        if context_rubrics:
            out: List[str] = []
            for r in context_rubrics:
                full = abs_url(base_url, r)
                if full:
                    out.append(full)
            return uniq_keep_order(out)

        site_cfg = self.site_cfg(base_url)
        if site_cfg.get('api_source', {}).get('enabled'):
            return []          
        strategies = site_cfg.get("rubric_strategy") or self.global_cfg().get(
            "rubric_strategies_order"
        ) or ["menu_scraping"]

        # Load main page
        try:
            html = self.fetch_html(base_url)
        except Exception:
            return []
        soup = BeautifulSoup(html, "html.parser")

        all_rubrics: List[str] = []

        # Process strategies
        for strat in strategies:
            if strat == "menu_scraping":
                # Exact selectors from site profile
                found_any = False
                for sel in site_cfg.get("rubric_selectors", []):
                    try:
                        for a in self._safe_select(soup, sel):
                            href = a.get("href") if hasattr(a, "get") else None
                            full = abs_url(base_url, href)
                            if full:
                                all_rubrics.append(full)
                                found_any = True
                    except Exception:
                        pass

                # If exact selectors weren't enough (< 3), enable heuristics
                if not found_any or len(set(all_rubrics)) < 3:
                    # Traverse all common navigation blocks
                    for container_sel in [
                        "nav", "header", ".menu", ".nav", ".navigation",
                        "ul.menu", "ul.nav", "div.nav", "div.menu",
                        "ul.main-nav__list", "ul.navbar-nav", "ul.nav__list",
                        "div.main-menu", "ul.the-menu", "ul#main-menu",
                        "ul#menu-primary-menu", "div.np-widget-rubric",
                    ]:
                        for container in soup.select(container_sel):
                            for a_tag in container.find_all("a", href=True):
                                href = a_tag.get("href")
                                if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
                                    continue
                                full = abs_url(base_url, href)
                                if full:
                                    all_rubrics.append(full)

            elif strat == "pattern_fallback":
                all_rubrics.extend(self._rubrics_pattern_fallback(base_url, soup, site_cfg))

            elif strat == "archive_generation":
                all_rubrics.extend(self._rubrics_archive_generation(base_url, site_cfg))

            elif strat == "nuxt_state_parsing":
                all_rubrics.extend(self._rubrics_nuxt_state(base_url, html, site_cfg))

            elif strat == "auto_detect_multi":
                all_rubrics.extend(self._rubrics_auto_detect_multi(base_url, soup, html, site_cfg))

            elif strat == "context_passed":
                # already handled above
                continue

        # Cleanup: domains, duplicates, noise
        same_domain_only = self.domain_only()
        uniq: List[str] = []
        seen: Set[str] = set()
        for u in all_rubrics:
            if not u or not is_valid_http_url(u):
                continue
            if same_domain_only and not same_domain(base_url, u):
                continue
            if is_noise_url(u):
                continue
            if u not in seen:
                seen.add(u)
                uniq.append(u)

        # Debug output
        logger.info(f"Collected {len(uniq)} rubrics for {base_url}")
        print(f"[RUBRICS] Found: {len(uniq)}")
        for r in uniq[:10]:
            print(f"  {r}")
        if len(uniq) > 10:
            print(f"  ... and {len(uniq)-10} more")

        return uniq

    # -------------------------------------------------
    # Candidate discovery
    # -------------------------------------------------

    def extract_page_meta_stub(self, soup: BeautifulSoup, url: str, site_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Lightweight metadata extraction for listing pages."""
        title = None
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
                    title = val
                    break

        if not title and soup.title and soup.title.string:
            title = clean_text(soup.title.string)

        locale_map = self.get_date_locale_map(url)
        date = None
        for sel in site_cfg.get("date_selectors", []):
            try:
                el = self._safe_select_one(soup, sel)
            except Exception:
                el = None
            if el:
                val = el.get("content") or el.get("datetime") or el.get_text(" ", strip=True)
                if val:
                    date = parse_datetime_value(val, locale_map)
                    if date:
                        break

        return {
            "title": title,
            "date": date,
            "author": None,
            "category": None,
            "image_url": None,
            "language": site_cfg.get("default_language"),
        }

    def _should_exclude_card(self, card: Tag, exclude_classes: List[str]) -> bool:
        if not exclude_classes:
            return False
        card_classes = card.get("class", [])
        if not isinstance(card_classes, list):
            return False
        return any(excl in card_classes for excl in exclude_classes)

    def extract_listing_items_stub(self, html: str, page_url: str) -> List[Dict[str, Any]]:
        soup = BeautifulSoup(html, "html.parser")
        site_cfg = self.site_cfg(page_url)
        locale_map = self.get_date_locale_map(page_url)
        page_meta = self.extract_page_meta_stub(soup, page_url, site_cfg)

        items: List[Dict[str, Any]] = []
        seen_urls: Set[str] = set()
        same_domain_only = self.domain_only()
        exclude_classes = site_cfg.get("card_exclude_classes", [])

        for sel in site_cfg.get("card_selectors", []):
            for card in self._safe_select(soup, sel):
                if not isinstance(card, Tag):
                    continue
                try:
                    if self._should_exclude_card(card, exclude_classes):
                        continue

                    url = None
                    link_node = None
                    for link_sel in site_cfg.get("article_link_selectors", []):
                        link_node = self._safe_select_one(card, link_sel)
                        if link_node and link_node.get("href"):
                            url = abs_url(page_url, link_node.get("href"))
                            if url:
                                break
                    if not url:
                        a = card.find("a", href=True)
                        if a:
                            url = abs_url(page_url, a.get("href"))
                            link_node = a

                    if not url or not is_valid_http_url(url):
                        continue
                    if same_domain_only and not same_domain(page_url, url):
                        continue
                    if is_noise_url(url):
                        continue

                    # Apply article URL filters / excludes
                    low = url.lower()
                    if site_cfg.get("article_url_excludes") and any(x.lower() in low for x in site_cfg.get("article_url_excludes", [])):
                        continue
                    if site_cfg.get("article_url_filters") and not any(x.lower() in low for x in site_cfg.get("article_url_filters", [])):
                        if not any(x in low for x in ["/news/", "/article/", "/story/", "/post/", "/a/"]):
                            continue

                    if url in seen_urls:
                        continue
                    seen_urls.add(url)

                    title = None
                    if link_node:
                        title = clean_text(link_node.get_text(" ", strip=True))
                        noise_words = site_cfg.get("_noise_words", [])
                        if title and is_noise_text(title, noise_words):
                            continue 
                    if not title:
                        for tag_name in ["h1", "h2", "h3", "h4", "a"]:
                            n = card.find(tag_name)
                            if n:
                                t = clean_text(n.get_text(" ", strip=True))
                                if t:
                                    title = t
                                    break

                    snippet = ""
                    for p in card.find_all(["p", "div", "span"]):
                        txt = clean_text(p.get_text(" ", strip=True))
                        if txt and len(txt) > len(snippet):
                            snippet = txt

                    raw_time = None
                    for node in card.find_all(["time", "span", "a"]):
                        txt = clean_text(node.get_text(" ", strip=True))
                        if re.search(r"\b\d{1,2}:\d{2}\b", txt) or re.search(r"\d{1,2}\s+[A-Za-z]+\s+\d{4}", txt):
                            raw_time = txt
                            break

                    item = {
                        "url": url,
                        "title": title or page_meta.get("title"),
                        "content": snippet or title or page_meta.get("title"),
                        "date": parse_datetime_value(raw_time, locale_map) if raw_time else page_meta.get("date"),
                        "author": page_meta.get("author"),
                        "category": page_meta.get("category"),
                        "time": extract_time_part(raw_time or page_meta.get("date") or ""),
                        "site": get_domain(url),
                        "hash": sha256_hex((title or "") + "|" + (snippet or "") + "|" + url, trunc=32),
                        "scraped_at": now_iso(),
                        "source_page": page_url,
                        "page_type": "listing_item",
                    }
                    items.append(item)
                except Exception as e:
                    logger.error(f"Error processing card: {e}")
                    continue

        # Anchors fallback
        if len(items) < 3:
            for a in soup.find_all("a", href=True):
                url = abs_url(page_url, a.get("href"))
                if not url or not is_valid_http_url(url):
                    continue
                if same_domain_only and not same_domain(page_url, url):
                    continue
                if is_noise_url(url):
                    continue

                low = url.lower()
                if site_cfg.get("article_url_excludes") and any(x.lower() in low for x in site_cfg.get("article_url_excludes", [])):
                    continue
                if site_cfg.get("article_url_filters") and not any(x.lower() in low for x in site_cfg.get("article_url_filters", [])):
                    if not any(x in low for x in ["/news/", "/article/", "/story/", "/post/", "/a/"]):
                        continue

                if url in seen_urls:
                    continue
                txt = clean_text(a.get_text(" ", strip=True))                
                if is_noise_text(txt, site_cfg.get("_noise_words", [])):
                    continue
                if len(txt) < 8:
                    continue
                seen_urls.add(url)
                items.append(
                    {
                        "url": url,
                        "title": txt or page_meta.get("title"),
                        "content": txt or page_meta.get("title"),
                        "date": page_meta.get("date"),
                        "author": page_meta.get("author"),
                        "category": page_meta.get("category"),
                        "time": extract_time_part(page_meta.get("date") or txt),
                        "site": get_domain(url),
                        "hash": sha256_hex((txt or "") + "|" + url, trunc=32),
                        "scraped_at": now_iso(),
                        "source_page": page_url,
                        "page_type": "listing_item",
                    }
                )
        return items

    def detect_page_candidates(
        self,
        start_url: str,
        context_rubrics: Optional[Iterable[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        site_cfg_main = self.site_cfg(start_url)
        if site_cfg_main.get('api_source', {}).get('enabled'):
            return self.detect_page_candidates_api(
                start_url, site_cfg_main
            )
        
        limits = self.limits_cfg()
        max_pages = int(limits.get("max_pages", 250))
        max_depth = int(limits.get("max_depth", 3))
        max_items = int(limits.get("max_items", 10000))

        queue: Deque[Tuple[str, int]] = deque()
        visited_pages: Set[str] = set()
        discovered: Dict[str, Dict[str, Any]] = {}

        queue.append((start_url, 0))

        rubric_links = []
        try:
            rubric_links = self.collect_rubrics(start_url, context_rubrics=context_rubrics)
        except Exception:
            rubric_links = []

        for r in rubric_links:
            queue.append((r, 1))

        logger.info(f"Starting crawl: {start_url} (max_pages={max_pages}, max_depth={max_depth}, max_items={max_items})")

        pbar = tqdm(
            total=max_pages,
            desc="[CRAWL] Page traversal",
            unit="page",
            dynamic_ncols=True,
        )

        while queue and len(visited_pages) < max_pages and len(discovered) < max_items:
            current_url, depth = queue.popleft()
            current_url = urldefrag(current_url)[0]

            if current_url in visited_pages:
                continue
            if depth > max_depth:
                continue
            if not is_valid_http_url(current_url):
                continue
            if self.domain_only() and not same_domain(start_url, current_url):
                continue
            if not self.can_fetch_robots(current_url):
                continue

            visited_pages.add(current_url)
            pbar.update(1)
            pbar.set_postfix_str(
                f"candidates: {len(discovered)}, depth: {depth}"
            )

            site_cfg = self.site_cfg(current_url)

            # Try to load HTML
            try:
                html = self.fetch_html(current_url)
            except Exception:
                continue

            # Safely create soup – skip page if invalid or binary
            try:
                soup = BeautifulSoup(html, "html.parser")
            except Exception:
                continue

            page_type = self.detect_page_type(soup, current_url)
            page_meta = self.extract_page_meta_stub(soup, current_url, site_cfg)

            # listing/homepage: discover from cards
            if page_type in ("listing", "unknown"):
                try:
                    listing_items = self.extract_listing_items_stub(html, current_url)
                except Exception:
                    listing_items = []
                for item in listing_items:
                    u = item.get("url")
                    if u and u not in discovered:
                        discovered[u] = item

            # Every visible article-like link
            for a in soup.find_all("a", href=True):
                href = abs_url(current_url, a.get("href"))
                if not href:
                    continue
                if self.domain_only() and not same_domain(start_url, href):
                    continue
                if is_noise_url(href):
                    continue

                low = href.lower()
                excluded = site_cfg.get("article_url_excludes", [])
                if excluded and any(x.lower() in low for x in excluded):
                    continue

                txt = clean_text(a.get_text(" ", strip=True))
                if is_noise_text(txt, site_cfg.get("_noise_words", [])):
                    continue

                article_like = False
                if site_cfg.get("article_url_filters"):
                    article_like = any(x.lower() in low for x in site_cfg.get("article_url_filters", []))
                if not article_like:
                    article_like = any(x in low for x in ["/news/", "/article/", "/story/", "/post/", "/a/"])

                if article_like and href not in discovered:
                    discovered[href] = {
                        "url": href,
                        "title": txt or page_meta.get("title"),
                        "content": txt or page_meta.get("title"),
                        "date": page_meta.get("date"),
                        "author": page_meta.get("author"),
                        "category": page_meta.get("category"),
                        "time": extract_time_part(page_meta.get("date") or txt),
                        "site": get_domain(href),
                        "hash": sha256_hex((txt or "") + "|" + href, trunc=32),
                        "scraped_at": now_iso(),
                        "source_page": current_url,
                        "page_type": "article_candidate",
                    }

            # next page
            next_url = self.get_next_page_url(soup, current_url)
            if next_url and next_url not in visited_pages:
                queue.append((next_url, depth))

            # rubric expansion
            for a in soup.find_all("a", href=True):
                href = abs_url(current_url, a.get("href"))
                if not href:
                    continue
                if self.domain_only() and not same_domain(start_url, href):
                    continue
                if is_noise_url(href):
                    continue
                if href in visited_pages:
                    continue

                low = href.lower()
                is_rubricish = any(x in low for x in ["/rubric/", "/category/", "/tag/", "/section/", "/archive/"])
                if is_rubricish and href not in [u for u, _ in queue]:
                    queue.append((href, depth))

            delay = random.uniform(
                float(self.request_cfg().get("default_delay_seconds", 1.2)),
                float(self.request_cfg().get("default_delay_seconds", 1.2)) + 0.8,
            )
            time.sleep(delay)

        pbar.close()
        logger.info(f"Crawl finished: {len(discovered)} candidates found")
        return discovered

    def detect_page_candidates_api(
        self,
        start_url: str,
        site_cfg: Dict[str, Any],
        max_items: int = None
    ) -> Dict[str, Dict[str, Any]]:
        
        api_cfg = site_cfg['api_source']
        last_page = self._get_api_last_page(site_cfg)
        delay = api_cfg.get('delay', 0.15)
        discovered = {}
        
        max_items = max_items or self.limits_cfg().get('max_items', 10000)
        logger.info(f"Starting API crawl: last_page={last_page}, max_items={max_items}")
        
        for page in range(1, last_page + 1):
            if len(discovered) >= max_items:
                break
            url = self._make_api_page_url(site_cfg, page)
            try:
                data = self.fetch_api_json(url, site_cfg)
                items = self._extract_api_items(data, site_cfg)
                
                for item in items:
                    article_url = item.get('url')
                    if not article_url:
                        continue
                    # Convert to candidate format
                    discovered[article_url] = {
                        'url': article_url,
                        'title': item.get('title'),
                        'content': item.get('content_html', ''),
                        'date': item.get('date'),
                        'author': site_cfg.get('default_author'),
                        'category': ', '.join(item.get('categories', [])),
                        'images': item.get('images', []),
                        'raw': item.get('raw'),
                        'scraped_at': now_iso(),
                        'source_page': url,
                        'page_type': 'api_candidate',
                    }
                    if len(discovered) >= max_items:
                        break
            except Exception as e:
                logger.error(f"Error on page {page}: {e}")
                print(f"[ERROR] Page {page}: {e}")
                continue
            
            time.sleep(delay)
        
        logger.info(f"API crawl finished: {len(discovered)} candidates")
        return discovered
    
    # -------------------------------------------------
    # Convenience API
    # -------------------------------------------------

    def discover(self, start_url: str, context_rubrics: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, Any]]:
        return self.detect_page_candidates(start_url, context_rubrics=context_rubrics)

    def stats(self, start_url: str, context_rubrics: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        cands = self.detect_page_candidates(start_url, context_rubrics=context_rubrics)
        return {
            "start_url": start_url,
            "site_key": self.site_key(start_url),
            "candidates": len(cands),
            "domain": get_domain(start_url),
        }

    def _get_nested_value(self, obj: Any, path_str: str) -> Any:
        """Extract value by dot-separated path: 'news_type.name'."""
        if not obj or not path_str:
            return None
        parts = path_str.split('.')
        current = obj
        for p in parts:
            if isinstance(current, dict):
                current = current.get(p)
            else:
                return None
        return current

    def _get_api_last_page(self, site_cfg: Dict[str, Any]) -> int:
        api_cfg = site_cfg['api_source']

        # Simple fixed limit
        if 'fixed_last_page' in api_cfg:
            return int(api_cfg['fixed_last_page'])

        # Flexible configuration via last_page_detection
        detection = api_cfg.get('last_page_detection', {})
        mode = detection.get('mode', 'fixed')
        if mode == 'fixed':
            return detection.get('value', 1)

        # Auto-detection by binary search
        start = detection.get('initial_page', 1)
        max_probe = detection.get('max_probe', 1000000)

        def nonempty(page):
            try:
                url = self._make_api_page_url(site_cfg, page)
                data = self.fetch_api_json(url, site_cfg)
                items = self._extract_api_items(data, site_cfg)
                return len(items) > 0
            except Exception:
                return False

        if not nonempty(start):
            return start - 1 if start > 1 else 0

        lo = start
        hi = start * 2
        while hi <= max_probe and nonempty(hi):
            lo = hi
            hi *= 2
            time.sleep(api_cfg.get('delay', 0.1))

        left, right = lo, hi
        while left + 1 < right:
            mid = (left + right) // 2
            if nonempty(mid):
                left = mid
            else:
                right = mid
            time.sleep(api_cfg.get('delay', 0.1))
        return left

    def _make_api_page_url(self, site_cfg: Dict[str, Any], page: int) -> str:
        api_cfg = site_cfg['api_source']
        base = api_cfg['base_api_url'].rstrip('/')
        endpoint = api_cfg['list_endpoint'].format(page=page)
        return base + endpoint if endpoint.startswith('/') else base + '/' + endpoint


# =====================================================
# Smoke test
# =====================================================

if __name__ == "__main__":
    import sys
    import json

    # Default values
    yaml_path = "config/universal.yaml"
    start = "https://khovar.tj"

    # Override from arguments if passed
    if len(sys.argv) >= 2:
        yaml_path = sys.argv[1]
    if len(sys.argv) >= 3:
        start = sys.argv[2]

    logger.info(f"Smoke test: config={yaml_path}, url={start}")
    engine = PipelineEngine(yaml_path=yaml_path)
    print(json.dumps(engine.stats(start), ensure_ascii=False, indent=2))