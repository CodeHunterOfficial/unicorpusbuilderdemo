# tools/_common.py
"""Shared helpers for the tools/ scripts."""
from __future__ import annotations

import os
import re
import statistics
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from datasets import load_dataset


WORD_RE = re.compile(r"\w+", re.UNICODE)


def get_token() -> str | None:
    """Return the Hugging Face token from the environment, or None."""
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        v = os.environ.get(var)
        if v:
            return v.strip()
    return None


def load_ds(repo: str, token, streaming: bool):
    """Load a dataset with two fallbacks."""
    try:
        return load_dataset(
            repo,
            token=token,
            streaming=streaming,
            verification_mode="no_checks",
        )
    except Exception as e1:
        print(f"  [WARN] {type(e1).__name__}: {str(e1)[:160]}")
        print("  [RETRY] via parquet loader...")

    return load_dataset(
        "parquet",
        data_files=f"hf://datasets/{repo}/data/*.parquet",
        token=token,
        streaming=streaming,
    )


def stats_summary(values) -> dict | None:
    """Return n / mean / median / min / max for a list, or None if empty."""
    if not values:
        return None
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def discover_urls_for_site(engine, site_key, site_cfg, limit: int) -> list[str]:
    """
    Discovers and returns up to ``limit`` article URLs from a given site.
    
    Uses the engine's YAML-driven extraction as the primary method, 
    falling back to a generic BeautifulSoup parser for sites with 
    absolute URLs or language subdomains (e.g., asiaplus.news).
    """
    match = site_cfg.get("match", [])
    start_url = site_cfg.get("start_url") or (f"https://{match[0]}" if match else None)
    if not start_url:
        return []

    urls: list[str] = []
    seen: set[str] = set()
    
    excludes = site_cfg.get("article_url_excludes", [])
    regex_raw = site_cfg.get("article_url_regex", r"/\d{4}/\d{2}/\d{2}/")
    
    if isinstance(regex_raw, list) and regex_raw:
        regex_str = str(regex_raw[0])
    elif isinstance(regex_raw, str):
        regex_str = regex_raw
    else:
        regex_str = r"/\d{4}/\d{2}/\d{2}/"

    try:
        article_re = re.compile(regex_str)
    except (re.error, TypeError):
        article_re = re.compile(r"/\d{4}/\d{2}/\d{2}/")

    def _generic_extract(html: str, base_url: str) -> list[str]:
        soup = BeautifulSoup(html, "html.parser")
        found: list[str] = []
        for a in soup.find_all("a", href=True):
            full_url = urljoin(base_url, a["href"])
            path = urlparse(full_url).path
            if any(ex in path for ex in excludes):
                continue
            if article_re.search(path) and full_url not in seen:
                seen.add(full_url)
                found.append(full_url)
        return found

    def _process_page(page_url: str):
        try:
            html = engine.fetch_html(page_url)
        except Exception:
            return
        
        try:
            items = engine.extract_listing_items_stub(html, page_url)
            for it in items:
                u = it.get("url")
                if u and u not in seen:
                    seen.add(u)
                    urls.append(u)
                    if len(urls) >= limit:
                        return
        except Exception:
            pass

        if len(urls) < limit:
            for u in _generic_extract(html, page_url):
                if u not in seen:
                    seen.add(u)
                    urls.append(u)
                    if len(urls) >= limit:
                        return

    _process_page(start_url)
    if len(urls) >= limit:
        return urls

    rubrics: list[str] = []
    try:
        rubrics = engine.collect_rubrics(start_url)
    except Exception:
        rubrics = []

    if not rubrics:
        base = start_url.rstrip("/")
        for guess in [
            f"{base}/category/",
            f"{base}/tg/category/",
            f"{base}/news/",
            f"{base}/tg/",
        ]:
            try:
                r = engine.fetch_html(guess)
                if r and len(r) > 2000:
                    rubrics.append(guess)
            except Exception:
                continue

    for rubric_url in rubrics:
        if len(urls) >= limit:
            break
        _process_page(rubric_url)

    return urls
