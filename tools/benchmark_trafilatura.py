# tools/benchmark_trafilatura.py
"""
Benchmark UniCorpusBuilder (UCB) against Trafilatura on identical article URLs.

Compared fields
---------------
    date, author, title, category (both systems), body length

Body-extraction quality is scored against a gold text using ROUGE-1,
ROUGE-L, word-level precision/recall/F1, and length ratio.

ROUGE tokenizer
---------------
``rouge_score.RougeScorer`` accepts a ``tokenizer`` object and calls
``tokenizer.tokenize(text)``. The default ``tokenize.tokenize`` keeps
only ``[a-z0-9]``, so ROUGE returns 0 on identical Cyrillic text. This
module passes ``_UnicodeTokenizer``, a subclass of
``rouge_score.tokenizers.Tokenizer`` whose ``tokenize`` splits on
``\\w+`` under ``re.UNICODE`` (see ``rouge_score/tokenizers.py``,
Google Research, 2026, lines 13-17).

Date parsing
------------
``dateutil.parser.parse`` raises ``UnknownTimezoneWarning`` when it
sees a token it believes is a timezone name but cannot resolve (for
example ``IV`` on some Tatar pages). ``normalize_date`` passes
``ignoretz=True``, which per the dateutil 2.8+ documentation returns a
naive ``datetime`` and suppresses timezone-name resolution.

Category guard
--------------
``_acceptable_category`` mirrors the guard in
``pipeline_extraction.ExtractionEngine.extract_category``: serialized
JSON-LD blobs (leading ``{`` / ``[``, or embedding ``@context`` /
``@type``) and over-long strings are rejected. The same guard is
applied to both UCB and Trafilatura so that a JSON-LD breadcrumb blob
is never counted as a successful category extraction.

Trafilatura API
---------------
``trafilatura.extract_metadata()`` returns a ``Document`` object in
trafilatura 2.1.x (2026). ``Document.categories`` is a ``list[str]``.
``_trafilatura_category`` handles ``None``, ``str``, ``list``,
``tuple`` and ``dict`` shapes so the benchmark is not broken by a
future minor version.

Output
------
    trafilatura_comparison.csv   one row per URL
    trafilatura_coverage.csv     aggregate per site

Parent directories of both output files are created automatically.

Installation
------------
    pip install trafilatura rouge-score python-dateutil tqdm

Usage
-----
    python tools/benchmark_trafilatura.py
    python tools/benchmark_trafilatura.py --sites-per-lang 2 --articles-per-site 20
    python tools/benchmark_trafilatura.py --exclude sssr ozodi
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from collections import defaultdict
from typing import Any, Optional

# Both the project root and tools/ must be importable:
#   * project root → config.*, pipeline.*
#   * tools/       → _common
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _TOOLS_DIR)

from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from tqdm import tqdm

import trafilatura
from config.loader import load_modular_config
from pipeline.pipeline_extraction import ExtractionEngine
from logger_setup import get_file_logger

from _common import WORD_RE, discover_urls_for_site, stats_summary


logger = get_file_logger("benchmark_trafilatura", "logs/benchmark_trafilatura.log")


# ---------------------------------------------------------------------------
# ROUGE with a Unicode-aware tokenizer
# ---------------------------------------------------------------------------
# rouge_score.tokenizers.Tokenizer is an ABC with a single abstract
# method ``tokenize(self, text)``. Subclass it; ``__call__`` is added
# for callers that pass the tokenizer as a plain function.

try:
    from rouge_score.tokenizers import Tokenizer as _RougeTokenizerBase
except ImportError:  # pragma: no cover - rouge-score not installed
    class _RougeTokenizerBase:  # type: ignore[no-redef]
        def tokenize(self, text: str) -> list[str]:
            raise NotImplementedError


_UNICODE_WORD_RE = re.compile(r"\w+", re.UNICODE)


class _UnicodeTokenizer(_RougeTokenizerBase):
    r"""
    Unicode-aware tokenizer compatible with rouge_score.

    Splits on ``\w+`` under ``re.UNICODE``, lowercases the input, and
    ignores extra keyword arguments (``stemmer``) that some
    rouge-score versions forward to ``tokenize``.
    """

    def tokenize(self, text: str, *args: Any, **kwargs: Any) -> list[str]:
        return _UNICODE_WORD_RE.findall((text or "").lower())

    def __call__(self, text: str, *args: Any, **kwargs: Any) -> list[str]:
        return self.tokenize(text, *args, **kwargs)


_ROUGE_ERROR_LOGGED = False


def _build_scorer() -> Optional[Any]:
    """Return a configured RougeScorer, or None if rouge-score is unavailable."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("[WARN] rouge-score is not installed; ROUGE will be skipped.")
        print("       Install with: pip install rouge-score")
        return None

    try:
        return rouge_scorer.RougeScorer(
            ["rouge1", "rougeL"],
            use_stemmer=False,
            tokenizer=_UnicodeTokenizer(),
        )
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[WARN] ROUGE scorer init failed: {type(exc).__name__}: {exc}")
        return None


_SCORER: Optional[Any] = _build_scorer()


URL_DATE_RE = re.compile(r"/(\d{4})/(\d{2})/")


# ---------------------------------------------------------------------------
# Reference metadata
# ---------------------------------------------------------------------------

def url_year_month(url: str) -> Optional[str]:
    """Return ``YYYY-MM`` from a URL path, or None."""
    match = URL_DATE_RE.search(url)
    return f"{match.group(1)}-{match.group(2)}" if match else None


def meta_date(soup: BeautifulSoup) -> Optional[str]:
    """Return the reference publication date from meta tags."""
    for prop in (
        "article:published_time",
        "og:published_time",
        "article:modified_time",
    ):
        tag = soup.find("meta", attrs={"property": prop})
        if tag and tag.get("content"):
            return tag["content"]
    for name in (
        "datePublished",
        "pubdate",
        "publishdate",
        "date",
        "DC.date",
        "sailthru.date",
    ):
        tag = soup.find("meta", attrs={"name": name})
        if tag and tag.get("content"):
            return tag["content"]
    return None


def meta_author(soup: BeautifulSoup) -> Optional[str]:
    """Return the reference author from meta tags only."""
    for name in (
        "author",
        "article:author",
        "DC.creator",
        "sailthru.author",
        "twitter:creator",
    ):
        tag = soup.find("meta", attrs={"name": name})
        if tag and tag.get("content"):
            return tag["content"]
    for prop in ("article:author", "og:article:author"):
        tag = soup.find("meta", attrs={"property": prop})
        if tag and tag.get("content"):
            return tag["content"]
    return None


def meta_title(soup: BeautifulSoup) -> Optional[str]:
    """Return the reference title from OpenGraph / Twitter / generic tags."""
    for prop in ("og:title", "twitter:title"):
        tag = soup.find("meta", attrs={"property": prop})
        if tag and tag.get("content"):
            return tag["content"]
    tag = soup.find("meta", attrs={"name": "title"})
    if tag and tag.get("content"):
        return tag["content"]
    return None


def normalize_date(value: Optional[str]) -> Optional[str]:
    """
    Parse a date string into ``YYYY-MM-DD``, or return None.

    ``ignoretz=True`` is required because dateutil raises
    ``UnknownTimezoneWarning`` on tokens such as ``IV`` that look like
    timezone names but are not. The date-only result does not need a
    timezone, so ignoring it is lossless for this benchmark.
    """
    if not value:
        return None
    try:
        parsed = dateparser.parse(
            str(value),
            fuzzy=True,
            ignoretz=True,
        )
        return parsed.date().isoformat() if parsed else None
    except (ValueError, TypeError, OverflowError):
        return None


def year_month(iso_date: Optional[str]) -> Optional[str]:
    """Return ``YYYY-MM`` from an ISO date, or None."""
    return iso_date[:7] if iso_date else None


# ---------------------------------------------------------------------------
# Category guard (mirrors pipeline_extraction._acceptable_category)
# ---------------------------------------------------------------------------

def _acceptable_category(value: Optional[str]) -> Optional[str]:
    """
    Return a cleaned category label, or None if the value does not
    look like a real category.

    Rejected:
      * empty or whitespace-only values;
      * serialized JSON / JSON-LD (leading ``{`` or ``[``);
      * strings embedding ``@context`` or ``@type``;
      * labels longer than 120 characters.
    """
    if not value:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text:
        return None
    if text.startswith(("{", "[")):
        return None
    if "@context" in text or "@type" in text:
        return None
    if len(text) > 120:
        return None
    return text


def _trafilatura_field(meta: Any, name: str) -> Any:
    """Read a Trafilatura metadata field from either a Document or a dict."""
    if meta is None:
        return None
    if isinstance(meta, dict):
        return meta.get(name)
    return getattr(meta, name, None)


def _trafilatura_category(meta: Any) -> Optional[str]:
    """
    Extract the first acceptable category from a Trafilatura metadata object.

    ``categories`` may be ``None``, ``str``, ``list`` or ``tuple``
    depending on the trafilatura version (2.1.x in 2026 returns a
    ``list[str]`` on ``Document.categories``, sourced from breadcrumbs
    and ``<meta>`` elements).
    """
    raw = _trafilatura_field(meta, "categories")
    if not raw:
        return None
    if isinstance(raw, str):
        return _acceptable_category(raw)
    if isinstance(raw, (list, tuple)):
        for item in raw:
            label = _acceptable_category(item)
            if label:
                return label
    return None


# ---------------------------------------------------------------------------
# Author comparison (strict)
# ---------------------------------------------------------------------------

def _author_tokens(value: str) -> list[str]:
    """Lowercase word tokens with punctuation removed."""
    cleaned = re.sub(r"[^\w\s]", " ", (value or "").lower(), flags=re.UNICODE)
    return [token for token in cleaned.split() if token]


def _author_correct(extracted: Optional[str], reference: Optional[str]) -> bool:
    """
    Strict author comparison.

    The previous substring rule (``reference[:15] in extracted``)
    treated different names as a match, e.g. "Aleksandr Alekseev" and
    "Aleksandr Aleksandrov". Token-level comparison fixes that:

      * two or more shared full tokens → match;
      * single token on both sides and equal → match;
      * otherwise → no match.

    Empty values never match.
    """
    if not reference or not extracted:
        return False
    extracted_tokens = _author_tokens(extracted)
    reference_tokens = _author_tokens(reference)
    if not extracted_tokens or not reference_tokens:
        return False
    shared = set(extracted_tokens) & set(reference_tokens)
    if len(shared) >= 2:
        return True
    if (
        len(extracted_tokens) == 1
        and len(reference_tokens) == 1
        and extracted_tokens[0] == reference_tokens[0]
    ):
        return True
    return False


# ---------------------------------------------------------------------------
# Gold text and body-quality metrics
# ---------------------------------------------------------------------------

def _clean_text(value: Optional[str]) -> str:
    """Collapse whitespace and strip."""
    return re.sub(r"\s+", " ", value or "").strip()


def _tokenize(value: str) -> list[str]:
    """Split text into lowercase word units via the shared WORD_RE."""
    return WORD_RE.findall((value or "").lower())


def extract_gold_text(soup: BeautifulSoup) -> str:
    """
    Return the gold article body used to score both systems.

    Resolution order:
      1. A fixed list of narrow selectors for common CMS layouts.
      2. ``<article>`` / ``<main>`` / ``[role=main]``.
      3. ``<body>`` with boilerplate tags removed.
    """
    narrow_selectors = [
        "div.page-main__text",
        "div.shortcode-content",
        "section.cols > div",
        ".news-single .content",
        ".single__content",
        "div.tdb-block-inner.td-fix-index",
        "div.td-post-content",
        "[itemprop=articleBody]",
        ".article-body",
        ".post-content",
        ".entry-content",
        ".article-content",
        ".news-text",
        ".article__body",
        ".story-body",
        ".content-body",
        ".article-body__content",
        ".content",
    ]
    for selector in narrow_selectors:
        try:
            node = soup.select_one(selector)
        except Exception:
            node = None
        if node:
            text = _clean_text(node.get_text(" ", strip=True))
            if len(text) > 200:
                return text

    for selector in ("article", "main", "[role=main]"):
        try:
            node = soup.select_one(selector)
        except Exception:
            node = None
        if node:
            text = _clean_text(node.get_text(" ", strip=True))
            if len(text) > 300:
                return text

    body = soup.find("body")
    if body:
        soup_copy = BeautifulSoup(str(body), "html.parser")
        for tag in soup_copy(
            ["script", "style", "nav", "header", "footer",
             "aside", "form", "button", "noscript"]
        ):
            tag.decompose()
        return _clean_text(soup_copy.get_text(" ", strip=True))
    return ""


def word_prf1(prediction: str, gold: str) -> tuple[float, float, float]:
    """Word-level precision, recall and F1 over unique words."""
    pred_set = set(_tokenize(prediction))
    gold_set = set(_tokenize(gold))
    if not pred_set or not gold_set:
        return 0.0, 0.0, 0.0
    intersection = pred_set & gold_set
    precision = len(intersection) / len(pred_set)
    recall = len(intersection) / len(gold_set)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def content_metrics(prediction: Optional[str], gold: Optional[str]) -> dict[str, Optional[float]]:
    """
    Compute ROUGE-1, ROUGE-L, word-level P/R/F1 and length ratio.

    Fields are None when they cannot be computed, e.g. gold text is
    shorter than 300 characters. ROUGE scoring failures are logged
    once and do not abort the run.
    """
    global _ROUGE_ERROR_LOGGED

    out: dict[str, Optional[float]] = {
        "len_ratio": None,
        "word_precision": None,
        "word_recall": None,
        "word_f1": None,
        "rouge_1_f1": None,
        "rouge_l_f1": None,
    }
    if not prediction or not gold or len(gold) < 300:
        return out

    out["len_ratio"] = round(len(prediction) / len(gold), 2)

    precision, recall, f1 = word_prf1(prediction, gold)
    out["word_precision"] = round(precision, 4)
    out["word_recall"] = round(recall, 4)
    out["word_f1"] = round(f1, 4)

    if _SCORER is not None:
        try:
            scores = _SCORER.score(gold, prediction)
            out["rouge_1_f1"] = round(scores["rouge1"].fmeasure, 4)
            out["rouge_l_f1"] = round(scores["rougeL"].fmeasure, 4)
        except Exception as exc:
            if not _ROUGE_ERROR_LOGGED:
                logger.warning(
                    "ROUGE scoring failed once: %s: %s",
                    type(exc).__name__,
                    exc,
                )
                _ROUGE_ERROR_LOGGED = True

    return out


# ---------------------------------------------------------------------------
# URL collection
# ---------------------------------------------------------------------------

def gather_from_config(
    engine: ExtractionEngine,
    langs: list[str],
    sites_per_lang: int,
    articles_per_site: int,
    exclude: Optional[list[str]] = None,
) -> list[dict[str, str]]:
    """Select sites by language and collect article URLs from each."""
    exclude_set = set(exclude or [])
    config = load_modular_config(engine.yaml_path)
    sites = config.get("sites", {})

    by_lang: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for site_key, site_cfg in sites.items():
        if not isinstance(site_cfg, dict):
            continue
        if site_key in exclude_set:
            continue
        lang = site_cfg.get("default_language")
        if lang not in langs:
            continue
        if len(by_lang[lang]) >= sites_per_lang:
            continue
        by_lang[lang].append((site_key, site_cfg))

    print("\n[PLAN] Sites to process:")
    if exclude_set:
        print(f"    (excluded: {', '.join(sorted(exclude_set))})")
    for lang in sorted(by_lang):
        names = [key for key, _ in by_lang[lang]]
        print(f"    {lang}: {', '.join(names)}")

    urls: list[dict[str, str]] = []
    for lang in sorted(by_lang):
        print(f"\n[{lang.upper()}]")
        for site_key, site_cfg in by_lang[lang]:
            match = site_cfg.get("match", [])
            domain = match[0] if match else "?"
            print(f"  [DISCOVER] {site_key} ({domain})")
            site_urls = discover_urls_for_site(
                engine, site_key, site_cfg, articles_per_site
            )
            for url in site_urls:
                urls.append({"url": url, "site": site_key, "lang": lang})
            print(f"    -> {len(site_urls)} URLs")
    return urls


# ---------------------------------------------------------------------------
# Per-URL comparison
# ---------------------------------------------------------------------------

def compare_one(engine: ExtractionEngine, article: dict[str, str]) -> dict[str, Any]:
    """Run both extractors on one URL and compute all metrics."""
    url = article["url"]
    result: dict[str, Any] = {
        "url": url,
        "site": article["site"],
        "lang": article["lang"],
        "error": None,
        # Reference values
        "ref_url_ym": url_year_month(url),
        "ref_meta_date": None,
        "ref_meta_author": None,
        "ref_meta_title": None,
        # UniCorpusBuilder
        "ucb_date": None,
        "ucb_author": None,
        "ucb_title": None,
        "ucb_category": None,
        "ucb_content_len": 0,
        # Trafilatura
        "traf_date": None,
        "traf_author": None,
        "traf_title": None,
        "traf_category": None,
        "traf_content_len": 0,
        # Gold body
        "gold_len": 0,
        # Quality metrics: UniCorpusBuilder
        "ucb_len_ratio": None,
        "ucb_word_precision": None,
        "ucb_word_recall": None,
        "ucb_word_f1": None,
        "ucb_rouge_1_f1": None,
        "ucb_rouge_l_f1": None,
        # Quality metrics: Trafilatura
        "traf_len_ratio": None,
        "traf_word_precision": None,
        "traf_word_recall": None,
        "traf_word_f1": None,
        "traf_rouge_1_f1": None,
        "traf_rouge_l_f1": None,
    }

    try:
        html = engine.fetch_html(url)
    except Exception as exc:
        result["error"] = f"fetch: {exc}"
        logger.warning("fetch failed for %s: %s", url, exc)
        return result

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception as exc:
        result["error"] = f"parse: {exc}"
        logger.warning("parse failed for %s: %s", url, exc)
        return result

    # Reference metadata
    result["ref_meta_date"] = normalize_date(meta_date(soup))
    result["ref_meta_author"] = meta_author(soup)
    result["ref_meta_title"] = meta_title(soup)

    # UniCorpusBuilder
    ucb_content = ""
    try:
        data = engine.extract_article_fields(html, url)
        result["ucb_date"] = normalize_date(data.get("date"))
        result["ucb_author"] = data.get("author")
        result["ucb_title"] = data.get("title")
        result["ucb_category"] = _acceptable_category(data.get("category"))
        ucb_content = data.get("content") or ""
        result["ucb_content_len"] = len(ucb_content)
    except Exception as exc:
        result["error"] = f"ucb: {exc}"
        logger.warning("UCB extraction failed for %s: %s", url, exc)

    # Trafilatura: metadata
    try:
        meta = trafilatura.extract_metadata(html)
        if meta:
            result["traf_date"] = normalize_date(_trafilatura_field(meta, "date"))
            result["traf_author"] = _trafilatura_field(meta, "author")
            result["traf_title"] = _trafilatura_field(meta, "title")
            result["traf_category"] = _trafilatura_category(meta)
    except Exception as exc:
        logger.warning("Trafilatura metadata failed for %s: %s", url, exc)

    # Trafilatura: body text
    traf_content = ""
    try:
        text = trafilatura.extract(
            html,
            output_format="txt",
            include_comments=False,
            include_tables=True,
        )
        if text:
            traf_content = text
            result["traf_content_len"] = len(text)
    except Exception as exc:
        logger.warning("Trafilatura body failed for %s: %s", url, exc)

    # Gold body
    gold = extract_gold_text(soup)
    result["gold_len"] = len(gold)

    # Quality metrics
    ucb_metrics = content_metrics(ucb_content, gold)
    result["ucb_len_ratio"] = ucb_metrics["len_ratio"]
    result["ucb_word_precision"] = ucb_metrics["word_precision"]
    result["ucb_word_recall"] = ucb_metrics["word_recall"]
    result["ucb_word_f1"] = ucb_metrics["word_f1"]
    result["ucb_rouge_1_f1"] = ucb_metrics["rouge_1_f1"]
    result["ucb_rouge_l_f1"] = ucb_metrics["rouge_l_f1"]

    traf_metrics = content_metrics(traf_content, gold)
    result["traf_len_ratio"] = traf_metrics["len_ratio"]
    result["traf_word_precision"] = traf_metrics["word_precision"]
    result["traf_word_recall"] = traf_metrics["word_recall"]
    result["traf_word_f1"] = traf_metrics["word_f1"]
    result["traf_rouge_1_f1"] = traf_metrics["rouge_1_f1"]
    result["traf_rouge_l_f1"] = traf_metrics["rouge_l_f1"]

    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def evaluate(results: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Aggregate per-URL results into global, per-site and per-language views."""
    ev: dict[str, Any] = {
        "total": 0,
        "errors": 0,
        # Date from URL
        "total_url_date": 0,
        "ucb_url_ok": 0,
        "ucb_url_empty": 0,
        "ucb_url_wrong": 0,
        "traf_url_ok": 0,
        "traf_url_empty": 0,
        "traf_url_wrong": 0,
        # Date from meta
        "total_meta_date": 0,
        "ucb_meta_ok": 0,
        "ucb_meta_empty": 0,
        "ucb_meta_wrong": 0,
        "traf_meta_ok": 0,
        "traf_meta_empty": 0,
        "traf_meta_wrong": 0,
        # Author from meta
        "total_meta_author": 0,
        "ucb_author_ok": 0,
        "traf_author_ok": 0,
        # Title from meta
        "total_meta_title": 0,
        "ucb_title_ok": 0,
        "traf_title_ok": 0,
        # Coverage
        "ucb_has_author": 0,
        "traf_has_author": 0,
        "ucb_has_title": 0,
        "traf_has_title": 0,
        "ucb_has_date": 0,
        "traf_has_date": 0,
        "ucb_has_category": 0,
        "traf_has_category": 0,
        # Body length
        "ucb_len": [],
        "traf_len": [],
        # Quality metrics: UniCorpusBuilder
        "ucb_rouge_1": [],
        "ucb_rouge_l": [],
        "ucb_word_p": [],
        "ucb_word_r": [],
        "ucb_word_f1": [],
        "ucb_len_ratio": [],
        # Quality metrics: Trafilatura
        "traf_rouge_1": [],
        "traf_rouge_l": [],
        "traf_word_p": [],
        "traf_word_r": [],
        "traf_word_f1": [],
        "traf_len_ratio": [],
    }

    cov: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "total": 0,
            "ucb_author": 0,
            "traf_author": 0,
            "ucb_title": 0,
            "traf_title": 0,
            "ucb_date": 0,
            "traf_date": 0,
            "ucb_category": 0,
            "traf_category": 0,
            "ucb_rouge_l": [],
            "traf_rouge_l": [],
            "ucb_word_f1": [],
            "traf_word_f1": [],
        }
    )
    by_lang: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "total": 0,
            "ucb_author": 0,
            "traf_author": 0,
            "ucb_title": 0,
            "traf_title": 0,
            "ucb_date": 0,
            "traf_date": 0,
            "ucb_rouge_l": [],
            "traf_rouge_l": [],
            "ucb_word_f1": [],
            "traf_word_f1": [],
        }
    )

    for row in results:
        if row["error"]:
            ev["errors"] += 1
            continue
        ev["total"] += 1

        site = row["site"]
        lang = row["lang"]
        cov[site]["total"] += 1
        by_lang[lang]["total"] += 1

        # Date from URL
        ref = row["ref_url_ym"]
        if ref:
            ev["total_url_date"] += 1
            ucb_ym = year_month(row["ucb_date"])
            traf_ym = year_month(row["traf_date"])
            if ucb_ym == ref:
                ev["ucb_url_ok"] += 1
            elif not ucb_ym:
                ev["ucb_url_empty"] += 1
            else:
                ev["ucb_url_wrong"] += 1
            if traf_ym == ref:
                ev["traf_url_ok"] += 1
            elif not traf_ym:
                ev["traf_url_empty"] += 1
            else:
                ev["traf_url_wrong"] += 1

        # Date from meta
        ref = row["ref_meta_date"]
        if ref:
            ev["total_meta_date"] += 1
            if row["ucb_date"] == ref:
                ev["ucb_meta_ok"] += 1
            elif not row["ucb_date"]:
                ev["ucb_meta_empty"] += 1
            else:
                ev["ucb_meta_wrong"] += 1
            if row["traf_date"] == ref:
                ev["traf_meta_ok"] += 1
            elif not row["traf_date"]:
                ev["traf_meta_empty"] += 1
            else:
                ev["traf_meta_wrong"] += 1

        # Author from meta (strict comparison)
        ref = row["ref_meta_author"]
        if ref:
            ev["total_meta_author"] += 1
            if _author_correct(row["ucb_author"], ref):
                ev["ucb_author_ok"] += 1
            if _author_correct(row["traf_author"], ref):
                ev["traf_author_ok"] += 1

        # Title from meta
        ref = row["ref_meta_title"]
        if ref:
            ev["total_meta_title"] += 1
            if row["ucb_title"] and ref[:30].lower() in row["ucb_title"].lower():
                ev["ucb_title_ok"] += 1
            if row["traf_title"] and ref[:30].lower() in row["traf_title"].lower():
                ev["traf_title_ok"] += 1

        # Coverage
        ucb_category_ok = _acceptable_category(row["ucb_category"]) is not None
        traf_category_ok = _acceptable_category(row["traf_category"]) is not None

        if row["ucb_author"]:
            ev["ucb_has_author"] += 1
        if row["traf_author"]:
            ev["traf_has_author"] += 1
        if row["ucb_title"]:
            ev["ucb_has_title"] += 1
        if row["traf_title"]:
            ev["traf_has_title"] += 1
        if row["ucb_date"]:
            ev["ucb_has_date"] += 1
        if row["traf_date"]:
            ev["traf_has_date"] += 1
        if ucb_category_ok:
            ev["ucb_has_category"] += 1
        if traf_category_ok:
            ev["traf_has_category"] += 1

        # Coverage per site
        if row["ucb_author"]:
            cov[site]["ucb_author"] += 1
        if row["traf_author"]:
            cov[site]["traf_author"] += 1
        if row["ucb_title"]:
            cov[site]["ucb_title"] += 1
        if row["traf_title"]:
            cov[site]["traf_title"] += 1
        if row["ucb_date"]:
            cov[site]["ucb_date"] += 1
        if row["traf_date"]:
            cov[site]["traf_date"] += 1
        if ucb_category_ok:
            cov[site]["ucb_category"] += 1
        if traf_category_ok:
            cov[site]["traf_category"] += 1

        # Coverage per language
        if row["ucb_author"]:
            by_lang[lang]["ucb_author"] += 1
        if row["traf_author"]:
            by_lang[lang]["traf_author"] += 1
        if row["ucb_title"]:
            by_lang[lang]["ucb_title"] += 1
        if row["traf_title"]:
            by_lang[lang]["traf_title"] += 1
        if row["ucb_date"]:
            by_lang[lang]["ucb_date"] += 1
        if row["traf_date"]:
            by_lang[lang]["traf_date"] += 1

        # Body length
        if row["ucb_content_len"]:
            ev["ucb_len"].append(row["ucb_content_len"])
        if row["traf_content_len"]:
            ev["traf_len"].append(row["traf_content_len"])

        # Quality metrics: UniCorpusBuilder
        if row["ucb_rouge_1_f1"] is not None:
            ev["ucb_rouge_1"].append(row["ucb_rouge_1_f1"])
        if row["ucb_rouge_l_f1"] is not None:
            ev["ucb_rouge_l"].append(row["ucb_rouge_l_f1"])
            cov[site]["ucb_rouge_l"].append(row["ucb_rouge_l_f1"])
            by_lang[lang]["ucb_rouge_l"].append(row["ucb_rouge_l_f1"])
        if row["ucb_word_f1"] is not None:
            ev["ucb_word_f1"].append(row["ucb_word_f1"])
            cov[site]["ucb_word_f1"].append(row["ucb_word_f1"])
            by_lang[lang]["ucb_word_f1"].append(row["ucb_word_f1"])
        if row["ucb_word_precision"] is not None:
            ev["ucb_word_p"].append(row["ucb_word_precision"])
        if row["ucb_word_recall"] is not None:
            ev["ucb_word_r"].append(row["ucb_word_recall"])
        if row["ucb_len_ratio"] is not None:
            ev["ucb_len_ratio"].append(row["ucb_len_ratio"])

        # Quality metrics: Trafilatura
        if row["traf_rouge_1_f1"] is not None:
            ev["traf_rouge_1"].append(row["traf_rouge_1_f1"])
        if row["traf_rouge_l_f1"] is not None:
            ev["traf_rouge_l"].append(row["traf_rouge_l_f1"])
            cov[site]["traf_rouge_l"].append(row["traf_rouge_l_f1"])
            by_lang[lang]["traf_rouge_l"].append(row["traf_rouge_l_f1"])
        if row["traf_word_f1"] is not None:
            ev["traf_word_f1"].append(row["traf_word_f1"])
            cov[site]["traf_word_f1"].append(row["traf_word_f1"])
            by_lang[lang]["traf_word_f1"].append(row["traf_word_f1"])
        if row["traf_word_precision"] is not None:
            ev["traf_word_p"].append(row["traf_word_precision"])
        if row["traf_word_recall"] is not None:
            ev["traf_word_r"].append(row["traf_word_recall"])
        if row["traf_len_ratio"] is not None:
            ev["traf_len_ratio"].append(row["traf_len_ratio"])

    return ev, cov, by_lang


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def pct(numerator: int, denominator: int) -> str:
    """Format a ratio as ``n/d = P%``, or an em dash when the denominator is zero."""
    return (
        f"{numerator}/{denominator} = {numerator / denominator * 100:.1f}%"
        if denominator
        else "—"
    )


def print_report(
    ev: dict[str, Any],
    cov: dict[str, dict[str, Any]],
    by_lang: dict[str, dict[str, Any]],
    results: list[dict[str, Any]],
    out_csv: str,
    out_cov: str,
) -> None:
    """Print all comparison tables and write both CSVs."""
    print(f"\n{'=' * 70}")
    print(f"Documents processed: {ev['total']}, errors: {ev['errors']}")
    print(f"{'=' * 70}")

    total = ev["total"]
    if total == 0:
        print("[WARN] No documents were processed successfully.")
        if results:
            _write_per_url_csv(results, out_csv)
        return

    # Date
    if ev["total_url_date"]:
        print(f"\n[DATE: year-month from URL]   documents: {ev['total_url_date']}")
        print(
            f"  UniCorpusBuilder:  {pct(ev['ucb_url_ok'], ev['total_url_date'])}  "
            f"(empty: {ev['ucb_url_empty']}, wrong: {ev['ucb_url_wrong']})"
        )
        print(
            f"  Trafilatura:       {pct(ev['traf_url_ok'], ev['total_url_date'])}  "
            f"(empty: {ev['traf_url_empty']}, wrong: {ev['traf_url_wrong']})"
        )
    else:
        print("\n[DATE: year-month from URL]   no data")

    if ev["total_meta_date"]:
        print(f"\n[DATE: HTML meta tags]        documents: {ev['total_meta_date']}")
        print(
            f"  UniCorpusBuilder:  {pct(ev['ucb_meta_ok'], ev['total_meta_date'])}  "
            f"(empty: {ev['ucb_meta_empty']}, wrong: {ev['ucb_meta_wrong']})"
        )
        print(
            f"  Trafilatura:       {pct(ev['traf_meta_ok'], ev['total_meta_date'])}  "
            f"(empty: {ev['traf_meta_empty']}, wrong: {ev['traf_meta_wrong']})"
        )

    # Author
    print("\n[AUTHOR extracted]")
    print(f"  UniCorpusBuilder:  {pct(ev['ucb_has_author'], total)}")
    print(f"  Trafilatura:       {pct(ev['traf_has_author'], total)}")

    if ev["total_meta_author"]:
        print(f"\n[AUTHOR matches meta-author] documents: {ev['total_meta_author']}")
        print(f"  UniCorpusBuilder:  {pct(ev['ucb_author_ok'], ev['total_meta_author'])}")
        print(f"  Trafilatura:       {pct(ev['traf_author_ok'], ev['total_meta_author'])}")

    # Title
    print("\n[TITLE extracted]")
    print(f"  UniCorpusBuilder:  {pct(ev['ucb_has_title'], total)}")
    print(f"  Trafilatura:       {pct(ev['traf_has_title'], total)}")

    # Category
    print("\n[CATEGORY extracted]")
    print(f"  UniCorpusBuilder:  {pct(ev['ucb_has_category'], total)}")
    print(f"  Trafilatura:       {pct(ev['traf_has_category'], total)}")

    # Body length
    if ev["ucb_len"]:
        ucb_avg = sum(ev["ucb_len"]) / len(ev["ucb_len"])
        traf_avg = (
            sum(ev["traf_len"]) / len(ev["traf_len"]) if ev["traf_len"] else 0
        )
        print("\n[AVERAGE BODY LENGTH]")
        print(f"  UniCorpusBuilder:  {ucb_avg:.0f} chars")
        print(f"  Trafilatura:       {traf_avg:.0f} chars")

    # Body-extraction quality
    print(f"\n{'=' * 70}")
    print("[BODY-EXTRACTION QUALITY] — gold: <article> / <main> / <body>")
    print(f"{'=' * 70}")

    for label, prefix in (("UniCorpusBuilder", "ucb"), ("Trafilatura", "traf")):
        print(f"\n  --- {label} ---")
        rouge_l = stats_summary(ev[f"{prefix}_rouge_l"])
        rouge_1 = stats_summary(ev[f"{prefix}_rouge_1"])
        word_f1 = stats_summary(ev[f"{prefix}_word_f1"])
        word_p = stats_summary(ev[f"{prefix}_word_p"])
        word_r = stats_summary(ev[f"{prefix}_word_r"])
        length_ratio = stats_summary(ev[f"{prefix}_len_ratio"])

        if rouge_l:
            print(
                f"  ROUGE-L F1:  N = {rouge_l['n']}  mean = {rouge_l['mean']:.4f}  "
                f"median = {rouge_l['median']:.4f}  "
                f"min/max = {rouge_l['min']:.4f}/{rouge_l['max']:.4f}"
            )
        else:
            print("  ROUGE-L F1:  no data")

        if rouge_1:
            print(
                f"  ROUGE-1 F1:  mean = {rouge_1['mean']:.4f}  "
                f"median = {rouge_1['median']:.4f}"
            )
        if word_f1:
            print(
                f"  Word-level:  P = {word_p['mean']:.4f}  R = {word_r['mean']:.4f}  "
                f"F1 = {word_f1['mean']:.4f}  median = {word_f1['median']:.4f}"
            )
        if length_ratio:
            print(
                f"  Length ratio: mean = {length_ratio['mean']:.2f}  "
                f"median = {length_ratio['median']:.2f}  "
                f"min/max = {length_ratio['min']:.2f}/{length_ratio['max']:.2f}"
            )

    # Coverage by site
    print(f"\n{'=' * 70}")
    print("[COVERAGE BY SITE]")
    print(f"{'=' * 70}")
    print(
        f"  {'Site':<18} {'N':>4} {'AuthU':>6} {'AuthT':>6} "
        f"{'CatU':>6} {'CatT':>6} {'RL U':>6} {'RL T':>6} "
        f"{'WF1 U':>6} {'WF1 T':>6}"
    )
    print("  " + "-" * 82)

    cov_rows: list[dict[str, Any]] = []
    for site in sorted(cov):
        entry = cov[site]
        site_total = entry["total"]
        if site_total == 0:
            continue

        rouge_ucb = stats_summary(entry["ucb_rouge_l"])
        rouge_traf = stats_summary(entry["traf_rouge_l"])
        word_ucb = stats_summary(entry["ucb_word_f1"])
        word_traf = stats_summary(entry["traf_word_f1"])

        rl_u = f"{rouge_ucb['mean']:.3f}" if rouge_ucb else "—"
        rl_t = f"{rouge_traf['mean']:.3f}" if rouge_traf else "—"
        wf_u = f"{word_ucb['mean']:.3f}" if word_ucb else "—"
        wf_t = f"{word_traf['mean']:.3f}" if word_traf else "—"

        print(
            f"  {site:<18} {site_total:>4} "
            f"{entry['ucb_author'] / site_total * 100:>5.1f}% "
            f"{entry['traf_author'] / site_total * 100:>5.1f}% "
            f"{entry['ucb_category'] / site_total * 100:>5.1f}% "
            f"{entry['traf_category'] / site_total * 100:>5.1f}% "
            f"{rl_u:>6} {rl_t:>6} {wf_u:>6} {wf_t:>6}"
        )

        cov_rows.append(
            {
                "site": site,
                "articles": site_total,
                "ucb_author_pct": round(entry["ucb_author"] / site_total * 100, 1),
                "traf_author_pct": round(entry["traf_author"] / site_total * 100, 1),
                "ucb_date_pct": round(entry["ucb_date"] / site_total * 100, 1),
                "traf_date_pct": round(entry["traf_date"] / site_total * 100, 1),
                "ucb_title_pct": round(entry["ucb_title"] / site_total * 100, 1),
                "traf_title_pct": round(entry["traf_title"] / site_total * 100, 1),
                "ucb_category_pct": round(entry["ucb_category"] / site_total * 100, 1),
                "traf_category_pct": round(entry["traf_category"] / site_total * 100, 1),
                "ucb_rouge_l_mean": round(rouge_ucb["mean"], 4) if rouge_ucb else None,
                "traf_rouge_l_mean": round(rouge_traf["mean"], 4) if rouge_traf else None,
                "ucb_word_f1_mean": round(word_ucb["mean"], 4) if word_ucb else None,
                "traf_word_f1_mean": round(word_traf["mean"], 4) if word_traf else None,
                "ucb_rouge_l_n": rouge_ucb["n"] if rouge_ucb else 0,
                "traf_rouge_l_n": rouge_traf["n"] if rouge_traf else 0,
            }
        )

    # Coverage by language
    print(f"\n{'=' * 70}")
    print("[BY LANGUAGE]")
    print(f"{'=' * 70}")
    print(
        f"  {'Lang':<6} {'N':>5} {'AuthUCB':>8} {'AuthTF':>8} "
        f"{'RL UCB':>8} {'RL TF':>8} {'WF1 UCB':>8} {'WF1 TF':>8}"
    )
    print("  " + "-" * 74)

    for lang in sorted(by_lang):
        entry = by_lang[lang]
        lang_total = entry["total"]
        if lang_total == 0:
            continue

        rouge_ucb = stats_summary(entry["ucb_rouge_l"])
        rouge_traf = stats_summary(entry["traf_rouge_l"])
        word_ucb = stats_summary(entry["ucb_word_f1"])
        word_traf = stats_summary(entry["traf_word_f1"])

        rl_u = f"{rouge_ucb['mean']:.3f}" if rouge_ucb else "—"
        rl_t = f"{rouge_traf['mean']:.3f}" if rouge_traf else "—"
        wf_u = f"{word_ucb['mean']:.3f}" if word_ucb else "—"
        wf_t = f"{word_traf['mean']:.3f}" if word_traf else "—"

        print(
            f"  {lang:<6} {lang_total:>5} "
            f"{entry['ucb_author'] / lang_total * 100:>7.1f}% "
            f"{entry['traf_author'] / lang_total * 100:>7.1f}% "
            f"{rl_u:>8} {rl_t:>8} {wf_u:>8} {wf_t:>8}"
        )

    # Final table
    print(f"\n{'=' * 70}")
    print("[FINAL TABLE FOR THE ARTICLE]")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<40} {'UniCorpusBuilder':>18} {'Trafilatura':>14}")
    print("  " + "-" * 74)
    print(
        f"  {'Title extracted':<40} "
        f"{ev['ucb_has_title'] / total * 100:>17.1f}% "
        f"{ev['traf_has_title'] / total * 100:>13.1f}%"
    )
    print(
        f"  {'Author extracted':<40} "
        f"{ev['ucb_has_author'] / total * 100:>17.1f}% "
        f"{ev['traf_has_author'] / total * 100:>13.1f}%"
    )
    if ev["total_meta_date"]:
        print(
            f"  {'Date matches meta (N=' + str(ev['total_meta_date']) + ')':<40} "
            f"{ev['ucb_meta_ok'] / ev['total_meta_date'] * 100:>17.1f}% "
            f"{ev['traf_meta_ok'] / ev['total_meta_date'] * 100:>13.1f}%"
        )
    if ev["total_url_date"]:
        print(
            f"  {'Date matches URL (N=' + str(ev['total_url_date']) + ')':<40} "
            f"{ev['ucb_url_ok'] / ev['total_url_date'] * 100:>17.1f}% "
            f"{ev['traf_url_ok'] / ev['total_url_date'] * 100:>13.1f}%"
        )
    print(
        f"  {'Category extracted':<40} "
        f"{ev['ucb_has_category'] / total * 100:>17.1f}% "
        f"{ev['traf_has_category'] / total * 100:>13.1f}%"
    )
    if ev["ucb_len"]:
        ucb_avg = sum(ev["ucb_len"]) / len(ev["ucb_len"])
        traf_avg = (
            sum(ev["traf_len"]) / len(ev["traf_len"]) if ev["traf_len"] else 0
        )
        print(
            f"  {'Average body length (chars)':<40} "
            f"{ucb_avg:>17.0f} {traf_avg:>14.0f}"
        )

    rouge_l_ucb = stats_summary(ev["ucb_rouge_l"])
    rouge_l_traf = stats_summary(ev["traf_rouge_l"])
    rouge_1_ucb = stats_summary(ev["ucb_rouge_1"])
    rouge_1_traf = stats_summary(ev["traf_rouge_1"])
    word_f1_ucb = stats_summary(ev["ucb_word_f1"])
    word_f1_traf = stats_summary(ev["traf_word_f1"])
    word_p_ucb = stats_summary(ev["ucb_word_p"])
    word_p_traf = stats_summary(ev["traf_word_p"])
    word_r_ucb = stats_summary(ev["ucb_word_r"])
    word_r_traf = stats_summary(ev["traf_word_r"])
    ratio_ucb = stats_summary(ev["ucb_len_ratio"])
    ratio_traf = stats_summary(ev["traf_len_ratio"])

    if rouge_l_ucb and rouge_l_traf:
        print(
            f"  {'ROUGE-L F1 (N=' + str(rouge_l_ucb['n']) + ')':<40} "
            f"{rouge_l_ucb['mean']:>17.4f} {rouge_l_traf['mean']:>14.4f}"
        )
    if rouge_1_ucb and rouge_1_traf:
        print(
            f"  {'ROUGE-1 F1':<40} "
            f"{rouge_1_ucb['mean']:>17.4f} {rouge_1_traf['mean']:>14.4f}"
        )
    if word_p_ucb and word_p_traf:
        print(
            f"  {'Word-level Precision':<40} "
            f"{word_p_ucb['mean']:>17.4f} {word_p_traf['mean']:>14.4f}"
        )
    if word_r_ucb and word_r_traf:
        print(
            f"  {'Word-level Recall':<40} "
            f"{word_r_ucb['mean']:>17.4f} {word_r_traf['mean']:>14.4f}"
        )
    if word_f1_ucb and word_f1_traf:
        print(
            f"  {'Word-level F1':<40} "
            f"{word_f1_ucb['mean']:>17.4f} {word_f1_traf['mean']:>14.4f}"
        )
    if ratio_ucb and ratio_traf:
        print(
            f"  {'Length ratio':<40} "
            f"{ratio_ucb['mean']:>17.2f} {ratio_traf['mean']:>14.2f}"
        )

    _write_per_url_csv(results, out_csv)
    print(f"\n[OK] Per-URL CSV:       {out_csv}")

    _ensure_parent_dir(out_cov)
    with open(out_cov, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "site", "articles",
                "ucb_author_pct", "traf_author_pct",
                "ucb_date_pct", "traf_date_pct",
                "ucb_title_pct", "traf_title_pct",
                "ucb_category_pct", "traf_category_pct",
                "ucb_rouge_l_mean", "traf_rouge_l_mean",
                "ucb_rouge_l_n", "traf_rouge_l_n",
                "ucb_word_f1_mean", "traf_word_f1_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(cov_rows)
    print(f"[OK] Coverage by site:  {out_cov}")


def _ensure_parent_dir(path: str) -> None:
    """Create the parent directory of ``path`` if it does not exist."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _write_per_url_csv(results: list[dict[str, Any]], path: str) -> None:
    """Write the per-URL comparison CSV, creating the parent directory if needed."""
    if not results:
        logger.warning("No per-URL results to write to %s", path)
        return
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    """Run the benchmark and return a process exit code."""
    parser = argparse.ArgumentParser(
        description="Compare UniCorpusBuilder and Trafilatura on identical URLs."
    )
    parser.add_argument("--config", default="config/universal.yaml")
    parser.add_argument("--out", default="trafilatura_comparison.csv")
    parser.add_argument("--out-cov", default="trafilatura_coverage.csv")
    parser.add_argument("--sleep", type=float, default=0.3)
    parser.add_argument("--langs", nargs="+", default=["tg", "tt", "ba", "os"])
    parser.add_argument("--sites-per-lang", type=int, default=2)
    parser.add_argument("--articles-per-site", type=int, default=20)
    parser.add_argument("--exclude", nargs="*", default=["sssr", "ozodi"])

    args = parser.parse_args(argv)

    engine = ExtractionEngine(yaml_path=args.config)

    print(
        f"[MODE] Languages={args.langs}, "
        f"{args.sites_per_lang} sites/language, "
        f"{args.articles_per_site} articles/site"
    )

    urls = gather_from_config(
        engine,
        args.langs,
        args.sites_per_lang,
        args.articles_per_site,
        exclude=args.exclude,
    )
    if not urls:
        print("[ERROR] No URLs collected.")
        return 1

    print(f"\n[TOTAL] URLs: {len(urls)}")
    print("[FETCH] Comparing UniCorpusBuilder vs Trafilatura + ROUGE...\n")

    results: list[dict[str, Any]] = []
    for article in tqdm(urls, desc="[PROCESS]", unit="doc"):
        results.append(compare_one(engine, article))
        time.sleep(args.sleep)

    ev, cov, by_lang = evaluate(results)
    print_report(ev, cov, by_lang, results, args.out, args.out_cov)
    return 0


if __name__ == "__main__":
    sys.exit(main())
    