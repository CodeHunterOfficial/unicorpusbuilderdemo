# tools/benchmark_trafilatura.py
"""
Compare UniCorpusBuilder (UCB) against Trafilatura on the same set of
article URLs.

Metadata fields compared
------------------------
    date, author, title, category, body length

Body-extraction quality is additionally scored for both systems against
a gold text, using:

    ROUGE-1 F1, ROUGE-L F1
    Word-level Precision / Recall / F1
    Length ratio

The gold text is taken from the first of a fixed list of content
selectors, or from ``<article>`` / ``<main>`` / ``[role=main]`` /
``<body>`` when no selector matches.

Output
------
    trafilatura_comparison.csv  one row per URL
    trafilatura_coverage.csv    aggregate per site

Installation
------------
    pip install trafilatura rouge-score

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from tqdm import tqdm

import trafilatura
from config.loader import load_modular_config
from pipeline.pipeline_extraction import ExtractionEngine

from _common import WORD_RE, discover_urls_for_site, stats_summary


# --- Optional dependency: ROUGE ---
try:
    from rouge_score import rouge_scorer
    _SCORER = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
except ImportError:
    _SCORER = None
    print("[WARN] rouge-score is not installed. ROUGE will not be computed.")
    print("       Install with: pip install rouge-score")


URL_DATE_RE = re.compile(r"/(\d{4})/(\d{2})/")


# ------------------------------------------------------------------
# Reference metadata (date / author / title)
# ------------------------------------------------------------------

def url_year_month(url):
    """Extract 'YYYY-MM' from a URL path, or None if not present."""
    m = URL_DATE_RE.search(url)
    return f"{m.group(1)}-{m.group(2)}" if m else None


def meta_date(soup):
    """Reference date: read from OpenGraph / article meta tags, then fall back."""
    for prop in ["article:published_time", "og:published_time",
                 "article:modified_time"]:
        tag = soup.find("meta", attrs={"property": prop})
        if tag and tag.get("content"):
            return tag["content"]
    for name in ["datePublished", "pubdate", "publishdate", "date",
                 "DC.date", "sailthru.date"]:
        tag = soup.find("meta", attrs={"name": name})
        if tag and tag.get("content"):
            return tag["content"]
    return None


def meta_author(soup):
    """Reference author: read from meta tags only."""
    for name in ["author", "article:author", "DC.creator",
                 "sailthru.author", "twitter:creator"]:
        tag = soup.find("meta", attrs={"name": name})
        if tag and tag.get("content"):
            return tag["content"]
    for prop in ["article:author", "og:article:author"]:
        tag = soup.find("meta", attrs={"property": prop})
        if tag and tag.get("content"):
            return tag["content"]
    return None


def meta_title(soup):
    """Reference title: read from OpenGraph / Twitter / generic meta tags."""
    for prop in ["og:title", "twitter:title"]:
        tag = soup.find("meta", attrs={"property": prop})
        if tag and tag.get("content"):
            return tag["content"]
    tag = soup.find("meta", attrs={"name": "title"})
    if tag and tag.get("content"):
        return tag["content"]
    return None


def normalize_date(s):
    """Parse a date string into ISO 'YYYY-MM-DD', or None on failure."""
    if not s:
        return None
    try:
        return dateparser.parse(str(s), fuzzy=True).date().isoformat()
    except Exception:
        return None


def year_month(iso):
    """Return 'YYYY-MM' from an ISO date, or None."""
    return iso[:7] if iso else None


# ------------------------------------------------------------------
# Gold text and body-quality metrics
# ------------------------------------------------------------------

def _clean_text(s):
    """Collapse whitespace and strip."""
    return re.sub(r"\s+", " ", s or "").strip()


def _tokenize(s):
    """Tokenize a string into lowercase word units."""
    return WORD_RE.findall((s or "").lower())


def extract_gold_text(soup):
    """
    Return the gold article body used to score both systems.

    Resolution order:
      1. A fixed list of narrow selectors targeting common CMS layouts.
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
    for sel in narrow_selectors:
        try:
            node = soup.select_one(sel)
        except Exception:
            node = None
        if node:
            text = _clean_text(node.get_text(" ", strip=True))
            if len(text) > 200:
                return text

    for sel in ["article", "main", "[role=main]"]:
        try:
            node = soup.select_one(sel)
        except Exception:
            node = None
        if node:
            text = _clean_text(node.get_text(" ", strip=True))
            if len(text) > 300:
                return text

    body = soup.find("body")
    if body:
        soup_copy = BeautifulSoup(str(body), "html.parser")
        for tag in soup_copy(["script", "style", "nav", "header", "footer",
                              "aside", "form", "button", "noscript"]):
            tag.decompose()
        return _clean_text(soup_copy.get_text(" ", strip=True))
    return ""


def word_prf1(pred, gold):
    """Word-level Precision / Recall / F1 over unique words."""
    pred_set = set(_tokenize(pred))
    gold_set = set(_tokenize(gold))

    if not pred_set or not gold_set:
        return 0.0, 0.0, 0.0

    inter = pred_set & gold_set
    p = len(inter) / len(pred_set)
    r = len(inter) / len(gold_set)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def content_metrics(pred_text, gold_text):
    """
    Compute ROUGE-1, ROUGE-L, word-level P/R/F1, and length ratio.

    Returns a dict with all metrics. Individual fields are set to None
    when they cannot be computed (e.g. gold text shorter than 300 chars).
    """
    out = {
        "len_ratio": None,
        "word_precision": None,
        "word_recall": None,
        "word_f1": None,
        "rouge_1_f1": None,
        "rouge_l_f1": None,
    }
    if not pred_text or not gold_text or len(gold_text) < 300:
        return out

    out["len_ratio"] = round(len(pred_text) / len(gold_text), 2)

    wp, wr, wf1 = word_prf1(pred_text, gold_text)
    out["word_precision"] = round(wp, 4)
    out["word_recall"] = round(wr, 4)
    out["word_f1"] = round(wf1, 4)

    if _SCORER is not None:
        try:
            scores = _SCORER.score(gold_text, pred_text)
            out["rouge_1_f1"] = round(scores["rouge1"].fmeasure, 4)
            out["rouge_l_f1"] = round(scores["rougeL"].fmeasure, 4)
        except Exception:
            pass

    return out


# ------------------------------------------------------------------
# URL collection
# ------------------------------------------------------------------

def gather_from_config(engine, langs, sites_per_lang, articles_per_site,
                       exclude=None):
    """Select sites by language and collect article URLs from each."""
    exclude = set(exclude or [])
    config = load_modular_config(engine.yaml_path)
    sites = config.get("sites", {})

    by_lang = defaultdict(list)
    for site_key, site_cfg in sites.items():
        if not isinstance(site_cfg, dict):
            continue
        if site_key in exclude:
            continue
        lang = site_cfg.get("default_language")
        if lang not in langs:
            continue
        if len(by_lang[lang]) >= sites_per_lang:
            continue
        by_lang[lang].append((site_key, site_cfg))

    print(f"\n[PLAN] Sites to process:")
    if exclude:
        print(f"    (excluded: {', '.join(sorted(exclude))})")
    for lang in sorted(by_lang.keys()):
        names = [k for k, _ in by_lang[lang]]
        print(f"    {lang}: {', '.join(names)}")

    urls = []
    for lang in sorted(by_lang.keys()):
        print(f"\n[{lang.upper()}]")
        for site_key, site_cfg in by_lang[lang]:
            match = site_cfg.get("match", [])
            domain = match[0] if match else "?"
            print(f"  [DISCOVER] {site_key} ({domain})")
            site_urls = discover_urls_for_site(engine, site_key, site_cfg, articles_per_site)
            for u in site_urls:
                urls.append({"url": u, "site": site_key, "lang": lang})
            print(f"    -> {len(site_urls)} URLs")
    return urls


# ------------------------------------------------------------------
# Per-URL comparison
# ------------------------------------------------------------------

def compare_one(engine, art):
    """Run both extractors on one URL and compute all metrics."""
    url = art["url"]
    r = {
        "url": url,
        "site": art["site"],
        "lang": art["lang"],
        "error": None,
        # Reference values
        "ref_url_ym": url_year_month(url),
        "ref_meta_date": None,
        "ref_meta_author": None,
        "ref_meta_title": None,
        # UCB
        "ucb_date": None, "ucb_author": None, "ucb_title": None,
        "ucb_category": None, "ucb_content_len": 0,
        # Trafilatura
        "traf_date": None, "traf_author": None, "traf_title": None,
        "traf_content_len": 0,
        # Gold body
        "gold_len": 0,
        # Quality metrics: UCB
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
    except Exception as e:
        r["error"] = f"fetch: {e}"
        return r

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception as e:
        r["error"] = f"parse: {e}"
        return r

    # Reference metadata
    r["ref_meta_date"] = normalize_date(meta_date(soup))
    r["ref_meta_author"] = meta_author(soup)
    r["ref_meta_title"] = meta_title(soup)

    # UCB
    ucb_content = ""
    try:
        data = engine.extract_article_fields(html, url)
        r["ucb_date"] = normalize_date(data.get("date"))
        r["ucb_author"] = data.get("author")
        r["ucb_title"] = data.get("title")
        r["ucb_category"] = data.get("category")
        ucb_content = data.get("content") or ""
        r["ucb_content_len"] = len(ucb_content)
    except Exception as e:
        r["error"] = f"ucb: {e}"

    # Trafilatura: metadata
    try:
        meta = trafilatura.extract_metadata(html)
        if meta:
            r["traf_date"] = normalize_date(meta.date)
            r["traf_author"] = meta.author
            r["traf_title"] = meta.title
    except Exception:
        pass

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
            r["traf_content_len"] = len(text)
    except Exception:
        pass

    # Gold body
    gold = extract_gold_text(soup)
    r["gold_len"] = len(gold)

    # Quality metrics for UCB
    ucb_m = content_metrics(ucb_content, gold)
    r["ucb_len_ratio"] = ucb_m["len_ratio"]
    r["ucb_word_precision"] = ucb_m["word_precision"]
    r["ucb_word_recall"] = ucb_m["word_recall"]
    r["ucb_word_f1"] = ucb_m["word_f1"]
    r["ucb_rouge_1_f1"] = ucb_m["rouge_1_f1"]
    r["ucb_rouge_l_f1"] = ucb_m["rouge_l_f1"]

    # Quality metrics for Trafilatura
    traf_m = content_metrics(traf_content, gold)
    r["traf_len_ratio"] = traf_m["len_ratio"]
    r["traf_word_precision"] = traf_m["word_precision"]
    r["traf_word_recall"] = traf_m["word_recall"]
    r["traf_word_f1"] = traf_m["word_f1"]
    r["traf_rouge_1_f1"] = traf_m["rouge_1_f1"]
    r["traf_rouge_l_f1"] = traf_m["rouge_l_f1"]

    return r


# ------------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------------

def evaluate(results):
    """Aggregate per-URL results into global, per-site, and per-language views."""
    ev = {
        "total": 0, "errors": 0,
        # Date from URL
        "total_url_date": 0,
        "ucb_url_ok": 0, "ucb_url_empty": 0, "ucb_url_wrong": 0,
        "traf_url_ok": 0, "traf_url_empty": 0, "traf_url_wrong": 0,
        # Date from meta
        "total_meta_date": 0,
        "ucb_meta_ok": 0, "ucb_meta_empty": 0, "ucb_meta_wrong": 0,
        "traf_meta_ok": 0, "traf_meta_empty": 0, "traf_meta_wrong": 0,
        # Author from meta
        "total_meta_author": 0,
        "ucb_author_ok": 0, "traf_author_ok": 0,
        # Title from meta
        "total_meta_title": 0,
        "ucb_title_ok": 0, "traf_title_ok": 0,
        # Coverage
        "ucb_has_author": 0, "traf_has_author": 0,
        "ucb_has_title": 0, "traf_has_title": 0,
        "ucb_has_date": 0, "traf_has_date": 0,
        "ucb_has_category": 0,
        # Body length
        "ucb_len": [], "traf_len": [],
        # Quality metrics: UCB
        "ucb_rouge_1": [], "ucb_rouge_l": [],
        "ucb_word_p": [], "ucb_word_r": [], "ucb_word_f1": [],
        "ucb_len_ratio": [],
        # Quality metrics: Trafilatura
        "traf_rouge_1": [], "traf_rouge_l": [],
        "traf_word_p": [], "traf_word_r": [], "traf_word_f1": [],
        "traf_len_ratio": [],
    }

    cov = defaultdict(lambda: {
        "total": 0,
        "ucb_author": 0, "traf_author": 0,
        "ucb_title": 0, "traf_title": 0,
        "ucb_date": 0, "traf_date": 0,
        "ucb_category": 0,
        "ucb_rouge_l": [], "traf_rouge_l": [],
        "ucb_word_f1": [], "traf_word_f1": [],
    })

    by_lang = defaultdict(lambda: {
        "total": 0,
        "ucb_author": 0, "traf_author": 0,
        "ucb_title": 0, "traf_title": 0,
        "ucb_date": 0, "traf_date": 0,
        "ucb_rouge_l": [], "traf_rouge_l": [],
        "ucb_word_f1": [], "traf_word_f1": [],
    })

    for r in results:
        if r["error"]:
            ev["errors"] += 1
            continue
        ev["total"] += 1

        site = r["site"]
        lang = r["lang"]
        cov[site]["total"] += 1
        by_lang[lang]["total"] += 1

        # Date from URL
        ref = r["ref_url_ym"]
        if ref:
            ev["total_url_date"] += 1
            ucb_ym = year_month(r["ucb_date"])
            traf_ym = year_month(r["traf_date"])
            if ucb_ym == ref: ev["ucb_url_ok"] += 1
            elif not ucb_ym:   ev["ucb_url_empty"] += 1
            else:              ev["ucb_url_wrong"] += 1
            if traf_ym == ref: ev["traf_url_ok"] += 1
            elif not traf_ym:   ev["traf_url_empty"] += 1
            else:               ev["traf_url_wrong"] += 1

        # Date from meta
        ref = r["ref_meta_date"]
        if ref:
            ev["total_meta_date"] += 1
            if r["ucb_date"] == ref:   ev["ucb_meta_ok"] += 1
            elif not r["ucb_date"]:     ev["ucb_meta_empty"] += 1
            else:                       ev["ucb_meta_wrong"] += 1
            if r["traf_date"] == ref:   ev["traf_meta_ok"] += 1
            elif not r["traf_date"]:    ev["traf_meta_empty"] += 1
            else:                       ev["traf_meta_wrong"] += 1

        # Author from meta
        ref = r["ref_meta_author"]
        if ref:
            ev["total_meta_author"] += 1
            if r["ucb_author"] and ref.lower()[:15] in r["ucb_author"].lower():
                ev["ucb_author_ok"] += 1
            if r["traf_author"] and ref.lower()[:15] in r["traf_author"].lower():
                ev["traf_author_ok"] += 1

        # Title from meta
        ref = r["ref_meta_title"]
        if ref:
            ev["total_meta_title"] += 1
            if r["ucb_title"] and ref[:30].lower() in r["ucb_title"].lower():
                ev["ucb_title_ok"] += 1
            if r["traf_title"] and ref[:30].lower() in r["traf_title"].lower():
                ev["traf_title_ok"] += 1

        # Coverage (global)
        if r["ucb_author"]:   ev["ucb_has_author"] += 1
        if r["traf_author"]:  ev["traf_has_author"] += 1
        if r["ucb_title"]:    ev["ucb_has_title"] += 1
        if r["traf_title"]:   ev["traf_has_title"] += 1
        if r["ucb_date"]:     ev["ucb_has_date"] += 1
        if r["traf_date"]:    ev["traf_has_date"] += 1
        if r["ucb_category"]: ev["ucb_has_category"] += 1

        # Coverage (per site)
        if r["ucb_author"]:   cov[site]["ucb_author"] += 1
        if r["traf_author"]:  cov[site]["traf_author"] += 1
        if r["ucb_title"]:    cov[site]["ucb_title"] += 1
        if r["traf_title"]:   cov[site]["traf_title"] += 1
        if r["ucb_date"]:     cov[site]["ucb_date"] += 1
        if r["traf_date"]:    cov[site]["traf_date"] += 1
        if r["ucb_category"]: cov[site]["ucb_category"] += 1

        # Coverage (per language)
        if r["ucb_author"]:   by_lang[lang]["ucb_author"] += 1
        if r["traf_author"]:  by_lang[lang]["traf_author"] += 1
        if r["ucb_title"]:    by_lang[lang]["ucb_title"] += 1
        if r["traf_title"]:   by_lang[lang]["traf_title"] += 1
        if r["ucb_date"]:     by_lang[lang]["ucb_date"] += 1
        if r["traf_date"]:    by_lang[lang]["traf_date"] += 1

        # Body length
        if r["ucb_content_len"]:  ev["ucb_len"].append(r["ucb_content_len"])
        if r["traf_content_len"]: ev["traf_len"].append(r["traf_content_len"])

        # Quality metrics: UCB
        if r["ucb_rouge_1_f1"] is not None:
            ev["ucb_rouge_1"].append(r["ucb_rouge_1_f1"])
        if r["ucb_rouge_l_f1"] is not None:
            ev["ucb_rouge_l"].append(r["ucb_rouge_l_f1"])
            cov[site]["ucb_rouge_l"].append(r["ucb_rouge_l_f1"])
            by_lang[lang]["ucb_rouge_l"].append(r["ucb_rouge_l_f1"])
        if r["ucb_word_f1"] is not None:
            ev["ucb_word_f1"].append(r["ucb_word_f1"])
            cov[site]["ucb_word_f1"].append(r["ucb_word_f1"])
            by_lang[lang]["ucb_word_f1"].append(r["ucb_word_f1"])
        if r["ucb_word_precision"] is not None:
            ev["ucb_word_p"].append(r["ucb_word_precision"])
        if r["ucb_word_recall"] is not None:
            ev["ucb_word_r"].append(r["ucb_word_recall"])
        if r["ucb_len_ratio"] is not None:
            ev["ucb_len_ratio"].append(r["ucb_len_ratio"])

        # Quality metrics: Trafilatura
        if r["traf_rouge_1_f1"] is not None:
            ev["traf_rouge_1"].append(r["traf_rouge_1_f1"])
        if r["traf_rouge_l_f1"] is not None:
            ev["traf_rouge_l"].append(r["traf_rouge_l_f1"])
            cov[site]["traf_rouge_l"].append(r["traf_rouge_l_f1"])
            by_lang[lang]["traf_rouge_l"].append(r["traf_rouge_l_f1"])
        if r["traf_word_f1"] is not None:
            ev["traf_word_f1"].append(r["traf_word_f1"])
            cov[site]["traf_word_f1"].append(r["traf_word_f1"])
            by_lang[lang]["traf_word_f1"].append(r["traf_word_f1"])
        if r["traf_word_precision"] is not None:
            ev["traf_word_p"].append(r["traf_word_precision"])
        if r["traf_word_recall"] is not None:
            ev["traf_word_r"].append(r["traf_word_recall"])
        if r["traf_len_ratio"] is not None:
            ev["traf_len_ratio"].append(r["traf_len_ratio"])

    return ev, cov, by_lang


# ------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------

def pct(num, den):
    """Format a ratio as 'n/d = P%', or an em dash when the denominator is zero."""
    return f"{num}/{den} = {num/den*100:.1f}%" if den else "—"


def print_report(ev, cov, by_lang, results, out_csv, out_cov):
    """Print all comparison tables and write both CSVs."""
    print(f"\n{'='*70}")
    print(f"Documents processed: {ev['total']}, errors: {ev['errors']}")
    print(f"{'='*70}")

    # Date
    if ev["total_url_date"]:
        print(f"\n[DATE: year-month from URL]   documents: {ev['total_url_date']}")
        print(f"  UniCorpusBuilder:  {pct(ev['ucb_url_ok'], ev['total_url_date'])}  "
              f"(empty: {ev['ucb_url_empty']}, wrong: {ev['ucb_url_wrong']})")
        print(f"  Trafilatura:       {pct(ev['traf_url_ok'], ev['total_url_date'])}  "
              f"(empty: {ev['traf_url_empty']}, wrong: {ev['traf_url_wrong']})")
    else:
        print(f"\n[DATE: year-month from URL]   no data")

    if ev["total_meta_date"]:
        print(f"\n[DATE: HTML meta tags]        documents: {ev['total_meta_date']}")
        print(f"  UniCorpusBuilder:  {pct(ev['ucb_meta_ok'], ev['total_meta_date'])}  "
              f"(empty: {ev['ucb_meta_empty']}, wrong: {ev['ucb_meta_wrong']})")
        print(f"  Trafilatura:       {pct(ev['traf_meta_ok'], ev['total_meta_date'])}  "
              f"(empty: {ev['traf_meta_empty']}, wrong: {ev['traf_meta_wrong']})")

    # Author
    n = ev["total"]
    print(f"\n[AUTHOR extracted]")
    print(f"  UniCorpusBuilder:  {pct(ev['ucb_has_author'], n)}")
    print(f"  Trafilatura:       {pct(ev['traf_has_author'], n)}")

    if ev["total_meta_author"]:
        print(f"\n[AUTHOR matches meta-author] documents: {ev['total_meta_author']}")
        print(f"  UniCorpusBuilder:  {pct(ev['ucb_author_ok'], ev['total_meta_author'])}")
        print(f"  Trafilatura:       {pct(ev['traf_author_ok'], ev['total_meta_author'])}")

    # Title
    print(f"\n[TITLE extracted]")
    print(f"  UniCorpusBuilder:  {pct(ev['ucb_has_title'], n)}")
    print(f"  Trafilatura:       {pct(ev['traf_has_title'], n)}")

    # Category
    print(f"\n[CATEGORY extracted (UCB only)]")
    print(f"  UniCorpusBuilder:  {pct(ev['ucb_has_category'], n)}")

    # Body length
    if ev["ucb_len"]:
        u_avg = sum(ev["ucb_len"]) / len(ev["ucb_len"])
        t_avg = sum(ev["traf_len"]) / len(ev["traf_len"]) if ev["traf_len"] else 0
        print(f"\n[AVERAGE BODY LENGTH]")
        print(f"  UniCorpusBuilder:  {u_avg:.0f} chars")
        print(f"  Trafilatura:       {t_avg:.0f} chars")

    # ================================================================
    # Body-extraction quality, both systems
    # ================================================================
    print(f"\n{'='*70}")
    print("[BODY-EXTRACTION QUALITY] — gold: <article> / <main> / <body>")
    print(f"{'='*70}")

    for label, key_prefix in [("UniCorpusBuilder", "ucb"), ("Trafilatura", "traf")]:
        print(f"\n  --- {label} ---")
        rl = stats_summary(ev[f"{key_prefix}_rouge_l"])
        r1 = stats_summary(ev[f"{key_prefix}_rouge_1"])
        wf = stats_summary(ev[f"{key_prefix}_word_f1"])
        wp = stats_summary(ev[f"{key_prefix}_word_p"])
        wr = stats_summary(ev[f"{key_prefix}_word_r"])
        lr = stats_summary(ev[f"{key_prefix}_len_ratio"])

        if rl:
            print(f"  ROUGE-L F1:  N = {rl['n']}  mean = {rl['mean']:.4f}  "
                  f"median = {rl['median']:.4f}  min/max = {rl['min']:.4f}/{rl['max']:.4f}")
        else:
            print(f"  ROUGE-L F1:  no data")

        if r1:
            print(f"  ROUGE-1 F1:  mean = {r1['mean']:.4f}  median = {r1['median']:.4f}")

        if wf:
            print(f"  Word-level:  P = {wp['mean']:.4f}  R = {wr['mean']:.4f}  "
                  f"F1 = {wf['mean']:.4f}  median = {wf['median']:.4f}")

        if lr:
            print(f"  Length ratio: mean = {lr['mean']:.2f}  median = {lr['median']:.2f}  "
                  f"min/max = {lr['min']:.2f}/{lr['max']:.2f}")

    # Coverage by site
    print(f"\n{'='*70}")
    print("[COVERAGE BY SITE]")
    print(f"{'='*70}")
    print(f"  {'Site':<18} {'N':>4} {'AuthUCB':>8} {'AuthTF':>7} "
          f"{'RL UCB':>8} {'RL TF':>8} {'WF1 UCB':>8} {'WF1 TF':>8}")
    print("  " + "-" * 78)

    cov_rows = []
    for site in sorted(cov.keys()):
        c = cov[site]
        tot = c["total"]
        if tot == 0:
            continue

        rl_ucb = stats_summary(c["ucb_rouge_l"])
        rl_tf = stats_summary(c["traf_rouge_l"])
        wf_ucb = stats_summary(c["ucb_word_f1"])
        wf_tf = stats_summary(c["traf_word_f1"])

        rl_u = f"{rl_ucb['mean']:.3f}" if rl_ucb else "—"
        rl_t = f"{rl_tf['mean']:.3f}" if rl_tf else "—"
        wf_u = f"{wf_ucb['mean']:.3f}" if wf_ucb else "—"
        wf_t = f"{wf_tf['mean']:.3f}" if wf_tf else "—"

        print(f"  {site:<18} {tot:>4} "
              f"{c['ucb_author']/tot*100:>7.1f}% "
              f"{c['traf_author']/tot*100:>6.1f}% "
              f"{rl_u:>8} {rl_t:>8} {wf_u:>8} {wf_t:>8}")

        cov_rows.append({
            "site": site, "articles": tot,
            "ucb_author_pct": round(c["ucb_author"]/tot*100, 1),
            "traf_author_pct": round(c["traf_author"]/tot*100, 1),
            "ucb_date_pct": round(c["ucb_date"]/tot*100, 1),
            "traf_date_pct": round(c["traf_date"]/tot*100, 1),
            "ucb_title_pct": round(c["ucb_title"]/tot*100, 1),
            "traf_title_pct": round(c["traf_title"]/tot*100, 1),
            "ucb_rouge_l_mean": round(rl_ucb["mean"], 4) if rl_ucb else None,
            "traf_rouge_l_mean": round(rl_tf["mean"], 4) if rl_tf else None,
            "ucb_word_f1_mean": round(wf_ucb["mean"], 4) if wf_ucb else None,
            "traf_word_f1_mean": round(wf_tf["mean"], 4) if wf_tf else None,
            "ucb_rouge_l_n": rl_ucb["n"] if rl_ucb else 0,
            "traf_rouge_l_n": rl_tf["n"] if rl_tf else 0,
        })

    # By language
    print(f"\n{'='*70}")
    print("[BY LANGUAGE]")
    print(f"{'='*70}")
    print(f"  {'Lang':<6} {'N':>5} {'AuthUCB':>8} {'AuthTF':>8} "
          f"{'RL UCB':>8} {'RL TF':>8} {'WF1 UCB':>8} {'WF1 TF':>8}")
    print("  " + "-" * 74)

    lang_rows = []
    for lang in sorted(by_lang.keys()):
        c = by_lang[lang]
        tot = c["total"]
        if tot == 0:
            continue

        rl_ucb = stats_summary(c["ucb_rouge_l"])
        rl_tf = stats_summary(c["traf_rouge_l"])
        wf_ucb = stats_summary(c["ucb_word_f1"])
        wf_tf = stats_summary(c["traf_word_f1"])

        rl_u = f"{rl_ucb['mean']:.3f}" if rl_ucb else "—"
        rl_t = f"{rl_tf['mean']:.3f}" if rl_tf else "—"
        wf_u = f"{wf_ucb['mean']:.3f}" if wf_ucb else "—"
        wf_t = f"{wf_tf['mean']:.3f}" if wf_tf else "—"

        print(f"  {lang:<6} {tot:>5} "
              f"{c['ucb_author']/tot*100:>7.1f}% "
              f"{c['traf_author']/tot*100:>7.1f}% "
              f"{rl_u:>8} {rl_t:>8} {wf_u:>8} {wf_t:>8}")

        lang_rows.append({
            "lang": lang, "articles": tot,
            "ucb_author_pct": round(c["ucb_author"]/tot*100, 1),
            "traf_author_pct": round(c["traf_author"]/tot*100, 1),
            "ucb_date_pct": round(c["ucb_date"]/tot*100, 1),
            "traf_date_pct": round(c["traf_date"]/tot*100, 1),
            "ucb_rouge_l_mean": round(rl_ucb["mean"], 4) if rl_ucb else None,
            "traf_rouge_l_mean": round(rl_tf["mean"], 4) if rl_tf else None,
            "ucb_word_f1_mean": round(wf_ucb["mean"], 4) if wf_ucb else None,
            "traf_word_f1_mean": round(wf_tf["mean"], 4) if wf_tf else None,
        })

    # Final table (for the article)
    print(f"\n{'='*70}")
    print("[FINAL TABLE FOR THE ARTICLE]")
    print(f"{'='*70}")
    print(f"  {'Metric':<40} {'UniCorpusBuilder':>18} {'Trafilatura':>14}")
    print("  " + "-" * 74)
    print(f"  {'Title extracted':<40} "
          f"{ev['ucb_has_title']/n*100:>17.1f}% "
          f"{ev['traf_has_title']/n*100:>13.1f}%")
    print(f"  {'Author extracted':<40} "
          f"{ev['ucb_has_author']/n*100:>17.1f}% "
          f"{ev['traf_has_author']/n*100:>13.1f}%")
    if ev["total_meta_date"]:
        print(f"  {'Date matches meta (N=' + str(ev['total_meta_date']) + ')':<40} "
              f"{ev['ucb_meta_ok']/ev['total_meta_date']*100:>17.1f}% "
              f"{ev['traf_meta_ok']/ev['total_meta_date']*100:>13.1f}%")
    if ev["total_url_date"]:
        print(f"  {'Date matches URL (N=' + str(ev['total_url_date']) + ')':<40} "
              f"{ev['ucb_url_ok']/ev['total_url_date']*100:>17.1f}% "
              f"{ev['traf_url_ok']/ev['total_url_date']*100:>13.1f}%")
    print(f"  {'Category extracted':<40} "
          f"{ev['ucb_has_category']/n*100:>17.1f}% "
          f"{'—':>14}")
    if ev["ucb_len"]:
        u_avg = sum(ev["ucb_len"]) / len(ev["ucb_len"])
        t_avg = sum(ev["traf_len"]) / len(ev["traf_len"]) if ev["traf_len"] else 0
        print(f"  {'Average body length (chars)':<40} "
              f"{u_avg:>17.0f} "
              f"{t_avg:>14.0f}")

    rl_ucb = stats_summary(ev["ucb_rouge_l"])
    rl_tf = stats_summary(ev["traf_rouge_l"])
    r1_ucb = stats_summary(ev["ucb_rouge_1"])
    r1_tf = stats_summary(ev["traf_rouge_1"])
    wf_ucb = stats_summary(ev["ucb_word_f1"])
    wf_tf = stats_summary(ev["traf_word_f1"])
    wp_ucb = stats_summary(ev["ucb_word_p"])
    wp_tf = stats_summary(ev["traf_word_p"])
    wr_ucb = stats_summary(ev["ucb_word_r"])
    wr_tf = stats_summary(ev["traf_word_r"])
    lr_ucb = stats_summary(ev["ucb_len_ratio"])
    lr_tf = stats_summary(ev["traf_len_ratio"])

    if rl_ucb and rl_tf:
        print(f"  {'ROUGE-L F1 (N=' + str(rl_ucb['n']) + ')':<40} "
              f"{rl_ucb['mean']:>17.4f} {rl_tf['mean']:>14.4f}")
    if r1_ucb and r1_tf:
        print(f"  {'ROUGE-1 F1':<40} "
              f"{r1_ucb['mean']:>17.4f} {r1_tf['mean']:>14.4f}")
    if wp_ucb and wp_tf:
        print(f"  {'Word-level Precision':<40} "
              f"{wp_ucb['mean']:>17.4f} {wp_tf['mean']:>14.4f}")
    if wr_ucb and wr_tf:
        print(f"  {'Word-level Recall':<40} "
              f"{wr_ucb['mean']:>17.4f} {wr_tf['mean']:>14.4f}")
    if wf_ucb and wf_tf:
        print(f"  {'Word-level F1':<40} "
              f"{wf_ucb['mean']:>17.4f} {wf_tf['mean']:>14.4f}")
    if lr_ucb and lr_tf:
        print(f"  {'Length ratio':<40} "
              f"{lr_ucb['mean']:>17.2f} {lr_tf['mean']:>14.2f}")

    # CSV: per-URL
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        if results:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
    print(f"\n[OK] Per-URL CSV:       {out_csv}")

    # CSV: coverage by site
    with open(out_cov, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "site", "articles",
            "ucb_author_pct", "traf_author_pct",
            "ucb_date_pct", "traf_date_pct",
            "ucb_title_pct", "traf_title_pct",
            "ucb_rouge_l_mean", "traf_rouge_l_mean",
            "ucb_rouge_l_n", "traf_rouge_l_n",
            "ucb_word_f1_mean", "traf_word_f1_mean",
        ])
        w.writeheader()
        w.writerows(cov_rows)
    print(f"[OK] Coverage by site:  {out_cov}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/universal.yaml")
    ap.add_argument("--out", default="trafilatura_comparison.csv")
    ap.add_argument("--out-cov", default="trafilatura_coverage.csv")
    ap.add_argument("--sleep", type=float, default=0.3)

    ap.add_argument("--langs", nargs="+", default=["tg", "tt", "ba", "os"])
    ap.add_argument("--sites-per-lang", type=int, default=2)
    ap.add_argument("--articles-per-site", type=int, default=20)
    ap.add_argument("--exclude", nargs="*", default=["sssr", "ozodi"])

    args = ap.parse_args()

    engine = ExtractionEngine(yaml_path=args.config)

    print(f"[MODE] Languages={args.langs}, "
          f"{args.sites_per_lang} sites/language, "
          f"{args.articles_per_site} articles/site")

    urls = gather_from_config(
        engine, args.langs, args.sites_per_lang,
        args.articles_per_site, exclude=args.exclude
    )
    if not urls:
        print("[ERROR] No URLs collected.")
        return

    print(f"\n[TOTAL] URLs: {len(urls)}")
    print("[FETCH] Comparing UniCorpusBuilder vs Trafilatura + ROUGE (both systems)...\n")

    results = []
    for art in tqdm(urls, desc="[PROCESS]", unit="doc"):
        results.append(compare_one(engine, art))
        time.sleep(args.sleep)

    ev, cov, by_lang = evaluate(results)
    print_report(ev, cov, by_lang, results, args.out, args.out_cov)


if __name__ == "__main__":
    main()