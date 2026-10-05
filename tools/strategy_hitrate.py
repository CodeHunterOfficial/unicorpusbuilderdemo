# tools/strategy_hitrate.py
"""
Measure strategy-level precision and cost, plus body-extraction quality.

Per field (author, title, date, category, content) the script reports:
  * first-matching strategy distribution;
  * per-strategy precision p — fraction of correct extractions among
    firings, evaluated against meta-tag references;
  * per-strategy cost c — mean wall-clock time per document attempt;
  * score = p / c, used to justify the chain order in the article.

Body-extraction quality for UniCorpusBuilder is measured against an
independently built gold text.

URL sources
-----------
Config mode — sites selected by language from ``config/universal.yaml``;
File mode   — URLs read from local JSONL files.

Usage
-----
    python tools/strategy_hitrate.py
    python tools/strategy_hitrate.py --langs tg tt ba os --sites-per-lang 2 --articles-per-site 20
    python tools/strategy_hitrate.py --exclude sssr ozodi
    python tools/strategy_hitrate.py --input-files "D:\\...\\file1.jsonl" "D:\\...\\file2.jsonl"

Installation
------------
    pip install trafilatura rouge-score python-dateutil
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from tqdm import tqdm

from config.loader import load_modular_config
from pipeline.pipeline_extraction import ExtractionEngine

from _common import WORD_RE, discover_urls_for_site, stats_summary


try:
    from rouge_score import rouge_scorer
    _SCORER = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
except ImportError:
    _SCORER = None
    print("[WARN] rouge-score is not installed. ROUGE will not be computed.")
    print("       Install with: pip install rouge-score")


# ------------------------------------------------------------------
# Text metrics
# ------------------------------------------------------------------

def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _tokenize(s: str) -> list[str]:
    return WORD_RE.findall((s or "").lower())


def extract_gold_text(soup: BeautifulSoup) -> str:
    """
    Return the gold article body used to score UniCorpusBuilder.

    Resolution order:
      1. A fixed list of narrow selectors targeting common CMS layouts.
      2. <article> / <main> / [role=main].
      3. <body> with boilerplate tags removed.
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


def word_prf1(pred: str, gold: str) -> tuple[float, float, float]:
    pred_set = set(_tokenize(pred))
    gold_set = set(_tokenize(gold))
    if not pred_set or not gold_set:
        return 0.0, 0.0, 0.0
    inter = pred_set & gold_set
    p = len(inter) / len(pred_set)
    r = len(inter) / len(gold_set)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def compute_quality(gold: str, pred: str) -> dict:
    out = {
        "gold_len": len(gold or ""),
        "pred_len": len(pred or ""),
        "len_ratio": None,
        "word_precision": None,
        "word_recall": None,
        "word_f1": None,
        "rouge_1_f1": None,
        "rouge_l_f1": None,
    }
    if len(gold or "") < 300 or not pred:
        return out

    out["len_ratio"] = round(len(pred) / len(gold), 3)

    wp, wr, wf1 = word_prf1(pred, gold)
    out["word_precision"] = round(wp, 4)
    out["word_recall"] = round(wr, 4)
    out["word_f1"] = round(wf1, 4)

    if _SCORER is not None:
        try:
            scores = _SCORER.score(gold, pred)
            out["rouge_1_f1"] = round(scores["rouge1"].fmeasure, 4)
            out["rouge_l_f1"] = round(scores["rougeL"].fmeasure, 4)
        except Exception:
            pass

    return out


# ------------------------------------------------------------------
# Reference metadata (meta tags) and correctness checks
# ------------------------------------------------------------------

def ref_meta_author(soup: BeautifulSoup) -> Optional[str]:
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


def ref_meta_date(soup: BeautifulSoup) -> Optional[str]:
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


def ref_meta_title(soup: BeautifulSoup) -> Optional[str]:
    for prop in ["og:title", "twitter:title"]:
        tag = soup.find("meta", attrs={"property": prop})
        if tag and tag.get("content"):
            return tag["content"]
    tag = soup.find("meta", attrs={"name": "title"})
    if tag and tag.get("content"):
        return tag["content"]
    return None


def _author_correct(extracted: str, ref: str) -> bool:
    if not ref or not extracted:
        return False
    return ref.lower()[:15] in extracted.lower()


def _date_correct(extracted: str, ref: str) -> bool:
    if not ref or not extracted:
        return False
    try:
        d1 = dateparser.parse(str(extracted), fuzzy=True)
        d2 = dateparser.parse(str(ref), fuzzy=True)
        if not d1 or not d2:
            return False
        return d1.date() == d2.date()
    except Exception:
        return False


def _title_correct(extracted: str, ref: str) -> bool:
    if not ref or not extracted:
        return False
    return ref[:30].lower() in extracted.lower()


# ------------------------------------------------------------------
# Strategy tracker
# ------------------------------------------------------------------

@dataclass
class ProbeResult:
    strategy: Optional[str] = None
    value: Optional[str] = None
    time_ms: float = 0.0
    is_correct: Optional[bool] = None


class StrategyTracker:
    """
    Collect per-strategy precision and cost.

    For each (field, strategy) pair, accumulate:
      attempts        — documents where the strategy was invoked;
      fired           — invocations that returned a value;
      ref_available   — firings where a meta-tag reference existed;
      correct         — firings whose value matched the reference;
      time_all_total  — wall-clock milliseconds across all attempts;
      time_fire_total — wall-clock milliseconds on firings only.

    Derived:
      p        = correct / ref_available       (precision among firings);
      c        = time_all_total / attempts     (mean ms per document);
      score    = p / c.
    """

    FIELDS = ("author", "title", "date", "category", "content")

    def __init__(self) -> None:
        self.data: dict[tuple[str, str], dict] = defaultdict(
            lambda: {
                "attempts": 0,
                "fired": 0,
                "correct": 0,
                "ref_available": 0,
                "time_all_total": 0.0,
                "time_fire_total": 0.0,
            }
        )

    def record(
        self,
        field: str,
        strategy: str,
        time_ms: float,
        is_correct: Optional[bool],
        fired: bool,
    ) -> None:
        d = self.data[(field, strategy)]
        d["attempts"] += 1
        d["time_all_total"] += time_ms
        if fired:
            d["fired"] += 1
            d["time_fire_total"] += time_ms
            if is_correct is not None:
                d["ref_available"] += 1
                if is_correct:
                    d["correct"] += 1

    def rows(self) -> list[dict]:
        rows = []
        for (field, strategy), d in sorted(self.data.items()):
            p = (d["correct"] / d["ref_available"]) if d["ref_available"] else None
            c_all = (d["time_all_total"] / d["attempts"]) if d["attempts"] else None
            c_fire = (d["time_fire_total"] / d["fired"]) if d["fired"] else None
            score = (p / c_all) if (p is not None and c_all and c_all > 0) else None
            rows.append({
                "field": field,
                "strategy": strategy,
                "attempts": d["attempts"],
                "fired": d["fired"],
                "ref_available": d["ref_available"],
                "correct": d["correct"],
                "p": round(p, 4) if p is not None else None,
                "c_all_ms": round(c_all, 4) if c_all is not None else None,
                "c_fire_ms": round(c_fire, 4) if c_fire is not None else None,
                "p_over_c": round(score, 4) if score is not None else None,
            })
        return rows


# ------------------------------------------------------------------
# URL collection
# ------------------------------------------------------------------

def gather_from_config(engine, langs, sites_per_lang, articles_per_site,
                       min_strategies, exclude=None):
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
        n_strat = len(site_cfg.get("author_strategy", []) or [])
        if n_strat < min_strategies:
            continue
        if len(by_lang[lang]) >= sites_per_lang:
            continue
        by_lang[lang].append((site_key, site_cfg, n_strat))

    print(f"\n[PLAN] Sites to process by language:")
    if exclude:
        print(f"    (excluded: {', '.join(sorted(exclude))})")
    for lang in sorted(by_lang.keys()):
        names = [k for k, _, _ in by_lang[lang]]
        print(f"    {lang}: {', '.join(names)}")

    urls = []
    for lang in sorted(by_lang.keys()):
        print(f"\n[{lang.upper()}]")
        for site_key, site_cfg, n_strat in by_lang[lang]:
            match = site_cfg.get("match", [])
            domain = match[0] if match else "?"
            print(f"  [DISCOVER] {site_key} ({domain}) — {n_strat} author strategies")
            site_urls = discover_urls_for_site(engine, site_key, site_cfg, articles_per_site)
            for u in site_urls:
                urls.append({"url": u, "site": site_key, "lang": lang})
            print(f"    -> {len(site_urls)} URLs")
    return urls


def iter_articles_from_files(files, per_site):
    seen = set()
    counter = defaultdict(int)
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                url = obj.get("url")
                if not url or url in seen:
                    continue
                site = obj.get("site") or f.stem.split("-")[0]
                if counter[site] >= per_site:
                    continue
                seen.add(url)
                counter[site] += 1
                yield {"url": url, "site": site, "lang": obj.get("language", "?")}


# ------------------------------------------------------------------
# Per-field strategy probes
# ------------------------------------------------------------------

_AUTHOR_LABEL_RE = re.compile(
    r"^(Author|By|From the author|Author)\s*[:\-]?\s*", re.I
)


def _clean_probe_value(raw: Optional[str], max_len: int = 300) -> Optional[str]:
    """Reject JSON blobs and over-long strings; strip leading labels."""
    if not raw:
        return None
    s = raw.strip()
    if not s or len(s) > max_len:
        return None
    if s.startswith(("{", "[")):
        return None
    return s


def probe_author(
    engine,
    soup: BeautifulSoup,
    site_cfg: dict,
    url: str,
    tracker: Optional[StrategyTracker] = None,
    ref: Optional[str] = None,
) -> ProbeResult:
    strategies = (
        site_cfg.get("author_strategy")
        or engine.global_cfg().get("author_strategies_order")
        or []
    )
    strategies = list(dict.fromkeys(strategies))
    real = [s for s in strategies if s != "default_fallback"]

    for strat in real:
        tmp = dict(site_cfg)
        tmp["author_strategy"] = [strat]
        tmp["default_author"] = None
        tmp["author_priority"] = None

        t0 = time.perf_counter()
        try:
            raw = engine.extract_author_from_soup(soup, tmp, url)
        except Exception:
            raw = None
        dt_ms = (time.perf_counter() - t0) * 1000

        value = _clean_probe_value(raw, max_len=300)
        if value:
            value = _AUTHOR_LABEL_RE.sub("", value).strip() or None

        if tracker is not None:
            if value:
                correct = _author_correct(value, ref) if ref else None
                tracker.record("author", strat, dt_ms, correct, fired=True)
            else:
                tracker.record("author", strat, dt_ms, None, fired=False)

        if value:
            return ProbeResult(strat, value, dt_ms, None)

    if "default_fallback" in strategies and site_cfg.get("default_author"):
        value = _clean_probe_value(site_cfg["default_author"], max_len=300)
        if value and tracker is not None:
            correct = _author_correct(value, ref) if ref else None
            tracker.record("author", "default_fallback", 0.0, correct, fired=True)
        return ProbeResult("default_fallback", value, 0.0, None)

    return ProbeResult()


def probe_title(
    engine,
    soup: BeautifulSoup,
    site_cfg: dict,
    tracker: Optional[StrategyTracker] = None,
    ref: Optional[str] = None,
) -> ProbeResult:
    for sel in site_cfg.get("title_selectors", []):
        t0 = time.perf_counter()
        try:
            el = engine._safe_select_one(soup, sel)
        except Exception:
            el = None
        dt_ms = (time.perf_counter() - t0) * 1000

        value = None
        if el:
            if el.name == "meta" and el.get("content"):
                value = _clean_probe_value(el.get("content"))
            else:
                value = _clean_probe_value(el.get_text(" ", strip=True))

        if tracker is not None:
            if value:
                correct = _title_correct(value, ref) if ref else None
                tracker.record("title", sel, dt_ms, correct, fired=True)
            else:
                tracker.record("title", sel, dt_ms, None, fired=False)

        if value:
            return ProbeResult(sel, value, dt_ms, None)

    if soup.title and soup.title.string and soup.title.string.strip():
        value = _clean_probe_value(soup.title.string)
        if tracker is not None and value:
            correct = _title_correct(value, ref) if ref else None
            tracker.record("title", "<title>", 0.0, correct, fired=True)
        if value:
            return ProbeResult("<title>", value, 0.0, None)

    return ProbeResult()


def probe_date(
    engine,
    soup: BeautifulSoup,
    site_cfg: dict,
    url: str,
    tracker: Optional[StrategyTracker] = None,
    ref: Optional[str] = None,
) -> ProbeResult:
    from pipeline.pipeline_core import parse_datetime_value
    locale_map = engine.get_date_locale_map(url)

    t0 = time.perf_counter()
    try:
        jld = engine.extract_jsonld_date(soup)
    except Exception:
        jld = None
    dt_ms = (time.perf_counter() - t0) * 1000

    if jld:
        parsed = parse_datetime_value(jld, locale_map)
        if parsed:
            if tracker is not None:
                correct = _date_correct(parsed, ref) if ref else None
                tracker.record("date", "json_ld:datePublished", dt_ms, correct, fired=True)
            return ProbeResult("json_ld:datePublished", parsed, dt_ms, None)
    if tracker is not None:
        tracker.record("date", "json_ld:datePublished", dt_ms, None, fired=False)

    for sel in site_cfg.get("date_selectors", []):
        if "ld+json" in sel:
            continue
        t0 = time.perf_counter()
        try:
            el = engine._safe_select_one(soup, sel)
        except Exception:
            el = None
        dt_ms = (time.perf_counter() - t0) * 1000

        parsed = None
        if el and el.name != "script":
            val = el.get("content") or el.get("datetime") or el.get_text(" ", strip=True)
            if val:
                parsed = parse_datetime_value(val, locale_map)

        if tracker is not None:
            if parsed:
                correct = _date_correct(parsed, ref) if ref else None
                tracker.record("date", sel, dt_ms, correct, fired=True)
            else:
                tracker.record("date", sel, dt_ms, None, fired=False)

        if parsed:
            return ProbeResult(sel, parsed, dt_ms, None)

    return ProbeResult()


def probe_category(
    engine,
    soup: BeautifulSoup,
    url: str,
    site_cfg: dict,
    tracker: Optional[StrategyTracker] = None,
) -> ProbeResult:
    strategies = site_cfg.get("category_strategy") or []

    for strat in strategies:
        if strat == "meta_tag":
            for sel in site_cfg.get("category_selectors", []):
                t0 = time.perf_counter()
                try:
                    el = engine._safe_select_one(soup, sel)
                except Exception:
                    el = None
                dt_ms = (time.perf_counter() - t0) * 1000
                value = None
                if el:
                    raw = el.get("content") if el.name == "meta" else el.get_text(" ", strip=True)
                    value = _clean_probe_value(raw, max_len=200)
                label = f"{strat}:{sel[:40]}"
                if tracker is not None:
                    tracker.record("category", label, dt_ms, None, fired=bool(value))
                if value:
                    return ProbeResult(label, value, dt_ms, None)

        elif strat == "url_path_parsing":
            t0 = time.perf_counter()
            try:
                cat = engine.extract_category(soup, url, site_cfg)
            except Exception:
                cat = None
            dt_ms = (time.perf_counter() - t0) * 1000
            value = _clean_probe_value(cat, max_len=200)
            if tracker is not None:
                tracker.record("category", strat, dt_ms, None, fired=bool(value))
            if value:
                return ProbeResult(strat, value, dt_ms, None)

        elif strat == "breadcrumb":
            t0 = time.perf_counter()
            value = None
            for el in soup.select(".breadcrumb, .breadcrumbs, [class*=breadcrumb]"):
                txt = el.get_text(" ", strip=True)
                value = _clean_probe_value(txt, max_len=200)
                if value:
                    break
            dt_ms = (time.perf_counter() - t0) * 1000
            if tracker is not None:
                tracker.record("category", strat, dt_ms, None, fired=bool(value))
            if value:
                return ProbeResult(strat, value, dt_ms, None)

        elif strat == "context_passed":
            continue

    t0 = time.perf_counter()
    try:
        cat = engine.extract_category(soup, url, site_cfg)
    except Exception:
        cat = None
    dt_ms = (time.perf_counter() - t0) * 1000
    value = _clean_probe_value(cat, max_len=200)
    if tracker is not None:
        tracker.record("category", "fallback", dt_ms, None, fired=bool(value))
    if value:
        return ProbeResult("fallback", value, dt_ms, None)

    return ProbeResult()


def probe_content(
    engine,
    soup: BeautifulSoup,
    site_cfg: dict,
    tracker: Optional[StrategyTracker] = None,
) -> ProbeResult:
    for sel in site_cfg.get("content_selectors", []):
        t0 = time.perf_counter()
        try:
            node = engine._safe_select_one(soup, sel)
        except Exception:
            node = None
        dt_ms = (time.perf_counter() - t0) * 1000
        value = None
        if node:
            txt = node.get_text(" ", strip=True)
            if txt and len(txt) > 200:
                value = sel
        if tracker is not None:
            tracker.record("content", sel, dt_ms, None, fired=bool(value))
        if value:
            return ProbeResult(sel, value, dt_ms, None)

    t0 = time.perf_counter()
    container = None
    try:
        container = engine.find_best_content_container(soup, site_cfg)
    except Exception:
        container = None
    dt_ms = (time.perf_counter() - t0) * 1000
    value = "auto:heuristic" if (container is not None and container != soup) else None
    if tracker is not None:
        tracker.record("content", "auto:heuristic", dt_ms, None, fired=bool(value))
    if value:
        return ProbeResult("auto:heuristic", value, dt_ms, None)

    return ProbeResult()


# ------------------------------------------------------------------
# Main analysis loop
# ------------------------------------------------------------------

def run_analysis(engine, urls_to_process, sleep):
    stats = {name: Counter() for name in ("author", "title", "date", "category", "content")}
    coverage_total = Counter()
    coverage_ok = defaultdict(lambda: Counter())
    tracker = StrategyTracker()

    quality_by_site = defaultdict(lambda: {
        "rouge_l": [], "rouge_1": [], "word_f1": [],
        "word_p": [], "word_r": [], "len_ratio": [],
    })
    quality_by_lang = defaultdict(lambda: {
        "rouge_l": [], "rouge_1": [], "word_f1": [],
        "word_p": [], "word_r": [], "len_ratio": [],
    })

    total = 0
    failed_fetch = 0

    print(f"\n[FETCH] Processing {len(urls_to_process)} URLs...\n")

    for art in tqdm(urls_to_process, desc="[PROCESS]", unit="doc"):
        url = art["url"]
        site = art["site"]
        lang = art.get("lang", "?")

        try:
            html = engine.fetch_html(url)
        except Exception:
            failed_fetch += 1
            continue

        try:
            soup = BeautifulSoup(html, "html.parser")
            site_cfg = engine.site_cfg(url)
        except Exception:
            failed_fetch += 1
            continue

        ref_a = ref_meta_author(soup)
        ref_d = ref_meta_date(soup)
        ref_t = ref_meta_title(soup)

        coverage_total[site] += 1

        a = probe_author(engine, soup, site_cfg, url, tracker, ref_a)
        stats["author"][a.strategy or "[missed]"] += 1
        if a.strategy:
            coverage_ok[site]["author"] += 1

        t = probe_title(engine, soup, site_cfg, tracker, ref_t)
        stats["title"][t.strategy or "[missed]"] += 1
        if t.strategy:
            coverage_ok[site]["title"] += 1

        d = probe_date(engine, soup, site_cfg, url, tracker, ref_d)
        stats["date"][d.strategy or "[missed]"] += 1
        if d.strategy:
            coverage_ok[site]["date"] += 1

        c = probe_category(engine, soup, url, site_cfg, tracker)
        stats["category"][c.strategy or "[missed]"] += 1
        if c.strategy:
            coverage_ok[site]["category"] += 1

        ct = probe_content(engine, soup, site_cfg, tracker)
        stats["content"][ct.strategy or "[missed]"] += 1
        if ct.strategy:
            coverage_ok[site]["content"] += 1

        try:
            data = engine.extract_article_fields(html, url)
            ucb_content = data.get("content") or ""
        except Exception:
            ucb_content = ""

        gold = extract_gold_text(soup)
        q = compute_quality(gold, ucb_content)

        if q["rouge_l_f1"] is not None:
            quality_by_site[site]["rouge_l"].append(q["rouge_l_f1"])
            quality_by_lang[lang]["rouge_l"].append(q["rouge_l_f1"])
        if q["rouge_1_f1"] is not None:
            quality_by_site[site]["rouge_1"].append(q["rouge_1_f1"])
            quality_by_lang[lang]["rouge_1"].append(q["rouge_1_f1"])
        if q["word_f1"] is not None:
            quality_by_site[site]["word_f1"].append(q["word_f1"])
            quality_by_lang[lang]["word_f1"].append(q["word_f1"])
            quality_by_site[site]["word_p"].append(q["word_precision"])
            quality_by_site[site]["word_r"].append(q["word_recall"])
            quality_by_lang[lang]["word_p"].append(q["word_precision"])
            quality_by_lang[lang]["word_r"].append(q["word_recall"])
        if q["len_ratio"] is not None:
            quality_by_site[site]["len_ratio"].append(q["len_ratio"])
            quality_by_lang[lang]["len_ratio"].append(q["len_ratio"])

        total += 1
        time.sleep(sleep)

    return (stats, coverage_total, coverage_ok, quality_by_site,
            quality_by_lang, total, failed_fetch, tracker)


# ------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------

def print_stats(name, counter, total, top=8):
    print(f"\n--- {name.upper()} ---")
    if not counter:
        print("  (no data)")
        return []
    print(f"  {'Value':<45} {'Hits':>10} {'Share':>8}")
    print("  " + "-" * 65)

    rows = []
    for val, n in counter.most_common():
        if val == "[missed]":
            continue
        share = n / total * 100 if total else 0
        print(f"  {val[:45]:<45} {n:>10} {share:>7.1f}%")
        rows.append({"metric": name, "value": val, "hits": n,
                     "share_pct": round(share, 2)})

    missed = counter.get("[missed]", 0)
    if missed:
        share = missed / total * 100 if total else 0
        print(f"  {'[no strategy matched]':<45} {missed:>10} {share:>7.1f}%")
        rows.append({"metric": name, "value": "[missed]",
                     "hits": missed, "share_pct": round(share, 2)})
    return rows


def print_strategy_scores(rows: list[dict]) -> None:
    print(f"\n{'='*70}")
    print("[STRATEGY PRECISION AND COST]")
    print(f"{'='*70}")
    print("  p = correct / ref_available   (precision among firings)")
    print("  c = time_all_total / attempts (mean ms per document attempt)")
    print("  score = p / c\n")
    print(f"  {'Field':<10} {'Strategy':<28} {'Fired':>6} {'Ref':>5} "
          f"{'Corr':>5} {'p':>7} {'c, ms':>8} {'p/c':>7}")
    print("  " + "-" * 82)

    for r in rows:
        p_str = f"{r['p']:.3f}" if r["p"] is not None else "—"
        c_str = f"{r['c_all_ms']:.3f}" if r["c_all_ms"] is not None else "—"
        s_str = f"{r['p_over_c']:.3f}" if r["p_over_c"] is not None else "—"
        print(f"  {r['field']:<10} {r['strategy'][:28]:<28} "
              f"{r['fired']:>6} {r['ref_available']:>5} {r['correct']:>5} "
              f"{p_str:>7} {c_str:>8} {s_str:>7}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/universal.yaml")
    ap.add_argument("--out", default="strategy_hitrate.csv")
    ap.add_argument("--out-coverage", default="strategy_coverage.csv")
    ap.add_argument("--out-quality", default="strategy_quality.csv")
    ap.add_argument("--out-scores", default="strategy_scores.csv")
    ap.add_argument("--sleep", type=float, default=0.3)

    ap.add_argument("--langs", nargs="+", default=["tg", "tt", "ba", "os"])
    ap.add_argument("--sites-per-lang", type=int, default=2)
    ap.add_argument("--articles-per-site", type=int, default=20)
    ap.add_argument("--min-strategies", type=int, default=1)
    ap.add_argument("--exclude", nargs="*", default=["sssr", "ozodi"])

    ap.add_argument("--input-files", nargs="*", default=None)
    ap.add_argument("--input-dir", default=None)
    ap.add_argument("--per-site", type=int, default=30)
    ap.add_argument("--limit", type=int, default=300)

    args = ap.parse_args()

    engine = ExtractionEngine(yaml_path=args.config)

    if args.input_files or args.input_dir:
        if args.input_files:
            files = [Path(p) for p in args.input_files]
        else:
            files = sorted(Path(args.input_dir).glob("*.jsonl"))
        files = [f for f in files if f.exists()]
        if not files:
            print("[ERROR] No JSONL files found.")
            return
        print(f"[FILES] Found: {len(files)} files")
        for f in files:
            print(f"        {f.name}")

        articles = list(iter_articles_from_files(files, args.per_site))
        if len(articles) > args.limit:
            articles = articles[:args.limit]
        urls_to_process = articles
    else:
        print(f"[MODE] Config mode: languages={args.langs}, "
              f"{args.sites_per_lang} sites/language, "
              f"{args.articles_per_site} articles/site")
        urls_to_process = gather_from_config(
            engine,
            langs=args.langs,
            sites_per_lang=args.sites_per_lang,
            articles_per_site=args.articles_per_site,
            min_strategies=args.min_strategies,
            exclude=args.exclude,
        )

    if not urls_to_process:
        print("[ERROR] No URLs collected.")
        return

    print(f"\n[TOTAL] URLs to process: {len(urls_to_process)}")

    (stats, coverage_total, coverage_ok, quality_by_site,
     quality_by_lang, total, failed_fetch, tracker) = run_analysis(
        engine, urls_to_process, args.sleep
    )

    print(f"\n{'='*70}")
    print(f"Processed:      {total}")
    print(f"Fetch failures: {failed_fetch}")
    print(f"{'='*70}")

    all_rows = []
    for name in ("author", "title", "date", "category", "content"):
        rows = print_stats(name, stats[name], total, top=8)
        if rows:
            all_rows.extend(rows)

    print(f"\n{'='*70}")
    print("[BODY-EXTRACTION QUALITY] — gold: <article> / <main> / <body>")
    print(f"{'='*70}")

    all_rl, all_r1, all_wf, all_wp, all_wr, all_lr = [], [], [], [], [], []
    for _s, d in quality_by_site.items():
        all_rl.extend(d["rouge_l"])
        all_r1.extend(d["rouge_1"])
        all_wf.extend(d["word_f1"])
        all_wp.extend(d["word_p"])
        all_wr.extend(d["word_r"])
        all_lr.extend(d["len_ratio"])

    rl = stats_summary(all_rl)
    r1 = stats_summary(all_r1)
    wf = stats_summary(all_wf)
    wp = stats_summary(all_wp)
    wr = stats_summary(all_wr)
    lr = stats_summary(all_lr)

    if rl:
        print(f"\n  ROUGE-L F1:")
        print(f"    N = {rl['n']}   mean = {rl['mean']:.4f}   "
              f"median = {rl['median']:.4f}   min/max = {rl['min']:.4f}/{rl['max']:.4f}")
    else:
        print(f"\n  ROUGE-L F1: no data "
              f"(pip install rouge-score, or no gold texts >= 300 chars)")

    if r1:
        print(f"\n  ROUGE-1 F1:")
        print(f"    mean = {r1['mean']:.4f}   median = {r1['median']:.4f}")

    if wf:
        print(f"\n  Word-level (unique words):")
        print(f"    Precision:  mean = {wp['mean']:.4f}")
        print(f"    Recall:     mean = {wr['mean']:.4f}")
        print(f"    F1:         mean = {wf['mean']:.4f}   median = {wf['median']:.4f}")

    if lr:
        print(f"\n  Length ratio (UCB / gold):")
        print(f"    mean = {lr['mean']:.2f}   median = {lr['median']:.2f}   "
              f"min/max = {lr['min']:.2f}/{lr['max']:.2f}")

    print(f"\n{'='*70}")
    print("[COVERAGE BY SITE]")
    print(f"{'='*70}")
    print(f"  {'Site':<20} {'N':>4} {'Author':>8} {'Title':>8} {'Date':>8} "
          f"{'Categ':>8} {'Cont':>8} {'ROUGE-L':>9} {'WordF1':>8}")
    print("  " + "-" * 92)

    cov_rows = []
    for site in sorted(coverage_total.keys()):
        tot = coverage_total[site]
        if tot == 0:
            continue
        ok = coverage_ok[site]
        a_pct = ok["author"] / tot * 100
        t_pct = ok["title"] / tot * 100
        d_pct = ok["date"] / tot * 100
        c_pct = ok["category"] / tot * 100
        ct_pct = ok["content"] / tot * 100

        rl_s = stats_summary(quality_by_site[site]["rouge_l"])
        wf_s = stats_summary(quality_by_site[site]["word_f1"])
        rl_disp = f"{rl_s['mean']:.3f}" if rl_s else "—"
        wf_disp = f"{wf_s['mean']:.3f}" if wf_s else "—"

        print(f"  {site:<20} {tot:>4} {a_pct:>7.1f}% {t_pct:>7.1f}% {d_pct:>7.1f}% "
              f"{c_pct:>7.1f}% {ct_pct:>7.1f}% {rl_disp:>9} {wf_disp:>8}")

        cov_rows.append({
            "site": site, "articles": tot,
            "author_pct": round(a_pct, 1),
            "title_pct": round(t_pct, 1),
            "date_pct": round(d_pct, 1),
            "category_pct": round(c_pct, 1),
            "content_pct": round(ct_pct, 1),
            "rouge_l_mean": round(rl_s["mean"], 4) if rl_s else None,
            "rouge_l_n": rl_s["n"] if rl_s else 0,
            "word_f1_mean": round(wf_s["mean"], 4) if wf_s else None,
        })

    print(f"\n{'='*70}")
    print("[BY LANGUAGE]")
    print(f"{'='*70}")
    print(f"  {'Lang':<6} {'N':>5} {'Author':>10} {'Date':>10} "
          f"{'Content':>10} {'ROUGE-L':>10} {'WordF1':>10}")
    print("  " + "-" * 70)

    lang_rows = []
    lang_site = defaultdict(set)
    for art in urls_to_process:
        lang_site[art.get("lang", "?")].add(art["site"])

    for lang in sorted(lang_site.keys()):
        tot = 0
        a_ok = t_ok = d_ok = c_ok = ct_ok = 0
        for site in lang_site[lang]:
            tot += coverage_total.get(site, 0)
            a_ok += coverage_ok[site]["author"]
            t_ok += coverage_ok[site]["title"]
            d_ok += coverage_ok[site]["date"]
            c_ok += coverage_ok[site]["category"]
            ct_ok += coverage_ok[site]["content"]
        if tot == 0:
            continue

        rl_s = stats_summary(quality_by_lang[lang]["rouge_l"])
        wf_s = stats_summary(quality_by_lang[lang]["word_f1"])
        rl_disp = f"{rl_s['mean']:.3f}" if rl_s else "—"
        wf_disp = f"{wf_s['mean']:.3f}" if wf_s else "—"

        print(f"  {lang:<6} {tot:>5} "
              f"{a_ok/tot*100:>9.1f}% "
              f"{d_ok/tot*100:>9.1f}% "
              f"{ct_ok/tot*100:>9.1f}% "
              f"{rl_disp:>10} {wf_disp:>10}")

        lang_rows.append({
            "lang": lang, "articles": tot,
            "author_pct": round(a_ok/tot*100, 1),
            "date_pct": round(d_ok/tot*100, 1),
            "content_pct": round(ct_ok/tot*100, 1),
            "rouge_l_mean": round(rl_s["mean"], 4) if rl_s else None,
            "word_f1_mean": round(wf_s["mean"], 4) if wf_s else None,
        })

    score_rows = tracker.rows()
    print_strategy_scores(score_rows)

    print(f"\n{'='*70}")
    print("[FINAL TABLE FOR THE ARTICLE]")
    print(f"{'='*70}")
    print(f"  {'Metric':<40} {'Value':>18}")
    print("  " + "-" * 60)

    if total:
        print(f"  {'Documents processed':<40} {total:>18}")
        print(f"  {'Author extracted':<40} "
              f"{sum(coverage_ok[s]['author'] for s in coverage_total)/total*100:>17.1f}%")
        print(f"  {'Title extracted':<40} "
              f"{sum(coverage_ok[s]['title'] for s in coverage_total)/total*100:>17.1f}%")
        print(f"  {'Date extracted':<40} "
              f"{sum(coverage_ok[s]['date'] for s in coverage_total)/total*100:>17.1f}%")
        print(f"  {'Category extracted':<40} "
              f"{sum(coverage_ok[s]['category'] for s in coverage_total)/total*100:>17.1f}%")
        print(f"  {'Content extracted':<40} "
              f"{sum(coverage_ok[s]['content'] for s in coverage_total)/total*100:>17.1f}%")
    if rl:
        print(f"  {'ROUGE-L F1 (N=' + str(rl['n']) + ')':<40} {rl['mean']:>18.4f}")
    if wf:
        print(f"  {'Word-level F1':<40} {wf['mean']:>18.4f}")
    if lr:
        print(f"  {'Length ratio (mean)':<40} {lr['mean']:>18.2f}")

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value", "hits", "share_pct"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n[OK] Strategy distribution: {args.out}")

    with open(args.out_coverage, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "site", "articles", "author_pct", "title_pct", "date_pct",
            "category_pct", "content_pct",
            "rouge_l_mean", "rouge_l_n", "word_f1_mean",
        ])
        w.writeheader()
        w.writerows(cov_rows)
    print(f"[OK] Coverage by site:     {args.out_coverage}")

    with open(args.out_quality, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "lang", "articles", "author_pct", "date_pct", "content_pct",
            "rouge_l_mean", "word_f1_mean",
        ])
        w.writeheader()
        w.writerows(lang_rows)
    print(f"[OK] Quality by language:  {args.out_quality}")

    with open(args.out_scores, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "field", "strategy", "attempts", "fired", "ref_available",
            "correct", "p", "c_all_ms", "c_fire_ms", "p_over_c",
        ])
        w.writeheader()
        w.writerows(score_rows)
    print(f"[OK] Strategy scores:      {args.out_scores}")


if __name__ == "__main__":
    main()
    