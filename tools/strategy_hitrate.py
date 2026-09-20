# tools/strategy_hitrate.py
"""
Measure two things in a single pass over a set of article URLs.

1. First-matching strategy distribution for the fields:
   author, title, date, category, content.

2. Body-extraction quality for UniCorpusBuilder:
   ROUGE-1, ROUGE-L, word-level P/R/F1, length ratio.
   Gold text is taken independently of the pipeline from
   ``<article>`` / ``<main>`` / ``[role=main]`` / ``<body>``.

Per-site and per-language coverage is also reported.

URL sources
-----------
Two modes are supported:

  * Config mode — the script reads ``config/universal.yaml`` and picks
    sites by language, then discovers article URLs from each site.
  * File mode — URLs are read from local JSONL files produced by an
    earlier pipeline run.

Usage
-----
    python tools/strategy_hitrate.py
    python tools/strategy_hitrate.py --langs tg tt ba os --sites-per-lang 2 --articles-per-site 20
    python tools/strategy_hitrate.py --exclude sssr ozodi
    python tools/strategy_hitrate.py --input-files "D:\\...\\file1.jsonl" "D:\\...\\file2.jsonl"

Installation
------------
    pip install trafilatura rouge-score
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
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bs4 import BeautifulSoup
from tqdm import tqdm

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


# ------------------------------------------------------------------
# Text metrics
# ------------------------------------------------------------------

def _clean_text(s):
    """Collapse whitespace and strip."""
    return re.sub(r"\s+", " ", s or "").strip()


def _tokenize(s):
    """Tokenize a string into lowercase word units."""
    return WORD_RE.findall((s or "").lower())


def extract_gold_text(soup):
    """
    Return the gold article body used to score UniCorpusBuilder.

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


def compute_quality(gold, pred):
    """
    Compute all body-quality metrics for one (gold, pred) pair.

    Returns a dict with all metrics. Individual fields are None when
    the gold text is shorter than 300 characters or the prediction is
    empty.
    """
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
# URL collection
# ------------------------------------------------------------------

def gather_from_config(engine, langs, sites_per_lang, articles_per_site,
                       min_strategies, exclude=None):
    """
    Select sites by language and collect article URLs from each.

    Sites with fewer than ``min_strategies`` author strategies are
    skipped, so that the first-matching-strategy analysis has enough
    configurations to be informative.
    """
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
    """Yield URLs from local JSONL files (file mode)."""
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

def try_author(engine, soup, site_cfg, url):
    """Return the name of the first author strategy that produces a value, or None."""
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
        try:
            result = engine.extract_author_from_soup(soup, tmp, url)
        except Exception:
            result = None
        if result and result.strip():
            s = result.strip()
            if s.startswith(("{", "[")):
                continue
            if len(s) > 300:
                continue
            return strat

    if "default_fallback" in strategies and site_cfg.get("default_author"):
        return "default_fallback"
    return None


def try_title(engine, soup, site_cfg):
    """Return the selector that first produces a title, or None."""
    for sel in site_cfg.get("title_selectors", []):
        try:
            el = engine._safe_select_one(soup, sel)
        except Exception:
            el = None
        if el:
            if el.name == "meta" and el.get("content"):
                val = (el.get("content") or "").strip()
            else:
                val = el.get_text(" ", strip=True)
            if val:
                return sel
    if soup.title and soup.title.string and soup.title.string.strip():
        return "<title>"
    return None


def try_date(engine, soup, site_cfg, url):
    """Return the source label that first produces a parsable date, or None."""
    from pipeline.pipeline_core import parse_datetime_value
    locale_map = engine.get_date_locale_map(url)

    jld = None
    try:
        jld = engine.extract_jsonld_date(soup)
    except Exception:
        jld = None
    if jld:
        parsed = parse_datetime_value(jld, locale_map)
        if parsed:
            return "json_ld:datePublished"

    for sel in site_cfg.get("date_selectors", []):
        if "ld+json" in sel:
            continue
        try:
            el = engine._safe_select_one(soup, sel)
        except Exception:
            el = None
        if el:
            if el.name == "script":
                continue
            val = el.get("content") or el.get("datetime") or el.get_text(" ", strip=True)
            if val:
                parsed = parse_datetime_value(val, locale_map)
                if parsed:
                    return sel
    return None


def try_category(engine, soup, url, site_cfg):
    """Return the category-strategy label that first produces a value, or None."""
    strategies = site_cfg.get("category_strategy") or []
    for strat in strategies:
        if strat == "meta_tag":
            for sel in site_cfg.get("category_selectors", []):
                try:
                    el = engine._safe_select_one(soup, sel)
                except Exception:
                    el = None
                if el:
                    val = el.get("content") if el.name == "meta" else el.get_text(" ", strip=True)
                    if val and val.strip():
                        return f"{strat}:{sel[:40]}"
        elif strat == "url_path_parsing":
            try:
                cat = engine.extract_category(soup, url, site_cfg)
                if cat:
                    return strat
            except Exception:
                pass
        elif strat == "breadcrumb":
            for el in soup.select(".breadcrumb, .breadcrumbs, [class*=breadcrumb]"):
                txt = el.get_text(" ", strip=True)
                if txt:
                    return strat
        elif strat == "context_passed":
            continue
    try:
        cat = engine.extract_category(soup, url, site_cfg)
        if cat:
            return "fallback"
    except Exception:
        pass
    return None


def try_content(engine, soup, site_cfg):
    """Return the content selector that first produces a non-trivial body, or None."""
    for sel in site_cfg.get("content_selectors", []):
        try:
            node = engine._safe_select_one(soup, sel)
        except Exception:
            node = None
        if node:
            txt = node.get_text(" ", strip=True)
            if txt and len(txt) > 200:
                return sel
    try:
        container = engine.find_best_content_container(soup, site_cfg)
        if container is not None and container != soup:
            return "auto:heuristic"
    except Exception:
        pass
    return None


# ------------------------------------------------------------------
# Main analysis loop
# ------------------------------------------------------------------

def run_analysis(engine, urls_to_process, sleep):
    """Run strategy probes and body-quality metrics over every URL."""
    stats = {
        "author": Counter(),
        "title": Counter(),
        "date": Counter(),
        "category": Counter(),
        "content": Counter(),
    }
    coverage_total = Counter()
    coverage_ok = defaultdict(lambda: Counter())

    # Quality metrics per site and per language
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

        coverage_total[site] += 1

        # Strategy probes per field
        a = try_author(engine, soup, site_cfg, url)
        stats["author"][a or "[missed]"] += 1
        if a:
            coverage_ok[site]["author"] += 1

        t = try_title(engine, soup, site_cfg)
        stats["title"][t or "[missed]"] += 1
        if t:
            coverage_ok[site]["title"] += 1

        d = try_date(engine, soup, site_cfg, url)
        stats["date"][d or "[missed]"] += 1
        if d:
            coverage_ok[site]["date"] += 1

        c = try_category(engine, soup, url, site_cfg)
        stats["category"][c or "[missed]"] += 1
        if c:
            coverage_ok[site]["category"] += 1

        ct = try_content(engine, soup, site_cfg)
        stats["content"][ct or "[missed]"] += 1
        if ct:
            coverage_ok[site]["content"] += 1

        # Body-quality metrics
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
            quality_by_lang, total, failed_fetch)


# ------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------

def print_stats(name, counter, total, top=8):
    """Print the strategy distribution for one field and return its CSV rows."""
    print(f"\n--- {name.upper()} ---")
    if not counter:
        print("  (no data)")
        return []
    print(f"  {'Value':<45} {'Hits':>10} {'Share':>8}")
    print("  " + "-" * 65)

    rows = []
    # Real strategy names first; "[missed]" is printed last.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/universal.yaml")
    ap.add_argument("--out", default="strategy_hitrate.csv")
    ap.add_argument("--out-coverage", default="strategy_coverage.csv")
    ap.add_argument("--out-quality", default="strategy_quality.csv")
    ap.add_argument("--sleep", type=float, default=0.3)

    # Config mode
    ap.add_argument("--langs", nargs="+", default=["tg", "tt", "ba", "os"])
    ap.add_argument("--sites-per-lang", type=int, default=2)
    ap.add_argument("--articles-per-site", type=int, default=20)
    ap.add_argument("--min-strategies", type=int, default=1)
    ap.add_argument("--exclude", nargs="*", default=["sssr", "ozodi"])

    # File mode
    ap.add_argument("--input-files", nargs="*", default=None)
    ap.add_argument("--input-dir", default=None)
    ap.add_argument("--per-site", type=int, default=30)
    ap.add_argument("--limit", type=int, default=300)

    args = ap.parse_args()

    engine = ExtractionEngine(yaml_path=args.config)

    # --- Determine URL source ---
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
     quality_by_lang, total, failed_fetch) = run_analysis(
        engine, urls_to_process, args.sleep
    )

    print(f"\n{'='*70}")
    print(f"Processed:      {total}")
    print(f"Fetch failures: {failed_fetch}")
    print(f"{'='*70}")

    # --- Per-field strategy distributions ---
    all_rows = []
    for name in ["author", "title", "date", "category", "content"]:
        rows = print_stats(name, stats[name], total, top=8)
        if rows:
            all_rows.extend(rows)

    # --- Body-extraction quality ---
    print(f"\n{'='*70}")
    print("[BODY-EXTRACTION QUALITY] — gold: <article> / <main> / <body>")
    print(f"{'='*70}")

    all_rl = []
    all_r1 = []
    all_wf = []
    all_wp = []
    all_wr = []
    all_lr = []
    for s, d in quality_by_site.items():
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

    # --- Coverage by site ---
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
        a = ok["author"] / tot * 100
        t = ok["title"] / tot * 100
        d = ok["date"] / tot * 100
        c = ok["category"] / tot * 100
        ct = ok["content"] / tot * 100

        rl_s = stats_summary(quality_by_site[site]["rouge_l"])
        wf_s = stats_summary(quality_by_site[site]["word_f1"])
        rl_disp = f"{rl_s['mean']:.3f}" if rl_s else "—"
        wf_disp = f"{wf_s['mean']:.3f}" if wf_s else "—"

        print(f"  {site:<20} {tot:>4} {a:>7.1f}% {t:>7.1f}% {d:>7.1f}% "
              f"{c:>7.1f}% {ct:>7.1f}% {rl_disp:>9} {wf_disp:>8}")

        cov_rows.append({
            "site": site, "articles": tot,
            "author_pct": round(a, 1),
            "title_pct": round(t, 1),
            "date_pct": round(d, 1),
            "category_pct": round(c, 1),
            "content_pct": round(ct, 1),
            "rouge_l_mean": round(rl_s["mean"], 4) if rl_s else None,
            "rouge_l_n": rl_s["n"] if rl_s else 0,
            "word_f1_mean": round(wf_s["mean"], 4) if wf_s else None,
        })

    # --- By language ---
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
        # Aggregate N and per-field coverage across all sites of this language.
        tot = 0
        a_ok = t_ok = d_ok = c_ok = ct_ok = 0
        for site in lang_site[lang]:
            st = coverage_total.get(site, 0)
            tot += st
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

    # --- Final table ---
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

    # --- CSV: strategies ---
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value", "hits", "share_pct"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n[OK] Strategy distribution: {args.out}")

    # --- CSV: coverage ---
    with open(args.out_coverage, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "site", "articles", "author_pct", "title_pct", "date_pct",
            "category_pct", "content_pct",
            "rouge_l_mean", "rouge_l_n", "word_f1_mean",
        ])
        w.writeheader()
        w.writerows(cov_rows)
    print(f"[OK] Coverage by site:     {args.out_coverage}")

    # --- CSV: quality by language ---
    with open(args.out_quality, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "lang", "articles", "author_pct", "date_pct", "content_pct",
            "rouge_l_mean", "word_f1_mean",
        ])
        w.writeheader()
        w.writerows(lang_rows)
    print(f"[OK] Quality by language:  {args.out_quality}")


if __name__ == "__main__":
    main()