# tools/author_verification.py
"""
Author verification helper for the trafilatura benchmark.

Purpose
-------
``tools/benchmark_trafilatura.py`` checks author correctness only on
documents that carry ``<meta name="author">`` (14 of 118 in the current
run). The remaining 104 have no meta reference, so correctness was
never verified there.

This tool does NOT verify authors automatically — that is a judgement
call that cannot be automated for multi-author bylines, editorial
signatures («Редакция», «Пресс-служба») and non-Latin scripts.

Instead it produces a review-friendly CSV:

    url, site, lang, ucb_author, traf_author,
    signature_found, signature_source, context_snippet,
    verdict_ucb, verdict_traf, notes

The ``verdict_*`` columns are filled by hand:
    correct    — matches a personal name on the page
    editorial  — page shows an editorial byline, not a person
    wrong      — does not match anything on the page
    unknown    — cannot determine
    (empty)    — not yet labelled

``stats`` mode aggregates the labelled CSV into four metrics per
system and reports them under unambiguous names:

    correct / labeled                    — precision on verified rows
    correct / filled                     — yield among filled fields
    (correct + editorial) / labeled      — byline precision
    (correct + editorial) / filled       — byline yield

``correct`` counts personal names only. ``editorial`` counts editorial
bylines («Редакция», «Пресс-служба»): they are valid bylines but not
personal names. ``unknown`` means the page carries no byline and the
verdict cannot be decided.

Design decisions
----------------
* HTML is cached on disk (``_cache_author_verification/``). A rerun
  reuses the cache unless ``--no-cache`` is passed.
* The output CSV is keyed by ``url``. On a rerun the ``verdict_*`` and
  ``notes`` columns are preserved, even if the signature columns are
  recomputed.
* ``prepare`` refuses to overwrite a file that already contains manual
  verdicts unless its name contains ``filled`` or ``--force`` is
  passed. This prevents the common mistake of clobbering the labelled
  sheet by re-running with the default ``--out``.
* Signature lookup is deliberately permissive: it returns the first
  matching pattern and a context snippet, so the human can see the
  whole byline block. Precision is not the goal here; coverage is.
* No automatic labelling, no ML. Every ``verdict`` is a human decision.

Usage
-----
    # 1. Prepare the verification sheet
    python tools/author_verification.py prepare

    # 2. Copy and fill verdicts
    copy author_verification.csv author_verification_filled.csv
    # edit author_verification_filled.csv, save

    # 3. Summarize
    python tools/author_verification.py stats author_verification_filled.csv

    # Optional: write the summary to a CSV for the article pipeline
    python tools/author_verification.py stats author_verification_filled.csv \\
        --out-csv author_verification_stats.csv

    # Custom paths, force re-fetch, limit for testing
    python tools/author_verification.py prepare \\
        --comparison trafilatura_comparison.csv \\
        --out author_verification.csv \\
        --no-cache --limit 20

Exit codes
----------
    prepare: 0 on success; 1 if the comparison CSV is missing or empty,
                or if writing would overwrite manual verdicts.
    stats:   0 on success; 1 if the verification CSV is missing.

See also
--------
    tools/benchmark_trafilatura.py   (produces the input CSV)
    tools/strategy_hitrate.py        (per-strategy precision)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bs4 import BeautifulSoup, Tag
from tqdm import tqdm

from pipeline.pipeline_extraction import ExtractionEngine
from logger_setup import get_file_logger


logger = get_file_logger("author_verification", "logs/author_verification.log")


# ------------------------------------------------------------------
# Signature patterns
# ------------------------------------------------------------------

# Byline markers in visible page text.
# These are the patterns a human would use to find "who wrote this"
# without opening the page source. JSON-LD blocks are ignored — using
# them here would be circular, because that is exactly what the systems
# under test read.
SIGNATURE_PATTERNS = [
    # Russian / common
    re.compile(r"\bАвтор(?:ы|ка)?\b\s*[:\-—–]?\s*[А-ЯЁA-Z][^\n.;|<]{1,80}", re.I),
    re.compile(r"\bПодготовил(?:а|и)?\b\s*[:\-—–]?\s*[А-ЯЁA-Z][^\n.;|<]{1,80}", re.I),
    re.compile(r"\bТекст\b\s*[:\-—–]?\s*[А-ЯЁA-Z][^\n.;|<]{1,80}", re.I),
    re.compile(r"\bКорреспондент\b\s*[:\-—–]?\s*[А-ЯЁA-Z][^\n.;|<]{1,80}", re.I),
    # Tajik
    re.compile(r"\bМуаллиф\b\s*[:\-—–]?\s*[А-ЯЁA-Z][^\n.;|<]{1,80}", re.I),
    re.compile(r"\bОмода\s+кард\b\s*[:\-—–]?\s*[А-ЯЁA-Z][^\n.;|<]{1,80}", re.I),
    re.compile(r"\bАкс\b\s*[:\-—–]?\s*[А-ЯЁA-Z][^\n.;|<]{1,80}", re.I),
    re.compile(r"\bСарчашма\b\s*[:\-—–]?\s*[А-ЯЁA-Z][^\n.;|<]{1,80}", re.I),
    # Tatar
    re.compile(r"\bЯзган\b\s*[:\-—–]?\s*[А-ЯЁӘӨҮҖҢҺA-Z][^\n.;|<]{1,80}", re.I),
    re.compile(r"\bӘзерләде\b\s*[:\-—–]?\s*[А-ЯЁӘӨҮҖҢҺA-Z][^\n.;|<]{1,80}", re.I),
    # Bashkir
    re.compile(r"\bӘҙерләне\b\s*[:\-—–]?\s*[А-ЯЁӘӨҮҖҢҺA-Z][^\n.;|<]{1,80}", re.I),
    # Editorial bylines, only when followed by a delimiter or at end of text
    re.compile(r"\bРедакция\b(?:\s*[:\-—–]|\s*$)", re.I),
    re.compile(r"\bПресс-служба\b(?:\s*[:\-—–]|\s*$)", re.I),
]


# Byline CSS selectors (visible blocks only).
BYLINE_SELECTORS = [
    ".author",
    ".article-author",
    ".byline",
    ".post-author",
    ".news-author",
    ".c-author",
    "[class*='author__']",
    "[class*='_author_']",
    "[itemprop='author']",
    "[rel='author']",
    "[class*='byline']",
]


# ------------------------------------------------------------------
# CSV schema
# ------------------------------------------------------------------

CSV_FIELDS = [
    "url", "site", "lang",
    "ucb_author", "traf_author",
    "signature_found", "signature_source", "context_snippet",
    "verdict_ucb", "verdict_traf",
    "notes",
]

STATS_FIELDS = [
    "system", "filled", "labeled", "correct", "editorial", "wrong",
    "unknown", "unlabeled",
    "correct_over_labeled", "correct_over_filled",
    "byline_over_labeled", "byline_over_filled",
]

# Verdicts that count as a labelled row. ``unknown`` is a human
# decision but excluded from every ratio, because there is nothing to
# compare against on the page.
LABELED_VERDICTS = {"correct", "editorial", "wrong"}


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _clean(s: str) -> str:
    """Collapse whitespace."""
    return re.sub(r"\s+", " ", s or "").strip()


def _truncate(s: str, n: int) -> str:
    """Truncate ``s`` to ``n`` chars, adding an ellipsis if cut."""
    s = _clean(s)
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _url_cache_key(url: str) -> str:
    """Stable short filename key for the HTML cache."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def _norm_verdict(value: str) -> str:
    """Lowercase and strip a verdict cell."""
    return (value or "").strip().lower()


def _ratio(num: int, den: int) -> Optional[float]:
    return (num / den) if den else None


def _fmt_pct(value: Optional[float]) -> str:
    return f"{value * 100:5.1f}%" if value is not None else "   N/A"


# ------------------------------------------------------------------
# Signature extraction
# ------------------------------------------------------------------

def find_signature(soup: BeautifulSoup) -> tuple[Optional[str], Optional[str]]:
    """
    Find a byline-looking string on the visible page.

    Returns ``(signature, source)``:
      * ``signature`` — the byline string, or ``None``;
      * ``source``    — ``'css:<selector>'`` or ``'regex:<index>'``.

    Both checks look only at visible text. JSON-LD blocks are ignored:
    using them here would be circular, because the systems under test
    read the same blocks.
    """
    # 1. Byline CSS selectors
    for sel in BYLINE_SELECTORS:
        try:
            el = soup.select_one(sel)
        except Exception:
            el = None
        if not el:
            continue
        txt = _clean(el.get_text(" ", strip=True))
        if 2 <= len(txt) <= 200 and re.search(r"[A-Za-zА-Яа-яЁё]", txt):
            return txt, f"css:{sel}"

    # 2. Regex markers over the whole visible body
    body = soup.find("body")
    if not body:
        return None, None
    body_text = _clean(body.get_text(" ", strip=True))
    if not body_text:
        return None, None

    for idx, pat in enumerate(SIGNATURE_PATTERNS):
        m = pat.search(body_text)
        if m:
            return _clean(m.group(0)), f"regex:{idx}"

    return None, None


def find_context(container: Tag, max_chars: int = 800) -> str:
    """
    Return up to ``max_chars`` of article text — first two paragraphs
    (header, often carries the byline) plus last three paragraphs
    (footer, often carries the byline).

    The reviewer sees a compact snippet instead of an entire article,
    which makes manual labeling practical.
    """
    if container is None:
        return ""

    paragraphs = [p for p in container.find_all("p") if p.get_text(strip=True)]
    if not paragraphs:
        return ""

    parts: list[str] = []
    for p in paragraphs[:2] + paragraphs[-3:]:
        txt = _clean(p.get_text(" ", strip=True))
        if txt and txt not in parts:
            parts.append(txt)

    return _truncate(" | ".join(parts), max_chars)


# ------------------------------------------------------------------
# HTML cache
# ------------------------------------------------------------------

def fetch_cached(
    engine: ExtractionEngine,
    url: str,
    cache_dir: Path,
    no_cache: bool,
) -> tuple[Optional[str], Optional[str], bool]:
    """
    Return ``(html, error, from_cache)``.

    Success  → ``html`` is a string, ``error`` is ``None``.
    Failure  → ``html`` is ``None``, ``error`` is a short message.
    """
    key = _url_cache_key(url)
    html_path = cache_dir / f"{key}.html"
    err_path = cache_dir / f"{key}.error"

    if not no_cache:
        if html_path.exists():
            return html_path.read_text(encoding="utf-8", errors="replace"), None, True
        if err_path.exists():
            return None, err_path.read_text(encoding="utf-8", errors="replace"), True

    try:
        html = engine.fetch_html(url)
    except Exception as e:
        cache_dir.mkdir(parents=True, exist_ok=True)
        err_path.write_text(str(e), encoding="utf-8")
        logger.warning(f"fetch failed for {url}: {e}")
        return None, str(e), False

    cache_dir.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html, encoding="utf-8")
    return html, None, False


# ------------------------------------------------------------------
# Prepare command
# ------------------------------------------------------------------

def read_comparison(path: Path, include_meta: bool = False) -> list[dict]:
    """
    Return rows from the benchmark CSV that need manual author
    verification:

      * no fetch / parse error;
      * no ``ref_meta_author`` (no ground truth to compare against);
      * at least one of UCB / Trafilatura returned an author.

    With ``include_meta=True`` the second filter is disabled, for a
    full re-verification.
    """
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    out: list[dict] = []
    for r in rows:
        if (r.get("error") or "").strip():
            continue
        if not include_meta and (r.get("ref_meta_author") or "").strip():
            continue
        if not (
            (r.get("ucb_author") or "").strip()
            or (r.get("traf_author") or "").strip()
        ):
            continue
        out.append(r)
    return out


def load_existing_verdicts(path: Path) -> dict[str, dict]:
    """Return ``{url: {'verdict_ucb', 'verdict_traf', 'notes'}}``."""
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            url = (r.get("url") or "").strip()
            if not url:
                continue
            out[url] = {
                "verdict_ucb": r.get("verdict_ucb", ""),
                "verdict_traf": r.get("verdict_traf", ""),
                "notes": r.get("notes", ""),
            }
    return out


def has_manual_verdicts(path: Path) -> bool:
    """Return True if any row in ``path`` already carries a verdict."""
    verdicts = load_existing_verdicts(path)
    return any(
        _norm_verdict(v["verdict_ucb"]) or _norm_verdict(v["verdict_traf"])
        for v in verdicts.values()
    )


def _is_filled_path(path: Path) -> bool:
    """Return True if the filename suggests a labelled copy."""
    return "filled" in path.stem.lower()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") or "" for k in CSV_FIELDS})


def cmd_prepare(args) -> int:
    comparison_path = Path(args.comparison)
    if not comparison_path.exists():
        print(f"[ERROR] Comparison CSV not found: {comparison_path}")
        print("        Run tools/benchmark_trafilatura.py first.")
        return 1
    if not Path(args.config).exists():
        print(f"[ERROR] Config not found: {args.config}")
        return 1

    out_path = Path(args.out)

    if (
        out_path.exists()
        and not _is_filled_path(out_path)
        and has_manual_verdicts(out_path)
        and not args.force
    ):
        print(
            f"[REFUSE] {out_path} already contains manual verdicts and its "
            f"name does not contain 'filled'."
        )
        print(
            "         This is the working sheet, not a scratch copy. Write "
            "to a *_filled.csv file,"
        )
        print(
            "         or pass --force to overwrite anyway. Re-running "
            "prepare on the base sheet"
        )
        print(
            "         is safe when no verdicts were entered yet."
        )
        return 1

    rows = read_comparison(comparison_path, include_meta=args.include_meta)
    if not rows:
        print("[INFO] No rows need manual verification.")
        print("       All benchmark rows either have a meta author or no "
              "system returned an author.")
        return 1

    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    print(f"[PLAN] {len(rows)} URLs need manual author verification")

    existing = load_existing_verdicts(out_path)
    if existing:
        print(f"[RESUME] Preserving verdicts for {len(existing)} existing URLs")

    logger.info(f"prepare: {len(rows)} URLs, out={out_path}, cache={args.cache_dir}")

    engine = ExtractionEngine(yaml_path=args.config)
    cache_dir = Path(args.cache_dir)

    out_rows: list[dict] = []
    n_cached = 0
    n_fetched = 0
    n_failed = 0

    for r in tqdm(rows, desc="[PREPARE]", unit="url"):
        url = (r.get("url") or "").strip()
        prev = existing.get(url, {})

        entry = {
            "url": url,
            "site": r.get("site", ""),
            "lang": r.get("lang", ""),
            "ucb_author": r.get("ucb_author", ""),
            "traf_author": r.get("traf_author", ""),
            "signature_found": "",
            "signature_source": "",
            "context_snippet": "",
            "verdict_ucb": prev.get("verdict_ucb", ""),
            "verdict_traf": prev.get("verdict_traf", ""),
            "notes": prev.get("notes", ""),
        }

        if not url:
            entry["notes"] = (entry["notes"] + " [empty url]").strip()
            out_rows.append(entry)
            continue

        html, err, from_cache = fetch_cached(engine, url, cache_dir, args.no_cache)

        if from_cache:
            n_cached += 1
        else:
            n_fetched += 1
            if err is None:
                time.sleep(args.sleep)

        if err or not html:
            n_failed += 1
            entry["notes"] = (entry["notes"] + f" [fetch_error: {err}]").strip()
            out_rows.append(entry)
            continue

        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception as e:
            n_failed += 1
            entry["notes"] = (entry["notes"] + f" [parse_error: {e}]").strip()
            out_rows.append(entry)
            continue

        sig, src = find_signature(soup)
        entry["signature_found"] = sig or ""
        entry["signature_source"] = src or ""

        try:
            site_cfg = engine.site_cfg(url)
            container = engine.find_best_content_container(soup, site_cfg)
        except Exception:
            container = soup
        entry["context_snippet"] = find_context(container)

        out_rows.append(entry)

    write_csv(out_path, out_rows)

    logger.info(
        f"prepare done: rows={len(out_rows)}, "
        f"cached={n_cached}, fetched={n_fetched}, failed={n_failed}"
    )

    print(f"\n[OK] Wrote {len(out_rows)} rows to {out_path}")
    print(f"     cached:  {n_cached}")
    print(f"     fetched: {n_fetched}")
    print(f"     failed:  {n_failed}")
    if not _is_filled_path(out_path):
        print()
        print("[NEXT] Make a working copy for manual labeling:")
        print(f"       copy {out_path} {out_path.with_stem(out_path.stem + '_filled')}")
        print("       Edit the *_filled.csv copy, NOT the base sheet.")
    print()
    print("[NEXT] Fill the ``verdict_ucb`` / ``verdict_traf`` columns:")
    print("       correct    — matches a personal name on the page")
    print("       editorial  — page shows an editorial byline "
          "(«Редакция», «Пресс-служба»)")
    print("       wrong      — does not match anything on the page")
    print("       unknown    — cannot determine")
    print("       Empty cells are treated as not yet labelled.")
    print(f"\n       Then run: python {sys.argv[0]} stats "
          f"{out_path.with_stem(out_path.stem + '_filled')}")
    return 0


# ------------------------------------------------------------------
# Stats command
# ------------------------------------------------------------------

def _system_metrics(rows: list[dict], key: str) -> dict:
    """Return the four metrics and the verdict counts for one system."""
    author_col = f"{key}_author"
    verdict_col = f"verdict_{key}"

    filled = [r for r in rows if _norm_verdict(r.get(author_col) or "")]
    labeled = [
        r for r in filled
        if _norm_verdict(r.get(verdict_col)) in LABELED_VERDICTS
    ]
    correct = [r for r in labeled if _norm_verdict(r.get(verdict_col)) == "correct"]
    editorial = [r for r in labeled if _norm_verdict(r.get(verdict_col)) == "editorial"]
    wrong = [r for r in labeled if _norm_verdict(r.get(verdict_col)) == "wrong"]
    unknown = [r for r in filled if _norm_verdict(r.get(verdict_col)) == "unknown"]
    unlabeled = [r for r in filled if not _norm_verdict(r.get(verdict_col))]

    n_filled = len(filled)
    n_labeled = len(labeled)
    n_correct = len(correct)
    n_editorial = len(editorial)

    return {
        "filled": n_filled,
        "labeled": n_labeled,
        "correct": n_correct,
        "editorial": n_editorial,
        "wrong": len(wrong),
        "unknown": len(unknown),
        "unlabeled": len(unlabeled),
        "correct_over_labeled": _ratio(n_correct, n_labeled),
        "correct_over_filled": _ratio(n_correct, n_filled),
        "byline_over_labeled": _ratio(n_correct + n_editorial, n_labeled),
        "byline_over_filled": _ratio(n_correct + n_editorial, n_filled),
    }


def _wrong_by_site(rows: list[dict], key: str) -> Counter:
    """Count ``wrong`` verdicts per site for one system."""
    verdict_col = f"verdict_{key}"
    counter: Counter = Counter()
    for r in rows:
        if _norm_verdict(r.get(verdict_col)) != "wrong":
            continue
        if not _norm_verdict(r.get(f"{key}_author") or ""):
            continue
        counter[(r.get("site") or "?")] += 1
    return counter


def _filled_and_labeled_by_site(rows: list[dict], key: str) -> dict[str, dict[str, int]]:
    """Per-site ``filled`` / ``labeled`` / ``correct`` / ``wrong`` for one system."""
    author_col = f"{key}_author"
    verdict_col = f"verdict_{key}"
    out: dict[str, dict[str, int]] = defaultdict(
        lambda: {"filled": 0, "labeled": 0, "correct": 0, "editorial": 0, "wrong": 0}
    )
    for r in rows:
        if not _norm_verdict(r.get(author_col) or ""):
            continue
        site = r.get("site") or "?"
        out[site]["filled"] += 1
        v = _norm_verdict(r.get(verdict_col))
        if v in LABELED_VERDICTS:
            out[site]["labeled"] += 1
            if v == "correct":
                out[site]["correct"] += 1
            elif v == "editorial":
                out[site]["editorial"] += 1
            elif v == "wrong":
                out[site]["wrong"] += 1
    return out


def _write_stats_csv(path: Path, systems: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=STATS_FIELDS)
        w.writeheader()
        for label, s in systems.items():
            w.writerow({
                "system": label,
                "filled": s["filled"],
                "labeled": s["labeled"],
                "correct": s["correct"],
                "editorial": s["editorial"],
                "wrong": s["wrong"],
                "unknown": s["unknown"],
                "unlabeled": s["unlabeled"],
                "correct_over_labeled": (
                    round(s["correct_over_labeled"], 4)
                    if s["correct_over_labeled"] is not None else None
                ),
                "correct_over_filled": (
                    round(s["correct_over_filled"], 4)
                    if s["correct_over_filled"] is not None else None
                ),
                "byline_over_labeled": (
                    round(s["byline_over_labeled"], 4)
                    if s["byline_over_labeled"] is not None else None
                ),
                "byline_over_filled": (
                    round(s["byline_over_filled"], 4)
                    if s["byline_over_filled"] is not None else None
                ),
            })


def cmd_stats(args) -> int:
    path = Path(args.infile)
    if not path.exists():
        print(f"[ERROR] Verification CSV not found: {path}")
        print("        Run: python tools/author_verification.py prepare")
        return 1

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("[INFO] Empty CSV.")
        return 1

    systems = {
        "UniCorpusBuilder": _system_metrics(rows, "ucb"),
        "Trafilatura": _system_metrics(rows, "traf"),
    }

    total = len(rows)
    both = sum(1 for r in rows
               if _norm_verdict(r.get("ucb_author") or "")
               and _norm_verdict(r.get("traf_author") or ""))
    only_ucb = sum(1 for r in rows
                   if _norm_verdict(r.get("ucb_author") or "")
                   and not _norm_verdict(r.get("traf_author") or ""))
    only_traf = sum(1 for r in rows
                    if _norm_verdict(r.get("traf_author") or "")
                    and not _norm_verdict(r.get("ucb_author") or ""))

    print(f"\n{'=' * 74}")
    print("[AUTHOR VERIFICATION SUMMARY]")
    print(f"{'=' * 74}")
    print(f"  Documents in CSV:                {total}")
    print(f"    both systems filled author:    {both}")
    print(f"    only UniCorpusBuilder filled:  {only_ucb}")
    print(f"    only Trafilatura filled:       {only_traf}")
    print()

    for label, s in systems.items():
        print(f"  --- {label} ---")
        print(f"    author filled:                        {s['filled']}")
        print(
            f"    manually labeled:                     {s['labeled']}"
            f"   (unknown: {s['unknown']}, unlabeled: {s['unlabeled']})"
        )
        print(f"    correct (personal name):              {s['correct']}")
        print(f"    editorial (not a person):             {s['editorial']}")
        print(f"    wrong:                                {s['wrong']}")
        print(
            f"    correct / labeled:                    "
            f"{_fmt_pct(s['correct_over_labeled'])}"
            f"   ({s['correct']}/{s['labeled']})"
        )
        print(
            f"    correct / filled:                     "
            f"{_fmt_pct(s['correct_over_filled'])}"
            f"   ({s['correct']}/{s['filled']})"
        )
        print(
            f"    (correct + editorial) / labeled:      "
            f"{_fmt_pct(s['byline_over_labeled'])}"
            f"   ({s['correct'] + s['editorial']}/{s['labeled']})"
        )
        print(
            f"    (correct + editorial) / filled:       "
            f"{_fmt_pct(s['byline_over_filled'])}"
            f"   ({s['correct'] + s['editorial']}/{s['filled']})"
        )
        print()

    print("[BY SITE] filled / labeled / correct / editorial / wrong")
    print(f"  {'Site':<16} "
          f"{'UCB fill':>9} {'UCB lab':>8} {'UCB cor':>8} {'UCB ed':>7} {'UCB wr':>7}  "
          f"{'TF fill':>8} {'TF lab':>7} {'TF cor':>7} {'TF ed':>6} {'TF wr':>6}")
    print("  " + "-" * 112)

    ucb_site = _filled_and_labeled_by_site(rows, "ucb")
    traf_site = _filled_and_labeled_by_site(rows, "traf")
    all_sites = sorted(set(ucb_site) | set(traf_site))

    for site in all_sites:
        u = ucb_site.get(site, {"filled": 0, "labeled": 0, "correct": 0,
                                "editorial": 0, "wrong": 0})
        t = traf_site.get(site, {"filled": 0, "labeled": 0, "correct": 0,
                                 "editorial": 0, "wrong": 0})
        print(
            f"  {site:<16} "
            f"{u['filled']:>9} {u['labeled']:>8} {u['correct']:>8} "
            f"{u['editorial']:>7} {u['wrong']:>7}  "
            f"{t['filled']:>8} {t['labeled']:>7} {t['correct']:>7} "
            f"{t['editorial']:>6} {t['wrong']:>6}"
        )

    print()
    print("[WRONG BY SITE]")
    for label, key in (("UniCorpusBuilder", "ucb"), ("Trafilatura", "traf")):
        counter = _wrong_by_site(rows, key)
        if not counter:
            print(f"  {label}: (none)")
            continue
        summary = ", ".join(f"{site}={n}" for site, n in counter.most_common())
        print(f"  {label}: {summary}")

    print()
    print("[NOTE] 'correct' counts personal names only; 'editorial' counts")
    print("       editorial bylines («Редакция», «Пресс-служба»). Report")
    print("       'correct / labeled' as precision and 'correct / filled' as")
    print("       yield. Do not average the two: they answer different questions.")
    print("       The four ratios above replace the earlier 'precision among labeled',")
    print("       which mixed personal names and editorial bylines in the numerator.")

    if args.out_csv:
        out_path = Path(args.out_csv)
        _write_stats_csv(out_path, systems)
        print(f"\n[OK] Stats CSV: {out_path}")

    logger.info(
        "stats: "
        + " ".join(
            f"{k}=correct:{v['correct']},editorial:{v['editorial']},"
            f"wrong:{v['wrong']},labeled:{v['labeled']},filled:{v['filled']}"
            for k, v in systems.items()
        )
    )
    return 0


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Manual author verification helper for the "
                    "trafilatura benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Typical workflow:

  1. Produce the benchmark CSV:
         python tools/benchmark_trafilatura.py

  2. Prepare the verification sheet (fetch HTML, fill signature
     and context columns):
         python tools/author_verification.py prepare

  3. Make a working copy and label it:
         copy author_verification.csv author_verification_filled.csv
         # edit author_verification_filled.csv, NOT the base sheet

  4. Summarize:
         python tools/author_verification.py stats author_verification_filled.csv
        """,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser(
        "prepare",
        help="Create or update the verification CSV",
    )
    p_prep.add_argument("--comparison", default="trafilatura_comparison.csv",
                        help="Benchmark CSV produced by "
                             "tools/benchmark_trafilatura.py")
    p_prep.add_argument("--out", default="author_verification.csv",
                        help="Verification CSV (created or updated)")
    p_prep.add_argument("--cache-dir", default="_cache_author_verification",
                        help="Directory for cached HTML")
    p_prep.add_argument("--config", default="config/universal.yaml",
                        help="Scraper config path")
    p_prep.add_argument("--no-cache", action="store_true",
                        help="Force re-fetch, ignore HTML cache")
    p_prep.add_argument("--include-meta", action="store_true",
                        help="Also include rows that already have a meta "
                             "author (for a full re-verification)")
    p_prep.add_argument("--limit", type=int, default=0,
                        help="Process only first N rows (0 = all)")
    p_prep.add_argument("--sleep", type=float, default=0.3,
                        help="Delay between fetches, seconds (default: 0.3)")
    p_prep.add_argument("--force", action="store_true",
                        help="Allow overwriting a sheet that already "
                             "contains manual verdicts and does not have "
                             "'filled' in its name")
    p_prep.set_defaults(func=cmd_prepare)

    p_stats = sub.add_parser(
        "stats",
        help="Summarize verdicts from the verification CSV",
    )
    p_stats.add_argument("infile", nargs="?", default="author_verification_filled.csv",
                         help="Verification CSV "
                              "(default: author_verification_filled.csv)")
    p_stats.add_argument("--out-csv", default=None,
                         help="Optional path to write the four metrics per "
                              "system as CSV")
    p_stats.set_defaults(func=cmd_stats)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
    