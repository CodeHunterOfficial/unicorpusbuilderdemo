# tools/author_verification.py
"""
Author verification helper for the trafilatura benchmark.

Purpose
-------
``tools/benchmark_trafilatura.py`` checks author correctness only on
documents that carry ``<meta name="author">`` (14 of 138 in the current
run). The remaining 124 are the ones the reviewer flagged: no meta
reference exists, so correctness was never verified.

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

``stats`` mode aggregates the labelled CSV into the numbers requested
by the reviewer (checked / correct / wrong / share of filled).

Design decisions
----------------
* HTML is cached on disk (``_cache_author_verification/``). A rerun
  reuses the cache unless ``--no-cache`` is passed.
* The output CSV is keyed by ``url``. On a rerun the ``verdict_*`` and
  ``notes`` columns are preserved, even if the signature columns are
  recomputed.
* Signature lookup is deliberately permissive: it returns the first
  matching pattern AND a context snippet, so the human can see the
  whole byline block. Precision is not the goal here; coverage is.
* No automatic labelling, no ML. Every ``verdict`` is a human decision.

Usage
-----
    # 1. Prepare the verification sheet
    python tools/author_verification.py prepare

    # 2. Fill ``verdict_ucb`` / ``verdict_traf`` in the CSV, save

    # 3. Summarize
    python tools/author_verification.py stats

    # Custom paths, force re-fetch, limit for testing
    python tools/author_verification.py prepare \\
        --comparison trafilatura_comparison.csv \\
        --out author_verification.csv \\
        --no-cache --limit 20

Exit codes
----------
    prepare: 0 on success; 1 if the comparison CSV is missing or empty.
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

# Verdicts that count as "a correct byline was extracted" in stats.
# ``editorial`` is a valid byline (Редакция, Пресс-служба) but not a
# personal name; it is reported separately in ``stats``.
CORRECT_VERDICTS = {"correct", "editorial"}


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
        if r.get("error"):
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

    rows = read_comparison(comparison_path, include_meta=args.include_meta)
    if not rows:
        print("[INFO] No rows need manual verification.")
        print("       All benchmark rows either have a meta author or no "
              "system returned an author.")
        return 1

    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    print(f"[PLAN] {len(rows)} URLs need manual author verification")

    out_path = Path(args.out)
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
    print(f"\n[NEXT] Fill the ``verdict_ucb`` / ``verdict_traf`` columns:")
    print("       correct    — matches a personal name on the page")
    print("       editorial  — page shows an editorial byline "
          "(«Редакция», «Пресс-служба»)")
    print("       wrong      — does not match anything on the page")
    print("       unknown    — cannot determine")
    print("       Empty cells are treated as not yet labelled.")
    print(f"\n       Then run: python {sys.argv[0]} stats")
    return 0


# ------------------------------------------------------------------
# Stats command
# ------------------------------------------------------------------

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

    def norm(v: str) -> str:
        return (v or "").strip().lower()

    systems: dict[str, dict] = {}
    for key in ("ucb", "traf"):
        author_col = f"{key}_author"
        verdict_col = f"verdict_{key}"

        filled = [r for r in rows if norm(r.get(author_col))]
        labeled = [
            r for r in filled
            if norm(r.get(verdict_col)) in {"correct", "editorial", "wrong"}
        ]
        correct = [r for r in labeled if norm(r.get(verdict_col)) == "correct"]
        editorial = [r for r in labeled if norm(r.get(verdict_col)) == "editorial"]
        wrong = [r for r in labeled if norm(r.get(verdict_col)) == "wrong"]
        unknown = [r for r in filled if norm(r.get(verdict_col)) == "unknown"]
        unlabeled = [r for r in filled if not norm(r.get(verdict_col))]

        precision_labeled = (
            (len(correct) + len(editorial)) / len(labeled) if labeled else None
        )
        correctness_over_filled = (
            (len(correct) + len(editorial)) / len(filled) if filled else None
        )

        systems[key] = {
            "filled": len(filled),
            "labeled": len(labeled),
            "correct": len(correct),
            "editorial": len(editorial),
            "wrong": len(wrong),
            "unknown": len(unknown),
            "unlabeled": len(unlabeled),
            "precision_labeled": precision_labeled,
            "correctness_over_filled": correctness_over_filled,
        }

    total = len(rows)
    both = sum(1 for r in rows
               if norm(r.get("ucb_author")) and norm(r.get("traf_author")))
    only_ucb = sum(1 for r in rows
                   if norm(r.get("ucb_author")) and not norm(r.get("traf_author")))
    only_traf = sum(1 for r in rows
                    if norm(r.get("traf_author")) and not norm(r.get("ucb_author")))

    print(f"\n{'=' * 70}")
    print("[AUTHOR VERIFICATION SUMMARY]")
    print(f"{'=' * 70}")
    print(f"  Documents in CSV:                {total}")
    print(f"    both systems filled author:    {both}")
    print(f"    only UniCorpusBuilder filled:  {only_ucb}")
    print(f"    only Trafilatura filled:       {only_traf}")
    print()

    for key, label in (("ucb", "UniCorpusBuilder"), ("traf", "Trafilatura")):
        s = systems[key]
        print(f"  --- {label} ---")
        print(f"    author filled:                 {s['filled']}")
        print(f"    manually labeled:              {s['labeled']}"
              f"   (unknown: {s['unknown']}, unlabeled: {s['unlabeled']})")
        print(f"    correct (personal name):       {s['correct']}")
        print(f"    editorial (not a person):      {s['editorial']}")
        print(f"    wrong:                         {s['wrong']}")
        if s["precision_labeled"] is not None:
            print(f"    precision among labeled:       "
                  f"{s['precision_labeled'] * 100:.1f}%")
        else:
            print(f"    precision among labeled:       N/A")
        if s["correctness_over_filled"] is not None:
            print(f"    correct+editorial / filled:    "
                  f"{s['correctness_over_filled'] * 100:.1f}%")
        print()

    print("[NOTE] 'editorial' counts as a correct byline but is not a "
          "personal name.\n"
          "       Report precision (correct / labeled) and share of "
          "filled (correct+editorial / filled)\n"
          "       separately — they answer different questions.")

    logger.info(
        "stats: "
        + " ".join(
            f"{k}=correct:{v['correct']},editorial:{v['editorial']},"
            f"wrong:{v['wrong']},labeled:{v['labeled']}"
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

  3. Open author_verification.csv in Excel / LibreOffice / Sheets,
     fill ``verdict_ucb`` and ``verdict_traf``, save.

  4. Summarize:
         python tools/author_verification.py stats
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
    p_prep.set_defaults(func=cmd_prepare)

    p_stats = sub.add_parser(
        "stats",
        help="Summarize verdicts from the verification CSV",
    )
    p_stats.add_argument("infile", nargs="?", default="author_verification.csv",
                         help="Verification CSV "
                              "(default: author_verification.csv)")
    p_stats.set_defaults(func=cmd_stats)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())