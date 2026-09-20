# tools/_common.py
"""
Shared helpers for the tools/ scripts.

Only infrastructure helpers live here. Functions that define the
reported metric (gold-text extraction, word-level P/R/F1) are
intentionally kept local to each script so that a reader can audit
the metric in the same file that prints it.
"""
from __future__ import annotations

import os
import re
import statistics

from datasets import load_dataset


WORD_RE = re.compile(r"\w+", re.UNICODE)


# ---------------------------------------------------------------------
# Hugging Face helpers
# ---------------------------------------------------------------------

def get_token() -> str | None:
    """Return the Hugging Face token from the environment, or None."""
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        v = os.environ.get(var)
        if v:
            return v.strip()
    return None


def load_ds(repo: str, token, streaming: bool):
    """
    Load a dataset with two fallbacks.

    1. ``verification_mode="no_checks"`` — fixes SplitInfo mismatch when
       ``dataset_info.json`` disagrees with the actual parquet files.
    2. Direct parquet loader — fixes schema mismatch or CastError on the
       same class of datasets.

    If both attempts fail, the exception propagates to the caller.
    """
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


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# URL discovery (shared by benchmark_trafilatura and strategy_hitrate)
# ---------------------------------------------------------------------

def discover_urls_for_site(engine, site_key, site_cfg, limit):
    """
    Return up to ``limit`` article URLs from a single site.

    Order of attempts:
      1. Parse the site's start page with ``extract_listing_items_stub``.
      2. Walk rubrics collected by ``engine.collect_rubrics``.
      3. Try a small set of conventional /category/ and /news/ URLs.
    """
    match = site_cfg.get("match", [])
    start_url = site_cfg.get("start_url") or (f"https://{match[0]}" if match else None)
    if not start_url:
        return []

    urls, seen = [], set()

    # Step 1: try the start page directly
    try:
        html = engine.fetch_html(start_url)
        items = engine.extract_listing_items_stub(html, start_url)
        for it in items:
            u = it.get("url")
            if u and u not in seen:
                seen.add(u)
                urls.append(u)
                if len(urls) >= limit:
                    return urls
    except Exception:
        pass

    # Step 2: discover rubric pages
    rubrics = []
    try:
        rubrics = engine.collect_rubrics(start_url)
    except Exception:
        rubrics = []

    # Step 3: fall back to conventional URL shapes
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
        try:
            html = engine.fetch_html(rubric_url)
        except Exception:
            continue

        try:
            items = engine.extract_listing_items_stub(html, rubric_url)
        except Exception:
            items = []

        for it in items:
            u = it.get("url")
            if u and u not in seen:
                seen.add(u)
                urls.append(u)
                if len(urls) >= limit:
                    break

    return urls