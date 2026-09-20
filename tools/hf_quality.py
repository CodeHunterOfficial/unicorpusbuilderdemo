# tools/hf_quality.py
"""
Measure quality metrics for the primary corpora of the four organisations
BashkirNLPWorld, TatarNLPWorld, OssetianNLPWorld, and TajikNLPWorld.

For each dataset the script reports:
  * language purity  — fraction of documents detected as the target
    language by fast-langdetect (lid.176)
  * exact-duplicate rate  — MD5 over NFKC-normalised, lowercased,
    punctuation-stripped text
  * near-duplicate rate  — MinHash LSH over character shingles
    (computed only when ``--near-dups`` is passed)

Only primary corpora (web, wiki, crisis) are processed. Derived datasets
(multiclass / multilabel / binary / cluster) are excluded because they
are subsets of the web corpora and do not add new documents.

Near-duplicate parameters
-------------------------
``NEAR_DUP_MIN_LEN = 500`` filters out short stubs and template-heavy
pages such as Wikipedia infoboxes and news cards. Shingles are
5-character sequences, which is more robust for agglutinative
morphology than word-level MinHash. Threshold is 0.9 with 256
permutations.

Output
------
    hf_quality.csv          one row per dataset
    hf_quality_summary.csv  aggregate per language

Installation
------------
    pip install datasets fast-langdetect datasketch tqdm

The lid.176.bin model (~126 MB) is downloaded on first use by
fast-langdetect and cached in the current directory.

Usage
-----
    python tools/hf_quality.py                       # sample=5000
    python tools/hf_quality.py --sample 2000
    python tools/hf_quality.py --groups tatar
    python tools/hf_quality.py --near-dups
    python tools/hf_quality.py --no-lang
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import os
import re
import unicodedata
from collections import defaultdict

# Silence Hugging Face and transformers noise before those modules are imported.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

from tqdm import tqdm

from _common import get_token, load_ds


# ============================================================
# Optional dependencies
# ============================================================

# Language detection via fast-langdetect
try:
    from fast_langdetect import detect as ft_detect
    FASTTEXT_OK = True
except ImportError:
    FASTTEXT_OK = False
    print("[WARN] fast-langdetect is not installed.")
    print("       Install with: pip install fast-langdetect")

# MinHash for near-duplicate detection
try:
    from datasketch import MinHash, MinHashLSH
    DATASKETCH_OK = True
except ImportError:
    DATASKETCH_OK = False
    print("[WARN] datasketch is not installed. Near-duplicates will be disabled.")
    print("       Install with: pip install datasketch")


# ============================================================
# Dataset registry — primary corpora only
# ============================================================

DATASETS = [
    # Bashkir
    ("BashkirNLPWorld/bashkir-web-corpus",     "ba", "web"),
    ("BashkirNLPWorld/bashkir-wiki-corpus",    "ba", "wiki"),
    # Tatar
    ("TatarNLPWorld/tatar-web-corpus-v3",      "tt", "web"),
    ("TatarNLPWorld/tatar-wiki-corpus",        "tt", "wiki"),
    # Ossetian
    ("OssetianNLPWorld/ossetian-web-corpus",   "os", "web"),
    # Tajik
    ("TajikNLPWorld/tajik-web-corpus",         "tg", "web"),
    ("TajikNLPWorld/tajik-wiki-corpus",        "tg", "wiki"),
    ("TajikNLPWorld/khf_news_labeled",         "tg", "crisis"),
]

GROUPS = {
    "bashkir":  [d for d in DATASETS if d[1] == "ba"],
    "tatar":    [d for d in DATASETS if d[1] == "tt"],
    "ossetian": [d for d in DATASETS if d[1] == "os"],
    "tajik":    [d for d in DATASETS if d[1] == "tg"],
}

# Text field priority; the first non-empty field is used for each example.
# hf_quality uses "body" and "article" in addition to the fields used by
# hf_count, because the web corpora in this project expose those fields.
TEXT_FIELDS = ["content", "text", "body", "article"]

# Minimum text length for a document to be counted at all.
MIN_TEXT_LEN = 50

# Number of leading characters passed to fastText.
FASTTEXT_MAX_CHARS = 500

# Near-duplicate parameters.
NEAR_DUP_MIN_LEN = 500      # shorter texts are skipped (template stubs)
NEAR_DUP_NUM_PERM = 256
NEAR_DUP_THRESHOLD = 0.9
NEAR_DUP_SHINGLE = 5        # 5-character shingles


# ============================================================
# Helpers
# ============================================================

def pick_text(example: dict) -> str:
    """Return the first non-empty text field from a dataset example."""
    for field in TEXT_FIELDS:
        v = example.get(field)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def normalize_for_hash(text: str) -> str:
    """
    Normalize text for exact-duplicate hashing.

    Applies NFKC normalization, lowercases, collapses whitespace, and
    strips punctuation. Two documents that differ only in these respects
    are treated as exact duplicates.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return text.strip()


def md5_hash(text: str) -> str:
    """Return the MD5 hex digest of the UTF-8 encoding of ``text``."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def minhash_of_chars(text: str,
                     num_perm: int = NEAR_DUP_NUM_PERM,
                     shingle_size: int = NEAR_DUP_SHINGLE) -> "MinHash":
    """
    Build a MinHash signature over character-level shingles.

    For agglutinative languages (Tatar, Bashkir, Ossetian) word-level
    MinHash produces false duplicates: the same lemma in different
    inflected forms appears as different tokens, while boilerplate
    blocks appear identical. Character shingles are more robust.
    """
    m = MinHash(num_perm=num_perm)
    if len(text) < shingle_size:
        return m
    for i in range(len(text) - shingle_size + 1):
        shingle = text[i:i + shingle_size]
        m.update(shingle.encode("utf-8"))
    return m


def detect_lang(text: str) -> str | None:
    """Return the language code predicted by fast_langdetect, or None on failure."""
    try:
        result = ft_detect(text, model="auto")
        if isinstance(result, list) and result:
            return result[0].get("lang")
    except Exception:
        pass
    return None


# ============================================================
# Per-dataset evaluation
# ============================================================

def evaluate_dataset(
    repo: str,
    lang: str,
    dtype: str,
    split_ds,
    sample_size: int,
    near_dups: bool,
    use_lang: bool,
):
    """Compute purity and duplicate metrics for a single dataset split."""
    total = 0
    lang_correct = 0
    lang_distribution = defaultdict(int)

    seen_hashes = set()
    exact_dups = 0

    if near_dups and DATASKETCH_OK:
        lsh = MinHashLSH(threshold=NEAR_DUP_THRESHOLD,
                         num_perm=NEAR_DUP_NUM_PERM)
    else:
        lsh = None

    near_dup_count = 0
    near_dup_total = 0   # denominator: only texts >= NEAR_DUP_MIN_LEN

    for i, ex in enumerate(tqdm(split_ds, desc=f"{repo} [{lang}/{dtype}]",
                                 unit="doc", leave=False)):
        if i >= sample_size:
            break

        text = pick_text(ex)
        if len(text) < MIN_TEXT_LEN:
            continue

        total += 1

        # Language purity
        if use_lang and FASTTEXT_OK:
            snippet = text[:FASTTEXT_MAX_CHARS].replace("\n", " ").strip()
            if snippet:
                pred = detect_lang(snippet)
                if pred:
                    lang_distribution[pred] += 1
                    if pred == lang:
                        lang_correct += 1

        # Exact duplicates
        norm = normalize_for_hash(text)
        h = md5_hash(norm)
        if h in seen_hashes:
            exact_dups += 1
            continue   # counted separately; not included in near-dup denominator
        seen_hashes.add(h)

        # Near-duplicates (long texts only)
        if lsh is not None and len(norm) >= NEAR_DUP_MIN_LEN:
            near_dup_total += 1
            m = minhash_of_chars(norm)
            if lsh.query(m):
                near_dup_count += 1
            else:
                lsh.insert(f"doc_{i}", m)

    near_dup_pct = (
        round(near_dup_count / near_dup_total * 100, 2)
        if near_dup_total > 0 else ""
    )

    return {
        "dataset": repo,
        "lang": lang,
        "type": dtype,
        "total": total,
        "lang_correct": lang_correct,
        "lang_purity_pct": round(lang_correct / total * 100, 1) if total else 0.0,
        "exact_dups": exact_dups,
        "exact_dup_pct": round(exact_dups / total * 100, 2) if total else 0.0,
        "near_dup_base": near_dup_total,
        "near_dups": near_dup_count if lsh is not None else "",
        "near_dup_pct": near_dup_pct,
        "top_other_langs": "; ".join(
            f"{l}={c}" for l, c in
            sorted(lang_distribution.items(), key=lambda x: -x[1])
            if l != lang
        )[:200],
    }


# ============================================================
# Entry point
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="*", choices=list(GROUPS.keys()),
                    default=list(GROUPS.keys()))
    ap.add_argument("--sample", type=int, default=5000,
                    help="Documents per dataset (default: 5000)")
    ap.add_argument("--near-dups", action="store_true",
                    help="Compute MinHash near-duplicates (slower)")
    ap.add_argument("--no-lang", action="store_true",
                    help="Disable language detection")
    ap.add_argument("--out", default="hf_quality.csv")
    ap.add_argument("--out-summary", default="hf_quality_summary.csv")
    ap.add_argument("--streaming", action="store_true")
    args = ap.parse_args()

    use_lang = not args.no_lang

    token = get_token()
    if token:
        print(f"[AUTH] HF_TOKEN: {token[:7]}...")
    else:
        print("[AUTH] HF_TOKEN not set; public datasets will be downloaded anonymously.")

    print(f"[MODE] sample={args.sample}, near-dups={args.near_dups}, lang={use_lang}")
    print(f"[LIBS] fast-langdetect={FASTTEXT_OK}, datasketch={DATASKETCH_OK}")
    if args.near_dups:
        print(f"[NEAR-DUP] threshold={NEAR_DUP_THRESHOLD}, "
              f"num_perm={NEAR_DUP_NUM_PERM}, "
              f"shingle={NEAR_DUP_SHINGLE} chars, "
              f"min_len={NEAR_DUP_MIN_LEN}")
    print()

    all_results = []

    for gname in args.groups:
        print(f"\n{'='*78}")
        print(f"GROUP: {gname.upper()}")
        print(f"{'='*78}")

        for repo, lang, dtype in GROUPS[gname]:
            print(f"\n[DATASET] {repo}  ({lang}, {dtype})")

            try:
                ds = load_ds(repo, token, args.streaming)
            except Exception as e:
                print(f"  [ERROR] {e}")
                all_results.append({
                    "dataset": repo, "lang": lang, "type": dtype,
                    "total": 0, "lang_correct": 0, "lang_purity_pct": 0.0,
                    "exact_dups": 0, "exact_dup_pct": 0.0,
                    "near_dup_base": 0, "near_dups": "", "near_dup_pct": "",
                    "top_other_langs": "",
                    "error": str(e)[:200],
                })
                continue

            split_name = None
            for candidate in ("full", "train"):
                if candidate in ds:
                    split_name = candidate
                    break
            if split_name is None:
                split_name = list(ds.keys())[0]

            print(f"  [SPLIT] using {split_name}")

            try:
                r = evaluate_dataset(
                    repo, lang, dtype, ds[split_name],
                    args.sample, args.near_dups, use_lang,
                )
                r["error"] = ""
                all_results.append(r)

                msg = (f"  -> total={r['total']:,}  "
                       f"purity={r['lang_purity_pct']}%  "
                       f"exact_dup={r['exact_dup_pct']}%")
                if args.near_dups and r["near_dup_pct"] != "":
                    msg += (f"  near_dup={r['near_dup_pct']}% "
                            f"(base={r['near_dup_base']:,})")
                print(msg)

                if r["top_other_langs"]:
                    print(f"     other languages: {r['top_other_langs']}")
            except Exception as e:
                print(f"  [ERROR] evaluate: {e}")
                all_results.append({
                    "dataset": repo, "lang": lang, "type": dtype,
                    "total": 0, "lang_correct": 0, "lang_purity_pct": 0.0,
                    "exact_dups": 0, "exact_dup_pct": 0.0,
                    "near_dup_base": 0, "near_dups": "", "near_dup_pct": "",
                    "top_other_langs": "",
                    "error": str(e)[:200],
                })

    # --- Summary by language ---
    print(f"\n\n{'='*78}")
    print("SUMMARY BY LANGUAGE")
    print(f"{'='*78}")
    print(f"{'Lang':<6} {'DS':>4} {'Docs':>9} "
          f"{'Purity':>9} {'Exact dup':>10} {'Near dup':>10}")
    print("-" * 78)

    by_lang = defaultdict(lambda: {
        "datasets": 0, "total": 0,
        "correct": 0, "exact": 0,
        "near": 0, "near_base": 0,
    })

    for r in all_results:
        if r["error"] or r["total"] == 0:
            continue
        l = by_lang[r["lang"]]
        l["datasets"] += 1
        l["total"] += r["total"]
        l["correct"] += r["lang_correct"]
        l["exact"] += r["exact_dups"]
        if isinstance(r["near_dups"], int):
            l["near"] += r["near_dups"]
            l["near_base"] += r["near_dup_base"]

    summary_rows = []
    for lang in sorted(by_lang.keys()):
        d = by_lang[lang]
        purity = round(d["correct"] / d["total"] * 100, 1) if d["total"] else 0
        edup = round(d["exact"] / d["total"] * 100, 2) if d["total"] else 0
        ndup = (round(d["near"] / d["near_base"] * 100, 2)
                if d["near_base"] > 0 else "")
        print(f"{lang:<6} {d['datasets']:>4} {d['total']:>9,} "
              f"{purity:>8.1f}% {edup:>9.2f}% "
              f"{(str(ndup) + '%') if ndup != '' else '—':>10}")
        summary_rows.append({
            "lang": lang,
            "datasets": d["datasets"],
            "total": d["total"],
            "purity_pct": purity,
            "exact_dup_pct": edup,
            "near_dup_pct": ndup if args.near_dups else "",
        })

    # --- Per-row CSV ---
    fields = ["dataset", "lang", "type", "total", "lang_correct",
              "lang_purity_pct", "exact_dups", "exact_dup_pct",
              "near_dup_base", "near_dups", "near_dup_pct",
              "top_other_langs", "error"]
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_results:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\n[OK] Per-row CSV:  {args.out}  (rows: {len(all_results)})")

    # --- Summary CSV ---
    with open(args.out_summary, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["lang", "datasets", "total",
                                           "purity_pct", "exact_dup_pct",
                                           "near_dup_pct"])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"[OK] Summary:      {args.out_summary}")


if __name__ == "__main__":
    main()