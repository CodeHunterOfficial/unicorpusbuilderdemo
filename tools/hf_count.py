# tools/hf_count.py
"""
Measure the size of every dataset published by the four organisations
BashkirNLPWorld, TatarNLPWorld, OssetianNLPWorld, and TajikNLPWorld.

For each split of each dataset the script reports:
  * number of documents
  * number of characters
  * number of tokens under xlm-roberta-base and facebook/mbart-large-50

Aggregate summaries are printed per group and in total.

Authentication
--------------
The Hugging Face token is read from the HF_TOKEN environment variable.
Public datasets can be downloaded anonymously; gated datasets require
a token.

Dataset loading
---------------
Some repositories in TatarNLPWorld and OssetianNLPWorld carry a
``dataset_info.json`` that disagrees with the underlying Parquet files.
The shared ``_common.load_ds`` helper handles this transparently with
two fallbacks (see its docstring).

Installation
------------
    pip install datasets transformers tqdm

Usage
-----
    # All groups, all primary splits (full / train / validation)
    python tools/hf_count.py

    # Tajik datasets only
    python tools/hf_count.py --groups tajik

    # Ossetian datasets only
    python tools/hf_count.py --groups ossetian

    # Include ``sample`` splits (skipped by default)
    python tools/hf_count.py --include-samples

    # Restrict to specific split names
    python tools/hf_count.py --splits full train

    # Stream without caching to disk
    python tools/hf_count.py --streaming

Output
------
    hf_count.csv          one row per (dataset, split)
    hf_count_summary.csv  aggregate per group and in total
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
from collections import defaultdict

# Silence Hugging Face and transformers noise before those modules are imported.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

from transformers import AutoTokenizer
from tqdm import tqdm

from _common import get_token, load_ds


# ============================================================
# Dataset registry
# ============================================================

DATASETS_BASHKIR = [
    ("BashkirNLPWorld/bashkir-web-corpus",           "ba", "web"),
    ("BashkirNLPWorld/bashkir-news-multiclass",      "ba", "multiclass"),
    ("BashkirNLPWorld/bashkir-news-multilabel",      "ba", "multilabel"),
    ("BashkirNLPWorld/bashkir-news-binary",          "ba", "binary"),
    ("BashkirNLPWorld/bashkir-news-cluster",         "ba", "cluster"),
    ("BashkirNLPWorld/bashkir-wiki-corpus",          "ba", "wiki"),
]

DATASETS_TATAR = [
    ("TatarNLPWorld/tatar-news-analysis-multilabel", "tt", "multilabel"),
    ("TatarNLPWorld/tatar-news-analysis-multiclass", "tt", "multiclass"),
    ("TatarNLPWorld/tatar-news-analysis-binary",     "tt", "binary"),
    ("TatarNLPWorld/tatar-news-cluster",             "tt", "cluster"),
    ("TatarNLPWorld/tatar-web-corpus",               "tt", "web_v2"),
    ("TatarNLPWorld/tatar-web-corpus-v3",            "tt", "web_v3"),
    ("TatarNLPWorld/tatar-wiki-corpus",              "tt", "wiki"),
]

DATASETS_OSSETIAN = [
    ("OssetianNLPWorld/ossetian-web-corpus",         "os", "web"),
    ("OssetianNLPWorld/ossetian-news-cluster",       "os", "cluster"),
    ("OssetianNLPWorld/ossetian-news-multilabel",    "os", "multilabel"),
    ("OssetianNLPWorld/ossetian-news-binary",        "os", "binary"),
    ("OssetianNLPWorld/ossetian-news-multiclass",    "os", "multiclass"),
]

DATASETS_TAJIK = [
    ("TajikNLPWorld/tajik-web-corpus",               "tg", "web"),
    ("TajikNLPWorld/tajik-news-multiclass",          "tg", "multiclass"),
    ("TajikNLPWorld/tajik-news-binary",              "tg", "binary"),
    ("TajikNLPWorld/tajik-news-multilabel",          "tg", "multilabel"),
    ("TajikNLPWorld/tajik-news-cluster",             "tg", "cluster"),
    ("TajikNLPWorld/tajik-wiki-corpus",              "tg", "wiki"),
    ("TajikNLPWorld/khf_news_labeled",               "tg", "crisis"),
]

GROUPS = {
    "bashkir":  DATASETS_BASHKIR,
    "tatar":    DATASETS_TATAR,
    "ossetian": DATASETS_OSSETIAN,
    "tajik":    DATASETS_TAJIK,
}

# Text field priority; the first non-empty field is used for each example.
TEXT_FIELDS = ["content", "text", "title"]

# Number of examples passed to the tokenizer at once.
TOKENIZE_BATCH = 256


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


def count_split(dataset, tokenizers, desc: str, batch_size: int = TOKENIZE_BATCH):
    """
    Count documents, characters, and tokens over a single split.

    Tokenization is batched: web corpora are an order of magnitude
    faster when the tokenizer receives several examples at once.
    """
    docs = 0
    chars = 0
    tokens = {name: 0 for name in tokenizers}

    def flush(batch):
        nonlocal docs, chars
        if not batch:
            return
        docs += len(batch)
        chars += sum(len(t) for t in batch)
        for name, tok in tokenizers.items():
            enc = tok(batch, add_special_tokens=False)
            tokens[name] += sum(len(ids) for ids in enc["input_ids"])

    batch = []
    for ex in tqdm(dataset, desc=desc, unit="doc", leave=False):
        txt = pick_text(ex)
        if not txt:
            continue
        batch.append(txt)
        if len(batch) >= batch_size:
            flush(batch)
            batch = []
    flush(batch)

    return docs, chars, tokens


def iter_targets(args):
    """Yield (group_name, repo_id, language, dataset_type) for each dataset to process."""
    if args.datasets:
        for repo in args.datasets:
            yield ("custom", repo, "?", "?")
        return
    for gname in args.groups:
        for repo, lang, dtype in GROUPS[gname]:
            yield (gname, repo, lang, dtype)


# ============================================================
# Entry point
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="*", choices=list(GROUPS.keys()),
                    default=list(GROUPS.keys()))
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Explicit list of repo_id values (overrides --groups)")
    ap.add_argument("--splits", nargs="*", default=None,
                    help="Restrict to these split names")
    ap.add_argument("--include-samples", action="store_true",
                    help="Include sample splits (skipped by default)")
    ap.add_argument("--out", default="hf_count.csv")
    ap.add_argument("--out-summary", default="hf_count_summary.csv")
    ap.add_argument("--streaming", action="store_true")
    args = ap.parse_args()

    token = get_token()
    if token:
        print(f"[AUTH] HF_TOKEN: {token[:7]}... (len={len(token)})")
    else:
        print("[AUTH] HF_TOKEN not set; public datasets will be downloaded anonymously.")

    print("\n[LOAD] Tokenizers...")
    tokenizers = {}
    for name, hf in (("xlm_r", "xlm-roberta-base"),
                     ("mbart", "facebook/mbart-large-50")):
        print(f"  - {name:<6} {hf}")
        tokenizers[name] = AutoTokenizer.from_pretrained(hf)
    print()

    if args.streaming:
        print("[MODE] STREAMING — no on-disk caching.\n")

    # Groups to print in the summary even if every dataset fails.
    requested_groups = ["custom"] if args.datasets else list(args.groups)

    rows = []
    group_totals = defaultdict(lambda: {
        "docs": 0, "chars": 0,
        "tokens_xlm_r": 0, "tokens_mbart": 0,
        "datasets": 0,
    })

    current_group = None
    for group_name, repo, lang, dtype in iter_targets(args):
        if group_name != current_group:
            current_group = group_name
            print(f"\n{'='*78}")
            print(f"GROUP: {group_name.upper()}")
            print(f"{'='*78}")

        print(f"\n[DATASET] {repo}  ({lang}, {dtype})")

        try:
            ds = load_ds(repo, token, args.streaming)
        except Exception as e:
            print(f"  [ERROR] {e}")
            rows.append({
                "group": group_name, "lang": lang, "type": dtype,
                "dataset": repo, "split": "", "docs": 0, "chars": 0,
                "tokens_xlm_r": 0, "tokens_mbart": 0,
                "tokens_xlm_r_per_doc": 0.0, "tokens_mbart_per_doc": 0.0,
                "error": str(e)[:200],
            })
            continue

        for split_name, split_ds in ds.items():
            if args.splits:
                if split_name not in args.splits:
                    print(f"  [SKIP] split={split_name}")
                    continue
            elif not args.include_samples and split_name.startswith("sample"):
                print(f"  [SKIP] split={split_name} (use --include-samples)")
                continue

            try:
                docs, chars, tokens = count_split(
                    split_ds, tokenizers, f"{repo} [{split_name}]"
                )
            except Exception as e:
                print(f"  [ERROR] split={split_name}: {e}")
                rows.append({
                    "group": group_name, "lang": lang, "type": dtype,
                    "dataset": repo, "split": split_name,
                    "docs": 0, "chars": 0,
                    "tokens_xlm_r": 0, "tokens_mbart": 0,
                    "tokens_xlm_r_per_doc": 0.0, "tokens_mbart_per_doc": 0.0,
                    "error": f"split error: {str(e)[:180]}",
                })
                continue

            if docs == 0:
                print(f"  split={split_name:<11} docs=0 (empty)")
                continue

            tpd_x = tokens["xlm_r"] / docs
            tpd_m = tokens["mbart"] / docs

            print(f"  split={split_name:<11} "
                  f"docs={docs:>10,}  "
                  f"chars={chars:>13,}  "
                  f"xlm-r={tokens['xlm_r']:>14,} ({tpd_x:>6.0f}/doc)  "
                  f"mbart={tokens['mbart']:>14,} ({tpd_m:>6.0f}/doc)")

            rows.append({
                "group": group_name, "lang": lang, "type": dtype,
                "dataset": repo, "split": split_name,
                "docs": docs, "chars": chars,
                "tokens_xlm_r": tokens["xlm_r"],
                "tokens_mbart": tokens["mbart"],
                "tokens_xlm_r_per_doc": round(tpd_x, 1),
                "tokens_mbart_per_doc": round(tpd_m, 1),
                "error": "",
            })

            g = group_totals[group_name]
            g["docs"] += docs
            g["chars"] += chars
            g["tokens_xlm_r"] += tokens["xlm_r"]
            g["tokens_mbart"] += tokens["mbart"]
            g["datasets"] += 1

    # --- Summary by group ---
    print(f"\n\n{'='*78}")
    print("SUMMARY BY GROUP")
    print(f"{'='*78}")
    print(f"{'Group':<12} {'Datasets':>10} {'Documents':>14} "
          f"{'Characters':>16} {'xlm-r tokens':>16} {'mBART tokens':>16}")
    print("-" * 92)

    empty = {"datasets": 0, "docs": 0, "chars": 0,
             "tokens_xlm_r": 0, "tokens_mbart": 0}

    grand_docs = grand_chars = grand_x = grand_m = 0
    for gname in requested_groups:
        g = group_totals.get(gname) or empty
        print(f"{gname:<12} {g['datasets']:>10} {g['docs']:>14,} "
              f"{g['chars']:>16,} {g['tokens_xlm_r']:>16,} {g['tokens_mbart']:>16,}")
        grand_docs  += g["docs"]
        grand_chars += g["chars"]
        grand_x     += g["tokens_xlm_r"]
        grand_m     += g["tokens_mbart"]

    print("-" * 92)
    print(f"{'TOTAL':<12} {'':>10} {grand_docs:>14,} "
          f"{grand_chars:>16,} {grand_x:>16,} {grand_m:>16,}")

    # --- Per-row CSV ---
    fields = ["group", "lang", "type", "dataset", "split",
              "docs", "chars", "tokens_xlm_r", "tokens_mbart",
              "tokens_xlm_r_per_doc", "tokens_mbart_per_doc", "error"]
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[OK] Per-row CSV:    {args.out}  (rows: {len(rows)})")

    # --- Summary CSV ---
    with open(args.out_summary, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "datasets", "docs", "chars",
                    "tokens_xlm_r", "tokens_mbart"])
        for gname in requested_groups:
            g = group_totals.get(gname) or empty
            w.writerow([gname, g["datasets"], g["docs"], g["chars"],
                        g["tokens_xlm_r"], g["tokens_mbart"]])
        w.writerow(["TOTAL", sum((group_totals.get(g) or empty)["datasets"]
                                 for g in requested_groups),
                    grand_docs, grand_chars, grand_x, grand_m])
    print(f"[OK] Summary by group: {args.out_summary}")


if __name__ == "__main__":
    main()