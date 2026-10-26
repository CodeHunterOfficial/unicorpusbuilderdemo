# tools/hf_count.py
"""
Measure dataset sizes for BashkirNLPWorld, TatarNLPWorld,
OssetianNLPWorld, and TajikNLPWorld.

For every split the script reports:
  * document count
  * character count
  * token counts under one or more tokenizers

Default tokenizers:
  * xlm-roberta-base              (XLM-R, SentencePiece, ~250k vocab)
  * google/gemma-3-270m           (Gemma 3, ~262k vocab)

Gemma 3 is gated on Hugging Face. Accept the license at
https://huggingface.co/google/gemma-3-270m and provide HF_TOKEN.

Authentication
--------------
HF_TOKEN environment variable. Public datasets can be downloaded
anonymously; gated datasets (Gemma 3, some corpora) require a token.

Dataset loading
---------------
Some repositories carry a `dataset_info.json` that disagrees with the
underlying Parquet shards. `load_dataset_with_fallback` handles this
with two attempts:
  1. standard `load_dataset(..., verification_mode="no_checks")`
  2. direct Parquet loader via `load_dataset("parquet", ...)`

Installation
------------
    pip install "datasets>=5.0" "transformers>=4.50" tqdm

Usage
-----
    python tools/hf_count.py
    python tools/hf_count.py --groups tajik
    python tools/hf_count.py --tokenizers xlm-roberta-base google/gemma-3-270m
    python tools/hf_count.py --gemma3-model google/gemma-3-1b-it
    python tools/hf_count.py --no-gemma3
    python tools/hf_count.py --streaming
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
from collections import defaultdict
from typing import Iterable

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

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

# Text field priority; first non-empty field is used.
TEXT_FIELDS = ["content", "text", "title"]

# Batch size for tokenizer calls.
TOKENIZE_BATCH = 256

# Default tokenizer set.
DEFAULT_TOKENIZERS = {
    "xlm_r": "xlm-roberta-base",
    "gemma3": "google/gemma-3-270m",
}


# ============================================================
# Helpers
# ============================================================

def get_token() -> str | None:
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        value = os.environ.get(var)
        if value:
            return value.strip()
    return None


def load_dataset_with_fallback(repo: str, token: str | None, streaming: bool):
    """
    Load a dataset with two attempts.

    1. Standard loader with verification_mode="no_checks".
    2. Direct Parquet loader.
    """
    try:
        return load_dataset(
            repo,
            token=token,
            streaming=streaming,
            verification_mode="no_checks",
        )
    except Exception as exc:
        print(f"  [WARN] standard loader failed: {type(exc).__name__}: {exc}")
        print("  [RETRY] parquet loader...")

    return load_dataset(
        "parquet",
        data_files=f"hf://datasets/{repo}/data/*.parquet",
        token=token,
        streaming=streaming,
    )


def pick_text(example: dict) -> str:
    """Return the first non-empty text field."""
    for field in TEXT_FIELDS:
        value = example.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def count_split(
    dataset,
    tokenizers: dict[str, AutoTokenizer],
    desc: str,
    batch_size: int = TOKENIZE_BATCH,
) -> tuple[int, int, dict[str, int]]:
    """
    Count documents, characters, and tokens for one split.

    Tokenizer calls are batched. `return_length=True` returns the number
    of tokens per example without materializing full input_ids when the
    fast tokenizer supports it.
    """
    docs = 0
    chars = 0
    tokens = {name: 0 for name in tokenizers}

    def flush(batch: list[str]) -> None:
        nonlocal docs, chars
        if not batch:
            return
        docs += len(batch)
        chars += sum(len(text) for text in batch)
        for name, tok in tokenizers.items():
            encoded = tok(
                batch,
                add_special_tokens=False,
                return_length=True,
            )
            tokens[name] += sum(encoded["length"])

    batch: list[str] = []
    for example in tqdm(dataset, desc=desc, unit="doc", leave=False):
        text = pick_text(example)
        if not text:
            continue
        batch.append(text)
        if len(batch) >= batch_size:
            flush(batch)
            batch = []
    flush(batch)

    return docs, chars, tokens


def iter_targets(args) -> Iterable[tuple[str, str, str, str]]:
    """Yield (group, repo, language, dataset_type)."""
    if args.datasets:
        for repo in args.datasets:
            yield ("custom", repo, "?", "?")
        return
    for group_name in args.groups:
        for repo, lang, dtype in GROUPS[group_name]:
            yield (group_name, repo, lang, dtype)


def build_tokenizers(args) -> dict[str, AutoTokenizer]:
    """
    Build the tokenizer set.

    If `--tokenizers` is given, use exactly those model names.
    Otherwise use the default set, optionally skipping Gemma 3.
    """
    if args.tokenizers:
        names = {name.replace("/", "_").replace("-", "_"): name for name in args.tokenizers}
    else:
        names = dict(DEFAULT_TOKENIZERS)
        if args.no_gemma3:
            names.pop("gemma3", None)
        if args.gemma3_model:
            names["gemma3"] = args.gemma3_model

    tokenizers: dict[str, AutoTokenizer] = {}
    for key, model_name in names.items():
        print(f"  - {key:<12} {model_name}")
        try:
            tokenizers[key] = AutoTokenizer.from_pretrained(
                model_name,
                token=args.hf_token,
            )
        except Exception as exc:
            print(f"    [SKIP] {type(exc).__name__}: {exc}")
            if key == "gemma3":
                print("    Gemma 3 is gated. Accept the license at")
                print(f"    https://huggingface.co/{model_name}")
    return tokenizers


# ============================================================
# Entry point
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Count documents, characters, and tokens per dataset split."
    )
    ap.add_argument("--groups", nargs="*", choices=list(GROUPS.keys()),
                    default=list(GROUPS.keys()))
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Explicit repo IDs (overrides --groups)")
    ap.add_argument("--splits", nargs="*", default=None,
                    help="Restrict to these split names")
    ap.add_argument("--include-samples", action="store_true",
                    help="Include sample splits (skipped by default)")
    ap.add_argument("--tokenizers", nargs="*", default=None,
                    help="Explicit tokenizer model names (overrides defaults)")
    ap.add_argument("--gemma3-model", default=None,
                    help="Gemma 3 checkpoint (default: google/gemma-3-270m)")
    ap.add_argument("--no-gemma3", action="store_true",
                    help="Disable Gemma 3 tokenizer")
    ap.add_argument("--out", default="hf_count.csv")
    ap.add_argument("--out-summary", default="hf_count_summary.csv")
    ap.add_argument("--streaming", action="store_true")
    args = ap.parse_args()

    args.hf_token = get_token()
    if args.hf_token:
        print(f"[AUTH] HF_TOKEN: {args.hf_token[:7]}... (len={len(args.hf_token)})")
    else:
        print("[AUTH] HF_TOKEN not set; gated models will be skipped.")

    print("\n[LOAD] Tokenizers...")
    tokenizers = build_tokenizers(args)
    if not tokenizers:
        print("[ERROR] No tokenizer loaded. Aborting.")
        return
    print()

    if args.streaming:
        print("[MODE] streaming=True — no on-disk cache.\n")

    requested_groups = ["custom"] if args.datasets else list(args.groups)

    rows: list[dict] = []
    group_totals: dict[str, dict] = defaultdict(
        lambda: {"docs": 0, "chars": 0, "datasets": 0, **{k: 0 for k in tokenizers}}
    )

    current_group = None
    for group_name, repo, lang, dtype in iter_targets(args):
        if group_name != current_group:
            current_group = group_name
            print(f"\n{'=' * 78}")
            print(f"GROUP: {group_name.upper()}")
            print(f"{'=' * 78}")

        print(f"\n[DATASET] {repo}  ({lang}, {dtype})")

        try:
            ds = load_dataset_with_fallback(repo, args.hf_token, args.streaming)
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            rows.append({
                "group": group_name, "lang": lang, "type": dtype,
                "dataset": repo, "split": "", "docs": 0, "chars": 0,
                **{f"tokens_{k}": 0 for k in tokenizers},
                **{f"tokens_{k}_per_doc": 0.0 for k in tokenizers},
                "error": str(exc)[:200],
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
            except Exception as exc:
                print(f"  [ERROR] split={split_name}: {exc}")
                rows.append({
                    "group": group_name, "lang": lang, "type": dtype,
                    "dataset": repo, "split": split_name,
                    "docs": 0, "chars": 0,
                    **{f"tokens_{k}": 0 for k in tokenizers},
                    **{f"tokens_{k}_per_doc": 0.0 for k in tokenizers},
                    "error": f"split error: {str(exc)[:180]}",
                })
                continue

            if docs == 0:
                print(f"  split={split_name:<11} docs=0 (empty)")
                continue

            per_doc = {k: tokens[k] / docs for k in tokenizers}
            row = {
                "group": group_name, "lang": lang, "type": dtype,
                "dataset": repo, "split": split_name,
                "docs": docs, "chars": chars,
                **{f"tokens_{k}": tokens[k] for k in tokenizers},
                **{f"tokens_{k}_per_doc": round(per_doc[k], 1) for k in tokenizers},
                "error": "",
            }
            rows.append(row)

            print(f"  split={split_name:<11} docs={docs:>10,}  chars={chars:>13,}")
            for key in tokenizers:
                print(f"    {key:<12} {tokens[key]:>14,}  ({per_doc[key]:>6.0f}/doc)")

            total = group_totals[group_name]
            total["docs"] += docs
            total["chars"] += chars
            for key in tokenizers:
                total[key] += tokens[key]
            total["datasets"] += 1

    # ---- Summary by group ----
    print(f"\n\n{'=' * 78}")
    print("SUMMARY BY GROUP")
    print(f"{'=' * 78}")

    header = f"{'Group':<12} {'Datasets':>10} {'Documents':>14} {'Characters':>16}"
    for key in tokenizers:
        header += f" {key + ' tokens':>16}"
    print(header)
    print("-" * (70 + 17 * len(tokenizers)))

    empty = {"datasets": 0, "docs": 0, "chars": 0, **{k: 0 for k in tokenizers}}
    grand = {"docs": 0, "chars": 0, "datasets": 0, **{k: 0 for k in tokenizers}}

    for name in requested_groups:
        data = group_totals.get(name) or empty
        line = (f"{name:<12} {data['datasets']:>10} {data['docs']:>14,} "
                f"{data['chars']:>16,}")
        for key in tokenizers:
            line += f" {data[key]:>16,}"
        print(line)
        for field in ("docs", "chars", "datasets"):
            grand[field] += data[field]
        for key in tokenizers:
            grand[key] += data[key]

    print("-" * (70 + 17 * len(tokenizers)))
    line = (f"{'TOTAL':<12} {'':>10} {grand['docs']:>14,} "
            f"{grand['chars']:>16,}")
    for key in tokenizers:
        line += f" {grand[key]:>16,}"
    print(line)

    # ---- Per-row CSV ----
    fields = ["group", "lang", "type", "dataset", "split", "docs", "chars"]
    fields += [f"tokens_{k}" for k in tokenizers]
    fields += [f"tokens_{k}_per_doc" for k in tokenizers]
    fields += ["error"]

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[OK] Per-row CSV:    {args.out}  (rows: {len(rows)})")

    # ---- Summary CSV ----
    with open(args.out_summary, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "datasets", "docs", "chars"]
                        + [f"tokens_{k}" for k in tokenizers])
        for name in requested_groups:
            data = group_totals.get(name) or empty
            writer.writerow([name, data["datasets"], data["docs"], data["chars"]]
                            + [data[k] for k in tokenizers])
        writer.writerow(["TOTAL",
                         sum((group_totals.get(g) or empty)["datasets"]
                             for g in requested_groups),
                         grand["docs"], grand["chars"]]
                        + [grand[k] for k in tokenizers])
    print(f"[OK] Summary by group: {args.out_summary}")


if __name__ == "__main__":
    main()
    