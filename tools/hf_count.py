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
underlying Parquet shards (e.g. `tatar-web-corpus-v3` declares
`id/text/label/source/date` while the Parquet files have
`content/title/label/label_text/category/content_length/resource/date`).
The standard loader aborts with ``column names don't match`` and
`verification_mode="no_checks"` does not help. `load_dataset_with_fallback`
therefore tries four Parquet patterns before giving up.

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
    python tools/hf_count.py --max-docs 5000
    python tools/hf_count.py --resume hf_count.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import time
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

TEXT_FIELDS = ["content", "text", "title"]

# Larger batch = better CPU utilisation on 1M+ splits
TOKENIZE_BATCH = 1024

# Warn when a split is big enough to need --streaming
LARGE_DATASET_WARN = 500_000

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
    2. Direct Parquet loader, trying four file patterns.

    Handles the `column names don't match` schema mismatch that
    `tatar-web-corpus` (v2 and v3) hits: the declared
    `dataset_info.json` schema disagrees with the Parquet shards.
    """
    try:
        return load_dataset(
            repo,
            token=token,
            streaming=streaming,
            verification_mode="no_checks",
        )
    except Exception as exc:
        msg = str(exc).lower()
        if "column names don't match" in msg:
            print("  [SCHEMA-MISMATCH] declared schema does not match Parquet "
                  "shards; falling back to Parquet loader")
        else:
            print(f"  [WARN] standard loader failed: "
                  f"{type(exc).__name__}: {str(exc)[:200]}")
        print("  [RETRY] parquet loader...")

    patterns = [
        f"hf://datasets/{repo}/data/*.parquet",
        f"hf://datasets/{repo}/*.parquet",
        f"hf://datasets/{repo}/data/**/*.parquet",
        f"hf://datasets/{repo}/**/*.parquet",
    ]
    last_exc: Exception | None = None
    for pattern in patterns:
        try:
            print(f"    trying: {pattern}")
            return load_dataset(
                "parquet",
                data_files=pattern,
                token=token,
                streaming=streaming,
            )
        except Exception as exc:
            last_exc = exc
            print(f"    [PATTERN-FAIL] {str(exc)[:120]}")
    raise RuntimeError(f"All Parquet patterns failed for {repo}: {last_exc}")


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
    max_docs: int | None = None,
) -> tuple[int, int, dict[str, int]]:
    """
    Count documents, characters, and tokens for one split.

    Tokenizer calls are batched. `return_length=True` returns the number
    of tokens per example without materialising full input_ids when the
    fast tokenizer supports it.
    """
    docs = 0
    chars = 0
    tokens = {name: 0 for name in tokenizers}
    started = time.time()

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
    total = getattr(dataset, "num_rows", None)
    pbar = tqdm(dataset, total=total, desc=desc, unit="doc", leave=False)
    for example in pbar:
        text = pick_text(example)
        if not text:
            continue
        batch.append(text)
        if len(batch) >= batch_size:
            flush(batch)
            batch = []
            pbar.set_postfix({
                "docs": f"{docs:,}",
                "tok": f"{sum(tokens.values()):,}",
            })
        if max_docs and docs >= max_docs:
            break
    flush(batch)

    elapsed = time.time() - started
    if docs and elapsed > 0:
        rate = docs / elapsed
        tok_rate = sum(tokens.values()) / elapsed
        pbar.write(
            f"  [done] {docs:,} docs in {elapsed:.0f}s "
            f"({rate:.0f} doc/s, {tok_rate:,.0f} tok/s)"
        )
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
    if args.tokenizers:
        names = {name.replace("/", "_").replace("-", "_"): name
                 for name in args.tokenizers}
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


def load_previous_rows(path: str | None) -> dict[tuple[str, str], dict]:
    """Read previously-counted rows from a CSV (for --resume)."""
    if not path or not os.path.exists(path):
        return {}
    out: dict[tuple[str, str], dict] = {}
    try:
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                key = (row.get("dataset", ""), row.get("split", ""))
                if key[0] and key[1]:
                    out[key] = row
    except Exception as exc:
        print(f"[RESUME] failed to read {path}: {exc}")
    return out


def row_from_previous(row: dict, tokenizer_names: list[str]) -> dict | None:
    """
    Convert a previous CSV row back into an internal row dict.

    Returns None if a required tokenizer column is missing, so the split
    is re-counted instead of receiving a misleading zero.
    """
    for k in tokenizer_names:
        if f"tokens_{k}" not in row or row[f"tokens_{k}"] in (None, ""):
            return None
    out = dict(row)
    out["docs"] = int(out.get("docs") or 0)
    out["chars"] = int(out.get("chars") or 0)
    for k in tokenizer_names:
        out[f"tokens_{k}"] = int(out.get(f"tokens_{k}") or 0)
        out[f"tokens_{k}_per_doc"] = float(out.get(f"tokens_{k}_per_doc") or 0.0)
    return out


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
    ap.add_argument("--streaming", action="store_true",
                    help="Stream rows instead of caching (recommended for 1M+ docs)")
    ap.add_argument("--max-docs", type=int, default=None,
                    help="Limit documents per split (for quick tests)")
    ap.add_argument("--resume", default=None,
                    help="CSV with previously-counted rows; those splits are skipped")
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
    if args.max_docs:
        print(f"[MODE] max-docs={args.max_docs} per split.\n")

    previous = load_previous_rows(args.resume)
    if previous:
        print(f"[RESUME] loaded {len(previous)} rows from {args.resume}\n")

    requested_groups = ["custom"] if args.datasets else list(args.groups)

    rows: list[dict] = []
    tokenizer_names = list(tokenizers.keys())
    empty_tot = {"docs": 0, "chars": 0, "datasets": 0,
                 **{k: 0 for k in tokenizer_names}}
    group_totals: dict[str, dict] = defaultdict(lambda: dict(empty_tot))

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
                **{f"tokens_{k}": 0 for k in tokenizer_names},
                **{f"tokens_{k}_per_doc": 0.0 for k in tokenizer_names},
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

            key = (repo, split_name)
            if key in previous:
                prev = row_from_previous(previous[key], tokenizer_names)
                if prev is not None:
                    rows.append(prev)
                    group_totals[group_name]["docs"] += prev["docs"]
                    group_totals[group_name]["chars"] += prev["chars"]
                    group_totals[group_name]["datasets"] += 1
                    for k in tokenizer_names:
                        group_totals[group_name][k] += prev[f"tokens_{k}"]
                    print(f"  split={split_name:<11} [RESUME] docs={prev['docs']:,}")
                    continue
                print(f"  split={split_name:<11} [RESUME] tokenizer columns missing, re-counting")

            try:
                docs, chars, tokens = count_split(
                    split_ds, tokenizers,
                    f"{repo} [{split_name}]",
                    max_docs=args.max_docs,
                )
            except Exception as exc:
                print(f"  [ERROR] split={split_name}: {exc}")
                rows.append({
                    "group": group_name, "lang": lang, "type": dtype,
                    "dataset": repo, "split": split_name,
                    "docs": 0, "chars": 0,
                    **{f"tokens_{k}": 0 for k in tokenizer_names},
                    **{f"tokens_{k}_per_doc": 0.0 for k in tokenizer_names},
                    "error": f"split error: {str(exc)[:180]}",
                })
                continue

            if docs == 0:
                print(f"  split={split_name:<11} docs=0 (empty)")
                continue

            if docs >= LARGE_DATASET_WARN:
                print(f"  [LARGE] {docs:,} docs — consider --streaming next time")

            per_doc = {k: tokens[k] / docs for k in tokenizer_names}
            row = {
                "group": group_name, "lang": lang, "type": dtype,
                "dataset": repo, "split": split_name,
                "docs": docs, "chars": chars,
                **{f"tokens_{k}": tokens[k] for k in tokenizer_names},
                **{f"tokens_{k}_per_doc": round(per_doc[k], 1) for k in tokenizer_names},
                "error": "",
            }
            rows.append(row)

            print(f"  split={split_name:<11} docs={docs:>10,}  chars={chars:>13,}")
            for key_tok in tokenizer_names:
                print(f"    {key_tok:<12} {tokens[key_tok]:>14,}  ({per_doc[key_tok]:>6.0f}/doc)")

            total = group_totals[group_name]
            total["docs"] += docs
            total["chars"] += chars
            for k in tokenizer_names:
                total[k] += tokens[k]
            total["datasets"] += 1

    # ---- Summary ----
    print(f"\n\n{'=' * 78}")
    print("SUMMARY BY GROUP")
    print(f"{'=' * 78}")

    header = f"{'Group':<12} {'Datasets':>10} {'Documents':>14} {'Characters':>16}"
    for k in tokenizer_names:
        header += f" {k + ' tokens':>16}"
    print(header)
    print("-" * (70 + 17 * len(tokenizer_names)))

    empty = dict(empty_tot)
    grand = {"docs": 0, "chars": 0, "datasets": 0,
             **{k: 0 for k in tokenizer_names}}

    for name in requested_groups:
        data = group_totals.get(name) or empty
        line = (f"{name:<12} {data['datasets']:>10} {data['docs']:>14,} "
                f"{data['chars']:>16,}")
        for k in tokenizer_names:
            line += f" {data[k]:>16,}"
        print(line)
        for field in ("docs", "chars", "datasets"):
            grand[field] += data[field]
        for k in tokenizer_names:
            grand[k] += data[k]

    print("-" * (70 + 17 * len(tokenizer_names)))
    line = (f"{'TOTAL':<12} {'':>10} {grand['docs']:>14,} "
            f"{grand['chars']:>16,}")
    for k in tokenizer_names:
        line += f" {grand[k]:>16,}"
    print(line)

    # ---- Per-row CSV ----
    fields = ["group", "lang", "type", "dataset", "split", "docs", "chars"]
    fields += [f"tokens_{k}" for k in tokenizer_names]
    fields += [f"tokens_{k}_per_doc" for k in tokenizer_names]
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
                        + [f"tokens_{k}" for k in tokenizer_names])
        for name in requested_groups:
            data = group_totals.get(name) or empty
            writer.writerow([name, data["datasets"], data["docs"], data["chars"]]
                            + [data[k] for k in tokenizer_names])
        writer.writerow(["TOTAL",
                         sum((group_totals.get(g) or empty)["datasets"]
                             for g in requested_groups),
                         grand["docs"], grand["chars"]]
                        + [grand[k] for k in tokenizer_names])
    print(f"[OK] Summary by group: {args.out_summary}")


if __name__ == "__main__":
    main()
    