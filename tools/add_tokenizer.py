# tools/add_tokenizer.py
"""
Add one extra tokenizer column to an existing hf_count.csv.

Reads a CSV produced by tools/hf_count.py, and for every row with
docs > 0, loads the dataset from the Hugging Face cache and counts
tokens under one additional tokenizer. Columns already present in the
input CSV are preserved unchanged.

Default tokenizer
-----------------
    google-bert/bert-base-multilingual-cased   (mBERT)

mBERT uses WordPiece with ~120k vocabulary, whereas XLM-R uses
SentencePiece with ~250k. The same text yields noticeably different
token counts, so both are reported side by side.

Gemma 3 is also supported via ``--tokenizer google/gemma-3-270m``,
matching the default tokenizer set in hf_count.py.

Retry list
----------
A small set of datasets failed in an earlier run and are re-counted
from scratch here. Their rows are appended to the output CSV.

Usage
-----
    python tools/add_tokenizer.py --input hf_count.csv --out hf_count_v2.csv
    python tools/add_tokenizer.py --tokenizer google/gemma-3-270m
    python tools/add_tokenizer.py --tokenizer google-bert/bert-base-multilingual-cased --streaming
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
from collections import defaultdict
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

DEFAULT_TOKENIZER = "google-bert/bert-base-multilingual-cased"

# Text field priority; matches hf_count.py so token counts are comparable.
TEXT_FIELDS = ["content", "text", "title"]

# Datasets that failed in an earlier run.
# repo_id -> (group, lang, type, forced_split_or_None)
RETRY_DATASETS = {
    "OssetianNLPWorld/ossetian-web-corpus":      ("ossetian", "os", "web",        None),
    "OssetianNLPWorld/ossetian-news-cluster":    ("ossetian", "os", "cluster",    None),
    "OssetianNLPWorld/ossetian-news-multilabel": ("ossetian", "os", "multilabel", None),
    "OssetianNLPWorld/ossetian-news-binary":     ("ossetian", "os", "binary",     None),
    "OssetianNLPWorld/ossetian-news-multiclass": ("ossetian", "os", "multiclass", None),
    "TatarNLPWorld/tatar-news-cluster":          ("tatar",    "tt", "cluster",    "train"),
}


# ============================================================
# Helpers
# ============================================================

def get_token() -> str | None:
    """Return the Hugging Face token from the environment, if present."""
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


def key_for_model(model_name: str) -> str:
    """Derive a stable CSV column key from a tokenizer model name."""
    return model_name.replace("/", "_").replace("-", "_")


def count_tokens(
    dataset,
    tokenizer: AutoTokenizer,
    desc: str,
    batch_size: int = 256,
) -> int:
    """
    Count tokens under one tokenizer.

    Documents and character counts are already in the input CSV, so only
    the token total is returned. Tokenization is batched and uses
    return_length=True to avoid materializing input_ids.
    """
    total = 0
    batch: list[str] = []

    def flush() -> None:
        nonlocal total
        if not batch:
            return
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            return_length=True,
        )
        total += sum(encoded["length"])

    for example in tqdm(dataset, desc=desc, unit="doc", leave=False):
        text = pick_text(example)
        if not text:
            continue
        batch.append(text)
        if len(batch) >= batch_size:
            flush()
            batch = []
    flush()

    return total


def count_both(
    dataset,
    tokenizers: dict[str, AutoTokenizer],
    desc: str,
    batch_size: int = 256,
) -> tuple[int, int, dict[str, int]]:
    """
    Count documents, characters, and tokens for one split.

    Used only for retry datasets, where the input CSV has no prior row.
    """
    docs = 0
    chars = 0
    tokens = {name: 0 for name in tokenizers}
    batch: list[str] = []

    def flush() -> None:
        nonlocal docs, chars
        if not batch:
            return
        docs += len(batch)
        chars += sum(len(text) for text in batch)
        for name, tok in tokenizers.items():
            encoded = tok(batch, add_special_tokens=False, return_length=True)
            tokens[name] += sum(encoded["length"])

    for example in tqdm(dataset, desc=desc, unit="doc", leave=False):
        text = pick_text(example)
        if not text:
            continue
        batch.append(text)
        if len(batch) >= batch_size:
            flush()
            batch = []
    flush()

    return docs, chars, tokens


# ============================================================
# Entry point
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Add one tokenizer column to an existing hf_count.csv."
    )
    ap.add_argument("--input", default="hf_count.csv",
                    help="CSV produced by hf_count.py")
    ap.add_argument("--out", default="hf_count_v2.csv",
                    help="Output CSV with the extra tokenizer columns")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                    help=f"Tokenizer model name to add (default: {DEFAULT_TOKENIZER})")
    ap.add_argument("--streaming", action="store_true",
                    help="Use streaming mode (needed for some datasets)")
    ap.add_argument("--skip-retry", action="store_true",
                    help="Do not reprocess the hard-coded retry list")
    args = ap.parse_args()

    token = get_token()
    if token:
        print(f"[AUTH] HF_TOKEN: {token[:7]}...")
    else:
        print("[AUTH] HF_TOKEN not set; public datasets will be downloaded anonymously.")

    new_key = key_for_model(args.tokenizer)
    print(f"\n[LOAD] Adding tokenizer: {args.tokenizer}  (column key: {new_key})")
    new_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, token=token)
    print()

    # ---- Read the existing CSV ----
    with open(args.input, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    print(f"[READ] Rows loaded: {len(rows)}")

    # Detect the existing token keys so retry rows can be filled consistently.
    existing_token_keys: list[str] = []
    for name in fieldnames:
        if name.startswith("tokens_") and not name.endswith("_per_doc"):
            existing_token_keys.append(name[len("tokens_"):])
    print(f"[INFO] Existing token columns: {', '.join(existing_token_keys) or '(none)'}")

    # ---- Add the new tokenizer column to every row with docs > 0 ----
    out_rows: list[dict[str, Any]] = []
    for r in rows:
        docs = int(r.get("docs", 0) or 0)
        repo = r.get("dataset", "")
        split = r.get("split", "")

        new_tokens = 0
        if docs > 0:
            print(f"\n[TOKENS] {repo} [{split}]  (docs={docs:,})")
            try:
                ds = load_dataset_with_fallback(repo, token, args.streaming)
            except Exception as exc:
                print(f"  [ERROR] {exc}")
            else:
                if split in ds:
                    try:
                        new_tokens = count_tokens(
                            ds[split], new_tokenizer, f"{repo} [{split}]"
                        )
                    except Exception as exc:
                        print(f"  [ERROR] split={split}: {exc}")
                else:
                    print(f"  [SKIP] split={split} not present in the dataset")

        row = dict(r)
        row[f"tokens_{new_key}"] = new_tokens
        row[f"tokens_{new_key}_per_doc"] = (
            round(new_tokens / docs, 1) if docs else 0.0
        )
        # Drop the legacy column if present; it is no longer produced by hf_count.py.
        row.pop("tokens_mbart", None)
        row.pop("tokens_mbart_per_doc", None)
        out_rows.append(row)

    # ---- Retry the previously failed datasets ----
    if not args.skip_retry and RETRY_DATASETS:
        print(f"\n{'=' * 78}")
        print(f"[RETRY] Datasets to reprocess: {len(RETRY_DATASETS)}")
        print(f"{'=' * 78}")

        # Build tokenizer set: existing keys + the new one.
        retry_tokenizers: dict[str, AutoTokenizer] = {}
        for key in existing_token_keys:
            model_name = key.replace("_", "/", 1) if key.startswith("xlm") else key
            # Best-effort mapping; unknown keys are skipped.
            candidates = {
                "xlm_r": "xlm-roberta-base",
                "gemma3": "google/gemma-3-270m",
                "mbert": "google-bert/bert-base-multilingual-cased",
            }
            if key in candidates:
                try:
                    retry_tokenizers[key] = AutoTokenizer.from_pretrained(
                        candidates[key], token=token
                    )
                except Exception as exc:
                    print(f"  [WARN] Could not load {candidates[key]}: {exc}")
        retry_tokenizers[new_key] = new_tokenizer

        for repo, (group, lang, dtype, forced_split) in RETRY_DATASETS.items():
            print(f"\n[DATASET] {repo}  ({lang}, {dtype})")
            try:
                ds = load_dataset_with_fallback(repo, token, args.streaming)
            except Exception as exc:
                print(f"  [ERROR] {exc}")
                continue

            for split_name, split_ds in ds.items():
                if forced_split and split_name != forced_split:
                    continue
                if not forced_split and split_name == "sample":
                    continue

                try:
                    docs, chars, token_counts = count_both(
                        split_ds,
                        retry_tokenizers,
                        f"{repo} [{split_name}]",
                    )
                except Exception as exc:
                    print(f"  [ERROR] split={split_name}: {exc}")
                    continue

                if docs == 0:
                    print(f"  split={split_name:<11} docs=0 (empty)")
                    continue

                print(f"  split={split_name:<11} docs={docs:>10,}  chars={chars:>13,}")
                for key in retry_tokenizers:
                    print(f"    {key:<12} {token_counts[key]:>14,}")

                row: dict[str, Any] = {
                    "group": group,
                    "lang": lang,
                    "type": dtype,
                    "dataset": repo,
                    "split": split_name,
                    "docs": docs,
                    "chars": chars,
                    "error": "",
                }
                for key, value in token_counts.items():
                    row[f"tokens_{key}"] = value
                    row[f"tokens_{key}_per_doc"] = round(value / docs, 1)
                out_rows.append(row)

    # ---- Write the new CSV ----
    base_fields = ["group", "lang", "type", "dataset", "split", "docs", "chars"]
    for key in existing_token_keys:
        if key != new_key:
            base_fields.append(f"tokens_{key}")
            base_fields.append(f"tokens_{key}_per_doc")
    base_fields.append(f"tokens_{new_key}")
    base_fields.append(f"tokens_{new_key}_per_doc")
    base_fields.append("error")

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base_fields, extrasaction="ignore")
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

    print(f"\n[OK] New CSV: {args.out}  (rows: {len(out_rows)})")

    # ---- Summary by group ----
    print(f"\n{'=' * 78}")
    print(f"SUMMARY BY GROUP (after adding {new_key})")
    print(f"{'=' * 78}")
    header = f"{'Group':<12} {'Datasets':>10} {'Documents':>14}"
    for key in existing_token_keys + [new_key]:
        if key == "mbart":
            continue
        header += f" {key:>16}"
    print(header)
    print("-" * (36 + 17 * (len(existing_token_keys) + 1)))

    totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"datasets": 0, "docs": 0, **{k: 0 for k in existing_token_keys + [new_key]}}
    )
    for row in out_rows:
        docs = int(row.get("docs", 0) or 0)
        if docs == 0:
            continue
        group = row.get("group", "?")
        totals[group]["datasets"] += 1
        totals[group]["docs"] += docs
        for key in existing_token_keys + [new_key]:
            if key == "mbart":
                continue
            totals[group][key] += int(row.get(f"tokens_{key}", 0) or 0)

    grand_docs = 0
    grand_tokens: dict[str, int] = {k: 0 for k in existing_token_keys + [new_key] if k != "mbart"}
    for group in ("bashkir", "tatar", "ossetian", "tajik"):
        data = totals.get(group)
        if not data:
            continue
        line = f"{group:<12} {data['datasets']:>10} {data['docs']:>14,}"
        for key in existing_token_keys + [new_key]:
            if key == "mbart":
                continue
            line += f" {data[key]:>16,}"
        print(line)
        grand_docs += data["docs"]
        for key in existing_token_keys + [new_key]:
            if key == "mbart":
                continue
            grand_tokens[key] += data[key]

    print("-" * (36 + 17 * (len(existing_token_keys) + 1)))
    line = f"{'TOTAL':<12} {'':>10} {grand_docs:>14,}"
    for key in existing_token_keys + [new_key]:
        if key == "mbart":
            continue
        line += f" {grand_tokens[key]:>16,}"
    print(line)


if __name__ == "__main__":
    main()
    