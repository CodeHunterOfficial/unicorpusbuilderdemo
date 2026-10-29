#!/usr/bin/env python
# tools/hf_count.py
"""Measure dataset sizes for BashkirNLPWorld, TatarNLPWorld,
OssetianNLPWorld, and TajikNLPWorld.

Default behavior:
  * large datasets (1M+ docs) use streaming automatically
  * CSV is saved atomically after every split
  * existing hf_count.csv is auto-resumed (partial tokenizers are completed)

Usage:
    python tools/hf_count.py
    python tools/hf_count.py --groups tajik --streaming --no-gemma3
    python tools/hf_count.py --groups tatar --gemma3-model google/gemma-3-270m
    python tools/hf_count.py --no-resume
    python tools/hf_count.py --no-streaming            # force non-streaming
    python tools/hf_count.py --max-docs 100 --streaming
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Iterable

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


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
TOKENIZE_BATCH = 512
LARGE_DATASET_WARN = 500_000

DEFAULT_TOKENIZERS = {
    "xlm_r": "xlm-roberta-base",
    "gemma3": "google/gemma-3-270m",
}

# Datasets that time out / OOM without streaming.
LARGE_DATASETS = {
    "TatarNLPWorld/tatar-web-corpus",
    "TatarNLPWorld/tatar-web-corpus-v3",
}


def get_token() -> str | None:
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        v = os.environ.get(var)
        if v:
            return v.strip()
    return None


def should_stream(repo: str, args) -> bool:
    if args.no_streaming:
        return False
    if args.streaming:
        return True
    return repo in LARGE_DATASETS


def load_dataset_with_fallback(repo: str, token: str | None, streaming: bool):
    try:
        return load_dataset(
            repo, token=token, streaming=streaming,
            verification_mode="no_checks",
        )
    except Exception as exc:
        msg = str(exc).lower()
        if "column names don't match" in msg:
            print("  [SCHEMA-MISMATCH] falling back to parquet loader")
        else:
            print(f"  [WARN] standard loader: {type(exc).__name__}: {str(exc)[:160]}")
        print("  [RETRY] parquet loader...")

    patterns = [
        f"hf://datasets/{repo}/data/*.parquet",
        f"hf://datasets/{repo}/*.parquet",
        f"hf://datasets/{repo}/data/**/*.parquet",
        f"hf://datasets/{repo}/**/*.parquet",
    ]
    last_exc: Exception | None = None
    for pat in patterns:
        try:
            print(f"    trying: {pat}")
            return load_dataset("parquet", data_files=pat, token=token, streaming=streaming)
        except Exception as exc:
            last_exc = exc
            print(f"    [PATTERN-FAIL] {str(exc)[:120]}")
    raise RuntimeError(f"All parquet patterns failed for {repo}: {last_exc}")


def pick_text(example: dict) -> str:
    for field in TEXT_FIELDS:
        v = example.get(field)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def count_split(
    dataset,
    tokenizers: dict[str, AutoTokenizer],
    desc: str,
    batch_size: int = TOKENIZE_BATCH,
    max_docs: int | None = None,
) -> tuple[int, int, dict[str, int]]:
    docs = 0
    chars = 0
    tokens = {name: 0 for name in tokenizers}
    started = time.time()

    def flush(batch: list[str]) -> None:
        nonlocal docs, chars
        if not batch:
            return
        docs += len(batch)
        chars += sum(len(t) for t in batch)
        for name, tok in tokenizers.items():
            enc = tok(batch, add_special_tokens=False, return_length=True)
            tokens[name] += sum(enc["length"])

    batch: list[str] = []
    total = getattr(dataset, "num_rows", None)
    pbar = tqdm(
        dataset, total=total, desc=desc, unit="doc",
        leave=False, mininterval=1.0,
    )
    try:
        for example in pbar:
            text = pick_text(example)
            if not text:
                continue
            batch.append(text)

            if max_docs and (docs + len(batch)) >= max_docs:
                needed = max_docs - docs
                batch = batch[:needed]
                flush(batch)
                batch = []
                break

            if len(batch) >= batch_size:
                flush(batch)
                batch = []
                pbar.set_postfix({"docs": f"{docs:,}"})

        if batch:
            flush(batch)
    finally:
        pbar.close()

    elapsed = time.time() - started
    if docs:
        rate = docs / elapsed if elapsed > 0 else 0
        print(f"  [done] {docs:,} docs in {elapsed:.0f}s ({rate:.0f} doc/s)")

    return docs, chars, tokens


def iter_targets(args) -> Iterable[tuple[str, str, str, str]]:
    if args.datasets:
        for repo in args.datasets:
            yield ("custom", repo, "?", "?")
        return
    for group_name in args.groups:
        for repo, lang, dtype in GROUPS[group_name]:
            yield (group_name, repo, lang, dtype)


def build_tokenizers(args) -> dict[str, AutoTokenizer]:
    if args.tokenizers:
        names = {n.replace("/", "_").replace("-", "_"): n for n in args.tokenizers}
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
            tokenizers[key] = AutoTokenizer.from_pretrained(model_name, token=args.hf_token)
        except Exception as exc:
            print(f"    [SKIP] {type(exc).__name__}: {exc}")
            if key == "gemma3":
                print(f"    Gemma 3 is gated. Accept the license at https://huggingface.co/{model_name}")
    return tokenizers


def load_previous_rows(path: str | None) -> dict[tuple[str, str], dict]:
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


def existing_tokenizers(row: dict, tokenizer_names: list[str]) -> set[str]:
    present: set[str] = set()
    for k in tokenizer_names:
        v = row.get(f"tokens_{k}")
        if v not in (None, "", "0"):
            try:
                if int(v) > 0:
                    present.add(k)
            except (ValueError, TypeError):
                pass
    return present


def atomic_write_csv(rows: list[dict], fields: list[str], path: str) -> None:
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    except OSError:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        try:
            os.remove(tmp)
        except OSError:
            pass


def write_summary(
    group_totals: dict[str, dict],
    requested_groups: list[str],
    tokenizer_names: list[str],
    path: str,
) -> None:
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["group", "datasets", "docs", "chars"]
                       + [f"tokens_{k}" for k in tokenizer_names])
            grand = {"datasets": 0, "docs": 0, "chars": 0,
                     **{k: 0 for k in tokenizer_names}}
            for name in requested_groups:
                d = group_totals.get(name) or {}
                w.writerow([name,
                            d.get("datasets", 0), d.get("docs", 0), d.get("chars", 0)]
                           + [d.get(k, 0) for k in tokenizer_names])
                for field in ("datasets", "docs", "chars"):
                    grand[field] += d.get(field, 0)
                for k in tokenizer_names:
                    grand[k] += d.get(k, 0)
            w.writerow(["TOTAL", grand["datasets"], grand["docs"], grand["chars"]]
                       + [grand[k] for k in tokenizer_names])
        os.replace(tmp, path)
    except OSError:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Count docs/chars/tokens per dataset split.")
    ap.add_argument("--groups", nargs="*", choices=list(GROUPS.keys()),
                    default=list(GROUPS.keys()))
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--include-samples", action="store_true")
    ap.add_argument("--tokenizers", nargs="*", default=None)
    ap.add_argument("--gemma3-model", default=None)
    ap.add_argument("--no-gemma3", action="store_true")
    ap.add_argument("--out", default="hf_count.csv")
    ap.add_argument("--out-summary", default="hf_count_summary.csv")
    ap.add_argument("--streaming", action="store_true",
                    help="Force streaming for all datasets")
    ap.add_argument("--no-streaming", action="store_true",
                    help="Force disable streaming (large datasets may time out)")
    ap.add_argument("--no-resume", action="store_true",
                    help="Ignore existing --out file, start from scratch")
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--resume", default=None,
                    help="Explicit CSV to resume from (overrides auto-resume)")
    args = ap.parse_args()

    if args.streaming and args.no_streaming:
        print("[ERROR] --streaming and --no-streaming are mutually exclusive.")
        sys.exit(2)

    args.hf_token = get_token()
    if args.hf_token:
        print(f"[AUTH] HF_TOKEN: {args.hf_token[:7]}... (len={len(args.hf_token)})")
    else:
        print("[AUTH] HF_TOKEN not set; gated models will be skipped.")

    if args.no_resume:
        previous: dict = {}
    elif args.resume:
        previous = load_previous_rows(args.resume)
        if previous:
            print(f"[RESUME] {len(previous)} rows from {args.resume}")
    elif os.path.exists(args.out):
        previous = load_previous_rows(args.out)
        if previous:
            print(f"[AUTO-RESUME] {len(previous)} rows from {args.out}")
    else:
        previous = {}

    print("\n[LOAD] Tokenizers...")
    tokenizers = build_tokenizers(args)
    if not tokenizers:
        print("[ERROR] No tokenizer loaded. Aborting.")
        sys.exit(1)
    print()

    requested_groups = ["custom"] if args.datasets else list(args.groups)
    tokenizer_names = list(tokenizers.keys())

    empty = {"docs": 0, "chars": 0, "datasets": 0,
             **{k: 0 for k in tokenizer_names}}
    group_totals: dict[str, dict] = defaultdict(lambda: dict(empty))
    rows: list[dict] = []

    fields = ["group", "lang", "type", "dataset", "split", "docs", "chars"]
    fields += [f"tokens_{k}" for k in tokenizer_names]
    fields += [f"tokens_{k}_per_doc" for k in tokenizer_names]
    fields += ["error"]

    start_time = time.time()
    current_group = None

    try:
        for group_name, repo, lang, dtype in iter_targets(args):
            if group_name != current_group:
                current_group = group_name
                print(f"\n{'=' * 78}")
                print(f"GROUP: {group_name.upper()}")
                print(f"{'=' * 78}")

            use_streaming = should_stream(repo, args)
            stream_tag = "  [streaming]" if use_streaming else ""
            print(f"\n[DATASET] {repo}  ({lang}, {dtype}){stream_tag}")

            try:
                ds = load_dataset_with_fallback(repo, args.hf_token, use_streaming)
            except Exception as exc:
                print(f"  [ERROR] load failed: {exc}")
                rows.append({
                    "group": group_name, "lang": lang, "type": dtype,
                    "dataset": repo, "split": "", "docs": 0, "chars": 0,
                    **{f"tokens_{k}": 0 for k in tokenizer_names},
                    **{f"tokens_{k}_per_doc": 0.0 for k in tokenizer_names},
                    "error": f"load: {str(exc)[:200]}",
                })
                atomic_write_csv(rows, fields, args.out)
                continue

            for split_name, split_ds in ds.items():
                if args.splits and split_name not in args.splits:
                    print(f"  [SKIP] split={split_name}")
                    continue
                if not args.include_samples and split_name.startswith("sample"):
                    print(f"  [SKIP] split={split_name} (samples)")
                    continue

                key = (repo, split_name)

                # Resume: figure out which tokenizers are already done.
                prior_row = previous.get(key)
                present: set[str] = set()
                if prior_row:
                    present = existing_tokenizers(prior_row, tokenizer_names)
                    if present == set(tokenizer_names):
                        try:
                            row = {
                                "group": group_name, "lang": lang, "type": dtype,
                                "dataset": repo, "split": split_name,
                                "docs": int(prior_row.get("docs", 0) or 0),
                                "chars": int(prior_row.get("chars", 0) or 0),
                                **{f"tokens_{k}": int(prior_row.get(f"tokens_{k}", 0) or 0)
                                   for k in tokenizer_names},
                                **{f"tokens_{k}_per_doc": float(prior_row.get(f"tokens_{k}_per_doc", 0) or 0)
                                   for k in tokenizer_names},
                                "error": "",
                            }
                        except (ValueError, TypeError):
                            pass
                        else:
                            rows.append(row)
                            total = group_totals[group_name]
                            total["docs"] += row["docs"]
                            total["chars"] += row["chars"]
                            total["datasets"] += 1
                            for k in tokenizer_names:
                                total[k] += row[f"tokens_{k}"]
                            print(f"  split={split_name:<11} [RESUME] docs={row['docs']:,}")
                            atomic_write_csv(rows, fields, args.out)
                            continue

                # Partial or fresh counting
                to_count = {k: tokenizers[k]
                            for k in tokenizer_names if k not in present}
                if prior_row and present:
                    print(f"  split={split_name:<11} [PARTIAL] adding: {', '.join(sorted(to_count))}")

                try:
                    docs, chars, new_tokens = count_split(
                        split_ds, to_count,
                        f"{repo} [{split_name}]",
                        max_docs=args.max_docs,
                    )
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    print(f"  [ERROR] split={split_name}: {exc}")
                    rows.append({
                        "group": group_name, "lang": lang, "type": dtype,
                        "dataset": repo, "split": split_name,
                        "docs": 0, "chars": 0,
                        **{f"tokens_{k}": 0 for k in tokenizer_names},
                        **{f"tokens_{k}_per_doc": 0.0 for k in tokenizer_names},
                        "error": f"count: {str(exc)[:180]}",
                    })
                    atomic_write_csv(rows, fields, args.out)
                    continue

                if docs == 0:
                    print(f"  split={split_name:<11} docs=0 (empty)")
                    continue

                if docs >= LARGE_DATASET_WARN and not use_streaming:
                    print(f"  [LARGE] {docs:,} docs — use --streaming next time")

                merged_tokens = {}
                for k in tokenizer_names:
                    if k in new_tokens:
                        merged_tokens[k] = new_tokens[k]
                    else:
                        merged_tokens[k] = int((prior_row or {}).get(f"tokens_{k}", 0) or 0)

                per_doc = {k: merged_tokens[k] / docs for k in tokenizer_names}
                row = {
                    "group": group_name, "lang": lang, "type": dtype,
                    "dataset": repo, "split": split_name,
                    "docs": docs, "chars": chars,
                    **{f"tokens_{k}": merged_tokens[k] for k in tokenizer_names},
                    **{f"tokens_{k}_per_doc": round(per_doc[k], 1) for k in tokenizer_names},
                    "error": "",
                }
                rows.append(row)

                print(f"  split={split_name:<11} docs={docs:>10,}  chars={chars:>13,}")
                for k in tokenizer_names:
                    print(f"    {k:<12} {merged_tokens[k]:>14,}  ({per_doc[k]:>6.0f}/doc)")

                total = group_totals[group_name]
                total["docs"] += docs
                total["chars"] += chars
                for k in tokenizer_names:
                    total[k] += merged_tokens[k]
                total["datasets"] += 1

                atomic_write_csv(rows, fields, args.out)

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Saving partial results...")
    finally:
        if rows:
            atomic_write_csv(rows, fields, args.out)
            print(f"\n[OK] Per-row CSV:      {args.out}  (rows: {len(rows)})")
            write_summary(group_totals, requested_groups, tokenizer_names, args.out_summary)
            print(f"[OK] Summary by group: {args.out_summary}")

            # Console summary
            print(f"\n{'=' * 78}")
            print("SUMMARY BY GROUP")
            print(f"{'=' * 78}")
            header = f"{'Group':<12} {'Datasets':>10} {'Documents':>14} {'Characters':>16}"
            for k in tokenizer_names:
                header += f" {k + ' tokens':>16}"
            print(header)
            print("-" * (70 + 17 * len(tokenizer_names)))
            grand = {"docs": 0, "chars": 0, "datasets": 0,
                     **{k: 0 for k in tokenizer_names}}
            for name in requested_groups:
                d = group_totals.get(name) or empty
                line = (f"{name:<12} {d.get('datasets', 0):>10} "
                        f"{d.get('docs', 0):>14,} {d.get('chars', 0):>16,}")
                for k in tokenizer_names:
                    line += f" {d.get(k, 0):>16,}"
                print(line)
                for field in ("docs", "chars", "datasets"):
                    grand[field] += d.get(field, 0)
                for k in tokenizer_names:
                    grand[k] += d.get(k, 0)
            print("-" * (70 + 17 * len(tokenizer_names)))
            line = f"{'TOTAL':<12} {'':>10} {grand['docs']:>14,} {grand['chars']:>16,}"
            for k in tokenizer_names:
                line += f" {grand[k]:>16,}"
            print(line)

        elapsed = time.time() - start_time
        print(f"\n[DONE] {elapsed:.0f}s")


if __name__ == "__main__":
    main()
    