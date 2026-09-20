# tools/inspect_binary_labels.py
"""
Inspect the structure of the four binary news datasets.

For each dataset the script prints:
  * distribution of the ``label`` field
  * distribution of ``label_text`` per label class
  * distribution of ``category`` within each label class

Datasets covered
----------------
    TatarNLPWorld/tatar-news-analysis-binary     (tt)
    TajikNLPWorld/tajik-news-binary              (tg)
    BashkirNLPWorld/bashkir-news-binary          (ba)
    OssetianNLPWorld/ossetian-news-binary        (os)

Loading strategy
----------------
The script first tries the standard ``load_dataset`` path. If that
fails (some repositories carry a ``dataset_info.json`` that disagrees
with the underlying Parquet files), it falls back to downloading the
Parquet shards directly and rebuilding a ``Dataset`` from them.

Usage
-----
    python tools/inspect_binary_labels.py
    python tools/inspect_binary_labels.py --top 20
    python tools/inspect_binary_labels.py --dataset TajikNLPWorld/tajik-news-binary

Installation
------------
    pip install datasets huggingface_hub pandas pyarrow
"""
from __future__ import annotations

import argparse
from collections import Counter

from datasets import load_dataset, Dataset


DATASETS = [
    ("tatar",    "tt", "TatarNLPWorld/tatar-news-analysis-binary"),
    ("tajik",    "tg", "TajikNLPWorld/tajik-news-binary"),
    ("bashkir",  "ba", "BashkirNLPWorld/bashkir-news-binary"),
    ("ossetian", "os", "OssetianNLPWorld/ossetian-news-binary"),
]


# ============================================================
# Loading
# ============================================================

def try_load(repo_id: str):
    """
    Load a dataset, falling back to the Parquet shards when the
    standard loader fails.

    Returns
    -------
    (Dataset or None, method_label : str)
    """
    try:
        ds = load_dataset(repo_id, split="train")
        return ds, "datasets"
    except Exception as e:
        print(f"  [WARN] load_dataset: {type(e).__name__}")
        print("  [RETRY] via parquet loader...")

    try:
        from huggingface_hub import list_repo_files, hf_hub_download
        import pandas as pd

        files = list_repo_files(repo_id, repo_type="dataset")
        parquet_files = [f for f in files if f.endswith(".parquet")]
        if not parquet_files:
            return None, "no parquet"

        dfs = []
        for pf in parquet_files:
            path = hf_hub_download(repo_id=repo_id, filename=pf, repo_type="dataset")
            dfs.append(pd.read_parquet(path))
        df = pd.concat(dfs, ignore_index=True)
        ds = Dataset.from_pandas(df, preserve_index=False)
        return ds, "parquet"
    except Exception as e:
        print(f"  [ERROR] parquet: {e}")
        return None, "error"


# ============================================================
# Inspection
# ============================================================

def inspect_one(repo_id: str, lang: str, top_n: int, summary: dict):
    """Print the label / label_text / category distributions for one dataset."""
    print(f"\n{'=' * 72}")
    print(f"[DATASET] {repo_id}  ({lang})")
    print(f"{'=' * 72}")

    ds, method = try_load(repo_id)
    if ds is None:
        print("  [FAIL] could not load the dataset")
        summary[repo_id] = {"error": "load failed"}
        return

    print(f"  [OK] loaded via {method}, rows: {len(ds):,}")
    print(f"  Columns: {ds.column_names}")

    label_field = "label"
    label_text_field = "label_text" if "label_text" in ds.column_names else None
    category_field = "category" if "category" in ds.column_names else None

    # Overall label distribution
    label_counter = Counter(ds[label_field])
    print("\n[LABEL DISTRIBUTION]")
    for lbl in sorted(label_counter.keys(), key=str):
        print(f"  label={lbl!r}: {label_counter[lbl]:,}")

    # Per-label breakdown: label_text and category
    for target_label in sorted(label_counter.keys(), key=str):
        print(f"\n{'─' * 72}")
        print(f"[CLASS label={target_label!r}]  ({label_counter[target_label]:,} documents)")
        print(f"{'─' * 72}")

        rows = [r for r in ds if r[label_field] == target_label]

        if label_text_field:
            lt_counter = Counter(r[label_text_field] for r in rows)
            print("  label_text:")
            for lt, cnt in lt_counter.most_common(top_n):
                print(f"    {lt!r}: {cnt:,}")

        if category_field:
            cat_counter = Counter(
                r[category_field] for r in rows
                if r[category_field] is not None and str(r[category_field]).strip()
            )
            total_with_cat = sum(cat_counter.values())
            print(f"\n  Categories ({len(cat_counter)} unique, "
                  f"{total_with_cat:,} documents with a non-empty category):")
            for cat, cnt in cat_counter.most_common(top_n):
                pct = cnt / len(rows) * 100
                print(f"    {cat!r}: {cnt:,}  ({pct:.1f}%)")
            if len(cat_counter) > top_n:
                print(f"    ... and {len(cat_counter) - top_n} more categories")
        else:
            print("\n  [no 'category' field]")

    summary[repo_id] = {
        "lang": lang,
        "n_docs": len(ds),
        "dist": dict(label_counter),
    }


# ============================================================
# Entry point
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=15,
                    help="Number of categories to show per class (default: 15)")
    ap.add_argument("--dataset", default=None,
                    help="Inspect a single dataset by repo_id")
    args = ap.parse_args()

    summary: dict = {}

    if args.dataset:
        inspect_one(args.dataset, "?", args.top, summary)
    else:
        for _, lang, repo in DATASETS:
            try:
                inspect_one(repo, lang, args.top, summary)
            except Exception as e:
                print(f"\n[FATAL] {repo}: {e}")


if __name__ == "__main__":
    main()