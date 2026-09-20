# tools/add_tokenizer.py
"""
Add a second tokenizer column to an existing hf_count.csv.

Reads the CSV produced by ``tools/hf_count.py``, and for every row with
``docs > 0`` loads the dataset from cache and counts tokens under a
second tokenizer. The existing xlm-r column is NOT recalculated.

Which tokenizer is added
------------------------
    google-bert/bert-base-multilingual-cased

mBERT differs from XLM-R in tokenization scheme and vocabulary size:

    XLM-R   SentencePiece, ~250k tokens
    mBERT   WordPiece,     ~120k tokens

This matters for corpus size estimates: the same text yields noticeably
different token counts under the two tokenizers.

Retry list
----------
A small set of datasets (mostly in OssetianNLPWorld, plus
``tatar-news-cluster``) failed in an earlier run and are loaded again
here, with both tokenizers counted from scratch. Their rows are appended
to the output CSV.

Usage
-----
    python tools/add_tokenizer.py --input hf_count.csv --out hf_count_v2.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

from transformers import AutoTokenizer
from tqdm import tqdm

from _common import get_token, load_ds


# Tokenizer to add, and the key used in the output column names.
NEW_TOKENIZER_NAME = "google-bert/bert-base-multilingual-cased"
NEW_TOKENIZER_KEY = "mbert"

# Text field priority; matches hf_count.py so that token counts are
# comparable between the two scripts.
TEXT_FIELDS = ["content", "text", "title"]

# Datasets that failed in an earlier run. Values are
# (language, dataset_type, forced_split_or_None).
# A ``forced_split`` of None means "process every non-sample split".
RETRY_DATASETS = {
    "OssetianNLPWorld/ossetian-web-corpus":         ("os", "web",        None),
    "OssetianNLPWorld/ossetian-news-cluster":       ("os", "cluster",    None),
    "OssetianNLPWorld/ossetian-news-multilabel":    ("os", "multilabel", None),
    "OssetianNLPWorld/ossetian-news-binary":        ("os", "binary",     None),
    "OssetianNLPWorld/ossetian-news-multiclass":    ("os", "multiclass", None),
    "TatarNLPWorld/tatar-news-cluster":             ("tt", "cluster",    "train"),
}


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


def count_only_new_tokenizer(dataset, tokenizer, desc: str) -> int:
    """
    Count tokens under the new tokenizer only.

    Documents and character counts are already present in the input CSV,
    so this function returns just the token total.
    """
    tokens = 0
    for ex in tqdm(dataset, desc=desc, unit="doc", leave=False):
        txt = pick_text(ex)
        if not txt:
            continue
        tokens += len(tokenizer.encode(txt, add_special_tokens=False))
    return tokens


# ============================================================
# Entry point
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="hf_count.csv",
                    help="Existing CSV produced by hf_count.py")
    ap.add_argument("--out", default="hf_count_v2.csv",
                    help="New CSV with the extra tokenizer columns")
    ap.add_argument("--streaming", action="store_true",
                    help="Streaming mode (needed for some datasets)")
    args = ap.parse_args()

    token = get_token()
    if token:
        print(f"[AUTH] HF_TOKEN: {token[:7]}...")
    else:
        print("[AUTH] HF_TOKEN not set; public datasets will be downloaded anonymously.")

    print(f"\n[LOAD] New tokenizer: {NEW_TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(NEW_TOKENIZER_NAME)
    print()

    # --- Read the existing CSV ---
    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    print(f"[READ] Rows loaded: {len(rows)}")

    # --- Count the new tokenizer for every row with docs > 0 ---
    out_rows = []
    for r in rows:
        docs = int(r.get("docs", 0) or 0)
        repo = r.get("dataset", "")
        split = r.get("split", "")

        # Rows with docs == 0 are copied through unchanged.
        if docs == 0:
            new_tokens = 0
        else:
            print(f"\n[TOKENS] {repo} [{split}] (docs={docs:,})")
            try:
                ds = load_ds(repo, token, args.streaming)
            except Exception as e:
                print(f"  [ERROR] {e}")
                new_tokens = 0
            else:
                if split in ds:
                    try:
                        new_tokens = count_only_new_tokenizer(
                            ds[split], tokenizer, f"{repo} [{split}]"
                        )
                    except Exception as e:
                        print(f"  [ERROR] split={split}: {e}")
                        new_tokens = 0
                else:
                    print(f"  [SKIP] split={split} not present in the dataset")
                    new_tokens = 0

        nr = dict(r)
        nr[f"tokens_{NEW_TOKENIZER_KEY}"] = new_tokens
        nr[f"tokens_{NEW_TOKENIZER_KEY}_per_doc"] = (
            round(new_tokens / docs, 1) if docs else 0.0
        )
        out_rows.append(nr)

    # --- Retry the previously failed datasets ---
    print(f"\n{'='*78}")
    print(f"[RETRY] Datasets to reprocess: {len(RETRY_DATASETS)}")
    print(f"{'='*78}")

    for repo, (lang, dtype, forced_split) in RETRY_DATASETS.items():
        print(f"\n[DATASET] {repo}  ({lang}, {dtype})")
        try:
            ds = load_ds(repo, token, args.streaming)
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

        for split_name, split_ds in ds.items():
            if forced_split and split_name != forced_split:
                continue
            if not forced_split and split_name == "sample":
                continue

            try:
                # Count both tokenizers from scratch for these rows.
                xlm_tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
                mbert_tok = tokenizer

                docs = 0
                chars = 0
                x_tokens = 0
                m_tokens = 0

                for ex in tqdm(split_ds, desc=f"{repo} [{split_name}]",
                               unit="doc", leave=False):
                    txt = pick_text(ex)
                    if not txt:
                        continue
                    docs += 1
                    chars += len(txt)
                    x_tokens += len(xlm_tok.encode(txt, add_special_tokens=False))
                    m_tokens += len(mbert_tok.encode(txt, add_special_tokens=False))

                if docs == 0:
                    print(f"  split={split_name:<11} docs=0 (empty)")
                    continue

                print(f"  split={split_name:<11} docs={docs:>10,} "
                      f"xlm-r={x_tokens:>14,}  mbert={m_tokens:>14,}")

                out_rows.append({
                    "group": "ossetian" if lang == "os" else "tatar",
                    "lang": lang, "type": dtype,
                    "dataset": repo, "split": split_name,
                    "docs": docs, "chars": chars,
                    "tokens_xlm_r": x_tokens,
                    "tokens_mbart": 0,  # legacy column, kept at 0
                    "tokens_xlm_r_per_doc": round(x_tokens / docs, 1),
                    "tokens_mbart_per_doc": 0.0,
                    f"tokens_{NEW_TOKENIZER_KEY}": m_tokens,
                    f"tokens_{NEW_TOKENIZER_KEY}_per_doc": round(m_tokens / docs, 1),
                    "error": "",
                })
            except Exception as e:
                print(f"  [ERROR] split={split_name}: {e}")
                continue

    # --- Write the new CSV ---
    base_fields = ["group", "lang", "type", "dataset", "split",
                   "docs", "chars",
                   "tokens_xlm_r", f"tokens_{NEW_TOKENIZER_KEY}",
                   "tokens_xlm_r_per_doc", f"tokens_{NEW_TOKENIZER_KEY}_per_doc",
                   "error"]

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=base_fields, extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    print(f"\n[OK] New CSV: {args.out}  (rows: {len(out_rows)})")

    # --- Summary by group ---
    print(f"\n{'='*78}")
    print(f"SUMMARY BY GROUP (after adding {NEW_TOKENIZER_KEY})")
    print(f"{'='*78}")
    print(f"{'Group':<10} {'Datasets':>10} {'Documents':>14} "
          f"{'xlm-r':>16} {NEW_TOKENIZER_KEY:>16}")
    print("-" * 76)

    totals = defaultdict(lambda: {"datasets": 0, "docs": 0,
                                  "xlm_r": 0, NEW_TOKENIZER_KEY: 0})
    for r in out_rows:
        docs = int(r.get("docs", 0) or 0)
        if docs == 0:
            continue
        g = r.get("group", "?")
        totals[g]["datasets"] += 1
        totals[g]["docs"] += docs
        totals[g]["xlm_r"] += int(r.get("tokens_xlm_r", 0) or 0)
        totals[g][NEW_TOKENIZER_KEY] += int(r.get(f"tokens_{NEW_TOKENIZER_KEY}", 0) or 0)

    grand_docs = grand_x = grand_m = 0
    for g in ("bashkir", "tatar", "ossetian", "tajik"):
        if g not in totals:
            continue
        t = totals[g]
        print(f"{g:<10} {t['datasets']:>10} {t['docs']:>14,} "
              f"{t['xlm_r']:>16,} {t[NEW_TOKENIZER_KEY]:>16,}")
        grand_docs += t["docs"]
        grand_x += t["xlm_r"]
        grand_m += t[NEW_TOKENIZER_KEY]

    print("-" * 76)
    print(f"{'TOTAL':<10} {'':>10} {grand_docs:>14,} "
          f"{grand_x:>16,} {grand_m:>16,}")


if __name__ == "__main__":
    main()