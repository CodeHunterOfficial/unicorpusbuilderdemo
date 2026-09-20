# tools/list_sites.py
"""
List all sites defined in the project configuration files.

Reads ``config/universal.yaml`` (which pulls in every included file)
and prints the site inventory either as a table grouped by language,
or as JSON when ``--json`` is passed.

Usage
-----
    python tools/list_sites.py
    python tools/list_sites.py --lang tg
    python tools/list_sites.py --json
    python tools/list_sites.py --config config/universal.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Add project root to sys.path so that `config` is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.loader import load_modular_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default=None,
                    help="Restrict output to a single language code")
    ap.add_argument("--config", default="config/universal.yaml",
                    help="Path to the entry-point config file")
    ap.add_argument("--json", action="store_true",
                    help="Emit JSON instead of a formatted table")
    args = ap.parse_args()

    print(f"[DEBUG] Loading: {args.config}")
    config = load_modular_config(args.config)

    print(f"[DEBUG] Type: {type(config).__name__}")
    print(f"[DEBUG] Top-level keys: {list(config.keys())}")

    sites = config.get("sites", {})
    print(f"[DEBUG] sites: {type(sites).__name__}, "
          f"length: {len(sites) if hasattr(sites, '__len__') else '?'}")

    if not sites:
        # Help the user find where the sites actually live when the
        # entry-point config has been misconfigured.
        for k in config.keys():
            v = config.get(k)
            if isinstance(v, dict) and v and all(isinstance(x, dict) for x in v.values()):
                print(f"[DEBUG] Possible alternative: config['{k}'] — {len(v)} entries")
        print("\n[ERROR] 'sites' is empty. Check universal.yaml and its includes.")
        return

    # --- Group sites by language ---
    by_lang = {}
    for key, cfg in sites.items():
        if not isinstance(cfg, dict):
            continue
        lang = cfg.get("default_language", "?")
        if args.lang and lang != args.lang:
            continue
        by_lang.setdefault(lang, []).append((key, cfg))

    if args.json:
        result = {}
        for lang, items in sorted(by_lang.items()):
            result[lang] = [
                {
                    "site_key": k,
                    "domain": (c.get("match") or [None])[0],
                    "start_url": c.get("start_url"),
                    "author_strategy": c.get("author_strategy", []),
                }
                for k, c in items
            ]
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # --- Tabular output ---
    total = 0
    for lang in sorted(by_lang.keys()):
        items = by_lang[lang]
        print(f"\n{'='*70}")
        print(f"Language: {lang}  ({len(items)} sites)")
        print(f"{'='*70}")
        print(f"{'Site key':<25} {'Domain':<25} {'Strategies':>10}")
        print("-" * 70)
        for key, cfg in sorted(items):
            match = cfg.get("match", [])
            domain = match[0] if match else "?"
            n_strat = len(cfg.get("author_strategy", []))
            print(f"{key:<25} {domain:<25} {n_strat:>10}")
            total += 1

    print(f"\nTotal sites: {total}")


if __name__ == "__main__":
    main()