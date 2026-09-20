# tools/analyze_configs.py
"""
Compute statistics over the project's YAML configuration files.

Scans the configuration directories listed in ``CONFIG_DIRS`` and
reports, per file, the size in bytes and the number of lines. A summary
line prints the total number of files, total size, average size, and
total line count.

Usage
-----
    python tools/analyze_configs.py
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml


# Directories scanned recursively for *.yaml files.
CONFIG_DIRS = ["config", "social/config", "documents", "wiki"]


def analyze_yaml(path: Path):
    """
    Return size and line-count statistics for a single YAML file.

    Returns None when the file cannot be read or parsed as YAML.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        data = yaml.safe_load(text) or {}
    except Exception:
        return None
    return {
        "path": str(path),
        "bytes": len(text.encode("utf-8")),
        "lines": text.count("\n") + 1,
        "top_keys": len(data) if isinstance(data, dict) else 0,
    }


def main():
    results = []
    for d in CONFIG_DIRS:
        if not os.path.isdir(d):
            continue
        for p in Path(d).rglob("*.yaml"):
            r = analyze_yaml(p)
            if r:
                results.append(r)

    if not results:
        print("No YAML files found in:", ", ".join(CONFIG_DIRS))
        return

    total_bytes = sum(r["bytes"] for r in results)
    total_lines = sum(r["lines"] for r in results)

    print(f"Config files:      {len(results)}")
    print(f"Total size:        {total_bytes} bytes ({total_bytes/1024:.1f} KB)")
    print(f"Average size:      {total_bytes/len(results):.0f} bytes")
    print(f"Total lines:       {total_lines}")
    print()
    print(f"{'File':<60} {'Bytes':>8} {'Lines':>6}")
    for r in sorted(results, key=lambda x: -x["bytes"]):
        print(f"{r['path']:<60} {r['bytes']:>8} {r['lines']:>6}")


if __name__ == "__main__":
    main()