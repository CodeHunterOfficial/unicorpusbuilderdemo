# D:\Science\TajikPersianNLP\scraper_project\social\scripts\build_vk_dataset.py
"""
Build a cleaned, unified dataset from raw VK posts and comments
with strict language filtering.

Usage:
    python social/scripts/build_vk_dataset.py --lang tg
    python social/scripts/build_vk_dataset.py --lang os
    python social/scripts/build_vk_dataset.py --lang udm
"""

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from logger_setup import get_file_logger

logger = get_file_logger("build_vk_dataset", "logs/build_vk_dataset.log")

POSTS_DIR = PROJECT_ROOT / "output" / "social" / "posts"
COMMENTS_DIR = PROJECT_ROOT / "output" / "social" / "comments"
OUTPUT_FILE = PROJECT_ROOT / "output" / "social" / "vk_unified.jsonl"

# ============================================================
# EXTENSIBLE LANGUAGE FILTER RULES
# ============================================================
LANGUAGE_RULES = {
    "tg": {
        "required_letters": ['ғ', 'қ', 'ӣ', 'ӯ', 'ҳ', 'ҷ'],
        "forbidden_letters": ['ц', 'щ', 'ы', 'ь'],
    },
    "os": {
        "required_letters": ['æ', 'Æ'],
        "forbidden_letters": [],
    },
    "udm": {
        "required_letters": ['ӟ', 'ӝ', 'ӥ', 'ӧ', 'ӵ'],
        "forbidden_letters": [],
    },
    "ba": {
        "required_letters": ['ғ', 'ҙ', 'ҡ', 'ө', 'ҫ', 'ү', 'һ', 'ә'],
        "forbidden_letters": [],
    },
    "tt": {
        "required_letters": ['ә', 'җ', 'ң', 'ө', 'ү', 'һ'],
        "forbidden_letters": [],
    },
}

# ============================================================
# Emoji removal
# ============================================================
_emoji_pattern = re.compile(
    "[\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE
)

def remove_emoji(text: str) -> str:
    return _emoji_pattern.sub("", text).strip()


def load_jsonl(path: Path) -> list:
    items = []
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return items


def language_filter(text: str, lang_code: str) -> bool:
    """
    Return True if the text passes the language filter for the given code.
    If no rules are defined for the language, accept everything.
    """
    rules = LANGUAGE_RULES.get(lang_code)
    if not rules:
        return True

    if any(ch in text for ch in rules.get("forbidden_letters", [])):
        return False

    required = rules.get("required_letters", [])
    if required and not any(ch in text for ch in required):
        return False

    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build VK dataset with language filter")
    parser.add_argument("--lang", type=str, default=None,
                        help="Language code to filter (tg, os, udm, ba, tt, etc.)")
    args = parser.parse_args()

    target_lang = args.lang
    if target_lang:
        if target_lang not in LANGUAGE_RULES:
            logger.warning(f"No language rules defined for '{target_lang}', will keep all posts/comments")
        logger.info(f"Language filter active: {target_lang}")
    else:
        logger.info("No language filter – keeping all posts/comments")

    logger.info("Starting VK dataset builder")
    logger.info(f"Looking for posts in: {POSTS_DIR}")
    logger.info(f"Looking for comments in: {COMMENTS_DIR}")

    post_files = sorted(POSTS_DIR.glob("vk_*_posts.jsonl"))
    if not post_files:
        logger.error(f"No post files found in {POSTS_DIR}")
        print(f"[ERROR] No post files found in {POSTS_DIR}")
        return

    logger.info(f"Found {len(post_files)} post files")
    print(f"[FILES] Found {len(post_files)} domains")

    all_clean = []
    stats = {
        "total_posts": 0,
        "total_comments": 0,
        "filtered_posts_short": 0,
        "filtered_posts_lang": 0,
        "filtered_comments_short": 0,
        "filtered_comments_lang": 0,
    }

    for post_file in tqdm(post_files, desc="[PROCESS] Domains"):
        domain = post_file.stem.replace("_posts", "").replace("vk_", "")
        comment_file = COMMENTS_DIR / f"vk_{domain}_comments.jsonl"

        logger.info(f"Processing domain: {domain}")
        posts = load_jsonl(post_file)
        comments = load_jsonl(comment_file)

        stats["total_posts"] += len(posts)
        stats["total_comments"] += len(comments)
        print(f"  Domain {domain}: {len(posts)} posts, {len(comments)} comments")

        comments_by_post = defaultdict(list)
        for c in comments:
            pid = c.get("post_id")
            if pid is not None:
                comments_by_post[pid].append(c)

        for post in posts:
            content = post.get("content", "")
            clean_content = remove_emoji(content).strip()

            if len(clean_content) < 10:
                stats["filtered_posts_short"] += 1
                continue

            if target_lang and not language_filter(clean_content, target_lang):
                stats["filtered_posts_lang"] += 1
                continue

            post_comments = comments_by_post.get(post.get("post_id"), [])
            clean_comment_lines = []
            for c in post_comments:
                c_text = remove_emoji(c.get("content", "")).strip()
                if len(c_text) < 10:
                    stats["filtered_comments_short"] += 1
                    continue
                if target_lang and not language_filter(c_text, target_lang):
                    stats["filtered_comments_lang"] += 1
                    continue
                clean_comment_lines.append(c_text)

            combined_comments = "\n".join(clean_comment_lines)

            record = {
                "title": remove_emoji(post.get("title", "")).strip(),
                "content": clean_content,
                "category": domain,
                "comments": combined_comments,
                "date": post.get("date"),
                "url": post.get("url"),
                "language": post.get("language", "unknown"),
                "post_id": post.get("post_id"),
                "domain": domain,
                "likes": post.get("likes"),
                "reposts": post.get("reposts"),
                "views": post.get("views"),
            }
            all_clean.append(record)

    os.makedirs(OUTPUT_FILE.parent, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in all_clean:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(f"Dataset built: {len(all_clean)} records saved to {OUTPUT_FILE}")
    logger.info(f"Stats: {stats}")

    print(f"\n{'='*60}")
    print(f"[OK] Dataset built!")
    print(f"  Total posts loaded:         {stats['total_posts']}")
    print(f"    - removed (<10 chars):    {stats['filtered_posts_short']}")
    if target_lang:
        print(f"    - removed (not {target_lang}):  {stats['filtered_posts_lang']}")
    print(f"  Total comments loaded:      {stats['total_comments']}")
    print(f"    - removed (<10 chars):    {stats['filtered_comments_short']}")
    if target_lang:
        print(f"    - removed (not {target_lang}):  {stats['filtered_comments_lang']}")
    print(f"  Final records written:      {len(all_clean)}")
    print(f"  Saved to: {OUTPUT_FILE}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()