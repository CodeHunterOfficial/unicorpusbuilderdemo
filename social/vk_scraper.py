# social/vk_scraper.py
"""
VK scraper with logging support.
Collects posts and comments from VK public pages.

Usage:
    python social/vk_scraper.py --domains club1135692 irta_tv --lang ba --max-posts 5
    python social/vk_scraper.py --lang tt --max-posts 10
    python social/vk_scraper.py --max-posts 100
"""

import json
import os
import sys
import time
import hashlib
import argparse
import requests
from datetime import datetime
from typing import Optional, List, Dict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.loader import load_social_config
from logger_setup import get_file_logger

logger = get_file_logger("vk_scraper", "logs/vk_scraper.log")

VK_VERSION = "5.199"
OUTPUT_DIR = os.path.join("output", "social")


def sha256_hex(text: str, trunc: int = 32) -> str:
    h = hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()
    return h[:trunc] if trunc else h


def vk_api_request(method: str, params: dict, token: str, max_retries: int = 5) -> Optional[dict]:
    params.update({'access_token': token, 'v': VK_VERSION})
    for attempt in range(max_retries):
        try:
            resp = requests.get(f'https://api.vk.com/method/{method}', params=params, timeout=30).json()
            if 'error' in resp:
                error_msg = resp['error']['error_msg']
                if 'flood' in error_msg.lower() or 'too many requests' in error_msg.lower():
                    wait = (attempt + 1) * 3
                    print(f"\n[WAIT] Flood control, waiting {wait}s")
                    logger.warning(f"Flood control, waiting {wait}s")
                    time.sleep(wait)
                    continue
                print(f"\n[ERROR] {error_msg}")
                logger.error(f"VK API error: {error_msg}")
                return None
            return resp
        except Exception as e:
            logger.warning(f"Network error (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(1)
    return None


def get_comments(token: str, owner_id: int, post_id: int, total_comments: int) -> List[Dict]:
    comments = []
    offset = 0
    count = 100
    while offset < total_comments:
        params = {
            'owner_id': owner_id,
            'post_id': post_id,
            'count': count,
            'offset': offset,
            'need_likes': 1,
            'thread_items_count': 10,
            'extended': 0
        }
        resp = vk_api_request('wall.getComments', params, token)
        if not resp or 'error' in resp:
            break
        items = resp.get('response', {}).get('items', [])
        if not items:
            break
        for item in items:
            comment_date = datetime.fromtimestamp(item['date'])
            comment_data = {
                "comment_id": item['id'],
                "post_id": post_id,
                "parent_comment_id": 0,
                "content": item.get('text', ''),
                "date": comment_date.isoformat(),
                "author_id": item.get('from_id', 0),
                "likes": item.get('likes', {}).get('count', 0),
                "reply_to_user": item.get('reply_to_user', 0),
                "reply_to_comment": item.get('reply_to_comment', 0),
            }
            comments.append(comment_data)
            thread = item.get('thread', {})
            thread_items = thread.get('items', [])
            thread_count = thread.get('count', 0)
            if thread_count > 0 and len(thread_items) < thread_count:
                thread_comments = get_thread_comments(token, owner_id, post_id, item['id'], thread_count)
                comments.extend(thread_comments)
            elif thread_items:
                for reply in thread_items:
                    reply_date = datetime.fromtimestamp(reply['date'])
                    comments.append({
                        "comment_id": reply['id'],
                        "post_id": post_id,
                        "parent_comment_id": item['id'],
                        "content": reply.get('text', ''),
                        "date": reply_date.isoformat(),
                        "author_id": reply.get('from_id', 0),
                        "likes": reply.get('likes', {}).get('count', 0),
                        "reply_to_user": reply.get('reply_to_user', 0),
                        "reply_to_comment": reply.get('reply_to_comment', 0),
                    })
        offset += len(items)
        time.sleep(1.0)
    return comments


def get_thread_comments(token: str, owner_id: int, post_id: int, comment_id: int, total: int) -> List[Dict]:
    thread_comments = []
    offset = 0
    count = 100
    while offset < total:
        params = {
            'owner_id': owner_id,
            'post_id': post_id,
            'comment_id': comment_id,
            'count': count,
            'offset': offset,
            'need_likes': 1,
        }
        resp = vk_api_request('wall.getComments', params, token)
        if not resp or 'error' in resp:
            break
        items = resp.get('response', {}).get('items', [])
        if not items:
            break
        for item in items:
            reply_date = datetime.fromtimestamp(item['date'])
            thread_comments.append({
                "comment_id": item['id'],
                "post_id": post_id,
                "parent_comment_id": comment_id,
                "content": item.get('text', ''),
                "date": reply_date.isoformat(),
                "author_id": item.get('from_id', 0),
                "likes": item.get('likes', {}).get('count', 0),
                "reply_to_user": item.get('reply_to_user', 0),
                "reply_to_comment": item.get('reply_to_comment', 0),
            })
        offset += len(items)
        time.sleep(0.5)
    return thread_comments


def get_posts_with_comments(token: str, domain_id: str, language: str,
                            max_posts: Optional[int] = None,
                            with_comments: bool = True) -> tuple:
    posts = []
    all_comments = []
    offset = 0

    check = vk_api_request('wall.get', {'domain': domain_id, 'count': 1, 'offset': 0}, token)
    if not check or 'error' in check:
        print(f"[WARN] {domain_id}: public page not found, skipping")
        logger.warning(f"{domain_id}: public page not found")
        return [], []

    total = check.get('response', {}).get('count', 0)
    if max_posts:
        total = min(total, max_posts)

    logger.info(f"Starting collection for {domain_id}: {total} posts")
    print(f"   Total posts: {total:,}")

    with tqdm(total=total, desc=f"[VK] {domain_id}", unit="post", dynamic_ncols=True) as pbar:
        while offset < total:
            params = {'domain': domain_id, 'count': 100, 'offset': offset}
            resp = vk_api_request('wall.get', params, token)
            if not resp or 'error' in resp:
                break
            items = resp.get('response', {}).get('items', [])
            if not items:
                break
            for item in items:
                if 'id' not in item or 'date' not in item:
                    logger.debug(f"Skipping non-post item: {json.dumps(item, ensure_ascii=False)}")
                    continue

                content = item.get('text', '')
                post_date_timestamp = item['date']
                post_date = datetime.fromtimestamp(post_date_timestamp)

                post_data = {
                    "url": f"https://vk.com/wall{item.get('owner_id', 0)}_{item['id']}",
                    "title": content[:120] if content else "",
                    "content": content,
                    "excerpt": content[:260] + "..." if len(content) > 260 else content,
                    "date": post_date.isoformat(),
                    "author": str(item.get('from_id', '')),
                    "category": None,
                    "time": post_date.strftime("%H:%M"),
                    "site": f"vk.com/{domain_id}",
                    "hash": sha256_hex(content + str(item['id']), trunc=32),
                    "image_url": None,
                    "language": language,
                    "scraped_at": datetime.now().isoformat(),
                    "source_type": "vk",
                    "page_type": "article",
                    "domain": domain_id,
                    "post_id": item['id'],
                    "owner_id": item.get('owner_id', 0),
                    "likes": item.get('likes', {}).get('count', 0),
                    "comments_count": item.get('comments', {}).get('count', 0),
                    "reposts": item.get('reposts', {}).get('count', 0),
                    "views": item.get('views', {}).get('count', 0) if 'views' in item else 0,
                }
                if 'attachments' in item:
                    for att in item['attachments']:
                        if att['type'] == 'photo':
                            sizes = att['photo'].get('sizes', [])
                            if sizes:
                                post_data['image_url'] = sizes[-1].get('url')
                                break
                posts.append(post_data)
                pbar.update(1)

                if with_comments and post_data['comments_count'] > 0:
                    post_comments = get_comments(
                        token,
                        post_data['owner_id'],
                        item['id'],
                        post_data['comments_count']
                    )
                    all_comments.extend(post_comments)

                if max_posts and len(posts) >= max_posts:
                    break
            offset += len(items)
            if max_posts and len(posts) >= max_posts:
                break
            time.sleep(1.5)

    logger.info(f"Finished {domain_id}: {len(posts)} posts, {len(all_comments)} comments")
    return posts, all_comments


def main():
    parser = argparse.ArgumentParser(description="VK scraper")
    parser.add_argument('--token', help='VK API token (overrides config)')
    parser.add_argument('--domains', nargs='+', help='VK domain IDs (e.g., club1135692 irta_tv). Overrides config.')
    parser.add_argument('--lang', default=None, help='Filter by language when using config, or set language for manual domains')
    parser.add_argument('--max-posts', type=int, default=None, help='Maximum posts per domain (default: all)')
    args = parser.parse_args()

    logger.info("Starting VK scraper")

    config = load_social_config()

    domains_to_scrape = []

    if args.domains:
        lang = args.lang or 'unknown'
        for dom in args.domains:
            domains_to_scrape.append((dom, lang))
    else:
        vk_domains = config.get("vk_domains", [])
        if not vk_domains:
            logger.error("No VK domains found in config and none specified via --domains")
            print("[ERROR] No VK domains found. Use --domains or configure vk_domains in social config.")
            return

        for entry in vk_domains:
            domain_id = entry.get("id")
            if not domain_id:
                continue
            language = entry.get("language", "unknown")
            if args.lang and language != args.lang:
                continue
            domains_to_scrape.append((domain_id, language))

    if not domains_to_scrape:
        print(f"[INFO] No domains to scrape for language '{args.lang}'")
        return

    token = args.token or config.get("vk", {}).get("token")
    if not token:
        logger.error("VK token not found. Provide --token or set it in social config.")
        print("[ERROR] VK token not found. Use --token or configure vk.token in social config.")
        return

    posts_dir = os.path.join(OUTPUT_DIR, "posts")
    comments_dir = os.path.join(OUTPUT_DIR, "comments")
    os.makedirs(posts_dir, exist_ok=True)
    os.makedirs(comments_dir, exist_ok=True)

    for domain_id, language in domains_to_scrape:
        print(f"\n{'='*60}")
        print(f"[VK] Collecting posts from {domain_id} (lang: {language})")
        print(f"{'='*60}")

        posts, comments = get_posts_with_comments(
            token, domain_id, language, max_posts=args.max_posts
        )

        if posts:
            posts_file = os.path.join(posts_dir, f"vk_{domain_id}_posts.jsonl")
            with open(posts_file, "w", encoding="utf-8") as f:
                for p in posts:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
            print(f"[OK] Posts: {len(posts)} -> {posts_file}")
            logger.info(f"Posts saved: {len(posts)} to {posts_file}")

        if comments:
            comments_file = os.path.join(comments_dir, f"vk_{domain_id}_comments.jsonl")
            with open(comments_file, "w", encoding="utf-8") as f:
                for c in comments:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
            print(f"[OK] Comments: {len(comments)} -> {comments_file}")
            logger.info(f"Comments saved: {len(comments)} to {comments_file}")

        if not posts and not comments:
            print(f"[FAIL] {domain_id}: no data collected")

    logger.info("VK scraper finished")


if __name__ == "__main__":
    main()