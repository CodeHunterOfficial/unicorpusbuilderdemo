# social/vk_scraper_async.py (fixed accelerated version with progress bars)
"""
VK scraper with logging support (async + controlled parallelism + flood protection).
Collects posts and comments from VK public pages.

Usage:
    python social/vk_scraper.py --domains club1135692 irta_tv --lang ba --max-posts 5
    python social/vk_scraper.py --lang tt --max-posts 10
    python social/vk_scraper.py --max-posts 100
    python social/vk_scraper.py --lang tt --max-posts 100
"""

import asyncio
import aiohttp
import json
import os
import sys
import time
import hashlib
import argparse
from datetime import datetime
from typing import Optional, List, Dict
from tqdm.asyncio import tqdm_asyncio

# Dynamic import of configs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.loader import load_social_config
from logger_setup import get_file_logger

logger = get_file_logger("vk_scraper", "logs/vk_scraper.log")

# Speed settings (default constants)
CALLS_PER_SECOND = 2.5
MAX_CONCURRENT_API = 3
COMMENT_CONCURRENT = 5
DOMAIN_PAUSE = 3
VK_VERSION = "5.199"
OUTPUT_DIR = os.path.join("output", "social")

# ------------------------------------------------------------------------

def sha256_hex(text: str, trunc: int = 32) -> str:
    h = hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()
    return h[:trunc] if trunc else h


class RateLimiter:
    """Global rate limiter with support for parallel requests."""
    def __init__(self, calls_per_second: float = 2.5, max_concurrent: int = 3):
        self.min_interval = 1.0 / calls_per_second
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.last_call = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Acquire a parallel execution slot."""
        await self.semaphore.acquire()

    def release(self):
        """Release a slot."""
        self.semaphore.release()

    async def wait_interval(self):
        """Global time interval between any requests."""
        async with self._lock:
            now = time.monotonic()
            wait = self.last_call + self.min_interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self.last_call = time.monotonic()


class FloodDetectedError(Exception):
    """Exception for immediate interruption on VK flood error."""
    pass


class VKApiClient:
    def __init__(self, token: str, session: aiohttp.ClientSession,
                 max_concurrent_api: int = MAX_CONCURRENT_API,
                 comment_concurrent: int = COMMENT_CONCURRENT):
        self.token = token
        self.session = session
        self.max_concurrent_api = max_concurrent_api
        self.comment_concurrent = comment_concurrent
        self.limiter = RateLimiter(calls_per_second=CALLS_PER_SECOND,
                                   max_concurrent=max_concurrent_api)
        self.flood_detected = False

    async def request(self, method: str, params: dict, max_retries: int = 2) -> Optional[dict]:
        """Request with concurrency control and ban protection."""
        await self.limiter.acquire()
        try:
            return await self._execute_request(method, params, max_retries)
        finally:
            self.limiter.release()

    async def _execute_request(self, method: str, params: dict, max_retries: int) -> Optional[dict]:
        params = {**params, 'access_token': self.token, 'v': VK_VERSION}

        for attempt in range(max_retries):
            await self.limiter.wait_interval()

            try:
                async with self.session.get(
                    f'https://api.vk.com/method/{method}',
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    data = await resp.json()
            except Exception as e:
                logger.warning(f"Network error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                return None

            if 'error' in data:
                err = data['error']
                error_code = err.get('error_code', 0)
                error_msg = err.get('error_msg', '')

                if error_code in (6, 9, 10) or 'flood' in error_msg.lower():
                    msg = (f"\n[FLOOD] VK blocked requests for this token!\n"
                           f"[FLOOD] Error: {error_msg}\n"
                           f"[FLOOD] You MUST wait 30-60 minutes before the next run!")
                    print(msg)
                    logger.error(f"FLOOD DETECTED: {error_msg}")
                    self.flood_detected = True
                    raise FloodDetectedError(f"Flood control: {error_msg}")

                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                logger.error(f"VK API error [{error_code}]: {error_msg}")
                return None

            return data

        return None

    async def get_all_posts(self, domain_id: str, max_posts: Optional[int] = None) -> List[Dict]:
        """Parallel loading of all posts with progress bar."""
        try:
            resp = await self.request('wall.get', {'domain': domain_id, 'count': 1})
        except FloodDetectedError:
            return []

        if not resp or 'error' in resp:
            msg = resp['error'].get('error_msg', 'No response') if resp else 'No response'
            print(f"[ERROR] {domain_id}: {msg}")
            logger.warning(f"{domain_id}: {msg}")
            return []

        total = resp['response']['count']
        if max_posts:
            total = min(total, max_posts)

        print(f"   Total posts available: {total:,}")
        print(f"   Fetching in parallel (safe mode, up to {self.max_concurrent_api} simultaneous requests)")
        logger.info(f"Fetching {total} posts from {domain_id} in parallel")

        batch_size = 100
        offsets = list(range(0, total, batch_size))
        
        pbar = tqdm_asyncio(total=total, desc=f"[VK] {domain_id}", unit="post", dynamic_ncols=True)
        
        tasks = [self._fetch_posts_page(domain_id, off, batch_size, pbar) for off in offsets]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        pbar.close()

        raw_posts = []
        for res in results:
            if isinstance(res, FloodDetectedError):
                raise res
            if isinstance(res, Exception) or not res:
                continue
            items = res.get('response', {}).get('items', [])
            raw_posts.extend(items)

        result = raw_posts[:total]
        print(f"   Collected {len(result):,} posts")
        return result

    async def _fetch_posts_page(self, domain_id: str, offset: int, batch_size: int, pbar=None) -> Optional[dict]:
        """Helper function for parallel loading of a page of posts."""
        try:
            result = await self.request('wall.get', {
                'domain': domain_id,
                'count': batch_size,
                'offset': offset
            })
            if result and 'response' in result and pbar:
                items = result['response'].get('items', [])
                pbar.update(len(items))
            return result
        except FloodDetectedError:
            raise
        except Exception:
            return None

    async def get_thread_comments(self, owner_id: int, post_id: int, comment_id: int, total: int, pbar=None) -> List[Dict]:
        """Sequential retrieval of thread comments."""
        thread_comments = []
        offset = 0
        batch_size = 100

        while offset < total:
            try:
                data = await self.request('wall.getComments', {
                    'owner_id': owner_id,
                    'post_id': post_id,
                    'comment_id': comment_id,
                    'count': batch_size,
                    'offset': offset,
                    'need_likes': 1
                })
            except FloodDetectedError:
                return thread_comments

            if not data or 'error' in data:
                break

            items = data.get('response', {}).get('items', [])
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
            if pbar:
                pbar.update(len(items))
            await asyncio.sleep(0.2)

        return thread_comments

    async def get_comments_for_post(self, post: Dict, pbar=None) -> List[Dict]:
        """Retrieval of all comments for one post (sequential within post)."""
        owner_id = post['owner_id']
        post_id = post['id']
        total = post['comments']['count']
        if total == 0:
            if pbar:
                pbar.update(0)
            return []

        comments = []
        offset = 0
        batch_size = 100

        while offset < total:
            try:
                data = await self.request('wall.getComments', {
                    'owner_id': owner_id,
                    'post_id': post_id,
                    'count': batch_size,
                    'offset': offset,
                    'need_likes': 1,
                    'thread_items_count': 10,
                    'extended': 0
                })
            except FloodDetectedError:
                return comments

            if not data or 'error' in data:
                break

            items = data.get('response', {}).get('items', [])
            if not items:
                break

            for item in items:
                comment_date = datetime.fromtimestamp(item['date'])
                comments.append({
                    "comment_id": item['id'],
                    "post_id": post_id,
                    "parent_comment_id": 0,
                    "content": item.get('text', ''),
                    "date": comment_date.isoformat(),
                    "author_id": item.get('from_id', 0),
                    "likes": item.get('likes', {}).get('count', 0),
                    "reply_to_user": item.get('reply_to_user', 0),
                    "reply_to_comment": item.get('reply_to_comment', 0),
                })

                thread = item.get('thread', {})
                if thread.get('count', 0) > 0:
                    try:
                        thread_comms = await self.get_thread_comments(
                            owner_id, post_id, item['id'], thread['count'], pbar
                        )
                        comments.extend(thread_comms)
                    except FloodDetectedError:
                        return comments

            offset += len(items)
            if pbar:
                pbar.update(len(items))
            await asyncio.sleep(0.2)

        return comments

    async def collect_all_comments(self, posts: List[Dict]) -> List[Dict]:
        """Parallel comment collection across different posts with progress bar."""
        posts_with_comments = [p for p in posts if p.get('comments', {}).get('count', 0) > 0]
        if not posts_with_comments:
            return []

        total_comments_count = sum(p.get('comments', {}).get('count', 0) for p in posts_with_comments)
        print(f"   Collecting comments from {len(posts_with_comments):,} posts (total ~{total_comments_count:,} comments)...")
        logger.info(f"Fetching comments for {len(posts_with_comments)} posts in parallel (limit {self.comment_concurrent})")

        pbar = tqdm_asyncio(total=total_comments_count, desc="[Comments]", unit="comm", dynamic_ncols=True)
        
        sem = asyncio.Semaphore(self.comment_concurrent)
        all_comments = []

        async def bounded_get_comments(post):
            async with sem:
                if self.flood_detected:
                    return []
                return await self.get_comments_for_post(post, pbar)

        tasks = [bounded_get_comments(post) for post in posts_with_comments]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        pbar.close()

        for res in results:
            if isinstance(res, FloodDetectedError):
                raise res
            if isinstance(res, list):
                all_comments.extend(res)

        print(f"   Collected {len(all_comments):,} comments")
        return all_comments


def transform_post(item: Dict, domain_id: str, language: str) -> Dict:
    """Transforms raw VK post into a structure for saving."""
    content = item.get('text', '')
    post_date = datetime.fromtimestamp(item['date'])
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

    return post_data


async def scrape_domain(client: VKApiClient, domain_id: str, language: str, max_posts: Optional[int] = None):
    """Processes one domain."""
    print(f"\n{'='*60}")
    print(f"[VK] Collecting posts from {domain_id} (lang: {language})")
    print(f"{'='*60}")

    try:
        raw_posts = await client.get_all_posts(domain_id, max_posts)
    except FloodDetectedError:
        print(f"[FAIL] {domain_id}: flood detected, skipping")
        return

    if client.flood_detected:
        print(f"[CRITICAL] Flood detected globally, stopping!")
        return

    if not raw_posts:
        print(f"[FAIL] {domain_id}: no posts found")
        return

    posts = [transform_post(p, domain_id, language) for p in raw_posts]

    comments = []
    if not client.flood_detected:
        try:
            comments = await client.collect_all_comments(raw_posts)
        except FloodDetectedError:
            print(f"[WARN] Comments collection interrupted by flood")

    posts_dir = os.path.join(OUTPUT_DIR, "posts")
    comments_dir = os.path.join(OUTPUT_DIR, "comments")
    os.makedirs(posts_dir, exist_ok=True)
    os.makedirs(comments_dir, exist_ok=True)

    posts_file = os.path.join(posts_dir, f"vk_{domain_id}_posts.jsonl")
    with open(posts_file, "w", encoding="utf-8") as f:
        for p in posts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[OK] Posts: {len(posts):,} -> {posts_file}")
    logger.info(f"Posts saved: {len(posts)} to {posts_file}")

    if comments:
        comments_file = os.path.join(comments_dir, f"vk_{domain_id}_comments.jsonl")
        with open(comments_file, "w", encoding="utf-8") as f:
            for c in comments:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"[OK] Comments: {len(comments):,} -> {comments_file}")
        logger.info(f"Comments saved: {len(comments)} to {comments_file}")

    logger.info(f"Finished {domain_id}: {len(posts)} posts, {len(comments)} comments")

    if not client.flood_detected:
        print(f"   Waiting {DOMAIN_PAUSE} seconds before next domain...")
        await asyncio.sleep(DOMAIN_PAUSE)


async def main_async():
    parser = argparse.ArgumentParser(description="VK scraper (async, parallel safe mode)")
    parser.add_argument('--token', help='VK API token')
    parser.add_argument('--domains', nargs='+', help='VK domain IDs (overrides config)')
    parser.add_argument('--lang', default=None, help='Filter by language when using config, or set language for manual domains')
    parser.add_argument('--max-posts', type=int, default=None, help='Maximum posts per domain')
    parser.add_argument('--max-concurrent', type=int, default=MAX_CONCURRENT_API,
                        help=f'Max simultaneous API requests (default: {MAX_CONCURRENT_API})')
    parser.add_argument('--comment-concurrent', type=int, default=COMMENT_CONCURRENT,
                        help=f'Max simultaneous comment fetches per domain (default: {COMMENT_CONCURRENT})')
    args = parser.parse_args()

    print(f"[INFO] Parallel settings: API concurrency={args.max_concurrent}, "
          f"Comment concurrency={args.comment_concurrent}")
    logger.info("Starting VK scraper (parallel safe mode)")

    config = load_social_config()
    domains_to_scrape = []

    if args.domains:
        lang = args.lang or 'unknown'
        for dom in args.domains:
            domains_to_scrape.append((dom, lang))
        print(f"[INFO] Manual mode: {len(domains_to_scrape)} domain(s) specified")
    else:
        vk_domains = config.get("vk_domains", [])
        if not vk_domains:
            logger.error("No VK domains found in config and none specified via --domains")
            print("[ERROR] No VK domains found. Use --domains or configure vk_domains in social config.")
            return

        for entry in vk_domains:
            dom = entry.get("id")
            if not dom:
                continue
            language = entry.get("language", "unknown")
            if args.lang and language != args.lang:
                continue
            domains_to_scrape.append((dom, language))

        print(f"[INFO] Config mode: {len(domains_to_scrape)} domain(s) for language '{args.lang or 'all'}'")

    if not domains_to_scrape:
        print(f"[INFO] No domains to scrape for language '{args.lang}'")
        return

    token = args.token or config.get("vk", {}).get("token")
    if not token:
        logger.error("VK token not found. Use --token or configure vk.token in social config.")
        print("[ERROR] VK token not found. Use --token or configure vk.token in social config.")
        return

    print(f"[INFO] Starting with {len(domains_to_scrape)} domain(s)")
    print(f"[INFO] Rate limit: {CALLS_PER_SECOND} requests/second")
    print(f"[INFO] Domain pause: {DOMAIN_PAUSE} seconds")
    print()

    async with aiohttp.ClientSession() as session:
        client = VKApiClient(
            token, session,
            max_concurrent_api=args.max_concurrent,
            comment_concurrent=args.comment_concurrent
        )

        for i, (domain_id, language) in enumerate(domains_to_scrape, 1):
            if client.flood_detected:
                print(f"\n{'='*60}")
                print(f"[CRITICAL] VK flood block detected!")
                print(f"[CRITICAL] Completed: {i-1}/{len(domains_to_scrape)} domains")
                print(f"[CRITICAL] Wait 30-60 minutes before running again!")
                print(f"{'='*60}")
                break

            print(f"\n[Progress] Domain {i}/{len(domains_to_scrape)}")
            await scrape_domain(client, domain_id, language, args.max_posts)

    logger.info("VK scraper finished")
    print(f"\n{'='*60}")
    print(f"[DONE] VK scraping completed!")
    if client.flood_detected:
        print(f"[WARN] Some domains skipped due to flood protection")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main_async())