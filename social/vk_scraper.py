# social_scrapers/vk_scraper_full.py
import json
import os
import sys
import time
import hashlib
import yaml
import requests
from datetime import datetime
from typing import Optional, List, Dict
from tqdm import tqdm

VK_VERSION = "5.199"

def sha256_hex(text: str, trunc: int = 32) -> str:
    h = hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()
    return h[:trunc] if trunc else h


def vk_api_request(method: str, params: dict, token: str, max_retries: int = 5) -> Optional[dict]:
    """Запрос к VK API с обработкой flood control и повторными попытками."""
    params.update({'access_token': token, 'v': VK_VERSION})
    
    for attempt in range(max_retries):
        try:
            resp = requests.get(f'https://api.vk.com/method/{method}', params=params, timeout=30).json()
            
            if 'error' in resp:
                error_msg = resp['error']['error_msg']
                
                if 'flood' in error_msg.lower() or 'too many requests' in error_msg.lower():
                    wait = (attempt + 1) * 3
                    print(f"\n⏳ Flood control, ждём {wait} сек...")
                    time.sleep(wait)
                    continue
                
                print(f"\n❌ {params.get('domain', '')}: {error_msg}")
                return None
            
            return resp
            
        except Exception as e:
            print(f"\n⚠️ Сетевая ошибка: {e}, попытка {attempt + 1}/{max_retries}")
            time.sleep(1)
    
    return None


def get_comments(token: str, owner_id: int, post_id: int, total_comments: int) -> List[Dict]:
    """Собирает все комментарии к посту."""
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
            'thread_items_count': 10,  # получаем ответы на комментарии
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
            
            # Собираем ответы на комментарий (thread)
            thread = item.get('thread', {})
            thread_items = thread.get('items', [])
            thread_count = thread.get('count', 0)
            
            if thread_count > 0 and len(thread_items) < thread_count:
                # Догружаем все ответы
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
    """Собирает все ответы на конкретный комментарий."""
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


def get_posts_with_comments(token: str, domain: str, max_posts: Optional[int] = None, 
                            with_comments: bool = True) -> tuple:
    """
    Собирает все посты и комментарии из VK-паблика.
    Возвращает (posts, comments).
    """
    posts = []
    all_comments = []
    offset = 0
    
    # Проверяем паблик
    check = vk_api_request('wall.get', {'domain': domain, 'count': 1, 'offset': 0}, token)
    if not check or 'error' in check:
        print(f"⚠️ {domain}: паблик не найден, пропускаем")
        return [], []
    
    total = check.get('response', {}).get('count', 0)
    if max_posts:
        total = min(total, max_posts)
    
    print(f"   Всего постов: {total:,}")
    
    with tqdm(total=total, desc=f"📱 {domain}", unit="пост", dynamic_ncols=True) as pbar:
        while offset < total:
            params = {'domain': domain, 'count': 100, 'offset': offset}
            resp = vk_api_request('wall.get', params, token)
            
            if not resp or 'error' in resp:
                break
            
            items = resp.get('response', {}).get('items', [])
            if not items:
                break
            
            for item in items:
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
                    "site": f"vk.com/{domain}",
                    "hash": sha256_hex(content + str(item['id']), trunc=32),
                    "image_url": None,
                    "language": "tt",
                    "scraped_at": datetime.now().isoformat(),
                    "source_type": "vk",
                    "page_type": "article",
                    "domain": domain,
                    "post_id": item['id'],
                    "owner_id": item.get('owner_id', 0),
                    "likes": item.get('likes', {}).get('count', 0),
                    "comments_count": item.get('comments', {}).get('count', 0),
                    "reposts": item.get('reposts', {}).get('count', 0),
                    "views": item.get('views', {}).get('count', 0) if 'views' in item else 0,
                }
                
                # Изображения
                if 'attachments' in item:
                    for att in item['attachments']:
                        if att['type'] == 'photo':
                            sizes = att['photo'].get('sizes', [])
                            if sizes:
                                post_data['image_url'] = sizes[-1].get('url')
                                break
                
                posts.append(post_data)
                pbar.update(1)
                
                # Собираем комментарии к посту
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
    
    return posts, all_comments


def main():
    if len(sys.argv) < 2:
        config_path = "social_config.yaml"
    else:
        config_path = sys.argv[1]

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    vk_cfg = cfg.get("vk", {})
    token = vk_cfg["token"]
    output_dir = cfg.get("output_dir", "social_jsonl")
    posts_dir = os.path.join(output_dir, "posts")
    comments_dir = os.path.join(output_dir, "comments")
    
    os.makedirs(posts_dir, exist_ok=True)
    os.makedirs(comments_dir, exist_ok=True)

    for domain in vk_cfg.get("domains", []):
        print(f"\n{'='*60}")
        print(f"📱 Собираем посты и комментарии из {domain}...")
        print(f"{'='*60}")
        
        posts, comments = get_posts_with_comments(token, domain)
        
        # Сохраняем посты
        if posts:
            posts_file = os.path.join(posts_dir, f"vk_{domain}_posts.jsonl")
            with open(posts_file, "w", encoding="utf-8") as f:
                for p in posts:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
            print(f"✅ Посты: {len(posts)} -> {posts_file}")
        
        # Сохраняем комментарии
        if comments:
            comments_file = os.path.join(comments_dir, f"vk_{domain}_comments.jsonl")
            with open(comments_file, "w", encoding="utf-8") as f:
                for c in comments:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
            print(f"💬 Комментарии: {len(comments)} -> {comments_file}")
        
        if not posts and not comments:
            print(f"❌ {domain}: данные не собраны")


if __name__ == "__main__":
    main()

#python social_scrapers/vk_scraper.py social_config.yaml 