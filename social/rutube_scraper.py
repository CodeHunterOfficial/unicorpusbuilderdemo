# social/rutube_scraper.py
"""
Проверенный скрапер Rutube:
- Комментарии к видео (Playwright) — ваш рабочий метод
- Видео из плейлистов (Playwright) — новый метод
- Статистика канала (REST API)
"""

import json, os, sys, time, hashlib, yaml
import requests
from datetime import datetime
from typing import Optional, List, Dict
from tqdm import tqdm
from playwright.sync_api import sync_playwright

BASE_URL = "https://rutube.ru"

def sha256_hex(text: str, trunc: int = 32) -> str:
    h = hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()
    return h[:trunc] if trunc else h

class RutubeScraper:
    def __init__(self, config: dict):
        self.config = config
        self.chrome_path = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"

    # ── КОММЕНТАРИИ (ваш проверенный метод) ──
    def get_video_comments(self, video_url: str, max_comments: int = 100) -> List[Dict]:
        comments = []
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    executable_path=self.chrome_path,
                    headless=True
                )
                page = browser.new_page()
                page.goto(video_url)
                page.wait_for_timeout(5000)

                for _ in range(10):
                    page.mouse.wheel(0, 2000)
                    time.sleep(1)

                elements = page.locator("[class*=comment]").all()
                for el in elements[:max_comments]:
                    try:
                        text = el.inner_text()
                    except:
                        text = ""
                    if text.strip():
                        comments.append({
                            "video_url": video_url,
                            "content": text,
                            "scraped_at": datetime.now().isoformat()
                        })
                browser.close()
        except Exception as e:
            print(f"❌ Ошибка при сборе комментариев: {e}")

        print(f"💬 Найдено комментариев: {len(comments)}")
        return comments

    # ── ПЛЕЙЛИСТ (через Playwright, использует ваш Chrome) ──
    def get_playlist_videos(self, playlist_id: str, max_videos: int = None) -> List[str]:
        video_urls = []
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    executable_path=self.chrome_path,
                    headless=True
                )
                page = browser.new_page()
                page.goto(f"https://rutube.ru/plst/{playlist_id}/")
                page.wait_for_timeout(3000)

                # собираем все ссылки, ведущие на видео
                links = page.locator("a[href*='/video/']").all()
                for link in links:
                    href = link.get_attribute("href")
                    if href and "/video/" in href:
                        video_id = href.split("/video/")[1].split("/")[0]
                        full_url = f"https://rutube.ru/video/{video_id}/"
                        if full_url not in video_urls:
                            video_urls.append(full_url)
                            if max_videos and len(video_urls) >= max_videos:
                                break
                browser.close()
        except Exception as e:
            print(f"❌ Ошибка парсинга плейлиста: {e}")

        print(f"📺 Плейлист {playlist_id}: найдено видео {len(video_urls)}")
        return video_urls[:max_videos] if max_videos else video_urls

    # ── СТАТИСТИКА КАНАЛА (REST API) ──
    def get_channel_info(self, channel_id: int) -> Optional[Dict]:
        try:
            resp = requests.get(f"https://rutube.ru/api/channel/{channel_id}", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return {
                "channel_id": channel_id,
                "name": data.get("name"),
                "subscribers_count": data.get("subscribers_count", 0),
                "videos_count": data.get("videos_count", 0),
                "views_count": data.get("views_count", 0),
                "updated_at": datetime.now().isoformat(),
                "source_type": "rutube_channel_stats",
                "hash": sha256_hex(str(channel_id)),
            }
        except Exception as e:
            print(f"❌ Ошибка получения статистики канала: {e}")
            return None


# ── ЗАПУСК ──
def main():
    if len(sys.argv) < 2:
        config_path = "social/social_config.yaml"
    else:
        config_path = sys.argv[1]

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rt_cfg = cfg.get("rutube", {})
    if not rt_cfg:
        print("❌ Секция rutube не найдена в конфиге")
        return

    scraper = RutubeScraper(rt_cfg)
    output_dir = cfg.get("output_dir", "social_jsonl")
    os.makedirs(output_dir, exist_ok=True)

    # --- Статистика каналов ---
    for ch_id in rt_cfg.get("channel_ids", []):
        info = scraper.get_channel_info(ch_id)
        if info:
            fname = os.path.join(output_dir, f"rutube_channel_{ch_id}_stats.jsonl")
            with open(fname, "a", encoding="utf-8") as f:
                f.write(json.dumps(info, ensure_ascii=False) + "\n")
            print(f"📊 Канал {ch_id}: {info['subscribers_count']} подписчиков -> {fname}")

    # --- Плейлисты (видео + комментарии) ---
    for pl in rt_cfg.get("playlists", []):
        pl_id = pl if isinstance(pl, str) else pl.get("id", "")
        max_videos = pl.get("max_videos", None) if isinstance(pl, dict) else None
        max_comments = pl.get("max_comments", None) if isinstance(pl, dict) else None
        if not pl_id:
            continue
        print(f"\n📺 Плейлист: {pl_id}")
        video_urls = scraper.get_playlist_videos(pl_id, max_videos)
        if video_urls:
            vf = os.path.join(output_dir, f"rutube_playlist_{pl_id}_videos.jsonl")
            with open(vf, "w", encoding="utf-8") as f:
                for url in video_urls:
                    f.write(json.dumps({"url": url}, ensure_ascii=False) + "\n")
            print(f"✅ Видео: {len(video_urls)} -> {vf}")

            for url in video_urls:
                comments = scraper.get_video_comments(url, max_comments)
                if comments:
                    cf = os.path.join(output_dir, f"rutube_playlist_{pl_id}_comments.jsonl")
                    with open(cf, "a", encoding="utf-8") as f:
                        for c in comments:
                            f.write(json.dumps(c, ensure_ascii=False) + "\n")
                    print(f"   💬 {url.split('/')[-2]}: {len(comments)} комм.")

    # --- Отдельные видео (комментарии) ---
    for v in rt_cfg.get("videos", []):
        vid = v if isinstance(v, str) else v.get("id", "")
        max_com = v.get("max_comments", None) if isinstance(v, dict) else None
        if not vid:
            continue
        url = f"https://rutube.ru/video/{vid}/"
        print(f"\n💬 Видео: {url}")
        comments = scraper.get_video_comments(url, max_com)
        if comments:
            fname = os.path.join(output_dir, f"rutube_video_{vid}_comments.jsonl")
            with open(fname, "w", encoding="utf-8") as f:
                for c in comments:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
            print(f"✅ Комментарии: {len(comments)} -> {fname}")

if __name__ == "__main__":
    main()