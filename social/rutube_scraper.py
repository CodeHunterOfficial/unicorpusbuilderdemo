# social/rutube_scraper.py
"""
Rutube scraper with logging support:
- Video comments (Playwright)
- Videos from playlists (Playwright)
- Channel statistics (REST API)

Usage:
    python social/rutube_scraper.py --video-id fb6eadbddb4dd00cad0ea58b8b060b48 --max-comments 100
    python social/rutube_scraper.py --playlist-id 1329269 --max-videos 10 --max-comments 50
    python social/rutube_scraper.py --max-comments 200
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
from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.loader import load_social_config
from logger_setup import get_file_logger

logger = get_file_logger("rutube_scraper", "logs/rutube_scraper.log")

BASE_URL = "https://rutube.ru"
OUTPUT_DIR = os.path.join("output", "social")


def sha256_hex(text: str, trunc: int = 32) -> str:
    h = hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()
    return h[:trunc] if trunc else h


class RutubeScraper:
    def __init__(self, chrome_path: str = None):
        self.chrome_path = chrome_path or "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
        logger.info("RutubeScraper initialized")

    def get_video_comments(self, video_url: str, max_comments: int) -> List[Dict]:
        logger.info(f"Collecting comments for video: {video_url} (max: {max_comments})")
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

                with tqdm(total=max_comments, desc=f"[COMMENTS] {video_url.split('/')[-2]}", unit="comm") as pbar:
                    for _ in range(10):
                        page.mouse.wheel(0, 2000)
                        time.sleep(1)

                        elements = page.locator("[class*=comment]").all()
                        new_comments = 0
                        for el in elements:
                            if len(comments) >= max_comments:
                                break
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
                                new_comments += 1
                        if new_comments:
                            pbar.update(new_comments)
                        if len(comments) >= max_comments:
                            break
                browser.close()
                logger.info(f"Collected {len(comments)} comments for {video_url}")
        except Exception as e:
            logger.error(f"Error collecting comments for {video_url}: {e}")
            print(f"[ERROR] Error collecting comments: {e}")

        print(f"[OK] Comments found: {len(comments)}")
        return comments

    def get_playlist_videos(self, playlist_id: str, max_videos: int = None) -> List[str]:
        logger.info(f"Collecting videos from playlist {playlist_id} (max: {max_videos})")
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
                logger.info(f"Playlist {playlist_id}: found {len(video_urls)} videos")
        except Exception as e:
            logger.error(f"Error parsing playlist {playlist_id}: {e}")
            print(f"[ERROR] Error parsing playlist: {e}")

        print(f"[OK] Playlist {playlist_id}: found {len(video_urls)} videos")
        return video_urls[:max_videos] if max_videos else video_urls

    def get_channel_info(self, channel_id: int) -> Optional[Dict]:
        logger.info(f"Getting channel info for ID: {channel_id}")
        try:
            resp = requests.get(f"https://rutube.ru/api/channel/{channel_id}", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            info = {
                "channel_id": channel_id,
                "name": data.get("name"),
                "subscribers_count": data.get("subscribers_count", 0),
                "videos_count": data.get("videos_count", 0),
                "views_count": data.get("views_count", 0),
                "updated_at": datetime.now().isoformat(),
                "source_type": "rutube_channel_stats",
                "hash": sha256_hex(str(channel_id)),
            }
            logger.info(f"Channel {channel_id}: {data.get('name')} - {info['subscribers_count']} subscribers")
            return info
        except Exception as e:
            logger.error(f"Error getting channel {channel_id}: {e}")
            print(f"[ERROR] Error getting channel statistics: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description="Rutube scraper")
    parser.add_argument('--video-id', help='Single video ID to scrape comments from')
    parser.add_argument('--playlist-id', help='Single playlist ID to scrape videos from')
    parser.add_argument('--max-comments', type=int, default=None, help='Max comments per video (overrides config)')
    parser.add_argument('--max-videos', type=int, default=None, help='Max videos per playlist (overrides config)')
    parser.add_argument('--chrome-path', help='Path to Chrome executable')
    args = parser.parse_args()

    logger.info("Starting Rutube scraper")
    config = load_social_config()
    rt_cfg = config.get("rutube", {})

    chrome_path = args.chrome_path or "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
    scraper = RutubeScraper(chrome_path=chrome_path)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Single video mode
    if args.video_id:
        url = f"https://rutube.ru/video/{args.video_id}/"
        max_com = args.max_comments or 100
        print(f"\n[VIDEO] {url}")
        comments = scraper.get_video_comments(url, max_com)
        if comments:
            fname = os.path.join(OUTPUT_DIR, f"rutube_video_{args.video_id}_comments.jsonl")
            with open(fname, "w", encoding="utf-8") as f:
                for c in comments:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
            print(f"[OK] Comments: {len(comments)} -> {fname}")
        return

    # Single playlist mode
    if args.playlist_id:
        max_videos = args.max_videos
        max_comments = args.max_comments or 50
        print(f"\n[PLAYLIST] {args.playlist_id}")
        video_urls = scraper.get_playlist_videos(args.playlist_id, max_videos)
        if video_urls:
            vf = os.path.join(OUTPUT_DIR, f"rutube_playlist_{args.playlist_id}_videos.jsonl")
            with open(vf, "w", encoding="utf-8") as f:
                for url in video_urls:
                    f.write(json.dumps({"url": url}, ensure_ascii=False) + "\n")
            print(f"[OK] Videos: {len(video_urls)} -> {vf}")

            for url in video_urls:
                comments = scraper.get_video_comments(url, max_comments)
                if comments:
                    cf = os.path.join(OUTPUT_DIR, f"rutube_playlist_{args.playlist_id}_comments.jsonl")
                    with open(cf, "a", encoding="utf-8") as f:
                        for c in comments:
                            f.write(json.dumps(c, ensure_ascii=False) + "\n")
                    print(f"   [OK] {url.split('/')[-2]}: {len(comments)} comments.")
        return

    # Config mode: process all from config
    if not rt_cfg:
        logger.error("No Rutube section found in config")
        print("[ERROR] Rutube section not found in config")
        return

    # Channel IDs
    for ch_id in rt_cfg.get("channel_ids", []):
        info = scraper.get_channel_info(ch_id)
        if info:
            fname = os.path.join(OUTPUT_DIR, f"rutube_channel_{ch_id}_stats.jsonl")
            with open(fname, "a", encoding="utf-8") as f:
                f.write(json.dumps(info, ensure_ascii=False) + "\n")
            print(f"[OK] Channel {ch_id}: {info['subscribers_count']} subscribers -> {fname}")

    # Playlists
    for pl in rt_cfg.get("playlists", []):
        pl_id = pl if isinstance(pl, str) else pl.get("id", "")
        max_videos = args.max_videos or (pl.get("max_videos", None) if isinstance(pl, dict) else None)
        max_comments = args.max_comments or (pl.get("max_comments", 50) if isinstance(pl, dict) else 50)
        if not pl_id:
            continue

        print(f"\n[PLAYLIST] {pl_id}")
        video_urls = scraper.get_playlist_videos(pl_id, max_videos)
        if video_urls:
            vf = os.path.join(OUTPUT_DIR, f"rutube_playlist_{pl_id}_videos.jsonl")
            with open(vf, "w", encoding="utf-8") as f:
                for url in video_urls:
                    f.write(json.dumps({"url": url}, ensure_ascii=False) + "\n")
            print(f"[OK] Videos: {len(video_urls)} -> {vf}")

            for url in video_urls:
                comments = scraper.get_video_comments(url, max_comments)
                if comments:
                    cf = os.path.join(OUTPUT_DIR, f"rutube_playlist_{pl_id}_comments.jsonl")
                    with open(cf, "a", encoding="utf-8") as f:
                        for c in comments:
                            f.write(json.dumps(c, ensure_ascii=False) + "\n")
                    print(f"   [OK] {url.split('/')[-2]}: {len(comments)} comments.")

    # Individual videos
    for v in rt_cfg.get("videos", []):
        vid = v if isinstance(v, str) else v.get("id", "")
        max_com = args.max_comments or (v.get("max_comments", 100) if isinstance(v, dict) else 100)
        if not vid:
            continue
        url = f"https://rutube.ru/video/{vid}/"
        print(f"\n[VIDEO] {url}")
        comments = scraper.get_video_comments(url, max_com)
        if comments:
            fname = os.path.join(OUTPUT_DIR, f"rutube_video_{vid}_comments.jsonl")
            with open(fname, "w", encoding="utf-8") as f:
                for c in comments:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
            print(f"[OK] Comments: {len(comments)} -> {fname}")

    logger.info("Rutube scraper finished")


if __name__ == "__main__":
    main()