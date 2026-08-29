# social/telegram_scraper.py
"""
Telegram scraper with logging support.
Collects posts from Telegram channels.

Usage:
    python social/telegram_scraper.py --channels vatantat --lang tt --max-posts 50
    python social/telegram_scraper.py --lang tt --max-posts 100
    python social/telegram_scraper.py --max-posts 200
"""

import asyncio
import hashlib
import json
import os
import sys
import argparse
from datetime import datetime
from telethon import TelegramClient
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.loader import load_social_config
from logger_setup import get_file_logger

logger = get_file_logger("telegram_scraper", "logs/telegram_scraper.log")

OUTPUT_DIR = os.path.join("output", "social")


def sha256_hex(text: str, trunc: int = 32) -> str:
    h = hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()
    return h[:trunc] if trunc else h


async def fetch_channel(client, channel_name, language, output_dir, max_posts=None):
    logger.info(f"Starting channel: {channel_name}")
    print(f"[TG] Accessing channel: {channel_name}...")

    entity = None
    try:
        entity = await client.get_entity(channel_name)
    except Exception as e:
        logger.error(f"Could not find channel {channel_name}: {e}")
        print(f"[WARN] {channel_name}: channel not found, skipping")
        return

    total = 0
    try:
        async for _ in client.iter_messages(entity, limit=max_posts):
            total += 1
    except Exception as e:
        logger.error(f"Error accessing channel {channel_name}: {e}")
        print(f"[ERROR] Could not access channel {channel_name}: {e}")
        return

    if total == 0:
        print(f"[WARN] {channel_name}: no messages found")
        return

    print(f"   Messages to collect: {total:,}")
    logger.info(f"Channel {channel_name}: collecting {total} messages")

    messages = []

    with tqdm(total=total, desc=f"[TG] {channel_name}", unit="post", dynamic_ncols=True) as pbar:
        async for msg in client.iter_messages(entity, limit=max_posts):
            if msg.text:
                content = msg.text
                msg_date = msg.date.replace(tzinfo=None) if msg.date else datetime.now()

                post_data = {
                    "url": f"https://t.me/{channel_name}/{msg.id}",
                    "title": content.split("\n")[0][:120] if content else "",
                    "content": content,
                    "excerpt": content[:260] + "..." if len(content) > 260 else content,
                    "date": msg_date.isoformat(),
                    "author": getattr(msg.sender, 'username', None) or str(msg.sender_id) if msg.sender else str(msg.sender_id) if msg.sender_id else None,
                    "category": None,
                    "time": msg_date.strftime("%H:%M"),
                    "site": f"t.me/{channel_name}",
                    "hash": sha256_hex(content + str(msg.id), trunc=32),
                    "image_url": None,
                    "language": language,
                    "scraped_at": datetime.now().isoformat(),
                    "source_type": "telegram",
                    "page_type": "article",
                    "channel": channel_name,
                    "post_id": msg.id,
                    "views": getattr(msg, 'views', 0),
                    "forwards": getattr(msg, 'forwards', 0),
                }

                messages.append(post_data)
                pbar.update(1)

    if messages:
        filename = os.path.join(output_dir, f"telegram_{channel_name}.jsonl")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            for m in messages:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        logger.info(f"Channel {channel_name}: {len(messages)} messages saved to {filename}")
        print(f"[OK] {channel_name}: {len(messages)} messages -> {filename}")
    else:
        logger.warning(f"Channel {channel_name}: no messages collected")
        print(f"[WARN] {channel_name}: no messages collected")


async def main():
    parser = argparse.ArgumentParser(description="Telegram scraper")
    parser.add_argument('--api-id', type=int, help='Telegram API ID (overrides config)')
    parser.add_argument('--api-hash', help='Telegram API hash (overrides config)')
    parser.add_argument('--channels', nargs='+', help='Channel names without @ (overrides config)')
    parser.add_argument('--lang', default=None, help='Filter by language when using config, or set language for manual channels')
    parser.add_argument('--max-posts', type=int, default=None, help='Maximum posts per channel (default: all)')
    args = parser.parse_args()

    logger.info("Starting Telegram scraper")
    config = load_social_config()

    tg_cfg = config.get("telegram", {})
    api_id = args.api_id or tg_cfg.get("api_id")
    api_hash = args.api_hash or tg_cfg.get("api_hash")

    if not api_id or not api_hash:
        logger.error("Telegram API credentials not found")
        print("[ERROR] Telegram API credentials not found. Use --api-id/--api-hash or configure in social config.")
        return

    channels_to_scrape = []

    if args.channels:
        lang = args.lang or 'unknown'
        for ch in args.channels:
            channels_to_scrape.append((ch, lang))
    else:
        config_channels = tg_cfg.get("channels", [])
        if not config_channels:
            logger.error("No channels found in config and none specified via --channels")
            print("[ERROR] No channels found. Use --channels or configure telegram.channels in social config.")
            return

        for entry in config_channels:
            channel_name = entry if isinstance(entry, str) else entry.get("name")
            if not channel_name:
                continue
            language = entry.get("language", "unknown") if isinstance(entry, dict) else "unknown"
            if args.lang and language != args.lang:
                continue
            channels_to_scrape.append((channel_name, language))

    if not channels_to_scrape:
        print(f"[INFO] No channels to scrape for language '{args.lang}'")
        return

    client = TelegramClient(
        "session_scraper",
        api_id,
        api_hash,
        connection_retries=10
    )

    print("[TG] Connecting to Telegram...")
    logger.info("Connecting to Telegram...")

    try:
        await client.start()
        print("[OK] Connected!")
        logger.info("Connected successfully")
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        print(f"[ERROR] Connection failed: {e}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")

    for channel_name, language in channels_to_scrape:
        try:
            await fetch_channel(
                client, channel_name, language,
                output_dir=OUTPUT_DIR, max_posts=args.max_posts
            )
        except Exception as e:
            logger.error(f"Error in channel {channel_name}: {e}")
            print(f"[ERROR] {channel_name}: {e}")

    await client.disconnect()
    logger.info("Telegram scraper finished")


if __name__ == "__main__":
    asyncio.run(main())