# social_scrapers/telegram_scraper.py
import asyncio
import hashlib
import json
import os
import sys
import yaml
from datetime import datetime
from telethon import TelegramClient
from tqdm import tqdm


def sha256_hex(text: str, trunc: int = 32) -> str:
    h = hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()
    return h[:trunc] if trunc else h


async def fetch_channel(client, channel, output_dir):
    """Собирает посты из Telegram-канала."""
    
    print(f"📱 Подсчёт сообщений в канале: {channel}...")
    
    total = 0
    async for _ in client.iter_messages(channel, limit=None):
        total += 1
    
    print(f"   Найдено сообщений: {total:,}")
    
    messages = []
    count = 0
    
    with tqdm(total=total, desc=f"📱 {channel}", unit="пост", dynamic_ncols=True) as pbar:
        async for msg in client.iter_messages(channel, limit=None):
            if msg.text:
                content = msg.text
                msg_date = msg.date.replace(tzinfo=None) if msg.date else datetime.now()
                
                post_data = {
                    "url": f"https://t.me/{channel}/{msg.id}",
                    "title": content.split("\n")[0][:120] if content else "",
                    "content": content,
                    "excerpt": content[:260] + "..." if len(content) > 260 else content,
                    "date": msg_date.isoformat(),
                    "author": getattr(msg.sender, 'username', None) or str(msg.sender_id) if msg.sender else str(msg.sender_id) if msg.sender_id else None,
                    "category": None,
                    "time": msg_date.strftime("%H:%M"),
                    "site": f"t.me/{channel}",
                    "hash": sha256_hex(content + str(msg.id), trunc=32),
                    "image_url": None,
                    "language": "tt",
                    "scraped_at": datetime.now().isoformat(),
                    "source_type": "telegram",
                    "page_type": "article",
                    "channel": channel,
                    "post_id": msg.id,
                    "views": getattr(msg, 'views', 0),
                    "forwards": getattr(msg, 'forwards', 0),
                }
                
                messages.append(post_data)
                pbar.update(1)
                
                count += 1
                if count % 100 == 0:
                    await asyncio.sleep(0.5)
    
    if messages:
        filename = os.path.join(output_dir, f"telegram_{channel}.jsonl")
        with open(filename, "w", encoding="utf-8") as f:
            for m in messages:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        print(f"✅ {channel}: {len(messages)} сообщений -> {filename}")


async def main():
    if len(sys.argv) < 2:
        config_path = "social_config.yaml"
    else:
        config_path = sys.argv[1]

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tg_cfg = cfg.get("telegram", {})
    
    # MTProto прокси (закомментируй если не нужен)
    proxy = ("mtproto.telegram.org.il", 443, "ee4737edd06f4cdb6f1425b4d2e0d0ea")
    
    client = TelegramClient(
        "session_scraper", 
        tg_cfg["api_id"], 
        tg_cfg["api_hash"],
        # proxy=proxy,  # ← раскомментируй когда будет работать прокси/VPN
        connection_retries=10
    )
    
    print("🔌 Подключаемся к Telegram...")
    await client.start()
    print("✅ Подключено!")

    output_dir = cfg.get("output_dir", "social_jsonl")
    os.makedirs(output_dir, exist_ok=True)

    for channel in tg_cfg.get("channels", []):
        try:
            await fetch_channel(client, channel, output_dir)
        except Exception as e:
            print(f"❌ Ошибка в канале {channel}: {e}")

    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

# python social_scrapers\telegram_scraper.py social_config.yaml