import os
import requests


def send(text):
    token, chat = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("---- (Telegram non configurato, stampo a video) ----\n" + text)
        return
    for i in range(0, len(text), 4000):
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          data={"chat_id": chat, "text": text[i:i + 4000]}, timeout=30)
        r.raise_for_status()


def get_updates(offset=None):
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        return []
    r = requests.get(f"https://api.telegram.org/bot{token}/getUpdates",
                     params={"offset": offset, "timeout": 0}, timeout=30)
    r.raise_for_status()
    return r.json().get("result", [])
