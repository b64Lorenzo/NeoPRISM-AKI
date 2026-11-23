"""Minimal logging compatibility helper."""
from datetime import datetime, timezone

def log(level: str, component: str, message: str) -> None:
    print(f'{datetime.now(timezone.utc).isoformat()}|{level}|{component}|{message}')
