"""Non-fatal ntfy notifications for long-running experiments."""

from __future__ import annotations

import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass


def _ascii_title(value: str) -> str:
    cleaned = value.encode("ascii", "replace").decode("ascii").strip()
    return cleaned[:120] or "DTI notification"


@dataclass
class NtfyNotifier:
    topic: str | None = None
    server_url: str = "https://ntfy.sh"
    timeout: float = 10.0

    def __post_init__(self) -> None:
        self.topic = (self.topic or os.environ.get("NTFY_TOPIC") or "").strip() or None
        self.server_url = (
            os.environ.get("NTFY_SERVER_URL") or self.server_url
        ).rstrip("/")

    @property
    def enabled(self) -> bool:
        return self.topic is not None

    def send(self, title: str, content: str = "", priority: str = "default") -> bool:
        """Send one notification; network failures never abort training."""
        if not self.enabled:
            return False
        body = content
        if body and "host:" not in body.lower():
            body = f"{body.rstrip()}\nhost: {socket.gethostname()}"
        request = urllib.request.Request(
            f"{self.server_url}/{self.topic}",
            data=body.encode("utf-8"),
            method="POST",
            headers={
                "Title": _ascii_title(title),
                "Priority": priority,
                "Content-Type": "text/plain; charset=utf-8",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                ok = 200 <= int(response.status) < 300
            print(f"[ntfy] {'sent' if ok else 'failed'}: {_ascii_title(title)}", flush=True)
            return ok
        except (OSError, urllib.error.URLError, ValueError) as exc:
            print(f"[ntfy] non-fatal send failure: {exc}", flush=True)
            return False

