#!/usr/bin/env python3
"""Run a training command and notify when it dies from a native signal."""

from __future__ import annotations

import argparse
import faulthandler
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.notify import NtfyNotifier  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default=os.environ.get("NTFY_TOPIC"))
    parser.add_argument("--server-url", default=os.environ.get("NTFY_SERVER_URL", "https://ntfy.sh"))
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="command after --, e.g. -- python -u main.py Davis ...")
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if not args.topic:
        parser.error("--topic or NTFY_TOPIC is required")
    return args


def signal_name(returncode: int) -> str | None:
    signal_number = -returncode if returncode < 0 else (
        returncode - 128 if 128 <= returncode <= 255 else None
    )
    if signal_number is None:
        return None
    try:
        return signal.Signals(signal_number).name
    except ValueError:
        return f"signal {signal_number}"


def main(argv: list[str] | None = None) -> int:
    faulthandler.enable(all_threads=True)
    args = parse_args(argv)
    notifier = NtfyNotifier(args.topic, server_url=args.server_url)
    environment = os.environ.copy()
    environment["NTFY_TOPIC"] = args.topic
    environment["NTFY_SERVER_URL"] = args.server_url
    environment["PYTHONFAULTHANDLER"] = "1"
    command_text = shlex.join(args.command)
    print(f"[watchdog] starting: {command_text}", flush=True)
    completed = subprocess.run(args.command, env=environment)
    native_signal = signal_name(completed.returncode)
    if native_signal is not None:
        notifier.send(
            "DTI Native Crash",
            "Training process stopped because of a native signal.\n"
            f"signal: {native_signal}\n"
            f"return code: {completed.returncode}\n"
            f"command: {command_text}",
            priority="high",
        )
    return 128 + (-completed.returncode) if completed.returncode < 0 else completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
