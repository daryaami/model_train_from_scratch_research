#!/usr/bin/env python3
"""Watch the cloud log and stop only the cloud runner on the first error."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "mineru_cloud.log"


def cloud_pids() -> list[int]:
    result = subprocess.run(
        ["pgrep", "-f", "ocr_research/run_mineru_cloud.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    pids = []
    for line in result.stdout.split():
        try:
            pid = int(line)
        except ValueError:
            continue
        if pid != os.getpid():
            pids.append(pid)
    return pids


def main() -> int:
    log = LOG.open("r", encoding="utf-8", errors="replace")
    log.seek(0, os.SEEK_END)
    print(f"watching {LOG} from the end", flush=True)
    while True:
        line = log.readline()
        if not line:
            time.sleep(1)
            continue
        text = line.lstrip()
        if not (text.startswith("ERROR ") or text.startswith("STOP ")):
            continue
        pids = cloud_pids()
        print(text.rstrip(), flush=True)
        if not pids:
            print("cloud runner already stopped", flush=True)
            return 0
        for pid in pids:
            print(f"stopping cloud pid={pid}", flush=True)
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                continue
        time.sleep(3)
        for pid in cloud_pids():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                continue
        print("stopped on cloud error", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
