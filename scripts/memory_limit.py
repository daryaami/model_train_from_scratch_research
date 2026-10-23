"""Move this interpreter into the research_chemistry memory cgroup.

The cap lives in the user slice research-chemistry.slice (8 GiB, no swap).
Hitting it kills this process, not the rest of the machine. Set
RESEARCH_MEMORY_LIMIT=0 to skip the move.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

SLICE = "research-chemistry.slice"
BUSCTL = "/usr/bin/busctl"


def _cgroup() -> str:
    try:
        with open("/proc/self/cgroup", encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        return ""


def apply() -> None:
    if os.environ.get("RESEARCH_MEMORY_LIMIT", "1") == "0":
        return
    if SLICE in _cgroup():
        return
    if not os.path.exists(BUSCTL):
        sys.stderr.write("research_chemistry: нет busctl, лимит памяти не включён\n")
        return

    pid = str(os.getpid())
    result = subprocess.run(
        [
            BUSCTL,
            "--user",
            "call",
            "org.freedesktop.systemd1",
            "/org/freedesktop/systemd1",
            "org.freedesktop.systemd1.Manager",
            "StartTransientUnit",
            "ssa(sv)a(sa(sv))",
            f"research-chemistry-{pid}.scope",
            "fail",
            "3",
            "Slice",
            "s",
            SLICE,
            "PIDs",
            "au",
            "1",
            pid,
            "CollectMode",
            "s",
            "inactive-or-failed",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or str(result.returncode)
        sys.stderr.write(f"research_chemistry: не удалось включить лимит памяти ({detail})\n")
        return

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if SLICE in _cgroup():
            return
        time.sleep(0.05)
    sys.stderr.write("research_chemistry: лимит памяти не применился, процесс вне среза\n")
