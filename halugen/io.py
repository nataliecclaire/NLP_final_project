"""Crash-safe JSONL helpers for resumable runs on preemptible machines."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


def repair_truncated_tail(path: Path) -> int:
    """If the file's last line was cut off mid-write, remove it. Returns bytes removed."""
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with open(path, "rb+") as f:
        data = f.read()
        if data.endswith(b"\n"):
            return 0
        cut = data.rfind(b"\n") + 1  # 0 if no complete line at all
        f.truncate(cut)
        return len(data) - cut


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    repair_truncated_tail(path)
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


class JsonlAppender:
    """Append one record per line, flush + fsync each one.

    Google Drive mounted in Colab only uploads a file after it is *closed*: a file kept open
    for hours is lost if the VM dies, even though every write was fsynced. So the file is also
    closed and reopened every `reopen_every_s` seconds, which bounds the loss to that window.
    """

    def __init__(self, path: Path, reopen_every_s: float = 60.0):
        path.parent.mkdir(parents=True, exist_ok=True)
        repair_truncated_tail(path)
        self.path, self.reopen_every_s = path, reopen_every_s
        self.f = open(path, "a")
        self.t_open = time.monotonic()

    def write(self, rec: dict) -> None:
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.f.flush()
        os.fsync(self.f.fileno())
        if time.monotonic() - self.t_open >= self.reopen_every_s:
            self.f.close()
            self.f = open(self.path, "a")
            self.t_open = time.monotonic()

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def write_jsonl_atomic(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
