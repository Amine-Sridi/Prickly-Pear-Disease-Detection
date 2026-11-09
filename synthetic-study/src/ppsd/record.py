"""Small I/O helpers: atomic JSON writes, hashes, timestamps."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def git_blob_sha(data: bytes) -> str:
    """Same hash git uses for file contents (`git hash-object`)."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _default(o: Any):
    # numpy scalars / arrays, Paths, sets
    if hasattr(o, "item") and callable(o.item):
        try:
            return o.item()
        except (ValueError, TypeError):
            pass
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, (Path, os.PathLike)):
        return str(o)
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_json_atomic(path: str | os.PathLike, obj: Any) -> Path:
    """Write JSON via a temp file + rename, so a disconnect never leaves half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_default, sort_keys=False), encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_json(path: str | os.PathLike) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
