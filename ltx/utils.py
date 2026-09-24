from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            return os.environ.get(name, default or "")
        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, list):
        return [expand_env(x) for x in value]
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    return value


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def stable_id(*parts: Any, length: int = 20) -> str:
    raw = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:length]


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True, default=str)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def run_capture(command: Iterable[str], cwd: Path | None = None, timeout: int = 30) -> str:
    proc = subprocess.run(
        list(command), cwd=str(cwd) if cwd else None, check=False,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout,
    )
    return proc.stdout.strip()


def shell_join(command: Iterable[str]) -> str:
    return shlex.join([str(x) for x in command])


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        os.environ.setdefault(key, value)


def load_runtime_env(root: Path) -> None:
    """Load an untracked environment file without ever persisting its values.

    An explicit LTX_ENV_FILE wins.  Otherwise the independent runpack only
    reads its own ignored .env.local/.env files; it never reaches into another
    checkout for credentials.
    """
    explicit = os.environ.get("LTX_ENV_FILE", "").strip()
    if explicit:
        path = Path(explicit).expanduser().resolve()
        load_dotenv(path)
        return
    load_dotenv(root / ".env.local")
    load_dotenv(root / ".env")
