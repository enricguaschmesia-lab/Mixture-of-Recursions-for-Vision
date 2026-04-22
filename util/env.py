"""Minimal `.env` loader.

Hand-rolled to avoid an extra dependency (`python-dotenv`) for ~15 lines of code.
Behaves like `set -a; source .env; set +a`, with one deliberate difference:
variables that are *already* set in the environment are NOT overridden. This
lets CI / Slurm / `WANDB_MODE=offline python pretrain.py` override `.env`
without editing the file.
"""
from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path | None = None) -> dict[str, str]:
    """Load KEY=VALUE pairs from `path` into `os.environ`.

    Args:
        path: Path to the `.env` file. Defaults to `<repo-root>/.env`, where
            `<repo-root>` is the parent of the `util/` directory.

    Returns:
        A dict of the variables that were actually set (i.e. excluding ones
        already present in the environment).
    """
    if path is None:
        path = Path(__file__).resolve().parent.parent / ".env"
    path = Path(path)

    if not path.is_file():
        return {}

    loaded: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip matching surrounding quotes, if any.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if not key or key in os.environ:
            continue
        os.environ[key] = value
        loaded[key] = value
    return loaded
