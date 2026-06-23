from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TMP_DIR = ROOT / "tmp"
LOGS_DIR = ROOT / "logs"


def load_env(path: Path | None = None) -> dict[str, str]:
    env_path = path or ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    return dict(os.environ)


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def ensure_dirs() -> None:
    for path in [
        TMP_DIR,
        TMP_DIR / "scraped",
        TMP_DIR / "predictions",
        TMP_DIR / "retrain",
        LOGS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)

