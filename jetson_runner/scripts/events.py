from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from io import StringIO

from github_io import GitHubRepo


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_event(repo: GitHubRepo, event: str, status: str, message: str, duration_seconds: float = 0.0) -> None:
    row = {
        "timestamp": now_utc(),
        "event": event,
        "status": status,
        "duration_seconds": f"{duration_seconds:.2f}",
        "message": message,
    }

    path = "logs/pipeline_events.csv"
    existing = repo.get_file(path)
    rows: list[dict[str, str]] = []
    if existing:
        rows = list(csv.DictReader(StringIO(existing.decode("utf-8-sig"))))
    rows.append(row)
    rows = rows[-1000:]

    out = StringIO()
    writer = csv.DictWriter(out, fieldnames=["timestamp", "event", "status", "duration_seconds", "message"])
    writer.writeheader()
    writer.writerows(rows)
    repo.put_file(path, out.getvalue().encode("utf-8-sig"), f"Update pipeline event: {event}")
    repo.put_file("logs/latest_event.json", json.dumps(row, ensure_ascii=False, indent=2).encode("utf-8"), "Update latest pipeline event")

