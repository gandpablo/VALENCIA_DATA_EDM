from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from config import ROOT, TMP_DIR, ensure_dirs, load_env
from events import append_event
from github_io import GitHubRepo


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run(command: list[str]) -> str:
    result = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip(), file=sys.stderr)
    return result.stdout.strip()


def reset_tmp_child(name: str) -> Path:
    path = TMP_DIR / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def update_index(repo: GitHubRepo, index_path: str, item: dict, limit: int = 1000) -> None:
    index = repo.get_json(index_path, default=[])
    index = [entry for entry in index if entry.get("path") != item.get("path")]
    index.append(item)
    repo.put_json(index_path, index[-limit:], f"Update {index_path}")


def upload_scrape(repo: GitHubRepo, scrape_file: Path) -> None:
    github_history = f"data/scraped/history/{scrape_file.name}"
    repo.put_file(github_history, scrape_file.read_bytes(), "Add scraped pollution snapshot")
    repo.put_file("data/scraped/latest.csv", scrape_file.read_bytes(), "Update latest scraped pollution snapshot")
    update_index(repo, "data/scraped/index.json", {"path": github_history, "timestamp": scrape_file.stem})

    state = repo.get_json("state/pipeline_state.json", default={})
    state["last_scrape_utc"] = now_utc()
    state["last_scrape_path"] = github_history
    repo.put_json("state/pipeline_state.json", state, "Update pipeline state after scrape")


def upload_prediction(repo: GitHubRepo, prediction_file: Path) -> None:
    github_history = f"predictions/history/{prediction_file.name}"
    repo.put_file(github_history, prediction_file.read_bytes(), "Add prediction snapshot")
    repo.put_file("predictions/latest.csv", prediction_file.read_bytes(), "Update latest predictions")
    update_index(repo, "predictions/index.json", {"path": github_history, "timestamp": prediction_file.stem})

    state = repo.get_json("state/pipeline_state.json", default={})
    state["last_prediction_utc"] = now_utc()
    state["last_prediction_path"] = github_history
    repo.put_json("state/pipeline_state.json", state, "Update pipeline state after prediction")


def scrape_pipeline(repo: GitHubRepo) -> None:
    start = time.time()
    scrape_dir = reset_tmp_child("scraped")
    pred_dir = reset_tmp_child("predictions")

    output = run([sys.executable, "scraper/scrape_current.py", "--out-dir", str(scrape_dir)])
    scrape_file = Path(output.splitlines()[-1])
    if not scrape_file.exists():
        raise RuntimeError(f"Scrape file not found: {scrape_file}")

    upload_scrape(repo, scrape_file)

    pred_output = run([
        sys.executable,
        "scripts/predict_latest.py",
        "--latest-scrape",
        str(scrape_file),
        "--out-dir",
        str(pred_dir),
    ])
    prediction_file = Path(pred_output.splitlines()[-1])
    if not prediction_file.exists():
        raise RuntimeError(f"Prediction file not found: {prediction_file}")

    upload_prediction(repo, prediction_file)
    append_event(repo, "scrape_pipeline", "ok", f"uploaded {scrape_file.name} and predictions", time.time() - start)


def retrain_pipeline(repo: GitHubRepo, force: bool) -> None:
    start = time.time()
    command = [sys.executable, "scripts/retrain_models.py"]
    if force:
        command.append("--force")
    output = run(command)
    status = "skipped" if output.strip().startswith("skip:") else "ok"
    append_event(repo, "retrain_pipeline", status, output[-240:] if output else status, time.time() - start)


def cleanup_tmp() -> None:
    if TMP_DIR.exists():
        for child in TMP_DIR.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    ensure_dirs()


def main() -> None:
    load_env()
    ensure_dirs()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["scrape", "retrain"], required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-tmp", action="store_true")
    args = parser.parse_args()

    repo = GitHubRepo.from_env()
    try:
        if args.mode == "scrape":
            scrape_pipeline(repo)
        else:
            retrain_pipeline(repo, args.force)
    except Exception as exc:
        append_event(repo, f"{args.mode}_pipeline", "error", str(exc)[:240], 0.0)
        raise
    finally:
        if not args.keep_tmp:
            cleanup_tmp()


if __name__ == "__main__":
    main()

