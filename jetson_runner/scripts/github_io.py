from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from config import load_env, require_env


@dataclass
class GitHubRepo:
    owner: str
    repo: str
    branch: str
    token: str

    @classmethod
    def from_env(cls) -> "GitHubRepo":
        load_env()
        return cls(
            owner=require_env("GITHUB_OWNER"),
            repo=require_env("GITHUB_REPO"),
            branch=require_env("GITHUB_BRANCH"),
            token=require_env("GITHUB_TOKEN"),
        )

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def contents_url(self, path: str) -> str:
        return f"https://api.github.com/repos/{self.owner}/{self.repo}/contents/{path}"

    def get_file(self, path: str) -> bytes | None:
        response = requests.get(
            self.contents_url(path),
            headers=self.headers,
            params={"ref": self.branch},
            timeout=30,
        )
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            raise RuntimeError(f"Error reading {path}: {response.status_code} - {response.text}")
        content = response.json()["content"]
        return base64.b64decode(content)

    def get_json(self, path: str, default: Any) -> Any:
        raw = self.get_file(path)
        if raw is None:
            return default
        return json.loads(raw.decode("utf-8"))

    def put_file(self, path: str, content: bytes, message: str) -> None:
        sha = None
        current = requests.get(
            self.contents_url(path),
            headers=self.headers,
            params={"ref": self.branch},
            timeout=30,
        )
        if current.status_code == 200:
            sha = current.json()["sha"]
        elif current.status_code != 404:
            raise RuntimeError(f"Error checking {path}: {current.status_code} - {current.text}")

        payload: dict[str, Any] = {
            "message": message,
            "content": base64.b64encode(content).decode("utf-8"),
            "branch": self.branch,
        }
        if sha:
            payload["sha"] = sha

        response = requests.put(
            self.contents_url(path),
            headers=self.headers,
            json=payload,
            timeout=30,
        )
        if response.status_code not in (200, 201):
            raise RuntimeError(f"Error writing {path}: {response.status_code} - {response.text}")

    def put_json(self, path: str, data: Any, message: str) -> None:
        content = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.put_file(path, content, message)

    def list_dir(self, path: str) -> list[dict[str, Any]]:
        response = requests.get(
            self.contents_url(path),
            headers=self.headers,
            params={"ref": self.branch},
            timeout=30,
        )
        if response.status_code == 404:
            return []
        if response.status_code != 200:
            raise RuntimeError(f"Error listing {path}: {response.status_code} - {response.text}")
        data = response.json()
        return data if isinstance(data, list) else [data]

    def download_prefix(self, github_dir: str, local_dir: Path) -> list[Path]:
        local_dir.mkdir(parents=True, exist_ok=True)
        downloaded: list[Path] = []
        for item in self.list_dir(github_dir):
            item_path = item["path"]
            if item["type"] == "dir":
                downloaded.extend(self.download_prefix(item_path, local_dir / item["name"]))
                continue
            raw = self.get_file(item_path)
            if raw is None:
                continue
            local_path = local_dir / item["name"]
            local_path.write_bytes(raw)
            downloaded.append(local_path)
        return downloaded

