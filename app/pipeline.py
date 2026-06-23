from __future__ import annotations

import base64
import json
import math
import os
import re
from datetime import datetime
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


ROOT = Path(__file__).resolve().parents[1]
SCRAPED_DIR = ROOT / "data" / "scraped"
SCRAPED_HISTORY_DIR = SCRAPED_DIR / "history"
PREDICTIONS_DIR = ROOT / "predictions"
PREDICTIONS_HISTORY_DIR = PREDICTIONS_DIR / "history"
MODELS_DIR = ROOT / "models" / "builded"
TOKEN_FILE = ROOT / ".confing"

OWNER = "gandpablo"
REPO = "VALENCIA_DATA_EDM"
BRANCH = "main"
LOCAL_TIMEZONE = "Europe/Madrid"
SCRAPER_URL = "https://www.valencia.es/valenciaalminut/"
SCRAPER_TABLE_ID = "tabla_dinamica"

POLLUTANTS = ["SO2", "NO2", "O3", "PM10", "PM2.5"]
SCRAPER_COLUMNS = ["µg/m3", "SO2", "NO2", "O3", "PM-10", "PM-2.5"]
SCRAPER_TO_MODEL = {"SO2": "SO2", "NO2": "NO2", "O3": "O3", "PM-10": "PM10", "PM-2.5": "PM2.5"}
MODEL_TO_SCRAPER = {value: key for key, value in SCRAPER_TO_MODEL.items()}

NAME_MAP = {
    "AVDA.FRANCIA": "València - Av. França",
    "BULEVARD SUD": "València - Bulevard Sud",
    "MOLÍ DEL SOL": "València - Molí del Sol",
    "PISTA DE SILLA": "València - Pista de Silla",
    "POLITÈCNIC": "València - Politècnic",
    "VIVERS": "València - Vivers",
    "VALÈNCIA CENTRE": "València - Centre",
    "OLIVERETA": "València Olivereta",
}


def ensure_dirs() -> None:
    for path in [SCRAPED_HISTORY_DIR, PREDICTIONS_HISTORY_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def local_now() -> datetime:
    return datetime.now(ZoneInfo(LOCAL_TIMEZONE))


def timestamp_name() -> str:
    return local_now().strftime("%Y-%m-%d_%H-%M.csv")


def read_token() -> str:
    token = os.environ.get("EDM_GITHUB_TOKEN")
    if token:
        return token
    if not TOKEN_FILE.exists():
        raise RuntimeError("No GitHub token found. Set EDM_GITHUB_TOKEN or create .confing.")
    text = TOKEN_FILE.read_text(encoding="utf-8")
    match = re.search(r"EDM_GITHUB_TOKEN\s*=\s*['\"]([^'\"]+)['\"]", text)
    if not match:
        raise RuntimeError("Could not read EDM_GITHUB_TOKEN from .confing")
    return match.group(1)


def github_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def github_url(path: str) -> str:
    return f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{path}"


def remote_file(token: str, path: str) -> dict | None:
    response = requests.get(
        github_url(path),
        headers=github_headers(token),
        params={"ref": BRANCH},
        timeout=30,
    )
    if response.status_code == 404:
        return None
    if response.status_code != 200:
        raise RuntimeError(f"Error checking {path}: {response.status_code} - {response.text}")
    return response.json()


def upload_file(token: str, path: str, content: bytes, message: str) -> None:
    remote = remote_file(token, path)
    payload = {
        "message": message,
        "content": base64.b64encode(content).decode("utf-8"),
        "branch": BRANCH,
    }
    if remote:
        payload["sha"] = remote["sha"]

    response = requests.put(
        github_url(path),
        headers=github_headers(token),
        json=payload,
        timeout=60,
    )
    if response.status_code not in (200, 201):
        raise RuntimeError(f"Error uploading {path}: {response.status_code} - {response.text}")


def scrape_current_table() -> pd.DataFrame:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--disable-notifications")
    options.add_argument("--ignore-certificate-errors")

    driver = webdriver.Chrome(options=options)
    try:
        driver.get(SCRAPER_URL)
        table_html = WebDriverWait(driver, 40).until(
            EC.presence_of_element_located((By.ID, SCRAPER_TABLE_ID))
        ).get_attribute("outerHTML")
    finally:
        driver.quit()

    df = pd.read_html(StringIO(table_html))[0]
    df = df[SCRAPER_COLUMNS].dropna(how="all")
    if df.empty or df["µg/m3"].dropna().empty:
        raise RuntimeError("Scrape returned no station rows; keeping previous data.")
    return df


def save_scrape(df: pd.DataFrame, filename: str) -> Path:
    ensure_dirs()
    if df.empty or df["µg/m3"].dropna().empty:
        raise RuntimeError("Refusing to save an empty scrape.")
    history_path = SCRAPED_HISTORY_DIR / filename
    latest_path = SCRAPED_DIR / "latest.csv"
    df.to_csv(history_path, index=False, encoding="utf-8-sig")
    df.to_csv(latest_path, index=False, encoding="utf-8-sig")
    update_index(SCRAPED_DIR / "index.json", "data/scraped/history", SCRAPED_HISTORY_DIR)
    return history_path


def update_index(index_path: Path, remote_prefix: str, folder: Path) -> None:
    items = [
        {"path": f"{remote_prefix}/{path.name}", "timestamp": path.stem}
        for path in sorted(folder.glob("*.csv"))
    ]
    index_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def load_scrape(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["station_model"] = df["µg/m3"].map(NAME_MAP).fillna(df["µg/m3"])
    for col in SCRAPER_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_scraped_history(current_path: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(SCRAPED_HISTORY_DIR.glob("*.csv")):
        ts = pd.to_datetime(path.stem, format="%Y-%m-%d_%H-%M", errors="coerce")
        if pd.isna(ts):
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        if df.empty:
            continue
        df["timestamp"] = ts
        rows.append(df)

    if not rows:
        raise RuntimeError("No scraped history available to build prediction features.")
    data = pd.concat(rows, ignore_index=True)
    data["station_model"] = data["µg/m3"].map(NAME_MAP).fillna(data["µg/m3"])
    for col in SCRAPER_COLUMNS[1:]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data.dropna(subset=["timestamp"]).sort_values("timestamp")


def load_model(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def feature_row(history: pd.DataFrame, station: str, model: dict) -> pd.DataFrame | None:
    station_history = history[history["station_model"] == station].sort_values("timestamp")
    if station_history.empty:
        return None

    latest = station_history.iloc[-1]
    current_time = latest["timestamp"]
    features: dict[str, float] = {}

    for pollutant in POLLUTANTS:
        scraper_col = MODEL_TO_SCRAPER[pollutant]
        features[f"{pollutant}_current"] = float(latest.get(scraper_col, np.nan))

    target = model["pollutant"]
    target_col = MODEL_TO_SCRAPER[target]
    current_target = float(latest.get(target_col, np.nan))

    for lag in [8, 24, 48, 168]:
        desired = current_time - pd.Timedelta(hours=lag)
        past = station_history[station_history["timestamp"] <= desired]
        features[f"{target}_lag_{lag}h"] = float(past.iloc[-1][target_col]) if not past.empty else current_target

    recent_24 = station_history[station_history["timestamp"] >= current_time - pd.Timedelta(hours=24)]
    recent_7d = station_history[station_history["timestamp"] >= current_time - pd.Timedelta(hours=168)]
    features[f"{target}_rolling_24h"] = float(recent_24[target_col].mean()) if not recent_24.empty else current_target
    features[f"{target}_rolling_7d"] = float(recent_7d[target_col].mean()) if not recent_7d.empty else current_target

    hour = current_time.hour
    dow = current_time.dayofweek
    month = current_time.month
    features["hour_sin"] = math.sin(2 * math.pi * hour / 24)
    features["hour_cos"] = math.cos(2 * math.pi * hour / 24)
    features["dow_sin"] = math.sin(2 * math.pi * dow / 7)
    features["dow_cos"] = math.cos(2 * math.pi * dow / 7)
    features["month_sin"] = math.sin(2 * math.pi * month / 12)
    features["month_cos"] = math.cos(2 * math.pi * month / 12)

    row = pd.DataFrame([features])
    needed = model["feature_names"]
    if row[needed].isna().any(axis=None):
        return None
    return row


def predict_model(model: dict, row: pd.DataFrame) -> float:
    cols = model["feature_names"]
    means = pd.Series(model["feature_means"])[cols]
    stds = pd.Series(model["feature_stds"])[cols]
    scaled = ((row[cols] - means) / stds).to_numpy(dtype=float)
    return float(model["intercept"] + scaled @ np.array(model["coefficients"], dtype=float))


def make_predictions(current_path: Path, filename: str) -> Path:
    ensure_dirs()
    current = load_scrape(current_path)
    history = load_scraped_history(current_path)
    registry = json.loads((MODELS_DIR / "registry.json").read_text(encoding="utf-8"))

    predictions = current[["µg/m3"]].copy()
    for col in SCRAPER_COLUMNS[1:]:
        predictions[col] = np.nan

    for entry in registry:
        if int(entry.get("horizon_hours", 0)) != 8:
            continue
        model = load_model(entry["model_path"])
        station = model["station"]
        pollutant = model["pollutant"]
        scraper_col = MODEL_TO_SCRAPER[pollutant]
        row = feature_row(history, station, model)
        if row is None:
            continue
        value = max(0.0, predict_model(model, row))
        mask = current["station_model"] == station
        predictions.loc[mask, scraper_col] = round(value, 1)

    history_path = PREDICTIONS_HISTORY_DIR / filename
    latest_path = PREDICTIONS_DIR / "latest.csv"
    predictions.to_csv(history_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(latest_path, index=False, encoding="utf-8-sig")
    update_index(PREDICTIONS_DIR / "index.json", "predictions/history", PREDICTIONS_HISTORY_DIR)
    return history_path


def upload_outputs(scrape_path: Path, prediction_path: Path) -> None:
    token = read_token()
    files = [
        ("data/scraped/latest.csv", SCRAPED_DIR / "latest.csv"),
        (f"data/scraped/history/{scrape_path.name}", scrape_path),
        ("data/scraped/index.json", SCRAPED_DIR / "index.json"),
        ("predictions/latest.csv", PREDICTIONS_DIR / "latest.csv"),
        (f"predictions/history/{prediction_path.name}", prediction_path),
        ("predictions/index.json", PREDICTIONS_DIR / "index.json"),
    ]
    for remote_path, local_path in files:
        upload_file(token, remote_path, local_path.read_bytes(), f"Update {remote_path}")


def run_manual_pipeline() -> dict[str, str]:
    ensure_dirs()
    filename = timestamp_name()
    df = scrape_current_table()
    scrape_path = save_scrape(df, filename)
    prediction_path = make_predictions(scrape_path, filename)
    upload_outputs(scrape_path, prediction_path)
    return {
        "scrape_file": scrape_path.name,
        "prediction_file": prediction_path.name,
        "scrape_path": str(scrape_path),
        "prediction_path": str(prediction_path),
    }
