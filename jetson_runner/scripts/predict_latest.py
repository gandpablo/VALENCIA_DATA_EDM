from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from config import TMP_DIR, ensure_dirs, load_env
from github_io import GitHubRepo


SCRAPER_TO_MODEL = {"PM-10": "PM10", "PM-2.5": "PM2.5", "SO2": "SO2", "NO2": "NO2", "O3": "O3"}
MODEL_TO_SCRAPER = {v: k for k, v in SCRAPER_TO_MODEL.items()}
POLLUTANTS = ["SO2", "NO2", "O3", "PM10", "PM2.5"]
SCRAPER_COLUMNS = ["µg/m3", "SO2", "NO2", "O3", "PM-10", "PM-2.5"]

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


def parse_snapshot_time(name: str) -> pd.Timestamp:
    stem = Path(name).stem
    return pd.to_datetime(stem, format="%Y-%m-%d_%H-%M", errors="coerce")


def read_scrape_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["station_model"] = df["µg/m3"].map(NAME_MAP).fillna(df["µg/m3"])
    for col in SCRAPER_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_history(repo: GitHubRepo, local_latest: Path) -> pd.DataFrame:
    records = []
    index = repo.get_json("data/scraped/index.json", default=[])
    for item in index[-40:]:
        path = item.get("path")
        if not path:
            continue
        raw = repo.get_file(path)
        if raw is None:
            continue
        ts = parse_snapshot_time(Path(path).name)
        df = pd.read_csv(StringIO(raw.decode("utf-8-sig")))
        df["timestamp"] = ts
        records.append(df)

    latest = read_scrape_csv(local_latest).drop(columns=["station_model"])
    latest["timestamp"] = parse_snapshot_time(local_latest.name if local_latest.name != "latest.csv" else datetime.now().strftime("%Y-%m-%d_%H-%M.csv"))
    records.append(latest)

    if not records:
        return pd.DataFrame()

    data = pd.concat(records, ignore_index=True)
    data["station_model"] = data["µg/m3"].map(NAME_MAP).fillna(data["µg/m3"])
    for col in SCRAPER_COLUMNS[1:]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["timestamp"])
    return data.sort_values("timestamp")


def model_feature_row(history: pd.DataFrame, station: str, model: dict) -> pd.DataFrame | None:
    station_history = history[history["station_model"] == station].copy()
    if station_history.empty:
        return None

    station_history = station_history.sort_values("timestamp")
    latest = station_history.iloc[-1]
    current_time = latest["timestamp"]
    features: dict[str, float] = {}

    for pollutant in POLLUTANTS:
        scraper_col = MODEL_TO_SCRAPER[pollutant]
        if scraper_col in station_history.columns:
            features[f"{pollutant}_current"] = float(latest.get(scraper_col, np.nan))

    target = model["pollutant"]
    target_col = MODEL_TO_SCRAPER[target]
    for lag in [8, 24, 48, 168]:
        desired = current_time - pd.Timedelta(hours=lag)
        past = station_history[station_history["timestamp"] <= desired]
        features[f"{target}_lag_{lag}h"] = float(past.iloc[-1][target_col]) if not past.empty else np.nan

    recent_24 = station_history[station_history["timestamp"] >= current_time - pd.Timedelta(hours=24)]
    recent_7d = station_history[station_history["timestamp"] >= current_time - pd.Timedelta(hours=168)]
    features[f"{target}_rolling_24h"] = float(recent_24[target_col].mean()) if not recent_24.empty else np.nan
    features[f"{target}_rolling_7d"] = float(recent_7d[target_col].mean()) if not recent_7d.empty else np.nan

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
    if row[model["feature_names"]].isna().any(axis=None):
        return None
    return row


def predict_model(model: dict, row: pd.DataFrame) -> float:
    cols = model["feature_names"]
    means = pd.Series(model["feature_means"])[cols]
    stds = pd.Series(model["feature_stds"])[cols]
    scaled = ((row[cols] - means) / stds).to_numpy(dtype=float)
    return float(model["intercept"] + scaled @ np.array(model["coefficients"], dtype=float))


def main() -> None:
    load_env()
    ensure_dirs()
    parser = argparse.ArgumentParser()
    parser.add_argument("--latest-scrape", required=True)
    parser.add_argument("--out-dir", default=str(TMP_DIR / "predictions"))
    args = parser.parse_args()

    repo = GitHubRepo.from_env()
    latest_path = Path(args.latest_scrape)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    current = read_scrape_csv(latest_path)
    history = load_history(repo, latest_path)
    registry = repo.get_json("models/builded/registry.json", default=[])

    predictions = current[["µg/m3"]].copy()
    for col in SCRAPER_COLUMNS[1:]:
        predictions[col] = np.nan

    for entry in registry:
        if int(entry.get("horizon_hours", 0)) != 8:
            continue
        model_path = entry["model_path"]
        raw_model = repo.get_file(model_path)
        if raw_model is None:
            continue
        model = json.loads(raw_model.decode("utf-8"))
        station = model["station"]
        pollutant = model["pollutant"]
        scraper_col = MODEL_TO_SCRAPER[pollutant]
        feature_row = model_feature_row(history, station, model)
        if feature_row is None:
            continue
        value = max(0.0, predict_model(model, feature_row))
        mask = current["station_model"] == station
        predictions.loc[mask, scraper_col] = round(value, 1)

    timestamp_name = latest_path.name if latest_path.name != "latest.csv" else datetime.now().strftime("%Y-%m-%d_%H-%M.csv")
    history_path = out_dir / timestamp_name
    latest_out = out_dir / "latest.csv"
    predictions.to_csv(history_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(latest_out, index=False, encoding="utf-8-sig")
    print(history_path)


if __name__ == "__main__":
    main()

