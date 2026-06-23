from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import TMP_DIR, ensure_dirs, load_env
from github_io import GitHubRepo


POLLUTANTS = ["SO2", "NO2", "O3", "PM10", "PM2.5"]
SCRAPER_TO_MODEL = {"PM-10": "PM10", "PM-2.5": "PM2.5", "SO2": "SO2", "NO2": "NO2", "O3": "O3"}
TARGET_WINDOWS = {"SO2": 1, "NO2": 1, "O3": 8, "PM10": 24, "PM2.5": 24}
HORIZONS = [8, 24]
LAGS = [8, 24, 48, 168]
RIDGE_ALPHA = 5.0
TRAIN_WINDOW_DAYS = 365 * 3
TEST_WINDOW_DAYS = 31
MIN_TRAIN_ROWS = 500
MIN_TEST_ROWS = 100

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


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def should_retrain(repo: GitHubRepo, force: bool) -> tuple[bool, str, dict]:
    if force:
        return True, "forced", repo.get_json("state/pipeline_state.json", default={})

    env = load_env()
    state = repo.get_json("state/pipeline_state.json", default={})
    every_days = int(env.get("RETRAIN_EVERY_DAYS", "5"))
    min_new = int(env.get("MIN_NEW_SCRAPES_FOR_RETRAIN", "15"))
    scrape_index = repo.get_json("data/scraped/index.json", default=[])
    current_count = len(scrape_index)
    previous_count = int(state.get("last_retrain_scrape_count", 0))

    if current_count - previous_count < min_new:
        return False, f"only {current_count - previous_count} new scrapes", state

    last_raw = state.get("last_retrain_utc")
    if last_raw:
        last = datetime.fromisoformat(last_raw.replace("Z", "+00:00"))
        if (now_utc() - last).days < every_days:
            return False, f"last retrain less than {every_days} days ago", state

    return True, "scheduled", state


def slugify(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").lower()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value


def station_from_file(path: Path) -> str:
    return unicodedata.normalize("NFC", path.stem.replace("filtrado_", ""))


def numeric_column(df: pd.DataFrame, name: str) -> pd.Series | None:
    matches = [col for col in df.columns if str(col).strip() == name]
    if not matches:
        return None
    return pd.to_numeric(df[matches[0]], errors="coerce")


def load_historical(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8-sig")
    timestamps = pd.to_datetime(raw["FECHA"], errors="coerce") + pd.to_timedelta(
        pd.to_numeric(raw["HORA"], errors="coerce"), unit="h"
    )
    frame = pd.DataFrame(index=timestamps)
    frame.index.name = "timestamp"
    for pollutant in POLLUTANTS:
        values = numeric_column(raw, pollutant)
        if values is not None:
            frame[pollutant] = values.to_numpy()
    frame = frame[~frame.index.isna()]
    frame = frame[~frame.index.duplicated(keep="first")].sort_index()
    return frame.asfreq("h")


def parse_scrape_time(path: Path) -> pd.Timestamp:
    return pd.to_datetime(path.stem, format="%Y-%m-%d_%H-%M", errors="coerce")


def load_scraped_by_station(scraped_dir: Path) -> dict[str, pd.DataFrame]:
    rows = []
    for path in sorted(scraped_dir.glob("*.csv")):
        ts = parse_scrape_time(path)
        if pd.isna(ts):
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        df["timestamp"] = ts
        rows.append(df)
    if not rows:
        return {}

    data = pd.concat(rows, ignore_index=True)
    data["station"] = data["µg/m3"].map(NAME_MAP)
    data = data.dropna(subset=["station"])
    for scraper_col, model_col in SCRAPER_TO_MODEL.items():
        data[model_col] = pd.to_numeric(data.get(scraper_col), errors="coerce")

    out = {}
    for station, group in data.groupby("station"):
        frame = group.set_index("timestamp")[POLLUTANTS].sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        out[station] = frame
    return out


def scraper_targets(hourly: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=hourly.index)
    for pollutant in POLLUTANTS:
        if pollutant not in hourly:
            continue
        window = TARGET_WINDOWS[pollutant]
        min_periods = 1 if window == 1 else math.ceil(window * 0.75)
        out[pollutant] = hourly[pollutant].rolling(window, min_periods=min_periods).mean()
    return out


def make_features(targets: pd.DataFrame, pollutant: str, horizon: int) -> tuple[pd.DataFrame, pd.Series]:
    x = pd.DataFrame(index=targets.index)
    for col in targets.columns:
        x[f"{col}_current"] = targets[col]
    base = targets[pollutant]
    for lag in LAGS:
        x[f"{pollutant}_lag_{lag}h"] = base.shift(lag)
    x[f"{pollutant}_rolling_24h"] = base.rolling(24, min_periods=18).mean()
    x[f"{pollutant}_rolling_7d"] = base.rolling(168, min_periods=126).mean()
    x["hour_sin"] = np.sin(2 * np.pi * x.index.hour / 24)
    x["hour_cos"] = np.cos(2 * np.pi * x.index.hour / 24)
    x["dow_sin"] = np.sin(2 * np.pi * x.index.dayofweek / 7)
    x["dow_cos"] = np.cos(2 * np.pi * x.index.dayofweek / 7)
    x["month_sin"] = np.sin(2 * np.pi * x.index.month / 12)
    x["month_cos"] = np.cos(2 * np.pi * x.index.month / 12)
    y = base.shift(-horizon).rename("target")
    dataset = pd.concat([x, y], axis=1).dropna()
    return dataset.drop(columns="target"), dataset["target"]


def fit_ridge(x: pd.DataFrame, y: pd.Series) -> dict:
    mean = x.mean()
    std = x.std().replace(0, 1)
    xs = ((x - mean) / std).to_numpy(dtype=float)
    yd = y.to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(xs)), xs])
    penalty = np.eye(design.shape[1]) * RIDGE_ALPHA
    penalty[0, 0] = 0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ yd)
    return {
        "intercept": float(beta[0]),
        "coefficients": [float(v) for v in beta[1:]],
        "feature_means": {k: float(v) for k, v in mean.items()},
        "feature_stds": {k: float(v) for k, v in std.items()},
    }


def predict(model: dict, x: pd.DataFrame) -> np.ndarray:
    cols = model["feature_names"]
    mean = pd.Series(model["feature_means"])[cols]
    std = pd.Series(model["feature_stds"])[cols]
    xs = ((x[cols] - mean) / std).to_numpy(dtype=float)
    return model["intercept"] + xs @ np.array(model["coefficients"], dtype=float)


def metric_block(y: pd.Series, pred: np.ndarray) -> dict:
    actual = y.to_numpy(dtype=float)
    mae = float(np.mean(np.abs(actual - pred)))
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    denom = float(np.sum((actual - actual.mean()) ** 2))
    r2 = float("nan") if denom == 0 else float(1 - np.sum((actual - pred) ** 2) / denom)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def train_all(historical_dir: Path, scraped_dir: Path, out_dir: Path) -> tuple[list[dict], pd.DataFrame]:
    scraped = load_scraped_by_station(scraped_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trained_at = now_utc().isoformat(timespec="seconds")
    registry = []
    metrics = []

    for path in sorted(historical_dir.glob("*.csv")):
        station = station_from_file(path)
        hourly = load_historical(path)
        if station in scraped:
            hourly = pd.concat([hourly, scraped[station]], axis=0).sort_index()
            hourly = hourly[~hourly.index.duplicated(keep="last")]
        targets = scraper_targets(hourly)
        last = targets.index.max()
        train_min = last - pd.Timedelta(days=TRAIN_WINDOW_DAYS)
        test_min = last - pd.Timedelta(days=TEST_WINDOW_DAYS)

        for pollutant in POLLUTANTS:
            if pollutant not in targets or targets[pollutant].dropna().empty:
                continue
            for horizon in HORIZONS:
                x, y = make_features(targets, pollutant, horizon)
                train_mask = (x.index >= train_min) & (x.index < test_min)
                test_mask = x.index >= test_min
                x_train, y_train = x.loc[train_mask], y.loc[train_mask]
                x_test, y_test = x.loc[test_mask], y.loc[test_mask]
                if len(x_train) < MIN_TRAIN_ROWS or len(x_test) < MIN_TEST_ROWS:
                    continue

                model = fit_ridge(x_train, y_train)
                model.update({
                    "model_type": "ridge_numpy",
                    "station": station,
                    "pollutant": pollutant,
                    "horizon_hours": horizon,
                    "target_window_hours": TARGET_WINDOWS[pollutant],
                    "alpha": RIDGE_ALPHA,
                    "feature_names": list(x_train.columns),
                    "trained_at_utc": trained_at,
                    "source_file": f"data/time/{path.name}",
                    "train_rows": int(len(x_train)),
                    "test_rows": int(len(x_test)),
                })
                pred = predict(model, x_test)
                baseline = x_test[f"{pollutant}_current"].to_numpy(dtype=float)
                model["metrics"] = metric_block(y_test, pred)
                model["baseline_metrics"] = metric_block(y_test, baseline)
                model["beats_baseline_mae"] = bool(model["metrics"]["mae"] <= model["baseline_metrics"]["mae"])

                filename = f"model__{slugify(station)}__{slugify(pollutant)}__h{horizon:02d}.json"
                (out_dir / filename).write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
                model_path = f"models/builded/{filename}"
                registry.append({
                    "station": station,
                    "pollutant": pollutant,
                    "horizon_hours": horizon,
                    "model_path": model_path,
                    "mae": model["metrics"]["mae"],
                    "baseline_mae": model["baseline_metrics"]["mae"],
                    "beats_baseline_mae": model["beats_baseline_mae"],
                    "trained_at_utc": trained_at,
                })
                metrics.append({
                    "trained_at_utc": trained_at,
                    "station": station,
                    "pollutant": pollutant,
                    "horizon_hours": horizon,
                    "mae": model["metrics"]["mae"],
                    "rmse": model["metrics"]["rmse"],
                    "r2": model["metrics"]["r2"],
                    "baseline_mae": model["baseline_metrics"]["mae"],
                    "baseline_rmse": model["baseline_metrics"]["rmse"],
                    "baseline_r2": model["baseline_metrics"]["r2"],
                    "beats_baseline_mae": model["beats_baseline_mae"],
                    "train_rows": len(x_train),
                    "test_rows": len(x_test),
                    "model_path": model_path,
                })

    return registry, pd.DataFrame(metrics)


def main() -> None:
    load_env()
    ensure_dirs()
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo = GitHubRepo.from_env()
    allowed, reason, state = should_retrain(repo, args.force)
    if not allowed:
        print(f"skip: {reason}")
        return

    work = TMP_DIR / "retrain"
    historical_dir = work / "data_time"
    scraped_dir = work / "scraped_history"
    builded_dir = work / "builded"
    for path in [historical_dir, scraped_dir, builded_dir]:
        path.mkdir(parents=True, exist_ok=True)

    repo.download_prefix("data/time", historical_dir)
    repo.download_prefix("data/scraped/history", scraped_dir)

    registry, metrics = train_all(historical_dir, scraped_dir, builded_dir)
    if not registry:
        raise RuntimeError("No models trained")

    for model_file in builded_dir.glob("model__*.json"):
        repo.put_file(f"models/builded/{model_file.name}", model_file.read_bytes(), "Update trained model")

    repo.put_json("models/builded/registry.json", registry, "Update model registry")
    repo.put_json("models/registry.json", registry, "Update model registry")
    repo.put_file("metrics/latest.csv", metrics.to_csv(index=False).encode("utf-8-sig"), "Update latest metrics")
    latest_json = {
        "trained_at_utc": registry[0]["trained_at_utc"],
        "models": len(registry),
        "beats_baseline_mae": int(metrics["beats_baseline_mae"].sum()),
        "avg_r2": float(metrics["r2"].mean()),
    }
    repo.put_json("metrics/latest.json", latest_json, "Update latest metrics summary")
    stamp = now_utc().strftime("%Y-%m-%d_%H-%M_retrain.csv")
    repo.put_file(f"metrics/history/{stamp}", metrics.to_csv(index=False).encode("utf-8-sig"), "Add retrain metrics")
    metric_index = repo.get_json("metrics/index.json", default=[])
    metric_index.append({"path": f"metrics/history/{stamp}", "trained_at_utc": registry[0]["trained_at_utc"]})
    repo.put_json("metrics/index.json", metric_index[-500:], "Update metrics index")

    scrape_index = repo.get_json("data/scraped/index.json", default=[])
    state.update({
        "last_retrain_utc": now_utc().isoformat(timespec="seconds"),
        "last_retrain_scrape_count": len(scrape_index),
        "active_model_version": registry[0]["trained_at_utc"],
    })
    repo.put_json("state/pipeline_state.json", state, "Update pipeline state after retrain")
    print(f"trained {len(registry)} models")


if __name__ == "__main__":
    main()

