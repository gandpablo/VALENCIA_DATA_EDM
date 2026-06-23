from __future__ import annotations

import argparse
from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
from config import load_env


def scrape_table(url: str, table_id: str) -> pd.DataFrame:
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
        driver.get(url)
        table_html = WebDriverWait(driver, 40).until(
            EC.presence_of_element_located((By.ID, table_id))
        ).get_attribute("outerHTML")
    finally:
        driver.quit()

    return pd.read_html(StringIO(table_html))[0]


def timestamp_name() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M.csv")


def main() -> None:
    env = load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="tmp/scraped")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    url = env.get("SCRAPER_URL", "https://www.valencia.es/valenciaalminut/")
    table_id = env.get("SCRAPER_TABLE_ID", "tabla_dinamica")
    df = scrape_table(url, table_id)

    filename = timestamp_name()
    timestamp_path = out_dir / filename
    latest_path = out_dir / "latest.csv"
    df.to_csv(timestamp_path, index=False, encoding="utf-8-sig")
    df.to_csv(latest_path, index=False, encoding="utf-8-sig")

    print(timestamp_path)


if __name__ == "__main__":
    main()

