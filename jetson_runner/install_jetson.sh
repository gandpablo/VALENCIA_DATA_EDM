#!/usr/bin/env bash
set -euo pipefail

sudo apt update
sudo apt install -y git python3 python3-pip python3-venv cron chromium-browser chromium-chromedriver

python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

mkdir -p logs tmp/scraped tmp/predictions tmp/retrain

echo "Installation complete."
echo "Copy .env.example to .env and fill in your GitHub repository and token."

