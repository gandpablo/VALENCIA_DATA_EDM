# Jetson runner

Carpeta aislada para ejecutar el pipeline desde una Jetson Nano.

La Jetson no almacena historico largo. Solo mantiene codigo, logs pequenos y archivos temporales. GitHub es el almacen principal.

## 1. Instalacion

```bash
cd /home/jetson
cp -r jetson_runner valencia-air-runner
cd valencia-air-runner
bash install_jetson.sh
cp .env.example .env
nano .env
```

Configura en `.env`:

```env
GITHUB_OWNER=tu_usuario
GITHUB_REPO=tu_repo
GITHUB_BRANCH=main
GITHUB_TOKEN=tu_token
```

El token debe tener permisos para leer y escribir contenidos del repo.

## 2. Estructura esperada en GitHub

```text
data/
  estaciones_valencia.csv
  time/
    filtrado_*.csv
  scraped/
    latest.csv
    index.json
    history/

predictions/
  latest.csv
  index.json
  history/

models/
  builded/
    registry.json
    model__*.json

metrics/
  latest.csv
  latest.json
  index.json
  history/

logs/
  pipeline_events.csv
  latest_event.json

state/
  pipeline_state.json
```

## 3. Ejecucion manual

Scraper + prediccion + subida a GitHub:

```bash
.venv/bin/python scripts/jetson_pipeline.py --mode scrape
```

Reentrenamiento si toca:

```bash
.venv/bin/python scripts/jetson_pipeline.py --mode retrain
```

## 4. Cron

```bash
crontab -e
```

Copia el contenido de `cron.example` ajustando la ruta.

## 5. Logs

Locales en Jetson:

```text
logs/pipeline.log
logs/retrain.log
```

Resumidos en GitHub:

```text
logs/pipeline_events.csv
logs/latest_event.json
```

## 6. Nota de seguridad

No subas `.env` a GitHub. El token debe vivir solo en la Jetson.

