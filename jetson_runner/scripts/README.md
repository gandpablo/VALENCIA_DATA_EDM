# Scripts

- `jetson_pipeline.py`: entrada principal para cron.
- `predict_latest.py`: genera predicciones con los modelos publicados en GitHub.
- `retrain_models.py`: descarga historicos/scrapes desde GitHub, reentrena y sube modelos/metricas.
- `github_io.py`: cliente de GitHub Contents API, equivalente al enfoque del notebook de ejemplo.
- `events.py`: sube logs resumidos a `logs/pipeline_events.csv`.

