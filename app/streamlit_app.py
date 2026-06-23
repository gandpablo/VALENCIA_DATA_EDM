from __future__ import annotations

import html
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from pipeline import run_manual_pipeline


ROOT = Path(__file__).resolve().parents[1]
SCRAPED_DIR = ROOT / "data" / "scraped"
PREDICTIONS_DIR = ROOT / "predictions"
STATIONS_PATH = ROOT / "data" / "estaciones_valencia.csv"

POLLUTANTS = ["NO2", "O3", "SO2", "PM-10", "PM-2.5"]
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
EXTRA_COORDS = {
    "DR.LLUCH": (39.4664, -0.3283),
    "CABANYAL": (39.4700, -0.3320),
    "PATRAIX": (39.4623, -0.3958),
}
HISTORICAL_SCRAPER_STATIONS = set(NAME_MAP)
QUALITY_LEVELS = [
    "Buena",
    "Razonablemente buena",
    "Regular",
    "Desfavorable",
    "Muy desfavorable",
    "Extremadamente desfavorable",
]
QUALITY_COLORS = {
    "Buena": "#50f0e6",
    "Razonablemente buena": "#50ccaa",
    "Regular": "#f0e641",
    "Desfavorable": "#ff5050",
    "Muy desfavorable": "#960032",
    "Extremadamente desfavorable": "#7d2181",
    "No hay datos": "#64748b",
}
QUALITY_THRESHOLDS = {
    "SO2": [100, 200, 350, 500, 750, 1250],
    "NO2": [40, 90, 120, 230, 340, 1000],
    "O3": [50, 100, 130, 240, 380, 800],
    "PM-10": [20, 40, 50, 100, 150, 1200],
    "PM-2.5": [10, 20, 25, 50, 75, 800],
}
POLLUTANT_NAMES = {
    "SO2": "Dioxido de Azufre",
    "NO2": "Dioxido de Nitrogeno",
    "O3": "Ozono",
    "PM-10": "Particulas < 10 micras",
    "PM-2.5": "Particulas < 2.5 micras",
}


st.set_page_config(page_title="Valencia Air Lab", layout="wide", initial_sidebar_state="collapsed")


@st.cache_data
def station_coords() -> dict[str, tuple[float, float]]:
    stations = pd.read_csv(STATIONS_PATH)
    coords = {
        row["Estación"]: (float(row["Latitud"]), float(row["Longitud"]))
        for _, row in stations.iterrows()
    }
    coords.update(EXTRA_COORDS)
    return coords


@st.cache_data
def load_snapshot(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["station_display"] = df["µg/m3"]
    df["station"] = df["µg/m3"].map(NAME_MAP).fillna(df["µg/m3"])
    for col in POLLUTANTS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data
def load_scraped_history() -> pd.DataFrame:
    rows = []
    history_dir = SCRAPED_DIR / "history"
    for path in sorted(history_dir.glob("*.csv")):
        timestamp = pd.to_datetime(path.stem, format="%Y-%m-%d_%H-%M", errors="coerce")
        if pd.isna(timestamp):
            continue
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame["timestamp"] = timestamp
        rows.append(frame)

    if not rows:
        return pd.DataFrame(columns=["timestamp", "station_display", "station", *POLLUTANTS])

    history = pd.concat(rows, ignore_index=True)
    history["station_display"] = history["µg/m3"]
    history["station"] = history["µg/m3"].map(NAME_MAP).fillna(history["µg/m3"])
    for col in POLLUTANTS:
        history[col] = pd.to_numeric(history[col], errors="coerce")
    return history.dropna(subset=["timestamp"]).sort_values("timestamp")


def quality_for_value(pollutant: str, value: float | None) -> str:
    if pd.isna(value):
        return "No hay datos"
    for level, upper in zip(QUALITY_LEVELS, QUALITY_THRESHOLDS[pollutant]):
        if float(value) <= upper:
            return level
    return "Extremadamente desfavorable"


def format_timestamp(value: pd.Timestamp) -> str:
    return pd.to_datetime(value).strftime("%d/%m/%Y %H:%M")


def format_date_range(values: pd.Series) -> str:
    if values.empty:
        return "sin fechas"
    first = format_timestamp(values.min())
    last = format_timestamp(values.max())
    return f"{first} - {last}"


def threshold_range(pollutant: str, index: int) -> str:
    lower = 0 if index == 0 else QUALITY_THRESHOLDS[pollutant][index - 1] + 1
    upper = QUALITY_THRESHOLDS[pollutant][index]
    return f"{lower}-{upper}"


def color_for_value(pollutant: str, value: float | None) -> str:
    return QUALITY_COLORS[quality_for_value(pollutant, value)]


def radius_for_value(pollutant: str, value: float | None) -> int:
    if pd.isna(value):
        return 17
    max_reasonable = QUALITY_THRESHOLDS[pollutant][3]
    return int(17 + min(float(value) / max_reasonable, 1.0) * 17)


def map_points(df: pd.DataFrame, pollutant: str, mode: str) -> list[dict]:
    coords = station_coords()
    points = []
    for _, row in df.iterrows():
        if row["station"] not in coords:
            continue
        value = row[pollutant]
        missing_prediction = mode == "Prediccion" and row["station_display"] not in HISTORICAL_SCRAPER_STATIONS
        if missing_prediction or pd.isna(value):
            continue
        lat, lon = coords[row["station"]]
        quality = quality_for_value(pollutant, value)
        points.append(
            {
                "label": row["station_display"],
                "station": row["station"],
                "lat": lat,
                "lon": lon,
                "value_label": f"{float(value):.1f} ug/m3 · {quality}",
                "color": color_for_value(pollutant, value),
                "radius": radius_for_value(pollutant, value),
            }
        )
    return points


def style_page() -> None:
    st.markdown(
        """
        <style>
          .stApp {
            background:
              radial-gradient(circle at 8% 10%, rgba(20, 184, 166, .15), transparent 25%),
              radial-gradient(circle at 92% 8%, rgba(59, 130, 246, .14), transparent 26%),
              linear-gradient(135deg, #050816 0%, #07111f 46%, #041114 100%);
            color: #e5f8ff;
          }
          .block-container { max-width: 100%; padding: 1.1rem 1.25rem .8rem; }
          [data-testid="stHeader"] { background: transparent; }
          .access-shell {
            min-height: 78vh;
            display: flex;
            align-items: center;
            justify-content: center;
          }
          .access-panel {
            width: min(760px, 94vw);
            padding: 38px;
            border-radius: 28px;
            border: 1px solid rgba(125,249,255,.22);
            background:
              radial-gradient(circle at 20% 0%, rgba(34,211,238,.18), transparent 30%),
              linear-gradient(135deg, rgba(2,6,23,.86), rgba(7,17,31,.92));
            box-shadow: 0 34px 100px rgba(0,0,0,.38), inset 0 1px 0 rgba(255,255,255,.06);
            text-align: center;
          }
          .access-panel h1 { margin: 0; font-size: 44px; color: #f8feff; }
          .access-panel p { color: #cbd5e1; font-size: 16px; line-height: 1.55; margin: 14px auto 0; max-width: 620px; }
          .chat-card {
            min-height: 662px;
            padding: 18px 14px;
            border: 1px solid rgba(125, 249, 255, .18);
            border-radius: 22px;
            background: rgba(2, 6, 23, .62);
            box-shadow: inset 0 1px 0 rgba(255,255,255,.04);
          }
          .chat-form-shell {
            margin-top: 10px;
            padding: 10px;
            border: 1px solid rgba(125, 249, 255, .14);
            border-radius: 18px;
            background: rgba(2, 6, 23, .48);
          }
          .side-title {
            color: #67e8f9;
            text-transform: uppercase;
            font-size: 11px;
            margin-bottom: 10px;
          }
          .control-bar {
            padding: 12px 14px;
            margin-bottom: 12px;
            border: 1px solid rgba(125, 249, 255, .18);
            border-radius: 20px;
            background: rgba(2, 6, 23, .70);
            box-shadow: inset 0 1px 0 rgba(255,255,255,.05), 0 18px 42px rgba(0,0,0,.18);
          }
          .control-label {
            color: #67e8f9;
            text-transform: uppercase;
            font-size: 10px;
            margin: 0 0 7px 0;
          }
          .status-strip {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 8px;
            margin-bottom: 10px;
          }
          .status-strip.history-kpis { grid-template-columns: repeat(8, minmax(0, 1fr)); }
          .status-item {
            border: 1px solid rgba(125, 249, 255, .12);
            border-radius: 14px;
            background: rgba(8, 47, 73, .42);
            padding: 9px 11px;
            min-height: 64px;
          }
          .status-item span { display: block; color: #94a3b8; font-size: 10px; text-transform: uppercase; }
          .status-item b { display: block; margin-top: 2px; color: #f8feff; font-size: 15px; }
          .quality-dot {
            display: inline-block;
            width: 9px;
            height: 9px;
            border-radius: 999px;
            margin-right: 6px;
            box-shadow: 0 0 14px currentColor;
          }
          .history-card {
            padding: 20px;
            margin-bottom: 10px;
            border: 1px solid rgba(125,249,255,.30);
            border-radius: 24px;
            background:
              radial-gradient(circle at 14% 8%, rgba(34,211,238,.16), transparent 28%),
              linear-gradient(135deg, rgba(2,6,23,.76), rgba(7,17,31,.88));
            box-shadow: 0 30px 90px rgba(0,0,0,.34), inset 0 1px 0 rgba(255,255,255,.06);
          }
          .info-panel {
            min-height: 815px;
            padding: 20px;
            border: 1px solid rgba(125,249,255,.30);
            border-radius: 24px;
            background:
              radial-gradient(circle at 14% 8%, rgba(34,211,238,.16), transparent 28%),
              linear-gradient(135deg, rgba(2,6,23,.76), rgba(7,17,31,.88));
            box-shadow: 0 30px 90px rgba(0,0,0,.34), inset 0 1px 0 rgba(255,255,255,.06);
          }
          .history-title {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 16px;
            margin-bottom: 12px;
          }
          .history-title h2 { margin: 0; color: #f8feff; font-size: 28px; }
          .history-title p { margin: 7px 0 0; color: #cbd5e1; font-size: 13px; }
          .info-grid {
            display: grid;
            grid-template-columns: 1.15fr .85fr;
            gap: 14px;
            margin-top: 12px;
          }
          .info-box {
            border: 1px solid rgba(125,249,255,.16);
            border-radius: 16px;
            background: rgba(2,6,23,.46);
            padding: 14px;
          }
          .quality-table {
            width: 100%;
            border-collapse: collapse;
            color: #dbeafe;
            font-size: 12px;
          }
          .quality-table th, .quality-table td {
            border-bottom: 1px solid rgba(148,163,184,.16);
            padding: 9px 8px;
            text-align: left;
          }
          .quality-table th {
            color: #67e8f9;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0;
          }
          .abbr-list {
            display: grid;
            gap: 9px;
            color: #cbd5e1;
            font-size: 13px;
            line-height: 1.38;
          }
          .threshold-note {
            color: #94a3b8;
            font-size: 12px;
            line-height: 1.45;
            margin-top: 15px;
          }
          .chat-bubble-user {
            margin: 10px 0 6px auto;
            padding: 10px 12px;
            max-width: 92%;
            border-radius: 14px 14px 4px 14px;
            background: rgba(14, 165, 233, .22);
            border: 1px solid rgba(125, 249, 255, .20);
            color: #e0f2fe;
            font-size: 13px;
          }
          .chat-bubble-bot {
            margin: 6px auto 10px 0;
            padding: 10px 12px;
            max-width: 92%;
            border-radius: 14px 14px 14px 4px;
            background: rgba(15, 23, 42, .86);
            border: 1px solid rgba(148, 163, 184, .18);
            color: #d1fae5;
            font-size: 13px;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def leaflet_map(points: list[dict], pollutant: str, mode: str) -> None:
    component = f"""
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
          html, body {{ margin:0; padding:0; background:#020617; font-family:Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; }}
          .map-frame {{ position:relative; height:815px; overflow:hidden; border:1px solid rgba(125,249,255,.30); border-radius:24px; box-shadow:0 30px 90px rgba(0,0,0,.42), inset 0 1px 0 rgba(255,255,255,.08); background:#07111f; }}
          #map {{ position:absolute; inset:0; z-index:1; background:#07111f; }}
          .map-frame::after {{ content:""; pointer-events:none; position:absolute; inset:0; z-index:3; background:radial-gradient(circle at 20% 18%, rgba(34,211,238,.18), transparent 28%), radial-gradient(circle at 84% 72%, rgba(59,130,246,.15), transparent 28%), linear-gradient(180deg, rgba(2,6,23,.10), rgba(2,6,23,.22)); mix-blend-mode:screen; }}
          .hud {{ position:absolute; left:22px; top:20px; z-index:5; max-width:390px; padding:16px 18px; border:1px solid rgba(125,249,255,.28); border-radius:18px; background:rgba(2,6,23,.76); box-shadow:0 18px 44px rgba(0,0,0,.28); backdrop-filter:blur(12px); }}
          .hud small {{ display:block; color:#67e8f9; text-transform:uppercase; font-size:11px; margin-bottom:7px; }}
          .hud h2 {{ margin:0; color:#f8feff; font-size:28px; line-height:1.08; }}
          .hud p {{ margin:8px 0 0; color:#cbd5e1; font-size:13px; }}
          .legend {{ position:absolute; right:18px; bottom:18px; z-index:5; display:flex; gap:10px; flex-wrap:wrap; max-width:520px; padding:11px 13px; border:1px solid rgba(125,249,255,.22); border-radius:999px; background:rgba(2,6,23,.72); color:#e2e8f0; font-size:12px; backdrop-filter:blur(10px); }}
          .legend span {{ display:inline-flex; align-items:center; gap:6px; }}
          .legend i {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
          .leaflet-control-zoom a {{ background:rgba(2,6,23,.82) !important; color:#e0f2fe !important; border-color:rgba(125,249,255,.22) !important; }}
          .leaflet-popup-content-wrapper, .leaflet-popup-tip {{ background:rgba(2,6,23,.94); color:#e5f8ff; border:1px solid rgba(125,249,255,.22); box-shadow:0 18px 45px rgba(0,0,0,.45); }}
          .popup-title {{ color:#fff; font-weight:800; font-size:15px; margin-bottom:4px; }}
          .popup-subtitle {{ color:#93c5fd; font-size:12px; margin-bottom:10px; }}
          .popup-value {{ color:#a7f3d0; font-weight:800; font-size:22px; }}
          .marker-label {{ color:#e5f8ff; font-weight:800; text-shadow:0 2px 10px rgba(0,0,0,.75); background:rgba(2,6,23,.72); border:1px solid rgba(255,255,255,.18); border-radius:999px; padding:2px 8px; }}
        </style>
      </head>
      <body>
        <div class="map-frame">
          <div id="map"></div>
          <div class="hud">
            <small>{html.escape(pollutant)}</small>
            <h2>{'Valores actuales' if mode == 'Actual' else 'Prediccion +8h'}</h2>
            <p>Mapa interactivo de Valencia. Amplia, desplaza y pulsa cualquier punto para ver el detalle.</p>
          </div>
          <div class="legend">
            <span><i style="background:#50f0e6"></i>buena</span>
            <span><i style="background:#50ccaa"></i>raz. buena</span>
            <span><i style="background:#f0e641"></i>regular</span>
            <span><i style="background:#ff5050"></i>desfavorable</span>
            <span><i style="background:#960032"></i>muy desf.</span>
            <span><i style="background:#7d2181"></i>extrema</span>
          </div>
        </div>
        <script>
          const points = {json.dumps(points, ensure_ascii=False)};
          const map = L.map('map', {{ zoomControl:true, scrollWheelZoom:true, preferCanvas:true }}).setView([39.4699, -0.3763], 12.5);
          L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{ maxZoom:19, attribution:'&copy; OpenStreetMap' }}).addTo(map);
          const bounds = [];
          points.forEach((point) => {{
            const marker = L.circleMarker([point.lat, point.lon], {{
              radius: point.radius, color:'#f8fafc', weight:2, fillColor:point.color, fillOpacity:.82, opacity:.95
            }}).addTo(map);
            marker.bindTooltip(`<span class="marker-label">${{point.label}}</span>`, {{ permanent:true, direction:'top', offset:[0,-18], opacity:.96 }});
            marker.bindPopup(`<div class="popup-title">${{point.label}}</div><div class="popup-subtitle">${{point.station}}</div><div class="popup-value">${{point.value_label}}</div>`);
            bounds.push([point.lat, point.lon]);
          }});
          if (bounds.length > 0) map.fitBounds(bounds, {{ padding:[55,55], maxZoom:13 }});
        </script>
      </body>
    </html>
    """
    components.html(component, height=845, scrolling=False)


def access_screen() -> None:
    st.markdown(
        """
        <div class="access-shell">
          <div class="access-panel">
            <h1>Valencia Air Lab</h1>
            <p>Al acceder se descarga la medicion actual, se generan predicciones +8h, se publican los resultados en GitHub y se abre el mapa operativo.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _, center, _ = st.columns([1.2, 1, 1.2])
    with center:
        if st.button("ACCEDER", type="primary", use_container_width=True):
            with st.spinner("Scrapeando, prediciendo y subiendo a GitHub..."):
                result = run_manual_pipeline()
            st.cache_data.clear()
            st.session_state["access_granted"] = True
            st.session_state["last_pipeline_result"] = result
            st.rerun()


def chat_panel() -> None:
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [("bot", "Hola, esto es un chat generico")]

    messages = ['<div class="side-title">Chat</div>']
    for role, message in st.session_state["chat_messages"][-8:]:
        klass = "chat-bubble-user" if role == "user" else "chat-bubble-bot"
        messages.append(f'<div class="{klass}">{html.escape(message)}</div>')
    st.markdown(f'<section class="chat-card">{"".join(messages)}</section>', unsafe_allow_html=True)

    with st.form("mock_chat_form", clear_on_submit=True):
        prompt = st.text_input("Mensaje", placeholder="Escribe una pregunta")
        submitted = st.form_submit_button("Enviar", use_container_width=True)
        if submitted and prompt.strip():
            st.session_state["chat_messages"].append(("user", prompt.strip()))
            st.session_state["chat_messages"].append(("bot", "Hola, esto es un chat generico"))
            st.rerun()


def toolbar() -> tuple[str, str]:
    st.markdown('<div class="control-bar">', unsafe_allow_html=True)
    col1, col2 = st.columns([3.6, 2.2], gap="small")
    with col1:
        st.markdown('<div class="control-label">Contaminante</div>', unsafe_allow_html=True)
        pollutant = st.pills("Contaminante", POLLUTANTS, default="NO2", label_visibility="collapsed")
    with col2:
        st.markdown('<div class="control-label">Vista</div>', unsafe_allow_html=True)
        mode = st.segmented_control(
            "Vista",
            ["Actual", "Prediccion", "Historico", "Info"],
            default="Actual",
            label_visibility="collapsed",
        )
    st.markdown("</div>", unsafe_allow_html=True)
    return pollutant or "NO2", mode or "Actual"


def status_strip(df: pd.DataFrame, pollutant: str, mode: str) -> None:
    values = df[pollutant].dropna()
    total_stations = len(df)
    no_data = total_stations - len(values)
    mean = f"{values.mean():.1f} ug/m3" if len(values) else "sin datos"
    if len(values):
        max_index = values.idxmax()
        maximum = f"{values.max():.1f} ug/m3"
        max_station = df.loc[max_index, "station_display"]
        worst_quality = quality_for_value(pollutant, values.max())
        qualities = [quality_for_value(pollutant, value) for value in values]
        desired_count = sum(level in {"Buena", "Razonablemente buena"} for level in qualities)
        regular_or_worse = len(values) - desired_count
        desired_share = f"{desired_count}/{len(values)}"
    else:
        maximum = "sin datos"
        max_station = "-"
        worst_quality = "No hay datos"
        desired_count = 0
        regular_or_worse = 0
        desired_share = "0/0"
    quality_color = QUALITY_COLORS[worst_quality]
    st.markdown(
        f"""
        <div class="status-strip">
          <div class="status-item"><span>Media</span><b>{html.escape(mean)}</b></div>
          <div class="status-item"><span>Maximo</span><b>{html.escape(maximum)}</b><span>{html.escape(str(max_station))}</span></div>
          <div class="status-item"><span>Peor calidad</span><b><i class="quality-dot" style="background:{quality_color}; color:{quality_color};"></i>{html.escape(worst_quality)}</b></div>
          <div class="status-item"><span>Calidad deseada</span><b>{desired_share}</b><span>buena o razonable</span></div>
          <div class="status-item"><span>Regular o peor</span><b>{regular_or_worse}</b><span>sobre {len(values)} con dato</span></div>
          <div class="status-item"><span>Sin datos</span><b>{no_data}</b><span>no aparecen en mapa</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def historical_panel(pollutant: str) -> None:
    history = load_scraped_history()
    station_options = sorted(history["station_display"].dropna().unique()) if not history.empty else []

    st.markdown(
        f"""
        <section class="history-card">
        <div class="history-title" style="margin-bottom:0;">
          <div>
            <h2>Historico scrapeado</h2>
            <p>Evolucion de los ultimos CSV almacenados en <b>data/scraped/history</b> y publicados en GitHub.</p>
          </div>
          <div style="color:#67e8f9;font-size:12px;text-transform:uppercase;">{html.escape(POLLUTANT_NAMES[pollutant])}</div>
        </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if not station_options:
        st.warning("Todavia no hay historico scrapeado disponible.")
        return

    selected_station = st.selectbox("Estacion", station_options, label_visibility="collapsed", key=f"station_{pollutant}")
    station_history = history[history["station_display"] == selected_station][["timestamp", pollutant]].dropna()
    station_history = station_history.sort_values("timestamp")

    if station_history.empty:
        st.info("No hay datos de este contaminante para la estacion seleccionada.")
        return

    latest_value = float(station_history[pollutant].iloc[-1])
    previous_value = float(station_history[pollutant].iloc[-2]) if len(station_history) > 1 else np.nan
    delta_value = latest_value - previous_value if not pd.isna(previous_value) else np.nan
    latest_quality = quality_for_value(pollutant, latest_value)
    latest_color = QUALITY_COLORS[latest_quality]
    period_mean = float(station_history[pollutant].mean())
    period_max = float(station_history[pollutant].max())
    period_min = float(station_history[pollutant].min())
    quality_values = station_history[pollutant].map(lambda value: quality_for_value(pollutant, value))
    desired_count = int(quality_values.isin(["Buena", "Razonablemente buena"]).sum())
    desired_pct = desired_count / len(station_history) * 100
    delta_label = "sin previo" if pd.isna(delta_value) else f"{delta_value:+.1f} ug/m3"
    period_label = format_date_range(station_history["timestamp"])

    st.markdown(
        f"""
        <div class="status-strip history-kpis">
          <div class="status-item"><span>Ultimo valor</span><b>{latest_value:.1f} ug/m3</b></div>
          <div class="status-item"><span>Calidad actual</span><b><i class="quality-dot" style="background:{latest_color}; color:{latest_color};"></i>{html.escape(latest_quality)}</b></div>
          <div class="status-item"><span>Cambio ultimo</span><b>{html.escape(delta_label)}</b></div>
          <div class="status-item"><span>Media historica</span><b>{period_mean:.1f} ug/m3</b></div>
          <div class="status-item"><span>Minimo</span><b>{period_min:.1f} ug/m3</b></div>
          <div class="status-item"><span>Maximo historico</span><b>{period_max:.1f} ug/m3</b></div>
          <div class="status-item"><span>Calidad deseada</span><b>{desired_pct:.0f}%</b><span>{desired_count}/{len(station_history)} registros</span></div>
          <div class="status-item"><span>Periodo</span><b>{len(station_history)}</b><span>{html.escape(period_label)}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chart_data = station_history.rename(columns={"timestamp": "Fecha", pollutant: "Valor"}).copy()
    chart_data["Calidad"] = chart_data["Valor"].map(lambda value: quality_for_value(pollutant, value))
    chart_data["FechaTexto"] = chart_data["Fecha"].map(format_timestamp)
    chart_upper = max(period_max * 1.18, QUALITY_THRESHOLDS[pollutant][0] * 1.12)
    rule_data = pd.DataFrame(
        [
            {
                "Valor": limit,
                "Umbral": f"{level}: {limit} ug/m3",
                "Color": QUALITY_COLORS[level],
            }
            for level, limit in zip(QUALITY_LEVELS, QUALITY_THRESHOLDS[pollutant])
            if limit <= chart_upper
        ]
    )

    line = (
        alt.Chart(chart_data)
        .mark_line(color="#67e8f9", strokeWidth=3)
        .encode(
            x=alt.X("Fecha:T", title="Fecha scrapeo", axis=alt.Axis(format="%d/%m %H:%M", labelAngle=-30)),
            y=alt.Y("Valor:Q", title=f"{pollutant} (ug/m3)", scale=alt.Scale(zero=True)),
            tooltip=[
                alt.Tooltip("Fecha:T", title="Fecha", format="%d/%m/%Y %H:%M"),
                alt.Tooltip("Valor:Q", title=f"{pollutant} ug/m3", format=".1f"),
                alt.Tooltip("Calidad:N", title="Calidad"),
            ],
        )
    )
    points = (
        alt.Chart(chart_data)
        .mark_circle(size=82, stroke="#f8fafc", strokeWidth=1.4)
        .encode(
            x="Fecha:T",
            y="Valor:Q",
            color=alt.Color(
                "Calidad:N",
                scale=alt.Scale(domain=QUALITY_LEVELS, range=[QUALITY_COLORS[level] for level in QUALITY_LEVELS]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Fecha:T", title="Fecha", format="%d/%m/%Y %H:%M"),
                alt.Tooltip("Valor:Q", title=f"{pollutant} ug/m3", format=".1f"),
                alt.Tooltip("Calidad:N", title="Calidad"),
            ],
        )
    )
    if rule_data.empty:
        chart = line + points
    else:
        rules = (
            alt.Chart(rule_data)
            .mark_rule(strokeDash=[5, 5], opacity=0.58)
            .encode(
                y="Valor:Q",
                color=alt.Color("Color:N", scale=None, legend=None),
                tooltip=[alt.Tooltip("Umbral:N", title="Umbral")],
            )
        )
        chart = rules + line + points
    st.altair_chart(chart.properties(height=500), use_container_width=True)
    recent = chart_data.tail(8).sort_values("Fecha", ascending=False).copy()
    recent["Fecha"] = recent["Fecha"].map(format_timestamp)
    recent["Valor"] = recent["Valor"].map(lambda value: f"{value:.1f} ug/m3")
    st.dataframe(recent[["Fecha", "Valor", "Calidad"]], hide_index=True, use_container_width=True, height=300)
    st.markdown(
        """
        <div class="threshold-note">
          Umbrales de color basados en el Indice Nacional de Calidad del Aire
          (Orden TEC/351/2019 y Resolucion de 2 de septiembre de 2020).
          Elaboracion propia con datos del
          <a href="https://www.valencia.es/val/qualitataire/contaminacio-atmosferica" target="_blank" style="color:#67e8f9;">Servicio de mejora climatica</a>.
          ug/m3 = microgramos por metro cubico; PM-10 y PM-2.5 = particulas en suspension.
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_panel() -> None:
    header_cells = "".join(f"<th>{html.escape(pollutant)}</th>" for pollutant in ["SO2", "NO2", "O3", "PM-10", "PM-2.5"])
    rows = []
    for index, level in enumerate(QUALITY_LEVELS):
        color = QUALITY_COLORS[level]
        values = "".join(
            f"<td>{html.escape(threshold_range(pollutant, index))}</td>"
            for pollutant in ["SO2", "NO2", "O3", "PM-10", "PM-2.5"]
        )
        rows.append(
            f"""
            <tr>
              <td><i class="quality-dot" style="background:{color}; color:{color};"></i>{html.escape(level)}</td>
              {values}
            </tr>
            """
        )

    st.markdown(
        f"""
        <section class="info-panel">
          <div class="history-title">
            <div>
              <h2>Info calidad del aire</h2>
              <p>Los colores del mapa, las predicciones y el historico se calculan con estos rangos de calidad del aire.</p>
            </div>
            <div style="color:#67e8f9;font-size:12px;text-transform:uppercase;">umbrales oficiales</div>
          </div>
          <div class="info-grid">
            <div class="info-box">
              <table class="quality-table">
                <thead>
                  <tr>
                    <th>Calidad del aire</th>
                    {header_cells}
                  </tr>
                </thead>
                <tbody>
                  {''.join(rows)}
                </tbody>
              </table>
              <div class="threshold-note">
                Rangos en µg/m3. ND significa que no se han recabado suficientes datos para establecer criterio.
                Las estaciones sin dato para un contaminante no se muestran en el mapa.
              </div>
            </div>
            <div class="info-box">
              <div class="abbr-list">
                <div><b>SO2</b> Dioxido de Azufre</div>
                <div><b>NO2</b> Dioxido de Nitrogeno</div>
                <div><b>O3</b> Ozono</div>
                <div><b>PM-10</b> Particulas en suspension inferiores a 10 micras</div>
                <div><b>PM-2.5</b> Particulas en suspension inferiores a 2.5 micras</div>
                <div><b>µg/m3</b> Microgramos por metro cubico</div>
                <div><b>mg/m3</b> Miligramos por metro cubico</div>
              </div>
              <div class="threshold-note">
                Umbrales basados en el Indice Nacional de Calidad del Aire
                (Orden TEC/351/2019, de 18 de marzo) y Resolucion de 2 de septiembre de 2020.
                Elaboracion propia con datos proporcionados por el
                <a href="https://www.valencia.es/val/qualitataire/contaminacio-atmosferica" target="_blank" style="color:#67e8f9;">Servicio de mejora climatica</a>.
              </div>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def dashboard() -> None:
    st.title("Valencia Air Lab")
    st.caption("Datos actualizados al pulsar ACCEDER. Chat generico en modo mock.")
    if "last_pipeline_result" in st.session_state:
        st.success(f"Actualizado y subido a GitHub: {st.session_state['last_pipeline_result']['scrape_file']}")

    main, chat = st.columns([5.7, 1.35], gap="medium")

    current = load_snapshot(SCRAPED_DIR / "latest.csv")
    prediction = load_snapshot(PREDICTIONS_DIR / "latest.csv")

    with main:
        pollutant, mode = toolbar()
        if mode == "Historico":
            historical_panel(pollutant)
        elif mode == "Info":
            info_panel()
        else:
            shown = current if mode == "Actual" else prediction
            status_strip(shown, pollutant, mode)
            leaflet_map(map_points(shown, pollutant, mode), pollutant, mode)

    with chat:
        chat_panel()


def main() -> None:
    style_page()
    if not st.session_state.get("access_granted", False):
        access_screen()
    else:
        dashboard()


if __name__ == "__main__":
    main()
