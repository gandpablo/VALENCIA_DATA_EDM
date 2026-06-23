from __future__ import annotations

import html
import json
from pathlib import Path

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


def color_for_value(pollutant: str, value: float | None) -> str:
    if pd.isna(value):
        return "#64748b"
    limits = {
        "NO2": (25, 50, 80),
        "O3": (50, 80, 110),
        "SO2": (5, 15, 40),
        "PM-10": (20, 35, 50),
        "PM-2.5": (8, 14, 25),
    }[pollutant]
    if value < limits[0]:
        return "#22c55e"
    if value < limits[1]:
        return "#eab308"
    if value < limits[2]:
        return "#f97316"
    return "#ef4444"


def radius_for_value(pollutant: str, value: float | None) -> int:
    if pd.isna(value):
        return 17
    max_reasonable = {"NO2": 100, "O3": 120, "SO2": 80, "PM-10": 60, "PM-2.5": 35}[pollutant]
    return int(17 + min(float(value) / max_reasonable, 1.0) * 17)


def map_points(df: pd.DataFrame, pollutant: str, mode: str) -> list[dict]:
    coords = station_coords()
    points = []
    for _, row in df.iterrows():
        if row["station"] not in coords:
            continue
        value = row[pollutant]
        missing_prediction = mode == "Prediccion" and row["station_display"] not in HISTORICAL_SCRAPER_STATIONS
        if missing_prediction:
            value = np.nan
        lat, lon = coords[row["station"]]
        points.append(
            {
                "label": row["station_display"],
                "station": row["station"],
                "lat": lat,
                "lon": lon,
                "value_label": (
                    "sin historico"
                    if missing_prediction
                    else "sin dato"
                    if pd.isna(value)
                    else f"{float(value):.1f} ug/m3"
                ),
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
            min-height: 835px;
            padding: 18px 14px;
            border: 1px solid rgba(125, 249, 255, .18);
            border-radius: 22px;
            background: rgba(2, 6, 23, .62);
            box-shadow: inset 0 1px 0 rgba(255,255,255,.04);
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
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 8px;
            margin-bottom: 10px;
          }
          .status-item {
            border: 1px solid rgba(125, 249, 255, .12);
            border-radius: 14px;
            background: rgba(8, 47, 73, .42);
            padding: 9px 11px;
          }
          .status-item span { display: block; color: #94a3b8; font-size: 10px; text-transform: uppercase; }
          .status-item b { display: block; margin-top: 2px; color: #f8feff; font-size: 15px; }
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
            <span><i style="background:#22c55e"></i>bajo</span>
            <span><i style="background:#eab308"></i>medio</span>
            <span><i style="background:#f97316"></i>alto</span>
            <span><i style="background:#ef4444"></i>muy alto</span>
            <span><i style="background:#64748b"></i>sin dato</span>
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
    st.markdown('<div class="chat-card">', unsafe_allow_html=True)
    st.markdown('<div class="side-title">Chat</div>', unsafe_allow_html=True)
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [("bot", "Hola, esto es un chat generico")]

    for role, message in st.session_state["chat_messages"][-8:]:
        klass = "chat-bubble-user" if role == "user" else "chat-bubble-bot"
        st.markdown(f'<div class="{klass}">{html.escape(message)}</div>', unsafe_allow_html=True)

    with st.form("mock_chat_form", clear_on_submit=True):
        prompt = st.text_input("Mensaje", placeholder="Escribe una pregunta")
        submitted = st.form_submit_button("Enviar", use_container_width=True)
        if submitted and prompt.strip():
            st.session_state["chat_messages"].append(("user", prompt.strip()))
            st.session_state["chat_messages"].append(("bot", "Hola, esto es un chat generico"))
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def toolbar() -> tuple[str, str]:
    st.markdown('<div class="control-bar">', unsafe_allow_html=True)
    col1, col2 = st.columns([3.6, 1.7], gap="small")
    with col1:
        st.markdown('<div class="control-label">Contaminante</div>', unsafe_allow_html=True)
        pollutant = st.pills("Contaminante", POLLUTANTS, default="NO2", label_visibility="collapsed")
    with col2:
        st.markdown('<div class="control-label">Vista</div>', unsafe_allow_html=True)
        mode = st.segmented_control("Vista", ["Actual", "Prediccion"], default="Actual", label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
    return pollutant or "NO2", mode or "Actual"


def status_strip(df: pd.DataFrame, pollutant: str, mode: str) -> None:
    values = df[pollutant].dropna()
    mean = f"{values.mean():.1f} ug/m3" if len(values) else "sin datos"
    maximum = f"{values.max():.1f} ug/m3" if len(values) else "sin datos"
    st.markdown(
        f"""
        <div class="status-strip">
          <div class="status-item"><span>Media</span><b>{html.escape(mean)}</b></div>
          <div class="status-item"><span>Maximo</span><b>{html.escape(maximum)}</b></div>
          <div class="status-item"><span>Estaciones con dato</span><b>{len(values)}/11</b></div>
          <div class="status-item"><span>Vista</span><b>{html.escape(mode)}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def dashboard() -> None:
    st.title("Valencia Air Lab")
    st.caption("Datos actualizados al pulsar ACCEDER. Chat generico en modo mock.")
    if "last_pipeline_result" in st.session_state:
        st.success(f"Actualizado y subido a GitHub: {st.session_state['last_pipeline_result']['scrape_file']}")

    chat, main = st.columns([1.12, 5.88], gap="medium")
    with chat:
        chat_panel()

    current = load_snapshot(SCRAPED_DIR / "latest.csv")
    prediction = load_snapshot(PREDICTIONS_DIR / "latest.csv")

    with main:
        pollutant, mode = toolbar()
        shown = current if mode == "Actual" else prediction
        status_strip(shown, pollutant, mode)
        leaflet_map(map_points(shown, pollutant, mode), pollutant, mode)


def main() -> None:
    style_page()
    if not st.session_state.get("access_granted", False):
        access_screen()
    else:
        dashboard()


if __name__ == "__main__":
    main()
