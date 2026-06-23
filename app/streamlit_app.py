from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


ROOT = Path(__file__).resolve().parents[1]
SCRAPED_DIR = ROOT / "data" / "scraped"
SCRAPED_HISTORY_DIR = SCRAPED_DIR / "history"
PREDICTIONS_DIR = ROOT / "predictions"
PREDICTIONS_HISTORY_DIR = PREDICTIONS_DIR / "history"
STATIONS_PATH = ROOT / "data" / "estaciones_valencia.csv"
METRICS_PATH = ROOT / "models" / "builded" / "metrics.csv"

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


st.set_page_config(
    page_title="Valencia Air Lab",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data
def load_station_coords() -> dict[str, tuple[float, float]]:
    stations = pd.read_csv(STATIONS_PATH)
    coords = {
        row["Estación"]: (float(row["Latitud"]), float(row["Longitud"]))
        for _, row in stations.iterrows()
    }
    coords.update(EXTRA_COORDS)
    return coords


@st.cache_data
def list_snapshots() -> list[Path]:
    source = SCRAPED_HISTORY_DIR if SCRAPED_HISTORY_DIR.exists() else SCRAPED_DIR
    return sorted(path for path in source.glob("*.csv") if path.name != "latest.csv")


@st.cache_data
def load_snapshot(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["station_display"] = df["µg/m3"]
    df["station"] = df["µg/m3"].map(NAME_MAP).fillna(df["µg/m3"])
    for col in POLLUTANTS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data
def load_metrics() -> pd.DataFrame:
    metrics = pd.read_csv(METRICS_PATH, encoding="utf-8-sig")
    metrics["r_score"] = metrics["r2"].clip(lower=-1, upper=1)
    metrics["beats_baseline_mae"] = metrics["beats_baseline_mae"].astype(bool)
    metrics["accuracy"] = np.where(metrics["beats_baseline_mae"], 100.0, 0.0)
    return metrics


def parse_snapshot_label(path: Path) -> str:
    return path.stem.replace("_", " ")


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

    max_reasonable = {
        "NO2": 100,
        "O3": 120,
        "SO2": 80,
        "PM-10": 60,
        "PM-2.5": 35,
    }[pollutant]
    return int(17 + min(float(value) / max_reasonable, 1.0) * 17)


def map_points(df: pd.DataFrame, pollutant: str, mode: str) -> list[dict]:
    coords = load_station_coords()
    points = []
    for _, row in df.iterrows():
        if row["station"] not in coords:
            continue

        lat, lon = coords[row["station"]]
        value = row[pollutant]
        missing_prediction = mode == "Prediccion" and row["station_display"] not in HISTORICAL_SCRAPER_STATIONS
        if missing_prediction:
            value = np.nan

        points.append(
            {
                "label": row["station_display"],
                "station": row["station"],
                "lat": lat,
                "lon": lon,
                "value": None if pd.isna(value) else round(float(value), 1),
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


def prediction_frame(paths: list[Path], idx: int, current: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    prediction_for_snapshot = PREDICTIONS_HISTORY_DIR / paths[idx].name
    if prediction_for_snapshot.exists():
        return load_snapshot(prediction_for_snapshot), f"{parse_snapshot_label(paths[idx])} +8h"

    latest_prediction = PREDICTIONS_DIR / "latest.csv"
    if latest_prediction.exists():
        return load_snapshot(latest_prediction), "latest +8h"

    if idx + 1 < len(paths):
        return load_snapshot(paths[idx + 1]), parse_snapshot_label(paths[idx + 1])

    if idx == 0:
        return current.copy(), "mock +8h"

    previous = load_snapshot(paths[idx - 1])
    pred = current.copy()
    for col in POLLUTANTS:
        pred[col] = (current[col] + 0.65 * (current[col] - previous[col])).clip(lower=0)
    return pred, "mock +8h"


def mask_unavailable_predictions(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode != "Prediccion":
        return df

    out = df.copy()
    no_history = ~out["station_display"].isin(HISTORICAL_SCRAPER_STATIONS)
    out.loc[no_history, POLLUTANTS] = np.nan
    return out


def leaflet_map(points: list[dict], pollutant: str, mode: str, snapshot_label: str) -> None:
    points_json = json.dumps(points, ensure_ascii=False)
    title = "Valores actuales" if mode == "Actual" else "Prediccion +8h"
    safe_title = html.escape(title)
    safe_pollutant = html.escape(pollutant)
    safe_snapshot = html.escape(snapshot_label)

    component = f"""
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
          html, body {{
            margin: 0;
            padding: 0;
            background: #020617;
            font-family: Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif;
          }}
          .map-frame {{
            position: relative;
            height: 815px;
            overflow: hidden;
            border: 1px solid rgba(125, 249, 255, .30);
            border-radius: 24px;
            box-shadow: 0 30px 90px rgba(0,0,0,.42), inset 0 1px 0 rgba(255,255,255,.08);
            background: #07111f;
          }}
          #map {{
            position: absolute;
            inset: 0;
            z-index: 1;
            background: #07111f;
          }}
          .map-frame::after {{
            content: "";
            pointer-events: none;
            position: absolute;
            inset: 0;
            z-index: 3;
            background:
              radial-gradient(circle at 20% 18%, rgba(34,211,238,.18), transparent 28%),
              radial-gradient(circle at 84% 72%, rgba(59,130,246,.15), transparent 28%),
              linear-gradient(180deg, rgba(2,6,23,.10), rgba(2,6,23,.22));
            mix-blend-mode: screen;
          }}
          .hud {{
            position: absolute;
            left: 22px;
            top: 20px;
            z-index: 5;
            max-width: 390px;
            padding: 16px 18px;
            border: 1px solid rgba(125,249,255,.28);
            border-radius: 18px;
            background: rgba(2, 6, 23, .76);
            box-shadow: 0 18px 44px rgba(0,0,0,.28);
            backdrop-filter: blur(12px);
          }}
          .hud small {{
            display: block;
            color: #67e8f9;
            text-transform: uppercase;
            font-size: 11px;
            margin-bottom: 7px;
          }}
          .hud h2 {{
            margin: 0;
            color: #f8feff;
            font-size: 28px;
            line-height: 1.08;
          }}
          .hud p {{
            margin: 8px 0 0;
            color: #cbd5e1;
            font-size: 13px;
          }}
          .legend {{
            position: absolute;
            right: 18px;
            bottom: 18px;
            z-index: 5;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            max-width: 520px;
            padding: 11px 13px;
            border: 1px solid rgba(125,249,255,.22);
            border-radius: 999px;
            background: rgba(2, 6, 23, .72);
            color: #e2e8f0;
            font-size: 12px;
            backdrop-filter: blur(10px);
          }}
          .legend span {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
          }}
          .legend i {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
          }}
          .leaflet-control-zoom a {{
            background: rgba(2, 6, 23, .82) !important;
            color: #e0f2fe !important;
            border-color: rgba(125,249,255,.22) !important;
          }}
          .leaflet-popup-content-wrapper, .leaflet-popup-tip {{
            background: rgba(2, 6, 23, .94);
            color: #e5f8ff;
            border: 1px solid rgba(125,249,255,.22);
            box-shadow: 0 18px 45px rgba(0,0,0,.45);
          }}
          .popup-title {{
            color: #fff;
            font-weight: 800;
            font-size: 15px;
            margin-bottom: 4px;
          }}
          .popup-subtitle {{
            color: #93c5fd;
            font-size: 12px;
            margin-bottom: 10px;
          }}
          .popup-value {{
            color: #a7f3d0;
            font-weight: 800;
            font-size: 22px;
          }}
          .marker-label {{
            color: #e5f8ff;
            font-weight: 800;
            text-shadow: 0 2px 10px rgba(0,0,0,.75);
            background: rgba(2, 6, 23, .72);
            border: 1px solid rgba(255,255,255,.18);
            border-radius: 999px;
            padding: 2px 8px;
          }}
        </style>
      </head>
      <body>
        <div class="map-frame">
          <div id="map"></div>
          <div class="hud">
            <small>{safe_pollutant} · {safe_snapshot}</small>
            <h2>{safe_title}</h2>
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
          const points = {points_json};
          const map = L.map('map', {{
            zoomControl: true,
            scrollWheelZoom: true,
            preferCanvas: true
          }}).setView([39.4699, -0.3763], 12.5);

          L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            maxZoom: 19,
            attribution: '&copy; OpenStreetMap'
          }}).addTo(map);

          const bounds = [];
          points.forEach((point) => {{
            const marker = L.circleMarker([point.lat, point.lon], {{
              radius: point.radius,
              color: '#f8fafc',
              weight: 2,
              fillColor: point.color,
              fillOpacity: 0.82,
              opacity: 0.95
            }}).addTo(map);

            marker.bindTooltip(
              `<span class="marker-label">${{point.label}}</span>`,
              {{ permanent: true, direction: 'top', offset: [0, -18], opacity: 0.96 }}
            );

            marker.bindPopup(`
              <div class="popup-title">${{point.label}}</div>
              <div class="popup-subtitle">${{point.station}}</div>
              <div class="popup-value">${{point.value_label}}</div>
            `);
            bounds.push([point.lat, point.lon]);
          }});

          if (bounds.length > 0) {{
            map.fitBounds(bounds, {{ padding: [55, 55], maxZoom: 13 }});
          }}
        </script>
      </body>
    </html>
    """
    components.html(component, height=845, scrolling=False)


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
          .block-container {
            max-width: 100%;
            padding: 1.1rem 1.25rem .8rem;
          }
          [data-testid="stHeader"] {
            background: transparent;
          }
          div[data-testid="stMetric"] {
            background: rgba(8, 47, 73, .52);
            border: 1px solid rgba(125, 249, 255, .14);
            border-radius: 18px;
            padding: 14px;
          }
          div[data-testid="stDialog"] div[role="dialog"] {
            border: 1px solid rgba(125, 249, 255, .22);
            background:
              radial-gradient(circle at 20% 0%, rgba(34,211,238,.12), transparent 28%),
              linear-gradient(135deg, rgba(2,6,23,.98), rgba(7,17,31,.98));
          }
          .control-bar {
            display: block;
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
          .snapshot-chip {
            min-height: 38px;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 1px solid rgba(125, 249, 255, .16);
            border-radius: 12px;
            background: rgba(15, 23, 42, .72);
            color: #e0f2fe;
            font-size: 13px;
            font-weight: 700;
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
          .status-item span {
            display: block;
            color: #94a3b8;
            font-size: 10px;
            text-transform: uppercase;
          }
          .status-item b {
            display: block;
            margin-top: 2px;
            color: #f8feff;
            font-size: 15px;
          }
          .side-card {
            min-height: 835px;
            padding: 18px 14px;
            border: 1px solid rgba(125, 249, 255, .18);
            border-radius: 22px;
            background: rgba(2, 6, 23, .58);
            box-shadow: inset 0 1px 0 rgba(255,255,255,.04);
          }
          .side-title {
            color: #67e8f9;
            text-transform: uppercase;
            font-size: 11px;
            margin-bottom: 10px;
          }
          .chat-card {
            min-height: 835px;
            padding: 18px 14px;
            border: 1px solid rgba(125, 249, 255, .18);
            border-radius: 22px;
            background: rgba(2, 6, 23, .62);
            box-shadow: inset 0 1px 0 rgba(255,255,255,.04);
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
          .control-help {
            color: #94a3b8;
            font-size: 12px;
            line-height: 1.45;
            margin: 12px 0 18px;
          }
          h1, h2, h3 {
            letter-spacing: 0;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def top_bar() -> None:
    left, right = st.columns([0.78, 0.22], vertical_alignment="center")
    with left:
        st.title("Valencia Air Lab")
        st.caption("Mock operativo con mapa real, datos sinteticos y monitorizacion de modelos.")
    with right:
        if st.button("Admin modelos", type="primary", use_container_width=True):
            admin_dialog()


@st.dialog("Monitorizacion de modelos", width="large")
def admin_dialog() -> None:
    admin_panel()


def login_controls() -> bool:
    if st.session_state.get("admin_logged", False):
        return True

    user = st.text_input("Usuario", key="admin_user")
    password = st.text_input("Password", type="password", key="admin_password")
    if st.button("Entrar", type="primary", use_container_width=True):
        if user == "usuario" and password == "usuario":
            st.session_state["admin_logged"] = True
            st.rerun()
        st.error("Credenciales incorrectas")
    return False


def admin_panel() -> None:
    if not login_controls():
        st.caption("Credenciales mock: usuario / usuario")
        return

    metrics = load_metrics()
    st.success("Sesion admin activa")

    general_r = metrics["r_score"].mean()
    general_acc = metrics["accuracy"].mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("R score", f"{general_r:.2f}")
    c2.metric("Accuracy", f"{general_acc:.0f}%")
    c3.metric("Modelos", f"{len(metrics)}")

    stations = ["Todas"] + sorted(metrics["station"].unique())
    pollutants = ["Todos"] + sorted(metrics["pollutant"].unique())
    selected_station = st.selectbox("Zona", stations)
    selected_pollutant = st.selectbox("Contaminante", pollutants)

    filtered = metrics.copy()
    if selected_station != "Todas":
        filtered = filtered[filtered["station"] == selected_station]
    if selected_pollutant != "Todos":
        filtered = filtered[filtered["pollutant"] == selected_pollutant]

    if filtered.empty:
        st.warning("No hay modelos para ese filtro.")
        return

    summary_cols = st.columns(3)
    summary_cols[0].metric("R filtrado", f"{filtered['r_score'].mean():.2f}")
    summary_cols[1].metric("Accuracy filtrada", f"{filtered['accuracy'].mean():.0f}%")
    summary_cols[2].metric("Modelos filtrados", f"{len(filtered)}")

    if selected_pollutant == "Todos":
        chart = filtered.groupby("pollutant", as_index=True).agg(
            r_score=("r_score", "mean"),
            accuracy=("accuracy", "mean"),
        )
    else:
        chart = filtered.groupby("station", as_index=True).agg(
            r_score=("r_score", "mean"),
            accuracy=("accuracy", "mean"),
        )

    st.bar_chart(chart)
    st.dataframe(
        filtered[
            [
                "station",
                "pollutant",
                "horizon_hours",
                "r_score",
                "accuracy",
                "mae",
                "baseline_mae",
            ]
        ].rename(
            columns={
                "station": "Zona",
                "pollutant": "Contaminante",
                "horizon_hours": "Horizonte",
                "r_score": "R score",
                "accuracy": "Accuracy",
                "mae": "MAE",
                "baseline_mae": "MAE baseline",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    if st.button("Cerrar sesion", use_container_width=True):
        st.session_state["admin_logged"] = False
        st.rerun()


def metric_strip(df: pd.DataFrame, pollutant: str, mode: str, snapshot: str) -> None:
    values = df[pollutant].dropna()
    mean = f"{values.mean():.1f} ug/m3" if len(values) else "sin datos"
    maximum = f"{values.max():.1f} ug/m3" if len(values) else "sin datos"
    count = f"{len(values)}/11"
    st.markdown(
        f"""
        <div class="status-strip">
          <div class="status-item"><span>Media</span><b>{html.escape(mean)}</b></div>
          <div class="status-item"><span>Maximo</span><b>{html.escape(maximum)}</b></div>
          <div class="status-item"><span>Estaciones con dato</span><b>{count}</b></div>
          <div class="status-item"><span>Vista</span><b>{html.escape(mode)}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chat_panel() -> None:
    st.markdown('<div class="chat-card">', unsafe_allow_html=True)
    st.markdown('<div class="side-title">Chat</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="control-help">
          Chat sintetico para el mock. Mas adelante puede leer valores, predicciones y metricas.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [
            ("bot", "Hola, esto es un chat generico"),
        ]

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


def set_state_and_rerun(key: str, value: object) -> None:
    st.session_state[key] = value
    st.rerun()


def toolbar(labels: list[str]) -> tuple[str, str, int]:
    st.markdown('<div class="control-bar">', unsafe_allow_html=True)
    pollutant_col, mode_col, snapshot_col = st.columns([2.7, 1.55, 1.85], gap="small")

    with pollutant_col:
        st.markdown('<div class="control-label">Contaminante</div>', unsafe_allow_html=True)
        pollutant = st.pills(
            "Contaminante",
            POLLUTANTS,
            key="selected_pollutant",
            label_visibility="collapsed",
        )

    with mode_col:
        st.markdown('<div class="control-label">Vista</div>', unsafe_allow_html=True)
        mode = st.segmented_control(
            "Vista",
            ["Actual", "Prediccion"],
            key="selected_mode",
            label_visibility="collapsed",
        )

    with snapshot_col:
        st.markdown('<div class="control-label">Captura</div>', unsafe_allow_html=True)
        idx = st.session_state["snapshot_idx"]
        st.markdown(
            f'<div class="snapshot-chip">{html.escape(labels[idx])}</div>',
            unsafe_allow_html=True,
        )
        if st.button("Captura anterior", key="snapshot_prev", use_container_width=True, disabled=idx == 0):
            set_state_and_rerun("snapshot_idx", max(0, idx - 1))
        if st.button("Captura siguiente", key="snapshot_next", use_container_width=True, disabled=idx == len(labels) - 1):
            set_state_and_rerun("snapshot_idx", min(len(labels) - 1, idx + 1))

    st.markdown("</div>", unsafe_allow_html=True)
    return pollutant or "NO2", mode or "Actual", st.session_state["snapshot_idx"]


def main() -> None:
    style_page()
    top_bar()

    paths = list_snapshots()
    labels = [parse_snapshot_label(path) for path in paths]
    st.session_state.setdefault("selected_pollutant", "NO2")
    st.session_state.setdefault("selected_mode", "Actual")
    st.session_state.setdefault("snapshot_idx", max(0, len(labels) - 3))
    st.session_state["snapshot_idx"] = min(st.session_state["snapshot_idx"], len(labels) - 1)

    chat, center = st.columns([1.12, 5.88], gap="medium")
    with chat:
        chat_panel()

    with center:
        pollutant, mode, idx = toolbar(labels)

    selected_label = labels[idx]
    current = load_snapshot(paths[idx])
    predicted, pred_label = prediction_frame(paths, idx, current)
    shown_df = current if mode == "Actual" else predicted
    shown_df = mask_unavailable_predictions(shown_df, mode)
    shown_label = selected_label if mode == "Actual" else pred_label

    with center:
        metric_strip(shown_df, pollutant, mode, selected_label)
        leaflet_map(map_points(shown_df, pollutant, mode), pollutant, mode, shown_label)


if __name__ == "__main__":
    main()
